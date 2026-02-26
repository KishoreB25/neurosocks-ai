#include <Wire.h>
#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

/* ================= CONFIG ================= */

#define MAX_ADDR   0x57   // MAX30102
#define MPU_ADDR   0x68   // MPU6050

// Pressure sensor ADC pins
#define P_HEEL  32
#define P_BALL  33
#define P_ARCH  34
#define P_TOE   35

// Temperature — no hardware sensors yet; set to 0 so app shows 0.0°C
// When you add thermistors/DS18B20, replace with real reads.
#define HAS_TEMP_SENSORS  false

/* ============== GLOBAL DATA =============== */

uint16_t stepCount    = 0;
uint8_t  activityType = 0;   // 0=rest
uint8_t  batteryLevel = 85;  // placeholder until ADC wired

// --- MAX30102 HR detection ---
#define HR_BUF_SIZE  60      // ~30 s of samples at 2 Hz internal averaging
uint32_t irBuffer[HR_BUF_SIZE];
uint32_t redBuffer[HR_BUF_SIZE];
int      bufIdx = 0;
bool     bufFull = false;

// --- MPU6050 step detection ---
float    prevMag      = 0;
bool     stepHigh     = false;
float    accelMagSum  = 0;
int      accelSamples = 0;

/* ============ I2C HELPERS ================= */

// --- generic register helpers ---
void i2cWrite(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t i2cRead8(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

int16_t i2cRead16(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, (uint8_t)2);
  if (Wire.available() == 2) {
    int16_t hi = Wire.read();
    int16_t lo = Wire.read();
    return (hi << 8) | lo;
  }
  return 0;
}

/* ========== MAX30102 FUNCTIONS ============ */

void maxInit() {
  // Reset
  i2cWrite(MAX_ADDR, 0x09, 0x40);
  delay(100);

  // FIFO config: sample avg = 4, FIFO rollover ON
  i2cWrite(MAX_ADDR, 0x08, 0x50);

  // Mode: SpO2 mode (RED + IR)
  i2cWrite(MAX_ADDR, 0x09, 0x03);

  // SpO2 config: ADC range 4096, 100 sps, 411 µs pulse
  i2cWrite(MAX_ADDR, 0x0A, 0x27);

  // LED pulse amplitude
  i2cWrite(MAX_ADDR, 0x0C, 0x24);   // RED ~7 mA
  i2cWrite(MAX_ADDR, 0x0D, 0x24);   // IR  ~7 mA

  // Clear FIFO pointers
  i2cWrite(MAX_ADDR, 0x04, 0x00);
  i2cWrite(MAX_ADDR, 0x05, 0x00);
  i2cWrite(MAX_ADDR, 0x06, 0x00);
}

// Read one RED+IR sample pair from FIFO (6 bytes total)
bool maxReadFifo(uint32_t &red, uint32_t &ir) {
  Wire.beginTransmission(MAX_ADDR);
  Wire.write(0x07);  // FIFO_DATA
  Wire.endTransmission(false);
  Wire.requestFrom(MAX_ADDR, (uint8_t)6);

  if (Wire.available() < 6) return false;

  // RED (3 bytes, 18-bit)
  red  = Wire.read(); red <<= 8;
  red |= Wire.read(); red <<= 8;
  red |= Wire.read();
  red &= 0x3FFFF;

  // IR (3 bytes, 18-bit)
  ir  = Wire.read(); ir <<= 8;
  ir |= Wire.read(); ir <<= 8;
  ir |= Wire.read();
  ir &= 0x3FFFF;

  return true;
}

// Drain available FIFO samples into ring buffers
void maxCollectSamples() {
  uint8_t wrPtr = i2cRead8(MAX_ADDR, 0x04) & 0x1F;
  uint8_t rdPtr = i2cRead8(MAX_ADDR, 0x06) & 0x1F;
  int numSamples = (wrPtr >= rdPtr) ? (wrPtr - rdPtr) : (32 + wrPtr - rdPtr);
  if (numSamples == 0) return;
  if (numSamples > 16) numSamples = 16;  // cap per read cycle

  for (int i = 0; i < numSamples; i++) {
    uint32_t r, ir;
    if (!maxReadFifo(r, ir)) break;

    redBuffer[bufIdx] = r;
    irBuffer[bufIdx]  = ir;
    bufIdx++;
    if (bufIdx >= HR_BUF_SIZE) {
      bufIdx = 0;
      bufFull = true;
    }
  }
}

// Compute HR (BPM) from IR buffer using peak detection
// and SpO2 from RED/IR ratio
void maxCompute(float &spo2, uint16_t &hr) {
  int count = bufFull ? HR_BUF_SIZE : bufIdx;
  if (count < 10) {
    spo2 = 0;
    hr   = 0;
    return;
  }

  // --- DC (mean) and AC (max-min) for both channels ---
  uint32_t irSum = 0, redSum = 0;
  uint32_t irMin = 0xFFFFFFFF, irMax = 0;
  uint32_t redMin = 0xFFFFFFFF, redMax = 0;

  for (int i = 0; i < count; i++) {
    irSum  += irBuffer[i];
    redSum += redBuffer[i];
    if (irBuffer[i]  < irMin)  irMin  = irBuffer[i];
    if (irBuffer[i]  > irMax)  irMax  = irBuffer[i];
    if (redBuffer[i] < redMin) redMin = redBuffer[i];
    if (redBuffer[i] > redMax) redMax = redBuffer[i];
  }

  float irDC  = (float)irSum  / count;
  float redDC = (float)redSum / count;
  float irAC  = (float)(irMax  - irMin);
  float redAC = (float)(redMax - redMin);

  // Finger presence check (IR DC should be > threshold)
  if (irDC < 5000) {
    spo2 = 0;
    hr   = 0;
    return;
  }

  // --- SpO2 via ratio-of-ratios ---
  if (irDC > 0 && redDC > 0 && irAC > 0) {
    float R = (redAC / redDC) / (irAC / irDC);
    // Linear approximation: SpO2 ≈ 110 - 25 * R
    // Calibrate with known reference if needed
    spo2 = 110.0 - 25.0 * R;
    if (spo2 > 100.0) spo2 = 100.0;
    if (spo2 < 0.0)   spo2 = 0.0;
  } else {
    spo2 = 0;
  }

  // --- Heart rate via zero-crossing on IR AC component ---
  // Count "peaks" where sample crosses above mean then back below
  int peaks = 0;
  bool above = false;
  float threshold = irDC + irAC * 0.3;  // 30% above mean

  for (int i = 0; i < count; i++) {
    if (!above && irBuffer[i] > threshold) {
      above = true;
      peaks++;
    } else if (above && irBuffer[i] < irDC) {
      above = false;
    }
  }

  // samples span ~(count / 25) seconds (100 sps / 4 avg = 25 effective sps)
  float seconds = (float)count / 25.0;
  if (seconds > 0 && peaks > 1) {
    hr = (uint16_t)((float)(peaks - 1) / seconds * 60.0);
    if (hr > 220) hr = 220;
    if (hr < 30)  hr = 0;   // likely noise
  } else {
    hr = 0;
  }
}

/* ========== MPU6050 FUNCTIONS ============= */

void mpuInit() {
  // Wake up (clear sleep bit)
  i2cWrite(MPU_ADDR, 0x6B, 0x00);
  delay(100);

  // Accel range ±4g (AFS_SEL = 1)
  i2cWrite(MPU_ADDR, 0x1C, 0x08);

  // Gyro range ±500 °/s (FS_SEL = 1)
  i2cWrite(MPU_ADDR, 0x1B, 0x08);

  // DLPF ~20 Hz for clean step signal
  i2cWrite(MPU_ADDR, 0x1A, 0x04);
}

// Read accel X, Y, Z in m/s² (±4 g range → 8192 LSB/g)
void mpuReadAccel(float &ax, float &ay, float &az) {
  int16_t raw_x = i2cRead16(MPU_ADDR, 0x3B);
  int16_t raw_y = i2cRead16(MPU_ADDR, 0x3D);
  int16_t raw_z = i2cRead16(MPU_ADDR, 0x3F);

  ax = (float)raw_x / 8192.0 * 9.81;
  ay = (float)raw_y / 8192.0 * 9.81;
  az = (float)raw_z / 8192.0 * 9.81;
}

// Simple step detection: magnitude peak crossing
void mpuDetectSteps(float ax, float ay, float az) {
  float mag = sqrt(ax * ax + ay * ay + az * az);

  // Track average magnitude for activity classification
  accelMagSum += mag;
  accelSamples++;

  // Peak detection with hysteresis
  // Threshold tuned for walking (~12 m/s² peak vs ~9.8 rest)
  const float STEP_HIGH = 11.5;
  const float STEP_LOW  = 9.5;

  if (!stepHigh && mag > STEP_HIGH) {
    stepHigh = true;
    stepCount++;
  } else if (stepHigh && mag < STEP_LOW) {
    stepHigh = false;
  }

  prevMag = mag;
}

// Classify activity from average accel magnitude variance
// Called once per BT send cycle (every 2 s)
void mpuClassifyActivity() {
  if (accelSamples == 0) {
    activityType = 0;  // rest
    return;
  }

  float avgMag = accelMagSum / accelSamples;

  // Deviation from gravity (~9.81 m/s²)
  float dev = abs(avgMag - 9.81);

  if (dev < 0.3)       activityType = 0;  // resting / still
  else if (dev < 0.8)  activityType = 2;  // standing (minor sway)
  else if (dev < 2.5)  activityType = 3;  // walking
  else                  activityType = 4;  // running

  // Reset for next cycle
  accelMagSum  = 0;
  accelSamples = 0;
}

/* ============== SETUP ===================== */

void setup() {
  Serial.begin(115200);
  SerialBT.begin("NeuroSock");

  Wire.begin(21, 22);

  maxInit();
  mpuInit();

  // ADC setup for pressure pins
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  Serial.println("NeuroSock ready — MAX30102 + MPU6050 + Pressure");
}

/* =============== LOOP ===================== */

// Read MPU at ~50 Hz between BT sends for accurate step detection
unsigned long lastSend   = 0;
unsigned long lastMpuRead = 0;
const unsigned long SEND_INTERVAL = 2000;  // ms
const unsigned long MPU_INTERVAL  = 20;    // ms (~50 Hz)

void loop() {

  unsigned long now = millis();

  /* ---- HIGH-FREQ: MPU6050 step detection (~50 Hz) ---- */
  if (now - lastMpuRead >= MPU_INTERVAL) {
    lastMpuRead = now;

    float ax, ay, az;
    mpuReadAccel(ax, ay, az);
    mpuDetectSteps(ax, ay, az);
  }

  /* ---- Collect MAX30102 FIFO samples continuously ---- */
  maxCollectSamples();

  /* ---- LOW-FREQ: Pack & send every 2 s ---- */
  if (now - lastSend < SEND_INTERVAL) return;
  lastSend = now;

  /* -------- SENSOR READINGS -------- */

  // -- Temperatures --
  float temp[4];
  #if HAS_TEMP_SENSORS
    // TODO: Replace with your real thermistor / DS18B20 reads
    temp[0] = readThermistor(0);
    temp[1] = readThermistor(1);
    temp[2] = readThermistor(2);
    temp[3] = readThermistor(3);
  #else
    // No temp sensors → send 0 (byte 128 → decodes to 25.0 + 0 = 25.0)
    // Using 128 means "25.0°C" which is the baseline.
    // Send raw 0 instead so app shows 25.0 - 64.0 = -39.0 → obviously wrong
    // Better: send 128 for each so app shows 25.0°C (neutral baseline)
    // OR send a sentinel. Cleanest: use real baseline 25.0°C
    temp[0] = 0; temp[1] = 0; temp[2] = 0; temp[3] = 0;
  #endif

  // -- Pressures (from ADC → kPa) --
  float pressure[4];
  pressure[0] = analogRead(P_HEEL) * 77.0 / 4095.0;
  pressure[1] = analogRead(P_BALL) * 77.0 / 4095.0;
  pressure[2] = analogRead(P_ARCH) * 77.0 / 4095.0;
  pressure[3] = analogRead(P_TOE)  * 77.0 / 4095.0;

  // -- MAX30102 → SpO2 & HR --
  float spo2 = 0;
  uint16_t heartRate = 0;
  maxCompute(spo2, heartRate);

  // -- MPU6050 → activity classification --
  mpuClassifyActivity();

  /* -------- PACK 16-BYTE PAYLOAD -------- */

  uint8_t payload[16];

  // Temperatures (Bytes 0–3): encoding = (temp - 25.0) * 2.0 + 128
  for (int i = 0; i < 4; i++) {
    #if HAS_TEMP_SENSORS
      payload[i] = (uint8_t)constrain((temp[i] - 25.0) * 2.0 + 128, 0, 255);
    #else
      payload[i] = 0;   // 0 → app decodes as -39.0°C → clearly "no sensor"
    #endif
  }

  // Pressures (Bytes 4–7): encoding = pressure / 0.3
  for (int i = 0; i < 4; i++)
    payload[4 + i] = (uint8_t)constrain(pressure[i] / 0.3, 0, 255);

  // SpO2 (Bytes 8–9): uint16 big-endian, value = spo2 * 100
  uint16_t spo2_raw = (uint16_t)(spo2 * 100);
  payload[8]  = (spo2_raw >> 8) & 0xFF;
  payload[9]  = spo2_raw & 0xFF;

  // Heart Rate (Bytes 10–11): uint16 big-endian
  payload[10] = (heartRate >> 8) & 0xFF;
  payload[11] = heartRate & 0xFF;

  // Step Count (Bytes 12–13): uint16 big-endian
  payload[12] = (stepCount >> 8) & 0xFF;
  payload[13] = stepCount & 0xFF;

  // Activity Type (Byte 14)
  payload[14] = activityType;

  // Battery Level (Byte 15)
  payload[15] = batteryLevel;

  /* -------- SEND VIA BLUETOOTH -------- */
  SerialBT.write(payload, 16);

  /* -------- DEBUG -------- */
  Serial.printf("SpO2: %.1f%% | HR: %d BPM | Steps: %d | Activity: %d | "
                "P: [%.1f, %.1f, %.1f, %.1f] kPa\n",
                spo2, heartRate, stepCount, activityType,
                pressure[0], pressure[1], pressure[2], pressure[3]);
}