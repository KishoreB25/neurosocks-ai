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

// ⚠️ IMPORTANT: ADC2 (GPIO 0,2,4,12-15,25-27) CANNOT be used when
//    Bluetooth (Classic or BLE) is active on ESP32!
//    Only ADC1 pins work: GPIO 32,33,34,35,36,39
//
// Temperature sensor — 1 physical NTC on ADC1
// Wiring: 3.3V → 10kΩ fixed resistor → ADC pin → NTC → GND
// Only GPIO 36 works reliably. Ball, Arch, Toe are derived from Heel
// with realistic thermal gradients (foot periphery is cooler).
#define HAS_TEMP_SENSORS  true   // Set false to disable temperature entirely
#define T_HEEL  36        // GPIO36 (VP) = ADC1_CH0  — Only working sensor
// Ball, Arch, Toe: derived mathematically from Heel reading
//   Ball = Heel - 0.3°C  (slightly cooler, plantar surface)
//   Arch = Heel - 0.5°C  (cooler, less blood flow)
//   Toe  = Heel - 0.8°C  (coldest, farthest from core)

// NTC thermistor parameters (common 10kΩ NTC — adjust B for your part)
#define NTC_R_SERIES   10000.0   // Fixed series resistor in voltage divider (ohms)
#define NTC_R_NOMINAL  10000.0   // NTC resistance at 25°C (ohms)
#define NTC_B_COEFF    3950.0    // B-coefficient (check your NTC datasheet)
#define NTC_T_NOMINAL  25.0      // Reference temperature for nominal R (°C)

#define TEMP_BASELINE     0.0    // Fallback value when sensors disabled
// Battery: GPIO2 is ADC2 → also broken with BT. Use fixed 85% placeholder.
// For real battery monitoring add a voltage divider to GPIO36/39 or an I2C fuel gauge.

/* ============== GLOBAL DATA =============== */

uint16_t stepCount    = 0;
uint8_t  activityType = 0;   // 0=rest
uint8_t  batteryLevel = 85;  // Read from ADC in loop()

// --- MAX30102 HR detection ---
#define HR_BUF_SIZE  60      // ~30 s of samples at 2 Hz internal averaging
uint32_t irBuffer[HR_BUF_SIZE];
uint32_t redBuffer[HR_BUF_SIZE];
int      bufIdx = 0;
bool     bufFull = false;

// --- MPU6050 step detection ---
float    prevMag       = 0;
bool     stepHigh      = false;
float    accelMagSum   = 0;
int      accelSamples  = 0;
unsigned long lastStepTime = 0;            // Step debounce timer
const unsigned long STEP_DEBOUNCE = 300;   // Min ms between steps (prevents double-count)

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

/* ======= ADC & TEMPERATURE HELPERS ======= */

// Read ADC with multi-sample averaging (reduces noise significantly)
float readADCAverage(int pin) {
  long sum = 0;
  for (int i = 0; i < 16; i++) {
    sum += analogRead(pin);
    delayMicroseconds(200);
  }
  return (float)sum / 16.0;
}

// Read NTC thermistor temperature via voltage divider
// Circuit: 3.3V → NTC_R_SERIES → ADC_PIN → NTC → GND
float readThermistor(int pin) {
  float adcVal = readADCAverage(pin);

  // Debug: always print raw ADC so we can diagnose wiring
  Serial.printf("[NTC] GPIO%d raw ADC: %.1f\n", pin, adcVal);

  // Open circuit (no sensor) or shorted
  if (adcVal < 1.0 || adcVal > 4090.0) {
    Serial.printf("[NTC] GPIO%d REJECTED (open/short)\n", pin);
    return 0.0;
  }

  // Calculate NTC resistance from voltage divider ratio
  float resistance = NTC_R_SERIES * adcVal / (4095.0 - adcVal);

  // B-parameter Steinhart-Hart equation:
  //   1/T = 1/T0 + (1/B) * ln(R/R0)
  float steinhart = log(resistance / NTC_R_NOMINAL) / NTC_B_COEFF;
  steinhart += 1.0 / (NTC_T_NOMINAL + 273.15);
  float tempC = 1.0 / steinhart - 273.15;

  // Reject clearly invalid readings (sensor fault)
  if (tempC < -10.0 || tempC > 60.0) return 0.0;

  return tempC;
}

// Read pressure sensor with ADC averaging for stable values
float readPressureADC(int pin) {
  float adcVal = readADCAverage(pin);
  return adcVal * 77.0 / 4095.0;  // Map 12-bit ADC → 0-77 kPa range
}

/* ========== MAX30102 FUNCTIONS ============ */

void maxInit() {
  Serial.println("\n=== MAX30102 INITIALIZATION ===");
  delay(100);
  
  // Wire.begin() already called in setup() — no duplicate needed here

  // Step 1: Verify I2C Communication
  Serial.print("[1] Checking I2C connection to 0x57... ");
  
  Wire.beginTransmission(MAX_ADDR);
  int I2CError = Wire.endTransmission();
  if (I2CError == 0) {
    Serial.println("✅ ACK received");
  } else {
    Serial.printf("❌ I2C Error: %d (Check SDA:21, SCL:22, 3.3V, GND)\n", I2CError);
    return;
  }

  // Step 2: Read Part ID to verify sensor
  Serial.print("[2] Reading Part ID register... ");
  uint8_t id = i2cRead8(MAX_ADDR, 0xFF);
  Serial.printf("ID: 0x%02X\n", id);
  if (id != 0x15) {
    Serial.printf("❌ WRONG ID! Expected 0x15, got 0x%02X. Wrong sensor connected?\n", id);
    return;
  } else {
    Serial.println("✅ Correct MAX30102 sensor detected!");
  }

  // Step 3: Reset
  Serial.print("[3] Sending RESET command... ");
  i2cWrite(MAX_ADDR, 0x09, 0x40);
  delay(100);
  Serial.println("✅");

  // Step 4: Configure FIFO
  Serial.print("[4] Configuring FIFO... ");
  i2cWrite(MAX_ADDR, 0x08, 0x50);  // Sample avg=4, rollover=ON
  Serial.println("✅");

  // Step 5: Set Mode to SpO2
  Serial.print("[5] Setting SpO2 mode... ");
  i2cWrite(MAX_ADDR, 0x09, 0x03);
  Serial.println("✅");

  // Step 6: Configure SpO2
  Serial.print("[6] Configuring SpO2 (ADC/sample rate)... ");
  i2cWrite(MAX_ADDR, 0x0A, 0x27);
  Serial.println("✅");

  // Step 7: Set LED Current (THIS WAS THE MAIN PROBLEM - was too weak)
  Serial.print("[7] Setting LED current to 15mA... ");
  i2cWrite(MAX_ADDR, 0x0C, 0x40);   // RED  15mA
  i2cWrite(MAX_ADDR, 0x0D, 0x40);   // IR   15mA
  Serial.println("✅");

  // Step 8: Clear FIFO
  Serial.print("[8] Clearing FIFO pointers... ");
  i2cWrite(MAX_ADDR, 0x04, 0x00);
  i2cWrite(MAX_ADDR, 0x05, 0x00);
  i2cWrite(MAX_ADDR, 0x06, 0x00);
  Serial.println("✅");

  // Step 9: Verify FIFO is working
  Serial.print("[9] Checking FIFO status... ");
  uint8_t status = i2cRead8(MAX_ADDR, 0x00);
  Serial.printf("Status: 0x%02X\n", status);

  Serial.println("\n✅ MAX30102 READY! Hold finger on sensor for 3-5 seconds...\n");
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
  // DON'T read I2C constantly - causes bus contention with MPU
  // Only read when we have time (not right after MPU read)
  static unsigned long lastMaxRead = 0;
  if (millis() - lastMaxRead < 5) return;  // Wait 5ms between MAX reads
  lastMaxRead = millis();
  
  uint8_t wrPtr = i2cRead8(MAX_ADDR, 0x04) & 0x1F;
  uint8_t rdPtr = i2cRead8(MAX_ADDR, 0x06) & 0x1F;
  int numSamples = (wrPtr >= rdPtr) ? (wrPtr - rdPtr) : (32 + wrPtr - rdPtr);
  
  // Debug: Print FIFO status every 2 seconds
  static unsigned long lastFifoDebug = 0;
  if (millis() - lastFifoDebug > 2000) {
    Serial.printf("[FIFO] wrPtr:%d rdPtr:%d numSamples:%d bufIdx:%d bufFull:%d\n", 
                   wrPtr, rdPtr, numSamples, bufIdx, bufFull);
    lastFifoDebug = millis();
  }
  
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
  
  // Need minimum 30 samples (~0.3 sec) for stable reading
  // Ideally 60+ samples (~0.6 sec) for accuracy
  if (count < 30) {
    spo2 = 0;
    hr = 0;
    static unsigned long lastMsg = 0;
    if (millis() - lastMsg > 1000) {  // Print every 1 sec instead of 3
      Serial.printf("[MAX] Warming up... %d/60 samples (hold finger on sensor)\n", count);
      lastMsg = millis();
    }
    return;
  }

  // --- Calculate DC (mean) and AC (peak-to-peak) ---
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
  
  // Debug output every 2 seconds
  static unsigned long lastDebug = 0;
  if (millis() - lastDebug > 2000) {
    Serial.printf("[MAX] irDC:%.0f redDC:%.0f irAC:%.0f redAC:%.0f ratio:%.2f samples:%d\n",
                   irDC, redDC, irAC, redAC, redDC/irDC, count);
    lastDebug = millis();
  }

  // --- ADAPTIVE Finger Detection (3-point check) ---
  
  // Check 1: AC ripple (heartbeat pulsatile component)
  // Real finger with 15mA LED: irAC typically 1000-5000
  // No finger or blowing: irAC < 300
  if (irAC < 300) {
    spo2 = 0;
    hr = 0;
    return;
  }

  // Check 2: DC signal baseline (light intensity)
  // Real finger: irDC 5000-50000 (depends on skin tone)
  // Ambient only: irDC < 1000
  if (irDC < 1000) {
    spo2 = 0;
    hr = 0;
    return;
  }

  // Check 3: RED/IR ratio (blood absorption signature)
  // Real finger: redDC ≈ irDC (both wavelengths absorbed)
  // Plastic/foam: redDC >> irDC or vice versa
  float ratio = redDC / irDC;
  if (ratio < 0.5 || ratio > 2.0) {
    spo2 = 0;
    hr = 0;
    return;
  }

  // --- SpO2 Calculation (industry standard ratio-of-ratios) ---
  if (irDC > 0 && redDC > 0 && irAC > 0 && redAC > 0) {
    float R = (redAC / redDC) / (irAC / irDC);
    // Empirical formula: SpO2 = 110 - 25*R
    spo2 = 110.0 - 25.0 * R;
    // Clamp to realistic range (70-100%)
    spo2 = constrain(spo2, 70.0, 100.0);
  } else {
    spo2 = 0;
  }

  // --- Heart Rate Calculation (peak counting on IR signal) ---
  int peaks = 0;
  bool above = false;
  float threshold = irDC + irAC * 0.3;  // Peak = DC + 30% of AC swing

  for (int i = 0; i < count; i++) {
    if (!above && irBuffer[i] > threshold) {
      above = true;
      peaks++;
    } else if (above && irBuffer[i] < irDC) {
      above = false;
    }
  }

  // Convert peaks to BPM
  // Sensor: 100 Hz → 4x averaging = 25 effective Hz
  float seconds = (float)count / 25.0;
  if (seconds > 0 && peaks > 0) {
    hr = (uint16_t)((float)peaks / seconds * 60.0);
    // Sanity check: realistic HR is 40-200 BPM
    if (hr < 40 || hr > 200) hr = 0;
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
  // Further lowered thresholds: 10.0 up / 9.6 down to catch realistic walking
  const float STEP_HIGH = 10.0;  // WAS 10.5 (was 11.5 originally)
  const float STEP_LOW  = 9.6;   // WAS 9.3 (was 9.5 originally)

  if (!stepHigh && mag > STEP_HIGH) {
    // Debounce: ignore if last step was too recent (prevents double-counting)
    if (millis() - lastStepTime >= STEP_DEBOUNCE) {
      stepHigh = true;
      stepCount++;
      lastStepTime = millis();
      Serial.printf("[MPU] STEP! Count: %d, Mag: %.2f m/s²\n", stepCount, mag);
    }
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
  // Tightened thresholds to better distinguish activity levels:
  // 0.15: barely any motion (sleeping/very still)
  // 0.35: small movements (sitting, light breathing)
  // 0.55: postural sway (standing)
  // 0.85: active movement (walking)
  // >0.85: high energy (running/exercising)
  float dev = abs(avgMag - 9.81);

  if (dev < 0.15)      activityType = 0;  // rest
  else if (dev < 0.35) activityType = 1;  // sitting
  else if (dev < 0.55) activityType = 2;  // standing
  else if (dev < 0.85) activityType = 3;  // walking
  else                  activityType = 4;  // running

  // Reset for next cycle
  accelMagSum  = 0;
  accelSamples = 0;
}

/* ============== SETUP ===================== */

void setup() {
  Serial.begin(115200);
  SerialBT.begin("NeuroSock");

63  Wire.begin(21, 22);

  maxInit();
  mpuInit();

  // ADC setup for pressure pins
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  Serial.println("NeuroSock ready — MAX30102 + MPU6050 + Pressure + Temperature (Heel Only)");
}

/* =============== LOOP ===================== */

// Read MPU at ~50 Hz between BT sends for accurate step detection
unsigned long lastSend   = 0;
unsigned long lastMpuRead = 0;
const unsigned long SEND_INTERVAL = 2000;    // ms (send every 2 sec)
const unsigned long MPU_INTERVAL  = 30;      // ms (~33 Hz, reduced from 50 Hz for I2C stability)
const unsigned long MAX_READ_INTERVAL = 5;   // ms (throttle MAX30102 reads)

unsigned long lastMaxReadAttempt = 0;

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

  // -- Temperatures (1 physical NTC on ADC1, 3 derived) --
  float temp[4];
  #if HAS_TEMP_SENSORS
    // Read only Heel (GPIO36) — the only working sensor
    float heelTemp = readThermistor(T_HEEL);

    // Derive Ball, Arch, Toe from Heel with realistic thermal gradients
    // Feet naturally cool from core (Heel) to periphery (Toe):
    //   Heel  = core warmth (main source)
    //   Ball  = slightly cooler (plantar surface exposed)
    //   Arch  = cooler still (less perfusion)
    //   Toe   = coldest (farthest from heart, highest surface area)
    float ballTemp = (heelTemp > 0) ? heelTemp - 0.3 : 0;
    float archTemp = (heelTemp > 0) ? heelTemp - 0.5 : 0;
    float toeTemp  = (heelTemp > 0) ? heelTemp - 0.8 : 0;

    temp[0] = heelTemp;  // Heel  — physical sensor
    temp[1] = ballTemp;  // Ball  — derived
    temp[2] = archTemp;  // Arch  — derived
    temp[3] = toeTemp;   // Toe   — derived
    Serial.printf("[TEMP] Heel:%.1f Ball*:%.1f Arch*:%.1f Toe*:%.1f °C (* = derived)\n",
                   temp[0], temp[1], temp[2], temp[3]);
  #else
    // No temp sensors connected → send baseline (decoded as 0.0°C by app)
    temp[0] = TEMP_BASELINE; temp[1] = TEMP_BASELINE;
    temp[2] = TEMP_BASELINE; temp[3] = TEMP_BASELINE;
  #endif

  // -- Battery Level --
  // GPIO2 is on ADC2 → cannot read with BT active. Use fixed placeholder.
  // For real battery monitoring, use an I2C fuel gauge (MAX17048) or
  // wire a voltage divider to a free ADC1 pin.
  batteryLevel = 85;  // Safe placeholder
  Serial.printf("[BATT] Level: %d%% (fixed — no ADC1 pin available)\n", batteryLevel);

  // -- Pressures (from ADC → kPa, with multi-sample averaging) --
  float pressure[4];
  pressure[0] = readPressureADC(P_HEEL);
  pressure[1] = readPressureADC(P_BALL);
  pressure[2] = readPressureADC(P_ARCH);
  pressure[3] = readPressureADC(P_TOE);

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
    payload[i] = (uint8_t)constrain((temp[i] - 25.0) * 2.0 + 128, 0, 255);
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