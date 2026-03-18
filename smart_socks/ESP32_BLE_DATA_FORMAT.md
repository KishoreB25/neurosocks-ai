# 📡 ESP32 BLE Data Format - Smart Socks

This document shows your friend **EXACTLY** how to format data from ESP32 to send via Bluetooth to the mobile app.

---

## 🎯 BLE Packet Structure

### **Total: 16 Bytes (Minimum)**

```
┌─────┬──────────────────────────────────────────────────┐
│ Idx │ Field              │ Size │ Type    │ Description │
├─────┼────────────────────┼──────┼─────────┼─────────────┤
│ 0   │ Temperature 1      │ 1    │ uint8   │ Heel        │
│ 1   │ Temperature 2      │ 1    │ uint8   │ Ball        │
│ 2   │ Temperature 3      │ 1    │ uint8   │ Arch        │
│ 3   │ Temperature 4      │ 1    │ uint8   │ Toe         │
├─────┼────────────────────┼──────┼─────────┼─────────────┤
│ 4   │ Pressure 1         │ 1    │ uint8   │ Heel        │
│ 5   │ Pressure 2         │ 1    │ uint8   │ Ball        │
│ 6   │ Pressure 3         │ 1    │ uint8   │ Arch        │
│ 7   │ Pressure 4         │ 1    │ uint8   │ Toe         │
├─────┼────────────────────┼──────┼─────────┼─────────────┤
│ 8   │ SpO2 High Byte     │ 1    │ uint8   │ Part of     │
│ 9   │ SpO2 Low Byte      │ 1    │ uint8   │ uint16      │
├─────┼────────────────────┼──────┼─────────┼─────────────┤
│ 10  │ Heart Rate High    │ 1    │ uint8   │ Part of     │
│ 11  │ Heart Rate Low     │ 1    │ uint8   │ uint16      │
├─────┼────────────────────┼──────┼─────────┼─────────────┤
│ 12  │ Step Count High    │ 1    │ uint8   │ Part of     │
│ 13  │ Step Count Low     │ 1    │ uint8   │ uint16      │
├─────┼────────────────────┼──────┼─────────┼─────────────┤
│ 14  │ Activity Type      │ 1    │ uint8   │ 0-4         │
├─────┼────────────────────┼──────┼─────────┼─────────────┤
│ 15  │ Battery Level      │ 1    │ uint8   │ 0-100%      │
└─────┴────────────────────┴──────┴─────────┴─────────────┘
```

---

## 🔢 Detailed Data Conversion

### **Temperatures (Bytes 0-3)**

**Received Formula:**
```
Actual Temperature (°C) = 25.0 + (received_byte - 128) / 2.0
```

**Send Formula (for ESP32):**
```c
// If you have actual temperature in °C:
uint8_t temp_byte = (int)((actual_temp - 25.0) * 2.0) + 128;
```

**Range:**
- Actual Temperature: -40°C to +120°C
- Raw Byte Value: 0 to 255

**Example:**
- If sensor reads **32.5°C** → Send byte: `(32.5 - 25.0) * 2 + 128 = 143`
- Mobile receives: `25.0 + (143 - 128) / 2 = 32.5°C` ✅

---

### **Pressures (Bytes 4-7)**

**Received Formula:**
```
Actual Pressure (kPa) = received_byte * 0.3
```

**Send Formula (for ESP32):**
```c
// If you have actual pressure in kPa:
uint8_t pressure_byte = (int)(actual_pressure / 0.3);
```

**Range:**
- Actual Pressure: 0 to 77 kPa (roughly)
- Raw Byte Value: 0 to 255

**Example:**
- If sensor reads **45 kPa** → Send byte: `45 / 0.3 = 150`
- Mobile receives: `150 * 0.3 = 45 kPa` ✅

---

### **SpO2 - Blood Oxygen Level (Bytes 8-9)**

**16-bit unsigned integer (Big Endian)**

**Received Formula:**
```
Actual SpO2 (%) = ((byte8 << 8) | byte9) / 100.0
```

**Send Formula (for ESP32):**
```c
// If you have SpO2 as percentage (e.g., 98.5%):
uint16_t spo2_raw = (uint16_t)(actual_spo2 * 100);
uint8_t byte8 = (spo2_raw >> 8) & 0xFF;  // High byte
uint8_t byte9 = spo2_raw & 0xFF;         // Low byte
```

**Range:**
- Actual SpO2: 0.0% to 100.0%
- Raw Value: 0 to 10000

**Example:**
- If SpO2 is **98.5%** → Raw: `9850`
  - Byte 8: `9850 >> 8 = 38` (decimal)
  - Byte 9: `9850 & 0xFF = 154` (decimal)
- Mobile receives: `((38 << 8) | 154) / 100 = 98.5%` ✅

---

### **Heart Rate (Bytes 10-11)**

**16-bit unsigned integer (Big Endian)**

**Received Formula:**
```
Actual Heart Rate (BPM) = (byte10 << 8) | byte11
```

**Send Formula (for ESP32):**
```c
// If you have heart rate as integer (e.g., 72 BPM):
uint16_t hr_raw = actual_heart_rate;
uint8_t byte10 = (hr_raw >> 8) & 0xFF;  // High byte
uint8_t byte11 = hr_raw & 0xFF;         // Low byte
```

**Range:**
- Actual Heart Rate: 0 to 65535 BPM (but realistic: 40-200 BPM)
- Raw Value: 0 to 65535

**Example:**
- If heart rate is **72 BPM** → Raw: `72`
  - Byte 10: `72 >> 8 = 0`
  - Byte 11: `72 & 0xFF = 72`
- Mobile receives: `(0 << 8) | 72 = 72 BPM` ✅

---

### **Step Count (Bytes 12-13)**

**16-bit unsigned integer (Big Endian)**

**Received Formula:**
```
Actual Step Count = (byte12 << 8) | byte13
```

**Send Formula (for ESP32):**
```c
// If you have step count:
uint16_t steps_raw = total_steps;
uint8_t byte12 = (steps_raw >> 8) & 0xFF;  // High byte
uint8_t byte13 = steps_raw & 0xFF;         // Low byte
```

**Range:**
- Actual Steps: 0 to 65535 steps

**Example:**
- If step count is **1250** → Raw: `1250` (0x04E2)
  - Byte 12: `1250 >> 8 = 4`
  - Byte 13: `1250 & 0xFF = 226`
- Mobile receives: `(4 << 8) | 226 = 1250 steps` ✅

---

### **Activity Type (Byte 14)**

**Single byte, values 0-4**

**Mapping:**
```
0 = Resting
1 = Sitting
2 = Standing
3 = Walking
4 = Running
```

**Send Formula (for ESP32):**
```c
// Simply send the activity code:
uint8_t activity_byte = 3;  // For Walking
```

**Example:**
- If user is walking → Send byte: `3`
- Mobile receives and interprets as: `Walking` ✅

---

### **Battery Level (Byte 15)**

**Single byte, 0-100%**

**Formula:**
```
Actual Battery % = received_byte
```

**Send Formula (for ESP32):**
```c
// If you have battery percentage (0-100):
uint8_t battery_byte = (uint8_t)battery_percentage;
```

**Range:**
- Actual Battery: 0% to 100%
- Raw Byte Value: 0 to 100

**Example:**
- If battery is **85%** → Send byte: `85`
- Mobile receives: `Battery = 85%` ✅

---

## 💻 Complete ESP32 Code Example

```cpp
#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

void setup() {
  Serial.begin(115200);
  SerialBT.begin("NeuroSock");  // Device name for Bluetooth discovery
}

void loop() {
  // Read actual sensor values
  float temp_heel = readTemperature(HEEL);      // Example: 32.5°C
  float temp_ball = readTemperature(BALL);      // Example: 33.2°C
  float temp_arch = readTemperature(ARCH);      // Example: 31.8°C
  float temp_toe = readTemperature(TOE);        // Example: 32.9°C
  
  float pressure_heel = readPressure(HEEL);     // Example: 45 kPa
  float pressure_ball = readPressure(BALL);     // Example: 50 kPa
  float pressure_arch = readPressure(ARCH);     // Example: 20 kPa
  float pressure_toe = readPressure(TOE);       // Example: 40 kPa
  
  float spo2 = readSpO2();                      // Example: 98.5%
  int heart_rate = readHeartRate();             // Example: 72 BPM
  int steps = getTotalSteps();                  // Example: 1250
  int activity = detectActivity();              // 0-4
  int battery = getBatteryPercent();            // 0-100
  
  // Create 16-byte payload
  uint8_t payload[16];
  
  // Temperatures (Bytes 0-3)
  payload[0] = (int)((temp_heel - 25.0) * 2.0) + 128;
  payload[1] = (int)((temp_ball - 25.0) * 2.0) + 128;
  payload[2] = (int)((temp_arch - 25.0) * 2.0) + 128;
  payload[3] = (int)((temp_toe - 25.0) * 2.0) + 128;
  
  // Pressures (Bytes 4-7)
  payload[4] = (int)(pressure_heel / 0.3);
  payload[5] = (int)(pressure_ball / 0.3);
  payload[6] = (int)(pressure_arch / 0.3);
  payload[7] = (int)(pressure_toe / 0.3);
  
  // SpO2 (Bytes 8-9) - uint16 big endian
  uint16_t spo2_raw = (uint16_t)(spo2 * 100);
  payload[8] = (spo2_raw >> 8) & 0xFF;
  payload[9] = spo2_raw & 0xFF;
  
  // Heart Rate (Bytes 10-11) - uint16 big endian
  uint16_t hr_raw = (uint16_t)heart_rate;
  payload[10] = (hr_raw >> 8) & 0xFF;
  payload[11] = hr_raw & 0xFF;
  
  // Step Count (Bytes 12-13) - uint16 big endian
  uint16_t steps_raw = (uint16_t)steps;
  payload[12] = (steps_raw >> 8) & 0xFF;
  payload[13] = steps_raw & 0xFF;
  
  // Activity Type (Byte 14)
  payload[14] = (uint8_t)activity;
  
  // Battery Level (Byte 15)
  payload[15] = (uint8_t)battery;
  
  // Send via BLE
  SerialBT.write(payload, 16);
  
  // Send every 2 seconds (or adjust as needed)
  delay(2000);
}
```

---

## 📋 Example Payload (Raw Bytes)

**If sensors read:**
- Temperatures: 32.5°C, 33.2°C, 31.8°C, 32.9°C
- Pressures: 45 kPa, 50 kPa, 20 kPa, 40 kPa
- SpO2: 98.5%
- Heart Rate: 72 BPM
- Steps: 1250
- Activity: Walking (3)
- Battery: 85%

**Raw bytes to send (hex):**
```
8F 94 8B 91 96 A3 41 75 26 8A 00 48 04 E2 03 55
```

**Breakdown:**
```
8F  = Temperature 1:    (32.5 - 25.0) * 2 + 128 = 143 = 0x8F
94  = Temperature 2:    (33.2 - 25.0) * 2 + 128 = 148 = 0x94
8B  = Temperature 3:    (31.8 - 25.0) * 2 + 128 = 139 = 0x8B
91  = Temperature 4:    (32.9 - 25.0) * 2 + 128 = 145 = 0x91

96  = Pressure 1:       45 / 0.3 = 150 = 0x96
A3  = Pressure 2:       50 / 0.3 = 167 = 0xA3
41  = Pressure 3:       20 / 0.3 = 67  = 0x41
75  = Pressure 4:       40 / 0.3 = 133 = 0x75

26 8A = SpO2:           9850 = 0x268A
00 48 = Heart Rate:     72   = 0x0048
04 E2 = Steps:          1250 = 0x04E2
03    = Activity:       3 (Walking)
55    = Battery:        85%
```

---

## ✅ Checklist for ESP32 Implementation

- [ ] Read all 4 temperature sensors
- [ ] Read all 4 pressure sensors
- [ ] Read SpO2 sensor
- [ ] Read heart rate sensor
- [ ] Track step count
- [ ] Detect activity type (resting/sitting/standing/walking/running)
- [ ] Read battery level
- [ ] Convert all values using formulas above
- [ ] Pack into 16-byte array
- [ ] Send via BLE characteristic notify
- [ ] Send every 2 seconds (or your preferred interval)

---

## 🔧 BLE Setup on ESP32

### **Service UUID:**
```
Different UUIDs can be used, but ensure they're consistent
Example: 180D (Heart Rate Service - standard)
```

### **Characteristic UUID for Data:**
```
Example: 2A37 (Heart Rate Measurement - standard)
Or custom: 550e8400-e29b-41d4-a716-446655440000
```

---

## 📱 Verification

After implementing on ESP32:
1. **Turn on Mock mode OFF** in mobile app settings
2. Connect to "NeuroSock" device
3. Open app and start streaming
4. Check **browser console** for received data
5. Check **Firestore** for stored readings

You should see in console:
```
📊 Received reading - Temp: [32.5, 33.2, 31.8, 32.9], Pressure: [45.0, 50.0, 20.0, 40.0]
💾 Saving sensor reading...
✅ Sensor reading saved successfully
```

---

## 🎯 Key Points for Your Friend

✅ **16 bytes total** - No more, no less  
✅ **Big Endian** for multi-byte values (temperatures use formula)  
✅ **Send every 2 seconds** - Or adjust period as needed  
✅ **BLE Notify** - Use characteristic notifications  
✅ **Temperature formula** - Don't forget the (x - 128) / 2 conversion!  
✅ **Pressure formula** - Keep the × 0.3 scaling  

Good luck! 🚀
