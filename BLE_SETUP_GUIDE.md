# NeuroSocks Real BLE Setup & Testing Guide

## 🎯 Overview

Your Flutter app now properly supports **real BLE communication** with your ESP32 NeuroSock device. The app:
- ✅ Scans for "NeuroSock" devices
- ✅ Connects automatically via SPP (Serial Port Profile)
- ✅ Decodes the 16-byte payload from your ESP32
- ✅ Displays real sensor data in the dashboard
- ✅ Handles disconnect & reconnect
- ✅ Falls back to Firestore data when offline

---

## 🔧 Hardware Setup

### ESP32 Requirements
Your ESP32 code already handles:
- ✅ BluetoothSerial with name "NeuroSock"
- ✅ 16-byte payload every 2 seconds via UART RX/TX characteristics
- ✅ Proper encoding/decoding of sensor values

**Verify your ESP32 code has:**
```cpp
SerialBT.begin("NeuroSock");   // Device name
SerialBT.write(payload, 16);    // Send 16-byte packet
```

---

## 📱 Flutter App Setup

### 1. **Device Settings** ✅ (Already Configured)
- Settings Screen → Device & Sensor
- Shows "No device connected" initially
- Tap "Connect" button

### 2. **Scanning Process** ✅ (Device Scan Screen)
**Location:** `lib/ui/screens/home/device_scan_screen.dart`

**What happens:**
1. Tap "Connect" in Settings
2. App scans for all BLE devices (15 seconds)
3. Filters devices starting with "NeuroSock"
4. Shows list of found devices
5. Tap to connect to your device

### 3. **Connection Flow**
```
Settings Screen
    ↓ (Tap "Connect")
Device Scan Screen
    ↓ (Scan & Find NeuroSock)
Connect to Device
    ↓ (Discover Services)
Start Streaming
    ↓ (Receive 16-byte packets)
Dashboard Shows Data
```

---

## 🚀 Step-by-Step Usage

### **First Time Setup:**

1. **Enable Bluetooth on your phone** (Android/iOS)

2. **Power on your ESP32** with the NeuroSock code

3. **Open NeuroSocks App** → Go to Settings

4. **Tap "Connect"** in the Device section
   - Wait for scanning (15 seconds)
   - See your "NeuroSock" device in the list

5. **Tap the device** to connect
   - Connection takes 5-10 seconds
   - See success message
   - Device name appears in Settings

6. **Go to Dashboard**
   - You'll see real sensor data streaming
   - Real-time values for:
     - Temperature (4 zones: Heel, Ball, Arch, Toe)
     - Pressure (4 zones)
     - SpO2 (blood oxygen)
     - Heart Rate
     - Step Count
     - Battery Level

### **Reconnecting:**

- **If disconnected:** Tap "Connect" → Select device → Done
- **Auto-reconnect:** (Coming soon in next update)
- **Data persistence:** Runs if BLE is offline; syncs when back online

---

## 📊 Payload Format Explanation

Your ESP32 sends this **16-byte packet every 2 seconds:**

```
Byte 0-3:   Temperatures (Heel, Ball, Arch, Toe)
            Formula: temp = 25.0 + (byte - 128) / 2.0
            Range: 17°C to 33°C

Byte 4-7:   Pressures (Heel, Ball, Arch, Toe)
            Formula: pressure = byte * 0.3
            Range: 0 to 77 kPa

Byte 8-9:   SpO2 (16-bit big-endian)
            Formula: spo2 = value / 100.0
            Range: 0% to 100%

Byte 10-11: Heart Rate (16-bit big-endian)
            Range: 0 to 300 BPM

Byte 12-13: Step Count (16-bit big-endian)
            Range: 0 to 65536

Byte 14:    Activity Type (0-4)
            0: Resting, 1: Sitting, 2: Standing
            3: Walking, 4: Running

Byte 15:    Battery Level (0-100%)
```

**Example Packet (hex):**
```
85 85 85 85 | 7D 7D 7D 7D | 02 6D | 00 48 | 02 F2 | 03 | 55
```

---

## 🔍 Debugging & Troubleshooting

### **Issue: Device not found during scan**

**Check:**
1. ESP32 is powered on and running NeuroSock code
2. Bluetooth is enabled on phone
3. Device name starts with "NeuroSock" (not "neurosock")
4. Check Serial Monitor output on ESP32:
   ```
   System ready
   IR: 12345 | Sent 16-byte packet
   ```

**Fix:**
- Restart app and try scanning again
- Restart ESP32
- Check ESP32 Bluetooth module is working

### **Issue: Connected but no data showing**

**Check:**
1. Streaming started (check terminal logs: "📡 Starting sensor data stream...")
2. Data is received (console shows: "✅ Parsed: Temp=...")
3. Go to Dashboard to see the data

**Fix:**
- Check BLE UART characteristics discovered correctly
- Verify 16-byte packets are being sent from ESP32
- Check app logs for parsing errors

### **Issue: Data looks wrong**

**Common causes:**
1. ESP32 sending incorrect payload (check Arduino code)
2. Wrong decoding formula (verify against payload format above)
3. Sensor calibration issue (check sensor readings directly on ESP32)

**Check logs:**
```dart
// Terminal shows parsing details:
✅ Parsed: Temp=32.5,33.2,31.8,32.9 
           Pressure=20.3,15.1,17.8,18.2 
           SpO2=98.5 HR=72
```

### **Issue: Connection drops frequently**

**Causes:**
- BLE range too far (move closer to device)
- Bluetooth interference (away from WiFi/microwave)
- Phone Bluetooth unstable (restart Bluetooth)

**Fix:**
- Disconnect and reconnect manually
- Keep device within 5-10 meters
- Turn off WiFi if possible

---

## 📱 Dashboard Features

Once connected, Dashboard shows:

### **Connection Status** (Top banner)
- Green: Connected & Streaming
- Orange: Connected but not streaming
- Red: Disconnected (showing previous data)

### **Risk Gauge** (Center)
- Real-time risk score (0-100)
- Color: Green/Yellow/Orange/Red

### **Foot Heatmap** (Pressure & Temperature)
- 4 zones showing live data
- Color intensity = value level

### **Quick Stats**
- Current values for all sensors
- Last update time

### **Recent Alerts**
- Yesterday's critical alerts
- Risk factors detected

---

## 🔄 Data Flow

```
ESP32 Device (Every 2 seconds)
    ↓ BLE Serial (SPP)
Phone Bluetooth
    ↓ (decodes 16-byte packet)
RealBleService
    ↓
SensorProvider (updateUI)
    ↓
Dashboard, Sensors, Alerts Screens
    ↓ (async save)
Local Storage (Hive) + Firestore
```

---

## 🔐 Security & Privacy

✅ **Local First:** All data stored locally by default
✅ **Firebase Optional:** Only syncs when user logs in
✅ **No personal data:** Only health metrics, not location or identity
✅ **Offline Mode:** Works fully without internet

---

## 📋 Checklist

### Before First Use:
- [ ] ESP32 uploaded with NeuroSock code
- [ ] ESP32 powered on and running
- [ ] Phone Bluetooth enabled
- [ ] Flutter app installed

### During First Connection:
- [ ] Open Settings
- [ ] Tap "Connect"
- [ ] Wait 15 seconds for scan
- [ ] See "NeuroSock" in device list
- [ ] Tap to connect
- [ ] Wait for connected message
- [ ] Go to Dashboard

### After Connection:
- [ ] See sensor data in Dashboard
- [ ] See real temperatures/pressures
- [ ] See heart rate & SpO2
- [ ] See battery level
- [ ] See step count updating

---

## 🆘 Support

### Check These Logs:

**Flutter Console (VS Code):**
```
[RealBleService] 🔍 Starting BLE scan for NeuroSock devices...
[RealBleService] ✅ Found device: NeuroSock (80:7F:B4:XX:XX:XX)
[RealBleService] 🔗 Connecting to NeuroSock...
[RealBleService] ✅ Connected to device
[RealBleService] 📡 Starting sensor data stream...
[RealBleService] ✅ Parsed: Temp=32.5,33.2,31.8,32.9...
```

**ESP32 Serial Monitor (Arduino IDE):**
```
System ready
IR: 12345 | Sent 16-byte packet
IR: 12346 | Sent 16-byte packet
...
```

### Key Files:
- **BLE Service:** `lib/data/services/real_ble_service.dart`
- **Sensor Provider:** `lib/providers/sensor_provider.dart`
- **Device Scan Screen:** `lib/ui/screens/home/device_scan_screen.dart`
- **Settings Screen:** `lib/ui/screens/home/settings_screen.dart`
- **Dashboard:** `lib/ui/screens/home/dashboard_screen.dart`

---

## 🎉 You're Ready!

Your NeuroSocks app now has **full real BLE support**. Enjoy monitoring foot health in real-time! 🏥💙
