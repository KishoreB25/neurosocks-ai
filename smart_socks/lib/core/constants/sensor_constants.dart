/// Sensor thresholds, ranges, and configuration constants
/// Based on medical literature for diabetic foot monitoring
class SensorConstants {
  SensorConstants._();

  // ============== Temperature Thresholds (°C) ==============
  /// Normal foot temperature range
  static const double tempMin = 25.0;
  static const double tempMax = 37.0;
  static const double tempNormalMin = 28.0;
  static const double tempNormalMax = 34.0;

  /// Warning thresholds
  static const double tempWarningLow = 27.0; // Too cold - circulation issue
  static const double tempWarningHigh = 35.0; // Elevated - inflammation

  /// Critical thresholds
  static const double tempCriticalLow = 25.0; // Very cold
  static const double tempCriticalHigh = 37.0; // Hot - infection risk

  /// Asymmetry threshold (difference between left and right foot)
  static const double tempAsymmetryWarning = 2.0; // °C difference
  static const double tempAsymmetryCritical = 3.5; // °C difference

  /// Rate of change threshold (per hour)
  static const double tempRateOfChangeWarning = 1.5; // °C/hour

  // ============== Pressure Thresholds (kPa) ==============
  /// Normal pressure range during standing/walking
  static const double pressureMin = 0.0;
  static const double pressureMax = 150.0;
  static const double pressureNormalMin = 10.0;
  static const double pressureNormalMax = 60.0;

  /// Warning thresholds
  static const double pressureWarning = 80.0; // Elevated pressure
  static const double pressureHigh = 100.0; // High pressure

  /// Critical threshold
  static const double pressureCritical = 120.0; // Dangerous pressure

  /// Duration thresholds (sustained high pressure)
  static const int pressureSustainedWarningMinutes = 15;
  static const int pressureSustainedCriticalMinutes = 30;

  /// Pressure spike detection
  static const double pressureSpikeThreshold = 30.0; // Sudden increase

  // ============== Blood Oxygen (SpO2) Thresholds (%) ==============
  static const double spo2Min = 70.0;
  static const double spo2Max = 100.0;
  static const double spo2Normal = 95.0; // Normal lower bound
  static const double spo2Warning = 92.0; // Mild hypoxemia
  static const double spo2Low = 90.0; // Moderate hypoxemia
  static const double spo2Critical = 85.0; // Severe hypoxemia

  // ============== Heart Rate Thresholds (BPM) ==============
  static const int hrMin = 40;
  static const int hrMax = 200;
  static const int hrNormalMin = 60;
  static const int hrNormalMax = 100;
  static const int hrWarningLow = 50;
  static const int hrWarningHigh = 110;
  static const int hrCriticalLow = 40;
  static const int hrCriticalHigh = 130;

  // ============== IMU / Gait Thresholds ==============
  /// Accelerometer range (m/s²)
  static const double accMin = -20.0;
  static const double accMax = 20.0;
  static const double accNormalGravity = 9.8;

  /// Gyroscope range (deg/s)
  static const double gyroMin = -250.0;
  static const double gyroMax = 250.0;

  /// Gait stability threshold
  static const double gaitStabilityWarning = 0.7; // Below this = unstable
  static const double gaitStabilityCritical = 0.5;

  /// Step frequency (steps per minute) - normal walking
  static const int stepFrequencyMin = 80;
  static const int stepFrequencyMax = 130;

  // ============== Risk Score Weights ==============
  /// Weights for calculating overall risk score
  static const double weightTemperature = 0.30; // 30%
  static const double weightPressure = 0.35; // 35%
  static const double weightCirculation = 0.20; // 20% (SpO2 + HR)
  static const double weightGait = 0.15; // 15%

  // ============== Risk Score Ranges ==============
  static const int riskScoreMin = 0;
  static const int riskScoreMax = 100;
  static const int riskLowMax = 30;
  static const int riskModerateMax = 50;
  static const int riskHighMax = 70;
  // Above 70 = Critical

  // ============== Sampling Configuration ==============
  /// How often to read sensors (milliseconds)
  static const int sensorSamplingIntervalMs = 2000; // 2 seconds

  /// How often to calculate risk score
  static const int riskCalculationIntervalMs = 5000; // 5 seconds

  /// How often to sync to cloud (milliseconds)
  static const int cloudSyncIntervalMs = 60000; // 1 minute

  /// Data points to keep in memory for trend analysis
  static const int historyBufferSize = 100;

  /// Readings per day estimate (for storage planning)
  static const int readingsPerHour = 1800; // At 2-second intervals

  // ============== BLE Configuration ==============
  /// Service UUID for Smart Sock device
  static const String bleServiceUuid = '0000180D-0000-1000-8000-00805f9b34fb';

  /// Characteristic UUIDs
  static const String bleSensorCharUuid =
      '00002A37-0000-1000-8000-00805f9b34fb';
  static const String bleBatteryCharUuid =
      '00002A19-0000-1000-8000-00805f9b34fb';

  /// Device name prefix for scanning
  static const String bleDeviceNamePrefix = 'NeuroSock';

  /// Connection timeout
  static const int bleConnectionTimeoutSec = 10;

  /// Reconnection attempts
  static const int bleReconnectAttempts = 3;

  // ============== Sensor Zone Configuration ==============
  /// Number of temperature sensors
  static const int temperatureSensorCount = 4;

  /// Number of pressure sensors
  static const int pressureSensorCount = 4;

  /// Zone names
  static const List<String> zoneNames = ['Heel', 'Ball', 'Arch', 'Toe'];

  /// Zone indices
  static const int zoneHeel = 0;
  static const int zoneBall = 1;
  static const int zoneArch = 2;
  static const int zoneToe = 3;

  // ============== Alert Configuration ==============
  /// Cooldown period between same type alerts (minutes)
  static const int alertCooldownMinutes = 5;

  /// Maximum alerts to store
  static const int maxStoredAlerts = 100;

  /// Auto-dismiss info alerts after (seconds)
  static const int infoAlertDismissSeconds = 30;

  // ============== Battery Thresholds ==============
  static const int batteryLow = 20;
  static const int batteryCritical = 10;
  static const int batteryFull = 100;

  // ============== Activity Detection ==============
  /// Thresholds for activity classification based on accelerometer variance
  static const double activityRestingThreshold = 0.1;
  static const double activityStandingThreshold = 0.3;
  static const double activityWalkingThreshold = 1.0;
  static const double activityRunningThreshold = 3.0;

  // ============== Helper Methods ==============

  /// Check if temperature is in normal range
  static bool isTemperatureNormal(double temp) {
    return temp >= tempNormalMin && temp <= tempNormalMax;
  }

  /// Check if pressure is in normal range
  static bool isPressureNormal(double pressure) {
    return pressure >= pressureNormalMin && pressure <= pressureNormalMax;
  }

  /// Check if SpO2 is normal
  static bool isSpO2Normal(double spo2) {
    return spo2 >= spo2Normal;
  }

  /// Check if heart rate is normal
  static bool isHeartRateNormal(int hr) {
    return hr >= hrNormalMin && hr <= hrNormalMax;
  }

  /// Get risk level from score
  static String getRiskLevelName(int score) {
    if (score <= riskLowMax) return 'Low';
    if (score <= riskModerateMax) return 'Moderate';
    if (score <= riskHighMax) return 'High';
    return 'Critical';
  }

  /// Get zone name by index
  static String getZoneName(int index) {
    if (index >= 0 && index < zoneNames.length) {
      return zoneNames[index];
    }
    return 'Unknown';
  }
}
