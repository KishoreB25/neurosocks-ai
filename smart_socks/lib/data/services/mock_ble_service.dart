// Simulates ESP32 sensor data with realistic temperature, pressure, SpO2, IMU values. 
// Includes activity simulation, anomaly generation, and configurable streaming

import 'dart:async';
import 'dart:math';
import '../models/sensor_reading.dart';

/// Mock BLE Service that simulates sensor data from ESP32
/// Use this for development/testing until hardware is ready
class MockBleService {
  // Singleton pattern
  static final MockBleService _instance = MockBleService._internal();
  factory MockBleService() => _instance;
  MockBleService._internal();

  // Stream controller for sensor readings
  StreamController<SensorReading>? _streamController;
  Timer? _timer;
  final Random _random = Random();

  // Connection state
  bool _isConnected = false;
  bool _isStreaming = false;

  // Simulated device info
  String _deviceName = 'NeuroSock-Mock';
  int _batteryLevel = 100;

  // Baseline values (simulating a healthy foot)
  final List<double> _baseTemperatures = [31.5, 32.0, 31.0, 32.5]; // Heel, Ball, Arch, Toe
  final List<double> _basePressures = [35.0, 45.0, 20.0, 40.0]; // kPa
  double _baseSpO2 = 98.0;
  int _baseHeartRate = 72;
  int _stepCount = 0;

  // Anomaly simulation settings
  bool _simulateAnomalies = true;
  double _anomalyProbability = 0.05; // 5% chance per reading
  int _currentAnomalyZone = -1;
  int _anomalyDuration = 0;

  // Activity simulation
  ActivityType _currentActivity = ActivityType.resting;
  int _activityDuration = 0;

  // ============== Getters ==============

  bool get isConnected => _isConnected;
  bool get isStreaming => _isStreaming;
  String get deviceName => _deviceName;
  int get batteryLevel => _batteryLevel;

  /// Stream of sensor readings
  Stream<SensorReading>? get sensorStream => _streamController?.stream;

  // ============== Connection Methods ==============

  /// Simulate connecting to the device
  Future<bool> connect() async {
    if (_isConnected) return true;

    // Simulate connection delay
    await Future.delayed(const Duration(seconds: 2));

    // 95% success rate
    if (_random.nextDouble() < 0.95) {
      _isConnected = true;
      _deviceName = 'NeuroSock-${_random.nextInt(9999).toString().padLeft(4, '0')}';
      _batteryLevel = 80 + _random.nextInt(21); // 80-100%
      return true;
    }

    return false;
  }

  /// Simulate disconnecting from the device
  Future<void> disconnect() async {
    await stopStreaming();
    _isConnected = false;
  }

  // ============== Streaming Methods ==============

  /// Start streaming sensor data
  Future<void> startStreaming({
    Duration interval = const Duration(seconds: 2),
    bool simulateAnomalies = true,
  }) async {
    if (_isStreaming) return;
    if (!_isConnected) {
      final connected = await connect();
      if (!connected) throw Exception('Failed to connect to device');
    }

    _simulateAnomalies = simulateAnomalies;
    _streamController = StreamController<SensorReading>.broadcast();
    _isStreaming = true;

    // Start the timer to emit readings
    _timer = Timer.periodic(interval, (_) {
      if (_streamController != null && !_streamController!.isClosed) {
        final reading = _generateReading();
        _streamController!.add(reading);
      }
    });

    // Emit first reading immediately
    if (_streamController != null && !_streamController!.isClosed) {
      _streamController!.add(_generateReading());
    }
  }

  /// Stop streaming sensor data
  Future<void> stopStreaming() async {
    _isStreaming = false;
    _timer?.cancel();
    _timer = null;
    await _streamController?.close();
    _streamController = null;
  }

  // ============== Data Generation ==============

  /// Generate a realistic sensor reading
  SensorReading _generateReading() {
    // Update activity periodically
    _updateActivity();

    // Update battery (drain slowly)
    _updateBattery();

    // Check for anomaly state
    _updateAnomalyState();

    // Generate temperatures
    final temperatures = _generateTemperatures();

    // Generate pressures based on activity
    final pressures = _generatePressures();

    // Generate SpO2 and heart rate
    final spO2 = _generateSpO2();
    final heartRate = _generateHeartRate();

    // Generate IMU data
    final accelerometer = _generateAccelerometer();
    final gyroscope = _generateGyroscope();

    // Update step count
    _updateStepCount();

    return SensorReading(
      timestamp: DateTime.now(),
      temperatures: temperatures,
      pressures: pressures,
      spO2: spO2,
      heartRate: heartRate,
      accelerometer: accelerometer,
      gyroscope: gyroscope,
      stepCount: _stepCount,
      batteryLevel: _batteryLevel,
      activityType: _currentActivity,
    );
  }

  /// Generate temperature readings with realistic variations
  List<double> _generateTemperatures() {
    final temps = <double>[];

    for (int i = 0; i < 4; i++) {
      double temp = _baseTemperatures[i];

      // Add small random variation (±0.5°C)
      temp += (_random.nextDouble() - 0.5) * 1.0;

      // Activity-based adjustment
      if (_currentActivity == ActivityType.walking ||
          _currentActivity == ActivityType.running) {
        temp += 0.5 + _random.nextDouble() * 0.5; // Warmer when active
      }

      // Simulate anomaly (hotspot)
      if (_simulateAnomalies && _currentAnomalyZone == i && _anomalyDuration > 0) {
        temp += 2.0 + _random.nextDouble() * 2.0; // 2-4°C higher
      }

      temps.add(double.parse(temp.toStringAsFixed(1)));
    }

    return temps;
  }

  /// Generate pressure readings based on activity
  List<double> _generatePressures() {
    final pressures = <double>[];

    for (int i = 0; i < 4; i++) {
      double pressure = _basePressures[i];

      // Activity-based pressure
      switch (_currentActivity) {
        case ActivityType.resting:
          pressure *= 0.3; // Low pressure when resting
          break;
        case ActivityType.sitting:
          pressure *= 0.4;
          break;
        case ActivityType.standing:
          pressure *= 0.9;
          break;
        case ActivityType.walking:
          // Simulate walking pattern - alternating pressure
          final phase = DateTime.now().millisecond % 1000;
          if (phase < 500) {
            pressure *= (i == 0 || i == 3) ? 1.2 : 0.5; // Heel-toe
          } else {
            pressure *= (i == 1 || i == 2) ? 1.1 : 0.6; // Ball-arch
          }
          break;
        case ActivityType.running:
          pressure *= 1.5 + _random.nextDouble() * 0.5;
          break;
        case ActivityType.unknown:
          pressure *= 0.5;
          break;
      }

      // Add random variation (±10%)
      pressure *= (0.9 + _random.nextDouble() * 0.2);

      // Simulate pressure anomaly
      if (_simulateAnomalies &&
          _currentAnomalyZone == i &&
          _anomalyDuration > 0 &&
          _random.nextBool()) {
        pressure *= 1.8; // High pressure spike
      }

      pressures.add(double.parse(pressure.clamp(0, 150).toStringAsFixed(1)));
    }

    return pressures;
  }

  /// Generate SpO2 reading
  double _generateSpO2() {
    double spO2 = _baseSpO2;

    // Small random variation
    spO2 += (_random.nextDouble() - 0.5) * 2;

    // Occasional dip
    if (_simulateAnomalies && _random.nextDouble() < 0.02) {
      spO2 -= 3 + _random.nextDouble() * 5; // Drop 3-8%
    }

    return double.parse(spO2.clamp(85, 100).toStringAsFixed(1));
  }

  /// Generate heart rate reading
  int _generateHeartRate() {
    int hr = _baseHeartRate;

    // Activity-based adjustment
    switch (_currentActivity) {
      case ActivityType.resting:
        hr = 60 + _random.nextInt(15);
        break;
      case ActivityType.sitting:
        hr = 65 + _random.nextInt(15);
        break;
      case ActivityType.standing:
        hr = 70 + _random.nextInt(20);
        break;
      case ActivityType.walking:
        hr = 85 + _random.nextInt(25);
        break;
      case ActivityType.running:
        hr = 120 + _random.nextInt(40);
        break;
      case ActivityType.unknown:
        hr = 70 + _random.nextInt(20);
        break;
    }

    // Small random variation
    hr += _random.nextInt(5) - 2;

    return hr.clamp(40, 180);
  }

  /// Generate accelerometer data
  AccelerometerData _generateAccelerometer() {
    double x = 0, y = 0, z = 9.8;

    switch (_currentActivity) {
      case ActivityType.resting:
        x = (_random.nextDouble() - 0.5) * 0.1;
        y = (_random.nextDouble() - 0.5) * 0.1;
        z = 9.8 + (_random.nextDouble() - 0.5) * 0.1;
        break;
      case ActivityType.sitting:
      case ActivityType.standing:
        x = (_random.nextDouble() - 0.5) * 0.3;
        y = (_random.nextDouble() - 0.5) * 0.3;
        z = 9.8 + (_random.nextDouble() - 0.5) * 0.2;
        break;
      case ActivityType.walking:
        x = (_random.nextDouble() - 0.5) * 2;
        y = (_random.nextDouble() - 0.5) * 1;
        z = 9.8 + (_random.nextDouble() - 0.5) * 3;
        break;
      case ActivityType.running:
        x = (_random.nextDouble() - 0.5) * 5;
        y = (_random.nextDouble() - 0.5) * 3;
        z = 9.8 + (_random.nextDouble() - 0.5) * 6;
        break;
      case ActivityType.unknown:
        x = (_random.nextDouble() - 0.5) * 0.5;
        y = (_random.nextDouble() - 0.5) * 0.5;
        z = 9.8;
        break;
    }

    return AccelerometerData(
      x: double.parse(x.toStringAsFixed(2)),
      y: double.parse(y.toStringAsFixed(2)),
      z: double.parse(z.toStringAsFixed(2)),
    );
  }

  /// Generate gyroscope data
  GyroscopeData _generateGyroscope() {
    double x = 0, y = 0, z = 0;

    switch (_currentActivity) {
      case ActivityType.resting:
      case ActivityType.sitting:
        x = (_random.nextDouble() - 0.5) * 2;
        y = (_random.nextDouble() - 0.5) * 2;
        z = (_random.nextDouble() - 0.5) * 2;
        break;
      case ActivityType.standing:
        x = (_random.nextDouble() - 0.5) * 5;
        y = (_random.nextDouble() - 0.5) * 5;
        z = (_random.nextDouble() - 0.5) * 3;
        break;
      case ActivityType.walking:
        x = (_random.nextDouble() - 0.5) * 30;
        y = (_random.nextDouble() - 0.5) * 20;
        z = (_random.nextDouble() - 0.5) * 15;
        break;
      case ActivityType.running:
        x = (_random.nextDouble() - 0.5) * 60;
        y = (_random.nextDouble() - 0.5) * 40;
        z = (_random.nextDouble() - 0.5) * 30;
        break;
      case ActivityType.unknown:
        x = (_random.nextDouble() - 0.5) * 5;
        y = (_random.nextDouble() - 0.5) * 5;
        z = (_random.nextDouble() - 0.5) * 5;
        break;
    }

    return GyroscopeData(
      x: double.parse(x.toStringAsFixed(2)),
      y: double.parse(y.toStringAsFixed(2)),
      z: double.parse(z.toStringAsFixed(2)),
    );
  }

  // ============== State Updates ==============

  /// Update simulated activity
  void _updateActivity() {
    _activityDuration++;

    // Change activity every 10-30 readings
    if (_activityDuration > 10 + _random.nextInt(20)) {
      _activityDuration = 0;

      // Weighted random activity selection
      final roll = _random.nextDouble();
      if (roll < 0.3) {
        _currentActivity = ActivityType.resting;
      } else if (roll < 0.5) {
        _currentActivity = ActivityType.sitting;
      } else if (roll < 0.7) {
        _currentActivity = ActivityType.standing;
      } else if (roll < 0.95) {
        _currentActivity = ActivityType.walking;
      } else {
        _currentActivity = ActivityType.running;
      }
    }
  }

  /// Update step count based on activity
  void _updateStepCount() {
    if (_currentActivity == ActivityType.walking) {
      _stepCount += 1 + _random.nextInt(2); // 1-2 steps per reading
    } else if (_currentActivity == ActivityType.running) {
      _stepCount += 2 + _random.nextInt(3); // 2-4 steps per reading
    }
  }

  /// Update battery level (slow drain)
  void _updateBattery() {
    // Drain ~1% every 50 readings (about 100 seconds at 2s intervals)
    if (_random.nextInt(50) == 0 && _batteryLevel > 0) {
      _batteryLevel--;
    }
  }

  /// Update anomaly simulation state
  void _updateAnomalyState() {
    if (!_simulateAnomalies) return;

    if (_anomalyDuration > 0) {
      _anomalyDuration--;
      if (_anomalyDuration == 0) {
        _currentAnomalyZone = -1;
      }
    } else if (_random.nextDouble() < _anomalyProbability) {
      // Start new anomaly
      _currentAnomalyZone = _random.nextInt(4);
      _anomalyDuration = 3 + _random.nextInt(5); // 3-7 readings
    }
  }

  // ============== Configuration ==============

  /// Set anomaly simulation probability (0.0 - 1.0)
  void setAnomalyProbability(double probability) {
    _anomalyProbability = probability.clamp(0.0, 1.0);
  }

  /// Enable/disable anomaly simulation
  void setSimulateAnomalies(bool enable) {
    _simulateAnomalies = enable;
    if (!enable) {
      _currentAnomalyZone = -1;
      _anomalyDuration = 0;
    }
  }

  /// Force a specific activity (for testing)
  void setActivity(ActivityType activity) {
    _currentActivity = activity;
    _activityDuration = 0;
  }

  /// Reset step count
  void resetStepCount() {
    _stepCount = 0;
  }

  /// Set battery level (for testing low battery alerts)
  void setBatteryLevel(int level) {
    _batteryLevel = level.clamp(0, 100);
  }

  /// Trigger an anomaly manually (for testing)
  void triggerAnomaly({int? zone, int duration = 5}) {
    _currentAnomalyZone = zone ?? _random.nextInt(4);
    _anomalyDuration = duration;
  }

  /// Dispose resources
  void dispose() {
    stopStreaming();
  }
}
