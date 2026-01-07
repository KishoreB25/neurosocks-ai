// Manages BLE connection, sensor streaming, foot data, trends

import 'dart:async';
import 'package:flutter/foundation.dart';
import '../data/models/sensor_reading.dart';
import '../data/models/foot_data.dart';
import '../data/services/mock_ble_service.dart';
import '../data/services/storage_service.dart';

/// Provider for managing sensor data and BLE connection
class SensorProvider extends ChangeNotifier {
  final MockBleService _bleService = MockBleService();
  final StorageService _storageService = StorageService();

  // Current state
  SensorReading? _currentReading;
  FootData? _leftFootData;
  FootData? _rightFootData;
  bool _isConnected = false;
  bool _isStreaming = false;
  bool _isConnecting = false;
  String? _errorMessage;

  // Stream subscription
  StreamSubscription<SensorReading>? _subscription;

  // Reading history (in-memory for quick access)
  final List<SensorReading> _recentReadings = [];
  static const int _maxRecentReadings = 100;

  // ============== Getters ==============

  SensorReading? get currentReading => _currentReading;
  FootData? get leftFootData => _leftFootData;
  FootData? get rightFootData => _rightFootData;
  bool get isConnected => _isConnected;
  bool get isStreaming => _isStreaming;
  bool get isConnecting => _isConnecting;
  String? get errorMessage => _errorMessage;
  String get deviceName => _bleService.deviceName;
  int get batteryLevel => _bleService.batteryLevel;
  List<SensorReading> get recentReadings => List.unmodifiable(_recentReadings);

  // Convenience getters for current reading
  List<double> get temperatures => _currentReading?.temperatures ?? [];
  List<double> get pressures => _currentReading?.pressures ?? [];
  double get spO2 => _currentReading?.spO2 ?? 0;
  int get heartRate => _currentReading?.heartRate ?? 0;
  int get stepCount => _currentReading?.stepCount ?? 0;
  ActivityType get activityType =>
      _currentReading?.activityType ?? ActivityType.unknown;

  // ============== Connection Management ==============

  /// Connect to the smart sock device
  Future<bool> connect() async {
    if (_isConnected || _isConnecting) return _isConnected;

    _isConnecting = true;
    _errorMessage = null;
    notifyListeners();

    try {
      final success = await _bleService.connect();

      if (success) {
        _isConnected = true;
        _errorMessage = null;
      } else {
        _errorMessage = 'Failed to connect to device';
      }
    } catch (e) {
      _errorMessage = 'Connection error: $e';
      _isConnected = false;
    }

    _isConnecting = false;
    notifyListeners();

    return _isConnected;
  }

  /// Disconnect from the device
  Future<void> disconnect() async {
    await stopStreaming();
    await _bleService.disconnect();
    _isConnected = false;
    _currentReading = null;
    _leftFootData = null;
    _rightFootData = null;
    notifyListeners();
  }

  // ============== Streaming Management ==============

  /// Start receiving sensor data
  Future<void> startStreaming({
    Duration interval = const Duration(seconds: 2),
    bool simulateAnomalies = true,
  }) async {
    if (_isStreaming) return;

    try {
      await _bleService.startStreaming(
        interval: interval,
        simulateAnomalies: simulateAnomalies,
      );

      _subscription = _bleService.sensorStream?.listen(
        _onReadingReceived,
        onError: _onStreamError,
        onDone: _onStreamDone,
      );

      _isStreaming = true;
      _isConnected = true;
      _errorMessage = null;
      notifyListeners();
    } catch (e) {
      _errorMessage = 'Failed to start streaming: $e';
      notifyListeners();
    }
  }

  /// Stop receiving sensor data
  Future<void> stopStreaming() async {
    await _subscription?.cancel();
    _subscription = null;
    await _bleService.stopStreaming();
    _isStreaming = false;
    notifyListeners();
  }

  /// Handle incoming sensor reading
  void _onReadingReceived(SensorReading reading) {
    _currentReading = reading;

    // Update foot data models
    _updateFootData(reading);

    // Add to recent readings
    _recentReadings.insert(0, reading);
    if (_recentReadings.length > _maxRecentReadings) {
      _recentReadings.removeLast();
    }

    // Save to local storage (async, don't wait)
    _storageService.saveReading(reading);

    notifyListeners();
  }

  /// Update foot data from sensor reading
  void _updateFootData(SensorReading reading) {
    // For now, we'll use the same reading for both feet
    // In real implementation, you'd have separate sensors per foot
    
    // Create zones from sensor data
    // Index 0: Heel, 1: Ball, 2: Arch, 3: Toe
    final zones = <FootZone>[];
    
    for (int i = 0; i < 4 && i < reading.temperatures.length; i++) {
      final temp = reading.temperatures[i];
      final pressure = i < reading.pressures.length ? reading.pressures[i] : 0.0;
      
      zones.add(FootZone.fromReadings(
        index: i,
        temperature: temp,
        pressure: pressure,
      ));
    }

    if (zones.length >= 4) {
      _leftFootData = FootData(
        side: FootSide.left,
        heel: zones[0],
        ball: zones[1],
        arch: zones[2],
        toe: zones[3],
        timestamp: reading.timestamp,
      );

      // For demo, right foot has slight variation
      _rightFootData = FootData(
        side: FootSide.right,
        heel: zones[0].copyWith(
          temperature: zones[0].temperature + 0.2,
          pressure: zones[0].pressure * 0.95,
        ),
        ball: zones[1].copyWith(
          temperature: zones[1].temperature - 0.1,
          pressure: zones[1].pressure * 1.05,
        ),
        arch: zones[2].copyWith(
          temperature: zones[2].temperature + 0.1,
        ),
        toe: zones[3].copyWith(
          temperature: zones[3].temperature - 0.2,
        ),
        timestamp: reading.timestamp,
      );
    }
  }

  /// Handle stream error
  void _onStreamError(dynamic error) {
    _errorMessage = 'Stream error: $error';
    _isStreaming = false;
    notifyListeners();
  }

  /// Handle stream completion
  void _onStreamDone() {
    _isStreaming = false;
    notifyListeners();
  }

  // ============== Data Access ==============

  /// Get average temperature across all zones
  double get averageTemperature {
    if (_currentReading == null || _currentReading!.temperatures.isEmpty) {
      return 0;
    }
    return _currentReading!.averageTemperature;
  }

  /// Get max temperature
  double get maxTemperature {
    if (_currentReading == null || _currentReading!.temperatures.isEmpty) {
      return 0;
    }
    return _currentReading!.maxTemperature;
  }

  /// Get average pressure
  double get averagePressure {
    if (_currentReading == null || _currentReading!.pressures.isEmpty) {
      return 0;
    }
    return _currentReading!.averagePressure;
  }

  /// Get max pressure
  double get maxPressure {
    if (_currentReading == null || _currentReading!.pressures.isEmpty) {
      return 0;
    }
    return _currentReading!.maxPressure;
  }

  /// Get temperature trend (last N readings)
  List<double> getTemperatureTrend({int count = 20}) {
    return _recentReadings
        .take(count)
        .map((r) => r.averageTemperature)
        .toList()
        .reversed
        .toList();
  }

  /// Get pressure trend
  List<double> getPressureTrend({int count = 20}) {
    return _recentReadings
        .take(count)
        .map((r) => r.averagePressure)
        .toList()
        .reversed
        .toList();
  }

  /// Get SpO2 trend
  List<double> getSpO2Trend({int count = 20}) {
    return _recentReadings
        .take(count)
        .map((r) => r.spO2)
        .toList()
        .reversed
        .toList();
  }

  /// Get heart rate trend
  List<int> getHeartRateTrend({int count = 20}) {
    return _recentReadings
        .take(count)
        .map((r) => r.heartRate)
        .toList()
        .reversed
        .toList();
  }

  // ============== Mock Service Controls ==============

  /// Set anomaly simulation
  void setSimulateAnomalies(bool enable) {
    _bleService.setSimulateAnomalies(enable);
  }

  /// Trigger a test anomaly
  void triggerTestAnomaly({int? zone, int duration = 5}) {
    _bleService.triggerAnomaly(zone: zone, duration: duration);
  }

  /// Set test activity
  void setTestActivity(ActivityType activity) {
    _bleService.setActivity(activity);
  }

  /// Set test battery level
  void setTestBatteryLevel(int level) {
    _bleService.setBatteryLevel(level);
  }

  /// Reset step count
  void resetStepCount() {
    _bleService.resetStepCount();
  }

  // ============== Cleanup ==============

  /// Clear error message
  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  /// Clear recent readings cache
  void clearRecentReadings() {
    _recentReadings.clear();
    notifyListeners();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _bleService.dispose();
    super.dispose();
  }
}
