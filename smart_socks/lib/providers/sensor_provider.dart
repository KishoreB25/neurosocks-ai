// Manages BLE connection, sensor streaming, foot data, trends
// PRODUCTION ONLY - Real Bluetooth or Firestore Historical Data

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../data/models/sensor_reading.dart';
import '../data/models/foot_data.dart';
import '../data/models/risk_score.dart';
import '../data/services/real_ble_service.dart';
import '../data/services/storage_service.dart';
import '../data/services/foot_ulcer_prediction_service.dart';
import '../data/services/firebase/firebase_firestore_service.dart';

/// Provider for managing sensor data and BLE connection
/// PRODUCTION: Real Bluetooth only OR Firestore historical data
class SensorProvider extends ChangeNotifier {
  final RealBleService _realBleService = RealBleService();
  final StorageService _storageService = StorageService();
  final FirebaseFirestoreService _firestoreService = FirebaseFirestoreService();

  // Public getter for RealBleService (for device scanning)
  RealBleService get realBleService => _realBleService;
  
  // User context
  String? _currentUserId;

  // Current state
  SensorReading? _currentReading;
  FootData? _leftFootData;
  FootData? _rightFootData;
  bool _isConnected = false;
  bool _isStreaming = false;
  bool _isConnecting = false;
  bool _isLoadingFromFirestore = false;
  String? _errorMessage;
  String _dataSource = 'disconnected'; // 'bluetooth' | 'firestore' | 'disconnected'

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
  bool get isLoadingFromFirestore => _isLoadingFromFirestore;
  String? get errorMessage => _errorMessage;
  String get deviceName => _realBleService.deviceName;
  int get batteryLevel => _realBleService.batteryLevel;
  List<SensorReading> get recentReadings => List.unmodifiable(_recentReadings);
  String get dataSource => _dataSource;

  // Convenience getters for current reading
  List<double> get temperatures => _currentReading?.temperatures ?? [];
  List<double> get pressures => _currentReading?.pressures ?? [];
  double get spO2 => _currentReading?.spO2 ?? 0;
  int get heartRate => _currentReading?.heartRate ?? 0;
  int get stepCount => _currentReading?.stepCount ?? 0;
  ActivityType get activityType =>
      _currentReading?.activityType ?? ActivityType.unknown;

  /// Check if connection is in progress
  Future<bool> get isConnectingAsync async => _isConnecting;

  // ============== Initialization ==============

  /// Set current user ID for user-specific data operations
  void setCurrentUser(String userId) {
    _currentUserId = userId;
    // Try to load recent data from Firestore
    _loadRecentDataFromFirestore();
    notifyListeners();
  }

  /// Get current user ID
  String? get currentUserId => _currentUserId;

  // ============== Device Scanning (Real BLE Only) ==============

  /// Scan for available devices
  Future<List<ScanResult>> scanForDevices() async {
    try {
      return await _realBleService.scanForDevices();
    } catch (e) {
      _errorMessage = 'Scan error: $e';
      notifyListeners();
      rethrow;
    }
  }

  // ============== Connection Management ==============

  /// Connect to the smart sock device
  Future<bool> connectToDevice(BluetoothDevice device) async {
    if (_isConnected || _isConnecting) {
      debugPrint('⚠️ Already connected or connecting');
      return _isConnected;
    }

    _isConnecting = true;
    _errorMessage = null;
    notifyListeners();

    try {
      debugPrint('🔌 Connecting to device: ${device.platformName}...');
      await _realBleService.connectToDevice(device);
      
      _isConnected = true;
      _dataSource = 'disconnected';
      _errorMessage = null;
      
      debugPrint('✅ Connected to ${device.platformName}');
    } catch (e) {
      _errorMessage = 'Failed to connect: $e';
      _isConnected = false;
      _dataSource = 'disconnected';
      debugPrint('❌ Connection failed: $e');
    }

    _isConnecting = false;
    notifyListeners();
    return _isConnected;
  }

  /// Disconnect from the device
  Future<void> disconnect() async {
    try {
      await stopStreaming();
      await _realBleService.disconnect();
      
      _isConnected = false;
      _dataSource = 'disconnected';
      notifyListeners();
      
      debugPrint('✅ Disconnected from device');
    } catch (e) {
      _errorMessage = 'Disconnect error: $e';
      debugPrint('❌ Disconnect error: $e');
      notifyListeners();
    }
  }

  // ============== Streaming Management ==============

  /// Start receiving sensor data from Bluetooth
  /// Returns false if not connected to a device
  Future<bool> startStreaming() async {
    if (_isStreaming) {
      debugPrint('⚠️ Already streaming');
      return true;
    }

    // If not connected via Bluetooth, load from Firestore
    if (!_isConnected) {
      debugPrint('⚠️ Not connected via Bluetooth. Loading historical data from Firestore...');
      await _loadRecentDataFromFirestore();
      return false; // Return false to indicate we're not streaming live data
    }

    try {
      debugPrint('📡 Starting Bluetooth stream...');
      await _realBleService.startStreaming();
      
      _subscription = _realBleService.sensorStream?.listen(
        _onReadingReceived,
        onError: _onStreamError,
        onDone: _onStreamDone,
      );

      _isStreaming = true;
      _dataSource = 'bluetooth';
      _errorMessage = null;
      
      debugPrint('✅ Bluetooth stream started');
      notifyListeners();
      return true;
    } catch (e) {
      _errorMessage = 'Failed to start stream: $e';
      _isStreaming = false;
      _dataSource = 'disconnected';
      debugPrint('❌ Stream failed: $e');
      notifyListeners();
      return false;
    }
  }

  /// Stop receiving sensor data
  Future<void> stopStreaming() async {
    try {
      await _subscription?.cancel();
      _subscription = null;
      
      await _realBleService.stopStreaming();
      
      _isStreaming = false;
      notifyListeners();
      
      debugPrint('✅ Stream stopped');
    } catch (e) {
      debugPrint('❌ Stop stream error: $e');
    }
  }

  /// Handle incoming sensor reading from Bluetooth
  void _onReadingReceived(SensorReading reading) {
    _currentReading = reading;
    debugPrint('📊 Received reading - Temp: ${reading.temperatures}, Pressure: ${reading.pressures}');

    // Update foot data models
    _updateFootData(reading);

    // Add to recent readings
    _recentReadings.insert(0, reading);
    if (_recentReadings.length > _maxRecentReadings) {
      _recentReadings.removeLast();
    }

    // Save to local storage (async, don't wait)
    unawaited(_storageService.saveReading(reading));

    // Save to Firestore if user is logged in
    if (_currentUserId != null) {
      unawaited(_saveReadingToFirestore(reading));
      unawaited(_savePredictionToFirestore(reading));
    } else {
      debugPrint('⚠️ Cannot save to Firestore: userId is null');
    }

    notifyListeners();
  }

  /// Save sensor reading to Firestore (async, non-blocking)
  Future<void> _saveReadingToFirestore(SensorReading reading) async {
    if (_currentUserId == null) {
      debugPrint('❌ Cannot save sensor reading: userId is null');
      return;
    }
    
    try {
      await _firestoreService.saveSensorReading(
        userId: _currentUserId!,
        reading: reading,
      );
      debugPrint('💾 Sensor reading saved to Firestore');
    } catch (e) {
      debugPrint('❌ Firestore save error: $e');
    }
  }

  /// Save foot ulcer prediction to Firestore
  Future<void> _savePredictionToFirestore(SensorReading reading) async {
    if (_currentUserId == null) {
      debugPrint('❌ Cannot save prediction: userId is null');
      return;
    }

    try {
      // Generate prediction
      final prediction = FootUlcerPredictionService.predictRisk(
        reading,
        historicalReadings: _recentReadings,
      );

      // Convert to risk score for storage
      final riskScore = RiskScore(
        timestamp: prediction.timestamp,
        overallScore: prediction.riskScore.toInt(),
        riskLevel: _mapUlcerRiskToRiskLevel(prediction.level),
        pressureRisk: reading.maxPressure.toInt(),
        temperatureRisk: reading.maxTemperature.toInt(),
        circulationRisk: 0,
        gaitRisk: 0,
        factors: prediction.riskFactors,
        recommendations: [prediction.recommendation],
      );

      await _firestoreService.saveRiskScore(
        userId: _currentUserId!,
        riskScore: riskScore,
      );
      debugPrint('💾 Risk prediction saved to Firestore');
    } catch (e) {
      debugPrint('❌ Prediction save error: $e');
    }
  }

  /// Load recent data from Firestore (when not connected via Bluetooth)
  Future<void> _loadRecentDataFromFirestore() async {
    if (_currentUserId == null) {
      debugPrint('⚠️ Cannot load from Firestore: userId is null');
      _errorMessage = 'Not logged in';
      notifyListeners();
      return;
    }

    _isLoadingFromFirestore = true;
    notifyListeners();

    try {
      debugPrint('📥 Loading recent readings from Firestore...');
      
      final readings = await _firestoreService.getRecentReadings(
        userId: _currentUserId!,
        limit: 50,
      );

      if (readings.isEmpty) {
        _errorMessage = 'No previous data available';
        _dataSource = 'disconnected';
        debugPrint('⚠️ No readings found in Firestore');
      } else {
        // Load most recent reading
        final mostRecent = readings.first;
        _currentReading = mostRecent;
        _dataSource = 'firestore';
        _errorMessage = null;

        // Load all readings into history
        _recentReadings.clear();
        _recentReadings.addAll(readings);

        // Update foot data
        _updateFootData(mostRecent);

        debugPrint('✅ Loaded ${readings.length} readings from Firestore');
        debugPrint('📊 Latest: Temp=${mostRecent.temperatures}, Pressure=${mostRecent.pressures}');
      }
    } catch (e) {
      _errorMessage = 'Failed to load data: $e';
      _dataSource = 'disconnected';
      debugPrint('❌ Firestore load error: $e');
    }

    _isLoadingFromFirestore = false;
    notifyListeners();
  }

  /// Update foot data from sensor reading
  void _updateFootData(SensorReading reading) {
    // Create foot zone data from sensor readings
    // Sensor zones 1-5: Left foot (Heel, Ball, Arch, Toe)
    // Sensor zones 6-10: Right foot (Heel, Ball, Arch, Toe)
    
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

      _rightFootData = FootData(
        side: FootSide.right,
        heel: zones[0],
        ball: zones[1],
        arch: zones[2],
        toe: zones[3],
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

  // ============== Cleanup ==============

  /// Map UlcerRiskLevel to RiskLevel enum
  RiskLevel _mapUlcerRiskToRiskLevel(dynamic level) {
    final levelStr = level.toString().toLowerCase();
    if (levelStr.contains('low')) {
      return RiskLevel.low;
    } else if (levelStr.contains('moderate')) {
      return RiskLevel.moderate;
    } else if (levelStr.contains('high')) {
      return RiskLevel.high;
    } else {
      return RiskLevel.critical;
    }
  }

  /// Clear error message
  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    // RealBleService manages its own lifecycle
    super.dispose();
  }
}
