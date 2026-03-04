// Manages BLE connection, sensor streaming, foot data, trends
// PRODUCTION ONLY - Real Bluetooth or Firestore Historical Data

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../data/models/sensor_reading.dart';
import '../data/models/foot_data.dart';
import '../data/services/real_ble_service.dart';
import '../data/services/classic_bluetooth_service.dart';
import '../data/services/storage_service.dart';
import '../data/services/firebase/firebase_firestore_service.dart';
import 'risk_provider.dart';

/// Provider for managing sensor data and BLE connection
/// PRODUCTION: Real Bluetooth only OR Firestore historical data
class SensorProvider extends ChangeNotifier {
  final RealBleService _realBleService = RealBleService();
  final ClassicBluetoothService _classicBtService = ClassicBluetoothService();
  final StorageService _storageService = StorageService();
  final FirebaseFirestoreService _firestoreService = FirebaseFirestoreService();
  final RiskProvider _riskProvider = RiskProvider();  // ← NEW: Process readings for risk

  // Public getter for RealBleService (for device scanning)
  RealBleService get realBleService => _realBleService;
  // Public getter for ClassicBluetoothService
  ClassicBluetoothService get classicBtService => _classicBtService;
  
  // User context
  String? _currentUserId;

  // Current state
  SensorReading? _currentReading;
  FootData? _footData;  // Single foot (ESP32 sends data for one leg only)
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

  /// Returns the current reading ONLY when connected (otherwise null → zeros)
  SensorReading? get currentReading => _isConnected ? _currentReading : null;
  FootData? get footData => _isConnected ? _footData : null;
  /// Legacy getters for compatibility — both point to the single foot
  FootData? get leftFootData => footData;
  FootData? get rightFootData => null; // Only one leg connected
  bool get isConnected => _isConnected;
  bool get isStreaming => _isStreaming;
  bool get isConnecting => _isConnecting;
  bool get isLoadingFromFirestore => _isLoadingFromFirestore;
  String? get errorMessage => _errorMessage;
  String get deviceName {
    if (!_isConnected) return 'Not Connected';
    if (_classicBtService.isConnected) return _classicBtService.deviceName;
    return _realBleService.deviceName;
  }
  int get batteryLevel {
    if (!_isConnected) return 0;
    if (_classicBtService.isConnected) return _classicBtService.batteryLevel;
    return _realBleService.batteryLevel;
  }
  List<SensorReading> get recentReadings => List.unmodifiable(_recentReadings);
  String get dataSource => _dataSource;

  /// Raw reading (for internal/history use, bypasses connection check)
  SensorReading? get rawReading => _currentReading;

  // Convenience getters — return ZERO when device not connected
  List<double> get temperatures =>
      _isConnected ? (_currentReading?.temperatures ?? []) : [];
  List<double> get pressures =>
      _isConnected ? (_currentReading?.pressures ?? []) : [];
  double get spO2 =>
      _isConnected ? (_currentReading?.spO2 ?? 0) : 0;
  int get heartRate =>
      _isConnected ? (_currentReading?.heartRate ?? 0) : 0;
  int get stepCount =>
      _isConnected ? (_currentReading?.stepCount ?? 0) : 0;
  ActivityType get activityType =>
      _isConnected
          ? (_currentReading?.activityType ?? ActivityType.unknown)
          : ActivityType.unknown;

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
      debugPrint('🔌 SensorProvider: Connecting to ${device.platformName}...');
      final connected = await _realBleService.connectToDevice(device);
      
      if (connected) {
        _isConnected = true;
        _errorMessage = null;
        debugPrint('✅ SensorProvider: Connected to ${device.platformName}');
        notifyListeners();
        
        // Try to start streaming (don't fail connection if this fails)
        try {
          debugPrint('📡 SensorProvider: Attempting to start stream...');
          await startStreaming();
          debugPrint('✅ SensorProvider: Streaming started');
        } catch (e) {
          debugPrint('⚠️ SensorProvider: Streaming failed but connection OK: $e');
          // Connection is still valid
        }
      } else {
        _isConnected = false;
        _errorMessage = 'Connection returned false';
      }
      
    } catch (e) {
      _errorMessage = 'Failed to connect: $e';
      _isConnected = false;
      _dataSource = 'disconnected';
      debugPrint('❌ SensorProvider: Connection failed: $e');
    }

    _isConnecting = false;
    notifyListeners();
    return _isConnected;
  }

  // ============== Classic Bluetooth Connection ==============

  /// Connect to ESP32 via Classic Bluetooth (SPP)
  /// This is the correct protocol for ESP32 BluetoothSerial
  Future<bool> connectToClassicDevice(ClassicBtDevice device) async {
    if (_isConnected || _isConnecting) {
      debugPrint('⚠️ Already connected or connecting');
      return _isConnected;
    }

    _isConnecting = true;
    _errorMessage = null;
    notifyListeners();

    try {
      debugPrint('🔌 SensorProvider: Classic BT connecting to ${device.name}...');
      final connected = await _classicBtService.connect(device);

      if (connected) {
        _isConnected = true;
        _errorMessage = null;
        debugPrint('✅ SensorProvider: Classic BT connected to ${device.name}');
        notifyListeners();

        // Start listening for sensor data
        try {
          debugPrint('📡 SensorProvider: Starting Classic BT stream...');
          await _classicBtService.startListening();

          _subscription?.cancel();
          _subscription = _classicBtService.sensorStream?.listen(
            _onReadingReceived,
            onError: _onStreamError,
            onDone: _onStreamDone,
          );

          _isStreaming = true;
          _dataSource = 'bluetooth';
          debugPrint('✅ SensorProvider: Classic BT streaming started');
        } catch (e) {
          debugPrint('⚠️ SensorProvider: Classic BT streaming failed: $e');
        }
      } else {
        _isConnected = false;
        _errorMessage = 'Classic BT connection returned false';
      }
    } catch (e) {
      _errorMessage = 'Failed to connect via Classic BT: $e';
      _isConnected = false;
      _dataSource = 'disconnected';
      debugPrint('❌ SensorProvider: Classic BT connection failed: $e');
    }

    _isConnecting = false;
    notifyListeners();
    return _isConnected;
  }

  /// Disconnect from the device — clears all live data (shows zeros)
  Future<void> disconnect() async {
    try {
      await stopStreaming();
      // Disconnect both services (only the connected one will do anything)
      await _realBleService.disconnect();
      await _classicBtService.disconnect();
    } catch (e) {
      _errorMessage = 'Disconnect error: $e';
      debugPrint('❌ Disconnect error: $e');
    }

    // Always clear state so dashboard shows zeros
    _isConnected = false;
    _isStreaming = false;
    _currentReading = null;
    _footData = null;
    _dataSource = 'disconnected';
    notifyListeners();
    debugPrint('✅ Disconnected — all readings cleared to zero');
  }

  // ============== Streaming Management ==============

  /// Start receiving sensor data from Bluetooth
  /// Returns false if not connected to a device
  Future<bool> startStreaming() async {
    if (_isStreaming) {
      debugPrint('⚠️ Already streaming');
      return true;
    }

    // If not connected via Bluetooth, do nothing — dashboard shows zeros
    if (!_isConnected) {
      debugPrint('⚠️ Not connected via Bluetooth. Dashboard will show zeros.');
      return false;
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

    // ✅ NEW: Process reading for RISK CALCULATION (this was missing!)
    _riskProvider.processReading(reading);
    debugPrint('📈 Risk calculated - Overall: ${_riskProvider.currentScore}');

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
      // Predictions are now saved by RiskProvider (ML-based) via processReading()
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


  /// Load recent data from Firestore (for history/trends only, NOT as current reading)
  /// Current reading is ONLY populated by live Bluetooth data
  Future<void> _loadRecentDataFromFirestore() async {
    if (_currentUserId == null) {
      debugPrint('⚠️ Cannot load from Firestore: userId is null');
      return;
    }

    _isLoadingFromFirestore = true;
    notifyListeners();

    try {
      debugPrint('📥 Loading historical readings from Firestore...');
      
      final readings = await _firestoreService.getRecentReadings(
        userId: _currentUserId!,
        limit: 50,
      );

      if (readings.isEmpty) {
        debugPrint('⚠️ No readings found in Firestore');
      } else {
        // Load into history only — do NOT set as _currentReading
        // _currentReading must come from live Bluetooth connection
        _recentReadings.clear();
        _recentReadings.addAll(readings);

        debugPrint('✅ Loaded ${readings.length} historical readings from Firestore');
      }
    } catch (e) {
      debugPrint('❌ Firestore load error: $e');
    }

    _isLoadingFromFirestore = false;
    notifyListeners();
  }

  /// Update foot data from sensor reading (single foot — one ESP32)
  void _updateFootData(SensorReading reading) {
    // ESP32 sends 4 temperature + 4 pressure zones for ONE foot
    // Zones: 0=Heel, 1=Ball, 2=Arch, 3=Toe
    
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
      _footData = FootData(
        side: FootSide.left, // Single leg setup
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

  /// Get average temperature across all zones (0 when disconnected)
  double get averageTemperature {
    if (!_isConnected || _currentReading == null || _currentReading!.temperatures.isEmpty) {
      return 0;
    }
    return _currentReading!.averageTemperature;
  }

  /// Get max temperature (0 when disconnected)
  double get maxTemperature {
    if (!_isConnected || _currentReading == null || _currentReading!.temperatures.isEmpty) {
      return 0;
    }
    return _currentReading!.maxTemperature;
  }

  /// Get average pressure (0 when disconnected)
  double get averagePressure {
    if (!_isConnected || _currentReading == null || _currentReading!.pressures.isEmpty) {
      return 0;
    }
    return _currentReading!.averagePressure;
  }

  /// Get max pressure (0 when disconnected)
  double get maxPressure {
    if (!_isConnected || _currentReading == null || _currentReading!.pressures.isEmpty) {
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

  /// Clear error message
  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _classicBtService.disconnect();
    // RealBleService manages its own lifecycle
    super.dispose();
  }
}
