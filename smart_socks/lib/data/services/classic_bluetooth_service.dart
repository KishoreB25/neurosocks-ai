// Classic Bluetooth (SPP) Service for ESP32 BluetoothSerial
// Uses native Android platform channel — NO third-party package needed.
// The ESP32 sends 16-byte binary packets every 2 seconds via BluetoothSerial.

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import '../models/sensor_reading.dart';

/// Lightweight model for a Classic Bluetooth device (name + MAC address)
class ClassicBtDevice {
  final String name;
  final String address;
  ClassicBtDevice({required this.name, required this.address});

  @override
  String toString() => '$name ($address)';
}

/// Service for Classic Bluetooth (SPP) communication with ESP32.
/// Communicates with native Kotlin via MethodChannel + EventChannel.
class ClassicBluetoothService {
  // Singleton
  static final ClassicBluetoothService _instance =
      ClassicBluetoothService._internal();
  factory ClassicBluetoothService() => _instance;
  ClassicBluetoothService._internal();

  // Platform channels — must match MainActivity.kt exactly
  static const _methodChannel = MethodChannel('com.neurosocks.app/classic_bt');
  static const _dataEventChannel =
      EventChannel('com.neurosocks.app/classic_bt_data');
  static const _discoveryEventChannel =
      EventChannel('com.neurosocks.app/classic_bt_discovery');

  // Parsed sensor stream exposed to the provider
  StreamController<SensorReading>? _sensorController;
  StreamSubscription? _rawDataSubscription;

  // State
  bool _isConnected = false;
  bool _isConnecting = false;
  String _deviceName = '';
  int _batteryLevel = 0;

  // Buffer for assembling 16-byte packets
  final List<int> _buffer = [];

  // ============== Getters ==============
  bool get isConnected => _isConnected;
  bool get isConnecting => _isConnecting;
  String get deviceName => _deviceName;
  int get batteryLevel => _batteryLevel;
  Stream<SensorReading>? get sensorStream => _sensorController?.stream;

  // ============== Bluetooth State ==============

  /// Check if Bluetooth adapter is enabled
  Future<bool> isEnabled() async {
    try {
      return await _methodChannel.invokeMethod<bool>('isEnabled') ?? false;
    } catch (e) {
      _log('isEnabled error: $e');
      return false;
    }
  }

  // ============== Discovery ==============

  /// Get list of paired/bonded devices
  Future<List<ClassicBtDevice>> getBondedDevices() async {
    try {
      final result = await _methodChannel.invokeMethod('getBondedDevices');
      final list = (result as List).cast<Map>();
      return list
          .map((m) => ClassicBtDevice(
                name: m['name']?.toString() ?? '',
                address: m['address']?.toString() ?? '',
              ))
          .toList();
    } catch (e) {
      _log('getBondedDevices error: $e');
      return [];
    }
  }

  /// Start scanning for nearby Classic Bluetooth devices.
  /// Returns a stream of discovered [ClassicBtDevice] objects.
  Stream<ClassicBtDevice> startDiscovery() {
    // Ask native to begin scanning
    _methodChannel.invokeMethod('startDiscovery').catchError((e) {
      _log('startDiscovery error: $e');
    });

    // Listen on the discovery EventChannel
    return _discoveryEventChannel.receiveBroadcastStream().map((event) {
      final m = Map<String, dynamic>.from(event as Map);
      return ClassicBtDevice(
        name: m['name']?.toString() ?? '',
        address: m['address']?.toString() ?? '',
      );
    });
  }

  /// Stop discovery
  Future<void> stopDiscovery() async {
    try {
      await _methodChannel.invokeMethod('stopDiscovery');
    } catch (e) {
      _log('stopDiscovery error: $e');
    }
  }

  // ============== Connection ==============

  /// Connect to an ESP32 Classic BT device by address
  Future<bool> connect(ClassicBtDevice device) async {
    if (_isConnecting || _isConnected) {
      _log('Already connected or connecting');
      return _isConnected;
    }

    _isConnecting = true;
    _deviceName = device.name.isNotEmpty ? device.name : 'ESP32';

    try {
      _log('Connecting to $_deviceName (${device.address})...');

      await _methodChannel.invokeMethod('connect', {'address': device.address});

      _isConnected = true;
      _isConnecting = false;
      _buffer.clear();
      _log('✅ Connected to $_deviceName');
      return true;
    } on PlatformException catch (e) {
      _log('❌ Connection failed: ${e.message}');
      _isConnected = false;
      _isConnecting = false;
      rethrow;
    } catch (e) {
      _log('❌ Connection failed: $e');
      _isConnected = false;
      _isConnecting = false;
      rethrow;
    }
  }

  /// Disconnect from the current device
  Future<void> disconnect() async {
    try {
      _log('Disconnecting...');
      await stopListening();
      await _methodChannel.invokeMethod('disconnect');
    } catch (e) {
      _log('⚠️ Disconnect error: $e');
    }

    _isConnected = false;
    _deviceName = '';
    _buffer.clear();
    _log('✅ Disconnected');
  }

  // ============== Data Listening ==============

  /// Start listening for 16-byte sensor packets from ESP32.
  /// Raw bytes arrive via EventChannel, get buffered and parsed here.
  Future<void> startListening() async {
    if (!_isConnected) {
      throw Exception('Not connected to any device');
    }

    // Create fresh sensor stream
    _sensorController?.close();
    _sensorController = StreamController<SensorReading>.broadcast();
    _buffer.clear();

    _log('📡 Listening for sensor data...');

    _rawDataSubscription =
        _dataEventChannel.receiveBroadcastStream().listen(
      (event) {
        // Native sends List<int> (bytes)
        final bytes = (event as List).cast<int>();
        _onDataReceived(bytes);
      },
      onError: (error) {
        _log('❌ Data stream error: $error');
        _isConnected = false;
        _sensorController?.addError(error);
      },
      onDone: () {
        _log('⚠️ Data stream ended (device disconnected?)');
        _isConnected = false;
      },
    );
  }

  /// Stop listening for data
  Future<void> stopListening() async {
    await _rawDataSubscription?.cancel();
    _rawDataSubscription = null;
    await _sensorController?.close();
    _sensorController = null;
    _buffer.clear();
  }

  // ============== Data Processing ==============

  /// Accumulate bytes and parse complete 16-byte packets
  void _onDataReceived(List<int> data) {
    _buffer.addAll(data);

    _log(
      '📥 +${data.length} bytes (buffer: ${_buffer.length}) '
      'raw: ${data.map((b) => b.toRadixString(16).padLeft(2, '0')).join(' ')}',
    );

    // Process all complete 16-byte packets
    while (_buffer.length >= 16) {
      final packet = _buffer.sublist(0, 16);
      _buffer.removeRange(0, 16);

      final reading = _parsePayload(packet);
      if (reading != null) {
        _log('🎉 Parsed reading → emitting');
        _sensorController?.add(reading);
      }
    }
  }

  // ============== 16-Byte Payload Parsing ==============
  //
  // ESP32 payload layout (matches the Arduino code exactly):
  //
  // Byte  0-3  : Temperature zones (Heel, Ball, Arch, Toe)
  //              Encoding: byte = (temp - 25.0) * 2.0 + 128
  //              Decoding: temp = 25.0 + (byte - 128) / 2.0
  //
  // Byte  4-7  : Pressure zones (Heel, Ball, Arch, Toe)
  //              Encoding: byte = pressure / 0.3
  //              Decoding: pressure = byte * 0.3
  //
  // Byte  8-9  : SpO2 (uint16 big-endian, value = realSpO2 * 100)
  //              Decoding: spO2 = ((byte8 << 8) | byte9) / 100.0
  //
  // Byte 10-11 : Heart Rate (uint16 big-endian, BPM)
  //
  // Byte 12-13 : Step Count (uint16 big-endian)
  //
  // Byte 14    : Activity Type (0=rest, 1=sit, 2=stand, 3=walk, 4=run)
  //
  // Byte 15    : Battery Level (0-100 %)

  SensorReading? _parsePayload(List<int> packet) {
    try {
      if (packet.length != 16) return null;

      // ---- Temperatures (Bytes 0-3) ----
      final temperatures = <double>[];
      for (int i = 0; i < 4; i++) {
        final raw = packet[i];
        final temp = 25.0 + (raw - 128) / 2.0;
        temperatures.add(temp);
      }

      // ---- Pressures (Bytes 4-7) ----
      final pressures = <double>[];
      for (int i = 0; i < 4; i++) {
        final raw = packet[4 + i];
        final pressure = raw * 0.3;
        pressures.add(pressure);
      }

      // ---- SpO2 (Bytes 8-9) ----
      final spO2Raw = (packet[8] << 8) | packet[9];
      double spO2 = spO2Raw / 100.0;
      if (spO2 > 100.0) spO2 = 100.0;
      if (spO2 < 0.0) spO2 = 0.0;

      // ---- Heart Rate (Bytes 10-11) ----
      int heartRate = (packet[10] << 8) | packet[11];
      if (heartRate > 250) heartRate = 250;
      if (heartRate < 0) heartRate = 0;

      // ---- Step Count (Bytes 12-13) ----
      final stepCount = (packet[12] << 8) | packet[13];

      // ---- Activity Type (Byte 14) ----
      final activityType = _parseActivityType(packet[14]);

      // ---- Battery Level (Byte 15) ----
      _batteryLevel = packet[15].clamp(0, 100);

      // Placeholder IMU (not transmitted by ESP32)
      final acc = AccelerometerData(
        x: stepCount > 0 ? 0.5 : 0.0,
        y: stepCount > 0 ? 0.3 : 0.0,
        z: 9.8,
      );
      final gyro = GyroscopeData(
        x: stepCount > 0 ? 5.0 : 0.5,
        y: stepCount > 0 ? 3.0 : 0.5,
        z: stepCount > 0 ? 2.0 : 0.5,
      );

      final reading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: temperatures,
        pressures: pressures,
        spO2: spO2,
        heartRate: heartRate,
        accelerometer: acc,
        gyroscope: gyro,
        stepCount: stepCount,
        batteryLevel: _batteryLevel,
        activityType: activityType,
      );

      _log(
        '✅ T=[${temperatures.map((t) => t.toStringAsFixed(1)).join(',')}]°C '
        'P=[${pressures.map((p) => p.toStringAsFixed(1)).join(',')}]kPa '
        'SpO2=${spO2.toStringAsFixed(1)}% HR=$heartRate STP=$stepCount '
        'ACT=$activityType BAT=$_batteryLevel%',
      );

      return reading;
    } catch (e) {
      _log('❌ Parse error: $e');
      return null;
    }
  }

  ActivityType _parseActivityType(int byte) {
    switch (byte & 0x0F) {
      case 0:
        return ActivityType.resting;
      case 1:
        return ActivityType.sitting;
      case 2:
        return ActivityType.standing;
      case 3:
        return ActivityType.walking;
      case 4:
        return ActivityType.running;
      default:
        return ActivityType.unknown;
    }
  }

  // ============== Logging ==============

  static void _log(String msg) {
    if (kDebugMode) {
      // ignore: avoid_print
      print('[ClassicBT] $msg');
    }
  }
}
