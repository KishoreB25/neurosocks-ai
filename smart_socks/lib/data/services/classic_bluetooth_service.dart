// Classic Bluetooth (SPP) Service for ESP32 BluetoothSerial
// The ESP32 sends 16-byte binary packets every 2 seconds via BluetoothSerial.
// BLE (flutter_blue_plus) CANNOT read this data — only Classic BT can.

import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import '../models/sensor_reading.dart';

/// Service for Classic Bluetooth (SPP) communication with ESP32.
/// The ESP32 uses BluetoothSerial.write(payload, 16) — this is SPP, not BLE.
class ClassicBluetoothService {
  // Singleton
  static final ClassicBluetoothService _instance =
      ClassicBluetoothService._internal();
  factory ClassicBluetoothService() => _instance;
  ClassicBluetoothService._internal();

  // Connection
  BluetoothConnection? _connection;
  BluetoothDevice? _device;

  // Stream
  StreamController<SensorReading>? _streamController;
  StreamSubscription<Uint8List>? _inputSubscription;

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
  Stream<SensorReading>? get sensorStream => _streamController?.stream;

  // ============== Connection ==============

  /// Connect to an ESP32 Classic Bluetooth device
  Future<bool> connect(BluetoothDevice device) async {
    if (_isConnecting || _isConnected) {
      debugPrint('[ClassicBT] Already connected or connecting');
      return _isConnected;
    }

    _isConnecting = true;
    _device = device;
    _deviceName = device.name ?? 'ESP32';

    try {
      debugPrint('[ClassicBT] Connecting to $_deviceName (${device.address})...');

      _connection = await BluetoothConnection.toAddress(device.address);

      _isConnected = true;
      _isConnecting = false;
      _buffer.clear();

      debugPrint('[ClassicBT] ✅ Connected to $_deviceName');
      return true;
    } catch (e) {
      debugPrint('[ClassicBT] ❌ Connection failed: $e');
      _isConnected = false;
      _isConnecting = false;
      _connection = null;
      rethrow;
    }
  }

  /// Disconnect
  Future<void> disconnect() async {
    try {
      debugPrint('[ClassicBT] Disconnecting...');
      await stopListening();
      await _connection?.close();
      _connection?.dispose();
    } catch (e) {
      debugPrint('[ClassicBT] ⚠️ Disconnect error: $e');
    }

    _connection = null;
    _isConnected = false;
    _device = null;
    _buffer.clear();
    debugPrint('[ClassicBT] ✅ Disconnected');
  }

  // ============== Data Listening ==============

  /// Start listening for 16-byte sensor packets from ESP32
  Future<void> startListening() async {
    if (_connection == null || !_isConnected) {
      throw Exception('Not connected to any device');
    }

    // Create fresh stream controller
    _streamController?.close();
    _streamController = StreamController<SensorReading>.broadcast();
    _buffer.clear();

    debugPrint('[ClassicBT] 📡 Listening for sensor data...');

    _inputSubscription = _connection!.input?.listen(
      (Uint8List data) {
        _onDataReceived(data);
      },
      onError: (error) {
        debugPrint('[ClassicBT] ❌ Stream error: $error');
        _streamController?.addError(error);
      },
      onDone: () {
        debugPrint('[ClassicBT] ⚠️ Stream ended (device disconnected?)');
        _isConnected = false;
      },
      cancelOnError: false,
    );
  }

  /// Stop listening
  Future<void> stopListening() async {
    await _inputSubscription?.cancel();
    _inputSubscription = null;
    await _streamController?.close();
    _streamController = null;
    _buffer.clear();
  }

  // ============== Data Processing ==============

  /// Accumulate bytes and parse complete 16-byte packets
  void _onDataReceived(Uint8List data) {
    _buffer.addAll(data);

    debugPrint(
      '[ClassicBT] 📥 +${data.length} bytes (buffer: ${_buffer.length}) '
      'raw: ${data.map((b) => b.toRadixString(16).padLeft(2, '0')).join(' ')}',
    );

    // Process all complete 16-byte packets in the buffer
    while (_buffer.length >= 16) {
      final packet = _buffer.sublist(0, 16);
      _buffer.removeRange(0, 16);

      final reading = _parsePayload(packet);
      if (reading != null) {
        debugPrint('[ClassicBT] 🎉 Parsed reading → emitting to stream');
        _streamController?.add(reading);
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
      // Clamp to physiological range
      if (spO2 > 100.0) spO2 = 100.0;
      if (spO2 < 0.0) spO2 = 0.0;

      // ---- Heart Rate (Bytes 10-11) ----
      int heartRate = (packet[10] << 8) | packet[11];
      // Clamp to physiological range
      if (heartRate > 250) heartRate = 250;
      if (heartRate < 0) heartRate = 0;

      // ---- Step Count (Bytes 12-13) ----
      final stepCount = (packet[12] << 8) | packet[13];

      // ---- Activity Type (Byte 14) ----
      final activityType = _parseActivityType(packet[14]);

      // ---- Battery Level (Byte 15) ----
      _batteryLevel = packet[15].clamp(0, 100);

      // Dummy IMU (not transmitted by ESP32)
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

      debugPrint(
        '[ClassicBT] ✅ T=[${temperatures.map((t) => t.toStringAsFixed(1)).join(',')}]°C '
        'P=[${pressures.map((p) => p.toStringAsFixed(1)).join(',')}]kPa '
        'SpO2=${spO2.toStringAsFixed(1)}% HR=$heartRate STP=$stepCount '
        'ACT=$activityType BAT=$_batteryLevel%',
      );

      return reading;
    } catch (e) {
      debugPrint('[ClassicBT] ❌ Parse error: $e');
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

  // ============== Debug ==============

  static void debugPrint(String msg) {
    if (kDebugMode) {
      // ignore: avoid_print
      print(msg);
    }
  }
}
