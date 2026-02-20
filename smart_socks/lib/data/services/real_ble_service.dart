// Real BLE Service using flutter_blue_plus
// Connects to actual smart socks hardware via Bluetooth Low Energy (SPP - Serial Port Profile)
// ESP32 sends 16-byte packets every 2 seconds

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import '../models/sensor_reading.dart';
import '../../core/constants/sensor_constants.dart';

/// Service for real BLE communication with smart sock device (ESP32 BluetoothSerial)
class RealBleService {
  // Singleton pattern
  static final RealBleService _instance = RealBleService._internal();
  factory RealBleService() => _instance;
  RealBleService._internal();

  // Device connection
  BluetoothDevice? _device;
  BluetoothCharacteristic? _rxCharacteristic;

  // Stream controllers
  StreamController<SensorReading>? _streamController;
  StreamSubscription? _rxSubscription;

  // Connection state
  bool _isConnected = false;
  bool _isStreaming = false;
  bool _isConnecting = false;
  String _deviceName = '';
  int _batteryLevel = 0;

  // Buffer for incomplete packets
  final List<int> _dataBuffer = [];

  // ============== Getters ==============
  bool get isConnected => _isConnected;
  bool get isStreaming => _isStreaming;
  bool get isConnecting => _isConnecting;
  String get deviceName => _deviceName;
  int get batteryLevel => _batteryLevel;
  Stream<SensorReading>? get sensorStream => _streamController?.stream;

  // ============== Scanning & Connection ==============

  /// Scan for nearby NeuroSock devices
  Future<List<ScanResult>> scanForDevices({int timeoutSeconds = 15}) async {
    try {
      if (kIsWeb) {
        throw Exception('Bluetooth not supported on web platform');
      }

      // Check Bluetooth adapter state
      final adapterState = await FlutterBluePlus.adapterState.first;
      if (adapterState != BluetoothAdapterState.on) {
        throw Exception('Bluetooth is disabled. Please enable Bluetooth.');
      }

      debugPrint('🔍 Starting BLE scan for NeuroSock devices...');

      // Stop any previous scan
      await FlutterBluePlus.stopScan();

      final discoveredDevices = <ScanResult>[];

      await FlutterBluePlus.startScan(
        timeout: Duration(seconds: timeoutSeconds),
        androidScanMode: AndroidScanMode.lowLatency,
      );

      // Collect scan results while scanning
      final subscription = FlutterBluePlus.scanResults.listen((results) {
        for (final result in results) {
          // Looking for device name starting with "NeuroSock"
          if (result.device.platformName
              .startsWith(SensorConstants.bleDeviceNamePrefix)) {
            // Avoid duplicates
            if (!discoveredDevices
                .any((r) => r.device.remoteId == result.device.remoteId)) {
              debugPrint(
                  '✅ Found device: ${result.device.platformName} (${result.device.remoteId})');
              discoveredDevices.add(result);
            }
          }
        }
      });

      // Wait for scan to complete
      await Future.delayed(Duration(seconds: timeoutSeconds));

      // Cleanup
      await FlutterBluePlus.stopScan();
      await subscription.cancel();

      if (discoveredDevices.isEmpty) {
        debugPrint(
            '⚠️ No NeuroSock devices found. Make sure your ESP32 is powered on and advertising.');
      }

      return discoveredDevices;
    } catch (e) {
      debugPrint('❌ Scan error: $e');
      await FlutterBluePlus.stopScan();
      rethrow;
    }
  }

  /// Connect to a specific BLE device
  Future<bool> connectToDevice(BluetoothDevice device) async {
    try {
      if (kIsWeb) {
        throw Exception('Bluetooth not supported on web platform');
      }

      if (_isConnecting) {
        throw Exception('Connection already in progress');
      }

      _isConnecting = true;
      _device = device;
      _deviceName = device.platformName;

      debugPrint('🔗 Connecting to $_deviceName...');

      // Connect with timeout
      await device.connect(
        timeout: const Duration(seconds: 15),
        autoConnect: false,
      );

      debugPrint('✅ Connected to device');
      _isConnected = true;
      _isConnecting = false;

      // Discover services to find RX/TX characteristics
      await _discoverServices();

      return true;
    } catch (e) {
      debugPrint('❌ Connection error: $e');
      _isConnected = false;
      _isConnecting = false;
      _device = null;
      rethrow;
    }
  }

  /// Discover RX/TX characteristics for SPP communication
  Future<void> _discoverServices() async {
    try {
      debugPrint('🔎 Discovering services...');

      final services = await _device!.discoverServices();
      debugPrint('Found ${services.length} services');

      for (var service in services) {
        debugPrint('Service: ${service.uuid}');

        for (var char in service.characteristics) {
          debugPrint('  Characteristic: ${char.uuid}');
          debugPrint('    Properties: ${char.properties}');

          // ESP32 BluetoothSerial typically uses these standard UUIDs:
          // RX (notify): 6E400003-B5A3-F393-E0A9-E50E24DCCA9E
          // RX (notify): 6E400003-B5A3-F393-E0A9-E50E24DCCA9E

          const String uartServiceUuid =
              '6E400001-B5A3-F393-E0A9-E50E24DCCA9E';
          const String rxCharUuid =
              '6E400003-B5A3-F393-E0A9-E50E24DCCA9E';

          // Check if this is the UART service
          if (service.uuid.toString().toUpperCase() ==
                  uartServiceUuid.toUpperCase() ||
              service.uuid.toString().toUpperCase().contains('180A') ||
              service.uuid.toString().toUpperCase().contains('180D')) {
            // Look for RX characteristic (notify)
            if (char.uuid.toString().toUpperCase() ==
                    rxCharUuid.toUpperCase() ||
                char.uuid.toString().toUpperCase().contains('2A37')) {
              if (char.properties.notify) {
                _rxCharacteristic = char;
                debugPrint('✅ Found RX characteristic (notify)');
              }
            }
          }
        }
      }

      if (_rxCharacteristic == null) {
        throw Exception(
            'EX32 UART RX characteristic not found. Trying fallback...');
      }

      debugPrint('✅ Service discovery complete');
    } catch (e) {
      debugPrint('❌ Service discovery error: $e');
      throw Exception('Service discovery failed: $e');
    }
  }

  /// Disconnect from device
  Future<void> disconnect() async {
    try {
      debugPrint('🔌 Disconnecting...');

      _isConnecting = false;
      await stopStreaming();

      if (_device != null) {
        await _device!.disconnect();
      }

      _isConnected = false;
      _device = null;
      _rxCharacteristic = null;

      debugPrint('✅ Disconnected');
    } catch (e) {
      debugPrint('⚠️ Disconnect error: $e');
      _isConnected = false;
    }
  }

  // ============== Streaming ==============

  /// Start streaming sensor data
  Future<void> startStreaming() async {
    try {
      if (kIsWeb) {
        throw Exception('Bluetooth not available on web');
      }

      if (!_isConnected || _rxCharacteristic == null) {
        throw Exception('Device not connected or RX characteristic not found');
      }

      if (_isStreaming) {
        return; // Already streaming
      }

      debugPrint('📡 Starting sensor data stream...');

      _streamController = StreamController<SensorReading>.broadcast();
      _isStreaming = true;

      // Enable notifications on RX characteristic
      await _rxCharacteristic!.setNotifyValue(true);

      // Listen to incoming data
      _rxSubscription = _rxCharacteristic!.lastValueStream.listen(
        (value) {
          _onDataReceived(value);
        },
        onError: (e) {
          debugPrint('❌ Stream error: $e');
          _streamController?.addError(e);
        },
        onDone: () {
          debugPrint('⚠️ Stream closed');
          _isStreaming = false;
        },
      );

      debugPrint('✅ Streaming started');
    } catch (e) {
      _isStreaming = false;
      debugPrint('❌ Failed to start streaming: $e');
      rethrow;
    }
  }

  /// Stop streaming
  Future<void> stopStreaming() async {
    try {
      _isStreaming = false;

      if (_rxCharacteristic != null) {
        await _rxCharacteristic!.setNotifyValue(false);
      }

      await _rxSubscription?.cancel();
      await _streamController?.close();
      _streamController = null;

      debugPrint('✅ Streaming stopped');
    } catch (e) {
      debugPrint('⚠️ Error stopping stream: $e');
    }
  }

  // ============== Data Reception ==============

  /// Handle incoming BLE data
  void _onDataReceived(List<int> data) {
    try {
      // Add to buffer
      _dataBuffer.addAll(data);

      // Process complete 16-byte packets
      while (_dataBuffer.length >= 16) {
        // Extract 16 bytes
        final packet = _dataBuffer.sublist(0, 16);

        // Parse packet
        final reading = _parsePayload(packet);
        if (reading != null) {
          _streamController?.add(reading);
        }

        // Remove processed bytes from buffer
        _dataBuffer.removeRange(0, 16);
      }
    } catch (e) {
      debugPrint('❌ Data reception error: $e');
    }
  }

  // ============== Payload Parsing ==============

  /// Parse 16-byte ESP32 payload into SensorReading
  /// Format (as per your ESP32 code):
  /// Bytes 0-3: Temperatures (encoded as (temp - 25.0) * 2.0 + 128)
  /// Bytes 4-7: Pressures (encoded as pressure / 0.3)
  /// Bytes 8-9: SpO2 (uint16, encoded as spo2 * 100)
  /// Bytes 10-11: Heart Rate (uint16)
  /// Bytes 12-13: Step Count (uint16)
  /// Byte 14: Activity Type
  /// Byte 15: Battery Level
  SensorReading? _parsePayload(List<int> packet) {
    try {
      if (packet.length != 16) {
        return null;
      }

      // Parse temperatures (Bytes 0-3)
      // Formula: temp = 25.0 + (byte - 128) / 2.0
      final temperatures = <double>[];
      for (int i = 0; i < 4; i++) {
        final tempByte = packet[i];
        final temp = 25.0 + (tempByte - 128) / 2.0;
        temperatures.add(temp);
      }

      // Parse pressures (Bytes 4-7)
      // Formula: pressure = byte * 0.3
      final pressures = <double>[];
      for (int i = 0; i < 4; i++) {
        final pressureByte = packet[4 + i];
        final pressure = pressureByte * 0.3;
        pressures.add(pressure);
      }

      // Parse SpO2 (Bytes 8-9)
      // uint16 big-endian, divided by 100
      final spO2Raw = (packet[8] << 8) | packet[9];
      final spO2 = spO2Raw / 100.0;

      // Parse Heart Rate (Bytes 10-11)
      // uint16 big-endian
      final heartRate = (packet[10] << 8) | packet[11];

      // Parse Step Count (Bytes 12-13)
      // uint16 big-endian
      final stepCount = (packet[12] << 8) | packet[13];

      // Parse Activity Type (Byte 14)
      final activityType = _parseActivityType(packet[14]);

      // Parse Battery Level (Byte 15)
      _batteryLevel = packet[15];

      // Generate dummy IMU data (ESP32 doesn't send accelerometer/gyroscope in this format)
      final accData = _generateDummyAccelerometerData(stepCount);
      final gyroData = _generateDummyGyroscopeData(stepCount);

      // Create SensorReading
      final reading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: temperatures,
        pressures: pressures,
        spO2: spO2,
        heartRate: heartRate,
        accelerometer: accData,
        gyroscope: gyroData,
        stepCount: stepCount,
        batteryLevel: _batteryLevel,
        activityType: activityType,
      );

      debugPrint(
          '✅ Parsed: Temp=${temperatures.join(',')} Pressure=${pressures.join(',')} SpO2=$spO2 HR=$heartRate');

      return reading;
    } catch (e) {
      debugPrint('❌ Parse error: $e');
      return null;
    }
  }

  /// Parse activity type from byte value
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

  /// Generate dummy accelerometer data based on activity
  AccelerometerData _generateDummyAccelerometerData(int stepCount) {
    if (stepCount > 0) {
      // Simulate motion during walking
      return AccelerometerData(x: 0.5, y: 0.3, z: 9.8);
    }
    return AccelerometerData(x: 0.0, y: 0.0, z: 9.8);
  }

  /// Generate dummy gyroscope data based on activity
  GyroscopeData _generateDummyGyroscopeData(int stepCount) {
    if (stepCount > 0) {
      // Simulate rotation during walking
      return GyroscopeData(x: 5.0, y: 3.0, z: 2.0);
    }
    return GyroscopeData(x: 0.5, y: 0.5, z: 0.5);
  }

  /// Print debug message
  static void debugPrint(String message) {
    if (kDebugMode) {
      // ignore: avoid_print
      print('[RealBleService] $message');
    }
  }
}
