// Device Connection Service
// Manages device discovery and connection workflow

import 'package:flutter_blue_plus/flutter_blue_plus.dart';

class DeviceConnection {
  final BluetoothDevice device;
  final String name;
  final int rssi; // Signal strength
  final bool isConnected;

  DeviceConnection({
    required this.device,
    required this.name,
    required this.rssi,
    this.isConnected = false,
  });
}

class DeviceConnectionService {
  static final DeviceConnectionService _instance = 
    DeviceConnectionService._internal();
  factory DeviceConnectionService() => _instance;
  DeviceConnectionService._internal();

  final List<DeviceConnection> _discoveredDevices = [];
  DeviceConnection? _selectedDevice;

  List<DeviceConnection> get discoveredDevices => _discoveredDevices;
  DeviceConnection? get selectedDevice => _selectedDevice;

  /// Get signal strength indicator
  static String getSignalStrength(int rssi) {
    if (rssi > -50) return 'Excellent';
    if (rssi > -60) return 'Good';
    if (rssi > -70) return 'Fair';
    if (rssi > -80) return 'Weak';
    return 'Very Weak';
  }

  /// Select a device for connection
  void selectDevice(DeviceConnection device) {
    _selectedDevice = device;
  }

  /// Clear discovered devices
  void clearDiscoveredDevices() {
    _discoveredDevices.clear();
  }

  /// Add discovered device
  void addDiscoveredDevice(DeviceConnection device) {
    // Update if already exists
    final index = _discoveredDevices.indexWhere(
      (d) => d.device.remoteId == device.device.remoteId,
    );
    if (index >= 0) {
      _discoveredDevices[index] = device;
    } else {
      _discoveredDevices.add(device);
    }
  }
}
