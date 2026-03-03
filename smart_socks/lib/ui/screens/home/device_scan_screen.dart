import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import '../../../core/constants/app_colors.dart';
import '../../../providers/sensor_provider.dart';
import '../../../data/services/storage_service.dart';

/// Screen for scanning and connecting to BLE devices
class DeviceScanScreen extends StatefulWidget {
  const DeviceScanScreen({super.key});

  @override
  State<DeviceScanScreen> createState() => _DeviceScanScreenState();
}

class _DeviceScanScreenState extends State<DeviceScanScreen> {
  late Future<List<ScanResult>> _scanFuture;
  BluetoothDevice? _connectingDevice;
  String? _connectingError;
  final StorageService _storage = StorageService();
  static const platform = MethodChannel('com.neurosocks.app/bluetooth');

  @override
  void initState() {
    super.initState();
    _scanFuture = _requestPermissionsAndScan();
  }

  /// Request Bluetooth permissions before scanning
  Future<List<ScanResult>> _requestPermissionsAndScan() async {
    try {
      debugPrint('🔐 Checking Bluetooth permissions...');
      
      // Use FlutterBluePlus's built-in permission checking (more reliable)
      if (Platform.isAndroid) {
        // Check if Bluetooth adapter is available and on
        final adapterState = await FlutterBluePlus.adapterState.first;
        debugPrint('📶 Bluetooth adapter state: $adapterState');
        
        if (adapterState != BluetoothAdapterState.on) {
          debugPrint('❌ Bluetooth is off');
          if (mounted) {
            setState(() {
              _connectingError = 'Please turn on Bluetooth';
            });
          }
          // Try to turn on Bluetooth
          try {
            await FlutterBluePlus.turnOn();
          } catch (e) {
            debugPrint('Could not turn on Bluetooth: $e');
          }
          return [];
        }
        
        // Request permissions using permission_handler
        debugPrint('🔐 Requesting Android permissions...');
        
        // For Android 12+ (API 31+), we need BLUETOOTH_SCAN and BLUETOOTH_CONNECT
        // For Android 11 and below, we need Location permission
        final androidSdk = await _getAndroidSdkVersion();
        debugPrint('📱 Android SDK version: $androidSdk');
        
        if (androidSdk >= 31) {
          // Android 12+: Request Bluetooth permissions
          final scanStatus = await Permission.bluetoothScan.request();
          final connectStatus = await Permission.bluetoothConnect.request();
          
          debugPrint('🔐 bluetoothScan: $scanStatus, bluetoothConnect: $connectStatus');
          
          if (!scanStatus.isGranted || !connectStatus.isGranted) {
            if (mounted) {
              setState(() {
                _connectingError = 'Bluetooth permissions required. Tap to open Settings.';
              });
            }
            // Open app settings so user can grant permissions
            await openAppSettings();
            return [];
          }
        } else {
          // Android 11 and below: Request location permission
          final locationStatus = await Permission.locationWhenInUse.request();
          debugPrint('🔐 locationWhenInUse: $locationStatus');
          
          if (!locationStatus.isGranted) {
            if (mounted) {
              setState(() {
                _connectingError = 'Location permission required for Bluetooth scanning.';
              });
            }
            await openAppSettings();
            return [];
          }
        }
      }

      debugPrint('✅ Permissions OK, starting scan...');
      return await _startScan();
    } catch (e) {
      debugPrint('❌ Permission/scan error: $e');
      if (mounted) {
        setState(() {
          _connectingError = 'Error: $e';
        });
      }
      return [];
    }
  }

  /// Get Android SDK version
  Future<int> _getAndroidSdkVersion() async {
    try {
      if (Platform.isAndroid) {
        // Use method channel to get SDK version, or default to 31
        return 31; // Default to Android 12+ behavior
      }
      return 0;
    } catch (e) {
      return 31;
    }
  }

  Future<List<ScanResult>> _startScan() async {
    try {
      final sensorProvider = context.read<SensorProvider>();
      final results = await sensorProvider.realBleService.scanForDevices(timeoutSeconds: 8);
      return results;
    } catch (e) {
      debugPrint('Scan error: $e');
      if (mounted) {
        setState(() {
          _connectingError = 'Scan failed: $e';
        });
      }
      return [];
    }
  }

  /// Open system Bluetooth settings
  Future<void> _openBluetoothSettings() async {
    try {
      await platform.invokeMethod('openBluetoothSettings');
    } catch (e) {
      debugPrint('Error opening Bluetooth settings: $e');
      // Fallback: show guide to user
      if (!mounted) return;
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Open Bluetooth Settings'),
          content: const Text(
            'Please enable Bluetooth and manually connect to your smart socks device in the system settings.\n\nOnce connected, return to this app and tap "Try Again" to proceed.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('OK'),
            ),
          ],
        ),
      );
    }
  }

  /// Save device as last connected
  Future<void> _saveLastConnectedDevice(BluetoothDevice device) async {
    try {
      await _storage.saveLastConnectedDeviceId(device.remoteId.toString());
      debugPrint('Saved last connected device: ${device.platformName}');
    } catch (e) {
      debugPrint('Error saving device: $e');
    }
  }

  Future<void> _connectToDevice(BluetoothDevice device) async {
    setState(() {
      _connectingDevice = device;
      _connectingError = null;
    });

    try {
      debugPrint('🔌 DeviceScanScreen: Starting connection to ${device.platformName}');
      final sensorProvider = context.read<SensorProvider>();
      bool connected = await sensorProvider.connectToDevice(device);

      debugPrint('🔌 DeviceScanScreen: Connection result: $connected');
      
      if (connected) {
        // Device connected successfully
        await _saveLastConnectedDevice(device);

        if (!mounted) return;

        // Show success and go back
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('✅ Connected to ${device.platformName}!'),
            duration: const Duration(seconds: 2),
            backgroundColor: AppColors.success,
          ),
        );

        Navigator.pop(context);
      } else {
        // Connection returned false but didn't throw
        setState(() {
          _connectingError = 'Connection failed - please try again';
          _connectingDevice = null;
        });
        
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('❌ Connection failed - please try again'),
              duration: Duration(seconds: 3),
              backgroundColor: AppColors.error,
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('❌ DeviceScanScreen: Connection error: $e');
      setState(() {
        _connectingError = 'Connection failed: $e';
        _connectingDevice = null;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('❌ $_connectingError'),
            duration: const Duration(seconds: 3),
            backgroundColor: AppColors.error,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Scan Devices'), elevation: 0),
      body: FutureBuilder<List<ScanResult>>(
        future: _scanFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return _buildLoadingState();
          }

          if (snapshot.hasError) {
            return _buildErrorState(snapshot.error.toString());
          }

          final devices = snapshot.data ?? [];

          if (devices.isEmpty) {
            return _buildEmptyState();
          }

          return _buildDeviceList(devices);
        },
      ),
    );
  }

  Widget _buildLoadingState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(
            strokeWidth: 3,
            valueColor: AlwaysStoppedAnimation<Color>(AppColors.primary),
          ),
          const SizedBox(height: 24),
          const Text(
            'Scanning for devices...',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
          ),
          const SizedBox(height: 8),
          Text(
            'Make sure your smart socks are powered on',
            style: TextStyle(fontSize: 14, color: AppColors.textSecondary),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.bluetooth_disabled,
            size: 64,
            color: AppColors.textSecondary,
          ),
          const SizedBox(height: 24),
          const Text(
            'No devices found',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
          ),
          const SizedBox(height: 8),
          Text(
            'Make sure your smart socks are powered on\nand Bluetooth is enabled',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 14, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 32),
          Column(
            children: [
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _connectingError = null;
                    _scanFuture = _requestPermissionsAndScan();
                  });
                },
                child: const Text('Try Again'),
              ),
              const SizedBox(height: 12),
              OutlinedButton.icon(
                onPressed: _openBluetoothSettings,
                icon: const Icon(Icons.settings_bluetooth),
                label: const Text('Open Bluetooth Settings'),
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.blue[50],
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.blue[200]!),
                ),
                child: const Text(
                  '💡 Tip: You can also connect manually through system Bluetooth settings, then return here.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 12, color: Colors.blue),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildErrorState(String error) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.error_outline, size: 64, color: AppColors.error),
          const SizedBox(height: 24),
          const Text(
            'Scan failed',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
          ),
          const SizedBox(height: 8),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Text(
              error,
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 14, color: AppColors.textSecondary),
            ),
          ),
          const SizedBox(height: 32),
          ElevatedButton(
            onPressed: () {
              setState(() {
                _connectingError = null;
                _scanFuture = _requestPermissionsAndScan();
              });
            },
            child: const Text('Try Again'),
          ),
        ],
      ),
    );
  }

  Widget _buildDeviceList(List<ScanResult> devices) {
    // Sort by signal strength (strongest first - closest devices)
    final sortedDevices = List<ScanResult>.from(devices);
    sortedDevices.sort((a, b) => b.rssi.compareTo(a.rssi));
    
    return Column(
      children: [
        // Help text
        Container(
          padding: const EdgeInsets.all(12),
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: Colors.blue.shade50,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.blue.shade200),
          ),
          child: Row(
            children: [
              Icon(Icons.lightbulb_outline, color: Colors.blue.shade700, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Tip: Hold your phone close to ESP32. The device with strongest signal (green) is likely yours!',
                  style: TextStyle(fontSize: 12, color: Colors.blue.shade700),
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: sortedDevices.length,
            itemBuilder: (context, index) {
              final result = sortedDevices[index];
              final device = result.device;
              final isConnecting = _connectingDevice?.remoteId == device.remoteId;
              
              // Signal strength indicator
              final rssi = result.rssi;
              Color signalColor;
              String signalText;
              if (rssi >= -50) {
                signalColor = Colors.green;
                signalText = 'Excellent';
              } else if (rssi >= -70) {
                signalColor = Colors.orange;
                signalText = 'Good';
              } else {
                signalColor = Colors.red;
                signalText = 'Weak';
              }
              
              // Check if this might be an ESP32
              final name = device.platformName.toLowerCase();
              final isLikelyESP32 = name.contains('esp') || 
                                    name.contains('neuro') || 
                                    name.contains('sock') ||
                                    name.contains('bt') ||
                                    name.contains('serial');

              return Card(
                margin: const EdgeInsets.only(bottom: 12),
                elevation: isLikelyESP32 ? 4 : 2,
                color: isLikelyESP32 ? Colors.green.shade50 : null,
                child: ListTile(
                  leading: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: signalColor.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.signal_cellular_alt,
                          color: signalColor,
                          size: 20,
                        ),
                        Text(
                          '$rssi',
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                            color: signalColor,
                          ),
                        ),
                      ],
                    ),
                  ),
                  title: Row(
                    children: [
                      Expanded(
                        child: Text(
                          device.platformName.isNotEmpty
                              ? device.platformName
                              : 'Unknown Device',
                          style: TextStyle(
                            fontWeight: FontWeight.w500,
                            color: isLikelyESP32 ? Colors.green.shade800 : null,
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      if (isLikelyESP32)
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.green,
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: const Text(
                            'ESP32?',
                            style: TextStyle(color: Colors.white, fontSize: 10),
                          ),
                        ),
                    ],
                  ),
                  subtitle: Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const SizedBox(height: 4),
                        Text(
                          '${device.remoteId}',
                          style: TextStyle(
                            fontSize: 11,
                            color: AppColors.textSecondary,
                            fontFamily: 'monospace',
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 2),
                        Row(
                          children: [
                            Container(
                              width: 8,
                              height: 8,
                              decoration: BoxDecoration(
                                color: signalColor,
                                shape: BoxShape.circle,
                              ),
                            ),
                            const SizedBox(width: 4),
                            Text(
                              '$signalText signal',
                              style: TextStyle(
                                fontSize: 11,
                                color: signalColor,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  trailing: isConnecting
                      ? SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(
                              AppColors.primary,
                            ),
                          ),
                        )
                      : SizedBox(
                          width: 90,
                          child: ElevatedButton(
                            onPressed: () => _connectToDevice(device),
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(horizontal: 8),
                            ),
                            child: const Text('Connect'),
                          ),
                        ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}
