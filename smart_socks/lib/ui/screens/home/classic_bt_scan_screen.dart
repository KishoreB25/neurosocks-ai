// Classic Bluetooth Scan Screen
// Discovers ESP32 devices using native Android platform channel (SPP).
// The ESP32 uses BluetoothSerial.begin("NeuroSock") — Classic BT only.

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import '../../../core/constants/app_colors.dart';
import '../../../data/services/classic_bluetooth_service.dart';
import '../../../providers/sensor_provider.dart';

/// Screen for scanning and connecting to Classic Bluetooth devices (ESP32)
class ClassicBtScanScreen extends StatefulWidget {
  const ClassicBtScanScreen({super.key});

  @override
  State<ClassicBtScanScreen> createState() => _ClassicBtScanScreenState();
}

class _ClassicBtScanScreenState extends State<ClassicBtScanScreen> {
  final ClassicBluetoothService _btService = ClassicBluetoothService();

  List<ClassicBtDevice> _pairedDevices = [];
  List<ClassicBtDevice> _discoveredDevices = [];
  bool _isScanning = false;
  bool _isConnecting = false;
  String? _connectingAddress;
  String? _errorMessage;
  bool _btEnabled = true;
  StreamSubscription<ClassicBtDevice>? _discoverySub;

  @override
  void initState() {
    super.initState();
    _init();
  }

  @override
  void dispose() {
    _discoverySub?.cancel();
    _btService.stopDiscovery();
    super.dispose();
  }

  // ============== Init ==============

  Future<void> _init() async {
    // Check BT state
    try {
      _btEnabled = await _btService.isEnabled();
      debugPrint('[ClassicBtScan] BT enabled: $_btEnabled');
    } catch (e) {
      debugPrint('[ClassicBtScan] Could not check BT state: $e');
      _btEnabled = true; // Assume on
    }
    if (mounted) setState(() {});

    // Request permissions
    await _requestPermissions();

    // Load paired + start discovery
    await _loadPairedDevices();
    if (_btEnabled) await _startDiscovery();
  }

  Future<void> _requestPermissions() async {
    if (!Platform.isAndroid) return;
    try {
      final statuses = await [
        Permission.bluetooth,
        Permission.bluetoothScan,
        Permission.bluetoothConnect,
        Permission.locationWhenInUse,
      ].request();
      for (var e in statuses.entries) {
        debugPrint('[ClassicBtScan] ${e.key}: ${e.value}');
      }
      if (!statuses.values.every((s) => s.isGranted || s.isLimited)) {
        setState(() => _errorMessage = 'Bluetooth permissions required.');
      }
    } catch (e) {
      debugPrint('[ClassicBtScan] Permission error: $e');
    }
  }

  // ============== Paired Devices ==============

  Future<void> _loadPairedDevices() async {
    try {
      final bonded = await _btService.getBondedDevices();
      debugPrint('[ClassicBtScan] Paired devices: ${bonded.length}');
      if (mounted) setState(() => _pairedDevices = bonded);
    } catch (e) {
      debugPrint('[ClassicBtScan] Paired devices error: $e');
    }
  }

  // ============== Discovery ==============

  Future<void> _startDiscovery() async {
    if (_isScanning) return;
    setState(() {
      _isScanning = true;
      _discoveredDevices.clear();
      _errorMessage = null;
    });

    debugPrint('[ClassicBtScan] 🔍 Starting discovery...');
    try {
      _discoverySub?.cancel();
      _discoverySub = _btService.startDiscovery().listen(
        (device) {
          // Skip already-paired or already-found devices
          final isPaired =
              _pairedDevices.any((d) => d.address == device.address);
          final alreadyFound =
              _discoveredDevices.any((d) => d.address == device.address);

          if (!isPaired && !alreadyFound) {
            debugPrint(
              '[ClassicBtScan] Found: ${device.name.isEmpty ? "?" : device.name} (${device.address})',
            );
            if (mounted) {
              setState(() => _discoveredDevices.add(device));
            }
          }
        },
        onDone: () {
          debugPrint('[ClassicBtScan] Discovery done');
          if (mounted) setState(() => _isScanning = false);
        },
        onError: (e) {
          debugPrint('[ClassicBtScan] Discovery error: $e');
          if (mounted) {
            setState(() {
              _isScanning = false;
              _errorMessage = 'Discovery error: $e';
            });
          }
        },
      );
    } catch (e) {
      debugPrint('[ClassicBtScan] Start discovery failed: $e');
      setState(() {
        _isScanning = false;
        _errorMessage = 'Discovery failed: $e';
      });
    }
  }

  Future<void> _stopDiscovery() async {
    await _btService.stopDiscovery();
    await _discoverySub?.cancel();
    _discoverySub = null;
    if (mounted) setState(() => _isScanning = false);
  }

  // ============== Connect ==============

  Future<void> _connectToDevice(ClassicBtDevice device) async {
    if (_isConnecting) return;

    setState(() {
      _isConnecting = true;
      _connectingAddress = device.address;
      _errorMessage = null;
    });

    // Stop discovery first
    await _stopDiscovery();

    try {
      debugPrint('[ClassicBtScan] Connecting to ${device.name}...');
      final provider = context.read<SensorProvider>();
      final ok = await provider.connectToClassicDevice(device);

      if (ok && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              '✅ Connected to ${device.name.isNotEmpty ? device.name : device.address}',
            ),
            backgroundColor: AppColors.success,
          ),
        );
        Navigator.pop(context, true);
      } else if (mounted) {
        setState(() => _errorMessage = 'Connection returned false');
      }
    } catch (e) {
      debugPrint('[ClassicBtScan] Connection error: $e');
      if (mounted) {
        setState(() => _errorMessage = 'Connection failed: $e');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('❌ Failed: $e'),
            backgroundColor: AppColors.error,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isConnecting = false;
          _connectingAddress = null;
        });
      }
    }
  }

  // ============== Build ==============

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Connect Device'),
        elevation: 0,
        actions: [
          if (_isScanning)
            IconButton(
              icon: const Icon(Icons.stop),
              onPressed: _stopDiscovery,
              tooltip: 'Stop',
            )
          else
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: () {
                _loadPairedDevices();
                _startDiscovery();
              },
              tooltip: 'Refresh',
            ),
        ],
      ),
      body: !_btEnabled ? _buildBtOffView() : _buildDeviceList(),
      floatingActionButton: _btEnabled && !_isScanning
          ? FloatingActionButton.extended(
              onPressed: _startDiscovery,
              icon: const Icon(Icons.search),
              label: const Text('Scan'),
              backgroundColor: AppColors.primary,
            )
          : null,
    );
  }

  Widget _buildBtOffView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.bluetooth_disabled, size: 80, color: Colors.grey),
          const SizedBox(height: 16),
          const Text('Bluetooth is OFF',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Text('Please enable Bluetooth to scan for devices',
              style: TextStyle(color: Colors.grey)),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: () async {
              // Show system "Turn on Bluetooth?" dialog
              try {
                const channel = MethodChannel('com.neurosocks.app/classic_bt');
                final enabled = await channel.invokeMethod<bool>('requestEnable');
                if (enabled == true && mounted) {
                  setState(() => _btEnabled = true);
                  _init();
                }
              } catch (_) {
                // Fallback: open BT settings page
                try {
                  const channel = MethodChannel('com.neurosocks.app/classic_bt');
                  await channel.invokeMethod('openBluetoothSettings');
                } catch (_) {}
                await Future.delayed(const Duration(seconds: 2));
                _init();
              }
            },
            icon: const Icon(Icons.bluetooth),
            label: const Text('Enable Bluetooth'),
          ),
        ],
      ),
    );
  }

  Widget _buildDeviceList() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Error banner
        if (_errorMessage != null) ...[
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.red[50],
              border: Border.all(color: Colors.red),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(_errorMessage!,
                style: TextStyle(color: Colors.red[800])),
          ),
          const SizedBox(height: 16),
        ],

        // Scanning indicator
        if (_isScanning) ...[
          const LinearProgressIndicator(),
          const SizedBox(height: 8),
          const Center(
              child: Text('Scanning for devices...',
                  style: TextStyle(color: Colors.grey))),
          const SizedBox(height: 16),
        ],

        // ---- Paired Devices ----
        if (_pairedDevices.isNotEmpty) ...[
          const Text('PAIRED DEVICES',
              style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey,
                  letterSpacing: 1)),
          const SizedBox(height: 8),
          ..._pairedDevices.map((d) => _buildDeviceTile(d, isPaired: true)),
          const SizedBox(height: 24),
        ],

        // ---- Discovered Devices ----
        const Text('NEARBY DEVICES',
            style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: Colors.grey,
                letterSpacing: 1)),
        const SizedBox(height: 8),
        if (_discoveredDevices.isEmpty && !_isScanning)
          Center(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Column(
                children: [
                  Icon(Icons.bluetooth_searching,
                      size: 48, color: Colors.grey[400]),
                  const SizedBox(height: 12),
                  Text('No devices found',
                      style: TextStyle(color: Colors.grey[500])),
                  const SizedBox(height: 8),
                  Text(
                    'Make sure your ESP32 is powered on\nand within range.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey[400], fontSize: 13),
                  ),
                ],
              ),
            ),
          )
        else
          ..._discoveredDevices.map((d) => _buildDeviceTile(d)),
      ],
    );
  }

  Widget _buildDeviceTile(ClassicBtDevice device, {bool isPaired = false}) {
    final isThisConnecting =
        _isConnecting && _connectingAddress == device.address;
    final name = device.name.isNotEmpty ? device.name : device.address;
    final isNeuroSock =
        name.toLowerCase().contains('neuro') ||
        name.toLowerCase().contains('sock');

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: isNeuroSock
            ? BorderSide(
                color: AppColors.primary.withValues(alpha: 0.5), width: 2)
            : BorderSide.none,
      ),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: isNeuroSock
              ? AppColors.primary.withValues(alpha: 0.15)
              : Colors.grey.withValues(alpha: 0.15),
          child: Icon(
            isPaired ? Icons.bluetooth_connected : Icons.bluetooth,
            color: isNeuroSock ? AppColors.primary : Colors.grey,
          ),
        ),
        title: Row(
          children: [
            Expanded(
              child: Text(name,
                  style: const TextStyle(fontWeight: FontWeight.w600)),
            ),
            if (isNeuroSock)
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: AppColors.primary.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Text('ESP32',
                    style: TextStyle(
                        fontSize: 10,
                        color: AppColors.primary,
                        fontWeight: FontWeight.bold)),
              ),
          ],
        ),
        subtitle: Text(
          device.address,
          style: TextStyle(fontSize: 12, color: Colors.grey[500]),
        ),
        trailing: isThisConnecting
            ? const SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(strokeWidth: 2))
            : ElevatedButton(
                onPressed: () => _connectToDevice(device),
                style: ElevatedButton.styleFrom(
                  backgroundColor:
                      isNeuroSock ? AppColors.primary : Colors.grey[700],
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                ),
                child: const Text('Connect',
                    style: TextStyle(color: Colors.white, fontSize: 13)),
              ),
      ),
    );
  }
}
