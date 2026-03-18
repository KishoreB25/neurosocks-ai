// Example: How to use Real BLE and Predictions in Dashboard
// Copy relevant parts to your existing dashboard_screen.dart

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../providers/sensor_provider.dart';
import '../../providers/user_provider.dart';
import '../../data/services/foot_ulcer_prediction_service.dart';

class DashboardScreenExample extends StatefulWidget {
  const DashboardScreenExample({super.key});

  @override
  State<DashboardScreenExample> createState() => _DashboardScreenExampleState();
}

class _DashboardScreenExampleState extends State<DashboardScreenExample> {
  @override
  void initState() {
    super.initState();
    _initializeBluetoothConnection();
  }

  /// Initialize Bluetooth after user is logged in
  Future<void> _initializeBluetoothConnection() async {
    final userProvider = context.read<UserProvider>();
    final sensorProvider = context.read<SensorProvider>();

    // Set current user (should already be done in auth flow)
    if (userProvider.isLoggedIn) {
      sensorProvider.setCurrentUser(userProvider.userProfile!.id);
    }

    // Start BLE connection
    if (!sensorProvider.isStreaming) {
      try {
        // Real usage: Navigate to device scan screen for user to select device
        // For now, just start streaming (will use Firestore fallback if no device)
        
        // Start streaming with automatic BLE/Firestore fallback
        final isLive = await sensorProvider.startStreaming();
        
        if (!isLive && mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('No device connected. Showing recent data from Firestore.'),
            ),
          );
        }
      } catch (e) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Connection error: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Smart Socks Dashboard'),
      ),
      body: Consumer2<SensorProvider, UserProvider>(
        builder: (context, sensorProvider, userProvider, _) {
          // Show loading if not streaming
          if (!sensorProvider.isStreaming) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(
                    sensorProvider.errorMessage ?? 'Connecting...',
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
                ],
              ),
            );
          }

          // Show data when streaming
          final reading = sensorProvider.currentReading;
          if (reading == null) {
            return const Center(child: Text('No data received'));
          }

          // Generate prediction
          final prediction = FootUlcerPredictionService.predictRisk(
            reading,
            historicalReadings: sensorProvider.recentReadings,
          );

          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // User info
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Hello, ${userProvider.userName}',
                          style: Theme.of(context).textTheme.headlineSmall,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Device: ${sensorProvider.deviceName}',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        Text(
                          'Battery: ${sensorProvider.batteryLevel}%',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Risk Assessment (Foot Ulcer Prediction)
                Card(
                  color: _getRiskColor(prediction.level),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Foot Ulcer Risk Assessment',
                          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          'Risk Score: ${prediction.riskScore.toStringAsFixed(1)}%',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          prediction.level.toString().split('.').last.toUpperCase(),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          'Affected Zone: ${prediction.affectedZone}',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'Recommendation:',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                prediction.recommendation,
                                style: const TextStyle(color: Colors.white),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Risk Factors
                Text(
                  'Risk Factors Detected:',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 12),
                if (prediction.riskFactors.isEmpty)
                  const Text('No risk factors detected')
                else
                  Column(
                    children: prediction.riskFactors
                        .map((factor) => Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('• ', style: TextStyle(fontSize: 18)),
                              Expanded(child: Text(factor)),
                            ],
                          ),
                        ))
                        .toList(),
                  ),
                const SizedBox(height: 24),

                // Sensor Data
                Text(
                  'Current Readings:',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 12),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: [
                        _buildReadingRow('Avg Temperature', 
                          '${reading.averageTemperature.toStringAsFixed(1)}°C'),
                        _buildReadingRow('Max Pressure', 
                          '${reading.maxPressure.toStringAsFixed(1)} kPa'),
                        _buildReadingRow('Heart Rate', 
                          '${reading.heartRate} BPM'),
                        _buildReadingRow('SpO2', 
                          '${reading.spO2.toStringAsFixed(1)}%'),
                        _buildReadingRow('Steps', 
                          '${reading.stepCount}'),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  /// Build a reading display row
  Widget _buildReadingRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  /// Get color based on risk level
  Color _getRiskColor(UlcerRiskLevel level) {
    switch (level) {
      case UlcerRiskLevel.low:
        return Colors.green;
      case UlcerRiskLevel.moderate:
        return Colors.orange;
      case UlcerRiskLevel.high:
        return Colors.deepOrange;
      case UlcerRiskLevel.critical:
        return Colors.red;
    }
  }

  @override
  void dispose() {
    // Stop streaming when leaving screen
    context.read<SensorProvider>().stopStreaming();
    super.dispose();
  }
}
