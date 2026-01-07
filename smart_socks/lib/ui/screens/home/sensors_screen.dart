import 'package:flutter/material.dart' hide ConnectionState;
import 'package:provider/provider.dart';
import '../../../core/constants/app_colors.dart';
import '../../../core/constants/app_strings.dart';
import '../../../data/models/foot_data.dart';
import '../../../data/models/sensor_reading.dart';
import '../../../providers/sensor_provider.dart';
import '../../../providers/risk_provider.dart';
import '../../widgets/foot_heatmap.dart';
import '../../widgets/sensor_card.dart';
import '../../widgets/mini_chart.dart';

/// Detailed sensors screen with foot heatmaps and live readings
class SensorsScreen extends StatefulWidget {
  const SensorsScreen({super.key});

  @override
  State<SensorsScreen> createState() => _SensorsScreenState();
}

class _SensorsScreenState extends State<SensorsScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  HeatmapMode _heatmapMode = HeatmapMode.temperature;
  FootZone? _selectedZone;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Sensor Details'),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: 'Foot Map', icon: Icon(Icons.grid_view)),
            Tab(text: 'Live Data', icon: Icon(Icons.sensors)),
            Tab(text: 'History', icon: Icon(Icons.show_chart)),
          ],
        ),
      ),
      body: Consumer2<SensorProvider, RiskProvider>(
        builder: (context, sensorProvider, riskProvider, _) {
          return TabBarView(
            controller: _tabController,
            children: [
              _buildFootMapTab(sensorProvider),
              _buildLiveDataTab(sensorProvider, riskProvider),
              _buildHistoryTab(sensorProvider),
            ],
          );
        },
      ),
    );
  }

  // ============== Foot Map Tab ==============

  Widget _buildFootMapTab(SensorProvider provider) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Mode toggle
          _buildModeToggle(),
          const SizedBox(height: 20),

          // Dual foot heatmap
          DualFootHeatmap(
            leftFoot: provider.leftFootData,
            rightFoot: provider.rightFootData,
            mode: _heatmapMode,
            onZoneTap: (zone, side) {
              setState(() {
                _selectedZone = zone;
              });
              _showZoneDetails(provider, zone, side);
            },
          ),
          const SizedBox(height: 20),

          // Legend
          _buildLegend(),
          const SizedBox(height: 20),

          // Zone details card
          if (_selectedZone != null)
            _buildZoneDetailsCard(provider),
        ],
      ),
    );
  }

  Widget _buildModeToggle() {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: Colors.grey[200],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _buildToggleButton(
            label: AppStrings.temperature,
            icon: Icons.thermostat,
            isSelected: _heatmapMode == HeatmapMode.temperature,
            color: AppColors.temperature,
            onTap: () => setState(() => _heatmapMode = HeatmapMode.temperature),
          ),
          const SizedBox(width: 4),
          _buildToggleButton(
            label: AppStrings.pressure,
            icon: Icons.compress,
            isSelected: _heatmapMode == HeatmapMode.pressure,
            color: AppColors.pressure,
            onTap: () => setState(() => _heatmapMode = HeatmapMode.pressure),
          ),
        ],
      ),
    );
  }

  Widget _buildToggleButton({
    required String label,
    required IconData icon,
    required bool isSelected,
    required Color color,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        decoration: BoxDecoration(
          color: isSelected ? color : Colors.transparent,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Row(
          children: [
            Icon(
              icon,
              size: 18,
              color: isSelected ? Colors.white : Colors.grey[600],
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                color: isSelected ? Colors.white : Colors.grey[600],
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegend() {
    final isTemp = _heatmapMode == HeatmapMode.temperature;
    final labels = isTemp
        ? ['< 25°C', '25-30°C', '30-35°C', '35-38°C', '> 38°C']
        : ['< 20 kPa', '20-50 kPa', '50-100 kPa', '100-200 kPa', '> 200 kPa'];
    final colors = [
      Colors.blue,
      Colors.green,
      Colors.yellow,
      Colors.orange,
      Colors.red,
    ];

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            isTemp ? 'Temperature Scale' : 'Pressure Scale',
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: List.generate(5, (index) {
              return Column(
                children: [
                  Container(
                    width: 24,
                    height: 24,
                    decoration: BoxDecoration(
                      color: colors[index],
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    labels[index],
                    style: TextStyle(
                      fontSize: 9,
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              );
            }),
          ),
        ],
      ),
    );
  }

  void _showZoneDetails(SensorProvider provider, FootZone zone, FootSide side) {
    final footData = side == FootSide.left
        ? provider.leftFootData
        : provider.rightFootData;

    if (footData == null) return;

    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Container(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  _getZoneIcon(zone.index),
                  color: AppColors.primary,
                ),
                const SizedBox(width: 12),
                Text(
                  '${side == FootSide.left ? "Left" : "Right"} ${zone.name}',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const Divider(height: 24),
            _buildDetailRow(
              Icons.thermostat,
              AppStrings.temperature,
              '${zone.temperature.toStringAsFixed(1)}°C',
              AppColors.temperature,
            ),
            const SizedBox(height: 12),
            _buildDetailRow(
              Icons.compress,
              AppStrings.pressure,
              '${zone.pressure.toStringAsFixed(1)} kPa',
              AppColors.pressure,
            ),
            const SizedBox(height: 12),
            _buildDetailRow(
              Icons.warning_amber,
              'Status',
              zone.riskLevel.name.toUpperCase(),
              _getStatusColor(zone.riskLevel),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailRow(IconData icon, String label, String value, Color color) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: color, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            label,
            style: TextStyle(
              color: Colors.grey[600],
              fontSize: 14,
            ),
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 16,
          ),
        ),
      ],
    );
  }

  Widget _buildZoneDetailsCard(SensorProvider provider) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Selected Zone: ${_selectedZone?.name ?? "None"}',
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Tap on a zone in the foot map to see details',
            style: TextStyle(
              color: Colors.grey[600],
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }

  IconData _getZoneIcon(int zoneIndex) {
    switch (zoneIndex) {
      case 0: // Heel
        return Icons.radio_button_unchecked;
      case 1: // Ball
        return Icons.circle;
      case 2: // Arch
        return Icons.horizontal_rule;
      case 3: // Toe
        return Icons.more_horiz;
      default:
        return Icons.help_outline;
    }
  }

  Color _getStatusColor(ZoneRiskLevel riskLevel) {
    switch (riskLevel) {
      case ZoneRiskLevel.normal:
        return AppColors.success;
      case ZoneRiskLevel.moderate:
        return AppColors.warning;
      case ZoneRiskLevel.high:
        return AppColors.warning;
      case ZoneRiskLevel.critical:
        return AppColors.error;
      case ZoneRiskLevel.unknown:
        return Colors.grey;
    }
  }

  // ============== Live Data Tab ==============

  Widget _buildLiveDataTab(SensorProvider sensorProvider, RiskProvider riskProvider) {
    final reading = sensorProvider.currentReading;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Last updated
          Row(
            children: [
              Icon(Icons.access_time, size: 16, color: Colors.grey[500]),
              const SizedBox(width: 4),
              Text(
                '${AppStrings.lastUpdated}: ${_formatTime(reading?.timestamp)}',
                style: TextStyle(
                  color: Colors.grey[500],
                  fontSize: 12,
                ),
              ),
              const Spacer(),
              if (sensorProvider.isStreaming)
                Row(
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: AppColors.success,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 4),
                    const Text(
                      'Live',
                      style: TextStyle(
                        color: AppColors.success,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
            ],
          ),
          const SizedBox(height: 16),

          // Vital signs section
          const Text(
            'Vital Signs',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: SensorCard.spO2(
                  value: sensorProvider.spO2,
                  size: SensorCardSize.large,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: SensorCard.heartRate(
                  value: sensorProvider.heartRate,
                  size: SensorCardSize.large,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Temperature readings
          const Text(
            'Temperature by Zone',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          _buildZoneReadings(
            reading?.temperatures ?? [],
            AppStrings.celsius,
            AppColors.temperature,
          ),
          const SizedBox(height: 24),

          // Pressure readings
          const Text(
            'Pressure by Zone',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          _buildZoneReadings(
            reading?.pressures ?? [],
            AppStrings.kpa,
            AppColors.pressure,
          ),
          const SizedBox(height: 24),

          // Activity
          const Text(
            'Activity',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: SensorCard.steps(
                  value: sensorProvider.stepCount,
                  size: SensorCardSize.medium,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildActivityCard(sensorProvider.activityType),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildZoneReadings(List<double> values, String unit, Color color) {
    final zoneNames = ['Heel', 'Ball', 'Arch', 'Toe'];
    
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 2.2,
      ),
      itemCount: values.length.clamp(0, 4),
      itemBuilder: (context, index) {
        final value = values.length > index ? values[index] : 0.0;
        return Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: color.withValues(alpha: 0.2)),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  zoneNames[index][0],
                  style: TextStyle(
                    color: color,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    zoneNames[index],
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                    ),
                  ),
                  Text(
                    '${value.toStringAsFixed(1)} $unit',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildActivityCard(ActivityType activity) {
    IconData icon;
    String label;

    switch (activity) {
      case ActivityType.walking:
        icon = Icons.directions_walk;
        label = AppStrings.walking;
        break;
      case ActivityType.standing:
        icon = Icons.accessibility_new;
        label = AppStrings.standing;
        break;
      case ActivityType.sitting:
        icon = Icons.chair;
        label = AppStrings.sitting;
        break;
      case ActivityType.running:
        icon = Icons.directions_run;
        label = AppStrings.running;
        break;
      default:
        icon = Icons.help_outline;
        label = AppStrings.unknown;
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.gait.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.gait.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            AppStrings.activity,
            style: TextStyle(
              fontSize: 12,
              color: Colors.grey[600],
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Icon(icon, color: AppColors.gait),
              const SizedBox(width: 8),
              Text(
                label,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  String _formatTime(DateTime? timestamp) {
    if (timestamp == null) return 'Never';
    final now = DateTime.now();
    final diff = now.difference(timestamp);
    
    if (diff.inSeconds < 60) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${diff.inDays}d ago';
  }

  // ============== History Tab ==============

  Widget _buildHistoryTab(SensorProvider provider) {
    final readings = provider.recentReadings;

    if (readings.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.show_chart, size: 64, color: Colors.grey[300]),
            const SizedBox(height: 16),
            Text(
              AppStrings.noData,
              style: TextStyle(
                color: Colors.grey[500],
                fontSize: 16,
              ),
            ),
          ],
        ),
      );
    }

    // Extract trend data
    final tempTrend = readings
        .take(20)
        .map((r) => r.temperatures.isNotEmpty
            ? r.temperatures.reduce((a, b) => a + b) / r.temperatures.length
            : 0.0)
        .toList()
        .reversed
        .toList();

    final pressureTrend = readings
        .take(20)
        .map((r) => r.pressures.isNotEmpty
            ? r.pressures.reduce((a, b) => a + b) / r.pressures.length
            : 0.0)
        .toList()
        .reversed
        .toList();

    final spo2Trend = readings
        .take(20)
        .map((r) => r.spO2)
        .toList()
        .reversed
        .toList();

    final hrTrend = readings
        .take(20)
        .map((r) => r.heartRate.toDouble())
        .toList()
        .reversed
        .toList();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Time range selector
          Container(
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(
              color: Colors.grey[200],
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _buildTimeButton(AppStrings.day, true),
                _buildTimeButton(AppStrings.week, false),
                _buildTimeButton(AppStrings.month, false),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Temperature chart
          _buildChartCard(
            title: AppStrings.temperature,
            subtitle: 'Average across zones',
            data: tempTrend,
            color: AppColors.temperature,
            unit: '°C',
          ),
          const SizedBox(height: 16),

          // Pressure chart
          _buildChartCard(
            title: AppStrings.pressure,
            subtitle: 'Average across zones',
            data: pressureTrend,
            color: AppColors.pressure,
            unit: 'kPa',
          ),
          const SizedBox(height: 16),

          // SpO2 chart
          _buildChartCard(
            title: AppStrings.spO2,
            subtitle: 'Blood oxygen level',
            data: spo2Trend,
            color: AppColors.oxygen,
            unit: '%',
          ),
          const SizedBox(height: 16),

          // Heart rate chart
          _buildChartCard(
            title: AppStrings.heartRate,
            subtitle: 'Beats per minute',
            data: hrTrend,
            color: AppColors.heartRate,
            unit: 'BPM',
          ),
        ],
      ),
    );
  }

  Widget _buildTimeButton(String label, bool isSelected) {
    return GestureDetector(
      onTap: () {
        // TODO: Implement time range filtering
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? AppColors.primary : Colors.transparent,
          borderRadius: BorderRadius.circular(6),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.grey[600],
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            fontSize: 13,
          ),
        ),
      ),
    );
  }

  Widget _buildChartCard({
    required String title,
    required String subtitle,
    required List<double> data,
    required Color color,
    required String unit,
  }) {
    final currentValue = data.isNotEmpty ? data.last : 0.0;
    final avgValue = data.isNotEmpty
        ? data.reduce((a, b) => a + b) / data.length
        : 0.0;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                  Text(
                    subtitle,
                    style: TextStyle(
                      color: Colors.grey[500],
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '${currentValue.toStringAsFixed(1)} $unit',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 18,
                      color: color,
                    ),
                  ),
                  Text(
                    'Avg: ${avgValue.toStringAsFixed(1)}',
                    style: TextStyle(
                      color: Colors.grey[500],
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 80,
            child: MiniChart(
              data: data,
              lineColor: color,
              showGradient: true,
              showDots: false,
            ),
          ),
        ],
      ),
    );
  }
}
