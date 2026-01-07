import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../core/constants/sensor_constants.dart';

/// Types of sensors that can be displayed
enum SensorType {
  temperature,
  pressure,
  spO2,
  heartRate,
  steps,
  battery,
}

/// Card widget displaying a sensor reading with icon, value, and optional trend
class SensorCard extends StatelessWidget {
  /// Type of sensor
  final SensorType type;

  /// Current value
  final double value;

  /// Unit to display (e.g., °C, kPa, %)
  final String unit;

  /// Optional label override
  final String? label;

  /// Optional trend data (list of recent values)
  final List<double>? trend;

  /// Whether the value is in warning/critical range
  final bool? isWarning;
  final bool? isCritical;

  /// Custom icon override
  final IconData? icon;

  /// Custom color override
  final Color? color;

  /// Card size
  final SensorCardSize size;

  /// On tap callback
  final VoidCallback? onTap;

  const SensorCard({
    super.key,
    required this.type,
    required this.value,
    required this.unit,
    this.label,
    this.trend,
    this.isWarning,
    this.isCritical,
    this.icon,
    this.color,
    this.size = SensorCardSize.medium,
    this.onTap,
  });

  /// Create a temperature sensor card
  factory SensorCard.temperature({
    required double value,
    List<double>? trend,
    String? zone,
    VoidCallback? onTap,
    SensorCardSize size = SensorCardSize.medium,
  }) {
    final isWarning = value > SensorConstants.tempWarningHigh ||
        value < SensorConstants.tempWarningLow;
    final isCritical = value > SensorConstants.tempCriticalHigh ||
        value < SensorConstants.tempCriticalLow;

    return SensorCard(
      type: SensorType.temperature,
      value: value,
      unit: '°C',
      label: zone != null ? '$zone Temp' : null,
      trend: trend,
      isWarning: isWarning,
      isCritical: isCritical,
      size: size,
      onTap: onTap,
    );
  }

  /// Create a pressure sensor card
  factory SensorCard.pressure({
    required double value,
    List<double>? trend,
    String? zone,
    VoidCallback? onTap,
    SensorCardSize size = SensorCardSize.medium,
  }) {
    final isWarning = value > SensorConstants.pressureWarning;
    final isCritical = value > SensorConstants.pressureCritical;

    return SensorCard(
      type: SensorType.pressure,
      value: value,
      unit: 'kPa',
      label: zone != null ? '$zone Pressure' : null,
      trend: trend,
      isWarning: isWarning,
      isCritical: isCritical,
      size: size,
      onTap: onTap,
    );
  }

  /// Create an SpO2 sensor card
  factory SensorCard.spO2({
    required double value,
    List<double>? trend,
    VoidCallback? onTap,
    SensorCardSize size = SensorCardSize.medium,
  }) {
    final isWarning = value < SensorConstants.spo2Warning;
    final isCritical = value < SensorConstants.spo2Critical;

    return SensorCard(
      type: SensorType.spO2,
      value: value,
      unit: '%',
      trend: trend,
      isWarning: isWarning,
      isCritical: isCritical,
      size: size,
      onTap: onTap,
    );
  }

  /// Create a heart rate sensor card
  factory SensorCard.heartRate({
    required int value,
    List<double>? trend,
    VoidCallback? onTap,
    SensorCardSize size = SensorCardSize.medium,
  }) {
    final isWarning = value < SensorConstants.hrWarningLow ||
        value > SensorConstants.hrWarningHigh;
    final isCritical = value < SensorConstants.hrCriticalLow ||
        value > SensorConstants.hrCriticalHigh;

    return SensorCard(
      type: SensorType.heartRate,
      value: value.toDouble(),
      unit: 'BPM',
      trend: trend,
      isWarning: isWarning,
      isCritical: isCritical,
      size: size,
      onTap: onTap,
    );
  }

  /// Create a steps counter card
  factory SensorCard.steps({
    required int value,
    int? goal,
    VoidCallback? onTap,
    SensorCardSize size = SensorCardSize.medium,
  }) {
    return SensorCard(
      type: SensorType.steps,
      value: value.toDouble(),
      unit: goal != null ? '/ $goal' : 'steps',
      size: size,
      onTap: onTap,
    );
  }

  /// Create a battery level card
  factory SensorCard.battery({
    required int level,
    VoidCallback? onTap,
    SensorCardSize size = SensorCardSize.small,
  }) {
    return SensorCard(
      type: SensorType.battery,
      value: level.toDouble(),
      unit: '%',
      isWarning: level <= SensorConstants.batteryLow,
      isCritical: level <= SensorConstants.batteryCritical,
      size: size,
      onTap: onTap,
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final effectiveColor = _getEffectiveColor();
    final dimensions = _getDimensions();

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          width: dimensions.width,
          padding: EdgeInsets.all(dimensions.padding),
          decoration: BoxDecoration(
            color: theme.cardColor,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: (isCritical == true || isWarning == true)
                  ? effectiveColor.withOpacity(0.3)
                  : Colors.grey.withOpacity(0.1),
              width: 1.5,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.05),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header: Icon + Label
              Row(
                children: [
                  Container(
                    padding: EdgeInsets.all(dimensions.iconPadding),
                    decoration: BoxDecoration(
                      color: effectiveColor.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Icon(
                      icon ?? _getDefaultIcon(),
                      size: dimensions.iconSize,
                      color: effectiveColor,
                    ),
                  ),
                  SizedBox(width: dimensions.spacing),
                  Expanded(
                    child: Text(
                      label ?? _getDefaultLabel(),
                      style: TextStyle(
                        fontSize: dimensions.labelSize,
                        fontWeight: FontWeight.w500,
                        color: Colors.grey[600],
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  if (isCritical == true || isWarning == true)
                    Icon(
                      isCritical == true ? Icons.error : Icons.warning,
                      size: dimensions.iconSize * 0.8,
                      color: effectiveColor,
                    ),
                ],
              ),

              SizedBox(height: dimensions.spacing),

              // Value
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    _formatValue(),
                    style: TextStyle(
                      fontSize: dimensions.valueSize,
                      fontWeight: FontWeight.bold,
                      color: effectiveColor,
                    ),
                  ),
                  SizedBox(width: 4),
                  Padding(
                    padding: const EdgeInsets.only(bottom: 2),
                    child: Text(
                      unit,
                      style: TextStyle(
                        fontSize: dimensions.unitSize,
                        fontWeight: FontWeight.w500,
                        color: Colors.grey[500],
                      ),
                    ),
                  ),
                ],
              ),

              // Trend indicator (if available)
              if (trend != null && trend!.length >= 2) ...[
                SizedBox(height: dimensions.spacing),
                _buildTrendIndicator(dimensions),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTrendIndicator(_SensorCardDimensions dimensions) {
    final trendDirection = _calculateTrend();
    final trendColor = trendDirection > 0
        ? (type == SensorType.spO2 ? Colors.green : Colors.orange)
        : trendDirection < 0
            ? (type == SensorType.spO2 ? Colors.orange : Colors.green)
            : Colors.grey;

    return Row(
      children: [
        Icon(
          trendDirection > 0
              ? Icons.trending_up
              : trendDirection < 0
                  ? Icons.trending_down
                  : Icons.trending_flat,
          size: dimensions.iconSize * 0.7,
          color: trendColor,
        ),
        SizedBox(width: 4),
        Text(
          '${trendDirection > 0 ? '+' : ''}${trendDirection.toStringAsFixed(1)}',
          style: TextStyle(
            fontSize: dimensions.labelSize * 0.9,
            color: trendColor,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  double _calculateTrend() {
    if (trend == null || trend!.length < 2) return 0;
    return trend!.last - trend!.first;
  }

  String _formatValue() {
    if (type == SensorType.steps || type == SensorType.heartRate) {
      return value.round().toString();
    }
    if (type == SensorType.battery) {
      return value.round().toString();
    }
    return value.toStringAsFixed(1);
  }

  Color _getEffectiveColor() {
    if (color != null) return color!;
    if (isCritical == true) return AppColors.riskCritical;
    if (isWarning == true) return AppColors.riskHigh;

    switch (type) {
      case SensorType.temperature:
        return AppColors.temperature;
      case SensorType.pressure:
        return AppColors.pressure;
      case SensorType.spO2:
        return AppColors.oxygen;
      case SensorType.heartRate:
        return AppColors.heartRate;
      case SensorType.steps:
        return AppColors.gait;
      case SensorType.battery:
        if (value <= SensorConstants.batteryCritical) return AppColors.riskCritical;
        if (value <= SensorConstants.batteryLow) return AppColors.riskHigh;
        return Colors.green;
    }
  }

  IconData _getDefaultIcon() {
    switch (type) {
      case SensorType.temperature:
        return Icons.thermostat;
      case SensorType.pressure:
        return Icons.speed;
      case SensorType.spO2:
        return Icons.bloodtype;
      case SensorType.heartRate:
        return Icons.favorite;
      case SensorType.steps:
        return Icons.directions_walk;
      case SensorType.battery:
        if (value <= 20) return Icons.battery_alert;
        if (value <= 50) return Icons.battery_3_bar;
        return Icons.battery_full;
    }
  }

  String _getDefaultLabel() {
    switch (type) {
      case SensorType.temperature:
        return 'Temperature';
      case SensorType.pressure:
        return 'Pressure';
      case SensorType.spO2:
        return 'Blood Oxygen';
      case SensorType.heartRate:
        return 'Heart Rate';
      case SensorType.steps:
        return 'Steps';
      case SensorType.battery:
        return 'Battery';
    }
  }

  _SensorCardDimensions _getDimensions() {
    switch (size) {
      case SensorCardSize.small:
        return _SensorCardDimensions(
          width: 120,
          padding: 12,
          iconSize: 18,
          iconPadding: 6,
          labelSize: 11,
          valueSize: 20,
          unitSize: 11,
          spacing: 8,
        );
      case SensorCardSize.medium:
        return _SensorCardDimensions(
          width: 160,
          padding: 16,
          iconSize: 22,
          iconPadding: 8,
          labelSize: 12,
          valueSize: 28,
          unitSize: 13,
          spacing: 12,
        );
      case SensorCardSize.large:
        return _SensorCardDimensions(
          width: 200,
          padding: 20,
          iconSize: 26,
          iconPadding: 10,
          labelSize: 14,
          valueSize: 36,
          unitSize: 15,
          spacing: 16,
        );
    }
  }
}

/// Size variants for sensor cards
enum SensorCardSize { small, medium, large }

/// Internal dimensions helper
class _SensorCardDimensions {
  final double width;
  final double padding;
  final double iconSize;
  final double iconPadding;
  final double labelSize;
  final double valueSize;
  final double unitSize;
  final double spacing;

  const _SensorCardDimensions({
    required this.width,
    required this.padding,
    required this.iconSize,
    required this.iconPadding,
    required this.labelSize,
    required this.valueSize,
    required this.unitSize,
    required this.spacing,
  });
}
