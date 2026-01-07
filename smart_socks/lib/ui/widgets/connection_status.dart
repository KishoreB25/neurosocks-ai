import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../core/constants/sensor_constants.dart';

/// Connection state enum
enum ConnectionState {
  disconnected,
  connecting,
  connected,
  error,
}

/// Widget showing BLE connection status and battery level
class ConnectionStatus extends StatelessWidget {
  /// Current connection state
  final ConnectionState state;

  /// Device name (if connected)
  final String? deviceName;

  /// Battery level (0-100)
  final int? batteryLevel;

  /// Signal strength (RSSI in dBm, typically -30 to -100)
  final int? signalStrength;

  /// Callback when tapped
  final VoidCallback? onTap;

  /// Whether to show detailed info
  final bool showDetails;

  /// Size variant
  final ConnectionStatusSize size;

  const ConnectionStatus({
    super.key,
    required this.state,
    this.deviceName,
    this.batteryLevel,
    this.signalStrength,
    this.onTap,
    this.showDetails = false,
    this.size = ConnectionStatusSize.medium,
  });

  @override
  Widget build(BuildContext context) {
    if (showDetails) {
      return _buildDetailedView(context);
    }
    return _buildCompactView(context);
  }

  Widget _buildCompactView(BuildContext context) {
    final dimensions = _getDimensions();

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(dimensions.borderRadius),
        child: Container(
          padding: EdgeInsets.symmetric(
            horizontal: dimensions.paddingH,
            vertical: dimensions.paddingV,
          ),
          decoration: BoxDecoration(
            color: _getBackgroundColor(),
            borderRadius: BorderRadius.circular(dimensions.borderRadius),
            border: Border.all(
              color: _getStateColor().withValues(alpha: 0.3),
              width: 1,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Connection icon with animation
              _buildConnectionIcon(dimensions),
              SizedBox(width: dimensions.spacing),

              // Status text
              Text(
                _getStatusText(),
                style: TextStyle(
                  fontSize: dimensions.fontSize,
                  fontWeight: FontWeight.w500,
                  color: _getStateColor(),
                ),
              ),

              // Battery indicator (if connected)
              if (state == ConnectionState.connected &&
                  batteryLevel != null) ...[
                SizedBox(width: dimensions.spacing * 1.5),
                _buildBatteryIndicator(dimensions),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDetailedView(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: theme.cardColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header
          Row(
            children: [
              _buildConnectionIcon(_getDimensions()),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      deviceName ?? 'Smart Socks',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _getStatusText(),
                      style: TextStyle(
                        fontSize: 13,
                        color: _getStateColor(),
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
              if (onTap != null)
                IconButton(
                  onPressed: onTap,
                  icon: Icon(
                    state == ConnectionState.connected
                        ? Icons.bluetooth_connected
                        : Icons.bluetooth_searching,
                    color: _getStateColor(),
                  ),
                ),
            ],
          ),

          if (state == ConnectionState.connected) ...[
            const SizedBox(height: 16),
            const Divider(height: 1),
            const SizedBox(height: 16),

            // Stats row
            Row(
              children: [
                // Battery
                Expanded(
                  child: _buildStatItem(
                    icon: _getBatteryIcon(),
                    label: 'Battery',
                    value: batteryLevel != null ? '$batteryLevel%' : '--',
                    color: _getBatteryColor(),
                  ),
                ),

                // Signal
                Expanded(
                  child: _buildStatItem(
                    icon: _getSignalIcon(),
                    label: 'Signal',
                    value: _getSignalText(),
                    color: _getSignalColor(),
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStatItem({
    required IconData icon,
    required String label,
    required String value,
    required Color color,
  }) {
    return Column(
      children: [
        Icon(icon, size: 24, color: color),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            fontSize: 11,
            color: Colors.grey[600],
          ),
        ),
      ],
    );
  }

  Widget _buildConnectionIcon(_ConnectionDimensions dimensions) {
    IconData icon;
    Widget iconWidget;

    switch (state) {
      case ConnectionState.disconnected:
        icon = Icons.bluetooth_disabled;
        iconWidget = Icon(icon, size: dimensions.iconSize, color: _getStateColor());
        break;
      case ConnectionState.connecting:
        icon = Icons.bluetooth_searching;
        iconWidget = _AnimatedBluetoothIcon(
          size: dimensions.iconSize,
          color: _getStateColor(),
        );
        break;
      case ConnectionState.connected:
        icon = Icons.bluetooth_connected;
        iconWidget = Icon(icon, size: dimensions.iconSize, color: _getStateColor());
        break;
      case ConnectionState.error:
        icon = Icons.bluetooth_disabled;
        iconWidget = Icon(icon, size: dimensions.iconSize, color: _getStateColor());
        break;
    }

    return iconWidget;
  }

  Widget _buildBatteryIndicator(_ConnectionDimensions dimensions) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          _getBatteryIcon(),
          size: dimensions.iconSize * 0.9,
          color: _getBatteryColor(),
        ),
        const SizedBox(width: 4),
        Text(
          '$batteryLevel%',
          style: TextStyle(
            fontSize: dimensions.fontSize * 0.9,
            fontWeight: FontWeight.w500,
            color: _getBatteryColor(),
          ),
        ),
      ],
    );
  }

  Color _getStateColor() {
    switch (state) {
      case ConnectionState.disconnected:
        return Colors.grey;
      case ConnectionState.connecting:
        return AppColors.warning;
      case ConnectionState.connected:
        return AppColors.success;
      case ConnectionState.error:
        return AppColors.error;
    }
  }

  Color _getBackgroundColor() {
    return _getStateColor().withValues(alpha: 0.1);
  }

  String _getStatusText() {
    switch (state) {
      case ConnectionState.disconnected:
        return 'Disconnected';
      case ConnectionState.connecting:
        return 'Connecting...';
      case ConnectionState.connected:
        return 'Connected';
      case ConnectionState.error:
        return 'Error';
    }
  }

  IconData _getBatteryIcon() {
    if (batteryLevel == null) return Icons.battery_unknown;
    if (batteryLevel! <= SensorConstants.batteryCritical) {
      return Icons.battery_alert;
    }
    if (batteryLevel! <= SensorConstants.batteryLow) {
      return Icons.battery_2_bar;
    }
    if (batteryLevel! <= 50) return Icons.battery_4_bar;
    if (batteryLevel! <= 80) return Icons.battery_5_bar;
    return Icons.battery_full;
  }

  Color _getBatteryColor() {
    if (batteryLevel == null) return Colors.grey;
    if (batteryLevel! <= SensorConstants.batteryCritical) {
      return AppColors.riskCritical;
    }
    if (batteryLevel! <= SensorConstants.batteryLow) {
      return AppColors.riskHigh;
    }
    return AppColors.success;
  }

  IconData _getSignalIcon() {
    if (signalStrength == null) return Icons.signal_cellular_null;
    if (signalStrength! >= -50) return Icons.signal_cellular_4_bar;
    if (signalStrength! >= -60) return Icons.signal_cellular_4_bar;
    if (signalStrength! >= -70) return Icons.network_cell;
    return Icons.signal_cellular_0_bar;
  }

  Color _getSignalColor() {
    if (signalStrength == null) return Colors.grey;
    if (signalStrength! >= -60) return AppColors.success;
    if (signalStrength! >= -80) return AppColors.warning;
    return AppColors.error;
  }

  String _getSignalText() {
    if (signalStrength == null) return '--';
    if (signalStrength! >= -50) return 'Excellent';
    if (signalStrength! >= -60) return 'Good';
    if (signalStrength! >= -70) return 'Fair';
    return 'Weak';
  }

  _ConnectionDimensions _getDimensions() {
    switch (size) {
      case ConnectionStatusSize.small:
        return _ConnectionDimensions(
          iconSize: 16,
          fontSize: 11,
          paddingH: 8,
          paddingV: 4,
          spacing: 6,
          borderRadius: 8,
        );
      case ConnectionStatusSize.medium:
        return _ConnectionDimensions(
          iconSize: 20,
          fontSize: 13,
          paddingH: 12,
          paddingV: 6,
          spacing: 8,
          borderRadius: 10,
        );
      case ConnectionStatusSize.large:
        return _ConnectionDimensions(
          iconSize: 24,
          fontSize: 15,
          paddingH: 16,
          paddingV: 8,
          spacing: 10,
          borderRadius: 12,
        );
    }
  }
}

/// Animated bluetooth icon for connecting state
class _AnimatedBluetoothIcon extends StatefulWidget {
  final double size;
  final Color color;

  const _AnimatedBluetoothIcon({
    required this.size,
    required this.color,
  });

  @override
  State<_AnimatedBluetoothIcon> createState() => _AnimatedBluetoothIconState();
}

class _AnimatedBluetoothIconState extends State<_AnimatedBluetoothIcon>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat(reverse: true);
    _animation = Tween<double>(begin: 0.5, end: 1.0).animate(_controller);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Opacity(
          opacity: _animation.value,
          child: Icon(
            Icons.bluetooth_searching,
            size: widget.size,
            color: widget.color,
          ),
        );
      },
    );
  }
}

/// Size variants
enum ConnectionStatusSize { small, medium, large }

/// Internal dimensions
class _ConnectionDimensions {
  final double iconSize;
  final double fontSize;
  final double paddingH;
  final double paddingV;
  final double spacing;
  final double borderRadius;

  const _ConnectionDimensions({
    required this.iconSize,
    required this.fontSize,
    required this.paddingH,
    required this.paddingV,
    required this.spacing,
    required this.borderRadius,
  });
}

/// Simple battery indicator widget
class BatteryIndicator extends StatelessWidget {
  final int level;
  final bool showPercentage;
  final double size;

  const BatteryIndicator({
    super.key,
    required this.level,
    this.showPercentage = true,
    this.size = 24,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          _getIcon(),
          size: size,
          color: _getColor(),
        ),
        if (showPercentage) ...[
          const SizedBox(width: 4),
          Text(
            '$level%',
            style: TextStyle(
              fontSize: size * 0.5,
              fontWeight: FontWeight.w500,
              color: _getColor(),
            ),
          ),
        ],
      ],
    );
  }

  IconData _getIcon() {
    if (level <= SensorConstants.batteryCritical) return Icons.battery_alert;
    if (level <= SensorConstants.batteryLow) return Icons.battery_2_bar;
    if (level <= 50) return Icons.battery_4_bar;
    if (level <= 80) return Icons.battery_5_bar;
    return Icons.battery_full;
  }

  Color _getColor() {
    if (level <= SensorConstants.batteryCritical) return AppColors.riskCritical;
    if (level <= SensorConstants.batteryLow) return AppColors.riskHigh;
    return AppColors.success;
  }
}
