import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../data/models/foot_data.dart';

/// Interactive foot heatmap showing temperature/pressure zones
class FootHeatmap extends StatelessWidget {
  /// Foot data to display
  final FootData? footData;

  /// Which data to visualize
  final HeatmapMode mode;

  /// Size of the widget
  final double size;

  /// Whether to show zone labels
  final bool showLabels;

  /// Whether to show values on zones
  final bool showValues;

  /// Callback when a zone is tapped
  final void Function(FootZone zone)? onZoneTap;

  /// Whether to show the foot outline
  final bool showOutline;

  const FootHeatmap({
    super.key,
    this.footData,
    this.mode = HeatmapMode.temperature,
    this.size = 280,
    this.showLabels = true,
    this.showValues = true,
    this.onZoneTap,
    this.showOutline = true,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: size,
      height: size * 1.6, // Foot is taller than wide
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Foot outline
          if (showOutline) _buildFootOutline(),

          // Zone overlays
          _buildZones(),

          // Side label
          Positioned(
            top: 8,
            left: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                footData?.side == FootSide.left ? 'LEFT' : 'RIGHT',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),

          // Mode indicator
          Positioned(
            top: 8,
            right: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: mode == HeatmapMode.temperature
                    ? AppColors.temperature.withOpacity(0.8)
                    : AppColors.pressure.withOpacity(0.8),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                mode == HeatmapMode.temperature ? '°C' : 'kPa',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFootOutline() {
    return CustomPaint(
      size: Size(size, size * 1.6),
      painter: _FootOutlinePainter(
        isLeftFoot: footData?.side == FootSide.left,
      ),
    );
  }

  Widget _buildZones() {
    if (footData == null) {
      return _buildEmptyState();
    }

    final zones = [
      footData!.toe,
      footData!.ball,
      footData!.arch,
      footData!.heel,
    ];

    // Zone positions (relative to container)
    final zonePositions = [
      // Toe - top
      Rect.fromLTWH(size * 0.2, size * 0.05, size * 0.6, size * 0.35),
      // Ball - upper middle
      Rect.fromLTWH(size * 0.1, size * 0.45, size * 0.8, size * 0.35),
      // Arch - middle
      Rect.fromLTWH(size * 0.15, size * 0.85, size * 0.7, size * 0.3),
      // Heel - bottom
      Rect.fromLTWH(size * 0.2, size * 1.2, size * 0.6, size * 0.35),
    ];

    return Stack(
      children: List.generate(zones.length, (index) {
        final zone = zones[index];
        final position = zonePositions[index];
        final color = _getZoneColor(zone);

        return Positioned(
          left: position.left,
          top: position.top,
          child: GestureDetector(
            onTap: onZoneTap != null ? () => onZoneTap!(zone) : null,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              width: position.width,
              height: position.height,
              decoration: BoxDecoration(
                color: color.withOpacity(0.6),
                borderRadius: BorderRadius.circular(
                  index == 0 ? 30 : (index == 3 ? 20 : 15),
                ),
                border: Border.all(
                  color: color,
                  width: 2,
                ),
                boxShadow: [
                  BoxShadow(
                    color: color.withOpacity(0.3),
                    blurRadius: 8,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: _buildZoneContent(zone, index),
            ),
          ),
        );
      }),
    );
  }

  Widget _buildZoneContent(FootZone zone, int index) {
    final value = mode == HeatmapMode.temperature
        ? zone.temperature
        : zone.pressure;
    final unit = mode == HeatmapMode.temperature ? '°C' : 'kPa';

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        if (showLabels)
          Text(
            zone.name,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 11,
              fontWeight: FontWeight.bold,
              shadows: [
                Shadow(
                  color: Colors.black54,
                  blurRadius: 2,
                ),
              ],
            ),
          ),
        if (showValues) ...[
          const SizedBox(height: 2),
          Text(
            '${value.toStringAsFixed(1)}$unit',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w600,
              shadows: [
                Shadow(
                  color: Colors.black54,
                  blurRadius: 2,
                ),
              ],
            ),
          ),
        ],
        // Risk indicator
        if (zone.riskLevel != ZoneRiskLevel.normal &&
            zone.riskLevel != ZoneRiskLevel.unknown) ...[
          const SizedBox(height: 4),
          Icon(
            zone.riskLevel == ZoneRiskLevel.critical
                ? Icons.error
                : zone.riskLevel == ZoneRiskLevel.high
                    ? Icons.warning
                    : Icons.info,
            color: Colors.white,
            size: 16,
          ),
        ],
      ],
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.bluetooth_searching,
            size: 48,
            color: Colors.grey[400],
          ),
          const SizedBox(height: 8),
          Text(
            'No Data',
            style: TextStyle(
              color: Colors.grey[500],
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }

  Color _getZoneColor(FootZone zone) {
    if (mode == HeatmapMode.temperature) {
      return AppColors.getTemperatureColor(zone.temperature);
    } else {
      return AppColors.getPressureColor(zone.pressure);
    }
  }
}

/// Foot outline painter
class _FootOutlinePainter extends CustomPainter {
  final bool isLeftFoot;

  _FootOutlinePainter({this.isLeftFoot = true});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.grey[300]!
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final path = Path();

    // Simplified foot outline
    final w = size.width;
    final h = size.height;

    // Start from heel
    path.moveTo(w * 0.3, h * 0.95);

    // Left side up
    path.quadraticBezierTo(w * 0.1, h * 0.7, w * 0.15, h * 0.5);
    path.quadraticBezierTo(w * 0.1, h * 0.3, w * 0.15, h * 0.2);

    // Toes
    path.quadraticBezierTo(w * 0.2, h * 0.08, w * 0.35, h * 0.05);
    path.quadraticBezierTo(w * 0.5, h * 0.02, w * 0.65, h * 0.05);
    path.quadraticBezierTo(w * 0.8, h * 0.08, w * 0.85, h * 0.2);

    // Right side down
    path.quadraticBezierTo(w * 0.9, h * 0.3, w * 0.85, h * 0.5);
    path.quadraticBezierTo(w * 0.9, h * 0.7, w * 0.7, h * 0.95);

    // Heel
    path.quadraticBezierTo(w * 0.5, h, w * 0.3, h * 0.95);

    path.close();

    // Mirror for right foot
    if (!isLeftFoot) {
      final matrix = Matrix4.identity()
        ..translate(size.width, 0.0)
        ..scale(-1.0, 1.0);
      path.transform(matrix.storage);
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

/// Mode for heatmap visualization
enum HeatmapMode {
  temperature,
  pressure,
}

/// Compact foot heatmap for smaller spaces
class CompactFootHeatmap extends StatelessWidget {
  final FootData? footData;
  final HeatmapMode mode;
  final double size;
  final VoidCallback? onTap;

  const CompactFootHeatmap({
    super.key,
    this.footData,
    this.mode = HeatmapMode.temperature,
    this.size = 100,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: FootHeatmap(
        footData: footData,
        mode: mode,
        size: size,
        showLabels: false,
        showValues: false,
        showOutline: true,
      ),
    );
  }
}

/// Side-by-side view of both feet
class DualFootHeatmap extends StatelessWidget {
  final FootData? leftFoot;
  final FootData? rightFoot;
  final HeatmapMode mode;
  final double footSize;
  final void Function(FootZone zone, FootSide side)? onZoneTap;

  const DualFootHeatmap({
    super.key,
    this.leftFoot,
    this.rightFoot,
    this.mode = HeatmapMode.temperature,
    this.footSize = 140,
    this.onZoneTap,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        FootHeatmap(
          footData: leftFoot,
          mode: mode,
          size: footSize,
          showLabels: false,
          onZoneTap: onZoneTap != null
              ? (zone) => onZoneTap!(zone, FootSide.left)
              : null,
        ),
        FootHeatmap(
          footData: rightFoot,
          mode: mode,
          size: footSize,
          showLabels: false,
          onZoneTap: onZoneTap != null
              ? (zone) => onZoneTap!(zone, FootSide.right)
              : null,
        ),
      ],
    );
  }
}
