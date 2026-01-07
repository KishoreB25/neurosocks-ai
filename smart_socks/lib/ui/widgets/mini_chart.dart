import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../core/constants/app_colors.dart';

/// Compact sparkline chart for displaying trends
class MiniChart extends StatelessWidget {
  /// Data points to display
  final List<double> data;

  /// Color of the line
  final Color? lineColor;

  /// Whether to show gradient fill below line
  final bool showGradient;

  /// Width of the chart
  final double width;

  /// Height of the chart
  final double height;

  /// Line thickness
  final double lineWidth;

  /// Whether to show dots on data points
  final bool showDots;

  /// Whether to animate the chart
  final bool animate;

  /// Optional min Y value (auto-calculated if null)
  final double? minY;

  /// Optional max Y value (auto-calculated if null)
  final double? maxY;

  /// Whether to show the current value label
  final bool showCurrentValue;

  /// Curve smoothness (0 = sharp, 1 = very smooth)
  final double curveSmoothness;

  const MiniChart({
    super.key,
    required this.data,
    this.lineColor,
    this.showGradient = true,
    this.width = 120,
    this.height = 40,
    this.lineWidth = 2,
    this.showDots = false,
    this.animate = true,
    this.minY,
    this.maxY,
    this.showCurrentValue = false,
    this.curveSmoothness = 0.3,
  });

  /// Create a temperature trend chart
  factory MiniChart.temperature({
    required List<double> data,
    double width = 120,
    double height = 40,
  }) {
    return MiniChart(
      data: data,
      lineColor: AppColors.temperature,
      width: width,
      height: height,
    );
  }

  /// Create a pressure trend chart
  factory MiniChart.pressure({
    required List<double> data,
    double width = 120,
    double height = 40,
  }) {
    return MiniChart(
      data: data,
      lineColor: AppColors.pressure,
      width: width,
      height: height,
    );
  }

  /// Create a SpO2 trend chart
  factory MiniChart.spO2({
    required List<double> data,
    double width = 120,
    double height = 40,
  }) {
    return MiniChart(
      data: data,
      lineColor: AppColors.oxygen,
      width: width,
      height: height,
      minY: 85,
      maxY: 100,
    );
  }

  /// Create a heart rate trend chart
  factory MiniChart.heartRate({
    required List<double> data,
    double width = 120,
    double height = 40,
  }) {
    return MiniChart(
      data: data,
      lineColor: AppColors.heartRate,
      width: width,
      height: height,
    );
  }

  /// Create a risk score trend chart
  factory MiniChart.riskScore({
    required List<double> data,
    double width = 120,
    double height = 40,
  }) {
    return MiniChart(
      data: data,
      lineColor: AppColors.riskModerate,
      width: width,
      height: height,
      minY: 0,
      maxY: 100,
    );
  }

  @override
  Widget build(BuildContext context) {
    if (data.isEmpty) {
      return _buildEmptyState();
    }

    final effectiveColor = lineColor ?? Theme.of(context).primaryColor;
    final spots = _createSpots();
    final calculatedMinY = minY ?? _calculateMinY();
    final calculatedMaxY = maxY ?? _calculateMaxY();

    return SizedBox(
      width: width,
      height: height,
      child: Row(
        children: [
          Expanded(
            child: LineChart(
              LineChartData(
                gridData: const FlGridData(show: false),
                titlesData: const FlTitlesData(show: false),
                borderData: FlBorderData(show: false),
                minY: calculatedMinY,
                maxY: calculatedMaxY,
                lineTouchData: const LineTouchData(enabled: false),
                lineBarsData: [
                  LineChartBarData(
                    spots: spots,
                    isCurved: true,
                    curveSmoothness: curveSmoothness,
                    color: effectiveColor,
                    barWidth: lineWidth,
                    isStrokeCapRound: true,
                    dotData: FlDotData(
                      show: showDots,
                      getDotPainter: (spot, percent, bar, index) {
                        // Only show dot for last point
                        if (index == spots.length - 1) {
                          return FlDotCirclePainter(
                            radius: 3,
                            color: effectiveColor,
                            strokeWidth: 1.5,
                            strokeColor: Colors.white,
                          );
                        }
                        return FlDotCirclePainter(
                          radius: 0,
                          color: Colors.transparent,
                          strokeWidth: 0,
                          strokeColor: Colors.transparent,
                        );
                      },
                    ),
                    belowBarData: showGradient
                        ? BarAreaData(
                            show: true,
                            gradient: LinearGradient(
                              begin: Alignment.topCenter,
                              end: Alignment.bottomCenter,
                              colors: [
                                effectiveColor.withOpacity(0.3),
                                effectiveColor.withOpacity(0.0),
                              ],
                            ),
                          )
                        : BarAreaData(show: false),
                  ),
                ],
              ),
              duration: animate
                  ? const Duration(milliseconds: 300)
                  : Duration.zero,
            ),
          ),
          if (showCurrentValue && data.isNotEmpty) ...[
            const SizedBox(width: 8),
            Text(
              data.last.toStringAsFixed(1),
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
                color: effectiveColor,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return SizedBox(
      width: width,
      height: height,
      child: Center(
        child: Text(
          '—',
          style: TextStyle(
            color: Colors.grey[400],
            fontSize: 16,
          ),
        ),
      ),
    );
  }

  List<FlSpot> _createSpots() {
    return List.generate(
      data.length,
      (index) => FlSpot(index.toDouble(), data[index]),
    );
  }

  double _calculateMinY() {
    if (data.isEmpty) return 0;
    final min = data.reduce((a, b) => a < b ? a : b);
    final range = _calculateMaxY() - min;
    return min - (range * 0.1); // Add 10% padding
  }

  double _calculateMaxY() {
    if (data.isEmpty) return 100;
    final max = data.reduce((a, b) => a > b ? a : b);
    final min = data.reduce((a, b) => a < b ? a : b);
    final range = max - min;
    return max + (range * 0.1); // Add 10% padding
  }
}

/// Larger chart with more details for detailed views
class DetailedChart extends StatelessWidget {
  final List<double> data;
  final List<String>? labels;
  final Color? lineColor;
  final String? title;
  final String? unit;
  final double height;
  final bool showGrid;
  final bool showLabels;
  final double? warningThreshold;
  final double? criticalThreshold;

  const DetailedChart({
    super.key,
    required this.data,
    this.labels,
    this.lineColor,
    this.title,
    this.unit,
    this.height = 200,
    this.showGrid = true,
    this.showLabels = true,
    this.warningThreshold,
    this.criticalThreshold,
  });

  @override
  Widget build(BuildContext context) {
    if (data.isEmpty) {
      return _buildEmptyState();
    }

    final effectiveColor = lineColor ?? Theme.of(context).primaryColor;
    final spots = _createSpots();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (title != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Row(
              children: [
                Text(
                  title!,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                if (unit != null) ...[
                  const SizedBox(width: 8),
                  Text(
                    '($unit)',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              ],
            ),
          ),
        SizedBox(
          height: height,
          child: LineChart(
            LineChartData(
              gridData: FlGridData(
                show: showGrid,
                drawVerticalLine: false,
                horizontalInterval: _calculateInterval(),
                getDrawingHorizontalLine: (value) {
                  return FlLine(
                    color: Colors.grey[200],
                    strokeWidth: 1,
                  );
                },
              ),
              titlesData: FlTitlesData(
                show: showLabels,
                rightTitles: const AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                topTitles: const AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: labels != null,
                    getTitlesWidget: (value, meta) {
                      final index = value.toInt();
                      if (labels != null &&
                          index >= 0 &&
                          index < labels!.length) {
                        return Padding(
                          padding: const EdgeInsets.only(top: 8),
                          child: Text(
                            labels![index],
                            style: TextStyle(
                              fontSize: 10,
                              color: Colors.grey[600],
                            ),
                          ),
                        );
                      }
                      return const SizedBox();
                    },
                  ),
                ),
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 40,
                    getTitlesWidget: (value, meta) {
                      return Text(
                        value.toStringAsFixed(0),
                        style: TextStyle(
                          fontSize: 10,
                          color: Colors.grey[600],
                        ),
                      );
                    },
                  ),
                ),
              ),
              borderData: FlBorderData(show: false),
              lineTouchData: LineTouchData(
                touchTooltipData: LineTouchTooltipData(
                  getTooltipItems: (touchedSpots) {
                    return touchedSpots.map((spot) {
                      return LineTooltipItem(
                        '${spot.y.toStringAsFixed(1)}${unit ?? ''}',
                        TextStyle(
                          color: effectiveColor,
                          fontWeight: FontWeight.bold,
                        ),
                      );
                    }).toList();
                  },
                ),
              ),
              extraLinesData: ExtraLinesData(
                horizontalLines: [
                  if (warningThreshold != null)
                    HorizontalLine(
                      y: warningThreshold!,
                      color: AppColors.riskModerate.withOpacity(0.5),
                      strokeWidth: 1,
                      dashArray: [5, 5],
                    ),
                  if (criticalThreshold != null)
                    HorizontalLine(
                      y: criticalThreshold!,
                      color: AppColors.riskCritical.withOpacity(0.5),
                      strokeWidth: 1,
                      dashArray: [5, 5],
                    ),
                ],
              ),
              lineBarsData: [
                LineChartBarData(
                  spots: spots,
                  isCurved: true,
                  curveSmoothness: 0.3,
                  color: effectiveColor,
                  barWidth: 2.5,
                  isStrokeCapRound: true,
                  dotData: FlDotData(
                    show: true,
                    getDotPainter: (spot, percent, bar, index) {
                      return FlDotCirclePainter(
                        radius: 3,
                        color: effectiveColor,
                        strokeWidth: 1.5,
                        strokeColor: Colors.white,
                      );
                    },
                  ),
                  belowBarData: BarAreaData(
                    show: true,
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [
                        effectiveColor.withOpacity(0.2),
                        effectiveColor.withOpacity(0.0),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildEmptyState() {
    return SizedBox(
      height: height,
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.show_chart,
              size: 48,
              color: Colors.grey[300],
            ),
            const SizedBox(height: 8),
            Text(
              'No data available',
              style: TextStyle(
                color: Colors.grey[500],
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<FlSpot> _createSpots() {
    return List.generate(
      data.length,
      (index) => FlSpot(index.toDouble(), data[index]),
    );
  }

  double _calculateInterval() {
    if (data.isEmpty) return 10;
    final max = data.reduce((a, b) => a > b ? a : b);
    final min = data.reduce((a, b) => a < b ? a : b);
    final range = max - min;
    if (range <= 0) return 1;
    return (range / 5).ceilToDouble();
  }
}
