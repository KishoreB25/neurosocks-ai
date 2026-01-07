import 'package:flutter/material.dart';

/// Card widget for displaying a summary statistic
class StatCard extends StatelessWidget {
  /// Title/label for the stat
  final String title;

  /// Main value to display
  final String value;

  /// Optional subtitle or unit
  final String? subtitle;

  /// Icon to display
  final IconData icon;

  /// Primary color for the card
  final Color color;

  /// Optional trend indicator (-1 = down, 0 = neutral, 1 = up)
  final int? trend;

  /// Optional trend value text
  final String? trendValue;

  /// Whether trend up is positive (e.g., steps) or negative (e.g., risk)
  final bool trendUpIsGood;

  /// Card size variant
  final StatCardSize size;

  /// On tap callback
  final VoidCallback? onTap;

  /// Optional background gradient
  final bool useGradient;

  const StatCard({
    super.key,
    required this.title,
    required this.value,
    this.subtitle,
    required this.icon,
    required this.color,
    this.trend,
    this.trendValue,
    this.trendUpIsGood = true,
    this.size = StatCardSize.medium,
    this.onTap,
    this.useGradient = false,
  });

  /// Create a stat card for step count
  factory StatCard.steps({
    required int steps,
    int? goal,
    int? trend,
    VoidCallback? onTap,
  }) {
    final percentage = goal != null ? (steps / goal * 100).round() : null;
    return StatCard(
      title: 'Steps Today',
      value: _formatNumber(steps),
      subtitle: goal != null ? '$percentage% of goal' : null,
      icon: Icons.directions_walk,
      color: Colors.blue,
      trend: trend,
      trendUpIsGood: true,
      onTap: onTap,
    );
  }

  /// Create a stat card for average risk
  factory StatCard.avgRisk({
    required double avgRisk,
    int? trend,
    String? trendValue,
    VoidCallback? onTap,
  }) {
    return StatCard(
      title: 'Avg Risk Score',
      value: '${avgRisk.round()}%',
      subtitle: 'Last 7 days',
      icon: Icons.analytics,
      color: _getRiskColor(avgRisk.round()),
      trend: trend,
      trendValue: trendValue,
      trendUpIsGood: false, // Lower risk is better
      onTap: onTap,
    );
  }

  /// Create a stat card for alerts count
  factory StatCard.alerts({
    required int count,
    int? criticalCount,
    VoidCallback? onTap,
  }) {
    return StatCard(
      title: 'Active Alerts',
      value: count.toString(),
      subtitle: criticalCount != null && criticalCount > 0
          ? '$criticalCount critical'
          : 'No critical alerts',
      icon: Icons.notifications_active,
      color: count > 0 ? Colors.orange : Colors.green,
      onTap: onTap,
    );
  }

  /// Create a stat card for streak
  factory StatCard.streak({
    required int days,
    VoidCallback? onTap,
  }) {
    return StatCard(
      title: 'Wearing Streak',
      value: '$days',
      subtitle: days == 1 ? 'day' : 'days',
      icon: Icons.local_fire_department,
      color: days >= 7 ? Colors.orange : Colors.grey,
      useGradient: days >= 7,
      onTap: onTap,
    );
  }

  /// Create a stat card for connection time
  factory StatCard.connectionTime({
    required Duration duration,
    VoidCallback? onTap,
  }) {
    final hours = duration.inHours;
    final minutes = duration.inMinutes % 60;
    return StatCard(
      title: 'Connected Time',
      value: hours > 0 ? '${hours}h ${minutes}m' : '${minutes}m',
      subtitle: 'Today',
      icon: Icons.bluetooth_connected,
      color: Colors.blue,
      onTap: onTap,
    );
  }

  /// Create a stat card for average temperature
  factory StatCard.avgTemperature({
    required double avgTemp,
    double? maxTemp,
    VoidCallback? onTap,
  }) {
    return StatCard(
      title: 'Avg Temperature',
      value: '${avgTemp.toStringAsFixed(1)}°C',
      subtitle: maxTemp != null ? 'Max: ${maxTemp.toStringAsFixed(1)}°C' : null,
      icon: Icons.thermostat,
      color: Colors.deepOrange,
      onTap: onTap,
    );
  }

  @override
  Widget build(BuildContext context) {
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
            gradient: useGradient
                ? LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      color,
                      color.withOpacity(0.7),
                    ],
                  )
                : null,
            color: useGradient ? null : Theme.of(context).cardColor,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: (useGradient ? color : Colors.black).withOpacity(0.1),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Header with icon
              Row(
                children: [
                  Container(
                    padding: EdgeInsets.all(dimensions.iconPadding),
                    decoration: BoxDecoration(
                      color: useGradient
                          ? Colors.white.withOpacity(0.2)
                          : color.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Icon(
                      icon,
                      size: dimensions.iconSize,
                      color: useGradient ? Colors.white : color,
                    ),
                  ),
                  const Spacer(),
                  if (trend != null) _buildTrendIndicator(dimensions),
                ],
              ),

              SizedBox(height: dimensions.spacing),

              // Value
              Text(
                value,
                style: TextStyle(
                  fontSize: dimensions.valueSize,
                  fontWeight: FontWeight.bold,
                  color: useGradient ? Colors.white : null,
                ),
              ),

              SizedBox(height: dimensions.spacing * 0.5),

              // Title
              Text(
                title,
                style: TextStyle(
                  fontSize: dimensions.titleSize,
                  fontWeight: FontWeight.w500,
                  color: useGradient
                      ? Colors.white.withOpacity(0.9)
                      : Colors.grey[600],
                ),
              ),

              // Subtitle
              if (subtitle != null) ...[
                SizedBox(height: dimensions.spacing * 0.3),
                Text(
                  subtitle!,
                  style: TextStyle(
                    fontSize: dimensions.subtitleSize,
                    color: useGradient
                        ? Colors.white.withOpacity(0.7)
                        : Colors.grey[500],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTrendIndicator(_StatCardDimensions dimensions) {
    final isUp = trend! > 0;
    final isNeutral = trend == 0;
    final isPositive = isNeutral || (isUp == trendUpIsGood);

    final trendColor = useGradient
        ? Colors.white
        : isPositive
            ? Colors.green
            : Colors.red;

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          isNeutral
              ? Icons.trending_flat
              : isUp
                  ? Icons.trending_up
                  : Icons.trending_down,
          size: dimensions.iconSize * 0.7,
          color: trendColor,
        ),
        if (trendValue != null) ...[
          const SizedBox(width: 4),
          Text(
            trendValue!,
            style: TextStyle(
              fontSize: dimensions.subtitleSize,
              fontWeight: FontWeight.w500,
              color: trendColor,
            ),
          ),
        ],
      ],
    );
  }

  _StatCardDimensions _getDimensions() {
    switch (size) {
      case StatCardSize.small:
        return _StatCardDimensions(
          width: 140,
          padding: 14,
          iconSize: 20,
          iconPadding: 8,
          valueSize: 22,
          titleSize: 11,
          subtitleSize: 10,
          spacing: 10,
        );
      case StatCardSize.medium:
        return _StatCardDimensions(
          width: 170,
          padding: 18,
          iconSize: 24,
          iconPadding: 10,
          valueSize: 28,
          titleSize: 13,
          subtitleSize: 11,
          spacing: 12,
        );
      case StatCardSize.large:
        return _StatCardDimensions(
          width: 200,
          padding: 22,
          iconSize: 28,
          iconPadding: 12,
          valueSize: 34,
          titleSize: 15,
          subtitleSize: 12,
          spacing: 14,
        );
    }
  }

  static String _formatNumber(int number) {
    if (number >= 1000000) {
      return '${(number / 1000000).toStringAsFixed(1)}M';
    } else if (number >= 1000) {
      return '${(number / 1000).toStringAsFixed(1)}K';
    }
    return number.toString();
  }

  static Color _getRiskColor(int risk) {
    if (risk >= 80) return Colors.red;
    if (risk >= 60) return Colors.orange;
    if (risk >= 40) return Colors.amber;
    return Colors.green;
  }
}

/// Size variants for stat cards
enum StatCardSize { small, medium, large }

/// Internal dimensions helper
class _StatCardDimensions {
  final double width;
  final double padding;
  final double iconSize;
  final double iconPadding;
  final double valueSize;
  final double titleSize;
  final double subtitleSize;
  final double spacing;

  const _StatCardDimensions({
    required this.width,
    required this.padding,
    required this.iconSize,
    required this.iconPadding,
    required this.valueSize,
    required this.titleSize,
    required this.subtitleSize,
    required this.spacing,
  });
}

/// Row of stat cards with automatic spacing
class StatCardRow extends StatelessWidget {
  final List<StatCard> cards;
  final double spacing;
  final bool scrollable;

  const StatCardRow({
    super.key,
    required this.cards,
    this.spacing = 12,
    this.scrollable = true,
  });

  @override
  Widget build(BuildContext context) {
    if (scrollable) {
      return SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        padding: EdgeInsets.symmetric(horizontal: spacing),
        child: Row(
          children: cards
              .map((card) => Padding(
                    padding: EdgeInsets.only(right: spacing),
                    child: card,
                  ))
              .toList(),
        ),
      );
    }

    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: cards,
    );
  }
}
