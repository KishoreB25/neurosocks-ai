import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../data/models/risk_score.dart';

/// Circular gauge widget displaying risk score (0-100)
/// Shows animated arc with gradient colors based on risk level
class RiskGauge extends StatefulWidget {
  /// Risk score value (0-100)
  final int score;

  /// Risk level for color and label
  final RiskLevel level;

  /// Size of the gauge
  final double size;

  /// Whether to animate on value change
  final bool animate;

  /// Animation duration
  final Duration animationDuration;

  /// Show the score number in center
  final bool showScore;

  /// Show the risk level label
  final bool showLabel;

  /// Custom center widget (overrides score display)
  final Widget? centerWidget;

  /// Stroke width of the arc
  final double strokeWidth;

  const RiskGauge({
    super.key,
    required this.score,
    required this.level,
    this.size = 200,
    this.animate = true,
    this.animationDuration = const Duration(milliseconds: 1000),
    this.showScore = true,
    this.showLabel = true,
    this.centerWidget,
    this.strokeWidth = 12,
  });

  /// Create from RiskScore model
  factory RiskGauge.fromRiskScore(
    RiskScore riskScore, {
    double size = 200,
    bool animate = true,
    bool showScore = true,
    bool showLabel = true,
  }) {
    return RiskGauge(
      score: riskScore.overallScore,
      level: riskScore.riskLevel,
      size: size,
      animate: animate,
      showScore: showScore,
      showLabel: showLabel,
    );
  }

  @override
  State<RiskGauge> createState() => _RiskGaugeState();
}

class _RiskGaugeState extends State<RiskGauge>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  double _previousScore = 0;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.animationDuration,
    );

    _setupAnimation();

    if (widget.animate) {
      _controller.forward();
    } else {
      _controller.value = 1.0;
    }
  }

  void _setupAnimation() {
    _animation = Tween<double>(
      begin: _previousScore,
      end: widget.score.toDouble(),
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    ));
  }

  @override
  void didUpdateWidget(RiskGauge oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.score != widget.score) {
      _previousScore = oldWidget.score.toDouble();
      _setupAnimation();
      _controller.forward(from: 0);
    }
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
        final currentScore = _animation.value;
        return SizedBox(
          width: widget.size,
          height: widget.size,
          child: Stack(
            alignment: Alignment.center,
            children: [
              // Background arc
              CustomPaint(
                size: Size(widget.size, widget.size),
                painter: _GaugeBackgroundPainter(
                  strokeWidth: widget.strokeWidth,
                ),
              ),
              // Foreground arc (colored)
              CustomPaint(
                size: Size(widget.size, widget.size),
                painter: _GaugeForegroundPainter(
                  score: currentScore,
                  strokeWidth: widget.strokeWidth,
                ),
              ),
              // Center content
              widget.centerWidget ?? _buildCenterContent(currentScore),
            ],
          ),
        );
      },
    );
  }

  Widget _buildCenterContent(double currentScore) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (widget.showScore) ...[
          Text(
            currentScore.round().toString(),
            style: TextStyle(
              fontSize: widget.size * 0.22,
              fontWeight: FontWeight.bold,
              color: widget.level.color,
            ),
          ),
          Text(
            '%',
            style: TextStyle(
              fontSize: widget.size * 0.08,
              fontWeight: FontWeight.w500,
              color: Colors.grey[600],
            ),
          ),
        ],
        if (widget.showLabel) ...[
          SizedBox(height: widget.size * 0.02),
          Container(
            padding: EdgeInsets.symmetric(
              horizontal: widget.size * 0.06,
              vertical: widget.size * 0.02,
            ),
            decoration: BoxDecoration(
              color: widget.level.backgroundColor,
              borderRadius: BorderRadius.circular(widget.size * 0.04),
            ),
            child: Text(
              widget.level.shortName,
              style: TextStyle(
                fontSize: widget.size * 0.07,
                fontWeight: FontWeight.w600,
                color: widget.level.color,
              ),
            ),
          ),
        ],
      ],
    );
  }
}

/// Background arc painter (grey track)
class _GaugeBackgroundPainter extends CustomPainter {
  final double strokeWidth;

  _GaugeBackgroundPainter({required this.strokeWidth});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width - strokeWidth) / 2;

    final paint = Paint()
      ..color = Colors.grey[200]!
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    // Draw arc from 135° to 405° (270° sweep)
    const startAngle = 135 * math.pi / 180;
    const sweepAngle = 270 * math.pi / 180;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle,
      false,
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

/// Foreground arc painter (gradient colored based on score)
class _GaugeForegroundPainter extends CustomPainter {
  final double score;
  final double strokeWidth;

  _GaugeForegroundPainter({
    required this.score,
    required this.strokeWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (score <= 0) return;

    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width - strokeWidth) / 2;

    // Create gradient shader
    final gradient = SweepGradient(
      startAngle: 135 * math.pi / 180,
      endAngle: 405 * math.pi / 180,
      colors: const [
        AppColors.riskLow,
        AppColors.riskModerate,
        AppColors.riskHigh,
        AppColors.riskCritical,
      ],
      stops: const [0.0, 0.33, 0.66, 1.0],
    );

    final paint = Paint()
      ..shader = gradient.createShader(
        Rect.fromCircle(center: center, radius: radius),
      )
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    // Calculate sweep angle based on score (0-100)
    const startAngle = 135 * math.pi / 180;
    final sweepAngle = (score / 100) * 270 * math.pi / 180;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle,
      false,
      paint,
    );

    // Draw end cap dot for better visual
    if (score > 0) {
      final endAngle = startAngle + sweepAngle;
      final endX = center.dx + radius * math.cos(endAngle);
      final endY = center.dy + radius * math.sin(endAngle);

      final dotPaint = Paint()
        ..color = AppColors.getRiskColor(score.round())
        ..style = PaintingStyle.fill;

      canvas.drawCircle(
        Offset(endX, endY),
        strokeWidth / 2,
        dotPaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _GaugeForegroundPainter oldDelegate) {
    return oldDelegate.score != score;
  }
}

/// Compact version of risk gauge for smaller spaces
class CompactRiskGauge extends StatelessWidget {
  final int score;
  final RiskLevel level;
  final double size;

  const CompactRiskGauge({
    super.key,
    required this.score,
    required this.level,
    this.size = 60,
  });

  @override
  Widget build(BuildContext context) {
    return RiskGauge(
      score: score,
      level: level,
      size: size,
      strokeWidth: 6,
      showLabel: false,
      centerWidget: Text(
        '$score',
        style: TextStyle(
          fontSize: size * 0.28,
          fontWeight: FontWeight.bold,
          color: level.color,
        ),
      ),
    );
  }
}
