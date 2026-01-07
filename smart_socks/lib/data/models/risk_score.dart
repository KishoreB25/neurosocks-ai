import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../core/constants/sensor_constants.dart';

/// Represents the calculated risk assessment from sensor data
class RiskScore extends Equatable {
  /// Overall risk score (0-100)
  final int overallScore;

  /// Risk level classification
  final RiskLevel riskLevel;

  /// Temperature component risk score (0-100)
  final int temperatureRisk;

  /// Pressure component risk score (0-100)
  final int pressureRisk;

  /// Circulation (SpO2 + HR) component risk score (0-100)
  final int circulationRisk;

  /// Gait/stability component risk score (0-100)
  final int gaitRisk;

  /// List of factors contributing to the risk score
  final List<String> factors;

  /// List of recommendations based on the risk assessment
  final List<String> recommendations;

  /// Timestamp when this score was calculated
  final DateTime timestamp;

  /// Optional: ID for storing in database
  final String? id;

  const RiskScore({
    required this.overallScore,
    required this.riskLevel,
    required this.temperatureRisk,
    required this.pressureRisk,
    required this.circulationRisk,
    required this.gaitRisk,
    required this.factors,
    required this.recommendations,
    required this.timestamp,
    this.id,
  });

  /// Create a RiskScore from component scores
  factory RiskScore.fromComponents({
    required int temperatureRisk,
    required int pressureRisk,
    required int circulationRisk,
    required int gaitRisk,
    required List<String> factors,
    DateTime? timestamp,
    String? id,
  }) {
    // Calculate weighted overall score
    final overall = (
      (temperatureRisk * SensorConstants.weightTemperature) +
      (pressureRisk * SensorConstants.weightPressure) +
      (circulationRisk * SensorConstants.weightCirculation) +
      (gaitRisk * SensorConstants.weightGait)
    ).round().clamp(0, 100);

    // Determine risk level
    final level = RiskLevel.fromScore(overall);

    // Generate recommendations based on factors
    final recommendations = _generateRecommendations(
      level: level,
      factors: factors,
      temperatureRisk: temperatureRisk,
      pressureRisk: pressureRisk,
      circulationRisk: circulationRisk,
      gaitRisk: gaitRisk,
    );

    return RiskScore(
      overallScore: overall,
      riskLevel: level,
      temperatureRisk: temperatureRisk,
      pressureRisk: pressureRisk,
      circulationRisk: circulationRisk,
      gaitRisk: gaitRisk,
      factors: factors,
      recommendations: recommendations,
      timestamp: timestamp ?? DateTime.now(),
      id: id,
    );
  }

  /// Create an empty/default risk score
  factory RiskScore.empty() {
    return RiskScore(
      overallScore: 0,
      riskLevel: RiskLevel.low,
      temperatureRisk: 0,
      pressureRisk: 0,
      circulationRisk: 0,
      gaitRisk: 0,
      factors: [],
      recommendations: ['Start monitoring to get risk assessment'],
      timestamp: DateTime.now(),
    );
  }

  /// Generate recommendations based on risk factors
  static List<String> _generateRecommendations({
    required RiskLevel level,
    required List<String> factors,
    required int temperatureRisk,
    required int pressureRisk,
    required int circulationRisk,
    required int gaitRisk,
  }) {
    final recommendations = <String>[];

    // Level-based recommendations
    switch (level) {
      case RiskLevel.low:
        recommendations.add('Continue regular monitoring');
        recommendations.add('Maintain good foot hygiene');
        break;
      case RiskLevel.moderate:
        recommendations.add('Increase monitoring frequency');
        recommendations.add('Check your footwear for proper fit');
        break;
      case RiskLevel.high:
        recommendations.add('Rest your feet and elevate if possible');
        recommendations.add('Consider consulting a healthcare provider');
        break;
      case RiskLevel.critical:
        recommendations.add('Seek medical attention promptly');
        recommendations.add('Avoid putting weight on affected areas');
        break;
    }

    // Factor-specific recommendations
    if (temperatureRisk > 50) {
      recommendations.add('Apply cool compress to hot spots');
      recommendations.add('Check for signs of infection or inflammation');
    }

    if (pressureRisk > 50) {
      recommendations.add('Redistribute weight or change position');
      recommendations.add('Consider using cushioned insoles');
    }

    if (circulationRisk > 50) {
      recommendations.add('Move around to improve blood flow');
      recommendations.add('Avoid tight socks or footwear');
    }

    if (gaitRisk > 50) {
      recommendations.add('Walk slowly and carefully');
      recommendations.add('Use assistive devices if needed');
    }

    return recommendations;
  }

  // ============== Getters ==============

  /// Get the primary color for this risk level
  Color get color => riskLevel.color;

  /// Get the background color for this risk level
  Color get backgroundColor => riskLevel.backgroundColor;

  /// Get icon for this risk level
  IconData get icon => riskLevel.icon;

  /// Get display message for this risk level
  String get message => riskLevel.message;

  /// Check if this is a high-risk score
  bool get isHighRisk => 
      riskLevel == RiskLevel.high || riskLevel == RiskLevel.critical;

  /// Check if any action is needed
  bool get needsAction => riskLevel != RiskLevel.low;

  /// Get the highest contributing factor
  String get primaryFactor {
    if (factors.isEmpty) return 'No issues detected';
    return factors.first;
  }

  /// Get score as percentage string
  String get scorePercentage => '$overallScore%';

  /// Get component with highest risk
  String get highestRiskComponent {
    final components = {
      'Temperature': temperatureRisk,
      'Pressure': pressureRisk,
      'Circulation': circulationRisk,
      'Gait': gaitRisk,
    };
    
    return components.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
  }

  // ============== Serialization ==============

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'overallScore': overallScore,
      'riskLevel': riskLevel.name,
      'temperatureRisk': temperatureRisk,
      'pressureRisk': pressureRisk,
      'circulationRisk': circulationRisk,
      'gaitRisk': gaitRisk,
      'factors': factors,
      'recommendations': recommendations,
      'timestamp': timestamp.millisecondsSinceEpoch,
    };
  }

  factory RiskScore.fromJson(Map<String, dynamic> json) {
    return RiskScore(
      id: json['id'],
      overallScore: json['overallScore'] ?? 0,
      riskLevel: RiskLevel.fromString(json['riskLevel']),
      temperatureRisk: json['temperatureRisk'] ?? 0,
      pressureRisk: json['pressureRisk'] ?? 0,
      circulationRisk: json['circulationRisk'] ?? 0,
      gaitRisk: json['gaitRisk'] ?? 0,
      factors: List<String>.from(json['factors'] ?? []),
      recommendations: List<String>.from(json['recommendations'] ?? []),
      timestamp: DateTime.fromMillisecondsSinceEpoch(
        json['timestamp'] ?? DateTime.now().millisecondsSinceEpoch,
      ),
    );
  }

  /// Create a copy with modified fields
  RiskScore copyWith({
    int? overallScore,
    RiskLevel? riskLevel,
    int? temperatureRisk,
    int? pressureRisk,
    int? circulationRisk,
    int? gaitRisk,
    List<String>? factors,
    List<String>? recommendations,
    DateTime? timestamp,
    String? id,
  }) {
    return RiskScore(
      overallScore: overallScore ?? this.overallScore,
      riskLevel: riskLevel ?? this.riskLevel,
      temperatureRisk: temperatureRisk ?? this.temperatureRisk,
      pressureRisk: pressureRisk ?? this.pressureRisk,
      circulationRisk: circulationRisk ?? this.circulationRisk,
      gaitRisk: gaitRisk ?? this.gaitRisk,
      factors: factors ?? this.factors,
      recommendations: recommendations ?? this.recommendations,
      timestamp: timestamp ?? this.timestamp,
      id: id ?? this.id,
    );
  }

  @override
  List<Object?> get props => [
        overallScore,
        riskLevel,
        temperatureRisk,
        pressureRisk,
        circulationRisk,
        gaitRisk,
        factors,
        recommendations,
        timestamp,
        id,
      ];

  @override
  String toString() => 
      'RiskScore($overallScore - ${riskLevel.displayName}, '
      'temp: $temperatureRisk, pressure: $pressureRisk, '
      'circ: $circulationRisk, gait: $gaitRisk)';
}

/// Risk level classification
enum RiskLevel {
  low,
  moderate,
  high,
  critical;

  /// Create from score (0-100)
  static RiskLevel fromScore(int score) {
    if (score <= SensorConstants.riskLowMax) return RiskLevel.low;
    if (score <= SensorConstants.riskModerateMax) return RiskLevel.moderate;
    if (score <= SensorConstants.riskHighMax) return RiskLevel.high;
    return RiskLevel.critical;
  }

  /// Create from string
  static RiskLevel fromString(String? value) {
    switch (value?.toLowerCase()) {
      case 'low':
        return RiskLevel.low;
      case 'moderate':
        return RiskLevel.moderate;
      case 'high':
        return RiskLevel.high;
      case 'critical':
        return RiskLevel.critical;
      default:
        return RiskLevel.low;
    }
  }

  /// Display name for UI
  String get displayName {
    switch (this) {
      case RiskLevel.low:
        return 'Low Risk';
      case RiskLevel.moderate:
        return 'Moderate Risk';
      case RiskLevel.high:
        return 'High Risk';
      case RiskLevel.critical:
        return 'Critical Risk';
    }
  }

  /// Short display name
  String get shortName {
    switch (this) {
      case RiskLevel.low:
        return 'Low';
      case RiskLevel.moderate:
        return 'Moderate';
      case RiskLevel.high:
        return 'High';
      case RiskLevel.critical:
        return 'Critical';
    }
  }

  /// Get color for this risk level
  Color get color {
    switch (this) {
      case RiskLevel.low:
        return AppColors.riskLow;
      case RiskLevel.moderate:
        return AppColors.riskModerate;
      case RiskLevel.high:
        return AppColors.riskHigh;
      case RiskLevel.critical:
        return AppColors.riskCritical;
    }
  }

  /// Get background color
  Color get backgroundColor {
    switch (this) {
      case RiskLevel.low:
        return AppColors.riskLowBg;
      case RiskLevel.moderate:
        return AppColors.riskModerateBg;
      case RiskLevel.high:
        return AppColors.riskHighBg;
      case RiskLevel.critical:
        return AppColors.riskCriticalBg;
    }
  }

  /// Get icon for this risk level
  IconData get icon {
    switch (this) {
      case RiskLevel.low:
        return Icons.check_circle;
      case RiskLevel.moderate:
        return Icons.info;
      case RiskLevel.high:
        return Icons.warning;
      case RiskLevel.critical:
        return Icons.error;
    }
  }

  /// Get message for this risk level
  String get message {
    switch (this) {
      case RiskLevel.low:
        return 'Looking good! Continue monitoring.';
      case RiskLevel.moderate:
        return 'Some attention needed. Check recommendations.';
      case RiskLevel.high:
        return 'Take action to reduce risk factors.';
      case RiskLevel.critical:
        return 'Urgent! Seek medical advice promptly.';
    }
  }

  /// Get score range for this level
  String get scoreRange {
    switch (this) {
      case RiskLevel.low:
        return '0-${SensorConstants.riskLowMax}';
      case RiskLevel.moderate:
        return '${SensorConstants.riskLowMax + 1}-${SensorConstants.riskModerateMax}';
      case RiskLevel.high:
        return '${SensorConstants.riskModerateMax + 1}-${SensorConstants.riskHighMax}';
      case RiskLevel.critical:
        return '${SensorConstants.riskHighMax + 1}-100';
    }
  }
}

/// Daily summary of risk scores
class DailyRiskSummary extends Equatable {
  /// Date for this summary
  final DateTime date;

  /// Average risk score for the day
  final int averageScore;

  /// Highest risk score recorded
  final int highestScore;

  /// Lowest risk score recorded
  final int lowestScore;

  /// Number of readings taken
  final int readingCount;

  /// Number of high/critical alerts
  final int alertCount;

  /// Most common risk level during the day
  final RiskLevel dominantRiskLevel;

  /// Key factors that contributed to risk
  final List<String> keyFactors;

  const DailyRiskSummary({
    required this.date,
    required this.averageScore,
    required this.highestScore,
    required this.lowestScore,
    required this.readingCount,
    required this.alertCount,
    required this.dominantRiskLevel,
    required this.keyFactors,
  });

  factory DailyRiskSummary.fromScores(DateTime date, List<RiskScore> scores) {
    if (scores.isEmpty) {
      return DailyRiskSummary(
        date: date,
        averageScore: 0,
        highestScore: 0,
        lowestScore: 0,
        readingCount: 0,
        alertCount: 0,
        dominantRiskLevel: RiskLevel.low,
        keyFactors: [],
      );
    }

    final overallScores = scores.map((s) => s.overallScore).toList();
    final avg = (overallScores.reduce((a, b) => a + b) / scores.length).round();
    final highest = overallScores.reduce((a, b) => a > b ? a : b);
    final lowest = overallScores.reduce((a, b) => a < b ? a : b);

    // Count alerts
    final alerts = scores.where((s) => s.isHighRisk).length;

    // Find dominant risk level
    final levelCounts = <RiskLevel, int>{};
    for (final score in scores) {
      levelCounts[score.riskLevel] = (levelCounts[score.riskLevel] ?? 0) + 1;
    }
    final dominant = levelCounts.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;

    // Collect unique factors
    final allFactors = <String>{};
    for (final score in scores) {
      allFactors.addAll(score.factors);
    }

    return DailyRiskSummary(
      date: date,
      averageScore: avg,
      highestScore: highest,
      lowestScore: lowest,
      readingCount: scores.length,
      alertCount: alerts,
      dominantRiskLevel: dominant,
      keyFactors: allFactors.take(5).toList(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'date': date.millisecondsSinceEpoch,
      'averageScore': averageScore,
      'highestScore': highestScore,
      'lowestScore': lowestScore,
      'readingCount': readingCount,
      'alertCount': alertCount,
      'dominantRiskLevel': dominantRiskLevel.name,
      'keyFactors': keyFactors,
    };
  }

  factory DailyRiskSummary.fromJson(Map<String, dynamic> json) {
    return DailyRiskSummary(
      date: DateTime.fromMillisecondsSinceEpoch(json['date']),
      averageScore: json['averageScore'] ?? 0,
      highestScore: json['highestScore'] ?? 0,
      lowestScore: json['lowestScore'] ?? 0,
      readingCount: json['readingCount'] ?? 0,
      alertCount: json['alertCount'] ?? 0,
      dominantRiskLevel: RiskLevel.fromString(json['dominantRiskLevel']),
      keyFactors: List<String>.from(json['keyFactors'] ?? []),
    );
  }

  @override
  List<Object?> get props => [
        date,
        averageScore,
        highestScore,
        lowestScore,
        readingCount,
        alertCount,
        dominantRiskLevel,
        keyFactors,
      ];
}
