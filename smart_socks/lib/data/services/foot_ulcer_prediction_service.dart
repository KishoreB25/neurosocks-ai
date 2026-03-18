// Foot Ulcer Prediction Model
// Analyzes sensor data to predict foot ulcer risk based on pressure, temperature, and activity patterns

import '../models/sensor_reading.dart';

enum UlcerRiskLevel {
  low,      // Green - < 30%
  moderate, // Yellow - 30-60%
  high,     // Orange - 60-80%
  critical  // Red - > 80%
}

class FootUlcerPrediction {
  final double riskScore; // 0-100
  final UlcerRiskLevel level;
  final List<String> riskFactors; // List of detected risk factors
  final DateTime timestamp;
  final String affectedZone; // Heel, Ball, Arch, Toe
  final String recommendation;

  FootUlcerPrediction({
    required this.riskScore,
    required this.level,
    required this.riskFactors,
    required this.timestamp,
    required this.affectedZone,
    required this.recommendation,
  });

  bool get isCritical => level == UlcerRiskLevel.critical;
  bool get isHigh => level == UlcerRiskLevel.high;

  Map<String, dynamic> toJson() => {
    'riskScore': riskScore,
    'level': level.toString(),
    'riskFactors': riskFactors,
    'timestamp': timestamp.toIso8601String(),
    'affectedZone': affectedZone,
    'recommendation': recommendation,
  };
}

/// Foot Ulcer Prediction Service
/// Uses ML model (simplified here) to predict foot ulcer risk
class FootUlcerPredictionService {
  // Risk thresholds
  static const double tempExcessiveHigh = 35.0; // Above normal foot temp
  static const double tempModerateHigh = 33.5;
  static const double pressureCritical = 60.0; // kPa
  static const double pressureHigh = 50.0;     // kPa
  static const double pressureModerate = 40.0; // kPa

  // Temperature difference thresholds (between zones)
  static const double tempZoneDiffCritical = 2.5;
  static const double tempZoneDiffHigh = 1.8;

  /// Predict foot ulcer risk from sensor reading
  static FootUlcerPrediction predictRisk(
    SensorReading reading, {
    List<SensorReading>? historicalReadings,
  }) {
    double riskScore = 0;
    final riskFactors = <String>[];
    String affectedZone = '';

    // Analyze pressure patterns
    final (pressureRisk, pressureZone, pressureFactors) = 
      _analyzePressure(reading);
    riskScore += pressureRisk;
    riskFactors.addAll(pressureFactors);
    if (pressureRisk > 20) affectedZone = pressureZone;

    // Analyze temperature patterns
    final (tempRisk, tempZone, tempFactors) =
      _analyzeTemperature(reading);
    riskScore += tempRisk;
    riskFactors.addAll(tempFactors);
    if (tempRisk > 15 && affectedZone.isEmpty) affectedZone = tempZone;

    // Analyze activity patterns
    final (activityRisk, activityFactors) = 
      _analyzeActivity(reading);
    riskScore += activityRisk;
    riskFactors.addAll(activityFactors);

    // Analyze historical trends
    if (historicalReadings != null && historicalReadings.isNotEmpty) {
      final (trendRisk, trendFactors) = 
        _analyzeTrends(reading, historicalReadings);
      riskScore += trendRisk;
      riskFactors.addAll(trendFactors);
    }

    // Clamp score to 0-100
    riskScore = riskScore.clamp(0, 100);

    // Determine risk level
    final level = _getRiskLevel(riskScore);

    // Generate recommendation
    final rec = _getRecommendation(level, riskFactors);

    return FootUlcerPrediction(
      riskScore: double.parse(riskScore.toStringAsFixed(1)),
      level: level,
      riskFactors: riskFactors,
      timestamp: reading.timestamp,
      affectedZone: affectedZone.isEmpty ? 'Multiple' : affectedZone,
      recommendation: rec,
    );
  }

  /// Analyze pressure patterns
  static (double, String, List<String>) _analyzePressure(SensorReading reading) {
    double risk = 0;
    final factors = <String>[];
    String zone = '';

    for (int i = 0; i < reading.pressures.length; i++) {
      final pressure = reading.pressures[i];
      final zoneName = _getZoneName(i);

      if (pressure > pressureCritical) {
        risk += 25;
        factors.add('Critical pressure in $zoneName (${pressure.toStringAsFixed(1)} kPa)');
        zone = zoneName;
      } else if (pressure > pressureHigh) {
        risk += 15;
        factors.add('High pressure in $zoneName (${pressure.toStringAsFixed(1)} kPa)');
        if (zone.isEmpty) zone = zoneName;
      } else if (pressure > pressureModerate) {
        risk += 5;
        factors.add('Moderate pressure in $zoneName');
      }
    }

    return (risk, zone, factors);
  }

  /// Analyze temperature patterns
  static (double, String, List<String>) _analyzeTemperature(SensorReading reading) {
    double risk = 0;
    final factors = <String>[];
    String zone = '';

    for (int i = 0; i < reading.temperatures.length; i++) {
      final temp = reading.temperatures[i];
      final zoneName = _getZoneName(i);

      if (temp > tempExcessiveHigh) {
        risk += 20;
        factors.add('Excessive temperature in $zoneName (${temp.toStringAsFixed(1)}°C)');
        zone = zoneName;
      } else if (temp > tempModerateHigh) {
        risk += 10;
        factors.add('Elevated temperature in $zoneName (${temp.toStringAsFixed(1)}°C)');
        if (zone.isEmpty) zone = zoneName;
      }
    }

    // Check temperature asymmetry between zones
    if (reading.temperatures.isNotEmpty) {
      final maxTemp = reading.temperatures.reduce((a, b) => a > b ? a : b);
      final minTemp = reading.temperatures.reduce((a, b) => a < b ? a : b);
      final diff = maxTemp - minTemp;

      if (diff > tempZoneDiffCritical) {
        risk += 15;
        factors.add('Critical temperature asymmetry (${diff.toStringAsFixed(1)}°C difference)');
      } else if (diff > tempZoneDiffHigh) {
        risk += 8;
        factors.add('Significant temperature difference between zones');
      }
    }

    return (risk, zone, factors);
  }

  /// Analyze activity patterns
  static (double, List<String>) _analyzeActivity(SensorReading reading) {
    double risk = 0;
    final factors = <String>[];

    // Excessive activity can increase ulcer risk
    if (reading.stepCount > 10000) {
      risk += 5;
      factors.add('High activity level (${reading.stepCount} steps)');
    }

    // Running poses higher risk than walking
    if (reading.activityType == ActivityType.running) {
      risk += 8;
      factors.add('Running activity detected');
    } else if (reading.activityType == ActivityType.walking && reading.stepCount > 5000) {
      risk += 3;
      factors.add('Prolonged walking');
    }

    return (risk, factors);
  }

  /// Analyze historical trends
  static (double, List<String>) _analyzeTrends(
    SensorReading current,
    List<SensorReading> history,
  ) {
    double risk = 0;
    final factors = <String>[];

    if (history.length < 2) return (risk, factors);

    // Check for pressure trending upward
    if (history.length >= 3) {
      final recentPressures = history.take(3).map((r) => r.pressures.isNotEmpty ? r.pressures[0] : 0).toList();
      if (recentPressures[0] > recentPressures[1] && recentPressures[1] > recentPressures[2]) {
        risk += 8;
        factors.add('Pressure trending upward');
      }
    }

    // Check for temperature trending upward
    if (history.length >= 3) {
      final recentTemps = history.take(3).map((r) => r.temperatures.isNotEmpty ? r.temperatures[0] : 0).toList();
      if (recentTemps[0] > recentTemps[1] && recentTemps[1] > recentTemps[2]) {
        risk += 6;
        factors.add('Temperature trending upward');
      }
    }

    return (risk, factors);
  }

  /// Determine risk level from score
  static UlcerRiskLevel _getRiskLevel(double score) {
    if (score < 30) return UlcerRiskLevel.low;
    if (score < 60) return UlcerRiskLevel.moderate;
    if (score < 80) return UlcerRiskLevel.high;
    return UlcerRiskLevel.critical;
  }

  /// Get zone name from index
  static String _getZoneName(int index) {
    switch (index) {
      case 0:
        return 'Heel';
      case 1:
        return 'Ball';
      case 2:
        return 'Arch';
      case 3:
        return 'Toe';
      default:
        return 'Unknown';
    }
  }

  /// Generate personalized recommendation
  static String _getRecommendation(UlcerRiskLevel level, List<String> factors) {
    switch (level) {
      case UlcerRiskLevel.critical:
        return 'URGENT: Reduce activity immediately. Seek medical attention. Remove offending footwear.';
      case UlcerRiskLevel.high:
        return 'Reduce physical activity. Offload pressure areas. Consider changing footwear. Monitor closely.';
      case UlcerRiskLevel.moderate:
        return 'Monitor foot health daily. Practice good foot hygiene. Consider activity modification.';
      case UlcerRiskLevel.low:
        return 'Continue regular foot monitoring. Maintain good foot hygiene. Stay active within limits.';
    }
  }
}
