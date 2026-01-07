import 'package:equatable/equatable.dart';
import '../../core/constants/app_colors.dart';
import '../../core/constants/sensor_constants.dart';
import 'package:flutter/material.dart';

/// Represents a single zone on the foot with its sensor data
class FootZone extends Equatable {
  /// Zone identifier (0-3)
  final int index;

  /// Zone name (Heel, Ball, Arch, Toe)
  final String name;

  /// Temperature reading in °C
  final double temperature;

  /// Pressure reading in kPa
  final double pressure;

  /// Calculated risk level for this zone
  final ZoneRiskLevel riskLevel;

  const FootZone({
    required this.index,
    required this.name,
    required this.temperature,
    required this.pressure,
    required this.riskLevel,
  });

  /// Create FootZone from sensor readings at a specific index
  factory FootZone.fromReadings({
    required int index,
    required double temperature,
    required double pressure,
  }) {
    return FootZone(
      index: index,
      name: SensorConstants.getZoneName(index),
      temperature: temperature,
      pressure: pressure,
      riskLevel: _calculateZoneRisk(temperature, pressure),
    );
  }

  /// Create an empty/default zone
  factory FootZone.empty(int index) {
    return FootZone(
      index: index,
      name: SensorConstants.getZoneName(index),
      temperature: 0.0,
      pressure: 0.0,
      riskLevel: ZoneRiskLevel.unknown,
    );
  }

  /// Calculate risk level based on temperature and pressure
  static ZoneRiskLevel _calculateZoneRisk(double temp, double pressure) {
    int riskScore = 0;

    // Temperature risk
    if (temp > SensorConstants.tempCriticalHigh || 
        temp < SensorConstants.tempCriticalLow) {
      riskScore += 3;
    } else if (temp > SensorConstants.tempWarningHigh || 
               temp < SensorConstants.tempWarningLow) {
      riskScore += 2;
    } else if (!SensorConstants.isTemperatureNormal(temp) && temp > 0) {
      riskScore += 1;
    }

    // Pressure risk
    if (pressure > SensorConstants.pressureCritical) {
      riskScore += 3;
    } else if (pressure > SensorConstants.pressureHigh) {
      riskScore += 2;
    } else if (pressure > SensorConstants.pressureWarning) {
      riskScore += 1;
    }

    // Determine risk level
    if (riskScore >= 5) return ZoneRiskLevel.critical;
    if (riskScore >= 3) return ZoneRiskLevel.high;
    if (riskScore >= 1) return ZoneRiskLevel.moderate;
    if (temp > 0 || pressure > 0) return ZoneRiskLevel.normal;
    return ZoneRiskLevel.unknown;
  }

  /// Get the color for this zone based on risk level
  Color get color => riskLevel.color;

  /// Get the background color for this zone
  Color get backgroundColor => riskLevel.backgroundColor;

  /// Check if this zone needs attention
  bool get needsAttention => 
      riskLevel == ZoneRiskLevel.high || 
      riskLevel == ZoneRiskLevel.critical;

  /// Check if temperature is elevated
  bool get isTemperatureElevated => 
      temperature > SensorConstants.tempWarningHigh;

  /// Check if pressure is elevated
  bool get isPressureElevated => 
      pressure > SensorConstants.pressureWarning;

  /// Create a copy with modified fields
  FootZone copyWith({
    int? index,
    String? name,
    double? temperature,
    double? pressure,
    ZoneRiskLevel? riskLevel,
  }) {
    return FootZone(
      index: index ?? this.index,
      name: name ?? this.name,
      temperature: temperature ?? this.temperature,
      pressure: pressure ?? this.pressure,
      riskLevel: riskLevel ?? this.riskLevel,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'index': index,
      'name': name,
      'temperature': temperature,
      'pressure': pressure,
      'riskLevel': riskLevel.name,
    };
  }

  factory FootZone.fromJson(Map<String, dynamic> json) {
    return FootZone(
      index: json['index'] ?? 0,
      name: json['name'] ?? 'Unknown',
      temperature: (json['temperature'] ?? 0).toDouble(),
      pressure: (json['pressure'] ?? 0).toDouble(),
      riskLevel: ZoneRiskLevel.fromString(json['riskLevel']),
    );
  }

  @override
  List<Object?> get props => [index, name, temperature, pressure, riskLevel];

  @override
  String toString() => 
      'FootZone($name: ${temperature.toStringAsFixed(1)}°C, '
      '${pressure.toStringAsFixed(1)}kPa, $riskLevel)';
}

/// Complete foot data containing all 4 zones
class FootData extends Equatable {
  /// Heel zone (index 0)
  final FootZone heel;

  /// Ball of foot zone (index 1)
  final FootZone ball;

  /// Arch zone (index 2)
  final FootZone arch;

  /// Toe zone (index 3)
  final FootZone toe;

  /// Timestamp of this data
  final DateTime timestamp;

  /// Which foot this data represents (for future dual-sock support)
  final FootSide side;

  const FootData({
    required this.heel,
    required this.ball,
    required this.arch,
    required this.toe,
    required this.timestamp,
    this.side = FootSide.left,
  });

  /// Create FootData from temperature and pressure lists
  factory FootData.fromLists({
    required List<double> temperatures,
    required List<double> pressures,
    DateTime? timestamp,
    FootSide side = FootSide.left,
  }) {
    return FootData(
      heel: FootZone.fromReadings(
        index: 0,
        temperature: temperatures.isNotEmpty ? temperatures[0] : 0,
        pressure: pressures.isNotEmpty ? pressures[0] : 0,
      ),
      ball: FootZone.fromReadings(
        index: 1,
        temperature: temperatures.length > 1 ? temperatures[1] : 0,
        pressure: pressures.length > 1 ? pressures[1] : 0,
      ),
      arch: FootZone.fromReadings(
        index: 2,
        temperature: temperatures.length > 2 ? temperatures[2] : 0,
        pressure: pressures.length > 2 ? pressures[2] : 0,
      ),
      toe: FootZone.fromReadings(
        index: 3,
        temperature: temperatures.length > 3 ? temperatures[3] : 0,
        pressure: pressures.length > 3 ? pressures[3] : 0,
      ),
      timestamp: timestamp ?? DateTime.now(),
      side: side,
    );
  }

  /// Create empty foot data
  factory FootData.empty({FootSide side = FootSide.left}) {
    return FootData(
      heel: FootZone.empty(0),
      ball: FootZone.empty(1),
      arch: FootZone.empty(2),
      toe: FootZone.empty(3),
      timestamp: DateTime.now(),
      side: side,
    );
  }

  /// Get all zones as a list
  List<FootZone> get allZones => [heel, ball, arch, toe];

  /// Get zone by index (0-3)
  FootZone getZoneByIndex(int index) {
    switch (index) {
      case 0:
        return heel;
      case 1:
        return ball;
      case 2:
        return arch;
      case 3:
        return toe;
      default:
        return heel;
    }
  }

  /// Get zone by name
  FootZone? getZoneByName(String name) {
    switch (name.toLowerCase()) {
      case 'heel':
        return heel;
      case 'ball':
        return ball;
      case 'arch':
        return arch;
      case 'toe':
        return toe;
      default:
        return null;
    }
  }

  // ============== Aggregate Calculations ==============

  /// Average temperature across all zones
  double get averageTemperature {
    return (heel.temperature + ball.temperature + 
            arch.temperature + toe.temperature) / 4;
  }

  /// Average pressure across all zones
  double get averagePressure {
    return (heel.pressure + ball.pressure + 
            arch.pressure + toe.pressure) / 4;
  }

  /// Maximum temperature among all zones
  double get maxTemperature {
    return [heel.temperature, ball.temperature, 
            arch.temperature, toe.temperature]
        .reduce((a, b) => a > b ? a : b);
  }

  /// Maximum pressure among all zones
  double get maxPressure {
    return [heel.pressure, ball.pressure, 
            arch.pressure, toe.pressure]
        .reduce((a, b) => a > b ? a : b);
  }

  /// Zone with highest temperature
  FootZone get hottestZone {
    return allZones.reduce((a, b) => 
        a.temperature > b.temperature ? a : b);
  }

  /// Zone with highest pressure
  FootZone get highestPressureZone {
    return allZones.reduce((a, b) => 
        a.pressure > b.pressure ? a : b);
  }

  /// Zones that need attention (high/critical risk)
  List<FootZone> get zonesNeedingAttention {
    return allZones.where((zone) => zone.needsAttention).toList();
  }

  /// Overall risk level based on all zones
  ZoneRiskLevel get overallRiskLevel {
    if (allZones.any((z) => z.riskLevel == ZoneRiskLevel.critical)) {
      return ZoneRiskLevel.critical;
    }
    if (allZones.any((z) => z.riskLevel == ZoneRiskLevel.high)) {
      return ZoneRiskLevel.high;
    }
    if (allZones.any((z) => z.riskLevel == ZoneRiskLevel.moderate)) {
      return ZoneRiskLevel.moderate;
    }
    if (allZones.any((z) => z.riskLevel == ZoneRiskLevel.normal)) {
      return ZoneRiskLevel.normal;
    }
    return ZoneRiskLevel.unknown;
  }

  /// Temperature variance (difference between hottest and coldest)
  double get temperatureVariance {
    final temps = [heel.temperature, ball.temperature, 
                   arch.temperature, toe.temperature];
    final max = temps.reduce((a, b) => a > b ? a : b);
    final min = temps.reduce((a, b) => a < b ? a : b);
    return max - min;
  }

  /// Pressure distribution score (0-1, 1 = perfectly even)
  double get pressureDistributionScore {
    final pressures = [heel.pressure, ball.pressure, 
                       arch.pressure, toe.pressure];
    if (pressures.every((p) => p == 0)) return 1.0;
    
    final avg = averagePressure;
    if (avg == 0) return 1.0;
    
    double variance = 0;
    for (final p in pressures) {
      variance += (p - avg) * (p - avg);
    }
    variance = variance / pressures.length;
    
    // Normalize: lower variance = higher score
    final normalizedVariance = variance / (avg * avg);
    return (1 - normalizedVariance).clamp(0.0, 1.0);
  }

  /// Check if there's significant temperature asymmetry
  bool get hasTemperatureAsymmetry {
    return temperatureVariance > SensorConstants.tempAsymmetryWarning;
  }

  // ============== Serialization ==============

  Map<String, dynamic> toJson() {
    return {
      'heel': heel.toJson(),
      'ball': ball.toJson(),
      'arch': arch.toJson(),
      'toe': toe.toJson(),
      'timestamp': timestamp.millisecondsSinceEpoch,
      'side': side.name,
    };
  }

  factory FootData.fromJson(Map<String, dynamic> json) {
    return FootData(
      heel: FootZone.fromJson(json['heel'] ?? {}),
      ball: FootZone.fromJson(json['ball'] ?? {}),
      arch: FootZone.fromJson(json['arch'] ?? {}),
      toe: FootZone.fromJson(json['toe'] ?? {}),
      timestamp: DateTime.fromMillisecondsSinceEpoch(
        json['timestamp'] ?? DateTime.now().millisecondsSinceEpoch,
      ),
      side: FootSide.fromString(json['side']),
    );
  }

  /// Create a copy with modified fields
  FootData copyWith({
    FootZone? heel,
    FootZone? ball,
    FootZone? arch,
    FootZone? toe,
    DateTime? timestamp,
    FootSide? side,
  }) {
    return FootData(
      heel: heel ?? this.heel,
      ball: ball ?? this.ball,
      arch: arch ?? this.arch,
      toe: toe ?? this.toe,
      timestamp: timestamp ?? this.timestamp,
      side: side ?? this.side,
    );
  }

  @override
  List<Object?> get props => [heel, ball, arch, toe, timestamp, side];

  @override
  String toString() => 
      'FootData(${side.name}: heel=${heel.temperature}°C, '
      'ball=${ball.temperature}°C, arch=${arch.temperature}°C, '
      'toe=${toe.temperature}°C)';
}

/// Risk level for a specific foot zone
enum ZoneRiskLevel {
  normal,
  moderate,
  high,
  critical,
  unknown;

  /// Create from string value
  static ZoneRiskLevel fromString(String? value) {
    switch (value?.toLowerCase()) {
      case 'normal':
        return ZoneRiskLevel.normal;
      case 'moderate':
        return ZoneRiskLevel.moderate;
      case 'high':
        return ZoneRiskLevel.high;
      case 'critical':
        return ZoneRiskLevel.critical;
      default:
        return ZoneRiskLevel.unknown;
    }
  }

  /// Get color for this risk level
  Color get color {
    switch (this) {
      case ZoneRiskLevel.normal:
        return AppColors.riskLow;
      case ZoneRiskLevel.moderate:
        return AppColors.riskModerate;
      case ZoneRiskLevel.high:
        return AppColors.riskHigh;
      case ZoneRiskLevel.critical:
        return AppColors.riskCritical;
      case ZoneRiskLevel.unknown:
        return AppColors.disabled;
    }
  }

  /// Get background color for this risk level
  Color get backgroundColor {
    switch (this) {
      case ZoneRiskLevel.normal:
        return AppColors.riskLowBg;
      case ZoneRiskLevel.moderate:
        return AppColors.riskModerateBg;
      case ZoneRiskLevel.high:
        return AppColors.riskHighBg;
      case ZoneRiskLevel.critical:
        return AppColors.riskCriticalBg;
      case ZoneRiskLevel.unknown:
        return AppColors.disabledBg;
    }
  }

  /// Display name for UI
  String get displayName {
    switch (this) {
      case ZoneRiskLevel.normal:
        return 'Normal';
      case ZoneRiskLevel.moderate:
        return 'Moderate';
      case ZoneRiskLevel.high:
        return 'High';
      case ZoneRiskLevel.critical:
        return 'Critical';
      case ZoneRiskLevel.unknown:
        return 'Unknown';
    }
  }
}

/// Which foot the data represents
enum FootSide {
  left,
  right;

  static FootSide fromString(String? value) {
    switch (value?.toLowerCase()) {
      case 'right':
        return FootSide.right;
      default:
        return FootSide.left;
    }
  }

  String get displayName {
    switch (this) {
      case FootSide.left:
        return 'Left Foot';
      case FootSide.right:
        return 'Right Foot';
    }
  }
}
