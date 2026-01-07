// Monitors readings against thresholds, generates alerts with cooldown to prevent spam, provides streams and statistics.

import 'dart:async';
import '../models/sensor_reading.dart';
import '../models/alert.dart';
import '../../core/constants/sensor_constants.dart';

/// Service for checking sensor readings and generating alerts
class AlertService {
  // Singleton pattern
  static final AlertService _instance = AlertService._internal();
  factory AlertService() => _instance;
  AlertService._internal();

  // Alert storage
  final List<Alert> _alerts = [];
  static const int _maxAlerts = 100;

  // Cooldown tracking to prevent alert spam
  final Map<String, DateTime> _alertCooldowns = {};
  static const Duration _cooldownDuration = Duration(minutes: 5);

  // Stream for new alerts
  final StreamController<Alert> _alertStreamController =
      StreamController<Alert>.broadcast();

  // Previous readings for comparison
  SensorReading? _previousReading;

  // ============== Getters ==============

  /// Get all stored alerts
  List<Alert> get alerts => List.unmodifiable(_alerts);

  /// Get unread alerts
  List<Alert> get unreadAlerts =>
      _alerts.where((a) => !a.isRead).toList();

  /// Get unread alert count
  int get unreadCount => unreadAlerts.length;

  /// Get critical alerts
  List<Alert> get criticalAlerts =>
      _alerts.where((a) => a.severity == AlertSeverity.critical).toList();

  /// Stream of new alerts
  Stream<Alert> get alertStream => _alertStreamController.stream;

  /// Check if there are any critical unread alerts
  bool get hasCriticalUnread => _alerts.any(
      (a) => a.severity == AlertSeverity.critical && !a.isRead);

  // ============== Alert Checking ==============

  /// Check a sensor reading for alerts
  /// Returns list of new alerts generated
  List<Alert> checkForAlerts(SensorReading reading) {
    final newAlerts = <Alert>[];

    // Temperature checks
    newAlerts.addAll(_checkTemperatureAlerts(reading));

    // Pressure checks
    newAlerts.addAll(_checkPressureAlerts(reading));

    // SpO2 checks
    newAlerts.addAll(_checkSpO2Alerts(reading));

    // Heart rate checks
    newAlerts.addAll(_checkHeartRateAlerts(reading));

    // Temperature asymmetry check
    final asymmetryAlert = _checkTemperatureAsymmetry(reading);
    if (asymmetryAlert != null) newAlerts.add(asymmetryAlert);

    // Battery check
    final batteryAlert = _checkBattery(reading);
    if (batteryAlert != null) newAlerts.add(batteryAlert);

    // Store previous reading for comparison
    _previousReading = reading;

    // Add new alerts to storage and stream
    for (final alert in newAlerts) {
      _addAlert(alert);
    }

    return newAlerts;
  }

  /// Check temperature for each zone
  List<Alert> _checkTemperatureAlerts(SensorReading reading) {
    final alerts = <Alert>[];

    for (int i = 0; i < reading.temperatures.length; i++) {
      final temp = reading.temperatures[i];
      final zoneName = SensorConstants.getZoneName(i);
      final cooldownKey = 'temp_high_$i';

      // Check for high temperature
      if (temp > SensorConstants.tempWarningHigh) {
        if (_canCreateAlert(cooldownKey)) {
          alerts.add(Alert.highTemperature(
            zone: zoneName,
            temperature: temp,
            threshold: SensorConstants.tempWarningHigh,
          ));
          _setCooldown(cooldownKey);
        }
      }

      // Check for low temperature (circulation issue)
      final coldCooldownKey = 'temp_low_$i';
      if (temp < SensorConstants.tempWarningLow && temp > 0) {
        if (_canCreateAlert(coldCooldownKey)) {
          alerts.add(Alert.create(
            type: AlertType.temperature,
            severity: temp < SensorConstants.tempCriticalLow
                ? AlertSeverity.critical
                : AlertSeverity.warning,
            title: 'Low Temperature',
            message: '$zoneName area showing low temperature '
                '(${temp.toStringAsFixed(1)}°C). '
                'This may indicate poor circulation.',
            affectedZone: zoneName,
            actualValue: temp,
            threshold: SensorConstants.tempWarningLow,
            action: 'Warm your feet and check circulation',
          ));
          _setCooldown(coldCooldownKey);
        }
      }
    }

    return alerts;
  }

  /// Check pressure for each zone
  List<Alert> _checkPressureAlerts(SensorReading reading) {
    final alerts = <Alert>[];

    for (int i = 0; i < reading.pressures.length; i++) {
      final pressure = reading.pressures[i];
      final zoneName = SensorConstants.getZoneName(i);
      final cooldownKey = 'pressure_$i';

      if (pressure > SensorConstants.pressureWarning) {
        if (_canCreateAlert(cooldownKey)) {
          alerts.add(Alert.highPressure(
            zone: zoneName,
            pressure: pressure,
            threshold: SensorConstants.pressureWarning,
          ));
          _setCooldown(cooldownKey);
        }
      }
    }

    // Check for pressure spike
    if (_previousReading != null) {
      for (int i = 0; i < reading.pressures.length; i++) {
        if (i < _previousReading!.pressures.length) {
          final diff = reading.pressures[i] - _previousReading!.pressures[i];
          final zoneName = SensorConstants.getZoneName(i);
          final cooldownKey = 'pressure_spike_$i';

          if (diff > SensorConstants.pressureSpikeThreshold) {
            if (_canCreateAlert(cooldownKey)) {
              alerts.add(Alert.create(
                type: AlertType.pressure,
                severity: AlertSeverity.warning,
                title: 'Pressure Spike',
                message: 'Sudden pressure increase at $zoneName '
                    '(+${diff.toStringAsFixed(1)} kPa).',
                affectedZone: zoneName,
                actualValue: reading.pressures[i],
                threshold: SensorConstants.pressureSpikeThreshold,
                action: 'Check foot position and redistribute weight',
              ));
              _setCooldown(cooldownKey);
            }
          }
        }
      }
    }

    return alerts;
  }

  /// Check SpO2 levels
  List<Alert> _checkSpO2Alerts(SensorReading reading) {
    final alerts = <Alert>[];
    const cooldownKey = 'spo2';

    if (reading.spO2 < SensorConstants.spo2Normal && reading.spO2 > 0) {
      if (_canCreateAlert(cooldownKey)) {
        alerts.add(Alert.lowSpO2(
          spO2: reading.spO2,
          threshold: SensorConstants.spo2Normal,
        ));
        _setCooldown(cooldownKey);
      }
    }

    return alerts;
  }

  /// Check heart rate
  List<Alert> _checkHeartRateAlerts(SensorReading reading) {
    final alerts = <Alert>[];
    final hr = reading.heartRate;

    if (hr <= 0) return alerts;

    // Low heart rate
    if (hr < SensorConstants.hrWarningLow) {
      const cooldownKey = 'hr_low';
      if (_canCreateAlert(cooldownKey)) {
        alerts.add(Alert.create(
          type: AlertType.circulation,
          severity: hr < SensorConstants.hrCriticalLow
              ? AlertSeverity.critical
              : AlertSeverity.warning,
          title: 'Low Heart Rate',
          message: 'Heart rate is low ($hr BPM). '
              'This may indicate a health concern.',
          actualValue: hr.toDouble(),
          threshold: SensorConstants.hrWarningLow.toDouble(),
          action: 'Rest and monitor. Seek help if symptoms persist.',
        ));
        _setCooldown(cooldownKey);
      }
    }

    // High heart rate
    if (hr > SensorConstants.hrWarningHigh) {
      const cooldownKey = 'hr_high';
      if (_canCreateAlert(cooldownKey)) {
        alerts.add(Alert.create(
          type: AlertType.circulation,
          severity: hr > SensorConstants.hrCriticalHigh
              ? AlertSeverity.critical
              : AlertSeverity.warning,
          title: 'High Heart Rate',
          message: 'Heart rate is elevated ($hr BPM). '
              'Consider resting if not exercising.',
          actualValue: hr.toDouble(),
          threshold: SensorConstants.hrWarningHigh.toDouble(),
          action: 'Rest and take slow, deep breaths',
        ));
        _setCooldown(cooldownKey);
      }
    }

    return alerts;
  }

  /// Check for temperature asymmetry
  Alert? _checkTemperatureAsymmetry(SensorReading reading) {
    if (reading.temperatures.length < 2) return null;

    const cooldownKey = 'temp_asymmetry';

    final temps = reading.temperatures;
    final maxTemp = temps.reduce((a, b) => a > b ? a : b);
    final minTemp = temps.reduce((a, b) => a < b ? a : b);
    final diff = maxTemp - minTemp;

    if (diff > SensorConstants.tempAsymmetryWarning) {
      if (_canCreateAlert(cooldownKey)) {
        final hotZoneIndex = temps.indexOf(maxTemp);
        final hotZone = SensorConstants.getZoneName(hotZoneIndex);

        _setCooldown(cooldownKey);
        return Alert.temperatureAsymmetry(
          difference: diff,
          threshold: SensorConstants.tempAsymmetryWarning,
          hotterZone: hotZone,
        );
      }
    }

    return null;
  }

  /// Check battery level
  Alert? _checkBattery(SensorReading reading) {
    const cooldownKey = 'battery';

    if (reading.batteryLevel <= SensorConstants.batteryLow) {
      if (_canCreateAlert(cooldownKey)) {
        _setCooldown(cooldownKey);
        return Alert.batteryLow(level: reading.batteryLevel);
      }
    }

    return null;
  }

  // ============== Cooldown Management ==============

  /// Check if we can create an alert (cooldown expired)
  bool _canCreateAlert(String key) {
    final lastAlert = _alertCooldowns[key];
    if (lastAlert == null) return true;

    return DateTime.now().difference(lastAlert) > _cooldownDuration;
  }

  /// Set cooldown for an alert type
  void _setCooldown(String key) {
    _alertCooldowns[key] = DateTime.now();
  }

  /// Clear all cooldowns (for testing)
  void clearCooldowns() {
    _alertCooldowns.clear();
  }

  // ============== Alert Management ==============

  /// Add an alert to storage
  void _addAlert(Alert alert) {
    _alerts.insert(0, alert); // Add to beginning (most recent first)

    // Trim if over max
    while (_alerts.length > _maxAlerts) {
      _alerts.removeLast();
    }

    // Emit to stream
    if (!_alertStreamController.isClosed) {
      _alertStreamController.add(alert);
    }
  }

  /// Add a custom alert
  void addAlert(Alert alert) {
    _addAlert(alert);
  }

  /// Mark an alert as read
  void markAsRead(String alertId) {
    final index = _alerts.indexWhere((a) => a.id == alertId);
    if (index != -1) {
      _alerts[index] = _alerts[index].markAsRead();
    }
  }

  /// Mark all alerts as read
  void markAllAsRead() {
    for (int i = 0; i < _alerts.length; i++) {
      if (!_alerts[i].isRead) {
        _alerts[i] = _alerts[i].markAsRead();
      }
    }
  }

  /// Remove an alert
  void removeAlert(String alertId) {
    _alerts.removeWhere((a) => a.id == alertId);
  }

  /// Clear all alerts
  void clearAlerts() {
    _alerts.clear();
  }

  /// Clear alerts older than specified duration
  void clearOldAlerts(Duration maxAge) {
    final cutoff = DateTime.now().subtract(maxAge);
    _alerts.removeWhere((a) => a.timestamp.isBefore(cutoff));
  }

  /// Get alerts filtered by type
  List<Alert> getAlertsByType(AlertType type) {
    return _alerts.where((a) => a.type == type).toList();
  }

  /// Get alerts filtered by severity
  List<Alert> getAlertsBySeverity(AlertSeverity severity) {
    return _alerts.where((a) => a.severity == severity).toList();
  }

  /// Get alerts from the last N hours
  List<Alert> getRecentAlerts(int hours) {
    final cutoff = DateTime.now().subtract(Duration(hours: hours));
    return _alerts.where((a) => a.timestamp.isAfter(cutoff)).toList();
  }

  /// Get alert statistics
  AlertStats getStats() {
    return AlertStats(
      total: _alerts.length,
      unread: unreadCount,
      critical: _alerts.where((a) => a.severity == AlertSeverity.critical).length,
      warning: _alerts.where((a) => a.severity == AlertSeverity.warning).length,
      info: _alerts.where((a) => a.severity == AlertSeverity.info).length,
      temperatureAlerts: _alerts.where((a) => a.type == AlertType.temperature).length,
      pressureAlerts: _alerts.where((a) => a.type == AlertType.pressure).length,
      circulationAlerts: _alerts.where((a) => a.type == AlertType.circulation).length,
      gaitAlerts: _alerts.where((a) => a.type == AlertType.gait).length,
    );
  }

  /// Dispose resources
  void dispose() {
    _alertStreamController.close();
  }
}

/// Statistics about alerts
class AlertStats {
  final int total;
  final int unread;
  final int critical;
  final int warning;
  final int info;
  final int temperatureAlerts;
  final int pressureAlerts;
  final int circulationAlerts;
  final int gaitAlerts;

  const AlertStats({
    required this.total,
    required this.unread,
    required this.critical,
    required this.warning,
    required this.info,
    required this.temperatureAlerts,
    required this.pressureAlerts,
    required this.circulationAlerts,
    required this.gaitAlerts,
  });

  @override
  String toString() =>
      'AlertStats(total: $total, unread: $unread, critical: $critical, '
      'warning: $warning, info: $info)';
}
