import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:flutter/foundation.dart';

/// Firebase Analytics Service
/// Tracks user behavior, events, and app usage
class FirebaseAnalyticsService {
  static final FirebaseAnalyticsService _instance =
      FirebaseAnalyticsService._internal();
  factory FirebaseAnalyticsService() => _instance;
  FirebaseAnalyticsService._internal();

  final FirebaseAnalytics _analytics = FirebaseAnalytics.instance;

  // ============== User Identification ==============

  /// Set user ID for tracking
  Future<void> setUserId(String userId) async {
    try {
      await _analytics.setUserId(id: userId);
    } catch (e) {
      debugPrint('Set User ID Error: $e');
    }
  }

  /// Set user properties
  Future<void> setUserProperty(String name, String value) async {
    try {
      await _analytics.setUserProperty(name: name, value: value);
    } catch (e) {
      debugPrint('Set User Property Error: $e');
    }
  }

  /// Clear user ID
  Future<void> clearUserId() async {
    try {
      await _analytics.setUserId(id: null);
    } catch (e) {
      debugPrint('Clear User ID Error: $e');
    }
  }

  // ============== Event Tracking ==============

  /// Track custom event
  Future<void> logEvent({
    required String name,
    Map<String, Object>? parameters,
  }) async {
    try {
      await _analytics.logEvent(
        name: name,
        parameters: parameters,
      );
    } catch (e) {
      debugPrint('Log Event Error: $e');
    }
  }

  /// Track screen view
  Future<void> logScreenView(String screenName) async {
    try {
      await _analytics.logScreenView(screenName: screenName);
    } catch (e) {
      debugPrint('Log Screen View Error: $e');
    }
  }

  /// Track login event
  Future<void> logLogin({
    required String method,
  }) async {
    try {
      await _analytics.logLogin(loginMethod: method);
    } catch (e) {
      debugPrint('Log Login Error: $e');
    }
  }

  /// Track signup event
  Future<void> logSignUp({
    required String method,
  }) async {
    try {
      await _analytics.logSignUp(signUpMethod: method);
    } catch (e) {
      debugPrint('Log Sign Up Error: $e');
    }
  }

  // ============== App Events ==============

  /// Track app open
  Future<void> logAppOpen() async {
    await logEvent(name: 'app_open');
  }

  /// Track profile setup completed
  Future<void> logProfileSetupCompleted({
    required String diabetesType,
  }) async {
    await logEvent(
      name: 'profile_setup_complete',
      parameters: {
        'diabetes_type': diabetesType,
      },
    );
  }

  /// Track BLE device connected
  Future<void> logBleDeviceConnected({
    required String deviceName,
    required String deviceId,
  }) async {
    await logEvent(
      name: 'ble_device_connected',
      parameters: {
        'device_name': deviceName,
        'device_id': deviceId,
      },
    );
  }

  /// Track high-risk alert triggered
  Future<void> logHighRiskAlert({
    required double riskLevel,
    required String zone,
    required String severity,
  }) async {
    await logEvent(
      name: 'high_risk_alert',
      parameters: {
        'risk_level': riskLevel,
        'zone': zone,
        'severity': severity,
      },
    );
  }

  /// Track sensor data received
  Future<void> logSensorDataReceived({
    required double temperature,
    required double humidity,
    required int sensorCount,
  }) async {
    await logEvent(
      name: 'sensor_data_received',
      parameters: {
        'temperature': temperature,
        'humidity': humidity,
        'sensor_count': sensorCount,
      },
    );
  }

  /// Track settings changed
  Future<void> logSettingsChanged({
    required String setting,
    required String value,
  }) async {
    await logEvent(
      name: 'settings_changed',
      parameters: {
        'setting': setting,
        'value': value,
      },
    );
  }

  /// Track data sync
  Future<void> logDataSync({
    required int recordsCount,
    required String syncStatus,
  }) async {
    await logEvent(
      name: 'data_sync',
      parameters: {
        'records_count': recordsCount,
        'sync_status': syncStatus,
      },
    );
  }

  /// Track error
  Future<void> logError({
    required String errorType,
    required String errorMessage,
    String? screen,
  }) async {
    await logEvent(
      name: 'app_error',
      parameters: {
        'error_type': errorType,
        'error_message': errorMessage,
        if (screen != null) 'screen': screen,
      },
    );
  }

  // ============== Analytics Settings ==============

  /// Enable/disable analytics collection
  Future<void> setAnalyticsCollectionEnabled(bool enabled) async {
    try {
      await _analytics.setAnalyticsCollectionEnabled(enabled);
    } catch (e) {
      debugPrint('Set Analytics Collection Error: $e');
    }
  }

  /// Check if analytics is enabled (always returns true by default)
  Future<bool> isAnalyticsCollectionEnabled() async {
    try {
      // Firebase Analytics is enabled by default
      return true;
    } catch (e) {
      debugPrint('Check Analytics Collection Error: $e');
      return false;
    }
  }
}
