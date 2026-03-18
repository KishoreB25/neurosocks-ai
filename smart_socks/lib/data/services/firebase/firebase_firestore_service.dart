import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import '../../models/user_profile.dart';
import '../../models/sensor_reading.dart';
import '../../models/risk_score.dart';
import '../../models/alert.dart';

/// Firebase Firestore Service
/// Handles all cloud database operations for user data, sensor readings, risk scores, alerts
class FirebaseFirestoreService {
  static final FirebaseFirestoreService _instance =
      FirebaseFirestoreService._internal();
  factory FirebaseFirestoreService() => _instance;
  FirebaseFirestoreService._internal();

  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  // Collection names
  static const String usersCollection = 'users';
  static const String sensorReadingsCollection = 'sensorReadings';
  static const String riskScoresCollection = 'riskScores';
  static const String alertsCollection = 'alerts';
  static const String dailySummariesCollection = 'dailySummaries';
  static const String tokensCollection = 'tokens';
  static const String predictionsCollection = 'predictions';
  static const String reportsCollection = 'reports';
  static const String activityLogsCollection = 'activityLogs';
  static const String deviceDataCollection = 'deviceData';
  static const String healthMetricsCollection = 'healthMetrics';
  static const String notificationsCollection = 'notifications';
  static const String userSettingsCollection = 'userSettings';

  // ============== User Profile ==============

  /// Save user profile to Firestore
  Future<void> saveUserProfile({
    required String userId,
    required UserProfile profile,
  }) async {
    try {
      await _firestore.collection(usersCollection).doc(userId).set(
        {
          'email': profile.email,
          'name': profile.name,
          'age': profile.age,
          'diabetesType': profile.diabetesType.toString(),
          'diabetesYears': profile.diabetesYears,
          'healthInfo': {
            'hasNeuropathy': profile.healthInfo?.hasNeuropathy ?? false,
            'hasPAD': profile.healthInfo?.hasPAD ?? false,
            'hasPreviousUlcer': profile.healthInfo?.hasPreviousUlcer ?? false,
            'hasHypertension': profile.healthInfo?.hasHypertension ?? false,
          },
          'settings': {
            'temperatureUnit':
                profile.settings.temperatureUnit.toString(),
            'notificationsEnabled':
                profile.settings.notificationsEnabled,
            'criticalAlertsEnabled':
                profile.settings.criticalAlertsEnabled,
          },
          'createdAt': profile.createdAt,
          'updatedAt': DateTime.now(),
        },
        SetOptions(merge: true),
      );
    } catch (e) {
      debugPrint('Save User Profile Error: $e');
      rethrow;
    }
  }

  /// Get user profile from Firestore
  Future<UserProfile?> getUserProfile(String userId) async {
    try {
      final doc =
          await _firestore.collection(usersCollection).doc(userId).get();
      if (!doc.exists) return null;

      final data = doc.data();
      if (data == null) return null;

      return UserProfile(
        id: userId,
        email: data['email'] ?? '',
        name: data['name'] ?? '',
        age: data['age'],
        diabetesType: _parseDiabetesType(data['diabetesType']),
        diabetesYears: data['diabetesYears'],
        healthInfo: HealthInfo(
          hasNeuropathy: data['healthInfo']?['hasNeuropathy'] ?? false,
          hasPAD: data['healthInfo']?['hasPAD'] ?? false,
          hasPreviousUlcer: data['healthInfo']?['hasPreviousUlcer'] ?? false,
          hasHypertension: data['healthInfo']?['hasHypertension'] ?? false,
        ),
        settings: UserSettings(
          temperatureUnit: _parseTemperatureUnit(
              data['settings']?['temperatureUnit'] ?? 'celsius'),
          notificationsEnabled:
              data['settings']?['notificationsEnabled'] ?? true,
          criticalAlertsEnabled:
              data['settings']?['criticalAlertsEnabled'] ?? true,
        ),
        createdAt: (data['createdAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
        updatedAt: (data['updatedAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
      );
    } catch (e) {
      debugPrint('Get User Profile Error: $e');
      return null;
    }
  }

  // ============== Sensor Readings ==============

  /// Save sensor reading to Firestore
  Future<void> saveSensorReading({
    required String userId,
    required SensorReading reading,
  }) async {
    try {
      final timestamp = reading.timestamp.millisecondsSinceEpoch.toString();
      await _firestore
          .collection(sensorReadingsCollection)
          .doc(userId)
          .collection('readings')
          .doc(timestamp)
          .set(reading.toJson());
    } catch (e) {
      debugPrint('Save Sensor Reading Error: $e');
      rethrow;
    }
  }

  /// Get sensor readings for user (paginated)
  Future<List<SensorReading>> getSensorReadings({
    required String userId,
    int limit = 100,
  }) async {
    try {
      final snapshot = await _firestore
          .collection(sensorReadingsCollection)
          .doc(userId)
          .collection('readings')
          .orderBy('timestamp', descending: true)
          .limit(limit)
          .get();

      return snapshot.docs
          .map((doc) => SensorReading.fromJson(
              doc.data()))
          .toList();
    } catch (e) {
      debugPrint('Get Sensor Readings Error: $e');
      return [];
    }
  }

  /// Get recent readings from Firestore (alias for getSensorReadings)
  Future<List<SensorReading>> getRecentReadings({
    required String userId,
    int limit = 50,
  }) async {
    return getSensorReadings(userId: userId, limit: limit);
  }

  // ============== Risk Scores ==============

  /// Save risk score to Firestore
  Future<void> saveRiskScore({
    required String userId,
    required RiskScore riskScore,
  }) async {
    try {
      final timestamp = riskScore.timestamp.millisecondsSinceEpoch.toString();
      await _firestore
          .collection(riskScoresCollection)
          .doc(userId)
          .collection('scores')
          .doc(timestamp)
          .set(riskScore.toJson());
    } catch (e) {
      debugPrint('Save Risk Score Error: $e');
      rethrow;
    }
  }

  /// Get latest risk score
  Future<RiskScore?> getLatestRiskScore(String userId) async {
    try {
      final snapshot = await _firestore
          .collection(riskScoresCollection)
          .doc(userId)
          .collection('scores')
          .orderBy('timestamp', descending: true)
          .limit(1)
          .get();

      if (snapshot.docs.isEmpty) return null;
      return RiskScore.fromJson(
          snapshot.docs.first.data());
    } catch (e) {
      debugPrint('Get Latest Risk Score Error: $e');
      return null;
    }
  }

  // ============== Alerts ==============

  /// Save alert to Firestore
  Future<void> saveAlert({
    required String userId,
    required Alert alert,
  }) async {
    try {
      await _firestore
          .collection(alertsCollection)
          .doc(userId)
          .collection('userAlerts')
          .doc()
          .set(alert.toJson());
    } catch (e) {
      debugPrint('Save Alert Error: $e');
      rethrow;
    }
  }

  /// Get alerts for user
  Future<List<Alert>> getAlerts({
    required String userId,
    int limit = 50,
  }) async {
    try {
      final snapshot = await _firestore
          .collection(alertsCollection)
          .doc(userId)
          .collection('userAlerts')
          .orderBy('timestamp', descending: true)
          .limit(limit)
          .get();

      return snapshot.docs
          .map((doc) => Alert.fromJson(doc.data()))
          .toList();
    } catch (e) {
      debugPrint('Get Alerts Error: $e');
      return [];
    }
  }

  // ============== Real-time Listeners ==============

  // ============== ML Predictions ==============

  /// Save ML prediction to Firestore (user-specific)
  Future<void> savePrediction({
    required String userId,
    required Map<String, dynamic> predictionData,
  }) async {
    try {
      final timestamp = predictionData['timestamp'] ?? DateTime.now().toIso8601String();
      final docId = DateTime.now().millisecondsSinceEpoch.toString();
      await _firestore
          .collection(predictionsCollection)
          .doc(userId)
          .collection('userPredictions')
          .doc(docId)
          .set({
            ...predictionData,
            'timestamp': timestamp,
            'createdAt': FieldValue.serverTimestamp(),
          });
    } catch (e) {
      debugPrint('Save Prediction Error: $e');
      rethrow;
    }
  }

  /// Get predictions for user
  Future<List<Map<String, dynamic>>> getPredictions({
    required String userId,
    int limit = 50,
  }) async {
    try {
      final snapshot = await _firestore
          .collection(predictionsCollection)
          .doc(userId)
          .collection('userPredictions')
          .orderBy('createdAt', descending: true)
          .limit(limit)
          .get();

      return snapshot.docs.map((doc) => doc.data()).toList();
    } catch (e) {
      debugPrint('Get Predictions Error: $e');
      return [];
    }
  }

  // ============== Real-time Listeners (streams) ==============

  /// Listen to user profile changes
  Stream<UserProfile?> userProfileStream(String userId) {
    return _firestore.collection(usersCollection).doc(userId).snapshots().map(
      (doc) {
        if (!doc.exists) return null;
        final data = doc.data() as Map<String, dynamic>;
        return UserProfile(
          id: userId,
          email: data['email'] ?? '',
          name: data['name'] ?? '',
          age: data['age'],
          diabetesType: _parseDiabetesType(data['diabetesType']),
          diabetesYears: data['diabetesYears'],
          healthInfo: HealthInfo(
            hasNeuropathy: data['healthInfo']?['hasNeuropathy'] ?? false,
            hasPAD: data['healthInfo']?['hasPAD'] ?? false,
            hasPreviousUlcer: data['healthInfo']?['hasPreviousUlcer'] ?? false,
            hasHypertension: data['healthInfo']?['hasHypertension'] ?? false,
          ),
          settings: UserSettings(
            temperatureUnit: _parseTemperatureUnit(
                data['settings']?['temperatureUnit'] ?? 'celsius'),
            notificationsEnabled:
                data['settings']?['notificationsEnabled'] ?? true,
            criticalAlertsEnabled:
                data['settings']?['criticalAlertsEnabled'] ?? true,
          ),
          createdAt:
              (data['createdAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
          updatedAt:
              (data['updatedAt'] as Timestamp?)?.toDate() ?? DateTime.now(),
        );
      },
    );
  }

  /// Listen to latest alerts
  Stream<List<Alert>> alertsStream(String userId) {
    return _firestore
        .collection(alertsCollection)
        .doc(userId)
        .collection('userAlerts')
        .orderBy('timestamp', descending: true)
        .limit(20)
        .snapshots()
        .map(
      (snapshot) {
        return snapshot.docs
            .map((doc) => Alert.fromJson(doc.data()))
            .toList();
      },
    );
  }

  // ============== FCM Tokens ==============

  /// Save FCM token for push notifications
  Future<void> saveFCMToken({
    required String userId,
    required String fcmToken,
    String deviceName = '',
  }) async {
    try {
      await _firestore
          .collection(usersCollection)
          .doc(userId)
          .collection('tokens')
          .doc('fcm')
          .set({
            'token': fcmToken,
            'deviceName': deviceName,
            'updatedAt': DateTime.now(),
            'platform': _getDevicePlatform(),
          }, SetOptions(merge: true));
    } catch (e) {
      debugPrint('Save FCM Token Error: $e');
    }
  }

  String _getDevicePlatform() {
    // This would use platform-specific code in real implementation
    return 'unknown';
  }

  // ============== Daily Summaries ==============

  /// Save daily summary
  Future<void> saveDailySummary({
    required String userId,
    required Map<String, dynamic> summaryData,
  }) async {
    try {
      final dateKey = DateTime.now().toIso8601String().split('T')[0]; // YYYY-MM-DD
      await _firestore
          .collection(usersCollection)
          .doc(userId)
          .collection('dailySummaries')
          .doc(dateKey)
          .set(summaryData, SetOptions(merge: true));
    } catch (e) {
      debugPrint('Save Daily Summary Error: $e');
    }
  }

  /// Get daily summary for a date
  Future<Map<String, dynamic>?> getDailySummary({
    required String userId,
    required DateTime date,
  }) async {
    try {
      final dateKey = date.toIso8601String().split('T')[0];
      final doc = await _firestore
          .collection(usersCollection)
          .doc(userId)
          .collection('dailySummaries')
          .doc(dateKey)
          .get();

      return doc.data();
    } catch (e) {
      debugPrint('Get Daily Summary Error: $e');
      return null;
    }
  }

  // ============== Health Metrics ==============

  /// Save health metric (aggregated stats)
  Future<void> saveHealthMetric({
    required String userId,
    required String metricName,
    required Map<String, dynamic> metricData,
  }) async {
    try {
      await _firestore
          .collection(usersCollection)
          .doc(userId)
          .collection('healthMetrics')
          .doc(metricName)
          .set(metricData, SetOptions(merge: true));
    } catch (e) {
      debugPrint('Save Health Metric Error: $e');
    }
  }

  // ============== Activity Logs ==============

  /// Log user activity
  Future<void> logActivity({
    required String userId,
    required String activityType,
    required Map<String, dynamic> activityData,
  }) async {
    try {
      await _firestore
          .collection(usersCollection)
          .doc(userId)
          .collection('activityLogs')
          .doc()
          .set({
            'type': activityType,
            'timestamp': DateTime.now(),
            ...activityData,
          });
    } catch (e) {
      debugPrint('Log Activity Error: $e');
    }
  }

  // ============== Helpers ==============

  DiabetesType _parseDiabetesType(String? type) {
    if (type == null) return DiabetesType.type2;
    if (type.contains('type1')) return DiabetesType.type1;
    if (type.contains('type2')) return DiabetesType.type2;
    if (type.contains('preDiabetes')) return DiabetesType.preDiabetes;
    return DiabetesType.none;
  }

  TemperatureUnit _parseTemperatureUnit(String? unit) {
    if (unit == null) return TemperatureUnit.celsius;
    if (unit.contains('fahrenheit')) return TemperatureUnit.fahrenheit;
    return TemperatureUnit.celsius;
  }
}
