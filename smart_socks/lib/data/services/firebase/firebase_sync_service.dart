import 'package:flutter/foundation.dart';
import '../storage_service.dart';
import 'firebase_firestore_service.dart';
import '../../models/user_profile.dart';
import '../../models/sensor_reading.dart';
import '../../models/risk_score.dart';
import '../../models/alert.dart';

/// Firebase Sync Service
/// Synchronizes local Hive data with Firebase Firestore
/// Ensures data is synced when online, and queued when offline
class FirebaseSyncService {
  static final FirebaseSyncService _instance =
      FirebaseSyncService._internal();
  factory FirebaseSyncService() => _instance;
  FirebaseSyncService._internal();

  final _storageService = StorageService();
  final _firestoreService = FirebaseFirestoreService();

  bool _isSyncing = false;
  DateTime? _lastSyncTime;

  bool get isSyncing => _isSyncing;
  DateTime? get lastSyncTime => _lastSyncTime;

  // ============== Profile Sync ==============

  /// Sync user profile to Firebase
  Future<bool> syncUserProfile({
    required String userId,
    required UserProfile profile,
  }) async {
    try {
      await _firestoreService.saveUserProfile(
        userId: userId,
        profile: profile,
      );
      return true;
    } catch (e) {
      debugPrint('Sync User Profile Error: $e');
      return false;
    }
  }

  /// Pull user profile from Firebase
  Future<UserProfile?> pullUserProfile(String userId) async {
    try {
      return await _firestoreService.getUserProfile(userId);
    } catch (e) {
      debugPrint('Pull User Profile Error: $e');
      return null;
    }
  }

  // ============== Sensor Readings Sync ==============

  /// Sync all local sensor readings to Firebase
  Future<int> syncSensorReadings(String userId) async {
    try {
      _isSyncing = true;
      notifyListeners();

      final localReadings = _storageService.getAllReadings();
      int syncedCount = 0;

      for (final reading in localReadings) {
        try {
          await _firestoreService.saveSensorReading(
            userId: userId,
            reading: reading,
          );
          syncedCount++;
        } catch (e) {
          debugPrint('Sync Individual Reading Error: $e');
          // Continue syncing other readings even if one fails
        }
      }

      _lastSyncTime = DateTime.now();
      _isSyncing = false;
      notifyListeners();

      return syncedCount;
    } catch (e) {
      debugPrint('Sync Sensor Readings Error: $e');
      _isSyncing = false;
      notifyListeners();
      return 0;
    }
  }

  /// Pull sensor readings from Firebase (for recovery/sync)
  Future<List<SensorReading>> pullSensorReadings(String userId) async {
    try {
      return await _firestoreService.getSensorReadings(userId: userId);
    } catch (e) {
      debugPrint('Pull Sensor Readings Error: $e');
      return [];
    }
  }

  // ============== Risk Scores Sync ==============

  /// Sync risk score to Firebase
  Future<bool> syncRiskScore({
    required String userId,
    required RiskScore riskScore,
  }) async {
    try {
      await _firestoreService.saveRiskScore(
        userId: userId,
        riskScore: riskScore,
      );
      return true;
    } catch (e) {
      debugPrint('Sync Risk Score Error: $e');
      return false;
    }
  }

  /// Pull latest risk score from Firebase
  Future<RiskScore?> pullLatestRiskScore(String userId) async {
    try {
      return await _firestoreService.getLatestRiskScore(userId);
    } catch (e) {
      debugPrint('Pull Latest Risk Score Error: $e');
      return null;
    }
  }

  // ============== Alerts Sync ==============

  /// Sync alert to Firebase
  Future<bool> syncAlert({
    required String userId,
    required Alert alert,
  }) async {
    try {
      await _firestoreService.saveAlert(
        userId: userId,
        alert: alert,
      );
      return true;
    } catch (e) {
      debugPrint('Sync Alert Error: $e');
      return false;
    }
  }

  /// Pull alerts from Firebase
  Future<List<Alert>> pullAlerts(String userId) async {
    try {
      return await _firestoreService.getAlerts(userId: userId);
    } catch (e) {
      debugPrint('Pull Alerts Error: $e');
      return [];
    }
  }

  // ============== Complete Sync ==============

  /// Perform complete sync of all data
  Future<Map<String, dynamic>> performCompleteSync(String userId) async {
    if (_isSyncing) {
      debugPrint('Sync already in progress');
      return {'success': false, 'message': 'Sync already in progress'};
    }

    try {
      _isSyncing = true;
      notifyListeners();

      final results = <String, dynamic>{};

      // Sync user profile
      try {
        final profile = _storageService.getUserProfile();
        if (profile != null) {
          await syncUserProfile(userId: userId, profile: profile);
          results['profile'] = true;
        }
      } catch (e) {
        results['profile'] = false;
        debugPrint('Profile Sync Failed: $e');
      }

      // Sync sensor readings
      try {
        final syncedCount = await syncSensorReadings(userId);
        results['sensor_readings'] = syncedCount;
      } catch (e) {
        results['sensor_readings'] = 0;
        debugPrint('Sensor Readings Sync Failed: $e');
      }

      // Sync alerts
      try {
        final alerts = _storageService.getAllAlerts();
        int syncedAlerts = 0;
        for (final alert in alerts) {
          if (await syncAlert(userId: userId, alert: alert)) {
            syncedAlerts++;
          }
        }
        results['alerts'] = syncedAlerts;
      } catch (e) {
        results['alerts'] = 0;
        debugPrint('Alerts Sync Failed: $e');
      }

      results['success'] = true;
      results['timestamp'] = DateTime.now();
      _lastSyncTime = DateTime.now();

      _isSyncing = false;
      notifyListeners();

      return results;
    } catch (e) {
      debugPrint('Complete Sync Error: $e');
      _isSyncing = false;
      notifyListeners();
      return {'success': false, 'message': e.toString()};
    }
  }

  // ============== Sync Status ==============

  /// Get last sync time formatted
  String getLastSyncTimeFormatted() {
    if (_lastSyncTime == null) return 'Never';

    final now = DateTime.now();
    final diff = now.difference(_lastSyncTime!);

    if (diff.inSeconds < 60) {
      return 'Just now';
    } else if (diff.inMinutes < 60) {
      return '${diff.inMinutes} minutes ago';
    } else if (diff.inHours < 24) {
      return '${diff.inHours} hours ago';
    } else {
      return '${diff.inDays} days ago';
    }
  }

  /// Check if sync is needed (optional)
  bool shouldSync() {
    if (_lastSyncTime == null) return true;

    final now = DateTime.now();
    final diff = now.difference(_lastSyncTime!);

    // Sync if more than 1 hour has passed
    return diff.inHours >= 1;
  }

  // ============== Observable State ==============

  final List<VoidCallback> _listeners = [];

  void addListener(VoidCallback listener) {
    _listeners.add(listener);
  }

  void removeListener(VoidCallback listener) {
    _listeners.remove(listener);
  }

  void notifyListeners() {
    for (final listener in _listeners) {
      listener();
    }
  }

  void dispose() {
    _listeners.clear();
  }
}
