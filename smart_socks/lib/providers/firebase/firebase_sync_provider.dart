import 'package:flutter/foundation.dart';
import '../../data/services/firebase/firebase_sync_service.dart';

/// Firebase Sync Provider
/// Manages cloud-local data synchronization state
class FirebaseSyncProvider extends ChangeNotifier {
  final FirebaseSyncService _syncService = FirebaseSyncService();

  // State
  bool _isSyncing = false;
  DateTime? _lastSyncTime;
  String? _syncMessage;
  bool _autoSyncEnabled = true;

  // Getters
  bool get isSyncing => _isSyncing;
  DateTime? get lastSyncTime => _lastSyncTime;
  String? get syncMessage => _syncMessage;
  bool get autoSyncEnabled => _autoSyncEnabled;
  String get lastSyncTimeFormatted => _syncService.getLastSyncTimeFormatted();

  FirebaseSyncProvider() {
    _syncService.addListener(_onSyncStateChanged);
  }

  // ============== Listeners ==============

  /// Called when sync state changes
  void _onSyncStateChanged() {
    _isSyncing = _syncService.isSyncing;
    _lastSyncTime = _syncService.lastSyncTime;
    notifyListeners();
  }

  // ============== Sync Operations ==============

  /// Sync user profile
  Future<bool> syncUserProfile({
    required String userId,
    required dynamic profile,
  }) async {
    try {
      _isSyncing = true;
      _syncMessage = 'Syncing profile...';
      notifyListeners();

      final success = await _syncService.syncUserProfile(
        userId: userId,
        profile: profile,
      );

      _isSyncing = false;
      _syncMessage = success ? 'Profile synced' : 'Failed to sync profile';
      _lastSyncTime = DateTime.now();
      notifyListeners();

      return success;
    } catch (e) {
      _isSyncing = false;
      _syncMessage = 'Sync error: $e';
      notifyListeners();
      return false;
    }
  }

  /// Sync sensor readings
  Future<int> syncSensorReadings(String userId) async {
    try {
      _isSyncing = true;
      _syncMessage = 'Syncing sensor data...';
      notifyListeners();

      final count = await _syncService.syncSensorReadings(userId);

      _isSyncing = false;
      _syncMessage = 'Synced $count sensor readings';
      _lastSyncTime = DateTime.now();
      notifyListeners();

      return count;
    } catch (e) {
      _isSyncing = false;
      _syncMessage = 'Sync error: $e';
      notifyListeners();
      return 0;
    }
  }

  /// Sync alerts
  Future<bool> syncAlert({
    required String userId,
    required dynamic alert,
  }) async {
    try {
      final success = await _syncService.syncAlert(
        userId: userId,
        alert: alert,
      );

      if (success) {
        _lastSyncTime = DateTime.now();
      }

      notifyListeners();
      return success;
    } catch (e) {
      _syncMessage = 'Failed to sync alert: $e';
      notifyListeners();
      return false;
    }
  }

  /// Perform complete sync
  Future<Map<String, dynamic>> performCompleteSync(String userId) async {
    try {
      _isSyncing = true;
      _syncMessage = 'Syncing all data...';
      notifyListeners();

      final results = await _syncService.performCompleteSync(userId);

      _isSyncing = false;
      if (results['success'] == true) {
        _lastSyncTime = DateTime.now();
        _syncMessage = 'All data synced successfully';
      } else {
        _syncMessage = 'Sync completed with errors';
      }

      notifyListeners();
      return results;
    } catch (e) {
      _isSyncing = false;
      _syncMessage = 'Sync failed: $e';
      notifyListeners();
      return {'success': false, 'message': e.toString()};
    }
  }

  // ============== Auto Sync ==============

  /// Enable/disable auto sync
  void setAutoSync(bool enabled) {
    _autoSyncEnabled = enabled;
    notifyListeners();
  }

  /// Check if sync is needed
  bool shouldSync() {
    return _syncService.shouldSync();
  }

  // ============== Status ==============

  /// Get sync status message
  String getSyncStatus() {
    if (_isSyncing) {
      return _syncMessage ?? 'Syncing...';
    }
    return 'Last sync: $lastSyncTimeFormatted';
  }

  /// Clear sync message
  void clearSyncMessage() {
    _syncMessage = null;
    notifyListeners();
  }

  @override
  void dispose() {
    _syncService.removeListener(_onSyncStateChanged);
    _syncService.dispose();
    super.dispose();
  }
}
