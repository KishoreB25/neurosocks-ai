import 'package:flutter/foundation.dart';
import '../../data/services/firebase/firebase_messaging_service.dart';

/// Firebase Notifications Provider
/// Manages push notification state and FCM tokens
class FirebaseNotificationsProvider extends ChangeNotifier {
  final FirebaseMessagingService _messagingService = FirebaseMessagingService();

  // State
  bool _isInitialized = false;
  String? _fcmToken;
  bool _notificationsEnabled = true;
  final List<String> _subscribedTopics = [];
  String? _pendingNotificationData;

  // Getters
  bool get isInitialized => _isInitialized;
  String? get fcmToken => _fcmToken;
  bool get notificationsEnabled => _notificationsEnabled;
  List<String> get subscribedTopics => _subscribedTopics;
  String? get pendingNotificationData => _pendingNotificationData;

  // ============== Initialization ==============

  /// Initialize Firebase Cloud Messaging
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      await _messagingService.initialize();

      // Get FCM token
      _fcmToken = await _messagingService.getToken();

      // Listen to token refresh
      _messagingService.tokenStream.listen((token) {
        _fcmToken = token;
        notifyListeners();
      });

      _isInitialized = true;
      notifyListeners();
    } catch (e) {
      debugPrint('FCM Initialization Error: $e');
      _isInitialized = false;
      notifyListeners();
    }
  }

  // ============== Topic Management ==============

  /// Subscribe to topic
  Future<void> subscribeToTopic(String topic) async {
    try {
      await _messagingService.subscribeToTopic(topic);
      if (!_subscribedTopics.contains(topic)) {
        _subscribedTopics.add(topic);
      }
      notifyListeners();
    } catch (e) {
      debugPrint('Subscribe to Topic Error: $e');
    }
  }

  /// Unsubscribe from topic
  Future<void> unsubscribeFromTopic(String topic) async {
    try {
      await _messagingService.unsubscribeFromTopic(topic);
      _subscribedTopics.remove(topic);
      notifyListeners();
    } catch (e) {
      debugPrint('Unsubscribe from Topic Error: $e');
    }
  }

  /// Subscribe to user alerts
  Future<void> subscribeToUserAlerts(String userId) async {
    await subscribeToTopic('alerts_$userId');
  }

  /// Unsubscribe from user alerts
  Future<void> unsubscribeFromUserAlerts(String userId) async {
    await unsubscribeFromTopic('alerts_$userId');
  }

  /// Subscribe to critical alerts
  Future<void> subscribeToCriticalAlerts(String userId) async {
    await subscribeToTopic('critical_$userId');
  }

  /// Unsubscribe from critical alerts
  Future<void> unsubscribeFromCriticalAlerts(String userId) async {
    await unsubscribeFromTopic('critical_$userId');
  }

  // ============== Notifications Settings ==============

  /// Enable/disable notifications
  void setNotificationsEnabled(bool enabled) {
    _notificationsEnabled = enabled;
    notifyListeners();
  }

  /// Open notification settings
  Future<void> openNotificationSettings() async {
    await _messagingService.openNotificationSettings();
  }

  // ============== Token Management ==============

  /// Get FCM token
  Future<String?> getFcmToken() async {
    if (_fcmToken != null) return _fcmToken;
    _fcmToken = await _messagingService.getToken();
    notifyListeners();
    return _fcmToken;
  }

  /// Delete FCM token
  Future<void> deleteFcmToken() async {
    await _messagingService.deleteToken();
    _fcmToken = null;
    notifyListeners();
  }

  // ============== Notification Data ==============

  /// Set pending notification data
  void setPendingNotificationData(String? data) {
    _pendingNotificationData = data;
    notifyListeners();
  }

  /// Clear pending notification data
  void clearPendingNotificationData() {
    _pendingNotificationData = null;
    notifyListeners();
  }

  // ============== Status ==============

  /// Get notification status
  String getNotificationStatus() {
    if (!_isInitialized) return 'Notifications not initialized';
    if (!_notificationsEnabled) return 'Notifications disabled';
    if (_fcmToken == null) return 'No device token';
    return 'Notifications enabled';
  }

  /// Check if subscribed to topic
  bool isSubscribedToTopic(String topic) {
    return _subscribedTopics.contains(topic);
  }
}
