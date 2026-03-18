import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart';
import 'firebase_firestore_service.dart';

/// Firebase Cloud Messaging Service
/// Handles push notifications for alerts and updates
class FirebaseMessagingService {
  static final FirebaseMessagingService _instance =
      FirebaseMessagingService._internal();
  factory FirebaseMessagingService() => _instance;
  FirebaseMessagingService._internal();

  final FirebaseMessaging _messaging = FirebaseMessaging.instance;
  final FirebaseFirestoreService _firestoreService = FirebaseFirestoreService();

  bool _isInitialized = false;
  String? _currentUserId;

  // ============== Initialization ==============

  /// Set current user ID for token saving
  void setCurrentUserId(String userId) {
    _currentUserId = userId;
  }

  /// Initialize Firebase Cloud Messaging
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Request notification permissions
      await _requestPermissions();

      // Get FCM token for this device
      final token = await getToken();
      debugPrint('FCM Token: $token');

      // Save token to Firestore if user is logged in
      if (_currentUserId != null && token != null) {
        await _firestoreService.saveFCMToken(
          userId: _currentUserId!,
          fcmToken: token,
        );
      }

      // Handle foreground messages
      FirebaseMessaging.onMessage.listen(_handleForegroundMessage);

      // Handle background message tap
      FirebaseMessaging.onMessageOpenedApp.listen(_handleMessageOpenedApp);

      // Handle initial message (app launched via notification)
      final initialMessage = await _messaging.getInitialMessage();
      if (initialMessage != null) {
        _handleMessageOpenedApp(initialMessage);
      }

      // Listen to token refresh and save new token
      _messaging.onTokenRefresh.listen((token) async {
        if (_currentUserId != null) {
          await _firestoreService.saveFCMToken(
            userId: _currentUserId!,
            fcmToken: token,
          );
        }
      });

      _isInitialized = true;
    } catch (e) {
      debugPrint('FCM Initialization Error: $e');
    }
  }

  // ============== Permissions ==============

  /// Request notification permissions
  Future<NotificationSettings> _requestPermissions() async {
    return await _messaging.requestPermission(
      alert: true,
      announcement: true,
      badge: true,
      criticalAlert: true,
      provisional: false,
      sound: true,
    );
  }

  // ============== Token Management ==============

  /// Get FCM token for this device
  Future<String?> getToken() async {
    try {
      return await _messaging.getToken();
    } catch (e) {
      debugPrint('Get FCM Token Error: $e');
      return null;
    }
  }

  /// Get APNs token (iOS only)
  Future<String?> getAPNSToken() async {
    try {
      return await _messaging.getAPNSToken();
    } catch (e) {
      debugPrint('Get APNS Token Error: $e');
      return null;
    }
  }

  /// Listen to token refresh events
  Stream<String> get tokenStream => _messaging.onTokenRefresh;

  /// Delete FCM token
  Future<void> deleteToken() async {
    try {
      await _messaging.deleteToken();
    } catch (e) {
      debugPrint('Delete FCM Token Error: $e');
    }
  }

  // ============== Message Handlers ==============

  /// Handle foreground messages (app in focus)
  void _handleForegroundMessage(RemoteMessage message) {
    debugPrint('Foreground Message:');
    debugPrint('Title: ${message.notification?.title}');
    debugPrint('Body: ${message.notification?.body}');
    debugPrint('Data: ${message.data}');

    // You can update local state, show a custom overlay, etc.
    // This is where you'd notify your app about the incoming alert
  }

  /// Handle message opened (notification tapped)
  void _handleMessageOpenedApp(RemoteMessage message) {
    debugPrint('Message Opened App:');
    debugPrint('Title: ${message.notification?.title}');
    debugPrint('Body: ${message.notification?.body}');
    debugPrint('Data: ${message.data}');

    // Navigate to relevant screen based on notification data
    // Example: if (message.data['type'] == 'alert') { navigate to alerts screen }
  }

  // ============== Background Handler ==============

  /// Set background message handler (call from main.dart)
  /// Must be a top-level function
  /// ```
  /// static Future<void> handleBackgroundMessage(RemoteMessage message) async {
  ///   debugPrint('Background Message: ${message.messageId}');
  /// }
  /// ```

  // ============== Notification Settings ==============

  /// Check if notifications are enabled
  Future<NotificationSettings> getNotificationSettings() async {
    return await _messaging.getNotificationSettings();
  }

  /// Open app notification settings
  Future<void> openNotificationSettings() async {
    // Note: Platform-specific notification settings - handled by OS
    // This would be implemented in platform channels if needed
  }

  /// Subscribe to topic for targeted notifications
  Future<void> subscribeToTopic(String topic) async {
    try {
      await _messaging.subscribeToTopic(topic);
      debugPrint('Subscribed to topic: $topic');
    } catch (e) {
      debugPrint('Subscribe to Topic Error: $e');
    }
  }

  /// Unsubscribe from topic
  Future<void> unsubscribeFromTopic(String topic) async {
    try {
      await _messaging.unsubscribeFromTopic(topic);
      debugPrint('Unsubscribed from topic: $topic');
    } catch (e) {
      debugPrint('Unsubscribe from Topic Error: $e');
    }
  }

  // ============== Notification Helper ==============

  /// Subscribe to alert notifications for specific user
  Future<void> subscribeToUserAlerts(String userId) async {
    await subscribeToTopic('alerts_$userId');
  }

  /// Unsubscribe from alert notifications
  Future<void> unsubscribeFromUserAlerts(String userId) async {
    await unsubscribeFromTopic('alerts_$userId');
  }

  /// Subscribe to critical risk alerts
  Future<void> subscribeToCriticalAlerts(String userId) async {
    await subscribeToTopic('critical_$userId');
  }
}

/// Background message handler (must be top-level function)
Future<void> firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  debugPrint('Background Message Handler: ${message.messageId}');
  // Handle background message
  // Note: Limited by platform restrictions
}
