import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';

/// Firebase Authentication Service
/// Handles user sign up, login, logout, password reset
class FirebaseAuthService {
  static final FirebaseAuthService _instance = FirebaseAuthService._internal();
  factory FirebaseAuthService() => _instance;
  FirebaseAuthService._internal();

  final FirebaseAuth _auth = FirebaseAuth.instance;

  // Current user
  User? get currentUser => _auth.currentUser;
  String? get currentUserId => _auth.currentUser?.uid;
  String? get currentUserEmail => _auth.currentUser?.email;
  bool get isLoggedIn => _auth.currentUser != null;

  // User state stream
  Stream<User?> get authStateChanges => _auth.authStateChanges();

  // ============== Sign Up ==============

  /// Register new user with email and password
  Future<User?> signUp({
    required String email,
    required String password,
  }) async {
    try {
      final userCredential = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );
      return userCredential.user;
    } on FirebaseAuthException catch (e) {
      debugPrint('SignUp Error: ${e.code} - ${e.message}');
      rethrow;
    } catch (e) {
      debugPrint('SignUp Error: $e');
      rethrow;
    }
  }

  // ============== Login ==============

  /// Login user with email and password
  Future<User?> login({
    required String email,
    required String password,
  }) async {
    try {
      final userCredential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      return userCredential.user;
    } on FirebaseAuthException catch (e) {
      debugPrint('Login Error: ${e.code} - ${e.message}');
      rethrow;
    } catch (e) {
      debugPrint('Login Error: $e');
      rethrow;
    }
  }

  // ============== Logout ==============

  /// Logout current user
  Future<void> logout() async {
    try {
      await _auth.signOut();
    } catch (e) {
      debugPrint('Logout Error: $e');
      rethrow;
    }
  }

  // ============== Password Management ==============

  /// Send password reset email
  Future<void> sendPasswordResetEmail(String email) async {
    try {
      await _auth.sendPasswordResetEmail(email: email);
    } on FirebaseAuthException catch (e) {
      debugPrint('Password Reset Error: ${e.code} - ${e.message}');
      rethrow;
    } catch (e) {
      debugPrint('Password Reset Error: $e');
      rethrow;
    }
  }

  /// Update user password
  Future<void> updatePassword(String newPassword) async {
    try {
      await _auth.currentUser?.updatePassword(newPassword);
    } on FirebaseAuthException catch (e) {
      debugPrint('Update Password Error: ${e.code} - ${e.message}');
      rethrow;
    } catch (e) {
      debugPrint('Update Password Error: $e');
      rethrow;
    }
  }

  // ============== User Info ==============

  /// Update user email
  Future<void> updateEmail(String newEmail) async {
    try {
      await _auth.currentUser?.verifyBeforeUpdateEmail(newEmail);
    } on FirebaseAuthException catch (e) {
      debugPrint('Update Email Error: ${e.code} - ${e.message}');
      rethrow;
    } catch (e) {
      debugPrint('Update Email Error: $e');
      rethrow;
    }
  }

  /// Delete user account
  Future<void> deleteAccount() async {
    try {
      await _auth.currentUser?.delete();
    } on FirebaseAuthException catch (e) {
      debugPrint('Delete Account Error: ${e.code} - ${e.message}');
      rethrow;
    } catch (e) {
      debugPrint('Delete Account Error: $e');
      rethrow;
    }
  }

  /// Get ID token for authenticated API calls
  Future<String?> getIdToken() async {
    try {
      return await _auth.currentUser?.getIdToken();
    } catch (e) {
      debugPrint('Get ID Token Error: $e');
      return null;
    }
  }

  /// Verify email
  Future<void> sendEmailVerification() async {
    try {
      await _auth.currentUser?.sendEmailVerification();
    } catch (e) {
      debugPrint('Send Email Verification Error: $e');
      rethrow;
    }
  }

  bool get isEmailVerified => _auth.currentUser?.emailVerified ?? false;
}
