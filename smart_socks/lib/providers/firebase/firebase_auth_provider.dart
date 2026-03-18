import 'package:flutter/foundation.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../../data/models/user_profile.dart';
import '../../data/services/firebase/firebase_auth_service.dart';
import '../../data/services/firebase/firebase_firestore_service.dart';

/// Firebase Authentication Provider
/// Manages authentication state using Provider pattern
class FirebaseAuthProvider extends ChangeNotifier {
  final FirebaseAuthService _authService = FirebaseAuthService();
  final FirebaseFirestoreService _firestoreService = FirebaseFirestoreService();

  // State
  User? _currentUser;
  bool _isLoading = false;
  String? _errorMessage;
  bool _isEmailVerified = false;

  // Getters
  User? get currentUser => _currentUser;
  String? get currentUserId => _currentUser?.uid;
  String? get currentUserEmail => _currentUser?.email;
  bool get isLoggedIn => _currentUser != null;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  bool get isEmailVerified => _isEmailVerified;

  FirebaseAuthProvider() {
    _initializeAuthState();
  }

  // ============== Initialization ==============

  /// Initialize authentication state
  void _initializeAuthState() {
    _authService.authStateChanges.listen((user) {
      _currentUser = user;
      _isEmailVerified = user?.emailVerified ?? false;
      notifyListeners();
    });
  }

  // ============== Sign Up ==============

  /// Sign up with email and password
  Future<bool> signUp({
    required String email,
    required String password,
  }) async {
    try {
      _isLoading = true;
      _errorMessage = null;
      notifyListeners();

      final user = await _authService.signUp(
        email: email,
        password: password,
      );

      _currentUser = user;
      _isLoading = false;
      notifyListeners();

      return user != null;
    } on FirebaseAuthException catch (e) {
      _errorMessage = _getErrorMessage(e.code);
      _isLoading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _errorMessage = 'An unexpected error occurred';
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  // ============== Login ==============

  /// Login with email and password
  Future<bool> login({
    required String email,
    required String password,
  }) async {
    try {
      _isLoading = true;
      _errorMessage = null;
      notifyListeners();

      final user = await _authService.login(
        email: email,
        password: password,
      );

      _currentUser = user;
      _isLoading = false;
      notifyListeners();

      return user != null;
    } on FirebaseAuthException catch (e) {
      _errorMessage = _getErrorMessage(e.code);
      _isLoading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _errorMessage = 'An unexpected error occurred';
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  /// Sign in with email and password (alias for login)
  Future<bool> signInWithEmail({
    required String email,
    required String password,
  }) async {
    return login(email: email, password: password);
  }

  /// Sign up with email, password, and user profile
  Future<bool> signUpWithEmail({
    required String email,
    required String password,
    required UserProfile profile,
  }) async {
    try {
      _isLoading = true;
      _errorMessage = null;
      notifyListeners();

      // Create user account
      final user = await _authService.signUp(
        email: email,
        password: password,
      );

      if (user == null) {
        _errorMessage = 'Failed to create account';
        _isLoading = false;
        notifyListeners();
        return false;
      }

      // Save profile to Firestore
      final updatedProfile = profile.copyWith(id: user.uid);
      await _firestoreService.saveUserProfile(
        userId: user.uid,
        profile: updatedProfile,
      );

      _currentUser = user;
      _isLoading = false;
      notifyListeners();

      return true;
    } on FirebaseAuthException catch (e) {
      _errorMessage = _getErrorMessage(e.code);
      _isLoading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _errorMessage = 'An unexpected error occurred: ${e.toString()}';
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  // ============== Logout ==============

  /// Logout current user
  Future<void> logout() async {
    try {
      await _authService.logout();
      _currentUser = null;
      _errorMessage = null;
      notifyListeners();
    } catch (e) {
      _errorMessage = 'Logout failed. Please try again.';
      notifyListeners();
    }
  }

  // ============== Password Management ==============

  /// Send password reset email
  Future<bool> sendPasswordResetEmail(String email) async {
    try {
      _isLoading = true;
      _errorMessage = null;
      notifyListeners();

      await _authService.sendPasswordResetEmail(email);

      _isLoading = false;
      notifyListeners();
      return true;
    } on FirebaseAuthException catch (e) {
      _errorMessage = _getErrorMessage(e.code);
      _isLoading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _errorMessage = 'Failed to send reset email';
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  /// Update user password
  Future<bool> updatePassword(String newPassword) async {
    try {
      _isLoading = true;
      _errorMessage = null;
      notifyListeners();

      await _authService.updatePassword(newPassword);

      _isLoading = false;
      notifyListeners();
      return true;
    } on FirebaseAuthException catch (e) {
      _errorMessage = _getErrorMessage(e.code);
      _isLoading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _errorMessage = 'Failed to update password';
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  // ============== Email Verification ==============

  /// Send email verification
  Future<void> sendEmailVerification() async {
    try {
      await _authService.sendEmailVerification();
    } catch (e) {
      _errorMessage = 'Failed to send verification email';
      notifyListeners();
    }
  }

  /// Check email verification status
  Future<void> checkEmailVerification() async {
    try {
      await _currentUser?.reload();
      _isEmailVerified = _currentUser?.emailVerified ?? false;
      notifyListeners();
    } catch (e) {
      _errorMessage = 'Failed to check email verification';
      notifyListeners();
    }
  }

  // ============== User Management ==============

  /// Update email
  Future<bool> updateEmail(String newEmail) async {
    try {
      _isLoading = true;
      _errorMessage = null;
      notifyListeners();

      await _authService.updateEmail(newEmail);
      await _currentUser?.reload();

      _isLoading = false;
      notifyListeners();
      return true;
    } on FirebaseAuthException catch (e) {
      _errorMessage = _getErrorMessage(e.code);
      _isLoading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _errorMessage = 'Failed to update email';
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  /// Delete account
  Future<bool> deleteAccount() async {
    try {
      _isLoading = true;
      _errorMessage = null;
      notifyListeners();

      await _authService.deleteAccount();
      _currentUser = null;

      _isLoading = false;
      notifyListeners();
      return true;
    } on FirebaseAuthException catch (e) {
      _errorMessage = _getErrorMessage(e.code);
      _isLoading = false;
      notifyListeners();
      return false;
    } catch (e) {
      _errorMessage = 'Failed to delete account';
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  // ============== Helper Methods ==============

  /// Convert Firebase error codes to user-friendly messages
  String _getErrorMessage(String code) {
    switch (code) {
      case 'weak-password':
        return 'Password is too weak. Use at least 6 characters.';
      case 'email-already-in-use':
        return 'This email is already registered.';
      case 'invalid-email':
        return 'Invalid email address.';
      case 'user-not-found':
        return 'No account found with this email.';
      case 'wrong-password':
        return 'Incorrect password.';
      case 'too-many-requests':
        return 'Too many login attempts. Try again later.';
      case 'network-request-failed':
        return 'Network error. Check your connection.';
      default:
        return 'An authentication error occurred.';
    }
  }

  /// Clear error message
  void clearError() {
    _errorMessage = null;
    notifyListeners();
  }
}
