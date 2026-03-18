import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter/gestures.dart';
import '../../../core/constants/app_colors.dart';
import '../../../providers/firebase/firebase_auth_provider.dart';
import '../../../providers/user_provider.dart';

/// Sign In screen with email and password
class SignInScreen extends StatefulWidget {
  const SignInScreen({super.key});

  @override
  State<SignInScreen> createState() => _SignInScreenState();
}

class _SignInScreenState extends State<SignInScreen> {
  late final TextEditingController _emailController;
  late final TextEditingController _passwordController;
  late final FocusNode _emailFocus;
  late final FocusNode _passwordFocus;
  bool _obscurePassword = true;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _emailController = TextEditingController();
    _passwordController = TextEditingController();
    _emailFocus = FocusNode();
    _passwordFocus = FocusNode();
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _emailFocus.dispose();
    _passwordFocus.dispose();
    super.dispose();
  }

  void _handleSignIn() async {
    if (!_validateForm()) return;

    setState(() => _isLoading = true);

    try {
      final authProvider = context.read<FirebaseAuthProvider>();
      final success = await authProvider.signInWithEmail(
        email: _emailController.text.trim(),
        password: _passwordController.text,
      );

      if (success && mounted) {
        // Sync profile from Firestore
        final userProvider = context.read<UserProvider>();
        if (authProvider.currentUserId != null) {
          await userProvider.syncFromFirestore(authProvider.currentUserId!);
        }

        // Navigate to dashboard
        if (mounted) {
          Navigator.of(context).pushNamedAndRemoveUntil(
            '/dashboard',
            (route) => false,
          );
        }
      } else if (mounted) {
        _showErrorSnackbar(authProvider.errorMessage ?? 'Sign in failed');
      }
    } catch (e) {
      if (mounted) {
        _showErrorSnackbar('An error occurred: ${e.toString()}');
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  bool _validateForm() {
    if (_emailController.text.trim().isEmpty) {
      _showErrorSnackbar('Please enter your email');
      return false;
    }
    if (!_isValidEmail(_emailController.text.trim())) {
      _showErrorSnackbar('Please enter a valid email');
      return false;
    }
    if (_passwordController.text.isEmpty) {
      _showErrorSnackbar('Please enter your password');
      return false;
    }
    if (_passwordController.text.length < 6) {
      _showErrorSnackbar('Password must be at least 6 characters');
      return false;
    }
    return true;
  }

  bool _isValidEmail(String email) {
    return RegExp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        .hasMatch(email);
  }

  void _showErrorSnackbar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 3),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.transparent,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppColors.textPrimary),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 40),

              // Title
              Text(
                'Welcome Back',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: AppColors.textPrimary,
                    ),
              ),
              const SizedBox(height: 8),

              // Subtitle
              Text(
                'Sign in to continue monitoring your health',
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      color: AppColors.textSecondary,
                    ),
              ),
              const SizedBox(height: 48),

              // Email field
              _buildTextField(
                controller: _emailController,
                focusNode: _emailFocus,
                label: 'Email',
                hint: 'Enter your email',
                keyboardType: TextInputType.emailAddress,
                prefixIcon: Icons.email_outlined,
                onFieldSubmitted: (_) {
                  _emailFocus.unfocus();
                  _passwordFocus.requestFocus();
                },
              ),
              const SizedBox(height: 24),

              // Password field
              _buildTextField(
                controller: _passwordController,
                focusNode: _passwordFocus,
                label: 'Password',
                hint: 'Enter your password',
                obscureText: _obscurePassword,
                prefixIcon: Icons.lock_outline,
                suffixIcon: _obscurePassword
                    ? Icons.visibility_outlined
                    : Icons.visibility_off_outlined,
                onSuffixIconPressed: () {
                  setState(() => _obscurePassword = !_obscurePassword);
                },
                onFieldSubmitted: (_) {
                  _passwordFocus.unfocus();
                  _handleSignIn();
                },
              ),
              const SizedBox(height: 32),

              // Sign In button
              ElevatedButton(
                onPressed: _isLoading ? null : _handleSignIn,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  minimumSize: const Size(double.infinity, 56),
                ),
                child: _isLoading
                    ? const SizedBox(
                        height: 24,
                        width: 24,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor:
                              AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      )
                    : const Text(
                        'Sign In',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
              ),
              const SizedBox(height: 24),

              // Sign Up link
              Center(
                child: RichText(
                  text: TextSpan(
                    text: "Don't have an account? ",
                    style: TextStyle(
                      color: AppColors.textSecondary,
                      fontSize: 14,
                    ),
                    children: [
                      TextSpan(
                        text: 'Sign Up',
                        style: const TextStyle(
                          color: AppColors.primary,
                          fontWeight: FontWeight.w600,
                        ),
                        recognizer: TapGestureRecognizer()
                          ..onTap = () {
                            Navigator.of(context)
                                .pushReplacementNamed('/sign-up');
                          },
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required FocusNode focusNode,
    required String label,
    required String hint,
    TextInputType keyboardType = TextInputType.text,
    bool obscureText = false,
    IconData? prefixIcon,
    IconData? suffixIcon,
    VoidCallback? onSuffixIconPressed,
    Function(String)? onFieldSubmitted,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.labelLarge?.copyWith(
                color: AppColors.textPrimary,
                fontWeight: FontWeight.w600,
              ),
        ),
        const SizedBox(height: 8),
        TextField(
          controller: controller,
          focusNode: focusNode,
          keyboardType: keyboardType,
          obscureText: obscureText,
          decoration: InputDecoration(
            hintText: hint,
            prefixIcon: prefixIcon != null ? Icon(prefixIcon) : null,
            suffixIcon: suffixIcon != null
                ? IconButton(
                    icon: Icon(suffixIcon),
                    onPressed: onSuffixIconPressed,
                  )
                : null,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(
                color: Colors.grey,
                width: 1,
              ),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(
                color: AppColors.primary,
                width: 2,
              ),
            ),
            contentPadding: const EdgeInsets.symmetric(
              horizontal: 16,
              vertical: 14,
            ),
          ),
          onSubmitted: onFieldSubmitted,
        ),
      ],
    );
  }
}
