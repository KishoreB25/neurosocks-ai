import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter/gestures.dart';
import '../../../core/constants/app_colors.dart';
import '../../../data/models/user_profile.dart';
import '../../../providers/firebase/firebase_auth_provider.dart';
import '../../../providers/user_provider.dart';

/// Sign Up screen with complete profile setup
class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  late final TextEditingController _nameController;
  late final TextEditingController _ageController;
  late final TextEditingController _diabetesYearsController;
  late final TextEditingController _emailController;
  late final TextEditingController _passwordController;
  late final TextEditingController _confirmPasswordController;

  late final FocusNode _nameFocus;
  late final FocusNode _ageFocus;
  late final FocusNode _diabetesYearsFocus;
  late final FocusNode _emailFocus;
  late final FocusNode _passwordFocus;
  late final FocusNode _confirmPasswordFocus;

  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;
  bool _isLoading = false;

  // Health info
  DiabetesType _diabetesType = DiabetesType.type1;
  int _diabetesYears = 0;
  bool _hasNeuropathy = false;
  bool _hasPAD = false;
  bool _hasPreviousUlcer = false;
  bool _hasHypertension = false;

  // Page tracking
  int _currentPage = 0;
  final int _totalPages = 2;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController();
    _ageController = TextEditingController();
    _diabetesYearsController = TextEditingController();
    _emailController = TextEditingController();
    _passwordController = TextEditingController();
    _confirmPasswordController = TextEditingController();

    _nameFocus = FocusNode();
    _ageFocus = FocusNode();
    _diabetesYearsFocus = FocusNode();
    _emailFocus = FocusNode();
    _passwordFocus = FocusNode();
    _confirmPasswordFocus = FocusNode();
  }

  @override
  void dispose() {
    _nameController.dispose();
    _ageController.dispose();
    _diabetesYearsController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();

    _nameFocus.dispose();
    _ageFocus.dispose();
    _diabetesYearsFocus.dispose();
    _emailFocus.dispose();
    _passwordFocus.dispose();
    _confirmPasswordFocus.dispose();
    super.dispose();
  }

  bool _validatePage1() {
    if (_nameController.text.trim().isEmpty) {
      _showErrorSnackbar('Please enter your name');
      return false;
    }
    if (_ageController.text.trim().isEmpty) {
      _showErrorSnackbar('Please enter your age');
      return false;
    }
    final age = int.tryParse(_ageController.text.trim());
    if (age == null || age < 10 || age > 120) {
      _showErrorSnackbar('Please enter a valid age');
      return false;
    }
    return true;
  }

  bool _validatePage2() {
    if (_emailController.text.trim().isEmpty) {
      _showErrorSnackbar('Please enter your email');
      return false;
    }
    if (!_isValidEmail(_emailController.text.trim())) {
      _showErrorSnackbar('Please enter a valid email');
      return false;
    }
    if (_passwordController.text.isEmpty) {
      _showErrorSnackbar('Please enter a password');
      return false;
    }
    if (_passwordController.text.length < 6) {
      _showErrorSnackbar('Password must be at least 6 characters');
      return false;
    }
    if (_passwordController.text != _confirmPasswordController.text) {
      _showErrorSnackbar('Passwords do not match');
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

  Future<void> _handleSignUp() async {
    if (!_validatePage2()) return;

    setState(() => _isLoading = true);

    try {
      final authProvider = context.read<FirebaseAuthProvider>();

      final profile = UserProfile(
        id: '',
        email: _emailController.text.trim(),
        name: _nameController.text.trim(),
        age: int.parse(_ageController.text.trim()),
        diabetesType: _diabetesType,
        diabetesYears: _diabetesYears,
        healthInfo: HealthInfo(
          hasNeuropathy: _hasNeuropathy,
          hasPAD: _hasPAD,
          hasPreviousUlcer: _hasPreviousUlcer,
          hasHypertension: _hasHypertension,
        ),
        settings: UserSettings(
          temperatureUnit: TemperatureUnit.celsius,
          notificationsEnabled: true,
          criticalAlertsEnabled: true,
        ),
        createdAt: DateTime.now(),
        updatedAt: DateTime.now(),
      );

      final success = await authProvider.signUpWithEmail(
        email: _emailController.text.trim(),
        password: _passwordController.text,
        profile: profile,
      );

      if (success && mounted) {
        // Update local UserProvider with the created profile
        final userProvider = context.read<UserProvider>();
        final currentUser = authProvider.currentUser;
        if (currentUser != null) {
          final profileWithId = profile.copyWith(id: currentUser.uid);
          await userProvider.saveProfile(profileWithId);
        }

        if (mounted) {
          Navigator.of(context).pushNamedAndRemoveUntil(
            '/dashboard',
            (route) => false,
          );
        }
      } else if (mounted) {
        _showErrorSnackbar(authProvider.errorMessage ?? 'Sign up failed');
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.transparent,
        leading: _currentPage > 0
            ? IconButton(
                icon: const Icon(Icons.arrow_back, color: AppColors.textPrimary),
                onPressed: () => setState(() => _currentPage--),
              )
            : IconButton(
                icon: const Icon(Icons.arrow_back, color: AppColors.textPrimary),
                onPressed: () => Navigator.of(context).pop(),
              ),
        title: Text(
          'Step ${_currentPage + 1} of $_totalPages',
          style: const TextStyle(color: AppColors.textPrimary),
        ),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Progress indicator
            LinearProgressIndicator(
              value: (_currentPage + 1) / _totalPages,
              minHeight: 3,
              backgroundColor: Colors.grey[300],
              valueColor: const AlwaysStoppedAnimation<Color>(AppColors.primary),
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 32),
                child: _currentPage == 0 ? _buildPage1() : _buildPage2(),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPage1() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Title
        Text(
          'Personal Information',
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppColors.textPrimary,
              ),
        ),
        const SizedBox(height: 8),
        Text(
          'Tell us about yourself',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: AppColors.textSecondary,
              ),
        ),
        const SizedBox(height: 32),

        // Name field
        _buildTextField(
          controller: _nameController,
          focusNode: _nameFocus,
          label: 'Full Name',
          hint: 'Enter your full name',
          keyboardType: TextInputType.name,
          prefixIcon: Icons.person_outline,
          onFieldSubmitted: (_) {
            _nameFocus.unfocus();
            _ageFocus.requestFocus();
          },
        ),
        const SizedBox(height: 24),

        // Age field
        _buildTextField(
          controller: _ageController,
          focusNode: _ageFocus,
          label: 'Age',
          hint: 'Enter your age',
          keyboardType: TextInputType.number,
          prefixIcon: Icons.cake_outlined,
          onFieldSubmitted: (_) {
            _ageFocus.unfocus();
          },
        ),
        const SizedBox(height: 32),

        // Diabetes Type
        _buildDropdownField(
          label: 'Diabetes Type',
          value: _diabetesType,
          items: DiabetesType.values,
          itemBuilder: (type) => _getDiabetesTypeLabel(type),
          onChanged: (value) => setState(() => _diabetesType = value),
        ),
        const SizedBox(height: 24),

        // Diabetes Years
        _buildTextField(
          controller: _diabetesYearsController,
          focusNode: _diabetesYearsFocus,
          label: 'Years with Diabetes',
          hint: 'How many years',
          keyboardType: TextInputType.number,
          prefixIcon: Icons.schedule_outlined,
          onFieldSubmitted: (_) {
            if (_diabetesYearsController.text.isNotEmpty) {
              _diabetesYears = int.tryParse(_diabetesYearsController.text) ?? 0;
            }
          },
        ),
        const SizedBox(height: 48),

        // Next button
        ElevatedButton(
          onPressed: () {
            if (_validatePage1()) {
              if (_diabetesYearsController.text.isNotEmpty) {
                _diabetesYears = int.tryParse(_diabetesYearsController.text) ?? 0;
              }
              setState(() => _currentPage = 1);
            }
          },
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.primary,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            minimumSize: const Size(double.infinity, 56),
          ),
          child: const Text(
            'Next',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildPage2() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Title
        Text(
          'Health & Account',
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.bold,
                color: AppColors.textPrimary,
              ),
        ),
        const SizedBox(height: 8),
        Text(
          'Complete your setup',
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: AppColors.textSecondary,
              ),
        ),
        const SizedBox(height: 32),

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
          hint: 'Create a strong password',
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
            _confirmPasswordFocus.requestFocus();
          },
        ),
        const SizedBox(height: 24),

        // Confirm Password field
        _buildTextField(
          controller: _confirmPasswordController,
          focusNode: _confirmPasswordFocus,
          label: 'Confirm Password',
          hint: 'Re-enter your password',
          obscureText: _obscureConfirmPassword,
          prefixIcon: Icons.lock_outline,
          suffixIcon: _obscureConfirmPassword
              ? Icons.visibility_outlined
              : Icons.visibility_off_outlined,
          onSuffixIconPressed: () {
            setState(() =>
                _obscureConfirmPassword = !_obscureConfirmPassword);
          },
          onFieldSubmitted: (_) {
            _confirmPasswordFocus.unfocus();
          },
        ),
        const SizedBox(height: 32),

        // Health conditions
        Text(
          'Health Conditions',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
                color: AppColors.textPrimary,
              ),
        ),
        const SizedBox(height: 16),

        _buildCheckbox('Neuropathy', _hasNeuropathy,
            (value) => setState(() => _hasNeuropathy = value ?? false)),
        _buildCheckbox(
            'Peripheral Arterial Disease (PAD)',
            _hasPAD,
            (value) => setState(() => _hasPAD = value ?? false)),
        _buildCheckbox('Previous Foot Ulcer', _hasPreviousUlcer,
            (value) => setState(() => _hasPreviousUlcer = value ?? false)),
        _buildCheckbox('Hypertension', _hasHypertension,
            (value) => setState(() => _hasHypertension = value ?? false)),

        const SizedBox(height: 48),

        // Sign Up button
        ElevatedButton(
          onPressed: _isLoading ? null : _handleSignUp,
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
                    valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                  ),
                )
              : const Text(
                  'Create Account',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
        ),
        const SizedBox(height: 16),

        // Sign In link
        Center(
          child: RichText(
            text: TextSpan(
              text: 'Already have an account? ',
              style: TextStyle(
                color: AppColors.textSecondary,
                fontSize: 14,
              ),
              children: [
                TextSpan(
                  text: 'Sign In',
                  style: const TextStyle(
                    color: AppColors.primary,
                    fontWeight: FontWeight.w600,
                  ),
                  recognizer: TapGestureRecognizer()
                    ..onTap = () {
                      Navigator.of(context).pushReplacementNamed('/sign-in');
                    },
                ),
              ],
            ),
          ),
        ),
      ],
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

  Widget _buildDropdownField<T>({
    required String label,
    required T value,
    required List<T> items,
    required String Function(T) itemBuilder,
    required Function(T) onChanged,
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
        DropdownButtonFormField<T>(
          value: value,
          items: items
              .map((item) => DropdownMenuItem(
                    value: item,
                    child: Text(itemBuilder(item)),
                  ))
              .toList(),
          onChanged: (newValue) {
            if (newValue != null) {
              onChanged(newValue);
            }
          },
          decoration: InputDecoration(
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
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
        ),
      ],
    );
  }

  Widget _buildCheckbox(
    String label,
    bool value,
    Function(bool?) onChanged,
  ) {
    return CheckboxListTile(
      title: Text(
        label,
        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: AppColors.textPrimary,
            ),
      ),
      value: value,
      onChanged: onChanged,
      controlAffinity: ListTileControlAffinity.leading,
      contentPadding: EdgeInsets.zero,
      activeColor: AppColors.primary,
    );
  }

  String _getDiabetesTypeLabel(DiabetesType type) {
    switch (type) {
      case DiabetesType.none:
        return 'None';
      case DiabetesType.type1:
        return 'Type 1';
      case DiabetesType.type2:
        return 'Type 2';
      case DiabetesType.preDiabetes:
        return 'Pre-Diabetes';
      case DiabetesType.gestational:
        return 'Gestational';
    }
  }
}
