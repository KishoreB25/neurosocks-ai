import 'package:flutter/material.dart';
import '../../../core/constants/app_colors.dart';
import '../../../core/constants/app_strings.dart';

/// Landing screen with Sign In and Sign Up options
class LandingScreen extends StatelessWidget {
  const LandingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 40.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  children: [
                    // App Logo/Icon
                    Container(
                      height: 120,
                      width: 120,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient: LinearGradient(
                          colors: [
                            AppColors.primary,
                            AppColors.primary.withOpacity(0.7),
                          ],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                      ),
                      child: const Icon(
                        Icons.favorite,
                        color: Colors.white,
                        size: 60,
                      ),
                    ),
                    const SizedBox(height: 40),

                    // App Title
                    Text(
                      AppStrings.appName,
                      style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: AppColors.textPrimary,
                          ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),

                    // Subtitle
                    Text(
                      'Monitor your foot health with smart socks',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                            color: AppColors.textSecondary,
                            fontSize: 16,
                          ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 60),

                    // Feature highlights
                    _FeatureHighlight(
                      icon: Icons.sensors,
                      title: 'Real-time Monitoring',
                      description: 'Track temperature and pressure changes',
                    ),
                    const SizedBox(height: 24),
                    _FeatureHighlight(
                      icon: Icons.notifications_active,
                      title: 'Smart Alerts',
                      description: 'Get notified of potential issues early',
                    ),
                    const SizedBox(height: 24),
                    _FeatureHighlight(
                      icon: Icons.cloud_sync,
                      title: 'Cloud Sync',
                      description: 'Your data synced securely across devices',
                    ),
                  ],
                ),
                const SizedBox(height: 60),

                // Sign In / Sign Up buttons
                Column(
                  children: [
                    // Sign Up button (primary)
                    ElevatedButton(
                      onPressed: () {
                        Navigator.of(context).pushNamed('/sign-up');
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppColors.primary,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 48,
                          vertical: 16,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        minimumSize: const Size(double.infinity, 56),
                      ),
                      child: const Text(
                        'Create Account',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),

                    // Sign In button (secondary)
                    OutlinedButton(
                      onPressed: () {
                        Navigator.of(context).pushNamed('/sign-in');
                      },
                      style: OutlinedButton.styleFrom(
                        foregroundColor: AppColors.primary,
                        side: const BorderSide(
                          color: AppColors.primary,
                          width: 2,
                        ),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 48,
                          vertical: 16,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        minimumSize: const Size(double.infinity, 56),
                      ),
                      child: const Text(
                        'Sign In',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _FeatureHighlight extends StatelessWidget {
  final IconData icon;
  final String title;
  final String description;

  const _FeatureHighlight({
    required this.icon,
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          height: 56,
          width: 56,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: AppColors.primary.withOpacity(0.1),
          ),
          child: Icon(
            icon,
            color: AppColors.primary,
            size: 28,
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                      color: AppColors.textPrimary,
                    ),
              ),
              const SizedBox(height: 4),
              Text(
                description,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: AppColors.textSecondary,
                    ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
