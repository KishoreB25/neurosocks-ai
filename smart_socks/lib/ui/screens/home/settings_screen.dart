import 'package:flutter/material.dart' hide ConnectionState, ThemeMode;
import 'package:provider/provider.dart';
import '../../../core/constants/app_colors.dart';
import '../../../core/constants/app_strings.dart';
import '../../../data/models/user_profile.dart';
import '../../../providers/user_provider.dart';
import '../../../providers/sensor_provider.dart';
import 'classic_bt_scan_screen.dart';

/// Settings and profile screen
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: Consumer2<UserProvider, SensorProvider>(
        builder: (context, userProvider, sensorProvider, _) {
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              // Profile section
              _buildProfileSection(context, userProvider),
              const SizedBox(height: 24),

              // Device section
              _buildDeviceSection(context, sensorProvider),
              const SizedBox(height: 24),

              // Preferences section
              _buildPreferencesSection(context, userProvider),
              const SizedBox(height: 24),

              // Notifications section
              _buildNotificationsSection(context, userProvider),
              const SizedBox(height: 24),

              // Health Info section
              _buildHealthInfoSection(context, userProvider),
              const SizedBox(height: 24),

              // About section
              _buildAboutSection(context),
              const SizedBox(height: 24),

              // Logout button
              _buildLogoutButton(context, userProvider),
              const SizedBox(height: 32),
            ],
          );
        },
      ),
    );
  }

  Widget _buildProfileSection(BuildContext context, UserProvider provider) {
    final profile = provider.userProfile;

    return _buildSection(
      title: 'Profile',
      icon: Icons.person_outline,
      children: [
        ListTile(
          leading: CircleAvatar(
            radius: 28,
            backgroundColor: AppColors.primary.withValues(alpha: 0.1),
            child: Text(
              provider.userName.isNotEmpty ? provider.userName[0].toUpperCase() : '?',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: AppColors.primary,
              ),
            ),
          ),
          title: Text(
            provider.userName,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          subtitle: Text(profile?.email ?? 'No email'),
          trailing: const Icon(Icons.chevron_right),
          onTap: () => _showEditProfileDialog(context, provider),
        ),
        if (profile != null) ...[
          const Divider(),
          _buildInfoRow('Email', profile.email),
          if (profile.age != null && profile.age! > 0)
            _buildInfoRow('Age', '${profile.age} years'),
          if (profile.diabetesType != null)
            _buildInfoRow('Diabetes Type', profile.diabetesType!.displayName),
          if (profile.diabetesYears != null && profile.diabetesYears! > 0)
            _buildInfoRow(
              'Years with Diabetes',
              '${profile.diabetesYears} years',
            ),
          if (profile.phone != null && profile.phone!.isNotEmpty)
            _buildInfoRow('Phone', profile.phone!),
          // Health Info section
          if (profile.healthInfo != null) ...[
            const SizedBox(height: 12),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: Text(
                'Health Information',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 12,
                  color: AppColors.textSecondary,
                ),
              ),
            ),
            _buildHealthInfoItem(
              'Neuropathy',
              profile.healthInfo!.hasNeuropathy,
            ),
            _buildHealthInfoItem(
              'Peripheral Arterial Disease (PAD)',
              profile.healthInfo!.hasPAD,
            ),
            _buildHealthInfoItem(
              'Previous Foot Ulcer',
              profile.healthInfo!.hasPreviousUlcer,
            ),
            _buildHealthInfoItem(
              'Hypertension',
              profile.healthInfo!.hasHypertension,
            ),
          ],
          // Emergency Contact Info
          if (profile.emergencyContactName != null ||
              profile.emergencyContactPhone != null) ...[
            const SizedBox(height: 12),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: Text(
                'Emergency Contact',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 12,
                  color: AppColors.textSecondary,
                ),
              ),
            ),
            if (profile.emergencyContactName != null &&
                profile.emergencyContactName!.isNotEmpty)
              _buildInfoRow('Name', profile.emergencyContactName!),
            if (profile.emergencyContactPhone != null &&
                profile.emergencyContactPhone!.isNotEmpty)
              _buildInfoRow('Phone', profile.emergencyContactPhone!),
          ],
          // Account Dates
          const SizedBox(height: 12),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 16),
            child: Text(
              'Account',
              style: TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 12,
                color: AppColors.textSecondary,
              ),
            ),
          ),
          _buildInfoRow('Member Since', _formatDate(profile.createdAt)),
          if (profile.lastLoginAt != null)
            _buildInfoRow('Last Login', _formatDate(profile.lastLoginAt!)),
        ],
      ],
    );
  }

  /// Build health info item with checkmark
  Widget _buildHealthInfoItem(String label, bool value) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          Icon(
            value ? Icons.check_circle : Icons.cancel,
            size: 20,
            color: value ? AppColors.success : Colors.grey[400],
          ),
          const SizedBox(width: 12),
          Text(
            label,
            style: TextStyle(
              fontSize: 14,
              color: value ? AppColors.textPrimary : AppColors.textSecondary,
            ),
          ),
        ],
      ),
    );
  }

  /// Format date for display
  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }

  Widget _buildDeviceSection(BuildContext context, SensorProvider provider) {
    return _buildSection(
      title: 'Device & Sensor',
      icon: Icons.bluetooth,
      children: [
        ListTile(
          leading: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: provider.isConnected
                  ? AppColors.success.withValues(alpha: 0.1)
                  : Colors.grey.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(
              provider.isConnected
                  ? Icons.bluetooth_connected
                  : Icons.bluetooth,
              color: provider.isConnected ? AppColors.success : Colors.grey,
            ),
          ),
          title: Text(
            provider.isConnected ? provider.deviceName : 'No device connected',
            style: const TextStyle(fontWeight: FontWeight.w500),
            overflow: TextOverflow.ellipsis,
            maxLines: 1,
          ),
          subtitle: Text(
            provider.isConnected
                ? 'Battery: ${provider.batteryLevel}%'
                : 'Tap to connect',
            overflow: TextOverflow.ellipsis,
            maxLines: 1,
          ),
          trailing: SizedBox(
            width: 120,
            child: provider.isConnected
                ? TextButton(
                    onPressed: () => provider.disconnect(),
                    child: const Text('Disconnect'),
                  )
                : ElevatedButton(
                    onPressed: () {
                      // Navigate to device scan screen for Bluetooth connection
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (context) => const ClassicBtScanScreen(),
                        ),
                      );
                    },
                    child: const Text('Connect'),
                  ),
          ),
        ),
        const Divider(),
        SwitchListTile(
          title: const Text('Auto-reconnect'),
          subtitle: const Text('Reconnect when device is in range'),
          value: true, // TODO: Implement auto-reconnect setting
          onChanged: (value) {
            // TODO: Implement
          },
        ),
      ],
    );
  }

  Widget _buildPreferencesSection(BuildContext context, UserProvider provider) {
    final settings = provider.userProfile?.settings ?? const UserSettings();

    return _buildSection(
      title: 'Preferences',
      icon: Icons.tune,
      children: [
        ListTile(
          title: const Text('Temperature Unit'),
          subtitle: Text(
            settings.temperatureUnit == TemperatureUnit.celsius
                ? 'Celsius (°C)'
                : 'Fahrenheit (°F)',
          ),
          trailing: const Icon(Icons.chevron_right),
          onTap: () => _showTemperatureUnitDialog(context, provider, settings),
        ),
        const Divider(),
        ListTile(
          title: const Text('Theme'),
          subtitle: Text(
            settings.themeMode == ThemeMode.system
                ? 'System default'
                : settings.themeMode == ThemeMode.dark
                ? 'Dark'
                : 'Light',
          ),
          trailing: const Icon(Icons.chevron_right),
          onTap: () => _showThemeDialog(context, provider, settings),
        ),
        const Divider(),
        ListTile(
          title: const Text('Data Sync Interval'),
          subtitle: Text('${settings.syncIntervalMinutes} minutes'),
          trailing: const Icon(Icons.chevron_right),
          onTap: () => _showSyncIntervalDialog(context, provider, settings),
        ),
      ],
    );
  }

  Widget _buildNotificationsSection(
    BuildContext context,
    UserProvider provider,
  ) {
    final settings = provider.userProfile?.settings ?? const UserSettings();

    return _buildSection(
      title: 'Notifications',
      icon: Icons.notifications_outlined,
      children: [
        SwitchListTile(
          title: const Text('Enable Notifications'),
          subtitle: const Text('Receive alerts and updates'),
          value: settings.notificationsEnabled,
          onChanged: (value) {
            provider.updateSettings(
              settings.copyWith(notificationsEnabled: value),
            );
          },
        ),
        const Divider(),
        SwitchListTile(
          title: const Text('Critical Alerts'),
          subtitle: const Text('Sound alerts for critical readings'),
          value: settings.criticalAlertsEnabled,
          onChanged: settings.notificationsEnabled
              ? (value) {
                  provider.updateSettings(
                    settings.copyWith(criticalAlertsEnabled: value),
                  );
                }
              : null,
        ),
        const Divider(),
        SwitchListTile(
          title: const Text('Daily Summary'),
          subtitle: const Text('Daily health summary notification'),
          value: settings.dailySummaryEnabled,
          onChanged: settings.notificationsEnabled
              ? (value) {
                  provider.updateSettings(
                    settings.copyWith(dailySummaryEnabled: value),
                  );
                }
              : null,
        ),
      ],
    );
  }

  Widget _buildHealthInfoSection(BuildContext context, UserProvider provider) {
    final healthInfo = provider.userProfile?.healthInfo ?? const HealthInfo();

    return _buildSection(
      title: 'Health Information',
      icon: Icons.medical_information_outlined,
      children: [
        SwitchListTile(
          title: const Text('Neuropathy'),
          subtitle: const Text('History of peripheral neuropathy'),
          value: healthInfo.hasNeuropathy,
          onChanged: (value) {
            provider.updateHealthInfo(
              healthInfo.copyWith(hasNeuropathy: value),
            );
          },
        ),
        const Divider(),
        SwitchListTile(
          title: const Text('PAD'),
          subtitle: const Text('Peripheral Arterial Disease'),
          value: healthInfo.hasPAD,
          onChanged: (value) {
            provider.updateHealthInfo(healthInfo.copyWith(hasPAD: value));
          },
        ),
        const Divider(),
        SwitchListTile(
          title: const Text('Previous Ulcer'),
          subtitle: const Text('History of foot ulcers'),
          value: healthInfo.hasPreviousUlcer,
          onChanged: (value) {
            provider.updateHealthInfo(
              healthInfo.copyWith(hasPreviousUlcer: value),
            );
          },
        ),
        const Divider(),
        SwitchListTile(
          title: const Text('Hypertension'),
          subtitle: const Text('High blood pressure'),
          value: healthInfo.hasHypertension,
          onChanged: (value) {
            provider.updateHealthInfo(
              healthInfo.copyWith(hasHypertension: value),
            );
          },
        ),
      ],
    );
  }

  Widget _buildAboutSection(BuildContext context) {
    return _buildSection(
      title: 'About',
      icon: Icons.info_outline,
      children: [
        ListTile(
          title: const Text('App Version'),
          subtitle: const Text('1.0.0'),
          trailing: Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.success.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              'Latest',
              style: TextStyle(
                color: AppColors.success,
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
        const Divider(),
        ListTile(
          title: const Text('Privacy Policy'),
          trailing: const Icon(Icons.chevron_right),
          onTap: () {
            // TODO: Open privacy policy
          },
        ),
        const Divider(),
        ListTile(
          title: const Text('Terms of Service'),
          trailing: const Icon(Icons.chevron_right),
          onTap: () {
            // TODO: Open terms
          },
        ),
        const Divider(),
        ListTile(
          title: const Text('Contact Support'),
          trailing: const Icon(Icons.chevron_right),
          onTap: () {
            // TODO: Open support
          },
        ),
      ],
    );
  }

  Widget _buildLogoutButton(BuildContext context, UserProvider provider) {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: () => _showLogoutDialog(context, provider),
        icon: const Icon(Icons.logout, color: AppColors.error),
        label: const Text(
          AppStrings.logout,
          style: TextStyle(color: AppColors.error),
        ),
        style: OutlinedButton.styleFrom(
          side: const BorderSide(color: AppColors.error),
          padding: const EdgeInsets.symmetric(vertical: 12),
        ),
      ),
    );
  }

  Widget _buildSection({
    required String title,
    required IconData icon,
    required List<Widget> children,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Icon(icon, size: 20, color: AppColors.primary),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
          ...children,
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.grey[600])),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  void _showEditProfileDialog(BuildContext context, UserProvider provider) {
    final profile = provider.userProfile;
    final nameController = TextEditingController(text: profile?.name ?? '');
    final ageController = TextEditingController(
      text: profile?.age?.toString() ?? '',
    );
    final phoneController = TextEditingController(text: profile?.phone ?? '');
    final emergencyNameController = TextEditingController(
      text: profile?.emergencyContactName ?? '',
    );
    final emergencyPhoneController = TextEditingController(
      text: profile?.emergencyContactPhone ?? '',
    );
    final diabetesYearsController = TextEditingController(
      text: profile?.diabetesYears?.toString() ?? '',
    );

    DiabetesType? selectedDiabetesType = profile?.diabetesType;

    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) => AlertDialog(
          title: const Text('Edit Profile'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: nameController,
                  decoration: const InputDecoration(
                    labelText: 'Full Name',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.person),
                  ),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: ageController,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(
                          labelText: 'Age',
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: TextField(
                        controller: diabetesYearsController,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(
                          labelText: 'Years w/ Diabetes',
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                DropdownButtonFormField<DiabetesType>(
                  value: selectedDiabetesType,
                  decoration: const InputDecoration(
                    labelText: 'Diabetes Type',
                    border: OutlineInputBorder(),
                  ),
                  items: DiabetesType.values.map((type) {
                    return DropdownMenuItem(
                      value: type,
                      child: Text(type.displayName),
                    );
                  }).toList(),
                  onChanged: (value) {
                    setState(() {
                      selectedDiabetesType = value;
                    });
                  },
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: phoneController,
                  keyboardType: TextInputType.phone,
                  decoration: const InputDecoration(
                    labelText: 'Phone Number',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.phone),
                  ),
                ),
                const SizedBox(height: 24),
                const Divider(),
                const SizedBox(height: 8),
                const Text(
                  'Emergency Contact',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: emergencyNameController,
                  decoration: const InputDecoration(
                    labelText: 'Contact Name',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: emergencyPhoneController,
                  keyboardType: TextInputType.phone,
                  decoration: const InputDecoration(
                    labelText: 'Contact Phone',
                    border: OutlineInputBorder(),
                  ),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () async {
                if (profile == null || profile.id.isEmpty) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Error: User profile not found'),
                    ),
                  );
                  return;
                }

                try {
                  await provider.updateProfile(
                    name: nameController.text.trim().isEmpty
                        ? profile.name
                        : nameController.text.trim(),
                    age: int.tryParse(ageController.text.trim()),
                    phone: phoneController.text.trim(),
                    diabetesType: selectedDiabetesType,
                    diabetesYears: int.tryParse(
                      diabetesYearsController.text.trim(),
                    ),
                    emergencyContactName: emergencyNameController.text.trim(),
                    emergencyContactPhone: emergencyPhoneController.text.trim(),
                  );

                  if (!context.mounted) return;
                  Navigator.pop(context);

                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Profile updated and saved to Firestore'),
                      duration: Duration(seconds: 2),
                    ),
                  );
                } catch (e) {
                  if (!context.mounted) return;
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Error saving profile: $e')),
                  );
                }
              },
              child: const Text('Save Changes'),
            ),
          ],
        ),
      ),
    );
  }

  void _showTemperatureUnitDialog(
    BuildContext context,
    UserProvider provider,
    UserSettings settings,
  ) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Temperature Unit'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RadioListTile<TemperatureUnit>(
              title: const Text('Celsius (°C)'),
              value: TemperatureUnit.celsius,
              groupValue: settings.temperatureUnit,
              onChanged: (value) {
                provider.updateSettings(
                  settings.copyWith(temperatureUnit: value),
                );
                Navigator.pop(context);
              },
            ),
            RadioListTile<TemperatureUnit>(
              title: const Text('Fahrenheit (°F)'),
              value: TemperatureUnit.fahrenheit,
              groupValue: settings.temperatureUnit,
              onChanged: (value) {
                provider.updateSettings(
                  settings.copyWith(temperatureUnit: value),
                );
                Navigator.pop(context);
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showThemeDialog(
    BuildContext context,
    UserProvider provider,
    UserSettings settings,
  ) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Theme'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            RadioListTile<ThemeMode>(
              title: const Text('System default'),
              value: ThemeMode.system,
              groupValue: settings.themeMode,
              onChanged: (value) {
                if (value != null) {
                  provider.updateSettings(settings.copyWith(themeMode: value));
                }
                Navigator.pop(context);
              },
            ),
            RadioListTile<ThemeMode>(
              title: const Text('Light'),
              value: ThemeMode.light,
              groupValue: settings.themeMode,
              onChanged: (value) {
                if (value != null) {
                  provider.updateSettings(settings.copyWith(themeMode: value));
                }
                Navigator.pop(context);
              },
            ),
            RadioListTile<ThemeMode>(
              title: const Text('Dark'),
              value: ThemeMode.dark,
              groupValue: settings.themeMode,
              onChanged: (value) {
                if (value != null) {
                  provider.updateSettings(settings.copyWith(themeMode: value));
                }
                Navigator.pop(context);
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showSyncIntervalDialog(
    BuildContext context,
    UserProvider provider,
    UserSettings settings,
  ) {
    final intervals = [1, 2, 5, 10, 30];

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Data Sync Interval'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: intervals.map((interval) {
            return RadioListTile<int>(
              title: Text('$interval minutes'),
              value: interval,
              groupValue: settings.syncIntervalMinutes,
              onChanged: (value) {
                provider.updateSettings(
                  settings.copyWith(syncIntervalMinutes: value),
                );
                Navigator.pop(context);
              },
            );
          }).toList(),
        ),
      ),
    );
  }

  void _showLogoutDialog(BuildContext context, UserProvider provider) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Logout'),
        content: const Text('Are you sure you want to logout?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              provider.deleteProfile();
              Navigator.of(
                context,
              ).pushNamedAndRemoveUntil('/welcome', (route) => false);
            },
            style: ElevatedButton.styleFrom(backgroundColor: AppColors.error),
            child: const Text('Logout'),
          ),
        ],
      ),
    );
  }
}
