import 'package:flutter/material.dart' hide ThemeMode;
import 'package:provider/provider.dart';
import 'core/theme/app_theme.dart';
import 'core/constants/app_strings.dart';
import 'data/models/user_profile.dart';
import 'providers/sensor_provider.dart';
import 'providers/risk_provider.dart';
import 'providers/user_provider.dart';
import 'providers/firebase/firebase_auth_provider.dart';
import 'providers/firebase/firebase_sync_provider.dart';
import 'providers/firebase/firebase_notifications_provider.dart';
import 'ui/screens/auth/landing_screen.dart';
import 'ui/screens/auth/sign_in_screen.dart';
import 'ui/screens/auth/sign_up_screen.dart';
import 'ui/screens/home/dashboard_screen.dart';
import 'ui/screens/home/sensors_screen.dart';
import 'ui/screens/home/alerts_screen.dart';
import 'ui/screens/home/settings_screen.dart';

// Use Flutter's ThemeMode for MaterialApp
import 'package:flutter/material.dart' as flutter show ThemeMode;

/// Main application widget with providers and routing
class SmartSocksApp extends StatelessWidget {
  const SmartSocksApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        // User provider (manages profile, settings, theme)
        ChangeNotifierProvider(create: (_) => UserProvider()),
        
        // Sensor provider (manages BLE, sensor data)
        ChangeNotifierProvider(create: (_) => SensorProvider()),
        
        // Risk provider (manages risk calculations, alerts)
        ChangeNotifierProvider(create: (_) => RiskProvider()),
        
        // 🔥 Firebase Authentication Provider
        ChangeNotifierProvider(create: (_) => FirebaseAuthProvider()),
        
        // 🔥 Firebase Sync Provider
        ChangeNotifierProvider(create: (_) => FirebaseSyncProvider()),
        
        // 🔥 Firebase Notifications Provider
        ChangeNotifierProvider(create: (_) => FirebaseNotificationsProvider()),
      ],
      child: Consumer2<UserProvider, FirebaseAuthProvider>(
        builder: (context, userProvider, authProvider, _) {
          return MaterialApp(
            title: AppStrings.appName,
            debugShowCheckedModeBanner: false,
            
            // Theme configuration
            theme: AppTheme.lightTheme,
            darkTheme: AppTheme.darkTheme,
            themeMode: _mapThemeMode(userProvider.userProfile?.settings.themeMode),
            
            // Initial route based on login status
            initialRoute: _getInitialRoute(context),
            
            // Named routes
            routes: {
              '/landing': (context) => const LandingScreen(),
              '/sign-in': (context) => const SignInScreen(),
              '/sign-up': (context) => const SignUpScreen(),
              '/dashboard': (context) => const DashboardScreen(),
              '/sensors': (context) => const SensorsScreen(),
              '/alerts': (context) => const AlertsScreen(),
              '/settings': (context) => const SettingsScreen(),
            },
            
            // Handle unknown routes
            onUnknownRoute: (settings) {
              return MaterialPageRoute(
                builder: (context) => const LandingScreen(),
              );
            },
          );
        },
      ),
    );
  }

  /// Determine initial route based on user state
  String _getInitialRoute(BuildContext context) {
    final authProvider = context.read<FirebaseAuthProvider>();
    
    // If user is logged in, go to dashboard
    if (authProvider.isLoggedIn) {
      return '/dashboard';
    }
    
    // Otherwise, show landing screen
    return '/landing';
  }

  /// Map user_profile ThemeMode to Flutter ThemeMode
  flutter.ThemeMode _mapThemeMode(ThemeMode? mode) {
    if (mode == null) return flutter.ThemeMode.system;
    switch (mode) {
      case ThemeMode.light:
        return flutter.ThemeMode.light;
      case ThemeMode.dark:
        return flutter.ThemeMode.dark;
      case ThemeMode.system:
        return flutter.ThemeMode.system;
    }
  }
}

/// Home screen with bottom navigation
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;

  final List<Widget> _screens = const [
    DashboardScreen(),
    SensorsScreen(),
    AlertsScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.dashboard_outlined),
            selectedIcon: const Icon(Icons.dashboard),
            label: AppStrings.navDashboard,
          ),
          NavigationDestination(
            icon: const Icon(Icons.sensors_outlined),
            selectedIcon: const Icon(Icons.sensors),
            label: AppStrings.navFootMap,
          ),
          NavigationDestination(
            icon: Consumer<RiskProvider>(
              builder: (context, provider, child) {
                return Badge(
                  isLabelVisible: provider.unreadAlertCount > 0,
                  label: Text('${provider.unreadAlertCount}'),
                  child: const Icon(Icons.notifications_outlined),
                );
              },
            ),
            selectedIcon: Consumer<RiskProvider>(
              builder: (context, provider, child) {
                return Badge(
                  isLabelVisible: provider.unreadAlertCount > 0,
                  label: Text('${provider.unreadAlertCount}'),
                  child: const Icon(Icons.notifications),
                );
              },
            ),
            label: AppStrings.navAlerts,
          ),
          NavigationDestination(
            icon: const Icon(Icons.settings_outlined),
            selectedIcon: const Icon(Icons.settings),
            label: AppStrings.navSettings,
          ),
        ],
      ),
    );
  }
}
