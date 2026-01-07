import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'app.dart';

/// Main entry point for Smart Socks application
void main() async {
  // Ensure Flutter bindings are initialized
  WidgetsFlutterBinding.ensureInitialized();

  // Set preferred orientations (portrait only for mobile)
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Set system UI overlay style
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
      statusBarBrightness: Brightness.light,
    ),
  );

  // Initialize Hive for local storage
  await _initializeHive();

  // Run the app
  runApp(const SmartSocksApp());
}

/// Initialize Hive database
Future<void> _initializeHive() async {
  // Initialize Hive with Flutter
  await Hive.initFlutter();

  // Open boxes for persistent storage
  // Using dynamic boxes since we're storing JSON-serializable data
  await Future.wait([
    Hive.openBox('sensor_readings'),
    Hive.openBox('foot_data'),
    Hive.openBox('risk_scores'),
    Hive.openBox('alerts'),
    Hive.openBox('user_profile'),
    Hive.openBox('settings'),
  ]);
}
