import 'package:flutter/material.dart';

/// App color palette for NeuroSocks
/// Designed for healthcare UI with clear risk indicators
class AppColors {
  AppColors._();

  // ============== Primary Brand Colors =============
  static const Color primary = Color(0xFF2563EB); // Blue
  static const Color primaryLight = Color(0xFF3B82F6);
  static const Color primaryDark = Color(0xFF1D4ED8);

  static const Color secondary = Color(0xFF7C3AED); // Purple
  static const Color secondaryLight = Color(0xFF8B5CF6);
  static const Color secondaryDark = Color(0xFF6D28D9);

  // ============== Risk Level Colors ==============
  /// Low risk - Safe, normal readings
  static const Color riskLow = Color(0xFF10B981); // Green
  static const Color riskLowLight = Color(0xFF34D399);
  static const Color riskLowBg = Color(0xFFD1FAE5);

  /// Moderate risk - Attention needed
  static const Color riskModerate = Color(0xFFF59E0B); // Yellow/Amber
  static const Color riskModerateLight = Color(0xFFFBBF24);
  static const Color riskModerateBg = Color(0xFFFEF3C7);

  /// High risk - Take action
  static const Color riskHigh = Color(0xFFF97316); // Orange
  static const Color riskHighLight = Color(0xFFFB923C);
  static const Color riskHighBg = Color(0xFFFFEDD5);

  /// Critical risk - Urgent attention
  static const Color riskCritical = Color(0xFFEF4444); // Red
  static const Color riskCriticalLight = Color(0xFFF87171);
  static const Color riskCriticalBg = Color(0xFFFEE2E2);

  // ============== Sensor Type Colors ==============
  static const Color temperature = Color(0xFFEF4444); // Red for heat
  static const Color temperatureLight = Color(0xFFFCA5A5);
  static const Color temperatureBg = Color(0xFFFEE2E2);

  static const Color pressure = Color(0xFF3B82F6); // Blue for pressure
  static const Color pressureLight = Color(0xFF93C5FD);
  static const Color pressureBg = Color(0xFFDBEAFE);

  static const Color oxygen = Color(0xFF8B5CF6); // Purple for SpO2
  static const Color oxygenLight = Color(0xFFC4B5FD);
  static const Color oxygenBg = Color(0xFFEDE9FE);

  static const Color heartRate = Color(0xFFEC4899); // Pink for heart
  static const Color heartRateLight = Color(0xFFF9A8D4);
  static const Color heartRateBg = Color(0xFFFCE7F3);

  static const Color gait = Color(0xFF14B8A6); // Teal for movement
  static const Color gaitLight = Color(0xFF5EEAD4);
  static const Color gaitBg = Color(0xFFCCFBF1);

  // ============== Background Colors ==============
  static const Color background = Color(0xFFF8FAFC);
  static const Color backgroundDark = Color(0xFF0F172A);

  static const Color surface = Color(0xFFFFFFFF);
  static const Color surfaceDark = Color(0xFF1E293B);

  static const Color card = Color(0xFFFFFFFF);
  static const Color cardDark = Color(0xFF334155);

  // ============== Text Colors ==============
  static const Color textPrimary = Color(0xFF1E293B);
  static const Color textPrimaryDark = Color(0xFFF1F5F9);

  static const Color textSecondary = Color(0xFF64748B);
  static const Color textSecondaryDark = Color(0xFF94A3B8);

  static const Color textHint = Color(0xFF94A3B8);
  static const Color textHintDark = Color(0xFF64748B);

  // ============== UI Element Colors ==============
  static const Color divider = Color(0xFFE2E8F0);
  static const Color dividerDark = Color(0xFF475569);

  static const Color border = Color(0xFFCBD5E1);
  static const Color borderDark = Color(0xFF475569);

  static const Color disabled = Color(0xFF94A3B8);
  static const Color disabledBg = Color(0xFFE2E8F0);

  static const Color error = Color(0xFFDC2626);
  static const Color errorLight = Color(0xFFFEE2E2);

  static const Color success = Color(0xFF16A34A);
  static const Color successLight = Color(0xFFDCFCE7);

  static const Color warning = Color(0xFFD97706);
  static const Color warningLight = Color(0xFFFEF3C7);

  static const Color info = Color(0xFF0284C7);
  static const Color infoLight = Color(0xFFE0F2FE);

  // ============== Chart Colors ==============
  static const List<Color> chartColors = [
    Color(0xFF3B82F6), // Blue
    Color(0xFF10B981), // Green
    Color(0xFFF59E0B), // Amber
    Color(0xFFEF4444), // Red
    Color(0xFF8B5CF6), // Purple
    Color(0xFFEC4899), // Pink
  ];

  // ============== Foot Zone Colors ==============
  static const Color zoneHeel = Color(0xFF3B82F6);
  static const Color zoneBall = Color(0xFF10B981);
  static const Color zoneArch = Color(0xFFF59E0B);
  static const Color zoneToe = Color(0xFF8B5CF6);

  // ============== Gradient Colors ==============
  static const LinearGradient primaryGradient = LinearGradient(
    colors: [primary, primaryLight],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient riskGradient = LinearGradient(
    colors: [riskLow, riskModerate, riskHigh, riskCritical],
    stops: [0.0, 0.33, 0.66, 1.0],
    begin: Alignment.centerLeft,
    end: Alignment.centerRight,
  );

  // ============== Helper Methods ==============

  /// Get color based on risk score (0-100)
  static Color getRiskColor(int score) {
    if (score <= 30) return riskLow;
    if (score <= 50) return riskModerate;
    if (score <= 70) return riskHigh;
    return riskCritical;
  }

  /// Get background color based on risk score
  static Color getRiskBgColor(int score) {
    if (score <= 30) return riskLowBg;
    if (score <= 50) return riskModerateBg;
    if (score <= 70) return riskHighBg;
    return riskCriticalBg;
  }

  /// Get color for temperature value
  static Color getTemperatureColor(double temp) {
    if (temp < 28) return info; // Too cold
    if (temp <= 34) return riskLow; // Normal
    if (temp <= 36) return riskModerate; // Warm
    if (temp <= 38) return riskHigh; // Hot
    return riskCritical; // Very hot
  }

  /// Get color for pressure value (kPa)
  static Color getPressureColor(double pressure) {
    if (pressure <= 50) return riskLow; // Normal
    if (pressure <= 80) return riskModerate; // Elevated
    if (pressure <= 100) return riskHigh; // High
    return riskCritical; // Very high
  }

  /// Get color for SpO2 value
  static Color getSpO2Color(double spo2) {
    if (spo2 >= 95) return riskLow; // Normal
    if (spo2 >= 90) return riskModerate; // Low
    if (spo2 >= 85) return riskHigh; // Very low
    return riskCritical; // Critical
  }
}
