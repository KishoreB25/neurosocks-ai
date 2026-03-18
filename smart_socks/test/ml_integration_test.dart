// Phase 4 Testing: ML Integration & Risk Calculation
// Verify: full pipeline works, fallback logic works
// Note: Tests use RiskCalculator directly (singleton).
//       RiskProvider tests are skipped as they require StorageService (Hive).

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_socks/data/models/sensor_reading.dart';
import 'package:smart_socks/data/models/risk_score.dart';
import 'package:smart_socks/data/services/risk_calculator.dart';

void main() {
  group('ML Integration & Risk Calculation', () {
    late RiskCalculator calculator;
    late SensorReading testReading;

    setUp(() {
      calculator = RiskCalculator();
      // Clear singleton history before each test
      calculator.clearHistory();

      testReading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [30.0, 31.0, 29.5, 32.0],  // Normal
        pressures: [50.0, 55.0, 45.0, 60.0],     // Elevated
        spO2: 98.5,                              // Normal
        heartRate: 72,                           // Normal
        stepCount: 100,
        batteryLevel: 85,
      );
    });

    // Test 1: RiskCalculator Basic Functionality
    test('RiskCalculator calculates risk score from sensor reading', () {
      final riskScore = calculator.calculate(testReading);

      expect(riskScore, isNotNull, reason: 'Should return RiskScore');
      expect(riskScore.overallScore, greaterThanOrEqualTo(0));
      expect(riskScore.overallScore, lessThanOrEqualTo(100),
          reason: 'Risk score should be 0-100');
    });

    // Test 2: Fallback Logic Works (when ML unavailable)
    test('Fallback to threshold-based calculation when ML not ready', () {
      // ML is not initialized in test env, so it should use fallback
      final riskScore = calculator.calculate(testReading);

      expect(riskScore, isNotNull);
      expect(riskScore.temperatureRisk, isNotNull);
      expect(riskScore.pressureRisk, isNotNull);
      expect(riskScore.circulationRisk, isNotNull);
      expect(riskScore.gaitRisk, isNotNull,
          reason: 'Fallback should calculate all component risks');
    });

    // Test 3: Risk Score Output Format
    test('Risk score has correct structure', () {
      final riskScore = calculator.calculate(testReading);

      expect(riskScore.factors, isList);
      expect(riskScore.recommendations, isList);
      expect(riskScore.riskLevel, isNotNull);
      expect(riskScore.timestamp, isNotNull);
    });

    // Test 4: Temperature Risk Calculation
    test('Temperature risk calculated correctly', () {
      // Normal temperatures
      var score1 = calculator.calculate(testReading);
      expect(score1.temperatureRisk, lessThanOrEqualTo(50),
          reason: 'Normal temps should have low risk');

      // Create reading with elevated temps
      final hotReading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [37.0, 38.0, 36.5, 39.0],  // High temps
        pressures: [50.0, 55.0, 45.0, 60.0],
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        batteryLevel: 85,
      );

      var score2 = calculator.calculate(hotReading);
      expect(score2.temperatureRisk, greaterThan(score1.temperatureRisk),
          reason: 'High temps should increase temperature risk');
    });

    // Test 5: Pressure Risk Calculation
    test('Pressure risk calculated correctly', () {
      // Normal pressures
      var score1 = calculator.calculate(testReading);
      expect(score1.pressureRisk, isNotNull);

      // Create reading with high pressures
      final highPressureReading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [30.0, 31.0, 29.5, 32.0],
        pressures: [100.0, 110.0, 95.0, 120.0],  // High pressures
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        batteryLevel: 85,
      );

      var score2 = calculator.calculate(highPressureReading);
      expect(score2.pressureRisk, greaterThan(score1.pressureRisk),
          reason: 'High pressure should increase pressure risk');
    });

    // Test 6: Risk History Tracking
    test('RiskCalculator tracks history correctly', () {
      // History cleared in setUp
      expect(calculator.history.length, 0, reason: 'Should start with empty history');

      calculator.calculate(testReading);
      expect(calculator.history.length, 1);

      calculator.calculate(testReading);
      expect(calculator.history.length, 2);
    });

    // Test 7: Risk Level Classification
    test('Risk level classified correctly from score', () {
      var score = calculator.calculate(testReading);
      expect(score.riskLevel, isNotNull);
      expect([RiskLevel.low, RiskLevel.moderate, RiskLevel.high, RiskLevel.critical],
          contains(score.riskLevel),
          reason: 'Should return valid RiskLevel');
    });

    // Test 8: Extreme Case - Very High Temperature
    test('Handle extreme sensor values (high temperature alert)', () {
      final extremeReading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [40.0, 41.0, 39.5, 42.0],  // Critical temps
        pressures: [50.0, 55.0, 45.0, 60.0],
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        batteryLevel: 85,
      );

      // Normal reading first for comparison
      final normalScore = calculator.calculate(testReading);
      final extremeScore = calculator.calculate(extremeReading);

      expect(extremeScore, isNotNull);
      expect(extremeScore.temperatureRisk, greaterThan(normalScore.temperatureRisk),
          reason: 'Critical temps should have higher temperature risk than normal');
      expect(extremeScore.factors, isNotEmpty,
          reason: 'Should identify risk factors');
    });

    // Test 9: Extreme Case - Very High Pressure
    test('Handle extreme sensor values (high pressure alert)', () {
      final extremeReading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [30.0, 31.0, 29.5, 32.0],
        pressures: [150.0, 160.0, 145.0, 170.0],  // Critical pressures
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        batteryLevel: 85,
      );

      final normalScore = calculator.calculate(testReading);
      final extremeScore = calculator.calculate(extremeReading);

      expect(extremeScore, isNotNull);
      expect(extremeScore.pressureRisk, greaterThan(normalScore.pressureRisk),
          reason: 'Critical pressures should have higher pressure risk than normal');
    });

    // Test 10: Multiple Readings - Risk Trend via RiskCalculator
    test('Track risk trend across multiple readings', () {
      // Low risk reading
      var score1 = calculator.calculate(testReading);

      // High pressure reading
      final highPressureReading = SensorReading(
        timestamp: DateTime.now().add(Duration(seconds: 5)),
        temperatures: [30.0, 31.0, 29.5, 32.0],
        pressures: [150.0, 160.0, 145.0, 170.0],
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        batteryLevel: 85,
      );

      var score2 = calculator.calculate(highPressureReading);

      expect(calculator.history.length, 2);
      expect(score2.pressureRisk, greaterThan(score1.pressureRisk),
          reason: 'Risk should increase with high pressure');
    });

    // Test 11: Consistent scoring for identical inputs
    test('Same input produces consistent risk scores', () {
      calculator.clearHistory();
      final score1 = calculator.calculate(testReading);

      calculator.clearHistory();
      final score2 = calculator.calculate(testReading);

      expect(score1.overallScore, equals(score2.overallScore),
          reason: 'Identical inputs should produce identical scores');
      expect(score1.temperatureRisk, equals(score2.temperatureRisk));
      expect(score1.pressureRisk, equals(score2.pressureRisk));
    });
  });
}
