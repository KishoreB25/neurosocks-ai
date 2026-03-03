// Phase 4 Testing: ML Integration & Risk Calculation
// Verify: full pipeline works, fallback logic works

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_socks/data/models/sensor_reading.dart';
import 'package:smart_socks/data/models/risk_score.dart';
import 'package:smart_socks/data/services/risk_calculator.dart';
import 'package:smart_socks/providers/risk_provider.dart';

void main() {
  group('ML Integration & Risk Calculation', () {
    late SensorReading testReading;

    setUp(() {
      testReading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [30.0, 31.0, 29.5, 32.0],  // Normal
        pressures: [50.0, 55.0, 45.0, 60.0],     // Elevated
        spO2: 98.5,                              // Normal
        heartRate: 72,                           // Normal
        stepCount: 100,
        battery: 85,
      );
    });

    // Test 1: RiskCalculator Basic Functionality
    test('RiskCalculator calculates risk score from sensor reading', () {
      final calculator = RiskCalculator();
      
      final riskScore = calculator.calculate(testReading);

      expect(riskScore, isNotNull, reason: 'Should return RiskScore');
      expect(riskScore.overallScore, greaterThanOrEqualTo(0));
      expect(riskScore.overallScore, lessThanOrEqualTo(100));
            reason: 'Risk score should be 0-100');
    });

    // Test 2: Fallback Logic Works (when ML unavailable)
    test('Fallback to threshold-based calculation when ML not ready', () {
      final calculator = RiskCalculator();
      // ML is not initialized, so it should use fallback
      
      final riskScore = calculator.calculate(testReading);

      expect(riskScore, isNotNull);
      expect(riskScore.temperatureRisk, isNotNull);
      expect(riskScore.pressureRisk, isNotNull);
      expect(riskScore.circulationRisk, isNotNull);
      expect(riskScore.gaitRisk, isNotNull);
          reason: 'Fallback should calculate all component risks');
    });

    // Test 3: Risk Score Output Format
    test('Risk score has correct structure', () {
      final calculator = RiskCalculator();
      
      final riskScore = calculator.calculate(testReading);

      expect(riskScore.factors, isList);
      expect(riskScore.recommendations, isList);
      expect(riskScore.riskLevel, isNotNull);
      expect(riskScore.timestamp, isNotNull);
    });

    // Test 4: Temperature Risk Calculation
    test('Temperature risk calculated correctly', () {
      final calculator = RiskCalculator();
      
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
        battery: 85,
      );
      
      var score2 = calculator.calculate(hotReading);
      expect(score2.temperatureRisk, greaterThan(score1.temperatureRisk),
          reason: 'High temps should increase temperature risk');
    });

    // Test 5: Pressure Risk Calculation
    test('Pressure risk calculated correctly', () {
      final calculator = RiskCalculator();
      
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
        battery: 85,
      );
      
      var score2 = calculator.calculate(highPressureReading);
      expect(score2.pressureRisk, greaterThan(score1.pressureRisk),
          reason: 'High pressure should increase pressure risk');
    });

    // Test 6: Risk History Tracking
    test('RiskCalculator tracks history correctly', () {
      final calculator = RiskCalculator();
      
      expect(calculator.history.length, 0, reason: 'Should start with empty history');
      
      calculator.calculate(testReading);
      expect(calculator.history.length, 1);
      
      calculator.calculate(testReading);
      expect(calculator.history.length, 2);
    });

    // Test 7: RiskProvider Integration
    test('RiskProvider processes readings correctly', () {
      final provider = RiskProvider();
      
      expect(provider.currentRiskScore, isNull, reason: 'Should start with no score');
      
      provider.processReading(testReading);
      
      expect(provider.currentRiskScore, isNotNull);
      expect(provider.currentScore, greaterThanOrEqualTo(0));
      expect(provider.currentScore, lessThanOrEqualTo(100));
    });

    // Test 8: Risk Level Classification
    test('Risk level classified correctly from score', () {
      final calculator = RiskCalculator();
      
      // Low risk reading (normal)
      var score = calculator.calculate(testReading);
      expect(score.riskLevel, isNotNull);
      expect([RiskLevel.low, RiskLevel.moderate, RiskLevel.high, RiskLevel.critical],
          contains(score.riskLevel),
          reason: 'Should return valid RiskLevel');
    });

    // Test 9: Extreme Case - Very High Temperature
    test('Handle extreme sensor values (high temperature alert)', () {
      final extremeReading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [40.0, 41.0, 39.5, 42.0],  // Critical temps
        pressures: [50.0, 55.0, 45.0, 60.0],
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        battery: 85,
      );

      final calculator = RiskCalculator();
      final score = calculator.calculate(extremeReading);

      expect(score, isNotNull);
      expect(score.overallScore, greaterThan(50),
          reason: 'Should detect high risk from extreme temps');
      expect(score.factors, isNotEmpty,
          reason: 'Should identify temperature as risk factor');
    });

    // Test 10: Extreme Case - Very High Pressure
    test('Handle extreme sensor values (high pressure alert)', () {
      final extremeReading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [30.0, 31.0, 29.5, 32.0],
        pressures: [150.0, 160.0, 145.0, 170.0],  // Critical pressures
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        battery: 85,
      );

      final calculator = RiskCalculator();
      final score = calculator.calculate(extremeReading);

      expect(score, isNotNull);
      expect(score.overallScore, greaterThan(50),
          reason: 'Should detect high risk from extreme pressures');
    });

    // Test 11: Multiple Readings - Risk Trend
    test('Track risk trend across multiple readings', () {
      final provider = RiskProvider();
      
      // Low risk reading
      provider.processReading(testReading);
      var score1 = provider.currentScore;
      
      // High pressure reading
      final highPressureReading = SensorReading(
        timestamp: DateTime.now().add(Duration(seconds: 5)),
        temperatures: [30.0, 31.0, 29.5, 32.0],
        pressures: [150.0, 160.0, 145.0, 170.0],
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        battery: 85,
      );
      
      provider.processReading(highPressureReading);
      var score2 = provider.currentScore;
      
      expect(provider.riskHistory.length, 2);
      expect(score2, greaterThan(score1),
          reason: 'Risk should increase with high pressure');
    });
  });
}
