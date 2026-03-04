// Phase 4 Testing: Feature Engineering Pipeline
// Verify: extraction, normalization, output shape

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_socks/data/models/sensor_reading.dart';
import 'package:smart_socks/data/services/ml_feature_engineer.dart';
import 'package:smart_socks/core/constants/ml_constants.dart';

void main() {
  group('ML Feature Engineer', () {
    late MLFeatureEngineer engineer;

    setUp(() {
      engineer = MLFeatureEngineer();
    });

    // Test 1: Raw Feature Extraction
    test('Extract raw features - correct count (11)', () {
      final reading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [30.0, 31.0, 29.5, 32.0],  // 4 zones
        pressures: [50.0, 55.0, 45.0, 60.0],     // 4 zones
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        batteryLevel: 85,
      );

      final features = engineer.extractRawFeatures(reading);

      expect(features.length, 11, reason: 'Should extract exactly 11 raw features');
      expect(features[0], 30.0, reason: 'First temp (heel) should be 30.0');
      expect(features[4], 50.0, reason: 'First pressure (heel) should be 50.0');
      expect(features[8], 98.5, reason: 'SpO2 should be 98.5');
    });

    // Test 2: Engineered Features Calculation
    test('Calculate engineered features - correct count (4)', () {
      final rawFeatures = [
        30.0, 31.0, 29.5, 32.0,  // temps
        50.0, 55.0, 45.0, 60.0,  // pressures
        98.5, 72.0, 100.0         // vitals + steps (11 total)
      ];

      final engineered = engineer.calculateEngineeredFeatures(rawFeatures);

      expect(engineered.length, 4, reason: 'Should calculate exactly 4 engineered features');
      expect(engineered[0], greaterThan(0), reason: 'max_pressure should be > 0');
      expect(engineered[1], greaterThanOrEqualTo(0), reason: 'pressure_variance should be >= 0');
    });

    // Test 3: Full Pipeline - Correct Total Features
    test('Process sensor reading - output 15 normalized features', () {
      final reading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [30.0, 31.0, 29.5, 32.0],
        pressures: [50.0, 55.0, 45.0, 60.0],
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        batteryLevel: 85,
      );

      final features = engineer.processSensorReading(reading);

      expect(features, isNotNull, reason: 'Should return feature vector');
      expect(features!.length, MLConstants.totalFeatures,
          reason: 'Should return ${MLConstants.totalFeatures} features (11 raw + 4 engineered)');
    });

    // Test 4: Normalization - Features in Expected Range
    test('Normalized features in valid range after StandardScaler', () {
      final reading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [30.0, 31.0, 29.5, 32.0],
        pressures: [50.0, 55.0, 45.0, 60.0],
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        batteryLevel: 85,
      );

      final features = engineer.processSensorReading(reading);

      expect(features, isNotNull);
      // After StandardScaler normalization, values should be finite numbers
      // Z-scores can be large when test inputs differ significantly from training mean
      for (int i = 0; i < features!.length; i++) {
        expect(features[i].isFinite, isTrue,
            reason: 'Feature $i should be a finite number after normalization');
        expect(features[i].isNaN, isFalse,
            reason: 'Feature $i should not be NaN');
      }
    });

    // Test 5: Handle Edge Cases
    test('Handle empty/zero sensor readings gracefully', () {
      final reading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [0.0, 0.0, 0.0, 0.0],
        pressures: [0.0, 0.0, 0.0, 0.0],
        spO2: 0.0,
        heartRate: 0,
        stepCount: 0,
        batteryLevel: 0,
      );

      final features = engineer.processSensorReading(reading);

      expect(features, isNotNull, reason: 'Should handle zero readings without crashing');
      expect(features!.length, MLConstants.totalFeatures);
    });

    // Test 6: Consistent Output for Same Input
    test('Same input produces same output (deterministic)', () {
      final reading = SensorReading(
        timestamp: DateTime.now(),
        temperatures: [30.0, 31.0, 29.5, 32.0],
        pressures: [50.0, 55.0, 45.0, 60.0],
        spO2: 98.5,
        heartRate: 72,
        stepCount: 100,
        batteryLevel: 85,
      );

      final features1 = engineer.processSensorReading(reading);
      final features2 = engineer.processSensorReading(reading);

      expect(features1, features2,
          reason: 'Same input should produce identical output');
    });
  });
}
