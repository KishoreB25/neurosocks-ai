/// Machine Learning Feature Engineering Service
/// 
/// Converts raw SensorReading data into a normalized feature vector
/// ready for TFLite model inference.
///
/// Pipeline:
/// SensorReading → Extract 15 features → Validate → Normalize → [1×15] vector

import '../models/sensor_reading.dart';
import '../../core/constants/ml_constants.dart';
import '../../core/constants/ml_feature_order.dart';
import 'package:flutter/foundation.dart';

/// Extracts and engineers features from sensor readings for ML inference
class MLFeatureEngineer {
  // Singleton pattern
  static final MLFeatureEngineer _instance = MLFeatureEngineer._internal();
  
  factory MLFeatureEngineer() {
    return _instance;
  }
  
  MLFeatureEngineer._internal();

  // Debug logging
  static const bool debugLogging = MLConstants.debugLogging;

  // ============== Feature Extraction ==============

  /// Extract raw features from SensorReading
  /// Returns 11 raw features in exact order
  List<double> extractRawFeatures(SensorReading reading) {
    try {
      return [
        // Temperature features (4)
        reading.temperatures.isNotEmpty ? reading.temperatures[0] : 0.0,  // temp_heel
        reading.temperatures.length > 1 ? reading.temperatures[1] : 0.0,  // temp_ball
        reading.temperatures.length > 2 ? reading.temperatures[2] : 0.0,  // temp_arch
        reading.temperatures.length > 3 ? reading.temperatures[3] : 0.0,  // temp_toe
        
        // Pressure features (4)
        reading.pressures.isNotEmpty ? reading.pressures[0] : 0.0,        // press_heel
        reading.pressures.length > 1 ? reading.pressures[1] : 0.0,        // press_ball
        reading.pressures.length > 2 ? reading.pressures[2] : 0.0,        // press_arch
        reading.pressures.length > 3 ? reading.pressures[3] : 0.0,        // press_toe
        
        // Vital signs (2)
        reading.spO2,                                                       // spo2
        reading.heartRate.toDouble(),                                       // heartRate
        
        // Activity (1)
        reading.stepCount.toDouble(),                                       // stepCount
      ];
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Error extracting raw features: $e');
      }
      // Return zeros if extraction fails
      return List<double>.filled(11, 0.0);
    }
  }

  // ============== Feature Engineering ==============

  /// Calculate engineered features from raw features
  /// Returns 4 engineered features derived from raw data
  List<double> calculateEngineeredFeatures(List<double> rawFeatures) {
    try {
      if (rawFeatures.length < 11) {
        if (debugLogging) {
          debugPrint('⚠️ Raw features incomplete: ${rawFeatures.length}/11');
        }
        return List<double>.filled(4, 0.0);
      }

      // Extract pressure and temperature zones from raw features
      final pressures = [
        rawFeatures[4],  // press_heel
        rawFeatures[5],  // press_ball
        rawFeatures[6],  // press_arch
        rawFeatures[7],  // press_toe
      ];

      final temperatures = [
        rawFeatures[0],  // temp_heel
        rawFeatures[1],  // temp_ball
        rawFeatures[2],  // temp_arch
        rawFeatures[3],  // temp_toe
      ];

      // 1. max_pressure
      final maxPressure = pressures.reduce((a, b) => a > b ? a : b);

      // 2. pressure_variance
      final pressureMean = pressures.isNotEmpty 
          ? pressures.reduce((a, b) => a + b) / pressures.length 
          : 0.0;
      final pressureVariance = pressures.isNotEmpty
          ? pressures
              .map((p) => (p - pressureMean) * (p - pressureMean))
              .reduce((a, b) => a + b) / pressures.length
          : 0.0;

      // 3. max_temp
      final maxTemp = temperatures.reduce((a, b) => a > b ? a : b);

      // 4. temp_variance
      final tempMean = temperatures.isNotEmpty 
          ? temperatures.reduce((a, b) => a + b) / temperatures.length 
          : 0.0;
      final tempVariance = temperatures.isNotEmpty
          ? temperatures
              .map((t) => (t - tempMean) * (t - tempMean))
              .reduce((a, b) => a + b) / temperatures.length
          : 0.0;

      return [
        maxPressure,
        pressureVariance,
        maxTemp,
        tempVariance,
      ];
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Error calculating engineered features: $e');
      }
      return List<double>.filled(4, 0.0);
    }
  }

  // ============== Feature Normalization ==============

  /// Normalize features using StandardScaler coefficients
  /// Formula: (feature - mean) / std
  List<double> normalizeFeatures(List<double> allFeatures) {
    try {
      if (allFeatures.length != MLConstants.totalFeatures) {
        if (debugLogging) {
          debugPrint(
            '❌ Feature count mismatch: ${allFeatures.length} vs '
            '${MLConstants.totalFeatures} expected',
          );
        }
        return List<double>.filled(MLConstants.totalFeatures, 0.0);
      }

      if (MLConstants.featureMeans.length != MLConstants.totalFeatures ||
          MLConstants.featureStds.length != MLConstants.totalFeatures) {
        if (debugLogging) {
          debugPrint('❌ Scaler coefficients incomplete');
        }
        return List<double>.filled(MLConstants.totalFeatures, 0.0);
      }

      final normalized = <double>[];
      
      for (int i = 0; i < allFeatures.length; i++) {
        final raw = allFeatures[i];
        final mean = MLConstants.featureMeans[i];
        final std = MLConstants.featureStds[i];

        // Avoid division by zero
        if (std == 0.0) {
          normalized.add(0.0);
          continue;
        }

        final normalized_val = (raw - mean) / std;
        normalized.add(normalized_val);
      }

      return normalized;
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Error normalizing features: $e');
      }
      return List<double>.filled(MLConstants.totalFeatures, 0.0);
    }
  }

  // ============== Feature Validation ==============

  /// Validate features are within expected ranges
  /// Helps catch sensor errors or data transmission issues
  bool validateFeatures(List<double> allFeatures) {
    try {
      if (allFeatures.length != MLConstants.totalFeatures) {
        if (debugLogging) {
          debugPrint('⚠️ Invalid feature count: ${allFeatures.length}');
        }
        return false;
      }

      // Check for NaN or infinite values
      for (int i = 0; i < allFeatures.length; i++) {
        final value = allFeatures[i];
        if (value.isNaN || value.isInfinite) {
          if (debugLogging) {
            debugPrint('⚠️ Invalid value at index $i: $value (${MLFeatureOrder.orderedFeatures[i]})');
          }
          return false;
        }
      }

      // Check feature-specific ranges (before normalization)
      // Temperatures: -10°C to 50°C
      for (int i = 0; i < 4; i++) {
        if (allFeatures[i] < -10.0 || allFeatures[i] > 50.0) {
          if (debugLogging) {
            debugPrint('⚠️ Temperature out of range at index $i: ${allFeatures[i]}°C');
          }
          // Don't return false - outliers can happen, just log
        }
      }

      // Pressures: 0-100 kPa
      for (int i = 4; i < 8; i++) {
        if (allFeatures[i] < 0.0 || allFeatures[i] > 100.0) {
          if (debugLogging) {
            debugPrint('⚠️ Pressure out of range at index $i: ${allFeatures[i]} kPa');
          }
        }
      }

      // SpO2: 0-100%
      if (allFeatures[8] < 0.0 || allFeatures[8] > 100.0) {
        if (debugLogging) {
          debugPrint('⚠️ SpO2 out of range: ${allFeatures[8]}%');
        }
      }

      // Heart rate: 0-250 BPM
      if (allFeatures[9] < 0.0 || allFeatures[9] > 250.0) {
        if (debugLogging) {
          debugPrint('⚠️ Heart rate out of range: ${allFeatures[9]} BPM');
        }
      }

      return true;
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Error validating features: $e');
      }
      return false;
    }
  }

  // ============== End-to-End Pipeline ==============

  /// Complete feature engineering pipeline
  /// SensorReading → Normalized feature vector ready for ML inference
  /// 
  /// Returns null if validation fails, otherwise List<double> of 15 features
  List<double>? processSensorReading(SensorReading reading) {
    try {
      if (debugLogging) {
        debugPrint('🔄 Processing sensor reading...');
      }

      // 1. Extract raw features (11)
      final rawFeatures = extractRawFeatures(reading);

      // 2. Calculate engineered features (4)
      final engineeredFeatures = calculateEngineeredFeatures(rawFeatures);

      // 3. Combine all features (15)
      final allFeatures = [...rawFeatures, ...engineeredFeatures];

      // 4. Validate
      if (!validateFeatures(allFeatures)) {
        if (debugLogging) {
          debugPrint('⚠️ Feature validation failed, but proceeding...');
        }
        // Don't stop processing - outliers are possible
      }

      // 5. Normalize
      final normalized = normalizeFeatures(allFeatures);

      if (debugLogging) {
        debugPrint(
          '✅ Features processed: '
          'spO2=${reading.spO2.toStringAsFixed(1)}%, '
          'HR=${reading.heartRate}bpm, '
          'maxTemp=${allFeatures[13].toStringAsFixed(1)}°C, '
          'maxPres=${allFeatures[11].toStringAsFixed(1)}kPa',
        );
      }

      return normalized;
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Error in ML feature pipeline: $e');
      }
      return null;
    }
  }

  // ============== Debugging & Analysis ==============

  /// Get readable feature names and values (for debugging)
  Map<String, double> getFeatureMap(List<double> allFeatures) {
    try {
      if (allFeatures.length != MLFeatureOrder.orderedFeatures.length) {
        return {};
      }

      final featureMap = <String, double>{};
      for (int i = 0; i < allFeatures.length; i++) {
        featureMap[MLFeatureOrder.orderedFeatures[i]] = allFeatures[i];
      }
      return featureMap;
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Error creating feature map: $e');
      }
      return {};
    }
  }

  /// Print feature statistics (for debugging)
  void printFeatureDebug(List<double> allFeatures) {
    if (!debugLogging) return;

    debugPrint('\n📊 FEATURE DEBUG INFO');
    debugPrint('=' * 80);

    final featureMap = getFeatureMap(allFeatures);
    
    debugPrint('\nRaw Features (11):');
    for (int i = 0; i < 11 && i < featureMap.length; i++) {
      final name = MLFeatureOrder.orderedFeatures[i];
      final value = allFeatures[i];
      debugPrint('  ${i.toString().padLeft(2)}: $name = ${value.toStringAsFixed(2)}');
    }

    debugPrint('\nEngineered Features (4):');
    for (int i = 11; i < 15 && i < featureMap.length; i++) {
      final name = MLFeatureOrder.orderedFeatures[i];
      final value = allFeatures[i];
      debugPrint('  ${i.toString().padLeft(2)}: $name = ${value.toStringAsFixed(2)}');
    }

    debugPrint(List.filled(80, '=').join() + '\n');
  }
}
