/// Machine Learning Feature Order Definition
/// 
/// **CRITICAL**: This order MUST match exactly with the training data
/// Any mismatch will cause the model to produce garbage predictions
/// 
/// Total: 11 raw sensor features + 4 engineered = 15 features
/// (WITHOUT motion sensors - not available in BLE transmission)

class MLFeatureOrder {
  /// The exact order of features expected by the TFLite model
  /// This is extracted from feature_names.pkl from the training pipeline
  static const List<String> orderedFeatures = [
    // ===== RAW SENSOR FEATURES (11) =====
    // Temperature sensors (4)
    'temp_heel',      // [0] - Temperature at heel zone
    'temp_ball',      // [1] - Temperature at ball of foot
    'temp_arch',      // [2] - Temperature at arch
    'temp_toe',       // [3] - Temperature at toe
    
    // Pressure sensors (4)
    'press_heel',     // [4] - Pressure at heel zone (kPa)
    'press_ball',     // [5] - Pressure at ball of foot (kPa)
    'press_arch',     // [6] - Pressure at arch (kPa)
    'press_toe',      // [7] - Pressure at toe (kPa)
    
    // Vital signs (2)
    'spo2',           // [8] - Blood oxygen saturation (%)
    'heartRate',      // [9] - Heart rate (BPM)
    
    // Activity (1)
    'stepCount',      // [10] - Cumulative step count
    
    // ===== ENGINEERED FEATURES (4) =====
    // Derived from raw features (motion sensors excluded)
    'max_pressure',        // [11] - max(pressure sensors)
    'pressure_variance',   // [12] - var(pressure sensors)
    'max_temp',            // [13] - max(temperature sensors)
    'temp_variance',       // [14] - var(temperature sensors)
  ];

  /// Get index of a feature by name
  /// Returns -1 if feature not found
  static int getFeatureIndex(String featureName) {
    return orderedFeatures.indexOf(featureName);
  }

  /// Validate that feature vector has correct length
  static bool isValidFeatureCount(List<double> features) {
    return features.length == 23;
  }

  /// Get feature description
  static String getDescription(String featureName) {
    const descriptions = {
      'temp_heel': 'Temperature at heel',
      'temp_ball': 'Temperature at ball of foot',
      'temp_arch': 'Temperature at arch',
      'temp_toe': 'Temperature at toe',
      'press_heel': 'Pressure at heel zone (kPa)',
      'press_ball': 'Pressure at ball of foot (kPa)',
      'press_arch': 'Pressure at arch (kPa)',
      'press_toe': 'Pressure at toe (kPa)',
      'spo2': 'Blood oxygen saturation (%)',
      'heartRate': 'Heart rate (BPM)',
      'stepCount': 'Cumulative step count',
      'max_pressure': 'Maximum pressure across all zones',
      'pressure_variance': 'Variance of pressure distribution',
      'max_temp': 'Maximum temperature across all zones',
      'temp_variance': 'Variance of temperature distribution',
    };
    return descriptions[featureName] ?? 'Unknown feature';
  }

  /// Group features by category
  static const Map<String, List<String>> featuresByCategory = {
    'temperature': ['temp_heel', 'temp_ball', 'temp_arch', 'temp_toe'],
    'pressure': ['press_heel', 'press_ball', 'press_arch', 'press_toe'],
    'vitals': ['spo2', 'heartRate'],
    'activity': ['stepCount'],
    'engineered_pressure': ['max_pressure', 'pressure_variance'],
    'engineered_temp': ['max_temp', 'temp_variance'],
  };
}
