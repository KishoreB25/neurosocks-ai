/// Machine Learning Configuration Constants
/// 
/// Centralized configuration for TFLite model inference
/// Ensures consistency across the ML pipeline

class MLConstants {
  // ============== Model Configuration ==============
  /// Path to the TFLite model in assets
  /// Random Forest model trained on 15 features (without motion sensors)
  /// Performance: 92.45% accuracy, 97.98% ROC-AUC
  static const String modelPath = 'lib/assets/models/random_forest_model.tflite';

  // ============== TFLite I/O Configuration ==============
  /// Input tensor shape: [1, 15] (batch=1, features=15)
  static const List<int> inputShape = [1, 15];
  
  /// Input tensor type: Float32
  static const String inputType = 'float32';
  
  /// Output tensor shape: [1, 1] (single probability output)
  static const List<int> outputShape = [1, 1];
  
  /// Output type: Probability (0-1)
  static const String outputType = 'float32';

  // ============== Feature Configuration ==============
  /// Total features: 11 raw + 4 engineered = 15 (WITHOUT motion sensors)
  static const int totalFeatures = 15;
  
  /// Raw sensor features count
  static const int rawFeatureCount = 11;
  
  /// Engineered features count
  static const int engineeredFeatureCount = 4;

  // ============== Inference Configuration ==============
  /// Target inference latency (ms)
  static const int targetLatencyMs = 100;
  
  /// Timeout for model load (ms)
  static const int modelLoadTimeoutMs = 5000;
  
  /// Inference frequency: Run every Nth reading
  /// Set to 1 for every reading, 3 for every 3rd (6 sec interval)
  static const int inferenceFrequency = 1;

  // ============== Risk Score Thresholds ==============
  /// Risk score 0-1 mapping to risk levels
  /// Used to convert ML probability to RiskScore object
  
  static const double riskLowThreshold = 0.3;          // < 0.3 = LOW
  static const double riskModerateThreshold = 0.6;     // 0.3-0.6 = MODERATE
  // > 0.6 = HIGH

  // ============== Data Normalization ==============
  /// StandardScaler mean and std values (must match training data)
  /// These are extracted from scaler.pkl during preprocessing
  /// **CRITICAL**: Order must match feature order exactly
  /// Model Accuracy: 92.45% | ROC-AUC: 97.98% (trained without motion sensors)
  
  static const List<double> featureMeans = [
    34.377127267113607,  // temp_heel
    34.376776436691941,  // temp_ball
    34.384331094983494,  // temp_arch
    34.404600260634702,  // temp_toe
    44.134212921268379,  // press_heel
    43.937746057938966,  // press_ball
    43.832688437183343,  // press_arch
    44.349988690344688,  // press_toe
    92.488721981641774,  // spo2
    94.265249999999995,  // heartRate
    59.760500000000000,  // stepCount
    64.597849484194811,  // max_pressure
    358.746560145869012,  // pressure_variance
    36.387833158608558,  // max_temp
    3.330151855516803   // temp_variance
  ];

  static const List<double> featureStds = [
    0.565352534871213,  // temp_heel
    0.560907760479280,  // temp_ball
    0.569056385618529,  // temp_arch
    0.547891185470064,  // temp_toe
    0.054544126291903,  // press_heel
    0.054640308757458,  // press_ball
    0.054551972382659,  // press_arch
    0.052693993830820,  // press_toe
    0.231752998642186,  // spo2
    0.049385116193493,  // heartRate
    0.029076658782992,  // stepCount
    0.046230085918394,  // max_pressure
    0.002327744445261,  // pressure_variance
    0.499504192726976,  // max_temp
    0.274311583009886   // temp_variance
  ];

  // ============== Fallback Strategy ==============
  /// Enable fallback to threshold-based calculation if ML fails
  static const bool enableFallback = true;
  
  /// Maximum retries for model loading
  static const int modelLoadRetries = 3;

  // ============== Debugging ==============
  /// Enable ML debug logging
  static const bool debugLogging = true;
  
  /// Save prediction history for analysis
  static const bool savePredictionHistory = true;
  
  /// Max predictions to keep in history
  static const int maxHistorySize = 500;

  // ============== Feature Metadata ==============
  static const Map<String, String> featureDescriptions = {
    'temp_heel': 'Temperature at heel',
    'temp_ball': 'Temperature at ball of foot',
    'temp_arch': 'Temperature at arch',
    'temp_toe': 'Temperature at toe',
    'press_heel': 'Pressure at heel (kPa)',
    'press_ball': 'Pressure at ball of foot (kPa)',
    'press_arch': 'Pressure at arch (kPa)',
    'press_toe': 'Pressure at toe (kPa)',
    'spo2': 'Blood oxygen saturation (%)',
    'heartRate': 'Heart rate (BPM)',
    'acc_x': 'Acceleration X-axis (m/s²)',
    'acc_y': 'Acceleration Y-axis (m/s²)',
    'acc_z': 'Acceleration Z-axis (m/s²)',
    'gyro_x': 'Gyroscope X-axis (°/s)',
    'gyro_y': 'Gyroscope Y-axis (°/s)',
    'gyro_z': 'Gyroscope Z-axis (°/s)',
    'stepCount': 'Cumulative step count',
    'max_pressure': 'Maximum pressure across zones',
    'pressure_variance': 'Variance of pressure distribution',
    'max_temp': 'Maximum temperature across zones',
    'temp_variance': 'Variance of temperature distribution',
    'acc_magnitude': 'Magnitude of acceleration vector',
    'gyro_magnitude': 'Magnitude of gyroscope vector',
  };
}
