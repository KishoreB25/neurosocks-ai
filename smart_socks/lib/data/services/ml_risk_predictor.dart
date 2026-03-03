/// Machine Learning Risk Predictor Service
///
/// Runs TFLite model inference on normalized sensor features
/// Returns risk probability (0-1) for diabetic foot ulcer prediction

import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import '../models/sensor_reading.dart';
import '../../core/constants/ml_constants.dart';
import 'ml_feature_engineer.dart';
import 'ml_model_loader.dart';

/// Result from ML inference
class MLPredictionResult {
  /// Raw risk probability (0-1)
  final double riskProbability;

  /// Inference latency in milliseconds
  final int latencyMs;

  /// Whether inference was successful
  final bool success;

  /// Error message if inference failed
  final String? error;

  /// Timestamp of prediction
  final DateTime timestamp;

  /// Risk level classification
  RiskLevel get riskLevel {
    if (riskProbability < MLConstants.riskLowThreshold) {
      return RiskLevel.low;
    } else if (riskProbability < MLConstants.riskModerateThreshold) {
      return RiskLevel.moderate;
    } else {
      return RiskLevel.high;
    }
  }

  /// Risk score (0-100)
  int get riskScore => (riskProbability * 100).round();

  const MLPredictionResult({
    required this.riskProbability,
    required this.latencyMs,
    this.success = true,
    this.error,
    required this.timestamp,
  });

  /// Create error result
  factory MLPredictionResult.error(String message) {
    return MLPredictionResult(
      riskProbability: 0.0,
      latencyMs: 0,
      success: false,
      error: message,
      timestamp: DateTime.now(),
    );
  }

  @override
  String toString() {
    if (!success) {
      return '❌ Prediction failed: $error';
    }
    return '📊 Risk: ${riskScore}% (${riskLevel.name}) in ${latencyMs}ms';
  }
}

/// Risk level classification
enum RiskLevel {
  low,      // < 0.3
  moderate, // 0.3 - 0.6
  high      // > 0.6
}

/// Performs ML inference for risk prediction
class MLRiskPredictor {
  // Singleton pattern
  static final MLRiskPredictor _instance = MLRiskPredictor._internal();

  factory MLRiskPredictor() {
    return _instance;
  }

  MLRiskPredictor._internal() {
    _featureEngineer = MLFeatureEngineer();
    _modelLoader = MLModelLoader();
  }

  // Dependencies
  late final MLFeatureEngineer _featureEngineer;
  late final MLModelLoader _modelLoader;

  // Statistics
  int _totalPredictions = 0;
  double _totalLatency = 0.0;
  List<MLPredictionResult> _predictionHistory = [];
  static const int maxHistorySize = MLConstants.maxHistorySize;

  // Debug logging
  static const bool debugLogging = MLConstants.debugLogging;

  // ============== Getters ==============

  /// Total predictions made
  int get totalPredictions => _totalPredictions;

  /// Average inference latency
  double get averageLatency =>
      _totalPredictions > 0 ? _totalLatency / _totalPredictions : 0.0;

  /// Prediction history
  List<MLPredictionResult> get predictionHistory =>
      List.unmodifiable(_predictionHistory);

  /// Model loader instance
  MLModelLoader get modelLoader => _modelLoader;

  /// Feature engineer instance
  MLFeatureEngineer get featureEngineer => _featureEngineer;

  /// Check if ready to predict
  bool get isReady => _modelLoader.canPredict();

  // ============== Model Initialization ==============

  /// Initialize model (lazy load on first use)
  Future<void> initialize() async {
    try {
      if (debugLogging) {
        debugPrint('🔄 Initializing ML Risk Predictor...');
      }

      if (_modelLoader.isLoaded) {
        if (debugLogging) {
          debugPrint('ℹ️ Model already loaded');
        }
        return;
      }

      await _modelLoader.loadModel();

      if (debugLogging) {
        debugPrint('✅ ML Risk Predictor initialized');
      }
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Initialization failed: $e');
      }
      rethrow;
    }
  }

  // ============== Prediction ==============

  /// Predict risk from sensor reading
  /// Complete pipeline: extract features → normalize → infer
  Future<MLPredictionResult> predictFromReading(SensorReading reading) async {
    try {
      // 1. Extract and engineer features
      final features = _featureEngineer.processSensorReading(reading);

      if (features == null) {
        return MLPredictionResult.error('Failed to extract features');
      }

      // 2. Run inference
      return await predictFromFeatures(features);
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Prediction error: $e');
      }
      return MLPredictionResult.error('Prediction failed: $e');
    }
  }

  /// Predict risk from pre-computed normalized features
  Future<MLPredictionResult> predictFromFeatures(List<double> features) async {
    try {
      // Validate model is ready
      if (!_modelLoader.isLoaded) {
        return MLPredictionResult.error('Model not loaded');
      }

      if (_modelLoader.interpreter == null) {
        return MLPredictionResult.error('Interpreter not available');
      }

      // Validate feature count
      if (features.length != MLConstants.totalFeatures) {
        return MLPredictionResult.error(
          'Feature count mismatch: ${features.length} vs ${MLConstants.totalFeatures}',
        );
      }

      // Measure latency
      final stopwatch = Stopwatch()..start();

      // 1. Prepare input tensor
      final input = _prepareInput(features);

      // 2. Run inference
      final output = _runInference(input);

      if (output == null) {
        stopwatch.stop();
        return MLPredictionResult.error('Inference returned null output');
      }

      // 3. Extract probability
      final probability = _extractProbability(output);

      stopwatch.stop();
      final latency = stopwatch.elapsedMilliseconds;

      // Update statistics
      _totalPredictions++;
      _totalLatency += latency;

      // Create result
      final result = MLPredictionResult(
        riskProbability: probability,
        latencyMs: latency,
        timestamp: DateTime.now(),
      );

      // Store in history
      _addToHistory(result);

      if (debugLogging) {
        debugPrint(
          '🎯 Prediction: ${result.riskScore}% ${result.riskLevel.name} (${latency}ms)',
        );
      }

      return result;
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Inference error: $e');
      }
      return MLPredictionResult.error('Inference failed: $e');
    }
  }

  // ============== Tensor Operations ==============

  /// Prepare input tensor [1×15] from feature vector
  List<List<double>> _prepareInput(List<double> features) {
    if (features.length != MLConstants.totalFeatures) {
      throw ArgumentError(
        'Expected ${MLConstants.totalFeatures} features, got ${features.length}',
      );
    }

    // Reshape: 15 features → [1×15]
    return [features];
  }

  /// Run inference on TFLite model
  List? _runInference(List<List<double>> input) {
    try {
      final interpreter = _modelLoader.interpreter;
      if (interpreter == null) {
        if (debugLogging) debugPrint('❌ Interpreter is null');
        return null;
      }

      // Allocate tensors
      interpreter.allocateTensors();

      // Get input/output tensors
      final inputTensors = interpreter.getInputTensors();
      final outputTensors = interpreter.getOutputTensors();

      if (inputTensors.isEmpty || outputTensors.isEmpty) {
        if (debugLogging) {
          debugPrint('❌ No input/output tensors found');
        }
        return null;
      }

      // Convert input to tensor format (Float32List)
      final flatInput = input.expand((row) => row).toList();
      final inputTensor = Float32List.fromList(flatInput);

      // Set input tensor data
      interpreter.setInput(0, inputTensor);

      // Run inference
      interpreter.invoke();

      // Get output tensor
      final output = interpreter.getOutput(0);

      if (output == null) {
        if (debugLogging) debugPrint('❌ Output is null');
        return null;
      }

      return output;
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Inference error: $e');
      }
      return null;
    }
  }

  /// Extract probability from output tensor
  /// Output shape: [1×1], value: probability 0-1
  double _extractProbability(dynamic output) {
    try {
      if (output is List) {
        // Handle [[[probability]]] or [[probability]]
        dynamic value = output;

        while (value is List && value.isNotEmpty) {
          value = value[0];
        }

        if (value is num) {
          final probability = value.toDouble();

          // Clamp to [0, 1]
          return probability.clamp(0.0, 1.0);
        }
      }

      if (debugLogging) {
        debugPrint('⚠️ Unexpected output type: ${output.runtimeType}');
      }

      return 0.0;
    } catch (e) {
      if (debugLogging) {
        debugPrint('❌ Error extracting probability: $e');
      }
      return 0.0;
    }
  }

  // ============== History Management ==============

  /// Add prediction to history
  void _addToHistory(MLPredictionResult result) {
    _predictionHistory.insert(0, result);

    // Keep history size limited
    if (_predictionHistory.length > maxHistorySize) {
      _predictionHistory = _predictionHistory.sublist(0, maxHistorySize);
    }
  }

  /// Clear prediction history
  void clearHistory() {
    _predictionHistory.clear();
    if (debugLogging) {
      debugPrint('🗑️ Prediction history cleared');
    }
  }

  // ============== Statistics ==============

  /// Get recent average risk (last N predictions)
  double getRecentAverageRisk({int lastN = 10}) {
    if (_predictionHistory.isEmpty) return 0.0;

    final recent = _predictionHistory.take(lastN).toList();
    final sum = recent.fold<double>(
      0.0,
      (prev, result) => prev + result.riskProbability,
    );

    return sum / recent.length;
  }

  /// Get trend (increasing/decreasing risk)
  String getRiskTrend({int window = 5}) {
    if (_predictionHistory.length < 2) return 'neutral';

    final recent = _predictionHistory.take(window).toList();
    if (recent.length < 2) return 'neutral';

    final oldest = recent.last.riskProbability;
    final newest = recent.first.riskProbability;
    final change = newest - oldest;

    if (change > 0.05) return '📈 increasing';
    if (change < -0.05) return '📉 decreasing';
    return '➡️ stable';
  }

  /// Get prediction statistics
  String getStatistics() {
    final divider = List.filled(51, '=').join();
    return '''$divider
  ML PREDICTION STATISTICS
$divider
  Total Predictions:  $_totalPredictions
  Average Latency:    ${averageLatency.toStringAsFixed(2)} ms
  History Size:       ${_predictionHistory.length} / $maxHistorySize
  Recent Avg Risk:    ${(getRecentAverageRisk() * 100).toStringAsFixed(1)}%
  Risk Trend:         ${getRiskTrend()}
  Model Status:       ${_modelLoader.status}
$divider
    ''';
  }

  /// Print statistics to console
  void printStatistics() {
    if (debugLogging) {
      debugPrint(getStatistics());
    }
  }

  // ============== Cleanup ==============

  /// Unload model
  Future<void> dispose() async {
    await _modelLoader.unloadModel();
    clearHistory();

    if (debugLogging) {
      debugPrint('✅ ML Risk Predictor disposed');
    }
  }
}
