/// Machine Learning Model Loader Service
///
/// Handles loading and initialization of TFLite model
/// Manages interpreter lifecycle and error handling

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../../core/constants/ml_constants.dart';

/// Exception for model loading errors
class MLModelException implements Exception {
  final String message;
  final dynamic originalError;

  MLModelException(this.message, [this.originalError]);

  @override
  String toString() => '❌ ML Model Error: $message${originalError != null ? '\n  Caused by: $originalError' : ''}';
}

/// Loads and manages TFLite model interpreter
class MLModelLoader {
  // Singleton pattern
  static final MLModelLoader _instance = MLModelLoader._internal();

  factory MLModelLoader() {
    return _instance;
  }

  MLModelLoader._internal();

  // State management
  Interpreter? _interpreter;
  bool _isLoaded = false;
  bool _isLoading = false;
  DateTime? _loadedAt;
  String? _lastError;

  // Debug logging
  static const bool debugLogging = MLConstants.debugLogging;

  // ============== Getters ==============

  /// Check if model is loaded
  bool get isLoaded => _isLoaded;

  /// Check if model loading is in progress
  bool get isLoading => _isLoading;

  /// Get the interpreter instance
  Interpreter? get interpreter => _interpreter;

  /// When was the model loaded
  DateTime? get loadedAt => _loadedAt;

  /// Last error message
  String? get lastError => _lastError;

  /// Model status string (for UI display)
  String get status {
    if (_isLoading) return '⏳ Loading model...';
    if (_isLoaded) return '✅ Model loaded (${_getModelAge()})';
    if (_lastError != null) return '❌ Error: $_lastError';
    return '⚠️ Model not loaded';
  }

  /// Get how long ago model was loaded
  String _getModelAge() {
    if (_loadedAt == null) return 'unknown';
    final age = DateTime.now().difference(_loadedAt!);
    if (age.inSeconds < 60) return '${age.inSeconds}s ago';
    if (age.inMinutes < 60) return '${age.inMinutes}m ago';
    return '${age.inHours}h ago';
  }

  // ============== Model Loading ==============

  /// Load the TFLite model from assets
  /// Handles both Android and iOS asset loading
  Future<void> loadModel() async {
    // Don't reload if already loaded
    if (_isLoaded) {
      if (debugLogging) {
        debugPrint('ℹ️ Model already loaded, skipping...');
      }
      return;
    }

    // Prevent multiple simultaneous loads
    if (_isLoading) {
      if (debugLogging) {
        debugPrint('ℹ️ Model loading already in progress...');
      }
      return;
    }

    _isLoading = true;
    _lastError = null;

    try {
      if (debugLogging) {
        debugPrint('🔄 Loading TFLite model: ${MLConstants.modelPath}');
      }

      // Load model file from assets
      final modelBytes = await _loadModelBytes();
      
      if (debugLogging) {
        debugPrint('📦 Model bytes loaded: ${(modelBytes.length / 1024 / 1024).toStringAsFixed(2)} MB');
      }

      // Create interpreter from bytes
      _interpreter = await Interpreter.fromBuffer(modelBytes);

      // Verify model
      _verifyInterpreter();

      _isLoaded = true;
      _loadedAt = DateTime.now();
      _isLoading = false;

      if (debugLogging) {
        debugPrint('✅ Model loaded successfully!');
        _printModelInfo();
      }
    } on MLModelException catch (e) {
      _lastError = e.message;
      _isLoaded = false;
      _isLoading = false;
      if (debugLogging) debugPrint('$e');
      rethrow;
    } catch (e) {
      _lastError = 'Unknown error: $e';
      _isLoaded = false;
      _isLoading = false;
      if (debugLogging) debugPrint('❌ Unexpected error loading model: $e');
      throw MLModelException('Failed to load model', e);
    }
  }

  /// Load model bytes from assets
  /// Handles platform-specific asset loading
  Future<Uint8List> _loadModelBytes() async {
    try {
      // Extract path without 'lib/assets/' prefix
      final assetPath = MLConstants.modelPath.replaceFirst('lib/', '');
      
      if (debugLogging) {
        debugPrint('📂 Loading asset: $assetPath');
      }

      final data = await rootBundle.load(assetPath);
      
      if (data.lengthInBytes == 0) {
        throw MLModelException('Model file is empty or not found: $assetPath');
      }

      return data.buffer.asUint8List();
    } on PlatformException catch (e) {
      throw MLModelException('Asset loading failed: ${e.message}', e);
    } catch (e) {
      throw MLModelException('Error loading model bytes', e);
    }
  }

  /// Verify interpreter is properly initialized
  void _verifyInterpreter() {
    if (_interpreter == null) {
      throw MLModelException('Interpreter is null after creation');
    }

    try {
      final inputDetails = _interpreter!.getInputTensors();
      final outputDetails = _interpreter!.getOutputTensors();

      if (inputDetails.isEmpty) {
        throw MLModelException('Model has no input tensors');
      }
      if (outputDetails.isEmpty) {
        throw MLModelException('Model has no output tensors');
      }

      if (debugLogging) {
        debugPrint('✅ Interpreter verified:');
        debugPrint('   - Input tensors: ${inputDetails.length}');
        debugPrint('   - Output tensors: ${outputDetails.length}');
      }
    } catch (e) {
      throw MLModelException('Tensor verification failed', e);
    }
  }

  // ============== Model Information ==============

  /// Get input tensor details
  List<Object>? getInputTensors() {
    return _interpreter?.getInputTensors();
  }

  /// Get output tensor details
  List<Object>? getOutputTensors() {
    return _interpreter?.getOutputTensors();
  }

  /// Get model info as string (for debugging)
  String getModelInfo() {
    if (!_isLoaded || _interpreter == null) {
      return '❌ Model not loaded';
    }

    try {
      final inputs = _interpreter!.getInputTensors();
      final outputs = _interpreter!.getOutputTensors();

      String info = '📊 Model Information\n';
      info += List.filled(50, '─').join() + '\n';
      info += 'Status: ✅ Loaded\n';
      info += 'Loaded at: $_loadedAt\n';
      info += 'Input tensors: ${inputs.length}\n';
      info += 'Output tensors: ${outputs.length}\n';
      info += List.filled(50, '─').join();

      return info;
    } catch (e) {
      return '⚠️ Error getting model info: $e';
    }
  }

  /// Print model information to console
  void _printModelInfo() {
    if (!debugLogging) return;
    debugPrint('\n' + getModelInfo() + '\n');
  }

  // ============== Model Cleanup ==============

  /// Unload and dispose of model
  Future<void> unloadModel() async {
    try {
      if (_interpreter != null) {
        _interpreter!.close();
        _interpreter = null;
      }

      _isLoaded = false;
      _loadedAt = null;

      if (debugLogging) {
        debugPrint('✅ Model unloaded and cleaned up');
      }
    } catch (e) {
      if (debugLogging) {
        debugPrint('⚠️ Error unloading model: $e');
      }
    }
  }

  /// Reload model (unload then load)
  Future<void> reloadModel() async {
    if (debugLogging) {
      debugPrint('🔄 Reloading model...');
    }

    await unloadModel();
    await loadModel();

    if (debugLogging) {
      debugPrint('✅ Model reloaded');
    }
  }

  // ============== Utility Methods ==============

  /// Check if model is ready for inference
  bool canPredict() {
    return _isLoaded && _interpreter != null && !_isLoading;
  }

  /// Reset error state
  void clearError() {
    _lastError = null;
  }

  /// Get detailed status report
  String getStatusReport() {
    return '''
═══════════════════════════════════════════════════════
  ML MODEL STATUS REPORT
═══════════════════════════════════════════════════════
  Loaded:       $_isLoaded
  Loading:      $_isLoading
  Can Predict:  ${canPredict()}
  Loaded At:    $_loadedAt
  Last Error:   $_lastError
  Model Path:   ${MLConstants.modelPath}
  Input Shape:  ${MLConstants.inputShape}
  Output Shape: ${MLConstants.outputShape}
═══════════════════════════════════════════════════════
    ''';
  }
}
