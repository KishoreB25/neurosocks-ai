# ML Pipeline - Diabetic Foot Ulcer Risk Classification

This folder contains the complete machine learning pipeline for early diabetic foot ulcer detection using multi-sensor wearable data.

##  Folder Structure

```
ML/
├── 01_eda_visualization.ipynb       # Exploratory Data Analysis
├── 02_random_forest_model.ipynb     # Primary Random Forest Model
├── 03_logistic_regression_baseline.ipynb  # Baseline Logistic Regression
├── README.md                        # This file
├── plots/                           # Generated visualizations
│   ├── 01_class_distribution.png
│   ├── 02_temperature_distributions.png
│   ├── 03_temperature_boxplots.png
│   ├── 04_pressure_distributions.png
│   ├── 05_pressure_boxplots.png
│   ├── 06_vital_signs_distributions.png
│   ├── 07_accelerometer_distributions.png
│   ├── 08_gyroscope_distributions.png
│   ├── 09_correlation_heatmap.png
│   ├── 10_target_correlation.png
│   ├── 11_engineered_features.png
│   ├── rf_confusion_matrix.png
│   ├── rf_roc_curve.png
│   ├── rf_feature_importance.png
│   ├── lr_confusion_matrix.png
│   ├── lr_roc_curve.png
│   ├── lr_feature_coefficients.png
│   └── model_comparison.png
└── models/                          # Saved models
    ├── random_forest_model.pkl      # sklearn Random Forest
    ├── logistic_regression_model.pkl # sklearn Logistic Regression
    ├── scaler.pkl                   # StandardScaler for RF
    ├── scaler_lr.pkl                # StandardScaler for LR
    ├── feature_names.pkl            # Feature list
    ├── rf_neural_network.keras      # Keras model (RF equivalent)
    ├── lr_neural_network.keras      # Keras model (LR equivalent)
    ├── random_forest_model.tflite   # TFLite model for mobile
    └── logistic_regression_model.tflite # TFLite model for mobile
```

## 🔬 Dataset

**Source**: `../Synthetic_Data/synthetic_foot_ulcer_dataset_RISK.csv`

### Features Used (17 raw + 6 engineered = 23 total)

#### Raw Sensor Features (17):
| Category | Features |
|----------|----------|
| Temperature (4) | temp_heel, temp_ball, temp_arch, temp_toe |
| Pressure (4) | press_heel, press_ball, press_arch, press_toe |
| Vital Signs (2) | spo2, heartRate |
| Accelerometer (3) | acc_x, acc_y, acc_z |
| Gyroscope (3) | gyro_x, gyro_y, gyro_z |
| Activity (1) | stepCount |

#### Engineered Features (6):
| Feature | Formula |
|---------|---------|
| max_pressure | max(pressure sensors) |
| pressure_variance | var(pressure sensors) |
| max_temp | max(temperature sensors) |
| temp_variance | var(temperature sensors) |
| acc_magnitude | √(acc_x² + acc_y² + acc_z²) |
| gyro_magnitude | √(gyro_x² + gyro_y² + gyro_z²) |

#### Features EXCLUDED from ML:
- `batteryLevel` - System-level, not physiological
- `risk_score` - Model output, NOT input

#### Target:
- `label`: 0 = Normal, 1 = High Risk

## 📓 Notebooks

### 1. `01_eda_visualization.ipynb`
Comprehensive Exploratory Data Analysis:
- Dataset inspection and statistics
- Missing value analysis
- Class distribution analysis
- Feature distributions by class
- Correlation heatmap
- Feature engineering preview

### 2. `02_random_forest_model.ipynb`
Primary classification model:
- Feature engineering pipeline
- Random Forest training with class balancing
- Cross-validation (5-fold)
- Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Feature importance analysis
- Model export (.pkl and .tflite)

### 3. `03_logistic_regression_baseline.ipynb`
Baseline comparison model:
- Same preprocessing pipeline
- Logistic Regression training
- Evaluation and comparison with RF
- Feature coefficient analysis
- Model export (.pkl and .tflite)

##  Quick Start

```python
# 1. Run notebooks in order
#    - 01_eda_visualization.ipynb
#    - 02_random_forest_model.ipynb
#    - 03_logistic_regression_baseline.ipynb

# 2. Load trained model for inference
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Prepare new data (with feature engineering)
# X_new = ...  # Your sensor data with engineered features

# Scale and predict
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
risk_scores = model.predict_proba(X_scaled)[:, 1]
```

##  Mobile Deployment (TFLite)

```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='models/random_forest_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inference
def predict(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Example
# scaled_input = scaler.transform(raw_features).astype(np.float32)
# risk_score = predict(scaled_input)[0][0]
```

##  Risk Score Interpretation

| Risk Score | Level | Action |
|------------|-------|--------|
| < 0.3 | Low | Normal monitoring |
| 0.3 - 0.6 | Moderate | Increased monitoring |
| > 0.6 | High | Alert and recommendation |

##  Important Notes

1. **Risk score is MODEL OUTPUT, not input**
   - Generated as: `risk_score = model.predict_proba(X)[:, 1]`

2. **Always apply same preprocessing**
   - Feature engineering (max, variance, magnitude)
   - StandardScaler transformation

3. **This is for risk classification, NOT medical diagnosis**
   - Results should support clinical decisions, not replace them

## 🔧 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
joblib
```

##  Expected Results

| Metric | Random Forest | Logistic Regression |
|--------|---------------|---------------------|
| Accuracy | ~85-90% | ~80-85% |
| ROC-AUC | ~0.90+ | ~0.85+ |

*Actual values depend on dataset characteristics*
