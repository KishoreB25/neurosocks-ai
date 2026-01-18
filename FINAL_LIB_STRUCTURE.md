# Smart Socks - Enhanced Lib Directory Structure (Firebase + ML Integration)

## Project Architecture Overview

```
lib/
├── main.dart                          # App entry point with Firebase initialization
├── app.dart                           # App configuration & routing with auth state
├── firebase_options.dart              # Firebase configuration (auto-generated)
│
├── core/                              # Core application logic
│   ├── constants/                     # Application-wide constants
│   │   ├── app_colors.dart           # Color palette definitions
│   │   ├── app_strings.dart          # Text strings & localization
│   │   └── sensor_constants.dart     # Sensor-related constants
│   │
│   ├── theme/                         # App theming
│   │   └── app_theme.dart            # Theme configuration
│   │
│   └── utils/                         # NEW: Utility functions
│       ├── ml_helpers.dart           # ML model preprocessing utilities
│       └── validators.dart           # Form validation helpers
│
├── data/                              # Data layer (Models & Services)
│   ├── models/                        # Data models
│   │   ├── sensor_reading.dart       # Sensor data model
│   │   ├── foot_data.dart            # Foot zone data model
│   │   ├── risk_score.dart           # Risk assessment model
│   │   ├── alert.dart                # Alert/notification model
│   │   ├── user_profile.dart         # User profile model
│   │   ├── prediction_result.dart    # NEW: ML prediction results
│   │   └── ml_model_metadata.dart    # NEW: Model version & config
│   │
│   ├── services/                      # Business logic & external services
│   │   ├── firebase_auth_service.dart # NEW: Firebase Authentication
│   │   ├── firestore_service.dart     # NEW: Firestore data operations
│   │   ├── ml_service.dart            # NEW: TFLite model inference
│   │   ├── ble_service.dart           # Real BLE service (replace mock)
│   │   ├── storage_service.dart       # Local storage (Hive + SharedPrefs)
│   │   ├── risk_calculator.dart       # Risk calculation logic
│   │   └── alert_service.dart         # Alert management service
│   │
│   └── repositories/                  # NEW: Data access layer
│       ├── sensor_repository.dart     # Sensor data CRUD operations
│       ├── user_repository.dart       # User data management
│       └── prediction_repository.dart # ML predictions storage
│
├── providers/                         # State management (Provider pattern)
│   ├── auth_provider.dart             # NEW: Authentication state
│   ├── firebase_provider.dart         # NEW: Firebase connectivity
│   ├── ml_provider.dart               # NEW: ML model state & predictions
│   ├── risk_provider.dart            # Risk assessment state
│   ├── sensor_provider.dart          # Sensor data state
│   └── user_provider.dart            # User profile state
│
└── ui/                                # User Interface Layer
    ├── screens/                       # App screens/pages
    │   ├── auth/                      # Authentication screens
    │   │   ├── welcome_screen.dart   # Onboarding screen
    │   │   ├── login_screen.dart     # Email/password login
    │   │   ├── signup_screen.dart    # NEW: User registration
    │   │   ├── forgot_password_screen.dart # NEW: Password reset
    │   │   └── profile_setup_screen.dart # Profile completion
    │   │
    │   └── home/                      # Main app screens
    │       ├── dashboard_screen.dart # Main dashboard with predictions
    │       ├── sensors_screen.dart   # Sensor details & live data
    │       ├── alerts_screen.dart    # Alerts & notifications
    │       ├── predictions_screen.dart # NEW: ML predictions history
    │       └── settings_screen.dart  # App settings & profile
    │
    └── widgets/                       # Reusable UI components
        ├── alert_tile.dart           # Alert notification tile
        ├── connection_status.dart    # BLE/Firebase connection indicator
        ├── foot_heatmap.dart         # Foot sensor heatmap visualization
        ├── loading_shimmer.dart      # Loading skeleton screen
        ├── mini_chart.dart           # Small chart component
        ├── risk_gauge.dart           # Risk level gauge/indicator
        ├── sensor_card.dart          # Sensor data card component
        ├── stat_card.dart            # Statistics card component
        ├── prediction_chart.dart     # NEW: ML prediction visualization
        └── auth_form_field.dart      # NEW: Reusable auth form fields
```

## Architecture Pattern (Enhanced)

| Layer | Purpose | Key Components | New Additions |
|-------|---------|----------------|---------------|
| **UI Layer** | User Interface | Screens, Widgets | Auth screens, Prediction UI |
| **Providers** | State Management | Provider classes | Auth, Firebase, ML providers |
| **Repositories** | Data Access | CRUD operations | Sensor, User, Prediction repos |
| **Services** | Business Logic | External APIs, ML | Firebase Auth, Firestore, TFLite |
| **Data Layer** | Models & Storage | Data structures | Prediction models, ML metadata |
| **Core Layer** | App-wide utilities | Constants, Theme, Utils | ML helpers, Validators |

## New Components Overview

### Firebase Integration
- **Authentication:** Email/password, Google Sign-in, biometric
- **Firestore:** Real-time data sync, offline persistence
- **Cloud Functions:** Server-side prediction processing (optional)

### Machine Learning Integration
- **TFLite Model:** Risk prediction from sensor data
- **Live Inference:** Real-time predictions on streaming data
- **Model Management:** Version control, A/B testing support

### Enhanced Authentication Flow
```
Welcome → Login/Signup → Email Verification → Profile Setup → Dashboard
```

## Data Flow Architecture

### Sensor Data Flow
```
Hardware → BLE Service → Sensor Repository → Firestore ← ML Service
                              ↓                        ↑
                       Local Storage ←→ Prediction Repository
                              ↓                        ↑
                       Sensor Provider ←→ ML Provider
                              ↓                        ↑
                       UI Screens ←→ Prediction UI
```

### Authentication Flow
```
UI Forms → Auth Provider → Firebase Auth Service → Firestore
                              ↓
                       User Repository → Local Storage
                              ↓
                       User Provider → UI Updates
```

## File Count Summary (Enhanced)

- **Entry Points:** 3 files (main.dart, app.dart, firebase_options.dart)
- **Core Layer:** 7 files (constants, theme, utils)
- **Data Models:** 7 files (including prediction models)
- **Data Services:** 8 files (Firebase, ML, BLE, Storage, etc.)
- **Data Repositories:** 3 files (Sensor, User, Prediction)
- **State Providers:** 6 files (Auth, Firebase, ML, Risk, Sensor, User)
- **UI Screens:** 10 files (auth + home screens)
- **UI Widgets:** 10 files (existing + auth + prediction widgets)

**Total Files in Lib:** ~54 files (expanded from ~33)

## Key Integration Points

### Firebase Setup
1. Add Firebase dependencies to `pubspec.yaml`
2. Configure `firebase_options.dart`
3. Initialize Firebase in `main.dart`
4. Set up authentication guards in `app.dart`

### ML Model Integration
1. Add TFLite dependencies
2. Load model in `MLService`
3. Preprocess sensor data for inference
4. Store predictions in Firestore
5. Display results in UI

### Authentication Enhancement
1. Replace mock auth with Firebase Auth
2. Add proper form validation
3. Implement email verification
4. Add password reset functionality
5. Integrate with user profile management

## Migration Path

### Phase 1: Firebase Auth
- Implement `FirebaseAuthService`
- Update `AuthProvider`
- Enhance auth screens
- Add email verification

### Phase 2: Firestore Integration
- Implement `FirestoreService`
- Create repositories
- Update data flow
- Add offline sync

### Phase 3: ML Integration
- Implement `MLService`
- Add prediction models
- Update providers
- Add prediction UI

### Phase 4: Real BLE
- Replace `MockBleService` with real BLE implementation
- Update sensor data flow
- Test end-to-end integration

## Dependencies to Add

```yaml
dependencies:
  firebase_core: ^2.0.0
  firebase_auth: ^4.0.0
  cloud_firestore: ^4.0.0
  tflite_flutter: ^0.10.0
  tflite_flutter_helper: ^0.3.0
  provider: ^6.0.0
  # ... existing dependencies
```

This enhanced architecture provides a scalable foundation for Firebase authentication, Firestore data management, and machine learning predictions while maintaining clean separation of concerns and testability.</content>
<parameter name="filePath">e:\S4\IOC & IOT\PROJECT\neurosocks-ai\smart_socks\ENHANCED_LIB_STRUCTURE.md