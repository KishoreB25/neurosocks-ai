# Smart Socks - Lib Directory Structure

## Project Architecture Overview

```
lib/
├── main.dart                          # Entry point of the application
├── app.dart                           # App configuration & setup
│
├── core/                              # Core application logic
│   ├── constants/                     # Application-wide constants
│   │   ├── app_colors.dart           # Color palette definitions
│   │   ├── app_strings.dart          # Text strings & localization
│   │   └── sensor_constants.dart     # Sensor-related constants
│   │
│   └── theme/                         # App theming
│       └── app_theme.dart            # Theme configuration (colors, fonts, etc.)
│
├── data/                              # Data layer (Models & Services)
│   ├── models/                        # Data models
│   │   ├── alert.dart                # Alert model
│   │   ├── foot_data.dart            # Foot sensor data model
│   │   ├── risk_score.dart           # Risk score calculation model
│   │   ├── sensor_reading.dart       # Sensor reading data model
│   │   └── user_profile.dart         # User profile model
│   │
│   └── services/                      # Business logic & external services
│       ├── alert_service.dart        # Alert management service
│       ├── mock_ble_service.dart     # Bluetooth LE mock service
│       ├── risk_calculator.dart      # Risk calculation logic
│       └── storage_service.dart      # Local storage service
│
├── providers/                         # State management (Providers)
│   ├── risk_provider.dart            # Risk state management
│   ├── sensor_provider.dart          # Sensor data state management
│   └── user_provider.dart            # User data state management
│
└── ui/                                # User Interface Layer
    ├── screens/                       # App screens/pages
    │   ├── auth/                      # Authentication screens
    │   │   ├── welcome_screen.dart   # Welcome/onboarding screen
    │   │   ├── login_screen.dart     # User login screen
    │   │   └── profile_setup_screen.dart # Profile setup screen
    │   │
    │   └── home/                      # Home screens (main app)
    │       ├── dashboard_screen.dart # Main dashboard
    │       ├── sensors_screen.dart   # Sensor details screen
    │       ├── alerts_screen.dart    # Alerts/notifications screen
    │       └── settings_screen.dart  # App settings screen
    │
    └── widgets/                       # Reusable UI components
        ├── alert_tile.dart           # Alert notification tile
        ├── connection_status.dart    # Bluetooth connection indicator
        ├── foot_heatmap.dart         # Foot sensor heatmap visualization
        ├── loading_shimmer.dart      # Loading skeleton screen
        ├── mini_chart.dart           # Small chart component
        ├── risk_gauge.dart           # Risk level gauge/indicator
        ├── sensor_card.dart          # Sensor data card component
        └── stat_card.dart            # Statistics card component
```

## Architecture Pattern

| Layer | Purpose | Contains |
|-------|---------|----------|
| **UI Layer** | User Interface components | Screens, Widgets |
| **Providers** | State Management | Risk, Sensor, User providers |
| **Data Layer** | Business Logic & Data Models | Models, Services |
| **Core Layer** | App-wide utilities | Constants, Theme |

## File Count Summary

- **Entry Points**: 2 files (main.dart, app.dart)
- **Core Layer**: 4 files (constants & theme)
- **Data Models**: 5 files
- **Data Services**: 4 files
- **State Providers**: 3 files
- **UI Screens**: 7 files (auth + home)
- **UI Widgets**: 8 files

**Total Files in Lib**: ~33 files

## Key Components

### Authentication Flow
welcome_screen.dart → login_screen.dart → profile_setup_screen.dart

### Main App Flow
dashboard_screen.dart ← Main Entry
├── sensors_screen.dart
├── alerts_screen.dart
└── settings_screen.dart

### Data Flow
UI Screens → Providers → Services → Models
