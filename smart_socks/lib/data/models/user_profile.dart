import 'package:equatable/equatable.dart';

/// Represents a user's profile and health information
class UserProfile extends Equatable {
  /// Unique identifier (Firebase UID)
  final String id;

  /// User's email address
  final String email;

  /// Display name
  final String? name;

  /// User's age (for risk factor calculation)
  final int? age;

  /// Type of diabetes (if any)
  final DiabetesType? diabetesType;

  /// Years since diabetes diagnosis
  final int? diabetesYears;

  /// Phone number (optional, for emergency contacts)
  final String? phone;

  /// Emergency contact name
  final String? emergencyContactName;

  /// Emergency contact phone
  final String? emergencyContactPhone;

  /// Profile photo URL
  final String? photoUrl;

  /// Account creation timestamp
  final DateTime createdAt;

  /// Last profile update timestamp
  final DateTime? updatedAt;

  /// Last login timestamp
  final DateTime? lastLoginAt;

  /// User settings and preferences
  final UserSettings settings;

  /// Additional health information
  final HealthInfo? healthInfo;

  const UserProfile({
    required this.id,
    required this.email,
    this.name,
    this.age,
    this.diabetesType,
    this.diabetesYears,
    this.phone,
    this.emergencyContactName,
    this.emergencyContactPhone,
    this.photoUrl,
    required this.createdAt,
    this.updatedAt,
    this.lastLoginAt,
    this.settings = const UserSettings(),
    this.healthInfo,
  });

  /// Create a new user profile with default values
  factory UserProfile.create({
    required String id,
    required String email,
    String? name,
  }) {
    return UserProfile(
      id: id,
      email: email,
      name: name,
      createdAt: DateTime.now(),
      settings: const UserSettings(),
    );
  }

  /// Create an empty profile
  factory UserProfile.empty() {
    return UserProfile(
      id: '',
      email: '',
      createdAt: DateTime.now(),
    );
  }

  // ============== Getters ==============

  /// Display name or email prefix
  String get displayName {
    if (name != null && name!.isNotEmpty) return name!;
    return email.split('@').first;
  }

  /// First name (first word of name)
  String get firstName {
    if (name == null || name!.isEmpty) return displayName;
    return name!.split(' ').first;
  }

  /// Initials for avatar
  String get initials {
    if (name != null && name!.isNotEmpty) {
      final parts = name!.split(' ');
      if (parts.length >= 2) {
        return '${parts[0][0]}${parts[1][0]}'.toUpperCase();
      }
      return name![0].toUpperCase();
    }
    return email[0].toUpperCase();
  }

  /// Check if profile is complete
  bool get isProfileComplete {
    return name != null && 
           name!.isNotEmpty && 
           age != null && 
           diabetesType != null;
  }

  /// Check if has emergency contact
  bool get hasEmergencyContact {
    return emergencyContactName != null && 
           emergencyContactPhone != null &&
           emergencyContactName!.isNotEmpty &&
           emergencyContactPhone!.isNotEmpty;
  }

  /// Get diabetes status display string
  String get diabetesStatus {
    if (diabetesType == null) return 'Not specified';
    if (diabetesType == DiabetesType.none) return 'No diabetes';
    
    String status = diabetesType!.displayName;
    if (diabetesYears != null && diabetesYears! > 0) {
      status += ' ($diabetesYears years)';
    }
    return status;
  }

  /// Risk factor based on age and diabetes
  int get baseRiskFactor {
    int factor = 0;
    
    // Age factor
    if (age != null) {
      if (age! >= 65) {
        factor += 20;
      } else if (age! >= 55) {
        factor += 15;
      } else if (age! >= 45) {
        factor += 10;
      } else if (age! >= 35) {
        factor += 5;
      }
    }
    
    // Diabetes factor
    if (diabetesType != null) {
      switch (diabetesType!) {
        case DiabetesType.type1:
          factor += 25;
          break;
        case DiabetesType.type2:
          factor += 20;
          break;
        case DiabetesType.preDiabetes:
          factor += 10;
          break;
        case DiabetesType.gestational:
          factor += 15;
          break;
        case DiabetesType.none:
          break;
      }
    }
    
    // Duration factor
    if (diabetesYears != null) {
      if (diabetesYears! >= 20) {
        factor += 15;
      } else if (diabetesYears! >= 10) factor += 10;
      else if (diabetesYears! >= 5) factor += 5;
    }
    
    return factor.clamp(0, 50);
  }

  // ============== Serialization ==============

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'name': name,
      'age': age,
      'diabetesType': diabetesType?.name,
      'diabetesYears': diabetesYears,
      'phone': phone,
      'emergencyContactName': emergencyContactName,
      'emergencyContactPhone': emergencyContactPhone,
      'photoUrl': photoUrl,
      'createdAt': createdAt.millisecondsSinceEpoch,
      'updatedAt': updatedAt?.millisecondsSinceEpoch,
      'lastLoginAt': lastLoginAt?.millisecondsSinceEpoch,
      'settings': settings.toJson(),
      'healthInfo': healthInfo?.toJson(),
    };
  }

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      id: json['id'] ?? '',
      email: json['email'] ?? '',
      name: json['name'],
      age: json['age'],
      diabetesType: json['diabetesType'] != null 
          ? DiabetesType.fromString(json['diabetesType']) 
          : null,
      diabetesYears: json['diabetesYears'],
      phone: json['phone'],
      emergencyContactName: json['emergencyContactName'],
      emergencyContactPhone: json['emergencyContactPhone'],
      photoUrl: json['photoUrl'],
      createdAt: json['createdAt'] != null
          ? DateTime.fromMillisecondsSinceEpoch(json['createdAt'])
          : DateTime.now(),
      updatedAt: json['updatedAt'] != null
          ? DateTime.fromMillisecondsSinceEpoch(json['updatedAt'])
          : null,
      lastLoginAt: json['lastLoginAt'] != null
          ? DateTime.fromMillisecondsSinceEpoch(json['lastLoginAt'])
          : null,
      settings: json['settings'] != null
          ? UserSettings.fromJson(json['settings'])
          : const UserSettings(),
      healthInfo: json['healthInfo'] != null
          ? HealthInfo.fromJson(json['healthInfo'])
          : null,
    );
  }

  /// Create a copy with modified fields
  UserProfile copyWith({
    String? id,
    String? email,
    String? name,
    int? age,
    DiabetesType? diabetesType,
    int? diabetesYears,
    String? phone,
    String? emergencyContactName,
    String? emergencyContactPhone,
    String? photoUrl,
    DateTime? createdAt,
    DateTime? updatedAt,
    DateTime? lastLoginAt,
    UserSettings? settings,
    HealthInfo? healthInfo,
  }) {
    return UserProfile(
      id: id ?? this.id,
      email: email ?? this.email,
      name: name ?? this.name,
      age: age ?? this.age,
      diabetesType: diabetesType ?? this.diabetesType,
      diabetesYears: diabetesYears ?? this.diabetesYears,
      phone: phone ?? this.phone,
      emergencyContactName: emergencyContactName ?? this.emergencyContactName,
      emergencyContactPhone: emergencyContactPhone ?? this.emergencyContactPhone,
      photoUrl: photoUrl ?? this.photoUrl,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
      lastLoginAt: lastLoginAt ?? this.lastLoginAt,
      settings: settings ?? this.settings,
      healthInfo: healthInfo ?? this.healthInfo,
    );
  }

  @override
  List<Object?> get props => [
        id,
        email,
        name,
        age,
        diabetesType,
        diabetesYears,
        phone,
        emergencyContactName,
        emergencyContactPhone,
        photoUrl,
        createdAt,
        updatedAt,
        lastLoginAt,
        settings,
        healthInfo,
      ];

  @override
  String toString() => 'UserProfile(id: $id, email: $email, name: $name)';
}

/// User settings and preferences
class UserSettings extends Equatable {
  /// Enable push notifications
  final bool notificationsEnabled;

  /// Enable critical alerts (even if notifications disabled)
  final bool criticalAlertsEnabled;

  /// Enable warning alerts
  final bool warningAlertsEnabled;

  /// Enable daily summary notification
  final bool dailySummaryEnabled;

  /// Time for daily summary (hour of day, 0-23)
  final int dailySummaryHour;

  /// Enable sound for notifications
  final bool soundEnabled;

  /// Enable vibration for notifications
  final bool vibrationEnabled;

  /// Temperature unit (celsius/fahrenheit)
  final TemperatureUnit temperatureUnit;

  /// Theme mode (light/dark/system)
  final ThemeMode themeMode;

  /// Data sync interval in minutes
  final int syncIntervalMinutes;

  /// Enable offline data caching
  final bool offlineCacheEnabled;

  /// Share data with healthcare provider
  final bool shareWithProvider;

  /// Healthcare provider email (if sharing enabled)
  final String? providerEmail;

  const UserSettings({
    this.notificationsEnabled = true,
    this.criticalAlertsEnabled = true,
    this.warningAlertsEnabled = true,
    this.dailySummaryEnabled = true,
    this.dailySummaryHour = 20, // 8 PM
    this.soundEnabled = true,
    this.vibrationEnabled = true,
    this.temperatureUnit = TemperatureUnit.celsius,
    this.themeMode = ThemeMode.system,
    this.syncIntervalMinutes = 5,
    this.offlineCacheEnabled = true,
    this.shareWithProvider = false,
    this.providerEmail,
  });

  Map<String, dynamic> toJson() {
    return {
      'notificationsEnabled': notificationsEnabled,
      'criticalAlertsEnabled': criticalAlertsEnabled,
      'warningAlertsEnabled': warningAlertsEnabled,
      'dailySummaryEnabled': dailySummaryEnabled,
      'dailySummaryHour': dailySummaryHour,
      'soundEnabled': soundEnabled,
      'vibrationEnabled': vibrationEnabled,
      'temperatureUnit': temperatureUnit.name,
      'themeMode': themeMode.name,
      'syncIntervalMinutes': syncIntervalMinutes,
      'offlineCacheEnabled': offlineCacheEnabled,
      'shareWithProvider': shareWithProvider,
      'providerEmail': providerEmail,
    };
  }

  factory UserSettings.fromJson(Map<String, dynamic> json) {
    return UserSettings(
      notificationsEnabled: json['notificationsEnabled'] ?? true,
      criticalAlertsEnabled: json['criticalAlertsEnabled'] ?? true,
      warningAlertsEnabled: json['warningAlertsEnabled'] ?? true,
      dailySummaryEnabled: json['dailySummaryEnabled'] ?? true,
      dailySummaryHour: json['dailySummaryHour'] ?? 20,
      soundEnabled: json['soundEnabled'] ?? true,
      vibrationEnabled: json['vibrationEnabled'] ?? true,
      temperatureUnit: TemperatureUnit.fromString(json['temperatureUnit']),
      themeMode: ThemeMode.fromString(json['themeMode']),
      syncIntervalMinutes: json['syncIntervalMinutes'] ?? 5,
      offlineCacheEnabled: json['offlineCacheEnabled'] ?? true,
      shareWithProvider: json['shareWithProvider'] ?? false,
      providerEmail: json['providerEmail'],
    );
  }

  UserSettings copyWith({
    bool? notificationsEnabled,
    bool? criticalAlertsEnabled,
    bool? warningAlertsEnabled,
    bool? dailySummaryEnabled,
    int? dailySummaryHour,
    bool? soundEnabled,
    bool? vibrationEnabled,
    TemperatureUnit? temperatureUnit,
    ThemeMode? themeMode,
    int? syncIntervalMinutes,
    bool? offlineCacheEnabled,
    bool? shareWithProvider,
    String? providerEmail,
  }) {
    return UserSettings(
      notificationsEnabled: notificationsEnabled ?? this.notificationsEnabled,
      criticalAlertsEnabled: criticalAlertsEnabled ?? this.criticalAlertsEnabled,
      warningAlertsEnabled: warningAlertsEnabled ?? this.warningAlertsEnabled,
      dailySummaryEnabled: dailySummaryEnabled ?? this.dailySummaryEnabled,
      dailySummaryHour: dailySummaryHour ?? this.dailySummaryHour,
      soundEnabled: soundEnabled ?? this.soundEnabled,
      vibrationEnabled: vibrationEnabled ?? this.vibrationEnabled,
      temperatureUnit: temperatureUnit ?? this.temperatureUnit,
      themeMode: themeMode ?? this.themeMode,
      syncIntervalMinutes: syncIntervalMinutes ?? this.syncIntervalMinutes,
      offlineCacheEnabled: offlineCacheEnabled ?? this.offlineCacheEnabled,
      shareWithProvider: shareWithProvider ?? this.shareWithProvider,
      providerEmail: providerEmail ?? this.providerEmail,
    );
  }

  @override
  List<Object?> get props => [
        notificationsEnabled,
        criticalAlertsEnabled,
        warningAlertsEnabled,
        dailySummaryEnabled,
        dailySummaryHour,
        soundEnabled,
        vibrationEnabled,
        temperatureUnit,
        themeMode,
        syncIntervalMinutes,
        offlineCacheEnabled,
        shareWithProvider,
        providerEmail,
      ];
}

/// Additional health information
class HealthInfo extends Equatable {
  /// Known neuropathy
  final bool hasNeuropathy;

  /// Previous foot ulcer history
  final bool hasPreviousUlcer;

  /// Peripheral artery disease
  final bool hasPAD;

  /// Chronic kidney disease
  final bool hasCKD;

  /// Retinopathy
  final bool hasRetinopathy;

  /// Blood pressure issues
  final bool hasHypertension;

  /// Smoking status
  final SmokingStatus smokingStatus;

  /// Weight in kg
  final double? weight;

  /// Height in cm
  final double? height;

  /// Any additional notes
  final String? notes;

  const HealthInfo({
    this.hasNeuropathy = false,
    this.hasPreviousUlcer = false,
    this.hasPAD = false,
    this.hasCKD = false,
    this.hasRetinopathy = false,
    this.hasHypertension = false,
    this.smokingStatus = SmokingStatus.never,
    this.weight,
    this.height,
    this.notes,
  });

  /// Calculate BMI
  double? get bmi {
    if (weight == null || height == null || height == 0) return null;
    final heightM = height! / 100;
    return weight! / (heightM * heightM);
  }

  /// BMI category
  String? get bmiCategory {
    final bmiValue = bmi;
    if (bmiValue == null) return null;
    if (bmiValue < 18.5) return 'Underweight';
    if (bmiValue < 25) return 'Normal';
    if (bmiValue < 30) return 'Overweight';
    return 'Obese';
  }

  /// Additional risk score from health factors
  int get additionalRiskScore {
    int score = 0;
    if (hasNeuropathy) score += 20;
    if (hasPreviousUlcer) score += 25;
    if (hasPAD) score += 15;
    if (hasCKD) score += 10;
    if (hasRetinopathy) score += 10;
    if (hasHypertension) score += 5;
    if (smokingStatus == SmokingStatus.current) score += 15;
    if (smokingStatus == SmokingStatus.former) score += 5;
    return score.clamp(0, 50);
  }

  Map<String, dynamic> toJson() {
    return {
      'hasNeuropathy': hasNeuropathy,
      'hasPreviousUlcer': hasPreviousUlcer,
      'hasPAD': hasPAD,
      'hasCKD': hasCKD,
      'hasRetinopathy': hasRetinopathy,
      'hasHypertension': hasHypertension,
      'smokingStatus': smokingStatus.name,
      'weight': weight,
      'height': height,
      'notes': notes,
    };
  }

  factory HealthInfo.fromJson(Map<String, dynamic> json) {
    return HealthInfo(
      hasNeuropathy: json['hasNeuropathy'] ?? false,
      hasPreviousUlcer: json['hasPreviousUlcer'] ?? false,
      hasPAD: json['hasPAD'] ?? false,
      hasCKD: json['hasCKD'] ?? false,
      hasRetinopathy: json['hasRetinopathy'] ?? false,
      hasHypertension: json['hasHypertension'] ?? false,
      smokingStatus: SmokingStatus.fromString(json['smokingStatus']),
      weight: json['weight']?.toDouble(),
      height: json['height']?.toDouble(),
      notes: json['notes'],
    );
  }

  HealthInfo copyWith({
    bool? hasNeuropathy,
    bool? hasPreviousUlcer,
    bool? hasPAD,
    bool? hasCKD,
    bool? hasRetinopathy,
    bool? hasHypertension,
    SmokingStatus? smokingStatus,
    double? weight,
    double? height,
    String? notes,
  }) {
    return HealthInfo(
      hasNeuropathy: hasNeuropathy ?? this.hasNeuropathy,
      hasPreviousUlcer: hasPreviousUlcer ?? this.hasPreviousUlcer,
      hasPAD: hasPAD ?? this.hasPAD,
      hasCKD: hasCKD ?? this.hasCKD,
      hasRetinopathy: hasRetinopathy ?? this.hasRetinopathy,
      hasHypertension: hasHypertension ?? this.hasHypertension,
      smokingStatus: smokingStatus ?? this.smokingStatus,
      weight: weight ?? this.weight,
      height: height ?? this.height,
      notes: notes ?? this.notes,
    );
  }

  @override
  List<Object?> get props => [
        hasNeuropathy,
        hasPreviousUlcer,
        hasPAD,
        hasCKD,
        hasRetinopathy,
        hasHypertension,
        smokingStatus,
        weight,
        height,
        notes,
      ];
}

/// Types of diabetes
enum DiabetesType {
  none,
  type1,
  type2,
  preDiabetes,
  gestational;

  static DiabetesType fromString(String? value) {
    switch (value?.toLowerCase()) {
      case 'type1':
        return DiabetesType.type1;
      case 'type2':
        return DiabetesType.type2;
      case 'prediabetes':
      case 'pre_diabetes':
        return DiabetesType.preDiabetes;
      case 'gestational':
        return DiabetesType.gestational;
      case 'none':
      default:
        return DiabetesType.none;
    }
  }

  String get displayName {
    switch (this) {
      case DiabetesType.none:
        return 'None';
      case DiabetesType.type1:
        return 'Type 1 Diabetes';
      case DiabetesType.type2:
        return 'Type 2 Diabetes';
      case DiabetesType.preDiabetes:
        return 'Pre-Diabetes';
      case DiabetesType.gestational:
        return 'Gestational Diabetes';
    }
  }
}

/// Temperature unit preference
enum TemperatureUnit {
  celsius,
  fahrenheit;

  static TemperatureUnit fromString(String? value) {
    switch (value?.toLowerCase()) {
      case 'fahrenheit':
      case 'f':
        return TemperatureUnit.fahrenheit;
      default:
        return TemperatureUnit.celsius;
    }
  }

  String get symbol {
    switch (this) {
      case TemperatureUnit.celsius:
        return '°C';
      case TemperatureUnit.fahrenheit:
        return '°F';
    }
  }

  /// Convert celsius to this unit
  double convert(double celsius) {
    if (this == TemperatureUnit.fahrenheit) {
      return (celsius * 9 / 5) + 32;
    }
    return celsius;
  }
}

/// Theme mode preference
enum ThemeMode {
  light,
  dark,
  system;

  static ThemeMode fromString(String? value) {
    switch (value?.toLowerCase()) {
      case 'light':
        return ThemeMode.light;
      case 'dark':
        return ThemeMode.dark;
      default:
        return ThemeMode.system;
    }
  }

  String get displayName {
    switch (this) {
      case ThemeMode.light:
        return 'Light';
      case ThemeMode.dark:
        return 'Dark';
      case ThemeMode.system:
        return 'System';
    }
  }
}

/// Smoking status
enum SmokingStatus {
  never,
  former,
  current;

  static SmokingStatus fromString(String? value) {
    switch (value?.toLowerCase()) {
      case 'former':
        return SmokingStatus.former;
      case 'current':
        return SmokingStatus.current;
      default:
        return SmokingStatus.never;
    }
  }

  String get displayName {
    switch (this) {
      case SmokingStatus.never:
        return 'Never smoked';
      case SmokingStatus.former:
        return 'Former smoker';
      case SmokingStatus.current:
        return 'Current smoker';
    }
  }
}
