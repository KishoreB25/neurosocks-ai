/// All text strings used in the NeuroSocks app
/// Centralized for easy localization and maintenance
class AppStrings {
  AppStrings._();

  // ============== App Info ==============
  static const String appName = 'NeuroSocks';
  static const String appTagline = 'Smart Foot Health Monitoring';
  static const String appVersion = '1.0.0';

  // ============== Navigation ==============
  static const String navDashboard = 'Dashboard';
  static const String navFootMap = 'Foot Map';
  static const String navHistory = 'History';
  static const String navAlerts = 'Alerts';
  static const String navSettings = 'Settings';
  static const String navProfile = 'Profile';

  // ============== Auth Screens ==============
  static const String login = 'Login';
  static const String register = 'Register';
  static const String logout = 'Logout';
  static const String email = 'Email';
  static const String password = 'Password';
  static const String confirmPassword = 'Confirm Password';
  static const String forgotPassword = 'Forgot Password?';
  static const String resetPassword = 'Reset Password';
  static const String sendResetLink = 'Send Reset Link';
  static const String dontHaveAccount = "Don't have an account? ";
  static const String alreadyHaveAccount = 'Already have an account? ';
  static const String signUp = 'Sign Up';
  static const String signIn = 'Sign In';
  static const String orContinueWith = 'Or continue with';
  static const String google = 'Google';

  // ============== Dashboard ==============
  static const String dailyRiskScore = 'Daily Risk Score';
  static const String currentReadings = 'Current Readings';
  static const String lastUpdated = 'Last updated';
  static const String monitoring = 'Monitoring';
  static const String startMonitoring = 'Start Monitoring';
  static const String stopMonitoring = 'Stop Monitoring';
  static const String connected = 'Connected';
  static const String disconnected = 'Disconnected';
  static const String connecting = 'Connecting...';
  static const String batteryLevel = 'Battery';

  // ============== Risk Levels ==============
  static const String riskLow = 'Low Risk';
  static const String riskModerate = 'Moderate Risk';
  static const String riskHigh = 'High Risk';
  static const String riskCritical = 'Critical Risk';
  static const String riskScore = 'Risk Score';
  static const String riskFactors = 'Risk Factors';
  static const String recommendations = 'Recommendations';
  static const String lookingGood = 'Looking Good!';
  static const String needsAttention = 'Needs Attention';
  static const String takeAction = 'Take Action';
  static const String seekHelp = 'Seek Medical Advice';

  // ============== Sensors ==============
  static const String temperature = 'Temperature';
  static const String pressure = 'Pressure';
  static const String bloodOxygen = 'Blood Oxygen';
  static const String spO2 = 'SpO₂';
  static const String heartRate = 'Heart Rate';
  static const String gait = 'Gait';
  static const String activity = 'Activity';
  static const String steps = 'Steps';
  static const String stability = 'Stability';

  // ============== Foot Zones ==============
  static const String leftFoot = 'Left Foot';
  static const String rightFoot = 'Right Foot';
  static const String heel = 'Heel';
  static const String ball = 'Ball';
  static const String arch = 'Arch';
  static const String toe = 'Toe';
  static const String zone1 = 'Zone 1 (Heel)';
  static const String zone2 = 'Zone 2 (Ball)';
  static const String zone3 = 'Zone 3 (Arch)';
  static const String zone4 = 'Zone 4 (Toe)';

  // ============== Units ==============
  static const String celsius = '°C';
  static const String fahrenheit = '°F';
  static const String kpa = 'kPa';
  static const String percent = '%';
  static const String bpm = 'BPM';
  static const String stepsUnit = 'steps';

  // ============== Activity Types ==============
  static const String walking = 'Walking';
  static const String standing = 'Standing';
  static const String sitting = 'Sitting';
  static const String resting = 'Resting';
  static const String running = 'Running';
  static const String unknown = 'Unknown';

  // ============== History ==============
  static const String today = 'Today';
  static const String yesterday = 'Yesterday';
  static const String thisWeek = 'This Week';
  static const String thisMonth = 'This Month';
  static const String custom = 'Custom';
  static const String day = 'Day';
  static const String week = 'Week';
  static const String month = 'Month';
  static const String average = 'Average';
  static const String highest = 'Highest';
  static const String lowest = 'Lowest';
  static const String trend = 'Trend';
  static const String noData = 'No data available';
  static const String dailySummary = 'Daily Summary';

  // ============== Alerts ==============
  static const String alerts = 'Alerts';
  static const String noAlerts = 'No alerts';
  static const String allClear = 'All Clear!';
  static const String markAsRead = 'Mark as read';
  static const String clearAll = 'Clear all';
  static const String alertInfo = 'Info';
  static const String alertWarning = 'Warning';
  static const String alertCritical = 'Critical';
  static const String newAlert = 'New Alert';
  static const String viewDetails = 'View Details';

  // ============== Alert Messages ==============
  static const String highTempAlert = 'Elevated temperature detected';
  static const String highPressureAlert = 'High pressure detected';
  static const String lowSpO2Alert = 'Low blood oxygen level';
  static const String gaitAbnormalAlert = 'Gait abnormality detected';
  static const String tempAsymmetryAlert = 'Temperature asymmetry between feet';
  static const String pressureSpikeAlert = 'Sudden pressure spike detected';

  // ============== Settings ==============
  static const String settings = 'Settings';
  static const String general = 'General';
  static const String notifications = 'Notifications';
  static const String appearance = 'Appearance';
  static const String darkMode = 'Dark Mode';
  static const String lightMode = 'Light Mode';
  static const String systemDefault = 'System Default';
  static const String language = 'Language';
  static const String units = 'Units';
  static const String temperatureUnit = 'Temperature Unit';
  static const String pressureUnit = 'Pressure Unit';
  static const String about = 'About';
  static const String help = 'Help & Support';
  static const String privacy = 'Privacy Policy';
  static const String terms = 'Terms of Service';
  static const String version = 'Version';

  // ============== Notifications Settings ==============
  static const String enableNotifications = 'Enable Notifications';
  static const String criticalAlerts = 'Critical Alerts';
  static const String warningAlerts = 'Warning Alerts';
  static const String dailySummaryNotif = 'Daily Summary';
  static const String soundEnabled = 'Sound';
  static const String vibrationEnabled = 'Vibration';

  // ============== Profile ==============
  static const String profile = 'Profile';
  static const String editProfile = 'Edit Profile';
  static const String name = 'Name';
  static const String age = 'Age';
  static const String diabetesType = 'Diabetes Type';
  static const String type1 = 'Type 1';
  static const String type2 = 'Type 2';
  static const String preDiabetes = 'Pre-Diabetes';
  static const String none = 'None';
  static const String saveChanges = 'Save Changes';
  static const String deleteAccount = 'Delete Account';

  // ============== Onboarding ==============
  static const String welcome = 'Welcome to NeuroSocks';
  static const String onboarding1Title = 'Smart Monitoring';
  static const String onboarding1Desc =
      'Continuous monitoring of your foot health with advanced sensors';
  static const String onboarding2Title = 'Early Detection';
  static const String onboarding2Desc =
      'AI-powered risk prediction to detect potential issues before they become serious';
  static const String onboarding3Title = 'Stay Informed';
  static const String onboarding3Desc =
      'Real-time alerts and daily reports to keep you in control of your health';
  static const String getStarted = 'Get Started';
  static const String skip = 'Skip';
  static const String next = 'Next';

  // ============== Connection ==============
  static const String searchingDevices = 'Searching for devices...';
  static const String noDevicesFound = 'No devices found';
  static const String tapToConnect = 'Tap to connect';
  static const String connectionFailed = 'Connection failed';
  static const String tryAgain = 'Try Again';
  static const String bluetoothOff = 'Bluetooth is off';
  static const String enableBluetooth = 'Please enable Bluetooth to connect';

  // ============== Errors ==============
  static const String error = 'Error';
  static const String somethingWentWrong = 'Something went wrong';
  static const String networkError = 'Network error. Please check your connection.';
  static const String invalidEmail = 'Please enter a valid email';
  static const String invalidPassword = 'Password must be at least 6 characters';
  static const String passwordMismatch = 'Passwords do not match';
  static const String requiredField = 'This field is required';
  static const String loginFailed = 'Login failed. Please check your credentials.';
  static const String registrationFailed = 'Registration failed. Please try again.';

  // ============== Success Messages ==============
  static const String success = 'Success';
  static const String profileUpdated = 'Profile updated successfully';
  static const String settingsSaved = 'Settings saved';
  static const String passwordResetSent = 'Password reset email sent';
  static const String logoutSuccess = 'Logged out successfully';

  // ============== Buttons ==============
  static const String ok = 'OK';
  static const String cancel = 'Cancel';
  static const String confirm = 'Confirm';
  static const String save = 'Save';
  static const String delete = 'Delete';
  static const String edit = 'Edit';
  static const String close = 'Close';
  static const String retry = 'Retry';
  static const String done = 'Done';
  static const String continueBtn = 'Continue';

  // ============== Misc ==============
  static const String loading = 'Loading...';
  static const String refreshing = 'Refreshing...';
  static const String noInternetConnection = 'No internet connection';
  static const String pullToRefresh = 'Pull to refresh';
  static const String lastSync = 'Last sync';
  static const String syncNow = 'Sync Now';
  static const String offlineMode = 'Offline Mode';
}
