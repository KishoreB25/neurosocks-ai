// Smart Socks App Widget Tests
//
// Basic widget tests for the Smart Socks diabetic foot monitoring application.
// These tests verify the app launches correctly.

import 'package:flutter_test/flutter_test.dart';
import 'package:smart_socks/app.dart';

void main() {
  testWidgets('App launches smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const SmartSocksApp());

    // Wait for providers to initialize
    await tester.pumpAndSettle();

    // Verify the app renders (welcome screen should show)
    expect(find.textContaining('Smart'), findsWidgets);
  });
}
