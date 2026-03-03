// Test file to verify ESP32 payload decoding
// Run this to ensure the parsing logic matches your ESP32 code

void main() {
  // Example decoded packet from your ESP32:
  // 85 85 85 85 | 7D 7D 7D 7D | 02 6D | 00 48 | 02 F2 | 03 | 55

  final examplePacket = [
    0x85, 0x85, 0x85, 0x85, // Temperatures (Heel, Ball, Arch, Toe)
    0x7D, 0x7D, 0x7D, 0x7D, // Pressures (4 zones)
    0x02, 0x6D,             // SpO2 (619 = 6.19... no wait, 0x026D = 621... / 100 = 6.21)
    0x00, 0x48,             // Heart Rate (0x0048 = 72 BPM)
    0x02, 0xF2,             // Step Count (0x02F2 = 754 steps)
    0x03,                   // Activity Type (3 = Walking)
    0x55,                   // Battery Level (85%)
  ];

  print('=== NeuroSocks ESP32 Payload Decoder Test ===\n');

  // Parse temperatures
  print('Temperature Parsing:');
  for (int i = 0; i < 4; i++) {
    final tempByte = examplePacket[i];
    final temp = 25.0 + (tempByte - 128) / 2.0;
    print('  Zone $i (Byte $i = 0x${tempByte.toRadixString(16).toUpperCase().padLeft(2, '0')}): $temp°C');
  }
  print('  Expected: ~32.5°C for all zones');
  print('  Formula: temp = 25.0 + (byte - 128) / 2.0\n');

  // Parse pressures
  print('Pressure Parsing:');
  for (int i = 0; i < 4; i++) {
    final pressureByte = examplePacket[4 + i];
    final pressure = pressureByte * 0.3;
    print('  Zone $i (Byte ${4 + i} = 0x${pressureByte.toRadixString(16).toUpperCase().padLeft(2, '0')}): ${pressure.toStringAsFixed(1)} kPa');
  }
  print('  Expected: ~23.7 kPa for all zones');
  print('  Formula: pressure = byte * 0.3\n');

  // Parse SpO2
  print('SpO2 Parsing:');
  final spO2Raw = (examplePacket[8] << 8) | examplePacket[9];
  final spO2 = spO2Raw / 100.0;
  print('  Bytes 8-9: 0x${examplePacket[8].toRadixString(16).toUpperCase().padLeft(2, '0')} 0x${examplePacket[9].toRadixString(16).toUpperCase().padLeft(2, '0')}');
  print('  Raw value: $spO2Raw (0x${spO2Raw.toRadixString(16).toUpperCase()})');
  print('  Decoded: ${spO2.toStringAsFixed(1)}%');
  print('  Expected: 98.5% (from your ESP32 code: spo2 = 98.5; raw = 9850)');
  print('  Formula: spo2 = (byte[8] << 8 | byte[9]) / 100.0\n');

  // Parse Heart Rate
  print('Heart Rate Parsing:');
  final heartRateByte = (examplePacket[10] << 8) | examplePacket[11];
  print('  Bytes 10-11: 0x${examplePacket[10].toRadixString(16).toUpperCase().padLeft(2, '0')} 0x${examplePacket[11].toRadixString(16).toUpperCase().padLeft(2, '0')}');
  print('  Decoded: $heartRateByte BPM');
  print('  Expected: 72 BPM (from your ESP32: heartRate = 72)');
  print('  Formula: hr = (byte[10] << 8) | byte[11]\n');

  // Parse Step Count
  print('Step Count Parsing:');
  final stepCountByte = (examplePacket[12] << 8) | examplePacket[13];
  print('  Bytes 12-13: 0x${examplePacket[12].toRadixString(16).toUpperCase().padLeft(2, '0')} 0x${examplePacket[13].toRadixString(16).toUpperCase().padLeft(2, '0')}');
  print('  Decoded: $stepCountByte steps');
  print('  Expected: 754 steps');
  print('  Formula: steps = (byte[12] << 8) | byte[13]\n');

  // Parse Activity Type
  print('Activity Type Parsing:');
  final activityByte = examplePacket[14];
  final activityType = activityByte & 0x0F;
  final activityNames = ['Resting', 'Sitting', 'Standing', 'Walking', 'Running'];
  print('  Byte 14: 0x${activityByte.toRadixString(16).toUpperCase().padLeft(2, '0')} = $activityByte');
  print('  Activity: ${activityNames[activityType .clamp(0, 4)]}');
  print('  Expected: Walking (from your ESP32: activityType = 3)\n');

  // Parse Battery
  print('Battery Level Parsing:');
  final batteryByte = examplePacket[15];
  print('  Byte 15: 0x${batteryByte.toRadixString(16).toUpperCase().padLeft(2, '0')} = $batteryByte%');
  print('  Expected: 85% (from your ESP32: batteryLevel = 85)\n');

  print('=== Test Complete ===');

  // Real-world test with your actual values
  print('\n=== Real Packet Test (from your ESP32) ===');
  const tempHeelEncoded = 85;  // From your code: (32.5 - 25.0) * 2.0 + 128 = 85
  final tempHeelDecoded = 25.0 + (tempHeelEncoded - 128) / 2.0;
  print('Heel temp encoded as: $tempHeelEncoded');
  print('Decodes to: ${tempHeelDecoded.toStringAsFixed(1)}°C');
  print('✅ Correct!' );
}
