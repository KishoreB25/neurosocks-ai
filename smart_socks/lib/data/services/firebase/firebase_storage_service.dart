import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/foundation.dart';
import 'dart:io';

/// Firebase Cloud Storage Service
/// Handles file uploads (photos, reports, documents)
class FirebaseStorageService {
  static final FirebaseStorageService _instance =
      FirebaseStorageService._internal();
  factory FirebaseStorageService() => _instance;
  FirebaseStorageService._internal();

  final FirebaseStorage _storage = FirebaseStorage.instance;

  // Storage paths
  static const String userProfilePhotos = 'user_profile_photos';
  static const String userReports = 'user_reports';
  static const String footAnalysisImages = 'foot_analysis_images';

  // ============== Upload ==============

  /// Upload file to Firebase Storage
  Future<String?> uploadFile({
    required String userId,
    required String path,
    required File file,
    String? fileName,
  }) async {
    try {
      final name = fileName ?? file.path.split('/').last;
      final ref = _storage.ref('$path/$userId/$name');

      await ref.putFile(file);
      return await ref.getDownloadURL();
    } on FirebaseException catch (e) {
      debugPrint('Upload File Error: ${e.code} - ${e.message}');
      rethrow;
    } catch (e) {
      debugPrint('Upload File Error: $e');
      rethrow;
    }
  }

  /// Upload bytes to Firebase Storage
  Future<String?> uploadBytes({
    required String userId,
    required String path,
    required List<int> bytes,
    required String fileName,
    String? contentType,
  }) async {
    try {
      final ref = _storage.ref('$path/$userId/$fileName');

      await ref.putData(
        Uint8List.fromList(bytes),
        SettableMetadata(contentType: contentType ?? 'application/octet-stream'),
      );

      return await ref.getDownloadURL();
    } on FirebaseException catch (e) {
      debugPrint('Upload Bytes Error: ${e.code} - ${e.message}');
      rethrow;
    } catch (e) {
      debugPrint('Upload Bytes Error: $e');
      rethrow;
    }
  }

  // ============== Profile Photo ==============

  /// Upload user profile photo
  Future<String?> uploadProfilePhoto({
    required String userId,
    required File photoFile,
  }) async {
    return await uploadFile(
      userId: userId,
      path: userProfilePhotos,
      file: photoFile,
      fileName: 'profile_photo.jpg',
    );
  }

  /// Delete user profile photo
  Future<void> deleteProfilePhoto(String userId) async {
    try {
      await _storage.ref('$userProfilePhotos/$userId/profile_photo.jpg').delete();
    } catch (e) {
      debugPrint('Delete Profile Photo Error: $e');
    }
  }

  // ============== Reports ==============

  /// Upload health report (PDF or document)
  Future<String?> uploadHealthReport({
    required String userId,
    required File reportFile,
    required String reportName,
  }) async {
    return await uploadFile(
      userId: userId,
      path: userReports,
      file: reportFile,
      fileName: reportName,
    );
  }

  /// List all reports for user
  Future<List<String>> getUserReports(String userId) async {
    try {
      final result = await _storage.ref('$userReports/$userId').listAll();
      final urls = <String>[];

      for (final item in result.items) {
        final url = await item.getDownloadURL();
        urls.add(url);
      }

      return urls;
    } catch (e) {
      debugPrint('Get User Reports Error: $e');
      return [];
    }
  }

  /// Delete report
  Future<void> deleteReport({
    required String userId,
    required String reportName,
  }) async {
    try {
      await _storage.ref('$userReports/$userId/$reportName').delete();
    } catch (e) {
      debugPrint('Delete Report Error: $e');
    }
  }

  // ============== Foot Analysis Images ==============

  /// Upload foot analysis image
  Future<String?> uploadFootAnalysisImage({
    required String userId,
    required File imageFile,
    required String imageName,
  }) async {
    return await uploadFile(
      userId: userId,
      path: footAnalysisImages,
      file: imageFile,
      fileName: imageName,
    );
  }

  /// Get all foot analysis images
  Future<List<String>> getFootAnalysisImages(String userId) async {
    try {
      final result = await _storage.ref('$footAnalysisImages/$userId').listAll();
      final urls = <String>[];

      for (final item in result.items) {
        final url = await item.getDownloadURL();
        urls.add(url);
      }

      return urls;
    } catch (e) {
      debugPrint('Get Foot Analysis Images Error: $e');
      return [];
    }
  }

  // ============== Download ==============

  /// Download file from storage
  Future<List<int>?> downloadFile({
    required String userId,
    required String path,
    required String fileName,
  }) async {
    try {
      return await _storage
          .ref('$path/$userId/$fileName')
          .getData();
    } catch (e) {
      debugPrint('Download File Error: $e');
      return null;
    }
  }

  /// Get download URL
  Future<String?> getDownloadUrl({
    required String userId,
    required String path,
    required String fileName,
  }) async {
    try {
      return await _storage
          .ref('$path/$userId/$fileName')
          .getDownloadURL();
    } catch (e) {
      debugPrint('Get Download URL Error: $e');
      return null;
    }
  }

  // ============== Delete ==============

  /// Delete file
  Future<void> deleteFile({
    required String userId,
    required String path,
    required String fileName,
  }) async {
    try {
      await _storage.ref('$path/$userId/$fileName').delete();
    } catch (e) {
      debugPrint('Delete File Error: $e');
    }
  }

  // ============== File Metadata ==============

  /// Get file metadata
  Future<FullMetadata?> getFileMetadata({
    required String userId,
    required String path,
    required String fileName,
  }) async {
    try {
      return await _storage.ref('$path/$userId/$fileName').getMetadata();
    } catch (e) {
      debugPrint('Get File Metadata Error: $e');
      return null;
    }
  }

  /// Update file metadata
  Future<FullMetadata?> updateFileMetadata({
    required String userId,
    required String path,
    required String fileName,
    String? contentType,
    Map<String, String>? customMetadata,
  }) async {
    try {
      return await _storage
          .ref('$path/$userId/$fileName')
          .updateMetadata(
            SettableMetadata(
              contentType: contentType,
              customMetadata: customMetadata,
            ),
          );
    } catch (e) {
      debugPrint('Update File Metadata Error: $e');
      return null;
    }
  }
}
