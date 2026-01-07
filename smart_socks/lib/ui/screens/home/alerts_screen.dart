import 'package:flutter/material.dart' hide ConnectionState;
import 'package:provider/provider.dart';
import '../../../core/constants/app_colors.dart';
import '../../../core/constants/app_strings.dart';
import '../../../data/models/alert.dart';
import '../../../providers/risk_provider.dart';
import '../../widgets/alert_tile.dart';

/// Screen displaying all alerts with filtering options
class AlertsScreen extends StatefulWidget {
  const AlertsScreen({super.key});

  @override
  State<AlertsScreen> createState() => _AlertsScreenState();
}

class _AlertsScreenState extends State<AlertsScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  AlertType? _selectedType;
  AlertSeverity? _selectedSeverity;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Alerts'),
        actions: [
          // Filter button
          IconButton(
            icon: const Icon(Icons.filter_list),
            onPressed: _showFilterDialog,
          ),
          // Mark all as read
          Consumer<RiskProvider>(
            builder: (context, provider, _) {
              if (provider.unreadAlertCount > 0) {
                return IconButton(
                  icon: const Icon(Icons.done_all),
                  onPressed: () {
                    provider.markAllAlertsAsRead();
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('All alerts marked as read'),
                        duration: Duration(seconds: 2),
                      ),
                    );
                  },
                  tooltip: 'Mark all as read',
                );
              }
              return const SizedBox.shrink();
            },
          ),
        ],
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: 'All'),
            Tab(text: 'Unread'),
            Tab(text: 'Critical'),
          ],
        ),
      ),
      body: Consumer<RiskProvider>(
        builder: (context, provider, _) {
          return TabBarView(
            controller: _tabController,
            children: [
              _buildAlertsList(provider, _filterAlerts(provider.alerts)),
              _buildAlertsList(
                provider,
                _filterAlerts(provider.alerts.where((a) => !a.isRead).toList()),
              ),
              _buildAlertsList(
                provider,
                _filterAlerts(provider.alerts
                    .where((a) => a.severity == AlertSeverity.critical)
                    .toList()),
              ),
            ],
          );
        },
      ),
    );
  }

  List<Alert> _filterAlerts(List<Alert> alerts) {
    var filtered = alerts;

    if (_selectedType != null) {
      filtered = filtered.where((a) => a.type == _selectedType).toList();
    }

    if (_selectedSeverity != null) {
      filtered = filtered.where((a) => a.severity == _selectedSeverity).toList();
    }

    return filtered;
  }

  Widget _buildAlertsList(RiskProvider provider, List<Alert> alerts) {
    if (alerts.isEmpty) {
      return _buildEmptyState();
    }

    // Group alerts by date
    final groupedAlerts = _groupAlertsByDate(alerts);

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: groupedAlerts.length,
      itemBuilder: (context, index) {
        final group = groupedAlerts[index];
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Date header
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: Text(
                group.dateLabel,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey[600],
                ),
              ),
            ),
            // Alerts for this date
            ...group.alerts.map((alert) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: AlertTile(
                    alert: alert,
                    dismissible: true,
                    onTap: () => _showAlertDetails(alert, provider),
                    onDismiss: () => provider.removeAlert(alert.id),
                  ),
                )),
          ],
        );
      },
    );
  }

  List<_AlertGroup> _groupAlertsByDate(List<Alert> alerts) {
    final Map<String, List<Alert>> groups = {};
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final yesterday = today.subtract(const Duration(days: 1));

    for (final alert in alerts) {
      final alertDate = DateTime(
        alert.timestamp.year,
        alert.timestamp.month,
        alert.timestamp.day,
      );

      String label;
      if (alertDate == today) {
        label = 'Today';
      } else if (alertDate == yesterday) {
        label = 'Yesterday';
      } else if (now.difference(alertDate).inDays < 7) {
        label = _getWeekdayName(alertDate.weekday);
      } else {
        label = '${alertDate.month}/${alertDate.day}/${alertDate.year}';
      }

      groups.putIfAbsent(label, () => []);
      groups[label]!.add(alert);
    }

    return groups.entries
        .map((e) => _AlertGroup(dateLabel: e.key, alerts: e.value))
        .toList();
  }

  String _getWeekdayName(int weekday) {
    const names = [
      'Monday',
      'Tuesday',
      'Wednesday',
      'Thursday',
      'Friday',
      'Saturday',
      'Sunday'
    ];
    return names[weekday - 1];
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.notifications_off_outlined,
            size: 80,
            color: Colors.grey[300],
          ),
          const SizedBox(height: 16),
          Text(
            AppStrings.noAlerts,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Colors.grey[500],
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'You\'re all caught up!',
            style: TextStyle(
              color: Colors.grey[400],
            ),
          ),
          if (_selectedType != null || _selectedSeverity != null) ...[
            const SizedBox(height: 16),
            TextButton.icon(
              onPressed: () {
                setState(() {
                  _selectedType = null;
                  _selectedSeverity = null;
                });
              },
              icon: const Icon(Icons.filter_list_off),
              label: const Text('Clear filters'),
            ),
          ],
        ],
      ),
    );
  }

  void _showFilterDialog() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => StatefulBuilder(
        builder: (context, setModalState) => Container(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Filter Alerts',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  TextButton(
                    onPressed: () {
                      setModalState(() {
                        _selectedType = null;
                        _selectedSeverity = null;
                      });
                      setState(() {});
                    },
                    child: const Text('Reset'),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Alert Type Filter
              const Text(
                'Alert Type',
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: AlertType.values.map((type) {
                  final isSelected = _selectedType == type;
                  return FilterChip(
                    label: Text(_getTypeLabel(type)),
                    selected: isSelected,
                    onSelected: (selected) {
                      setModalState(() {
                        _selectedType = selected ? type : null;
                      });
                      setState(() {});
                    },
                    selectedColor: _getTypeColor(type).withValues(alpha: 0.2),
                    checkmarkColor: _getTypeColor(type),
                  );
                }).toList(),
              ),
              const SizedBox(height: 16),

              // Severity Filter
              const Text(
                'Severity',
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: AlertSeverity.values.map((severity) {
                  final isSelected = _selectedSeverity == severity;
                  return FilterChip(
                    label: Text(_getSeverityLabel(severity)),
                    selected: isSelected,
                    onSelected: (selected) {
                      setModalState(() {
                        _selectedSeverity = selected ? severity : null;
                      });
                      setState(() {});
                    },
                    selectedColor: _getSeverityColor(severity).withValues(alpha: 0.2),
                    checkmarkColor: _getSeverityColor(severity),
                  );
                }).toList(),
              ),
              const SizedBox(height: 24),

              // Apply button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Apply'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showAlertDetails(Alert alert, RiskProvider provider) {
    // Mark as read when viewing
    if (!alert.isRead) {
      provider.markAlertAsRead(alert.id);
    }

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.4,
        maxChildSize: 0.9,
        expand: false,
        builder: (context, scrollController) => Container(
          padding: const EdgeInsets.all(24),
          child: ListView(
            controller: scrollController,
            children: [
              // Header
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: _getSeverityColor(alert.severity).withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Icon(
                      _getTypeIcon(alert.type),
                      color: _getSeverityColor(alert.severity),
                      size: 28,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          alert.title,
                          style: const TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            _buildSeverityBadge(alert.severity),
                            const SizedBox(width: 8),
                            Text(
                              _formatTimestamp(alert.timestamp),
                              style: TextStyle(
                                color: Colors.grey[500],
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const Divider(height: 32),

              // Message
              Text(
                alert.message,
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey[700],
                  height: 1.5,
                ),
              ),
              const SizedBox(height: 24),

              // Details
              if (alert.actualValue != null || alert.affectedZone != null)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Details',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 14,
                        ),
                      ),
                      const SizedBox(height: 12),
                      if (alert.affectedZone != null)
                        _buildDetailRow('Zone', alert.affectedZone!),
                      if (alert.actualValue != null)
                        _buildDetailRow(
                          'Reading',
                          alert.actualValue!.toStringAsFixed(1),
                        ),
                      if (alert.threshold != null)
                        _buildDetailRow(
                          'Threshold',
                          alert.threshold!.toStringAsFixed(1),
                        ),
                    ],
                  ),
                ),
              const SizedBox(height: 24),

              // Recommended Action
              if (alert.action != null)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: AppColors.info.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppColors.info.withValues(alpha: 0.3)),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.lightbulb_outline, color: AppColors.info),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Recommended Action',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: AppColors.info,
                                fontSize: 12,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              alert.action!,
                              style: TextStyle(
                                color: Colors.grey[700],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              const SizedBox(height: 24),

              // Actions
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () {
                        provider.removeAlert(alert.id);
                        Navigator.pop(context);
                      },
                      icon: const Icon(Icons.delete_outline),
                      label: const Text('Dismiss'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: () {
                        Navigator.pop(context);
                        Navigator.of(context).pushNamed('/sensors');
                      },
                      icon: const Icon(Icons.sensors),
                      label: const Text('View Sensors'),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(color: Colors.grey[600]),
          ),
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }

  Widget _buildSeverityBadge(AlertSeverity severity) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: _getSeverityColor(severity).withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        _getSeverityLabel(severity),
        style: TextStyle(
          color: _getSeverityColor(severity),
          fontSize: 10,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  String _formatTimestamp(DateTime timestamp) {
    final now = DateTime.now();
    final diff = now.difference(timestamp);

    if (diff.inMinutes < 60) {
      return '${diff.inMinutes}m ago';
    } else if (diff.inHours < 24) {
      return '${diff.inHours}h ago';
    } else {
      return '${timestamp.month}/${timestamp.day} ${timestamp.hour}:${timestamp.minute.toString().padLeft(2, '0')}';
    }
  }

  String _getTypeLabel(AlertType type) {
    switch (type) {
      case AlertType.temperature:
        return AppStrings.temperature;
      case AlertType.pressure:
        return AppStrings.pressure;
      case AlertType.circulation:
        return 'Circulation';
      case AlertType.gait:
        return 'Gait';
      case AlertType.system:
        return 'System';
    }
  }

  Color _getTypeColor(AlertType type) {
    switch (type) {
      case AlertType.temperature:
        return AppColors.temperature;
      case AlertType.pressure:
        return AppColors.pressure;
      case AlertType.circulation:
        return AppColors.oxygen;
      case AlertType.gait:
        return AppColors.gait;
      case AlertType.system:
        return Colors.grey;
    }
  }

  IconData _getTypeIcon(AlertType type) {
    switch (type) {
      case AlertType.temperature:
        return Icons.thermostat;
      case AlertType.pressure:
        return Icons.compress;
      case AlertType.circulation:
        return Icons.bloodtype;
      case AlertType.gait:
        return Icons.directions_walk;
      case AlertType.system:
        return Icons.info_outline;
    }
  }

  String _getSeverityLabel(AlertSeverity severity) {
    switch (severity) {
      case AlertSeverity.info:
        return 'Info';
      case AlertSeverity.warning:
        return 'Warning';
      case AlertSeverity.critical:
        return 'Critical';
    }
  }

  Color _getSeverityColor(AlertSeverity severity) {
    switch (severity) {
      case AlertSeverity.info:
        return AppColors.info;
      case AlertSeverity.warning:
        return AppColors.warning;
      case AlertSeverity.critical:
        return AppColors.error;
    }
  }
}

/// Helper class for grouping alerts by date
class _AlertGroup {
  final String dateLabel;
  final List<Alert> alerts;

  _AlertGroup({required this.dateLabel, required this.alerts});
}
