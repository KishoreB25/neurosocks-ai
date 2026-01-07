import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';
import '../../data/models/alert.dart';

/// Tile widget for displaying a single alert
class AlertTile extends StatelessWidget {
  /// The alert to display
  final Alert alert;

  /// Callback when the tile is tapped
  final VoidCallback? onTap;

  /// Callback when dismissed
  final VoidCallback? onDismiss;

  /// Callback when marked as read
  final VoidCallback? onMarkRead;

  /// Whether to show the time
  final bool showTime;

  /// Whether to enable swipe to dismiss
  final bool dismissible;

  /// Whether to show action buttons
  final bool showActions;

  const AlertTile({
    super.key,
    required this.alert,
    this.onTap,
    this.onDismiss,
    this.onMarkRead,
    this.showTime = true,
    this.dismissible = true,
    this.showActions = false,
  });

  @override
  Widget build(BuildContext context) {
    final tile = _buildTile(context);

    if (dismissible && onDismiss != null) {
      return Dismissible(
        key: Key(alert.id),
        direction: DismissDirection.endToStart,
        onDismissed: (_) => onDismiss!(),
        background: _buildDismissBackground(),
        child: tile,
      );
    }

    return tile;
  }

  Widget _buildTile(BuildContext context) {
    final theme = Theme.of(context);
    final severityColor = _getSeverityColor();
    final typeIcon = _getTypeIcon();

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: alert.isRead
                ? theme.cardColor
                : severityColor.withOpacity(0.05),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: alert.isRead
                  ? Colors.grey.withOpacity(0.1)
                  : severityColor.withOpacity(0.3),
              width: 1.5,
            ),
            boxShadow: alert.isRead
                ? null
                : [
                    BoxShadow(
                      color: severityColor.withOpacity(0.1),
                      blurRadius: 8,
                      offset: const Offset(0, 2),
                    ),
                  ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header row
              Row(
                children: [
                  // Severity icon
                  Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: severityColor.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Icon(
                      typeIcon,
                      size: 20,
                      color: severityColor,
                    ),
                  ),
                  const SizedBox(width: 12),

                  // Title and type
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          alert.title,
                          style: TextStyle(
                            fontSize: 15,
                            fontWeight:
                                alert.isRead ? FontWeight.w500 : FontWeight.bold,
                            color: theme.textTheme.bodyLarge?.color,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Row(
                          children: [
                            _buildSeverityBadge(severityColor),
                            const SizedBox(width: 8),
                            Text(
                              alert.type.displayName,
                              style: TextStyle(
                                fontSize: 11,
                                color: Colors.grey[600],
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),

                  // Unread indicator
                  if (!alert.isRead)
                    Container(
                      width: 10,
                      height: 10,
                      decoration: BoxDecoration(
                        color: severityColor,
                        shape: BoxShape.circle,
                      ),
                    ),
                ],
              ),

              const SizedBox(height: 12),

              // Message
              Text(
                alert.message,
                style: TextStyle(
                  fontSize: 13,
                  color: Colors.grey[700],
                  height: 1.4,
                ),
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),

              // Value info (if available)
              if (alert.actualValue != null && alert.threshold != null) ...[
                const SizedBox(height: 8),
                _buildValueInfo(severityColor),
              ],

              // Action recommendation (if available)
              if (alert.action != null && alert.action!.isNotEmpty) ...[
                const SizedBox(height: 10),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.blue.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.lightbulb_outline,
                        size: 14,
                        color: Colors.blue[700],
                      ),
                      const SizedBox(width: 6),
                      Flexible(
                        child: Text(
                          alert.action!,
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.blue[700],
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],

              const SizedBox(height: 10),

              // Footer: Time and actions
              Row(
                children: [
                  // Zone (if available)
                  if (alert.affectedZone != null) ...[
                    Icon(
                      Icons.location_on,
                      size: 14,
                      color: Colors.grey[500],
                    ),
                    const SizedBox(width: 4),
                    Text(
                      alert.affectedZone!,
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.grey[600],
                      ),
                    ),
                    const SizedBox(width: 12),
                  ],

                  // Time
                  if (showTime) ...[
                    Icon(
                      Icons.access_time,
                      size: 14,
                      color: Colors.grey[500],
                    ),
                    const SizedBox(width: 4),
                    Text(
                      _formatTime(alert.timestamp),
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],

                  const Spacer(),

                  // Actions
                  if (showActions) ...[
                    if (!alert.isRead && onMarkRead != null)
                      TextButton.icon(
                        onPressed: onMarkRead,
                        icon: const Icon(Icons.check, size: 16),
                        label: const Text('Mark Read'),
                        style: TextButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 8),
                          minimumSize: Size.zero,
                          tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                        ),
                      ),
                    if (onDismiss != null)
                      IconButton(
                        onPressed: onDismiss,
                        icon: const Icon(Icons.close, size: 18),
                        padding: EdgeInsets.zero,
                        constraints: const BoxConstraints(),
                        color: Colors.grey[500],
                      ),
                  ],
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSeverityBadge(Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        alert.severity.displayName,
        style: TextStyle(
          fontSize: 10,
          fontWeight: FontWeight.bold,
          color: color,
        ),
      ),
    );
  }

  Widget _buildValueInfo(Color color) {
    return Row(
      children: [
        Icon(
          Icons.analytics,
          size: 14,
          color: Colors.grey[600],
        ),
        const SizedBox(width: 6),
        Text(
          'Measured: ',
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey[600],
          ),
        ),
        Text(
          alert.actualValue!.toStringAsFixed(1),
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(
          ' (threshold: ${alert.threshold!.toStringAsFixed(1)})',
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey[600],
          ),
        ),
      ],
    );
  }

  Widget _buildDismissBackground() {
    return Container(
      alignment: Alignment.centerRight,
      padding: const EdgeInsets.only(right: 24),
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Icon(
        Icons.delete_outline,
        color: Colors.red,
      ),
    );
  }

  Color _getSeverityColor() {
    switch (alert.severity) {
      case AlertSeverity.info:
        return AppColors.riskLow;
      case AlertSeverity.warning:
        return AppColors.riskModerate;
      case AlertSeverity.critical:
        return AppColors.riskCritical;
    }
  }

  IconData _getTypeIcon() {
    switch (alert.type) {
      case AlertType.temperature:
        return Icons.thermostat;
      case AlertType.pressure:
        return Icons.speed;
      case AlertType.circulation:
        return Icons.bloodtype;
      case AlertType.gait:
        return Icons.directions_walk;
      case AlertType.system:
        return Icons.settings;
    }
  }

  String _formatTime(DateTime time) {
    final now = DateTime.now();
    final diff = now.difference(time);

    if (diff.inMinutes < 1) {
      return 'Just now';
    } else if (diff.inMinutes < 60) {
      return '${diff.inMinutes}m ago';
    } else if (diff.inHours < 24) {
      return '${diff.inHours}h ago';
    } else if (diff.inDays == 1) {
      return 'Yesterday';
    } else if (diff.inDays < 7) {
      return '${diff.inDays}d ago';
    } else {
      return '${time.day}/${time.month}/${time.year}';
    }
  }
}

/// Compact version of alert tile for lists
class CompactAlertTile extends StatelessWidget {
  final Alert alert;
  final VoidCallback? onTap;

  const CompactAlertTile({
    super.key,
    required this.alert,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color = _getSeverityColor();

    return ListTile(
      onTap: onTap,
      leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Icon(
          _getTypeIcon(),
          size: 20,
          color: color,
        ),
      ),
      title: Text(
        alert.title,
        style: TextStyle(
          fontWeight: alert.isRead ? FontWeight.normal : FontWeight.bold,
          fontSize: 14,
        ),
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      ),
      subtitle: Text(
        alert.message,
        style: TextStyle(
          fontSize: 12,
          color: Colors.grey[600],
        ),
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      ),
      trailing: !alert.isRead
          ? Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
              ),
            )
          : null,
    );
  }

  Color _getSeverityColor() {
    switch (alert.severity) {
      case AlertSeverity.info:
        return AppColors.riskLow;
      case AlertSeverity.warning:
        return AppColors.riskModerate;
      case AlertSeverity.critical:
        return AppColors.riskCritical;
    }
  }

  IconData _getTypeIcon() {
    switch (alert.type) {
      case AlertType.temperature:
        return Icons.thermostat;
      case AlertType.pressure:
        return Icons.speed;
      case AlertType.circulation:
        return Icons.bloodtype;
      case AlertType.gait:
        return Icons.directions_walk;
      case AlertType.system:
        return Icons.settings;
    }
  }
}
