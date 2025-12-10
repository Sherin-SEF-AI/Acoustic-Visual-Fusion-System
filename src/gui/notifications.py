"""
System Tray and Notifications - Background operation and alerts.
"""

from PyQt6.QtWidgets import (
    QSystemTrayIcon, QMenu, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor
from loguru import logger


def create_tray_icon(color: str = "#58a6ff") -> QIcon:
    """Create a simple colored tray icon."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(8, 8, 48, 48)
    
    # Inner ring
    painter.setBrush(QColor("#0d1117"))
    painter.drawEllipse(16, 16, 32, 32)
    
    # Center dot
    painter.setBrush(QColor(color))
    painter.drawEllipse(24, 24, 16, 16)
    
    painter.end()
    
    return QIcon(pixmap)


class NotificationManager(QObject):
    """
    Manages system notifications and tray icon.
    """
    
    notification_clicked = pyqtSignal(str)  # notification_id
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.tray_icon: QSystemTrayIcon = None
        
        self._notification_queue = []
        self._notification_timer = QTimer()
        self._notification_timer.timeout.connect(self._show_next_notification)
        
        self._setup_tray()
        logger.info("NotificationManager initialized")
    
    def _setup_tray(self):
        """Setup system tray icon."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("System tray not available")
            return
        
        self.tray_icon = QSystemTrayIcon(self.main_window)
        self.tray_icon.setIcon(create_tray_icon())
        self.tray_icon.setToolTip("Acoustic-Visual Fusion System")
        
        # Create context menu
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background: #21262d;
                border: 1px solid #30363d;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 24px;
                color: #e6edf3;
            }
            QMenu::item:selected {
                background: #388bfd26;
            }
        """)
        
        show_action = menu.addAction("Show Window")
        show_action.triggered.connect(self._show_window)
        
        menu.addSeparator()
        
        start_action = menu.addAction("▶ Start")
        start_action.triggered.connect(self.main_window._on_start)
        
        stop_action = menu.addAction("⏹ Stop")
        stop_action.triggered.connect(self.main_window._on_stop)
        
        menu.addSeparator()
        
        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)
        
        self.tray_icon.setContextMenu(menu)
        
        # Connect signals
        self.tray_icon.activated.connect(self._on_tray_activated)
        
        self.tray_icon.show()
    
    def _show_window(self):
        """Show the main window."""
        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()
    
    def _on_tray_activated(self, reason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_window()
    
    def notify(self, title: str, message: str, 
               icon_type: str = "info",
               duration_ms: int = 5000):
        """
        Show a system notification.
        
        Args:
            title: Notification title
            message: Notification message
            icon_type: "info", "warning", or "critical"
            duration_ms: Duration to show notification
        """
        if not self.tray_icon:
            return
        
        icon_map = {
            "info": QSystemTrayIcon.MessageIcon.Information,
            "warning": QSystemTrayIcon.MessageIcon.Warning,
            "critical": QSystemTrayIcon.MessageIcon.Critical
        }
        
        icon = icon_map.get(icon_type, QSystemTrayIcon.MessageIcon.Information)
        
        self._notification_queue.append((title, message, icon, duration_ms))
        
        if not self._notification_timer.isActive():
            self._show_next_notification()
    
    def _show_next_notification(self):
        """Show next notification in queue."""
        if not self._notification_queue:
            self._notification_timer.stop()
            return
        
        title, message, icon, duration = self._notification_queue.pop(0)
        
        self.tray_icon.showMessage(title, message, icon, duration)
        
        if self._notification_queue:
            self._notification_timer.start(duration + 500)
    
    def set_status(self, status: str):
        """Update tray icon status."""
        if self.tray_icon:
            if status == "running":
                self.tray_icon.setIcon(create_tray_icon("#238636"))
            elif status == "recording":
                self.tray_icon.setIcon(create_tray_icon("#da3633"))
            elif status == "warning":
                self.tray_icon.setIcon(create_tray_icon("#d29922"))
            else:
                self.tray_icon.setIcon(create_tray_icon("#58a6ff"))
    
    def notify_detection(self, count: int, camera_id: str):
        """Notify about detections (rate limited)."""
        pass  # Don't spam notifications for detections
    
    def notify_alert(self, level: str, message: str):
        """Notify about system alert."""
        icon_type = "critical" if level == "critical" else "warning"
        self.notify("⚠️ Alert", message, icon_type)
    
    def notify_recording_started(self):
        """Notify recording started."""
        self.notify("🎬 Recording", "Session recording started", "info", 3000)
        self.set_status("recording")
    
    def notify_recording_stopped(self, path: str):
        """Notify recording stopped."""
        self.notify("🎬 Recording Saved", f"Saved to: {path}", "info", 4000)
        self.set_status("running")
