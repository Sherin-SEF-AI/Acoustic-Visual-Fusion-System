"""
Alert Panel - Display and manage system alerts.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QListWidget, QListWidgetItem, QPushButton, QComboBox,
    QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from dataclasses import dataclass
from enum import Enum
import time
from typing import Optional


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: AlertSeverity
    source: str  # e.g., "zone", "audio", "video"
    title: str
    message: str
    timestamp: float
    acknowledged: bool = False
    metadata: dict = None


class AlertCard(QFrame):
    """Individual alert display card."""
    
    acknowledged = pyqtSignal(str)  # alert_id
    
    COLORS = {
        AlertSeverity.INFO: "#6496ff",
        AlertSeverity.WARNING: "#ffd93d",
        AlertSeverity.CRITICAL: "#ff4444"
    }
    
    ICONS = {
        AlertSeverity.INFO: "ℹ️",
        AlertSeverity.WARNING: "⚠️",
        AlertSeverity.CRITICAL: "🚨"
    }
    
    def __init__(self, alert: Alert, parent=None):
        super().__init__(parent)
        self.alert = alert
        
        color = self.COLORS[alert.severity]
        self.setStyleSheet(f"""
            QFrame {{
                background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15);
                border: 1px solid {color}40;
                border-left: 3px solid {color};
                border-radius: 6px;
                margin: 2px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        
        icon = QLabel(self.ICONS[alert.severity])
        icon.setStyleSheet("font-size: 16px;")
        header.addWidget(icon)
        
        title = QLabel(alert.title)
        title.setStyleSheet(f"color: {color}; font-weight: 600;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Time
        time_str = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #666; font-size: 10px;")
        header.addWidget(time_label)
        
        layout.addLayout(header)
        
        # Message
        msg = QLabel(alert.message)
        msg.setWordWrap(True)
        msg.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(msg)
        
        # Source tag
        source = QLabel(f"[{alert.source}]")
        source.setStyleSheet("color: #555; font-size: 10px;")
        layout.addWidget(source)
        
        # Acknowledge button for warnings/critical
        if alert.severity != AlertSeverity.INFO and not alert.acknowledged:
            ack_btn = QPushButton("Acknowledge")
            ack_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 255, 255, 0.1);
                    border: none;
                    padding: 4px 12px;
                    border-radius: 4px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.2);
                }
            """)
            ack_btn.clicked.connect(lambda: self.acknowledged.emit(alert.alert_id))
            layout.addWidget(ack_btn, alignment=Qt.AlignmentFlag.AlignRight)


class AlertPanel(QFrame):
    """Panel displaying system alerts."""
    
    alert_acknowledged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("alert-panel")
        
        self.alerts: list[Alert] = []
        self.alert_widgets: dict[str, AlertCard] = {}
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("🔔 Alerts")
        title.setStyleSheet("font-size: 13px; font-weight: 500; color: #ff6b9d;")
        header.addWidget(title)
        
        self.count_label = QLabel("0")
        self.count_label.setStyleSheet("""
            background: rgba(255, 100, 100, 0.3);
            color: #ff6b6b;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
        """)
        header.addWidget(self.count_label)
        
        header.addStretch()
        
        # Filter
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Critical", "Warning", "Info"])
        self.filter_combo.setStyleSheet("background: rgba(0,0,0,0.3); padding: 4px;")
        self.filter_combo.currentTextChanged.connect(self._apply_filter)
        header.addWidget(self.filter_combo)
        
        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 100, 100, 0.2);
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
            }
        """)
        clear_btn.clicked.connect(self._clear_all)
        header.addWidget(clear_btn)
        
        layout.addLayout(header)
        
        # Scroll area for alerts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        self.alerts_container = QWidget()
        self.alerts_layout = QVBoxLayout(self.alerts_container)
        self.alerts_layout.setContentsMargins(0, 0, 0, 0)
        self.alerts_layout.setSpacing(4)
        self.alerts_layout.addStretch()
        
        scroll.setWidget(self.alerts_container)
        layout.addWidget(scroll, 1)
        
        # Summary
        self.summary = QLabel("No alerts")
        self.summary.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(self.summary)
    
    def add_alert(self, severity: AlertSeverity, source: str,
                  title: str, message: str, metadata: dict = None):
        """Add a new alert."""
        alert = Alert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            severity=severity,
            source=source,
            title=title,
            message=message,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.alerts.insert(0, alert)
        
        # Limit alerts
        if len(self.alerts) > 100:
            old_alert = self.alerts.pop()
            if old_alert.alert_id in self.alert_widgets:
                widget = self.alert_widgets.pop(old_alert.alert_id)
                widget.deleteLater()
        
        # Create widget
        card = AlertCard(alert)
        card.acknowledged.connect(self._on_acknowledged)
        
        self.alerts_layout.insertWidget(0, card)
        self.alert_widgets[alert.alert_id] = card
        
        self._update_counts()
    
    def _on_acknowledged(self, alert_id: str):
        """Handle alert acknowledgment."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                break
        
        if alert_id in self.alert_widgets:
            widget = self.alert_widgets[alert_id]
            widget.setStyleSheet(widget.styleSheet().replace("0.15", "0.05"))
        
        self.alert_acknowledged.emit(alert_id)
        self._update_counts()
    
    def _apply_filter(self, filter_text: str):
        """Filter alerts by severity."""
        for alert in self.alerts:
            if alert.alert_id in self.alert_widgets:
                widget = self.alert_widgets[alert.alert_id]
                
                if filter_text == "All":
                    widget.setVisible(True)
                else:
                    widget.setVisible(
                        alert.severity.value.lower() == filter_text.lower()
                    )
    
    def _clear_all(self):
        """Clear all alerts."""
        for widget in self.alert_widgets.values():
            widget.deleteLater()
        
        self.alerts.clear()
        self.alert_widgets.clear()
        self._update_counts()
    
    def _update_counts(self):
        """Update alert counts."""
        total = len(self.alerts)
        unack = sum(1 for a in self.alerts 
                    if not a.acknowledged and a.severity != AlertSeverity.INFO)
        
        self.count_label.setText(str(unack) if unack > 0 else str(total))
        
        if unack > 0:
            self.count_label.setStyleSheet("""
                background: rgba(255, 50, 50, 0.4);
                color: #ff4444;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
            """)
        else:
            self.count_label.setStyleSheet("""
                background: rgba(100, 150, 255, 0.2);
                color: #6496ff;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 11px;
            """)
        
        critical = sum(1 for a in self.alerts if a.severity == AlertSeverity.CRITICAL)
        warning = sum(1 for a in self.alerts if a.severity == AlertSeverity.WARNING)
        info = sum(1 for a in self.alerts if a.severity == AlertSeverity.INFO)
        
        self.summary.setText(
            f"🚨 {critical}  ⚠️ {warning}  ℹ️ {info}  |  Total: {total}"
        )
