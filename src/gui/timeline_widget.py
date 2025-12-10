"""
Event Timeline Widget - Visual timeline of audio/visual events.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath,
    QLinearGradient
)
import time
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class EventType(Enum):
    SPEECH = "speech"
    PERSON_DETECTED = "person"
    SOUND_LOCALIZED = "sound"
    ACTIVITY = "activity"
    ALERT = "alert"
    FUSION_MATCH = "fusion"


@dataclass
class TimelineEvent:
    event_type: EventType
    timestamp: float
    duration: float
    label: str
    confidence: float
    track_id: Optional[int] = None
    camera_id: Optional[str] = None
    color: Optional[str] = None
    
    @property
    def end_time(self) -> float:
        return self.timestamp + self.duration


class TimelineWidget(QWidget):
    """Interactive timeline showing events over time."""
    
    event_selected = pyqtSignal(object)  # TimelineEvent
    
    # Color scheme
    COLORS = {
        EventType.SPEECH: "#ff6b9d",
        EventType.PERSON_DETECTED: "#00d4aa",
        EventType.SOUND_LOCALIZED: "#6496ff",
        EventType.ACTIVITY: "#ffd93d",
        EventType.ALERT: "#ff4444",
        EventType.FUSION_MATCH: "#a855f7",
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.events: list[TimelineEvent] = []
        self.time_window = 60.0  # seconds visible
        self.current_time = time.time()
        self.start_time = self.current_time
        
        self.selected_event: Optional[TimelineEvent] = None
        self.hover_event: Optional[TimelineEvent] = None
        
        # Track lanes
        self.lanes = {
            EventType.SPEECH: 0,
            EventType.PERSON_DETECTED: 1,
            EventType.SOUND_LOCALIZED: 2,
            EventType.ACTIVITY: 3,
            EventType.ALERT: 4,
            EventType.FUSION_MATCH: 5,
        }
        
        self.setMouseTracking(True)
        
        # Auto-scroll timer
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self._auto_scroll)
        self.scroll_timer.start(100)
    
    def add_event(self, event: TimelineEvent):
        """Add event to timeline."""
        self.events.append(event)
        # Keep only recent events (last 5 minutes)
        cutoff = self.current_time - 300
        self.events = [e for e in self.events if e.end_time > cutoff]
        self.update()
    
    def _auto_scroll(self):
        """Update current time for auto-scroll."""
        self.current_time = time.time()
        self.update()
    
    def time_to_x(self, t: float) -> float:
        """Convert timestamp to x coordinate."""
        relative = t - (self.current_time - self.time_window)
        return (relative / self.time_window) * self.width()
    
    def x_to_time(self, x: float) -> float:
        """Convert x coordinate to timestamp."""
        relative = (x / self.width()) * self.time_window
        return (self.current_time - self.time_window) + relative
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(15, 15, 35))
        
        # Grid lines
        self._draw_grid(painter)
        
        # Lane labels
        self._draw_lanes(painter)
        
        # Events
        self._draw_events(painter)
        
        # Current time marker
        self._draw_now_marker(painter)
        
        # Hover tooltip
        if self.hover_event:
            self._draw_tooltip(painter, self.hover_event)
    
    def _draw_grid(self, painter: QPainter):
        """Draw time grid."""
        painter.setPen(QPen(QColor(40, 40, 60), 1))
        
        # Vertical grid lines (every 5 seconds)
        interval = 5.0
        t = self.current_time - self.time_window
        while t < self.current_time:
            t += interval
            x = self.time_to_x(t)
            painter.drawLine(int(x), 0, int(x), self.height())
        
        # Time labels
        painter.setPen(QColor(100, 100, 120))
        font = QFont("Segoe UI", 8)
        painter.setFont(font)
        
        t = self.current_time - self.time_window
        while t < self.current_time:
            t += 10.0
            x = self.time_to_x(t)
            rel_sec = int(t - self.start_time)
            mins, secs = divmod(rel_sec, 60)
            painter.drawText(int(x) - 15, self.height() - 5, f"{mins}:{secs:02d}")
    
    def _draw_lanes(self, painter: QPainter):
        """Draw lane separators and labels."""
        lane_height = self.height() / 6
        
        painter.setPen(QPen(QColor(50, 50, 70), 1, Qt.PenStyle.DashLine))
        for i in range(1, 6):
            y = int(i * lane_height)
            painter.drawLine(0, y, self.width(), y)
    
    def _draw_events(self, painter: QPainter):
        """Draw event blocks."""
        lane_height = (self.height() - 20) / 6
        
        for ev in self.events:
            x1 = self.time_to_x(ev.timestamp)
            x2 = self.time_to_x(ev.end_time)
            
            if x2 < 0 or x1 > self.width():
                continue
            
            lane = self.lanes.get(ev.event_type, 0)
            y = lane * lane_height + 2
            
            color = QColor(ev.color or self.COLORS.get(ev.event_type, "#888888"))
            
            # Highlight selected/hover
            if ev == self.selected_event:
                color = color.lighter(150)
            elif ev == self.hover_event:
                color = color.lighter(120)
            
            # Draw event bar
            rect = QRectF(x1, y, max(x2 - x1, 4), lane_height - 4)
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(rect, 3, 3)
            
            # Confidence indicator
            if ev.confidence < 1.0:
                painter.setOpacity(0.3)
                uncert_rect = QRectF(x1, y + lane_height - 6, 
                                     (x2 - x1) * (1 - ev.confidence), 2)
                painter.fillRect(uncert_rect, QColor(0, 0, 0))
                painter.setOpacity(1.0)
            
            # Label if wide enough
            if x2 - x1 > 40:
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Segoe UI", 8))
                painter.drawText(rect.adjusted(4, 2, -2, -2), 
                               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                               ev.label[:15])
    
    def _draw_now_marker(self, painter: QPainter):
        """Draw current time indicator."""
        x = self.time_to_x(self.current_time)
        
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        painter.drawLine(int(x), 0, int(x), self.height())
        
        # Now label
        painter.setBrush(QBrush(QColor(255, 100, 100)))
        painter.drawEllipse(int(x) - 4, 2, 8, 8)
    
    def _draw_tooltip(self, painter: QPainter, event: TimelineEvent):
        """Draw tooltip for hovered event."""
        text = f"{event.label}\n{event.event_type.value}\nConf: {event.confidence:.0%}"
        
        painter.setPen(QColor(255, 255, 255))
        painter.setBrush(QBrush(QColor(30, 30, 50, 230)))
        
        metrics = painter.fontMetrics()
        rect = metrics.boundingRect(0, 0, 200, 100, 
                                    Qt.TextFlag.TextWordWrap, text)
        
        x = self.time_to_x(event.timestamp)
        y = 30
        
        painter.drawRoundedRect(int(x), y, rect.width() + 16, rect.height() + 8, 4, 4)
        painter.drawText(int(x) + 8, y + 4, rect.width(), rect.height(),
                        Qt.TextFlag.TextWordWrap, text)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for hover effects."""
        pos = event.position()
        t = self.x_to_time(pos.x())
        lane_height = (self.height() - 20) / 6
        lane = int(pos.y() / lane_height)
        
        self.hover_event = None
        for ev in reversed(self.events):
            ev_lane = self.lanes.get(ev.event_type, 0)
            if ev_lane == lane and ev.timestamp <= t <= ev.end_time:
                self.hover_event = ev
                break
        
        self.update()
    
    def mousePressEvent(self, event):
        """Handle click to select event."""
        if self.hover_event:
            self.selected_event = self.hover_event
            self.event_selected.emit(self.hover_event)
            self.update()


class EventTimelinePanel(QFrame):
    """Panel containing timeline and controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("timeline-panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("📈 Event Timeline")
        title.setStyleSheet("font-size: 13px; font-weight: 500; color: #88aaff;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Legend
        for event_type, color in TimelineWidget.COLORS.items():
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 10px;")
            header.addWidget(dot)
            
            label = QLabel(event_type.value[:6])
            label.setStyleSheet("color: #888; font-size: 10px; margin-right: 8px;")
            header.addWidget(label)
        
        layout.addLayout(header)
        
        # Timeline
        self.timeline = TimelineWidget()
        layout.addWidget(self.timeline)
    
    def add_event(self, event_type: EventType, label: str, 
                  duration: float = 1.0, confidence: float = 1.0,
                  track_id: int = None, camera_id: str = None):
        """Add an event to the timeline."""
        event = TimelineEvent(
            event_type=event_type,
            timestamp=time.time(),
            duration=duration,
            label=label,
            confidence=confidence,
            track_id=track_id,
            camera_id=camera_id
        )
        self.timeline.add_event(event)
