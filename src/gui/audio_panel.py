"""
Audio Panel Widget for audio visualization and controls.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QProgressBar, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QLinearGradient
import numpy as np


class AudioLevelMeter(QWidget):
    """Real-time audio level meter."""
    
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.level = 0.0
        self.peak = 0.0
        self.peak_hold = 0
        
        if orientation == Qt.Orientation.Horizontal:
            self.setMinimumSize(100, 20)
        else:
            self.setMinimumSize(20, 100)
    
    def set_level(self, level: float):
        """Set current level (0.0 to 1.0)."""
        self.level = max(0, min(1, level))
        
        if self.level > self.peak:
            self.peak = self.level
            self.peak_hold = 30
        elif self.peak_hold > 0:
            self.peak_hold -= 1
        else:
            self.peak = max(0, self.peak - 0.02)
        
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect().adjusted(1, 1, -1, -1)
        
        # Background
        painter.fillRect(rect, QColor(20, 20, 40))
        
        # Level gradient
        if self.orientation == Qt.Orientation.Horizontal:
            level_width = int(rect.width() * self.level)
            level_rect = rect.adjusted(0, 0, -(rect.width() - level_width), 0)
        else:
            level_height = int(rect.height() * self.level)
            level_rect = rect.adjusted(0, rect.height() - level_height, 0, 0)
        
        gradient = QLinearGradient()
        if self.orientation == Qt.Orientation.Horizontal:
            gradient.setStart(rect.left(), 0)
            gradient.setFinalStop(rect.right(), 0)
        else:
            gradient.setStart(0, rect.bottom())
            gradient.setFinalStop(0, rect.top())
        
        gradient.setColorAt(0, QColor(76, 175, 80))
        gradient.setColorAt(0.6, QColor(255, 235, 59))
        gradient.setColorAt(0.85, QColor(255, 152, 0))
        gradient.setColorAt(1, QColor(244, 67, 54))
        
        painter.fillRect(level_rect, gradient)
        
        # Peak marker
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        if self.orientation == Qt.Orientation.Horizontal:
            peak_x = int(rect.left() + rect.width() * self.peak)
            painter.drawLine(peak_x, rect.top(), peak_x, rect.bottom())
        else:
            peak_y = int(rect.bottom() - rect.height() * self.peak)
            painter.drawLine(rect.left(), peak_y, rect.right(), peak_y)


class WaveformWidget(QWidget):
    """Audio waveform visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 80)
        self.samples = np.zeros(200)
    
    def set_samples(self, samples: np.ndarray):
        """Update waveform data."""
        if len(samples) > 0:
            # Resample to display size
            target_len = 200
            if len(samples) > target_len:
                indices = np.linspace(0, len(samples)-1, target_len, dtype=int)
                self.samples = samples[indices]
            else:
                self.samples = np.pad(samples, (0, target_len - len(samples)))
            self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        center_y = rect.height() // 2
        
        # Background
        painter.fillRect(rect, QColor(15, 15, 35))
        
        # Center line
        painter.setPen(QPen(QColor(50, 50, 80), 1))
        painter.drawLine(0, center_y, rect.width(), center_y)
        
        if len(self.samples) == 0:
            return
        
        # Waveform
        painter.setPen(QPen(QColor(100, 200, 255), 1))
        
        max_amp = max(np.max(np.abs(self.samples)), 0.01)
        scale = (rect.height() / 2 - 5) / max_amp
        
        step = rect.width() / len(self.samples)
        points = []
        
        for i, sample in enumerate(self.samples):
            x = int(i * step)
            y = center_y - int(sample * scale)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], 
                           points[i+1][0], points[i+1][1])


class MicrophoneLevelWidget(QFrame):
    """Individual microphone level display."""
    
    def __init__(self, mic_id: str, mic_name: str, parent=None):
        super().__init__(parent)
        self.mic_id = mic_id
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        self.name = QLabel(mic_name[:20])
        self.name.setStyleSheet("font-size: 11px; color: #aaa;")
        self.name.setFixedWidth(120)
        layout.addWidget(self.name)
        
        self.meter = AudioLevelMeter()
        layout.addWidget(self.meter, 1)
        
        self.db_label = QLabel("-∞ dB")
        self.db_label.setStyleSheet("font-size: 10px; color: #888; font-family: monospace;")
        self.db_label.setFixedWidth(50)
        layout.addWidget(self.db_label)
    
    def set_level(self, level: float, db: float):
        self.meter.set_level(level)
        if db < -60:
            self.db_label.setText("-∞ dB")
        else:
            self.db_label.setText(f"{db:.0f} dB")


class AudioPanel(QFrame):
    """Panel for audio visualization and microphone status."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("audio-panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("🎤 Audio Analysis")
        title.setObjectName("panel-title")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)
        
        # Waveform
        waveform_frame = QFrame()
        waveform_frame.setStyleSheet("QFrame { background: #0a0a18; border-radius: 8px; }")
        waveform_layout = QVBoxLayout(waveform_frame)
        waveform_layout.setContentsMargins(8, 8, 8, 8)
        
        self.waveform = WaveformWidget()
        waveform_layout.addWidget(self.waveform)
        layout.addWidget(waveform_frame)
        
        # Localization info
        loc_frame = QFrame()
        loc_layout = QGridLayout(loc_frame)
        loc_layout.setContentsMargins(8, 8, 8, 8)
        
        loc_layout.addWidget(QLabel("Sound Position:"), 0, 0)
        self.position_label = QLabel("-- m")
        self.position_label.setStyleSheet("color: #ff6b9d; font-weight: 500;")
        loc_layout.addWidget(self.position_label, 0, 1)
        
        loc_layout.addWidget(QLabel("Confidence:"), 0, 2)
        self.confidence_label = QLabel("--%")
        self.confidence_label.setStyleSheet("color: #00d4aa;")
        loc_layout.addWidget(self.confidence_label, 0, 3)
        
        loc_layout.addWidget(QLabel("Direction:"), 1, 0)
        self.direction_label = QLabel("--°")
        self.direction_label.setStyleSheet("color: #88aaff;")
        loc_layout.addWidget(self.direction_label, 1, 1)
        
        loc_layout.addWidget(QLabel("Event:"), 1, 2)
        self.event_label = QLabel("--")
        self.event_label.setStyleSheet("color: #ffaa00;")
        loc_layout.addWidget(self.event_label, 1, 3)
        
        layout.addWidget(loc_frame)
        
        # Microphone levels
        mic_label = QLabel("Microphone Levels")
        mic_label.setStyleSheet("font-size: 12px; color: #888; margin-top: 4px;")
        layout.addWidget(mic_label)
        
        self.mic_widgets: dict[str, MicrophoneLevelWidget] = {}
        self.mic_container = QWidget()
        self.mic_layout = QVBoxLayout(self.mic_container)
        self.mic_layout.setContentsMargins(0, 0, 0, 0)
        self.mic_layout.setSpacing(4)
        layout.addWidget(self.mic_container)
        
        layout.addStretch()
    
    def setup_microphones(self, microphones: list[dict]):
        """Initialize microphone level widgets."""
        # Clear existing
        for widget in self.mic_widgets.values():
            widget.deleteLater()
        self.mic_widgets.clear()
        
        # Clear layout
        while self.mic_layout.count():
            item = self.mic_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Create widgets for each mic (limit to 6 for UI)
        for mic in microphones[:6]:
            mic_id = mic.get('id', 'unknown')
            mic_name = mic.get('name', 'Microphone')[:25]
            
            widget = MicrophoneLevelWidget(mic_id, mic_name)
            self.mic_layout.addWidget(widget)
            self.mic_widgets[mic_id] = widget
    
    def update_waveform(self, samples: np.ndarray):
        """Update waveform display."""
        if samples is not None and len(samples) > 0:
            self.waveform.set_samples(samples)
    
    def update_mic_level(self, mic_id: str, level: float, db: float):
        """Update individual mic level."""
        if mic_id in self.mic_widgets:
            self.mic_widgets[mic_id].set_level(level, db)
    
    def update_localization(self, position: np.ndarray, confidence: float,
                           direction: float = 0, event: str = ""):
        """Update localization display."""
        if position is not None and len(position) >= 3:
            self.position_label.setText(f"[{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] m")
        self.confidence_label.setText(f"{confidence*100:.0f}%")
        self.direction_label.setText(f"{direction:.0f}°")
        self.event_label.setText(event or "--")
