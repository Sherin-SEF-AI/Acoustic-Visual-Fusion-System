"""
Analytics Panel - Statistics, charts, and performance metrics.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout, QProgressBar, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath
import numpy as np
from collections import deque
import time


class SparklineWidget(QWidget):
    """Mini chart showing recent values."""
    
    def __init__(self, color: str = "#6496ff", max_points: int = 60, parent=None):
        super().__init__(parent)
        self.color = QColor(color)
        self.max_points = max_points
        self.values = deque(maxlen=max_points)
        self.setMinimumSize(100, 30)
    
    def add_value(self, value: float):
        self.values.append(value)
        self.update()
    
    def paintEvent(self, event):
        if len(self.values) < 2:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(20, 20, 40))
        
        values = list(self.values)
        min_val = min(values)
        max_val = max(values)
        range_val = max(max_val - min_val, 0.001)
        
        # Create path
        path = QPainterPath()
        step = self.width() / (len(values) - 1)
        
        for i, v in enumerate(values):
            x = i * step
            y = self.height() - (v - min_val) / range_val * (self.height() - 4) - 2
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        
        # Draw filled area
        fill_path = QPainterPath(path)
        fill_path.lineTo(self.width(), self.height())
        fill_path.lineTo(0, self.height())
        fill_path.closeSubpath()
        
        painter.setPen(Qt.PenStyle.NoPen)
        fill_color = QColor(self.color)
        fill_color.setAlpha(50)
        painter.fillPath(fill_path, fill_color)
        
        # Draw line
        painter.setPen(QPen(self.color, 2))
        painter.drawPath(path)
        
        # Current value dot
        if values:
            x = self.width()
            y = self.height() - (values[-1] - min_val) / range_val * (self.height() - 4) - 2
            painter.setBrush(QBrush(self.color))
            painter.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)


class StatCard(QFrame):
    """Card showing a single statistic with sparkline."""
    
    def __init__(self, title: str, unit: str = "", color: str = "#6496ff", parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        
        self.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(title_label)
        
        # Value
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: 600;")
        layout.addWidget(self.value_label)
        
        # Sparkline
        self.sparkline = SparklineWidget(color)
        layout.addWidget(self.sparkline)
    
    def set_value(self, value: float, decimals: int = 1):
        self.value_label.setText(f"{value:.{decimals}f}{self.unit}")
        self.sparkline.add_value(value)


class HistogramWidget(QWidget):
    """Bar chart for distribution visualization."""
    
    def __init__(self, bins: int = 10, color: str = "#6496ff", parent=None):
        super().__init__(parent)
        self.bins = bins
        self.color = QColor(color)
        self.counts = np.zeros(bins)
        self.labels = [f"{i}" for i in range(bins)]
        self.setMinimumSize(200, 80)
    
    def set_data(self, counts: np.ndarray, labels: list = None):
        self.counts = counts
        if labels:
            self.labels = labels
        self.update()
    
    def paintEvent(self, event):
        if self.counts.sum() == 0:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(20, 20, 40))
        
        max_count = max(self.counts.max(), 1)
        bar_width = self.width() / len(self.counts) - 4
        
        for i, count in enumerate(self.counts):
            x = i * (bar_width + 4) + 2
            height = (count / max_count) * (self.height() - 20)
            y = self.height() - height - 15
            
            # Bar
            color = QColor(self.color)
            color.setAlpha(int(100 + 155 * (count / max_count)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(int(x), int(y), int(bar_width), int(height), 2, 2)
            
            # Label
            painter.setPen(QColor(100, 100, 120))
            if i < len(self.labels):
                painter.drawText(int(x), self.height() - 2, self.labels[i][:4])


class AnalyticsPanel(QFrame):
    """Panel showing system analytics and statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analytics-panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Header
        title = QLabel("📊 Analytics")
        title.setObjectName("panel-title")
        layout.addWidget(title)
        
        # Tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab { 
                padding: 8px 16px; 
                background: rgba(255,255,255,0.05);
                border-radius: 4px 4px 0 0;
                margin-right: 2px;
            }
            QTabBar::tab:selected { 
                background: rgba(100, 150, 255, 0.2);
            }
        """)
        
        # Performance tab
        perf_widget = QWidget()
        perf_layout = QGridLayout(perf_widget)
        perf_layout.setSpacing(8)
        
        self.fps_card = StatCard("Frame Rate", " fps", "#00d4aa")
        perf_layout.addWidget(self.fps_card, 0, 0)
        
        self.latency_card = StatCard("Latency", " ms", "#ff9800")
        perf_layout.addWidget(self.latency_card, 0, 1)
        
        self.cpu_card = StatCard("CPU Usage", "%", "#6496ff")
        perf_layout.addWidget(self.cpu_card, 1, 0)
        
        self.gpu_card = StatCard("GPU Usage", "%", "#a855f7")
        perf_layout.addWidget(self.gpu_card, 1, 1)
        
        tabs.addTab(perf_widget, "Performance")
        
        # Detection tab
        det_widget = QWidget()
        det_layout = QVBoxLayout(det_widget)
        
        self.detections_card = StatCard("Detections/sec", "", "#00d4aa")
        det_layout.addWidget(self.detections_card)
        
        det_layout.addWidget(QLabel("Detection Classes"))
        self.class_histogram = HistogramWidget(8, "#00d4aa")
        det_layout.addWidget(self.class_histogram)
        
        det_layout.addStretch()
        tabs.addTab(det_widget, "Detection")
        
        # Audio tab
        audio_widget = QWidget()
        audio_layout = QVBoxLayout(audio_widget)
        
        self.loc_accuracy_card = StatCard("Localization Conf", "%", "#ff6b9d")
        audio_layout.addWidget(self.loc_accuracy_card)
        
        self.events_card = StatCard("Audio Events", "/min", "#ff6b9d")
        audio_layout.addWidget(self.events_card)
        
        audio_layout.addStretch()
        tabs.addTab(audio_widget, "Audio")
        
        layout.addWidget(tabs, 1)
        
        # Session stats
        session_frame = QFrame()
        session_layout = QGridLayout(session_frame)
        session_layout.setContentsMargins(8, 8, 8, 8)
        
        session_layout.addWidget(QLabel("Session:"), 0, 0)
        self.session_time = QLabel("00:00:00")
        self.session_time.setStyleSheet("color: #888;")
        session_layout.addWidget(self.session_time, 0, 1)
        
        session_layout.addWidget(QLabel("Frames:"), 0, 2)
        self.frame_count = QLabel("0")
        self.frame_count.setStyleSheet("color: #888;")
        session_layout.addWidget(self.frame_count, 0, 3)
        
        session_layout.addWidget(QLabel("Tracks:"), 1, 0)
        self.track_count = QLabel("0")
        self.track_count.setStyleSheet("color: #888;")
        session_layout.addWidget(self.track_count, 1, 1)
        
        session_layout.addWidget(QLabel("Events:"), 1, 2)
        self.event_count = QLabel("0")
        self.event_count.setStyleSheet("color: #888;")
        session_layout.addWidget(self.event_count, 1, 3)
        
        layout.addWidget(session_frame)
        
        # State
        self.start_time = time.time()
        self.total_frames = 0
        self.total_tracks = 0
        self.total_events = 0
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_session_time)
        self.update_timer.start(1000)
    
    def _update_session_time(self):
        elapsed = int(time.time() - self.start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.session_time.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def update_performance(self, fps: float, latency: float, cpu: float, gpu: float):
        self.fps_card.set_value(fps)
        self.latency_card.set_value(latency, 0)
        self.cpu_card.set_value(cpu, 0)
        self.gpu_card.set_value(gpu, 0)
    
    def update_detection_stats(self, detections_per_sec: float, class_counts: dict):
        self.detections_card.set_value(detections_per_sec)
        
        # Update histogram
        classes = list(class_counts.keys())[:8]
        counts = np.array([class_counts.get(c, 0) for c in classes])
        self.class_histogram.set_data(counts, classes)
    
    def update_audio_stats(self, loc_confidence: float, events_per_min: float):
        self.loc_accuracy_card.set_value(loc_confidence * 100, 0)
        self.events_card.set_value(events_per_min)
    
    def increment_frame(self):
        self.total_frames += 1
        self.frame_count.setText(str(self.total_frames))
    
    def increment_track(self):
        self.total_tracks += 1
        self.track_count.setText(str(self.total_tracks))
    
    def increment_event(self):
        self.total_events += 1
        self.event_count.setText(str(self.total_events))
