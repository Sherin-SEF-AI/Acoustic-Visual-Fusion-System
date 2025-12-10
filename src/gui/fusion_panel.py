"""
Fusion Panel Widget for audio-visual correlation display.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
import numpy as np


class SpatialMapWidget(QWidget):
    """2D spatial map showing sound sources and visual tracks."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        
        self.sound_sources = []  # List of (x, y, confidence)
        self.visual_tracks = []  # List of (x, y, id, is_speaking)
        self.microphones = []  # List of (x, y)
        self.cameras = []  # List of (x, y, angle)
        
        self.scale = 50  # pixels per meter
        self.center_offset = (0, 0)
    
    def set_sound_sources(self, sources: list):
        self.sound_sources = sources
        self.update()
    
    def set_visual_tracks(self, tracks: list):
        self.visual_tracks = tracks
        self.update()
    
    def set_microphones(self, mics: list):
        self.microphones = mics
        self.update()
    
    def set_cameras(self, cams: list):
        self.cameras = cams
        self.update()
    
    def world_to_screen(self, x: float, y: float) -> tuple:
        """Convert world coords to screen coords."""
        cx = self.width() // 2 + self.center_offset[0]
        cy = self.height() // 2 + self.center_offset[1]
        return (int(cx + x * self.scale), int(cy - y * self.scale))
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(15, 15, 35))
        
        # Grid
        painter.setPen(QPen(QColor(40, 40, 60), 1))
        cx, cy = self.width() // 2, self.height() // 2
        
        for i in range(-5, 6):
            x = cx + i * self.scale
            y = cy + i * self.scale
            painter.drawLine(x, 0, x, self.height())
            painter.drawLine(0, y, self.width(), y)
        
        # Axes
        painter.setPen(QPen(QColor(80, 80, 100), 2))
        painter.drawLine(cx, 0, cx, self.height())
        painter.drawLine(0, cy, self.width(), cy)
        
        # Microphones
        painter.setPen(QPen(QColor(100, 150, 255), 2))
        painter.setBrush(QBrush(QColor(100, 150, 255, 100)))
        for mx, my in self.microphones:
            sx, sy = self.world_to_screen(mx, my)
            painter.drawEllipse(sx - 6, sy - 6, 12, 12)
        
        # Cameras
        painter.setPen(QPen(QColor(150, 255, 150), 2))
        painter.setBrush(QBrush(QColor(150, 255, 150, 100)))
        for cam in self.cameras:
            cx, cy = cam[:2]
            sx, sy = self.world_to_screen(cx, cy)
            painter.drawRect(sx - 8, sy - 5, 16, 10)
        
        # Visual tracks
        for track in self.visual_tracks:
            tx, ty = track[:2]
            sx, sy = self.world_to_screen(tx, ty)
            
            is_speaking = track[3] if len(track) > 3 else False
            if is_speaking:
                painter.setPen(QPen(QColor(255, 100, 150), 2))
                painter.setBrush(QBrush(QColor(255, 100, 150, 150)))
            else:
                painter.setPen(QPen(QColor(100, 200, 255), 2))
                painter.setBrush(QBrush(QColor(100, 200, 255, 150)))
            
            painter.drawEllipse(sx - 10, sy - 10, 20, 20)
            
            track_id = track[2] if len(track) > 2 else "?"
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(sx - 4, sy + 4, str(track_id))
        
        # Sound sources
        for source in self.sound_sources:
            sx, sy = source[:2]
            conf = source[2] if len(source) > 2 else 1.0
            
            px, py = self.world_to_screen(sx, sy)
            radius = int(15 + 10 * conf)
            
            # Pulsing effect
            painter.setPen(QPen(QColor(255, 100, 100, int(200 * conf)), 2))
            painter.setBrush(QBrush(QColor(255, 100, 100, int(100 * conf))))
            painter.drawEllipse(px - radius, py - radius, radius * 2, radius * 2)
            
            # Center dot
            painter.setBrush(QBrush(QColor(255, 50, 50)))
            painter.drawEllipse(px - 4, py - 4, 8, 8)


class CorrelationTableWidget(QTableWidget):
    """Table showing audio-visual correlations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels([
            "Track", "Audio Source", "Confidence", "Speaking", "Distance"
        ])
        
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                background: rgba(0, 0, 0, 0.2);
                border: none;
                gridline-color: rgba(255, 255, 255, 0.1);
            }
            QTableWidget::item {
                padding: 6px;
            }
            QHeaderView::section {
                background: rgba(100, 150, 255, 0.2);
                padding: 8px;
                border: none;
                font-weight: 500;
            }
        """)
    
    def update_correlations(self, correlations: list[dict]):
        """Update table with correlation data."""
        self.setRowCount(len(correlations))
        
        for i, corr in enumerate(correlations):
            self.setItem(i, 0, QTableWidgetItem(str(corr.get('track_id', '--'))))
            self.setItem(i, 1, QTableWidgetItem(f"{corr.get('audio_pos', '--')}"))
            
            conf = corr.get('confidence', 0)
            conf_item = QTableWidgetItem(f"{conf*100:.0f}%")
            conf_item.setForeground(QColor("#00d4aa") if conf > 0.7 else QColor("#ff9800"))
            self.setItem(i, 2, conf_item)
            
            speaking = "🗣️" if corr.get('is_speaking') else "—"
            self.setItem(i, 3, QTableWidgetItem(speaking))
            
            dist = corr.get('distance', 0)
            self.setItem(i, 4, QTableWidgetItem(f"{dist:.2f} m"))


class FusionPanel(QFrame):
    """Panel showing audio-visual fusion and correlations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fusion-panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("🔗 Audio-Visual Fusion")
        title.setObjectName("panel-title")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)
        
        # Spatial map
        map_label = QLabel("Spatial View")
        map_label.setStyleSheet("font-size: 12px; color: #888;")
        layout.addWidget(map_label)
        
        self.spatial_map = SpatialMapWidget()
        self.spatial_map.setStyleSheet("border-radius: 8px;")
        layout.addWidget(self.spatial_map, 1)
        
        # Correlation table
        table_label = QLabel("Active Correlations")
        table_label.setStyleSheet("font-size: 12px; color: #888;")
        layout.addWidget(table_label)
        
        self.correlation_table = CorrelationTableWidget()
        self.correlation_table.setMaximumHeight(150)
        layout.addWidget(self.correlation_table)
        
        # Stats
        stats_frame = QFrame()
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        
        self.active_sources = QLabel("Active Sources: 0")
        self.active_sources.setStyleSheet("color: #ff6b9d;")
        stats_layout.addWidget(self.active_sources)
        
        self.active_tracks = QLabel("Active Tracks: 0")
        self.active_tracks.setStyleSheet("color: #00d4aa;")
        stats_layout.addWidget(self.active_tracks)
        
        self.matched = QLabel("Matched: 0")
        self.matched.setStyleSheet("color: #a855f7;")
        stats_layout.addWidget(self.matched)
        
        layout.addWidget(stats_frame)
    
    def update_spatial_map(self, sources: list, tracks: list, mics: list = None, cams: list = None):
        """Update spatial map display."""
        self.spatial_map.set_sound_sources(sources)
        self.spatial_map.set_visual_tracks(tracks)
        if mics:
            self.spatial_map.set_microphones(mics)
        if cams:
            self.spatial_map.set_cameras(cams)
        
        self.active_sources.setText(f"Active Sources: {len(sources)}")
        self.active_tracks.setText(f"Active Tracks: {len(tracks)}")
    
    def update_correlations(self, correlations: list[dict]):
        """Update correlation table."""
        self.correlation_table.update_correlations(correlations)
        matched = sum(1 for c in correlations if c.get('confidence', 0) > 0.5)
        self.matched.setText(f"Matched: {matched}")
