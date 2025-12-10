"""
Heatmap Widget - Activity visualization with heat overlays.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QComboBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QImage, QPixmap, QLinearGradient
import time


class HeatmapWidget(QWidget):
    """Displays activity heatmap overlayed on spatial view."""
    
    # Colormap: blue -> cyan -> green -> yellow -> red
    COLORMAP = [
        (0.0, (0, 0, 128)),      # Dark blue
        (0.25, (0, 255, 255)),    # Cyan
        (0.5, (0, 255, 0)),       # Green
        (0.75, (255, 255, 0)),    # Yellow
        (1.0, (255, 0, 0))        # Red
    ]
    
    def __init__(self, grid_size: int = 50, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        
        # Grid configuration
        self.grid_size = grid_size
        self.heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Spatial bounds (meters)
        self.x_range = (-5, 5)
        self.y_range = (-5, 5)
        
        # Decay rate for temporal smoothing
        self.decay_rate = 0.98
        
        # Gaussian kernel for point spreading
        self.kernel_size = 5
        self.kernel = self._create_gaussian_kernel(self.kernel_size)
        
        # Mode
        self.mode = "positions"  # positions, detections, sound
        
        # Update timer for decay
        self.decay_timer = QTimer()
        self.decay_timer.timeout.connect(self._apply_decay)
        self.decay_timer.start(100)
    
    def _create_gaussian_kernel(self, size: int) -> np.ndarray:
        """Create a Gaussian kernel for heat spreading."""
        sigma = size / 3
        x = np.arange(size) - size // 2
        xx, yy = np.meshgrid(x, x)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.max()
    
    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * self.grid_size)
        gy = int((y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * self.grid_size)
        return (
            max(0, min(self.grid_size - 1, gx)),
            max(0, min(self.grid_size - 1, gy))
        )
    
    def add_point(self, x: float, y: float, intensity: float = 1.0):
        """Add a heat point at world coordinates."""
        gx, gy = self.world_to_grid(x, y)
        
        # Apply kernel around point
        half = self.kernel_size // 2
        for ki in range(self.kernel_size):
            for kj in range(self.kernel_size):
                ni = gy - half + ki
                nj = gx - half + kj
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    self.heatmap[ni, nj] += self.kernel[ki, kj] * intensity
        
        # Clamp to max
        self.heatmap = np.clip(self.heatmap, 0, 1)
        self.update()
    
    def add_track(self, track: dict):
        """Add heat from a tracking result."""
        bbox = track.get('bbox', [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # Map bbox position to world (simplified)
        x = (cx / 640 - 0.5) * 10  # Assuming 640 width
        y = (cy / 480 - 0.5) * 10
        
        intensity = 0.3 if track.get('is_speaking') else 0.1
        self.add_point(x, y, intensity)
    
    def add_sound_source(self, position: np.ndarray, confidence: float):
        """Add heat from sound localization."""
        self.add_point(position[0], position[1], confidence * 0.5)
    
    def _apply_decay(self):
        """Apply decay to heatmap."""
        self.heatmap *= self.decay_rate
        self.update()
    
    def clear(self):
        """Clear the heatmap."""
        self.heatmap.fill(0)
        self.update()
    
    def get_color(self, value: float) -> QColor:
        """Get color from colormap for a value 0-1."""
        value = max(0, min(1, value))
        
        # Find colormap segment
        for i in range(len(self.COLORMAP) - 1):
            t0, c0 = self.COLORMAP[i]
            t1, c1 = self.COLORMAP[i + 1]
            
            if t0 <= value <= t1:
                # Interpolate
                t = (value - t0) / (t1 - t0)
                r = int(c0[0] + t * (c1[0] - c0[0]))
                g = int(c0[1] + t * (c1[1] - c0[1]))
                b = int(c0[2] + t * (c1[2] - c0[2]))
                return QColor(r, g, b)
        
        return QColor(255, 0, 0)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(10, 10, 30))
        
        # Draw heatmap
        cell_w = self.width() / self.grid_size
        cell_h = self.height() / self.grid_size
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self.heatmap[i, j]
                if value > 0.01:
                    color = self.get_color(value)
                    color.setAlpha(int(150 * value + 50))
                    painter.fillRect(
                        int(j * cell_w), int(i * cell_h),
                        int(cell_w) + 1, int(cell_h) + 1,
                        color
                    )
        
        # Draw grid overlay
        painter.setPen(QColor(50, 50, 70))
        for i in range(0, self.grid_size + 1, 10):
            x = int(i * cell_w)
            y = int(i * cell_h)
            painter.drawLine(x, 0, x, self.height())
            painter.drawLine(0, y, self.width(), y)
        
        # Draw axes labels
        painter.setPen(QColor(100, 100, 120))
        painter.drawText(5, 15, f"{self.y_range[0]:.0f}m")
        painter.drawText(5, self.height() - 5, f"{self.y_range[1]:.0f}m")
        painter.drawText(self.width() - 25, self.height() - 5, f"{self.x_range[1]:.0f}m")


class HeatmapPanel(QFrame):
    """Panel containing heatmap with controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("heatmap-panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("🔥 Activity Heatmap")
        title.setStyleSheet("font-size: 13px; font-weight: 500; color: #ff9800;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Mode selector
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Positions", "Detections", "Sound", "Combined"])
        self.mode_combo.setStyleSheet("background: rgba(0,0,0,0.3); padding: 4px;")
        header.addWidget(self.mode_combo)
        
        layout.addLayout(header)
        
        # Heatmap
        self.heatmap = HeatmapWidget()
        layout.addWidget(self.heatmap, 1)
        
        # Stats
        stats = QHBoxLayout()
        
        self.max_label = QLabel("Max: 0.00")
        self.max_label.setStyleSheet("color: #888; font-size: 11px;")
        stats.addWidget(self.max_label)
        
        stats.addStretch()
        
        self.total_label = QLabel("Total: 0.00")
        self.total_label.setStyleSheet("color: #888; font-size: 11px;")
        stats.addWidget(self.total_label)
        
        layout.addLayout(stats)
        
        # Update stats timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(500)
    
    def _update_stats(self):
        max_val = self.heatmap.heatmap.max()
        total = self.heatmap.heatmap.sum()
        self.max_label.setText(f"Max: {max_val:.2f}")
        self.total_label.setText(f"Total: {total:.1f}")
    
    def add_track(self, track: dict):
        self.heatmap.add_track(track)
    
    def add_sound_source(self, position: np.ndarray, confidence: float):
        self.heatmap.add_sound_source(position, confidence)
