"""
Camera Panel Widget for displaying camera feeds.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import cv2


class CameraFeedWidget(QFrame):
    """Single camera feed display widget."""
    
    def __init__(self, camera_id: str, camera_name: str, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = camera_name
        
        self.setObjectName("camera-feed")
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #0a0a18;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.video_label, 1)
        
        # Info overlay
        self.info_frame = QFrame()
        self.info_frame.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 0.7);
                border-radius: 6px;
                padding: 4px;
            }
        """)
        info_layout = QHBoxLayout(self.info_frame)
        info_layout.setContentsMargins(10, 6, 10, 6)
        
        self.name_label = QLabel(camera_name)
        self.name_label.setStyleSheet("color: white; font-weight: 500; font-size: 12px;")
        info_layout.addWidget(self.name_label)
        
        info_layout.addStretch()
        
        self.fps_label = QLabel("0 FPS")
        self.fps_label.setStyleSheet("color: #00ff88; font-size: 11px;")
        info_layout.addWidget(self.fps_label)
        
        self.status_label = QLabel("●")
        self.status_label.setStyleSheet("color: #00ff88; font-size: 14px;")
        info_layout.addWidget(self.status_label)
        
        layout.addWidget(self.info_frame)
        
        # Tracking overlays
        self.detections = []
        self.tracks = []
        
        # No-feed placeholder
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder when no feed."""
        self.video_label.setText("📹 No Signal")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #0a0a18;
                border-radius: 8px;
                color: #555;
                font-size: 16px;
            }
        """)
    
    def update_frame(self, frame: np.ndarray, fps: float = 0):
        """Update with new video frame."""
        if frame is None:
            self._show_placeholder()
            return
        
        # Draw detections/tracks on frame
        frame_display = frame.copy()
        self._draw_overlays(frame_display)
        
        # Convert to QImage
        h, w, ch = frame_display.shape
        bytes_per_line = ch * w
        
        if ch == 3:
            rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            qimg = QImage(frame_display.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        # Scale to fit label
        label_size = self.video_label.size()
        pixmap = QPixmap.fromImage(qimg).scaled(
            label_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(pixmap)
        self.video_label.setStyleSheet("background-color: #0a0a18; border-radius: 8px;")
        
        self.fps_label.setText(f"{fps:.1f} FPS")
    
    def _draw_overlays(self, frame: np.ndarray):
        """Draw detection boxes and track info."""
        # Draw detections
        for det in self.detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{det.get('class', 'obj')} {det.get('conf', 0):.2f}"
            cv2.putText(frame, label, (x1, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw tracks
        for track in self.tracks:
            x1, y1, x2, y2 = map(int, track['bbox'])
            color = (255, 100, 100) if track.get('is_speaking') else (100, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track.get('id', '?')}"
            cv2.putText(frame, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def set_detections(self, detections: list):
        """Update detection overlays."""
        self.detections = detections
    
    def set_tracks(self, tracks: list):
        """Update track overlays."""
        self.tracks = tracks
    
    def set_status(self, connected: bool):
        """Update connection status indicator."""
        if connected:
            self.status_label.setStyleSheet("color: #00ff88; font-size: 14px;")
        else:
            self.status_label.setStyleSheet("color: #ff6464; font-size: 14px;")


class CameraPanel(QFrame):
    """Panel containing all camera feeds in a grid."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("camera-panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("📹 Camera Feeds")
        title.setObjectName("panel-title")
        header.addWidget(title)
        header.addStretch()
        
        self.status_label = QLabel("0 Active")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        header.addWidget(self.status_label)
        layout.addLayout(header)
        
        # Grid for camera feeds
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        layout.addWidget(self.grid_widget, 1)
        
        self.camera_widgets: dict[str, CameraFeedWidget] = {}
    
    def setup_cameras(self, cameras: list[dict]):
        """Initialize camera feed widgets."""
        # Clear existing
        for widget in self.camera_widgets.values():
            widget.deleteLater()
        self.camera_widgets.clear()
        
        # Create new widgets in grid
        cols = 2 if len(cameras) > 1 else 1
        for i, cam in enumerate(cameras):
            row, col = divmod(i, cols)
            
            widget = CameraFeedWidget(cam['id'], cam['name'])
            self.grid_layout.addWidget(widget, row, col)
            self.camera_widgets[cam['id']] = widget
        
        self.status_label.setText(f"{len(cameras)} Active")
    
    def update_camera(self, camera_id: str, frame: np.ndarray, fps: float = 0):
        """Update a specific camera feed."""
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].update_frame(frame, fps)
    
    def set_camera_detections(self, camera_id: str, detections: list):
        """Set detections for a camera."""
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].set_detections(detections)
    
    def set_camera_tracks(self, camera_id: str, tracks: list):
        """Set tracks for a camera."""
        if camera_id in self.camera_widgets:
            self.camera_widgets[camera_id].set_tracks(tracks)
