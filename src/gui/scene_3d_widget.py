"""
3D Scene Viewer - OpenGL-based 3D visualization of the acoustic-visual scene.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PyQt6.QtCore import QPointF
import math


class Scene3DWidget(QWidget):
    """3D scene visualization using 2D projection (no OpenGL dependency)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        
        # Scene bounds (meters)
        self.x_range = (-5, 5)
        self.y_range = (-5, 5)
        self.z_range = (0, 3)
        
        # Camera view
        self.camera_azimuth = 45  # degrees
        self.camera_elevation = 30  # degrees
        self.camera_distance = 15
        self.camera_target = np.array([0, 0, 1])
        
        # Scene objects
        self.microphones = []  # List of (x, y, z)
        self.cameras = []  # List of (x, y, z, yaw, fov)
        self.sound_sources = []  # List of (x, y, z, confidence, label)
        self.persons = []  # List of (x, y, z, height, id, is_speaking)
        self.sound_rays = []  # List of (source_pos, direction)
        
        # Interaction
        self.last_mouse_pos = None
        self.setMouseTracking(True)
        
        # Animation
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._animate)
        self.animation_timer.start(50)
        self.animation_phase = 0
    
    def set_microphones(self, positions: list):
        """Set microphone positions."""
        self.microphones = positions
        self.update()
    
    def set_cameras(self, cameras: list):
        """Set camera positions and orientations."""
        self.cameras = cameras
        self.update()
    
    def set_sound_sources(self, sources: list):
        """Set localized sound source positions."""
        self.sound_sources = sources
        self.update()
    
    def set_persons(self, persons: list):
        """Set detected person positions."""
        self.persons = persons
        self.update()
    
    def project_3d_to_2d(self, point: np.ndarray) -> tuple:
        """Project 3D point to 2D screen coordinates using isometric-like projection."""
        x, y, z = point
        
        # Rotate around Z axis (azimuth)
        az = math.radians(self.camera_azimuth)
        x_rot = x * math.cos(az) - y * math.sin(az)
        y_rot = x * math.sin(az) + y * math.cos(az)
        
        # Apply elevation
        el = math.radians(self.camera_elevation)
        y_proj = y_rot * math.cos(el) - z * math.sin(el)
        z_proj = y_rot * math.sin(el) + z * math.cos(el)
        
        # Scale and center
        scale = min(self.width(), self.height()) / 12
        cx = self.width() / 2
        cy = self.height() / 2
        
        screen_x = cx + x_rot * scale
        screen_y = cy - z_proj * scale  # Flip Y for screen coords
        
        return (screen_x, screen_y, y_proj)  # Return depth for sorting
    
    def _animate(self):
        """Animation tick."""
        self.animation_phase = (self.animation_phase + 1) % 100
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background with gradient
        gradient = QColor(15, 15, 35)
        painter.fillRect(self.rect(), gradient)
        
        # Draw grid floor
        self._draw_floor_grid(painter)
        
        # Draw axes
        self._draw_axes(painter)
        
        # Draw room bounds
        self._draw_room(painter)
        
        # Collect all drawable objects with depth
        drawables = []
        
        # Microphones
        for mic in self.microphones:
            sx, sy, depth = self.project_3d_to_2d(np.array(mic[:3]))
            drawables.append((depth, 'mic', sx, sy, mic))
        
        # Cameras
        for cam in self.cameras:
            sx, sy, depth = self.project_3d_to_2d(np.array(cam[:3]))
            drawables.append((depth, 'cam', sx, sy, cam))
        
        # Persons
        for person in self.persons:
            sx, sy, depth = self.project_3d_to_2d(np.array(person[:3]))
            drawables.append((depth, 'person', sx, sy, person))
        
        # Sound sources
        for source in self.sound_sources:
            sx, sy, depth = self.project_3d_to_2d(np.array(source[:3]))
            drawables.append((depth, 'sound', sx, sy, source))
        
        # Sort by depth (back to front)
        drawables.sort(key=lambda x: x[0], reverse=True)
        
        # Draw in order
        for item in drawables:
            if item[1] == 'mic':
                self._draw_microphone(painter, item[2], item[3], item[4])
            elif item[1] == 'cam':
                self._draw_camera(painter, item[2], item[3], item[4])
            elif item[1] == 'person':
                self._draw_person(painter, item[2], item[3], item[4])
            elif item[1] == 'sound':
                self._draw_sound_source(painter, item[2], item[3], item[4])
    
    def _draw_floor_grid(self, painter: QPainter):
        """Draw floor grid."""
        painter.setPen(QPen(QColor(40, 40, 60), 1))
        
        for x in range(-5, 6, 1):
            p1 = self.project_3d_to_2d(np.array([x, -5, 0]))
            p2 = self.project_3d_to_2d(np.array([x, 5, 0]))
            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
        
        for y in range(-5, 6, 1):
            p1 = self.project_3d_to_2d(np.array([-5, y, 0]))
            p2 = self.project_3d_to_2d(np.array([5, y, 0]))
            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
    
    def _draw_axes(self, painter: QPainter):
        """Draw coordinate axes."""
        origin = self.project_3d_to_2d(np.array([0, 0, 0]))
        
        # X axis (red)
        x_end = self.project_3d_to_2d(np.array([2, 0, 0]))
        painter.setPen(QPen(QColor(255, 80, 80), 2))
        painter.drawLine(int(origin[0]), int(origin[1]), int(x_end[0]), int(x_end[1]))
        painter.drawText(int(x_end[0]) + 5, int(x_end[1]), "X")
        
        # Y axis (green)
        y_end = self.project_3d_to_2d(np.array([0, 2, 0]))
        painter.setPen(QPen(QColor(80, 255, 80), 2))
        painter.drawLine(int(origin[0]), int(origin[1]), int(y_end[0]), int(y_end[1]))
        painter.drawText(int(y_end[0]) + 5, int(y_end[1]), "Y")
        
        # Z axis (blue)
        z_end = self.project_3d_to_2d(np.array([0, 0, 2]))
        painter.setPen(QPen(QColor(80, 80, 255), 2))
        painter.drawLine(int(origin[0]), int(origin[1]), int(z_end[0]), int(z_end[1]))
        painter.drawText(int(z_end[0]) + 5, int(z_end[1]), "Z")
    
    def _draw_room(self, painter: QPainter):
        """Draw room boundaries."""
        painter.setPen(QPen(QColor(60, 60, 80), 1, Qt.PenStyle.DashLine))
        
        # Floor corners
        corners = [
            np.array([-4, -4, 0]), np.array([4, -4, 0]),
            np.array([4, 4, 0]), np.array([-4, 4, 0])
        ]
        
        # Draw floor boundary
        for i in range(4):
            p1 = self.project_3d_to_2d(corners[i])
            p2 = self.project_3d_to_2d(corners[(i + 1) % 4])
            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
    
    def _draw_microphone(self, painter: QPainter, x: float, y: float, mic: tuple):
        """Draw microphone marker."""
        painter.setPen(QPen(QColor(100, 150, 255), 2))
        painter.setBrush(QBrush(QColor(100, 150, 255, 150)))
        
        # Draw as small diamond
        size = 8
        points = QPolygonF([
            QPointF(x, y - size),
            QPointF(x + size, y),
            QPointF(x, y + size),
            QPointF(x - size, y)
        ])
        painter.drawPolygon(points)
    
    def _draw_camera(self, painter: QPainter, x: float, y: float, cam: tuple):
        """Draw camera marker with FOV cone."""
        painter.setPen(QPen(QColor(100, 255, 150), 2))
        painter.setBrush(QBrush(QColor(100, 255, 150, 100)))
        
        # Camera body
        painter.drawRect(int(x) - 10, int(y) - 6, 20, 12)
        
        # Lens
        painter.drawEllipse(int(x) + 8, int(y) - 4, 8, 8)
        
        # FOV indicator if we have orientation
        if len(cam) >= 4:
            yaw = cam[3]
            fov = cam[4] if len(cam) >= 5 else 60
            
            painter.setOpacity(0.2)
            # Draw FOV cone projection
            painter.setOpacity(1.0)
    
    def _draw_person(self, painter: QPainter, x: float, y: float, person: tuple):
        """Draw person as a 3D-ish figure."""
        is_speaking = person[5] if len(person) > 5 else False
        person_id = person[4] if len(person) > 4 else "?"
        
        # Color based on speaking
        if is_speaking:
            color = QColor(255, 100, 150)
            head_color = QColor(255, 150, 180)
        else:
            color = QColor(100, 200, 255)
            head_color = QColor(150, 220, 255)
        
        # Body
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color.darker(120)))
        
        # Torso (ellipse)
        painter.drawEllipse(int(x) - 12, int(y) - 20, 24, 40)
        
        # Head
        painter.setBrush(QBrush(head_color))
        painter.drawEllipse(int(x) - 8, int(y) - 36, 16, 16)
        
        # ID label
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(int(x) - 5, int(y) + 55, str(person_id))
        
        # Speaking indicator
        if is_speaking:
            pulse = (self.animation_phase % 20) / 20.0
            radius = int(20 + 10 * pulse)
            painter.setPen(QPen(QColor(255, 100, 150, int(150 * (1 - pulse))), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(x) - radius, int(y) - 10 - radius, 
                               radius * 2, radius * 2)
    
    def _draw_sound_source(self, painter: QPainter, x: float, y: float, source: tuple):
        """Draw sound source with pulsing animation."""
        confidence = source[3] if len(source) > 3 else 1.0
        label = source[4] if len(source) > 4 else ""
        
        # Pulsing effect
        pulse = (self.animation_phase % 25) / 25.0
        base_radius = 12
        radius = int(base_radius + 8 * pulse)
        
        # Outer pulse ring
        alpha = int(200 * (1 - pulse) * confidence)
        painter.setPen(QPen(QColor(255, 100, 100, alpha), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(int(x) - radius, int(y) - radius, radius * 2, radius * 2)
        
        # Inner solid circle
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 80, 80, int(200 * confidence))))
        painter.drawEllipse(int(x) - 8, int(y) - 8, 16, 16)
        
        # Label
        if label:
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(int(x) - 20, int(y) + 20, label)
    
    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position()
    
    def mouseMoveEvent(self, event):
        if self.last_mouse_pos and event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.position() - self.last_mouse_pos
            self.camera_azimuth += delta.x() * 0.5
            self.camera_elevation = max(5, min(85, self.camera_elevation + delta.y() * 0.5))
            self.last_mouse_pos = event.position()
            self.update()
    
    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.camera_distance = max(5, min(30, self.camera_distance - delta * 0.01))
        self.update()


class Scene3DPanel(QFrame):
    """Panel containing 3D scene viewer with controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("scene3d-panel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("🌐 3D Scene View")
        title.setStyleSheet("font-size: 13px; font-weight: 500; color: #88aaff;")
        header.addWidget(title)
        
        header.addStretch()
        
        hint = QLabel("Drag to rotate, scroll to zoom")
        hint.setStyleSheet("color: #666; font-size: 10px;")
        header.addWidget(hint)
        
        layout.addLayout(header)
        
        # 3D view
        self.scene = Scene3DWidget()
        layout.addWidget(self.scene, 1)
