"""
Settings Dialog - Configuration and preferences.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QTabWidget, QWidget, QGridLayout,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QLineEdit, QGroupBox, QScrollArea,
    QColorDialog, QFileDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from loguru import logger


class SettingsDialog(QDialog):
    """Settings dialog for system configuration."""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.changes = {}
        
        self.setWindowTitle("⚙️ Settings")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("""
            QDialog {
                background: #1a1a2e;
            }
            QLabel {
                color: #ddd;
            }
            QGroupBox {
                font-weight: 500;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                color: #88aaff;
                subcontrol-origin: margin;
                left: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Tabs
        tabs = QTabWidget()
        
        # Video tab
        tabs.addTab(self._create_video_tab(), "📹 Video")
        
        # Audio tab
        tabs.addTab(self._create_audio_tab(), "🎤 Audio")
        
        # Fusion tab
        tabs.addTab(self._create_fusion_tab(), "🔗 Fusion")
        
        # Display tab
        tabs.addTab(self._create_display_tab(), "🎨 Display")
        
        # Advanced tab
        tabs.addTab(self._create_advanced_tab(), "⚡ Advanced")
        
        layout.addWidget(tabs)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        
        layout.addWidget(buttons)
    
    def _create_video_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)
        
        # Detection settings
        det_group = QGroupBox("Object Detection")
        det_layout = QGridLayout(det_group)
        
        det_layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self.model_combo.setCurrentText("yolov8m")
        det_layout.addWidget(self.model_combo, 0, 1)
        
        det_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.1, 0.9)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.setValue(0.5)
        det_layout.addWidget(self.conf_threshold, 1, 1)
        
        det_layout.addWidget(QLabel("IOU Threshold:"), 2, 0)
        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.1, 0.9)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.setValue(0.45)
        det_layout.addWidget(self.iou_threshold, 2, 1)
        
        self.detect_persons = QCheckBox("Detect Persons")
        self.detect_persons.setChecked(True)
        det_layout.addWidget(self.detect_persons, 3, 0, 1, 2)
        
        layout.addWidget(det_group)
        
        # Tracking settings
        track_group = QGroupBox("Multi-Object Tracking")
        track_layout = QGridLayout(track_group)
        
        track_layout.addWidget(QLabel("Track Buffer (frames):"), 0, 0)
        self.track_buffer = QSpinBox()
        self.track_buffer.setRange(10, 120)
        self.track_buffer.setValue(30)
        track_layout.addWidget(self.track_buffer, 0, 1)
        
        track_layout.addWidget(QLabel("Match Threshold:"), 1, 0)
        self.match_threshold = QDoubleSpinBox()
        self.match_threshold.setRange(0.1, 0.9)
        self.match_threshold.setValue(0.8)
        track_layout.addWidget(self.match_threshold, 1, 1)
        
        layout.addWidget(track_group)
        
        # Camera settings
        cam_group = QGroupBox("Camera")
        cam_layout = QGridLayout(cam_group)
        
        cam_layout.addWidget(QLabel("Resolution:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        cam_layout.addWidget(self.resolution_combo, 0, 1)
        
        cam_layout.addWidget(QLabel("Target FPS:"), 1, 0)
        self.target_fps = QSpinBox()
        self.target_fps.setRange(10, 60)
        self.target_fps.setValue(30)
        cam_layout.addWidget(self.target_fps, 1, 1)
        
        layout.addWidget(cam_group)
        layout.addStretch()
        
        return widget
    
    def _create_audio_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Capture settings
        cap_group = QGroupBox("Audio Capture")
        cap_layout = QGridLayout(cap_group)
        
        cap_layout.addWidget(QLabel("Sample Rate:"), 0, 0)
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["44100", "48000", "96000"])
        self.sample_rate.setCurrentText("48000")
        cap_layout.addWidget(self.sample_rate, 0, 1)
        
        cap_layout.addWidget(QLabel("Buffer Size:"), 1, 0)
        self.buffer_size = QSpinBox()
        self.buffer_size.setRange(128, 4096)
        self.buffer_size.setValue(1024)
        cap_layout.addWidget(self.buffer_size, 1, 1)
        
        layout.addWidget(cap_group)
        
        # Localization settings
        loc_group = QGroupBox("Sound Localization")
        loc_layout = QGridLayout(loc_group)
        
        loc_layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.loc_algorithm = QComboBox()
        self.loc_algorithm.addItems(["GCC-PHAT", "SRP-PHAT", "MUSIC"])
        loc_layout.addWidget(self.loc_algorithm, 0, 1)
        
        loc_layout.addWidget(QLabel("Speed of Sound (m/s):"), 1, 0)
        self.speed_of_sound = QDoubleSpinBox()
        self.speed_of_sound.setRange(300, 400)
        self.speed_of_sound.setValue(343.0)
        loc_layout.addWidget(self.speed_of_sound, 1, 1)
        
        self.enable_kalman = QCheckBox("Enable Kalman Filtering")
        self.enable_kalman.setChecked(True)
        loc_layout.addWidget(self.enable_kalman, 2, 0, 1, 2)
        
        layout.addWidget(loc_group)
        
        # Event detection
        event_group = QGroupBox("Event Detection")
        event_layout = QGridLayout(event_group)
        
        event_layout.addWidget(QLabel("Energy Threshold:"), 0, 0)
        self.energy_threshold = QDoubleSpinBox()
        self.energy_threshold.setRange(0.001, 0.1)
        self.energy_threshold.setDecimals(3)
        self.energy_threshold.setValue(0.01)
        event_layout.addWidget(self.energy_threshold, 0, 1)
        
        self.enable_vad = QCheckBox("Voice Activity Detection")
        self.enable_vad.setChecked(True)
        event_layout.addWidget(self.enable_vad, 1, 0, 1, 2)
        
        layout.addWidget(event_group)
        layout.addStretch()
        
        return widget
    
    def _create_fusion_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Spatial correlation
        spatial_group = QGroupBox("Spatial Correlation")
        spatial_layout = QGridLayout(spatial_group)
        
        spatial_layout.addWidget(QLabel("Max Distance (m):"), 0, 0)
        self.max_distance = QDoubleSpinBox()
        self.max_distance.setRange(0.5, 10.0)
        self.max_distance.setValue(3.0)
        spatial_layout.addWidget(self.max_distance, 0, 1)
        
        spatial_layout.addWidget(QLabel("Min Confidence:"), 1, 0)
        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0.1, 0.9)
        self.min_confidence.setValue(0.5)
        spatial_layout.addWidget(self.min_confidence, 1, 1)
        
        layout.addWidget(spatial_group)
        
        # Temporal
        temporal_group = QGroupBox("Temporal Alignment")
        temporal_layout = QGridLayout(temporal_group)
        
        temporal_layout.addWidget(QLabel("Window (ms):"), 0, 0)
        self.temporal_window = QSpinBox()
        self.temporal_window.setRange(50, 500)
        self.temporal_window.setValue(200)
        temporal_layout.addWidget(self.temporal_window, 0, 1)
        
        layout.addWidget(temporal_group)
        
        # Speaking detection
        speak_group = QGroupBox("Speaking Detection")
        speak_layout = QGridLayout(speak_group)
        
        self.use_lip_movement = QCheckBox("Use Lip Movement Analysis")
        self.use_lip_movement.setChecked(True)
        speak_layout.addWidget(self.use_lip_movement, 0, 0, 1, 2)
        
        self.use_head_pose = QCheckBox("Use Head Pose")
        self.use_head_pose.setChecked(True)
        speak_layout.addWidget(self.use_head_pose, 1, 0, 1, 2)
        
        layout.addWidget(speak_group)
        layout.addStretch()
        
        return widget
    
    def _create_display_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Overlay settings
        overlay_group = QGroupBox("Video Overlays")
        overlay_layout = QVBoxLayout(overlay_group)
        
        self.show_detections = QCheckBox("Show Detection Boxes")
        self.show_detections.setChecked(True)
        overlay_layout.addWidget(self.show_detections)
        
        self.show_tracks = QCheckBox("Show Track IDs")
        self.show_tracks.setChecked(True)
        overlay_layout.addWidget(self.show_tracks)
        
        self.show_confidence = QCheckBox("Show Confidence Scores")
        self.show_confidence.setChecked(True)
        overlay_layout.addWidget(self.show_confidence)
        
        self.show_pose = QCheckBox("Show Pose Skeleton")
        self.show_pose.setChecked(False)
        overlay_layout.addWidget(self.show_pose)
        
        self.show_speaking = QCheckBox("Highlight Speaking Persons")
        self.show_speaking.setChecked(True)
        overlay_layout.addWidget(self.show_speaking)
        
        layout.addWidget(overlay_group)
        
        # 3D View
        view_group = QGroupBox("3D View")
        view_layout = QGridLayout(view_group)
        
        view_layout.addWidget(QLabel("Grid Size (m):"), 0, 0)
        self.grid_size = QSpinBox()
        self.grid_size.setRange(5, 20)
        self.grid_size.setValue(10)
        view_layout.addWidget(self.grid_size, 0, 1)
        
        self.show_grid = QCheckBox("Show Floor Grid")
        self.show_grid.setChecked(True)
        view_layout.addWidget(self.show_grid, 1, 0, 1, 2)
        
        self.show_rays = QCheckBox("Show Sound Rays")
        self.show_rays.setChecked(True)
        view_layout.addWidget(self.show_rays, 2, 0, 1, 2)
        
        layout.addWidget(view_group)
        layout.addStretch()
        
        return widget
    
    def _create_advanced_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance
        perf_group = QGroupBox("Performance")
        perf_layout = QGridLayout(perf_group)
        
        perf_layout.addWidget(QLabel("Processing Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        perf_layout.addWidget(self.device_combo, 0, 1)
        
        perf_layout.addWidget(QLabel("Worker Threads:"), 1, 0)
        self.worker_threads = QSpinBox()
        self.worker_threads.setRange(1, 16)
        self.worker_threads.setValue(4)
        perf_layout.addWidget(self.worker_threads, 1, 1)
        
        self.half_precision = QCheckBox("Use FP16 (Half Precision)")
        self.half_precision.setChecked(False)
        perf_layout.addWidget(self.half_precision, 2, 0, 1, 2)
        
        layout.addWidget(perf_group)
        
        # Paths
        paths_group = QGroupBox("Paths")
        paths_layout = QGridLayout(paths_group)
        
        paths_layout.addWidget(QLabel("Recordings:"), 0, 0)
        self.recordings_path = QLineEdit("recordings")
        paths_layout.addWidget(self.recordings_path, 0, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_recordings)
        paths_layout.addWidget(browse_btn, 0, 2)
        
        paths_layout.addWidget(QLabel("Logs:"), 1, 0)
        self.logs_path = QLineEdit("logs")
        paths_layout.addWidget(self.logs_path, 1, 1)
        
        layout.addWidget(paths_group)
        
        # Debug
        debug_group = QGroupBox("Debug")
        debug_layout = QVBoxLayout(debug_group)
        
        self.verbose_logging = QCheckBox("Verbose Logging")
        debug_layout.addWidget(self.verbose_logging)
        
        self.save_debug_frames = QCheckBox("Save Debug Frames")
        debug_layout.addWidget(self.save_debug_frames)
        
        layout.addWidget(debug_group)
        layout.addStretch()
        
        return widget
    
    def _browse_recordings(self):
        path = QFileDialog.getExistingDirectory(self, "Select Recordings Directory")
        if path:
            self.recordings_path.setText(path)
    
    def _apply(self):
        self._collect_settings()
        self.settings_changed.emit(self.changes)
    
    def _collect_settings(self):
        self.changes = {
            'video': {
                'model': self.model_combo.currentText(),
                'confidence_threshold': self.conf_threshold.value(),
                'iou_threshold': self.iou_threshold.value(),
                'track_buffer': self.track_buffer.value(),
            },
            'audio': {
                'sample_rate': int(self.sample_rate.currentText()),
                'buffer_size': self.buffer_size.value(),
                'algorithm': self.loc_algorithm.currentText(),
            },
            'fusion': {
                'max_distance': self.max_distance.value(),
                'temporal_window': self.temporal_window.value(),
            },
            'display': {
                'show_detections': self.show_detections.isChecked(),
                'show_tracks': self.show_tracks.isChecked(),
                'show_pose': self.show_pose.isChecked(),
            },
            'performance': {
                'device': self.device_combo.currentText(),
                'threads': self.worker_threads.value(),
            }
        }
    
    def accept(self):
        self._collect_settings()
        self.settings_changed.emit(self.changes)
        super().accept()
