"""
Controls Panel Widget for system controls and status.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QCheckBox, QSlider, QProgressBar,
    QGroupBox, QGridLayout, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlsPanel(QFrame):
    """Panel for system controls and configuration."""
    
    # Signals
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    calibrate_clicked = pyqtSignal()
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controls-panel")
        self.setMaximumWidth(300)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Header
        title = QLabel("⚙️ Controls")
        title.setObjectName("panel-title")
        layout.addWidget(title)
        
        # System controls
        control_group = QGroupBox("System")
        control_layout = QVBoxLayout(control_group)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_clicked.emit)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        control_layout.addLayout(btn_layout)
        
        self.calibrate_btn = QPushButton("📐 Calibrate")
        self.calibrate_btn.clicked.connect(self.calibrate_clicked.emit)
        control_layout.addWidget(self.calibrate_btn)
        
        layout.addWidget(control_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QGridLayout(status_group)
        
        status_layout.addWidget(QLabel("State:"), 0, 0)
        self.state_label = QLabel("Stopped")
        self.state_label.setStyleSheet("color: #ff6464;")
        status_layout.addWidget(self.state_label, 0, 1)
        
        status_layout.addWidget(QLabel("FPS:"), 1, 0)
        self.fps_label = QLabel("0")
        self.fps_label.setStyleSheet("color: #00d4aa;")
        status_layout.addWidget(self.fps_label, 1, 1)
        
        status_layout.addWidget(QLabel("Latency:"), 2, 0)
        self.latency_label = QLabel("-- ms")
        self.latency_label.setStyleSheet("color: #88aaff;")
        status_layout.addWidget(self.latency_label, 2, 1)
        
        status_layout.addWidget(QLabel("CPU:"), 3, 0)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setMaximumHeight(12)
        self.cpu_bar.setValue(0)
        status_layout.addWidget(self.cpu_bar, 3, 1)
        
        status_layout.addWidget(QLabel("GPU:"), 4, 0)
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setMaximumHeight(12)
        self.gpu_bar.setValue(0)
        status_layout.addWidget(self.gpu_bar, 4, 1)
        
        layout.addWidget(status_group)
        
        # Detection settings
        det_group = QGroupBox("Detection")
        det_layout = QGridLayout(det_group)
        
        det_layout.addWidget(QLabel("Confidence:"), 0, 0)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self._on_settings_change)
        det_layout.addWidget(self.conf_slider, 0, 1)
        self.conf_label = QLabel("50%")
        det_layout.addWidget(self.conf_label, 0, 2)
        
        self.show_detections = QCheckBox("Show Detections")
        self.show_detections.setChecked(True)
        self.show_detections.stateChanged.connect(self._on_settings_change)
        det_layout.addWidget(self.show_detections, 1, 0, 1, 3)
        
        self.show_tracks = QCheckBox("Show Tracks")
        self.show_tracks.setChecked(True)
        self.show_tracks.stateChanged.connect(self._on_settings_change)
        det_layout.addWidget(self.show_tracks, 2, 0, 1, 3)
        
        self.show_pose = QCheckBox("Show Pose")
        self.show_pose.setChecked(False)
        self.show_pose.stateChanged.connect(self._on_settings_change)
        det_layout.addWidget(self.show_pose, 3, 0, 1, 3)
        
        layout.addWidget(det_group)
        
        # Audio settings
        audio_group = QGroupBox("Audio")
        audio_layout = QGridLayout(audio_group)
        
        audio_layout.addWidget(QLabel("Gain:"), 0, 0)
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 100)
        self.gain_slider.setValue(50)
        audio_layout.addWidget(self.gain_slider, 0, 1)
        
        self.enable_localization = QCheckBox("Enable Localization")
        self.enable_localization.setChecked(True)
        audio_layout.addWidget(self.enable_localization, 1, 0, 1, 2)
        
        self.enable_events = QCheckBox("Enable Event Detection")
        self.enable_events.setChecked(True)
        audio_layout.addWidget(self.enable_events, 2, 0, 1, 2)
        
        layout.addWidget(audio_group)
        
        # Fusion settings
        fusion_group = QGroupBox("Fusion")
        fusion_layout = QGridLayout(fusion_group)
        
        fusion_layout.addWidget(QLabel("Max Distance:"), 0, 0)
        self.max_dist_spin = QSpinBox()
        self.max_dist_spin.setRange(1, 10)
        self.max_dist_spin.setValue(3)
        self.max_dist_spin.setSuffix(" m")
        fusion_layout.addWidget(self.max_dist_spin, 0, 1)
        
        self.enable_fusion = QCheckBox("Enable Fusion")
        self.enable_fusion.setChecked(True)
        fusion_layout.addWidget(self.enable_fusion, 1, 0, 1, 2)
        
        layout.addWidget(fusion_group)
        
        layout.addStretch()
    
    def _on_settings_change(self):
        """Emit settings changed signal."""
        self.conf_label.setText(f"{self.conf_slider.value()}%")
        
        settings = {
            'confidence': self.conf_slider.value() / 100,
            'show_detections': self.show_detections.isChecked(),
            'show_tracks': self.show_tracks.isChecked(),
            'show_pose': self.show_pose.isChecked(),
            'enable_localization': self.enable_localization.isChecked(),
            'enable_events': self.enable_events.isChecked(),
            'enable_fusion': self.enable_fusion.isChecked(),
            'max_distance': self.max_dist_spin.value()
        }
        self.settings_changed.emit(settings)
    
    def set_running(self, running: bool):
        """Update UI for running state."""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.calibrate_btn.setEnabled(not running)
        
        if running:
            self.state_label.setText("Running")
            self.state_label.setStyleSheet("color: #00ff88;")
        else:
            self.state_label.setText("Stopped")
            self.state_label.setStyleSheet("color: #ff6464;")
    
    def update_stats(self, fps: float, latency: float, cpu: float, gpu: float):
        """Update performance stats."""
        self.fps_label.setText(f"{fps:.1f}")
        self.latency_label.setText(f"{latency:.0f} ms")
        self.cpu_bar.setValue(int(cpu))
        self.gpu_bar.setValue(int(gpu))
