"""
Recording Manager - Record and playback sessions.
"""

import os
import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QProgressBar, QFileDialog, QListWidget,
    QListWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class RecordingFrame:
    timestamp: float
    camera_id: str
    detections: list
    tracks: list
    audio_level: float
    localization: Optional[tuple]


class RecordingSession:
    """Manages a recording session."""
    
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.session_id: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.is_recording = False
        self.start_time = 0
        self.frame_count = 0
        
        # Writers
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.metadata_file = None
        self.frames_data = []
    
    def start(self) -> bool:
        """Start new recording session."""
        if self.is_recording:
            return False
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0
        self.frames_data = []
        
        logger.info(f"Recording started: {self.session_id}")
        return True
    
    def stop(self) -> Optional[str]:
        """Stop recording and save session."""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        # Close video writers
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
        
        # Save metadata
        metadata = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "duration": time.time() - self.start_time,
            "frame_count": self.frame_count,
            "frames": self.frames_data
        }
        
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Recording saved: {self.session_dir}")
        return str(self.session_dir)
    
    def add_frame(self, camera_id: str, frame: np.ndarray,
                  detections: list = None, tracks: list = None,
                  audio_level: float = 0, localization: tuple = None):
        """Add a frame to the recording."""
        if not self.is_recording or not CV2_AVAILABLE:
            return
        
        # Initialize video writer if needed
        if camera_id not in self.video_writers:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            path = str(self.session_dir / f"{camera_id}.mp4")
            self.video_writers[camera_id] = cv2.VideoWriter(path, fourcc, 30, (w, h))
        
        # Write video frame
        self.video_writers[camera_id].write(frame)
        
        # Store frame metadata
        frame_data = {
            "timestamp": time.time() - self.start_time,
            "camera_id": camera_id,
            "detections": detections or [],
            "tracks": tracks or [],
            "audio_level": audio_level,
            "localization": list(localization) if localization else None
        }
        self.frames_data.append(frame_data)
        self.frame_count += 1
    
    @property
    def duration(self) -> float:
        if self.is_recording:
            return time.time() - self.start_time
        return 0
    
    @staticmethod
    def list_recordings(output_dir: str = "recordings") -> list[dict]:
        """List available recordings."""
        recordings = []
        rec_dir = Path(output_dir)
        
        if not rec_dir.exists():
            return recordings
        
        for session_dir in rec_dir.iterdir():
            if session_dir.is_dir():
                meta_path = session_dir / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        recordings.append({
                            "path": str(session_dir),
                            "session_id": meta.get("session_id", session_dir.name),
                            "duration": meta.get("duration", 0),
                            "frame_count": meta.get("frame_count", 0),
                            "start_time": meta.get("start_time", 0)
                        })
                    except:
                        pass
        
        return sorted(recordings, key=lambda x: x.get("start_time", 0), reverse=True)


class RecordingWidget(QFrame):
    """Widget for recording controls and playback."""
    
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal(str)  # path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("recording-widget")
        
        self.session = RecordingSession()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("🎬 Recording")
        title.setStyleSheet("font-size: 13px; font-weight: 500; color: #ff6b9d;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)
        
        # Controls
        controls = QHBoxLayout()
        
        self.record_btn = QPushButton("⏺ Record")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 80, 80, 0.3);
                border: 1px solid rgba(255, 80, 80, 0.5);
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: rgba(255, 80, 80, 0.5);
            }
        """)
        self.record_btn.clicked.connect(self._toggle_recording)
        controls.addWidget(self.record_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_recording)
        controls.addWidget(self.stop_btn)
        
        layout.addLayout(controls)
        
        # Status
        status_layout = QHBoxLayout()
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #666; font-size: 16px;")
        status_layout.addWidget(self.status_indicator)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.duration_label = QLabel("00:00")
        self.duration_label.setStyleSheet("color: #888; font-family: monospace;")
        status_layout.addWidget(self.duration_label)
        
        layout.addLayout(status_layout)
        
        # Progress bar (for playback)
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(4)
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        
        # Recent recordings
        rec_label = QLabel("Recent Recordings")
        rec_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 8px;")
        layout.addWidget(rec_label)
        
        self.recordings_list = QListWidget()
        self.recordings_list.setMaximumHeight(100)
        self.recordings_list.setStyleSheet("""
            QListWidget {
                background: rgba(0, 0, 0, 0.2);
                border: none;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px;
            }
        """)
        layout.addWidget(self.recordings_list)
        
        # Duration timer
        self.duration_timer = QTimer()
        self.duration_timer.timeout.connect(self._update_duration)
        
        # Load recordings
        self._refresh_recordings()
    
    def _toggle_recording(self):
        if not self.session.is_recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _start_recording(self):
        if self.session.start():
            self.record_btn.setText("⏺ Recording...")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 80, 80, 0.6);
                    border: 1px solid rgba(255, 80, 80, 0.8);
                    padding: 8px 16px;
                }
            """)
            self.stop_btn.setEnabled(True)
            self.status_indicator.setStyleSheet("color: #ff4444; font-size: 16px;")
            self.status_label.setText("Recording")
            self.duration_timer.start(1000)
            self.recording_started.emit()
    
    def _stop_recording(self):
        path = self.session.stop()
        if path:
            self.record_btn.setText("⏺ Record")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 80, 80, 0.3);
                    border: 1px solid rgba(255, 80, 80, 0.5);
                    padding: 8px 16px;
                }
            """)
            self.stop_btn.setEnabled(False)
            self.status_indicator.setStyleSheet("color: #00ff88; font-size: 16px;")
            self.status_label.setText("Saved")
            self.duration_timer.stop()
            self._refresh_recordings()
            self.recording_stopped.emit(path)
    
    def _update_duration(self):
        duration = int(self.session.duration)
        mins, secs = divmod(duration, 60)
        self.duration_label.setText(f"{mins:02d}:{secs:02d}")
    
    def _refresh_recordings(self):
        self.recordings_list.clear()
        for rec in RecordingSession.list_recordings()[:5]:
            duration = int(rec["duration"])
            mins, secs = divmod(duration, 60)
            item = QListWidgetItem(
                f"{rec['session_id']} ({mins}:{secs:02d})"
            )
            self.recordings_list.addItem(item)
    
    def add_frame(self, camera_id: str, frame: np.ndarray, **kwargs):
        """Add frame to current recording."""
        if self.session.is_recording:
            self.session.add_frame(camera_id, frame, **kwargs)
    
    @property
    def is_recording(self) -> bool:
        return self.session.is_recording
