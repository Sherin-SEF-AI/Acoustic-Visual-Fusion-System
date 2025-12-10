"""
Export Manager - Export session data to various formats.
"""

import json
import csv
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
from loguru import logger

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False


@dataclass
class ExportSession:
    """Represents an export session."""
    session_id: str
    start_time: float
    end_time: float
    frames_exported: int
    tracks_exported: int
    events_exported: int
    output_path: str


class ExportManager:
    """
    Manages data export to various formats.
    """
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session data buffers
        self.detections: list[dict] = []
        self.tracks: list[dict] = []
        self.localizations: list[dict] = []
        self.events: list[dict] = []
        self.fusions: list[dict] = []
        
        self.session_start = 0
        self.is_recording = False
        
        logger.info(f"ExportManager initialized: {self.output_dir}")
    
    def start_session(self):
        """Start a new export session."""
        self.session_start = time.time()
        self.is_recording = True
        
        self.detections.clear()
        self.tracks.clear()
        self.localizations.clear()
        self.events.clear()
        self.fusions.clear()
    
    def stop_session(self):
        """Stop the export session."""
        self.is_recording = False
    
    def add_detection(self, camera_id: str, timestamp: float,
                      bbox: list, class_name: str, confidence: float):
        """Record a detection."""
        if not self.is_recording:
            return
        
        self.detections.append({
            "timestamp": timestamp,
            "camera_id": camera_id,
            "bbox": bbox,
            "class": class_name,
            "confidence": confidence
        })
    
    def add_track(self, track_id: int, timestamp: float,
                  position: list, class_name: str, is_speaking: bool):
        """Record a track update."""
        if not self.is_recording:
            return
        
        self.tracks.append({
            "timestamp": timestamp,
            "track_id": track_id,
            "position": position,
            "class": class_name,
            "is_speaking": is_speaking
        })
    
    def add_localization(self, timestamp: float, position: list,
                         confidence: float, direction: float = 0):
        """Record a sound localization."""
        if not self.is_recording:
            return
        
        self.localizations.append({
            "timestamp": timestamp,
            "position": position,
            "confidence": confidence,
            "direction": direction
        })
    
    def add_event(self, timestamp: float, event_type: str,
                  source: str, confidence: float, metadata: dict = None):
        """Record an audio event."""
        if not self.is_recording:
            return
        
        self.events.append({
            "timestamp": timestamp,
            "event_type": event_type,
            "source": source,
            "confidence": confidence,
            "metadata": metadata or {}
        })
    
    def add_fusion(self, timestamp: float, track_id: int,
                   sound_position: list, confidence: float):
        """Record a fusion result."""
        if not self.is_recording:
            return
        
        self.fusions.append({
            "timestamp": timestamp,
            "track_id": track_id,
            "sound_position": sound_position,
            "confidence": confidence
        })
    
    def export_json(self, filename: str = None) -> str:
        """Export all data to JSON."""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        
        data = {
            "session": {
                "start_time": self.session_start,
                "end_time": time.time(),
                "duration": time.time() - self.session_start
            },
            "summary": {
                "detections": len(self.detections),
                "tracks": len(self.tracks),
                "localizations": len(self.localizations),
                "events": len(self.events),
                "fusions": len(self.fusions)
            },
            "detections": self.detections,
            "tracks": self.tracks,
            "localizations": self.localizations,
            "events": self.events,
            "fusions": self.fusions
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported JSON: {output_path}")
        return str(output_path)
    
    def export_csv(self, data_type: str = "all") -> dict[str, str]:
        """Export data to CSV files."""
        paths = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if data_type in ["all", "detections"] and self.detections:
            path = self.output_dir / f"detections_{timestamp}.csv"
            self._write_csv(path, self.detections)
            paths["detections"] = str(path)
        
        if data_type in ["all", "tracks"] and self.tracks:
            path = self.output_dir / f"tracks_{timestamp}.csv"
            self._write_csv(path, self.tracks)
            paths["tracks"] = str(path)
        
        if data_type in ["all", "localizations"] and self.localizations:
            path = self.output_dir / f"localizations_{timestamp}.csv"
            self._write_csv(path, self.localizations)
            paths["localizations"] = str(path)
        
        if data_type in ["all", "events"] and self.events:
            path = self.output_dir / f"events_{timestamp}.csv"
            self._write_csv(path, self.events)
            paths["events"] = str(path)
        
        logger.info(f"Exported CSV files: {list(paths.keys())}")
        return paths
    
    def _write_csv(self, path: Path, data: list[dict]):
        """Write data to CSV file."""
        if not data:
            return
        
        # Flatten nested dicts
        flat_data = []
        for item in data:
            flat = {}
            for key, value in item.items():
                if isinstance(value, (list, tuple)):
                    flat[key] = str(value)
                elif isinstance(value, dict):
                    for k, v in value.items():
                        flat[f"{key}_{k}"] = v
                else:
                    flat[key] = value
            flat_data.append(flat)
        
        # Get all keys
        keys = set()
        for item in flat_data:
            keys.update(item.keys())
        keys = sorted(keys)
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_data)
    
    def export_numpy(self) -> dict[str, str]:
        """Export numerical data to NumPy format."""
        if not NP_AVAILABLE:
            logger.warning("NumPy not available for export")
            return {}
        
        paths = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Track positions
        if self.localizations:
            positions = np.array([
                loc["position"] for loc in self.localizations
            ])
            timestamps = np.array([
                loc["timestamp"] for loc in self.localizations
            ])
            confidences = np.array([
                loc["confidence"] for loc in self.localizations
            ])
            
            path = self.output_dir / f"localizations_{timestamp}.npz"
            np.savez(path, 
                     positions=positions,
                     timestamps=timestamps,
                     confidences=confidences)
            paths["localizations"] = str(path)
        
        logger.info(f"Exported NumPy files: {list(paths.keys())}")
        return paths
    
    def get_statistics(self) -> dict:
        """Get statistics about recorded data."""
        duration = time.time() - self.session_start if self.session_start > 0 else 0
        
        return {
            "duration_seconds": duration,
            "detections_count": len(self.detections),
            "tracks_count": len(self.tracks),
            "localizations_count": len(self.localizations),
            "events_count": len(self.events),
            "fusions_count": len(self.fusions),
            "detections_per_second": len(self.detections) / max(duration, 1),
            "events_per_minute": len(self.events) / max(duration / 60, 1)
        }
