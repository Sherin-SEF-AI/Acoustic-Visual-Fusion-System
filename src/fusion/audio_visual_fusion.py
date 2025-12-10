"""
Audio-Visual Fusion Module for correlating sound sources with visual tracks.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from loguru import logger


@dataclass
class FusionResult:
    """Result of audio-visual fusion."""
    audio_position: np.ndarray  # 3D sound source position
    visual_track_id: Optional[int]  # Associated visual track ID
    confidence: float  # Association confidence
    temporal_score: float  # Temporal alignment score
    spatial_score: float  # Spatial proximity score
    speaking_score: float  # Visual speaking indicator score
    timestamp: float = 0.0


class AudioVisualFusion:
    """
    Fuses audio localization with visual tracking for speaker identification.
    
    Uses spatial proximity, temporal alignment, and visual speaking cues
    to associate sound sources with visual tracks.
    """
    
    def __init__(self, max_distance: float = 2.0, temporal_window: float = 0.5,
                 spatial_weight: float = 0.4, temporal_weight: float = 0.3,
                 speaking_weight: float = 0.3):
        self.max_distance = max_distance
        self.temporal_window = temporal_window
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.speaking_weight = speaking_weight
        
        # Camera projection matrices (set during calibration)
        self.camera_matrices: dict[str, np.ndarray] = {}
        self.camera_positions: dict[str, np.ndarray] = {}
        
        logger.info("AudioVisualFusion initialized")
    
    def set_camera_calibration(self, camera_id: str, intrinsic: np.ndarray,
                                extrinsic: np.ndarray, position: np.ndarray):
        """Set camera calibration for 3D-2D projection."""
        projection = intrinsic @ extrinsic[:3]
        self.camera_matrices[camera_id] = projection
        self.camera_positions[camera_id] = position
    
    def project_3d_to_camera(self, point_3d: np.ndarray, camera_id: str) -> Optional[np.ndarray]:
        """Project 3D point to camera image coordinates."""
        if camera_id not in self.camera_matrices:
            return None
        
        P = self.camera_matrices[camera_id]
        point_h = np.append(point_3d, 1)
        projected = P @ point_h
        
        if projected[2] <= 0:
            return None  # Behind camera
        
        return projected[:2] / projected[2]
    
    def estimate_track_3d_position(self, track_bbox: np.ndarray, camera_id: str,
                                   depth: float = 2.0) -> Optional[np.ndarray]:
        """Estimate 3D position of track from 2D bbox and estimated depth."""
        if camera_id not in self.camera_positions:
            return None
        
        # Simple estimation: place at camera position + depth along view direction
        camera_pos = self.camera_positions[camera_id]
        # Simplified: assume track is at given depth in front of camera
        center_2d = np.array([(track_bbox[0]+track_bbox[2])/2, 
                              (track_bbox[1]+track_bbox[3])/2])
        
        # For proper estimation, we'd need inverse projection
        # Here we use a simplified forward estimation
        return camera_pos + np.array([0, 0, depth])
    
    def compute_spatial_score(self, audio_pos: np.ndarray, 
                               track_pos: np.ndarray) -> float:
        """Compute spatial proximity score."""
        distance = np.linalg.norm(audio_pos - track_pos)
        if distance > self.max_distance:
            return 0.0
        return 1.0 - (distance / self.max_distance)
    
    def compute_temporal_score(self, audio_timestamp: float,
                                track_timestamp: float) -> float:
        """Compute temporal alignment score."""
        time_diff = abs(audio_timestamp - track_timestamp)
        if time_diff > self.temporal_window:
            return 0.0
        return 1.0 - (time_diff / self.temporal_window)
    
    def fuse(self, audio_position: np.ndarray, audio_timestamp: float,
             visual_tracks: list[dict]) -> FusionResult:
        """
        Fuse audio localization with visual tracks.
        
        Args:
            audio_position: 3D localized sound position
            audio_timestamp: Audio event timestamp
            visual_tracks: List of dicts with keys:
                - track_id: int
                - position_3d: np.ndarray (estimated 3D position)
                - timestamp: float
                - is_speaking: bool (from lip detection)
                - camera_id: str
        
        Returns:
            FusionResult with best matching track
        """
        if not visual_tracks:
            return FusionResult(
                audio_position=audio_position, visual_track_id=None,
                confidence=0.0, temporal_score=0.0, spatial_score=0.0,
                speaking_score=0.0, timestamp=audio_timestamp
            )
        
        best_score = -1
        best_result = None
        
        for track in visual_tracks:
            track_pos = track.get('position_3d')
            if track_pos is None:
                continue
            
            spatial = self.compute_spatial_score(audio_position, track_pos)
            temporal = self.compute_temporal_score(audio_timestamp, track.get('timestamp', 0))
            speaking = 1.0 if track.get('is_speaking', False) else 0.0
            
            # Weighted combination
            score = (self.spatial_weight * spatial +
                    self.temporal_weight * temporal +
                    self.speaking_weight * speaking)
            
            if score > best_score:
                best_score = score
                best_result = FusionResult(
                    audio_position=audio_position,
                    visual_track_id=track.get('track_id'),
                    confidence=score,
                    temporal_score=temporal,
                    spatial_score=spatial,
                    speaking_score=speaking,
                    timestamp=audio_timestamp
                )
        
        return best_result or FusionResult(
            audio_position=audio_position, visual_track_id=None,
            confidence=0.0, temporal_score=0.0, spatial_score=0.0,
            speaking_score=0.0, timestamp=audio_timestamp
        )
    
    def fuse_batch(self, audio_events: list[dict], 
                   visual_tracks: list[dict]) -> list[FusionResult]:
        """Fuse multiple audio events with visual tracks."""
        results = []
        for event in audio_events:
            result = self.fuse(
                event['position'], event['timestamp'], visual_tracks
            )
            results.append(result)
        return results
