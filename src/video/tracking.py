"""
Multi-Object Tracking Module using ByteTrack-style tracking.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from loguru import logger
from .detection import Detection


@dataclass
class Track:
    """Represents a tracked object."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    camera_id: str = ""
    state: str = "active"  # active, lost, deleted
    age: int = 0  # frames since creation
    hits: int = 0  # successful matches
    time_since_update: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    features: Optional[np.ndarray] = None
    
    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2])
    
    def predict(self):
        """Predict next position using velocity."""
        self.bbox = self.bbox + self.velocity
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection: Detection):
        """Update track with new detection."""
        old_center = self.center
        self.bbox = detection.bbox.copy()
        self.confidence = detection.confidence
        new_center = self.center
        
        # Update velocity with exponential smoothing
        new_velocity = np.concatenate([new_center - old_center, [0, 0]])
        self.velocity = 0.7 * self.velocity + 0.3 * new_velocity
        
        self.hits += 1
        self.time_since_update = 0
        self.state = "active"
        if detection.features is not None:
            self.features = detection.features


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def linear_assignment(cost_matrix: np.ndarray, threshold: float = 0.5):
    """Simple greedy assignment for cost matrix."""
    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))
    
    # Sort by cost and greedily assign
    indices = np.argsort(cost_matrix.flatten())
    for idx in indices:
        row, col = divmod(idx, cost_matrix.shape[1])
        if row in unmatched_rows and col in unmatched_cols:
            if cost_matrix[row, col] < threshold:
                matches.append((row, col))
                unmatched_rows.discard(row)
                unmatched_cols.discard(col)
    
    return matches, list(unmatched_rows), list(unmatched_cols)


class MultiObjectTracker:
    """ByteTrack-style multi-object tracker."""
    
    def __init__(self, track_buffer: int = 30, high_thresh: float = 0.5,
                 low_thresh: float = 0.1, match_thresh: float = 0.8,
                 min_hits: int = 3):
        self.track_buffer = track_buffer
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.min_hits = min_hits
        
        self.tracks: list[Track] = []
        self.next_id = 1
        self.frame_id = 0
        
        logger.info("MultiObjectTracker initialized")
    
    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracks with new detections."""
        self.frame_id += 1
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]
        
        # Get active and lost tracks
        active_tracks = [t for t in self.tracks if t.state == "active"]
        lost_tracks = [t for t in self.tracks if t.state == "lost"]
        
        # First association: high confidence detections with active tracks
        if active_tracks and high_dets:
            cost = np.zeros((len(active_tracks), len(high_dets)))
            for i, track in enumerate(active_tracks):
                for j, det in enumerate(high_dets):
                    cost[i, j] = 1 - iou(track.bbox, det.bbox)
            
            matches, unmatched_tracks, unmatched_dets = linear_assignment(
                cost, 1 - self.match_thresh
            )
            
            for t_idx, d_idx in matches:
                active_tracks[t_idx].update(high_dets[d_idx])
            
            unmatched_active = [active_tracks[i] for i in unmatched_tracks]
            remaining_high = [high_dets[i] for i in unmatched_dets]
        else:
            unmatched_active = active_tracks
            remaining_high = high_dets
        
        # Second association: remaining with low confidence detections
        if unmatched_active and low_dets:
            cost = np.zeros((len(unmatched_active), len(low_dets)))
            for i, track in enumerate(unmatched_active):
                for j, det in enumerate(low_dets):
                    cost[i, j] = 1 - iou(track.bbox, det.bbox)
            
            matches, still_unmatched, _ = linear_assignment(cost, 1 - self.match_thresh)
            
            for t_idx, d_idx in matches:
                unmatched_active[t_idx].update(low_dets[d_idx])
            
            for t_idx in still_unmatched:
                unmatched_active[t_idx].state = "lost"
        else:
            for track in unmatched_active:
                track.state = "lost"
        
        # Third association: lost tracks with remaining high confidence
        if lost_tracks and remaining_high:
            cost = np.zeros((len(lost_tracks), len(remaining_high)))
            for i, track in enumerate(lost_tracks):
                for j, det in enumerate(remaining_high):
                    cost[i, j] = 1 - iou(track.bbox, det.bbox)
            
            matches, _, unmatched_dets = linear_assignment(cost, 1 - self.match_thresh * 0.8)
            
            for t_idx, d_idx in matches:
                lost_tracks[t_idx].update(remaining_high[d_idx])
            
            remaining_high = [remaining_high[i] for i in unmatched_dets]
        
        # Create new tracks for unmatched high confidence detections
        for det in remaining_high:
            new_track = Track(
                track_id=self.next_id, bbox=det.bbox.copy(),
                confidence=det.confidence, class_id=det.class_id,
                class_name=det.class_name, camera_id=det.camera_id,
                features=det.features
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.track_buffer
        ]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_hits and t.state == "active"]
    
    def get_all_tracks(self) -> list[Track]:
        """Get all tracks including lost ones."""
        return self.tracks.copy()
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_id = 0
