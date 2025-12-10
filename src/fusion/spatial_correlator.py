"""
Spatial Correlator for audio-visual spatial alignment.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class SpatialCorrelation:
    """Spatial correlation between audio and visual."""
    audio_position: np.ndarray
    visual_position: np.ndarray
    distance: float
    direction_alignment: float  # Cosine similarity
    confidence: float


class SpatialCorrelator:
    """Correlates audio and visual observations spatially."""
    
    def __init__(self, max_distance: float = 3.0):
        self.max_distance = max_distance
        self.reference_position = np.zeros(3)  # Observer/camera position
        logger.info("SpatialCorrelator initialized")
    
    def set_reference(self, position: np.ndarray):
        """Set reference position for direction calculations."""
        self.reference_position = position
    
    def compute_direction(self, position: np.ndarray) -> np.ndarray:
        """Compute direction from reference to position."""
        direction = position - self.reference_position
        norm = np.linalg.norm(direction)
        return direction / (norm + 1e-10)
    
    def correlate(self, audio_pos: np.ndarray, 
                  visual_pos: np.ndarray) -> SpatialCorrelation:
        """Compute spatial correlation between audio and visual positions."""
        distance = np.linalg.norm(audio_pos - visual_pos)
        
        audio_dir = self.compute_direction(audio_pos)
        visual_dir = self.compute_direction(visual_pos)
        direction_alignment = float(np.dot(audio_dir, visual_dir))
        
        # Confidence from distance and alignment
        dist_score = max(0, 1 - distance / self.max_distance)
        align_score = (direction_alignment + 1) / 2  # Map [-1,1] to [0,1]
        confidence = (dist_score + align_score) / 2
        
        return SpatialCorrelation(
            audio_position=audio_pos, visual_position=visual_pos,
            distance=distance, direction_alignment=direction_alignment,
            confidence=confidence
        )
    
    def find_best_match(self, audio_pos: np.ndarray,
                        visual_positions: list[np.ndarray]) -> tuple[int, SpatialCorrelation]:
        """Find best matching visual position for audio source."""
        best_idx = -1
        best_corr = None
        best_conf = -1
        
        for idx, vis_pos in enumerate(visual_positions):
            corr = self.correlate(audio_pos, vis_pos)
            if corr.confidence > best_conf:
                best_conf = corr.confidence
                best_corr = corr
                best_idx = idx
        
        return best_idx, best_corr
