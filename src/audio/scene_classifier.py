"""
Acoustic Scene Classification - Environment type identification.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
from loguru import logger


class SceneType(Enum):
    """Acoustic scene types."""
    QUIET_OFFICE = "quiet_office"
    ACTIVE_MEETING = "active_meeting"
    CROWDED_SPACE = "crowded_space"
    INDUSTRIAL = "industrial"
    OUTDOOR = "outdoor"
    CAFETERIA = "cafeteria"
    HALLWAY = "hallway"
    EMPTY_ROOM = "empty_room"


@dataclass
class SceneClassification:
    """Result of scene classification."""
    scene_type: SceneType
    confidence: float
    secondary_type: Optional[SceneType]
    features: Dict[str, float]
    recommended_thresholds: Dict[str, float]


class AcousticSceneClassifier:
    """
    Classifies the acoustic environment type.
    
    Adjusts detection thresholds based on scene type.
    """
    
    # Feature thresholds for each scene type
    SCENE_PROFILES = {
        SceneType.QUIET_OFFICE: {
            "avg_energy": (0.001, 0.05),
            "speech_ratio": (0.1, 0.4),
            "spectral_flatness": (0.2, 0.5),
            "event_rate": (0.0, 0.1)
        },
        SceneType.ACTIVE_MEETING: {
            "avg_energy": (0.05, 0.2),
            "speech_ratio": (0.4, 0.8),
            "spectral_flatness": (0.3, 0.6),
            "event_rate": (0.1, 0.4)
        },
        SceneType.CROWDED_SPACE: {
            "avg_energy": (0.15, 0.5),
            "speech_ratio": (0.5, 0.95),
            "spectral_flatness": (0.4, 0.7),
            "event_rate": (0.3, 0.8)
        },
        SceneType.INDUSTRIAL: {
            "avg_energy": (0.3, 1.0),
            "speech_ratio": (0.0, 0.3),
            "spectral_flatness": (0.5, 0.9),
            "event_rate": (0.2, 0.6)
        },
        SceneType.EMPTY_ROOM: {
            "avg_energy": (0.0, 0.01),
            "speech_ratio": (0.0, 0.1),
            "spectral_flatness": (0.1, 0.3),
            "event_rate": (0.0, 0.05)
        }
    }
    
    # Recommended thresholds per scene
    THRESHOLD_ADJUSTMENTS = {
        SceneType.QUIET_OFFICE: {"speech": 0.02, "event": 0.03, "alert": 0.1},
        SceneType.ACTIVE_MEETING: {"speech": 0.05, "event": 0.08, "alert": 0.2},
        SceneType.CROWDED_SPACE: {"speech": 0.15, "event": 0.2, "alert": 0.4},
        SceneType.INDUSTRIAL: {"speech": 0.3, "event": 0.25, "alert": 0.5},
        SceneType.EMPTY_ROOM: {"speech": 0.01, "event": 0.02, "alert": 0.05}
    }
    
    def __init__(self, classification_window: int = 50):
        self.classification_window = classification_window
        
        # Feature history
        self._feature_history: deque = deque(maxlen=classification_window)
        
        # Current classification
        self._current_scene: SceneType = SceneType.QUIET_OFFICE
        self._scene_confidence: float = 0.5
        self._scene_history: deque = deque(maxlen=20)
        
        logger.info("AcousticSceneClassifier initialized")
    
    def update(self,
               energy: float,
               speech_detected: bool,
               spectral_flatness: float = 0.5,
               num_events: int = 0,
               timestamp: Optional[float] = None) -> SceneClassification:
        """
        Update scene classification with new features.
        
        Returns:
            Current scene classification
        """
        timestamp = timestamp or time.time()
        
        # Store features
        self._feature_history.append({
            "timestamp": timestamp,
            "energy": energy,
            "speech": 1.0 if speech_detected else 0.0,
            "spectral_flatness": spectral_flatness,
            "num_events": num_events
        })
        
        # Classify if enough history
        if len(self._feature_history) >= 10:
            self._classify()
        
        return self.get_classification()
    
    def _classify(self):
        """Perform scene classification."""
        recent = list(self._feature_history)
        
        # Calculate aggregate features
        features = {
            "avg_energy": np.mean([f["energy"] for f in recent]),
            "speech_ratio": np.mean([f["speech"] for f in recent]),
            "spectral_flatness": np.mean([f["spectral_flatness"] for f in recent]),
            "event_rate": np.sum([f["num_events"] for f in recent]) / len(recent)
        }
        
        # Score each scene type
        scores = {}
        for scene_type, profile in self.SCENE_PROFILES.items():
            score = self._calculate_match_score(features, profile)
            scores[scene_type] = score
        
        # Select best match
        best_scene = max(scores, key=scores.get)
        best_score = scores[best_scene]
        
        # Get second best for secondary classification
        sorted_scenes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary = sorted_scenes[1][0] if len(sorted_scenes) > 1 else None
        
        # Apply temporal smoothing
        self._scene_history.append(best_scene)
        
        # Use mode of recent history for stable classification
        from collections import Counter
        scene_counts = Counter(self._scene_history)
        stable_scene = scene_counts.most_common(1)[0][0]
        
        self._current_scene = stable_scene
        self._scene_confidence = best_score
    
    def _calculate_match_score(self, features: dict, profile: dict) -> float:
        """Calculate how well features match a scene profile."""
        total_score = 0.0
        num_features = 0
        
        for feature_name, (min_val, max_val) in profile.items():
            if feature_name in features:
                value = features[feature_name]
                
                if min_val <= value <= max_val:
                    # Within range - high score
                    range_center = (min_val + max_val) / 2
                    range_width = max_val - min_val
                    distance = abs(value - range_center) / (range_width / 2)
                    score = 1.0 - (distance * 0.5)
                else:
                    # Outside range - penalize
                    if value < min_val:
                        distance = min_val - value
                    else:
                        distance = value - max_val
                    score = max(0, 0.5 - distance)
                
                total_score += score
                num_features += 1
        
        return total_score / num_features if num_features > 0 else 0
    
    def get_classification(self) -> SceneClassification:
        """Get current scene classification."""
        recent = list(self._feature_history)
        
        features = {}
        if recent:
            features = {
                "avg_energy": np.mean([f["energy"] for f in recent]),
                "speech_ratio": np.mean([f["speech"] for f in recent]),
                "spectral_flatness": np.mean([f["spectral_flatness"] for f in recent]),
                "event_rate": np.sum([f["num_events"] for f in recent]) / max(len(recent), 1)
            }
        
        thresholds = self.THRESHOLD_ADJUSTMENTS.get(
            self._current_scene,
            {"speech": 0.05, "event": 0.1, "alert": 0.2}
        )
        
        return SceneClassification(
            scene_type=self._current_scene,
            confidence=self._scene_confidence,
            secondary_type=None,
            features=features,
            recommended_thresholds=thresholds
        )
    
    def get_adjusted_threshold(self, threshold_type: str) -> float:
        """Get adjusted threshold for current scene."""
        thresholds = self.THRESHOLD_ADJUSTMENTS.get(
            self._current_scene,
            {"speech": 0.05, "event": 0.1, "alert": 0.2}
        )
        return thresholds.get(threshold_type, 0.1)
    
    @property
    def current_scene(self) -> SceneType:
        return self._current_scene
    
    @property
    def scene_name(self) -> str:
        return self._current_scene.value.replace("_", " ").title()


class AcousticFingerprint:
    """
    Learn and track the normal acoustic signature of a space.
    
    Detects deviations from learned baseline.
    """
    
    def __init__(self, learning_period: int = 1000):
        self.learning_period = learning_period
        self._learning = True
        self._samples_collected = 0
        
        # Baseline statistics
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._feature_buffer: List[np.ndarray] = []
        
        # Temporal patterns
        self._hourly_patterns: Dict[int, Dict] = {}
        
        # Anomaly tracking
        self._deviation_history: deque = deque(maxlen=100)
        
        logger.info("AcousticFingerprint initialized")
    
    def update(self, features: np.ndarray, 
               timestamp: Optional[float] = None) -> Tuple[float, bool]:
        """
        Update fingerprint with new features.
        
        Returns:
            (deviation_score, is_anomaly)
        """
        timestamp = timestamp or time.time()
        hour = int((timestamp % 86400) / 3600)  # Hour of day
        
        if self._learning:
            self._learn(features, hour)
            return 0.0, False
        
        # Calculate deviation
        deviation = self._calculate_deviation(features)
        is_anomaly = deviation > 2.5  # Threshold in standard deviations
        
        self._deviation_history.append((timestamp, deviation))
        
        # Continue slow learning
        self._slow_adapt(features)
        
        return deviation, is_anomaly
    
    def _learn(self, features: np.ndarray, hour: int):
        """Learn baseline during learning period."""
        self._feature_buffer.append(features)
        self._samples_collected += 1
        
        # Store hourly pattern
        if hour not in self._hourly_patterns:
            self._hourly_patterns[hour] = {"samples": [], "mean": None}
        self._hourly_patterns[hour]["samples"].append(features)
        
        if self._samples_collected >= self.learning_period:
            self._finalize_learning()
    
    def _finalize_learning(self):
        """Finalize baseline learning."""
        all_features = np.array(self._feature_buffer)
        self._baseline_mean = np.mean(all_features, axis=0)
        self._baseline_std = np.std(all_features, axis=0)
        self._baseline_std = np.maximum(self._baseline_std, 1e-6)
        
        # Compute hourly patterns
        for hour, data in self._hourly_patterns.items():
            if data["samples"]:
                data["mean"] = np.mean(data["samples"], axis=0)
        
        self._learning = False
        self._feature_buffer.clear()
        
        logger.info("Acoustic fingerprint learning complete")
    
    def _calculate_deviation(self, features: np.ndarray) -> float:
        """Calculate deviation from baseline."""
        if self._baseline_mean is None:
            return 0.0
        
        z_scores = (features - self._baseline_mean) / self._baseline_std
        deviation = np.sqrt(np.mean(z_scores ** 2))
        
        return float(deviation)
    
    def _slow_adapt(self, features: np.ndarray, alpha: float = 0.001):
        """Slowly adapt baseline to account for gradual changes."""
        if self._baseline_mean is not None:
            self._baseline_mean = (1 - alpha) * self._baseline_mean + alpha * features
    
    def get_recent_deviations(self) -> List[Tuple[float, float]]:
        """Get recent deviation history."""
        return list(self._deviation_history)
    
    @property
    def is_learning(self) -> bool:
        return self._learning
    
    @property
    def learning_progress(self) -> float:
        if not self._learning:
            return 1.0
        return self._samples_collected / self.learning_period
