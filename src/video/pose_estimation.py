"""
Pose Estimation Module using MediaPipe.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2
from loguru import logger

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logger.warning("MediaPipe not available")


@dataclass
class PoseResult:
    """Pose estimation result."""
    landmarks: np.ndarray  # Nx3 normalized landmarks
    world_landmarks: Optional[np.ndarray] = None  # Nx3 world coordinates
    visibility: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    
    @property
    def num_keypoints(self) -> int:
        return len(self.landmarks)
    
    def get_keypoint(self, idx: int) -> np.ndarray:
        return self.landmarks[idx]


class PoseEstimator:
    """MediaPipe-based pose estimation."""
    
    KEYPOINT_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    def __init__(self, static_mode: bool = False, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.static_mode = static_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.pose = None
        if MP_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=static_mode,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            logger.info("PoseEstimator initialized with MediaPipe")
        else:
            logger.warning("MediaPipe not available, pose estimation disabled")
    
    def estimate(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> Optional[PoseResult]:
        """Estimate pose from image or cropped region."""
        if self.pose is None:
            return None
        
        # Crop to bbox if provided
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            crop = image[y1:y2, x1:x2]
        else:
            crop = image
            x1, y1 = 0, 0
        
        if crop.size == 0:
            return None
        
        # Convert to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks is None:
            return None
        
        # Extract landmarks
        h, w = crop.shape[:2]
        landmarks = np.array([[lm.x * w + x1, lm.y * h + y1, lm.z] 
                              for lm in results.pose_landmarks.landmark])
        visibility = np.array([lm.visibility for lm in results.pose_landmarks.landmark])
        
        world_landmarks = None
        if results.pose_world_landmarks:
            world_landmarks = np.array([[lm.x, lm.y, lm.z] 
                                        for lm in results.pose_world_landmarks.landmark])
        
        return PoseResult(
            landmarks=landmarks, world_landmarks=world_landmarks,
            visibility=visibility, bbox=bbox
        )
    
    def draw_pose(self, image: np.ndarray, pose: PoseResult,
                  color: tuple = (0, 255, 0), thickness: int = 2):
        """Draw pose landmarks on image."""
        if MP_AVAILABLE:
            connections = self.mp_pose.POSE_CONNECTIONS
            for start, end in connections:
                if (pose.visibility[start] > 0.5 and pose.visibility[end] > 0.5):
                    pt1 = tuple(map(int, pose.landmarks[start][:2]))
                    pt2 = tuple(map(int, pose.landmarks[end][:2]))
                    cv2.line(image, pt1, pt2, color, thickness)
            
            for i, (x, y, _) in enumerate(pose.landmarks):
                if pose.visibility[i] > 0.5:
                    cv2.circle(image, (int(x), int(y)), 4, color, -1)
    
    def close(self):
        if self.pose:
            self.pose.close()
