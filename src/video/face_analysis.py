"""
Face Analysis Module for landmarks, head pose, and lip movement detection.
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


@dataclass  
class FaceAnalysisResult:
    """Face analysis result."""
    landmarks: np.ndarray  # Nx3 facial landmarks
    bbox: np.ndarray  # Face bounding box
    head_pose: Optional[np.ndarray] = None  # [pitch, yaw, roll]
    lip_distance: float = 0.0  # Lip opening
    is_speaking: bool = False
    gaze_direction: Optional[np.ndarray] = None


class FaceAnalyzer:
    """MediaPipe-based face analysis for landmarks and lip movement."""
    
    # Key landmark indices
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    NOSE_TIP = 1
    
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 lip_threshold: float = 0.02):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.lip_threshold = lip_threshold
        
        self.face_mesh = None
        if MP_AVAILABLE:
            self.mp_face = mp.solutions.face_mesh
            self.face_mesh = self.mp_face.FaceMesh(
                static_image_mode=False, max_num_faces=4,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            logger.info("FaceAnalyzer initialized")
        else:
            logger.warning("MediaPipe not available")
        
        # Lip movement tracking
        self.prev_lip_distances: dict[int, list[float]] = {}
    
    def analyze(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> list[FaceAnalysisResult]:
        """Analyze faces in image."""
        if self.face_mesh is None:
            return []
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return []
        
        h, w = image.shape[:2]
        face_results = []
        
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = np.array([[lm.x * w, lm.y * h, lm.z] 
                                  for lm in face_landmarks.landmark])
            
            # Compute bbox from landmarks
            xs = landmarks[:, 0]
            ys = landmarks[:, 1]
            face_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            
            # Lip distance (normalized by face height)
            face_height = ys.max() - ys.min()
            upper_lip = landmarks[self.UPPER_LIP]
            lower_lip = landmarks[self.LOWER_LIP]
            lip_dist = np.linalg.norm(upper_lip - lower_lip) / (face_height + 1e-6)
            
            # Detect speaking from lip movement
            is_speaking = self._detect_speaking(face_id, lip_dist)
            
            # Estimate head pose
            head_pose = self._estimate_head_pose(landmarks, w, h)
            
            face_results.append(FaceAnalysisResult(
                landmarks=landmarks, bbox=face_bbox,
                head_pose=head_pose, lip_distance=lip_dist,
                is_speaking=is_speaking
            ))
        
        return face_results
    
    def _detect_speaking(self, face_id: int, lip_dist: float) -> bool:
        """Detect speaking from lip movement variance."""
        if face_id not in self.prev_lip_distances:
            self.prev_lip_distances[face_id] = []
        
        history = self.prev_lip_distances[face_id]
        history.append(lip_dist)
        
        # Keep last 10 frames
        if len(history) > 10:
            history.pop(0)
        
        if len(history) < 5:
            return False
        
        # Speaking detected if variance is high
        variance = np.var(history)
        return variance > self.lip_threshold * 0.1
    
    def _estimate_head_pose(self, landmarks: np.ndarray, w: int, h: int) -> np.ndarray:
        """Estimate head pose (pitch, yaw, roll) from landmarks."""
        # Simplified pose estimation using key points
        nose = landmarks[self.NOSE_TIP][:2]
        left_eye = landmarks[self.LEFT_EYE_OUTER][:2]
        right_eye = landmarks[self.RIGHT_EYE_OUTER][:2]
        
        # Eye center
        eye_center = (left_eye + right_eye) / 2
        
        # Yaw from nose-eye horizontal offset
        face_width = np.linalg.norm(right_eye - left_eye)
        yaw = np.arcsin(np.clip((nose[0] - eye_center[0]) / (face_width * 0.5 + 1e-6), -1, 1))
        
        # Pitch from nose-eye vertical offset
        pitch = np.arcsin(np.clip((nose[1] - eye_center[1]) / (face_width * 0.5 + 1e-6), -1, 1))
        
        # Roll from eye angle
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        
        return np.degrees(np.array([pitch, yaw, roll]))
    
    def close(self):
        if self.face_mesh:
            self.face_mesh.close()
