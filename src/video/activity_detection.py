"""
Activity Detection Module for speaking and gesture recognition.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from loguru import logger

from .pose_estimation import PoseResult
from .face_analysis import FaceAnalysisResult


@dataclass
class ActivityResult:
    """Activity detection result."""
    is_speaking: bool = False
    speaking_confidence: float = 0.0
    is_hand_raised: bool = False
    is_pointing: bool = False
    is_standing: bool = True
    gesture: Optional[str] = None
    activity: str = "idle"


class ActivityDetector:
    """Detects activities from pose and face analysis results."""
    
    # Pose keypoint indices (MediaPipe)
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    NOSE = 0
    
    def __init__(self, speaking_threshold: float = 0.5):
        self.speaking_threshold = speaking_threshold
        logger.info("ActivityDetector initialized")
    
    def detect(self, pose: Optional[PoseResult] = None,
               face: Optional[FaceAnalysisResult] = None) -> ActivityResult:
        """Detect activities from pose and face."""
        result = ActivityResult()
        
        # Speaking from face analysis
        if face is not None:
            result.is_speaking = face.is_speaking
            result.speaking_confidence = min(1.0, face.lip_distance / 0.1)
        
        # Gestures from pose
        if pose is not None and pose.visibility is not None:
            result.is_hand_raised = self._detect_hand_raised(pose)
            result.is_pointing = self._detect_pointing(pose)
            result.is_standing = self._detect_standing(pose)
            result.gesture = self._classify_gesture(pose)
        
        # Overall activity
        if result.is_speaking:
            result.activity = "speaking"
        elif result.is_hand_raised:
            result.activity = "hand_raised"
        elif result.is_pointing:
            result.activity = "pointing"
        elif not result.is_standing:
            result.activity = "sitting"
        else:
            result.activity = "standing"
        
        return result
    
    def _detect_hand_raised(self, pose: PoseResult) -> bool:
        """Detect if either hand is raised above shoulder."""
        lm = pose.landmarks
        vis = pose.visibility
        
        # Check visibility
        if vis[self.LEFT_WRIST] < 0.5 and vis[self.RIGHT_WRIST] < 0.5:
            return False
        
        # Left hand raised
        if vis[self.LEFT_WRIST] > 0.5 and vis[self.LEFT_SHOULDER] > 0.5:
            if lm[self.LEFT_WRIST][1] < lm[self.LEFT_SHOULDER][1]:  # Y is inverted
                return True
        
        # Right hand raised
        if vis[self.RIGHT_WRIST] > 0.5 and vis[self.RIGHT_SHOULDER] > 0.5:
            if lm[self.RIGHT_WRIST][1] < lm[self.RIGHT_SHOULDER][1]:
                return True
        
        return False
    
    def _detect_pointing(self, pose: PoseResult) -> bool:
        """Detect pointing gesture (arm extended)."""
        lm = pose.landmarks
        vis = pose.visibility
        
        for side in [(self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST),
                     (self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST)]:
            shoulder, elbow, wrist = side
            
            if all(vis[i] > 0.5 for i in [shoulder, elbow, wrist]):
                # Check if arm is extended (shoulder-elbow-wrist roughly colinear)
                v1 = lm[elbow][:2] - lm[shoulder][:2]
                v2 = lm[wrist][:2] - lm[elbow][:2]
                
                arm_length = np.linalg.norm(v1) + np.linalg.norm(v2)
                direct_dist = np.linalg.norm(lm[wrist][:2] - lm[shoulder][:2])
                
                # Arm extended if direct distance is close to total length
                if direct_dist > arm_length * 0.9:
                    return True
        
        return False
    
    def _detect_standing(self, pose: PoseResult) -> bool:
        """Detect if person is standing or sitting."""
        lm = pose.landmarks
        vis = pose.visibility
        
        # Need hip and knee visibility
        if vis[self.LEFT_HIP] < 0.5 or vis[self.LEFT_KNEE] < 0.5:
            return True  # Default to standing if can't determine
        
        # Standing: hip significantly above knee in y-axis
        hip_y = (lm[self.LEFT_HIP][1] + lm[self.RIGHT_HIP][1]) / 2
        knee_y = (lm[self.LEFT_KNEE][1] + lm[self.RIGHT_KNEE][1]) / 2
        
        # If hip is much higher than knee, standing
        torso_length = abs(lm[self.NOSE][1] - hip_y)
        hip_knee_diff = knee_y - hip_y
        
        return hip_knee_diff > torso_length * 0.3
    
    def _classify_gesture(self, pose: PoseResult) -> Optional[str]:
        """Classify specific gestures."""
        if self._detect_hand_raised(pose):
            return "hand_raised"
        if self._detect_pointing(pose):
            return "pointing"
        if self._detect_arms_crossed(pose):
            return "arms_crossed"
        return None
    
    def _detect_arms_crossed(self, pose: PoseResult) -> bool:
        """Detect arms crossed gesture."""
        lm = pose.landmarks
        vis = pose.visibility
        
        if all(vis[i] > 0.5 for i in [self.LEFT_WRIST, self.RIGHT_WRIST, 
                                       self.LEFT_SHOULDER, self.RIGHT_SHOULDER]):
            # Wrists near opposite shoulders
            left_to_right = np.linalg.norm(lm[self.LEFT_WRIST][:2] - lm[self.RIGHT_SHOULDER][:2])
            right_to_left = np.linalg.norm(lm[self.RIGHT_WRIST][:2] - lm[self.LEFT_SHOULDER][:2])
            shoulder_width = np.linalg.norm(lm[self.LEFT_SHOULDER][:2] - lm[self.RIGHT_SHOULDER][:2])
            
            if left_to_right < shoulder_width * 0.5 and right_to_left < shoulder_width * 0.5:
                return True
        
        return False
