"""
Gesture Recognition - Detect and classify hand/body gestures.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from collections import deque
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class GestureType(Enum):
    """Recognized gesture types."""
    NONE = "none"
    WAVE = "wave"
    POINT = "point"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    STOP = "stop"
    COME_HERE = "come_here"
    RAISED_HAND = "raised_hand"
    CLAP = "clap"


@dataclass
class GestureResult:
    """Result of gesture detection."""
    gesture: GestureType
    confidence: float
    hand_position: Optional[tuple]
    bounding_box: Optional[list]
    track_id: Optional[int]


class GestureRecognizer:
    """
    Recognizes hand gestures from pose landmarks.
    Uses heuristics on landmark positions.
    """
    
    def __init__(self, min_confidence: float = 0.5,
                 smoothing_window: int = 5):
        self.min_confidence = min_confidence
        self.smoothing_window = smoothing_window
        
        # History for temporal smoothing
        self._gesture_history: dict[int, deque] = {}
        
        # Landmark indices (MediaPipe convention)
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        logger.info("GestureRecognizer initialized")
    
    def recognize(self, landmarks: np.ndarray, 
                  track_id: int = 0,
                  hand: str = "right") -> GestureResult:
        """
        Recognize gesture from hand landmarks.
        
        Args:
            landmarks: Array of shape (21, 3) with hand landmarks
            track_id: ID of the person/track
            hand: "left" or "right"
            
        Returns:
            GestureResult with detected gesture
        """
        if landmarks is None or len(landmarks) < 21:
            return GestureResult(GestureType.NONE, 0.0, None, None, track_id)
        
        # Get fingertip positions relative to wrist
        wrist = landmarks[self.WRIST]
        thumb = landmarks[self.THUMB_TIP]
        index = landmarks[self.INDEX_TIP]
        middle = landmarks[self.MIDDLE_TIP]
        ring = landmarks[self.RING_TIP]
        pinky = landmarks[self.PINKY_TIP]
        
        # Calculate finger extension (relative to wrist height)
        fingers_up = self._count_fingers_up(landmarks)
        hand_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        # Detect gestures based on finger positions
        gesture = GestureType.NONE
        confidence = 0.0
        
        # Raised hand / Stop - all fingers up, hand high
        if fingers_up >= 4:
            gesture = GestureType.RAISED_HAND
            confidence = 0.8
            
            # Check if palm facing forward (stop gesture)
            if self._is_palm_forward(landmarks):
                gesture = GestureType.STOP
                confidence = 0.85
        
        # Point - only index extended
        elif fingers_up == 1 and self._is_finger_extended(landmarks, self.INDEX_TIP):
            gesture = GestureType.POINT
            confidence = 0.8
        
        # Thumbs up - thumb up, others closed
        elif self._is_thumbs_up(landmarks):
            gesture = GestureType.THUMBS_UP
            confidence = 0.75
        
        # Thumbs down
        elif self._is_thumbs_down(landmarks):
            gesture = GestureType.THUMBS_DOWN
            confidence = 0.75
        
        # Apply temporal smoothing
        gesture, confidence = self._smooth_gesture(track_id, gesture, confidence)
        
        # Calculate bounding box
        bbox = [
            float(np.min(landmarks[:, 0])),
            float(np.min(landmarks[:, 1])),
            float(np.max(landmarks[:, 0])),
            float(np.max(landmarks[:, 1]))
        ]
        
        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            hand_position=(float(wrist[0]), float(wrist[1])),
            bounding_box=bbox,
            track_id=track_id
        )
    
    def detect_wave(self, position_history: list) -> bool:
        """Detect wave gesture from position history."""
        if len(position_history) < 10:
            return False
        
        # Check for oscillating horizontal movement
        x_positions = [p[0] for p in position_history[-10:]]
        
        # Calculate direction changes
        directions = np.diff(x_positions)
        sign_changes = np.sum(np.diff(np.sign(directions)) != 0)
        
        return sign_changes >= 3
    
    def _count_fingers_up(self, landmarks: np.ndarray) -> int:
        """Count number of extended fingers."""
        count = 0
        
        tips = [4, 8, 12, 16, 20]
        base = [2, 5, 9, 13, 17]
        
        for tip_idx, base_idx in zip(tips, base):
            if landmarks[tip_idx][1] < landmarks[base_idx][1]:
                count += 1
        
        return count
    
    def _is_finger_extended(self, landmarks: np.ndarray, tip_idx: int) -> bool:
        """Check if a finger is extended."""
        wrist_y = landmarks[self.WRIST][1]
        tip_y = landmarks[tip_idx][1]
        
        # Finger is extended if tip is above wrist
        return tip_y < wrist_y - 0.05
    
    def _is_palm_forward(self, landmarks: np.ndarray) -> bool:
        """Check if palm is facing forward (z-depth check)."""
        # Simplified: check if fingers are spread
        index = landmarks[self.INDEX_TIP]
        pinky = landmarks[self.PINKY_TIP]
        
        spread = np.abs(index[0] - pinky[0])
        return spread > 0.1
    
    def _is_thumbs_up(self, landmarks: np.ndarray) -> bool:
        """Detect thumbs up gesture."""
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        
        # Thumb above index, index below wrist
        return (thumb_tip[1] < index_tip[1] and 
                self._count_fingers_up(landmarks) <= 2)
    
    def _is_thumbs_down(self, landmarks: np.ndarray) -> bool:
        """Detect thumbs down gesture."""
        thumb_tip = landmarks[self.THUMB_TIP]
        wrist = landmarks[self.WRIST]
        
        return (thumb_tip[1] > wrist[1] and
                self._count_fingers_up(landmarks) <= 2)
    
    def _smooth_gesture(self, track_id: int, 
                        gesture: GestureType, 
                        confidence: float) -> tuple:
        """Apply temporal smoothing to gesture detection."""
        if track_id not in self._gesture_history:
            self._gesture_history[track_id] = deque(maxlen=self.smoothing_window)
        
        self._gesture_history[track_id].append((gesture, confidence))
        
        # Count gesture occurrences
        gesture_counts = {}
        for g, c in self._gesture_history[track_id]:
            if g not in gesture_counts:
                gesture_counts[g] = []
            gesture_counts[g].append(c)
        
        # Find most common gesture
        best_gesture = GestureType.NONE
        best_count = 0
        best_conf = 0.0
        
        for g, confs in gesture_counts.items():
            if len(confs) > best_count:
                best_count = len(confs)
                best_gesture = g
                best_conf = np.mean(confs)
        
        # Require majority for stable detection
        if best_count >= self.smoothing_window // 2:
            return best_gesture, best_conf
        
        return GestureType.NONE, 0.0


class BodyGestureRecognizer:
    """
    Recognizes body-level gestures from pose landmarks.
    """
    
    # Pose landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    def __init__(self):
        self._position_history: dict[int, deque] = {}
        logger.info("BodyGestureRecognizer initialized")
    
    def detect(self, pose_landmarks: np.ndarray,
               track_id: int = 0) -> dict:
        """
        Detect body gestures from pose landmarks.
        
        Returns dict with:
            - raised_hands: bool
            - waving: bool  
            - pointing_direction: Optional[str]
            - body_orientation: str
        """
        if pose_landmarks is None or len(pose_landmarks) < 25:
            return {'raised_hands': False, 'waving': False}
        
        result = {
            'raised_hands': False,
            'waving': False,
            'pointing_direction': None,
            'body_orientation': 'forward'
        }
        
        # Check raised hands
        left_wrist = pose_landmarks[self.LEFT_WRIST]
        right_wrist = pose_landmarks[self.RIGHT_WRIST]
        left_shoulder = pose_landmarks[self.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[self.RIGHT_SHOULDER]
        
        left_raised = left_wrist[1] < left_shoulder[1]
        right_raised = right_wrist[1] < right_shoulder[1]
        
        result['raised_hands'] = left_raised or right_raised
        
        # Track wrist positions for wave detection
        if track_id not in self._position_history:
            self._position_history[track_id] = deque(maxlen=30)
        
        self._position_history[track_id].append((left_wrist, right_wrist))
        
        # Detect waving
        if len(self._position_history[track_id]) >= 10:
            positions = list(self._position_history[track_id])
            
            # Check for oscillating movement
            for hand_idx in [0, 1]:  # left, right
                x_pos = [p[hand_idx][0] for p in positions[-10:]]
                directions = np.diff(x_pos)
                sign_changes = np.sum(np.diff(np.sign(directions)) != 0)
                
                if sign_changes >= 3:
                    result['waving'] = True
                    break
        
        # Pointing direction
        if result['raised_hands']:
            if left_raised and not right_raised:
                result['pointing_direction'] = 'left'
            elif right_raised and not left_raised:
                result['pointing_direction'] = 'right'
        
        return result
