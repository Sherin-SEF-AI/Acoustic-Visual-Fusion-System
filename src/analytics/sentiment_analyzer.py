"""
Sentiment Analysis - Meeting mood tracking with acoustic and visual fusion.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
from loguru import logger


class Emotion(Enum):
    """Detected emotions."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ENTHUSIASTIC = "enthusiastic"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    STRESSED = "stressed"
    BORED = "bored"
    ENGAGED = "engaged"


class MeetingMood(Enum):
    """Overall meeting mood states."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    TENSE = "tense"
    ENERGETIC = "energetic"
    FLAT = "flat"


@dataclass
class SentimentPoint:
    """A sentiment measurement at a point in time."""
    timestamp: float
    speaker_id: Optional[str]
    
    # Acoustic sentiment
    acoustic_valence: float  # -1 to 1 (negative to positive)
    acoustic_arousal: float  # 0 to 1 (calm to energetic)
    acoustic_confidence: float
    
    # Visual sentiment
    visual_emotion: Emotion
    visual_confidence: float
    
    # Fused sentiment
    combined_valence: float
    combined_arousal: float
    
    def get_mood(self) -> MeetingMood:
        """Determine mood from valence and arousal."""
        if self.combined_valence > 0.3:
            if self.combined_arousal > 0.5:
                return MeetingMood.ENERGETIC
            return MeetingMood.POSITIVE
        elif self.combined_valence < -0.3:
            return MeetingMood.TENSE
        else:
            if self.combined_arousal < 0.3:
                return MeetingMood.FLAT
            return MeetingMood.NEUTRAL


@dataclass
class SentimentShift:
    """A significant sentiment change."""
    timestamp: float
    speaker_id: Optional[str]
    from_valence: float
    to_valence: float
    shift_type: str  # "positive", "negative", "tension", "breakthrough"
    magnitude: float


class SentimentAnalyzer:
    """
    Analyzes sentiment from acoustic and visual features.
    
    Tracks meeting mood over time and correlates sentiment
    shifts with speakers.
    """
    
    # Acoustic sentiment thresholds
    STRESS_PITCH_RATIO = 1.3  # Pitch increase in stress
    FRUSTRATION_VOLUME_RATIO = 1.4
    ENTHUSIASM_ENERGY_RATIO = 1.5
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Window for averaging sentiment
        """
        self.window_size = window_size
        
        # Sentiment history
        self._sentiment_history: deque = deque(maxlen=1000)
        self._mood_history: deque = deque(maxlen=200)
        
        # Per-speaker sentiment
        self._speaker_sentiment: Dict[str, deque] = {}
        
        # Baselines (learned)
        self._baseline_pitch: float = 150.0
        self._baseline_volume: float = 0.1
        self._baseline_energy: float = 0.5
        
        # Shift detection
        self._recent_shifts: List[SentimentShift] = []
        self._shift_threshold = 0.3
        
        logger.info("SentimentAnalyzer initialized")
    
    def analyze_acoustic(self, 
                        pitch: float,
                        volume: float,
                        energy: float,
                        speech_rate: float = 1.0,
                        spectral_features: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        Analyze sentiment from acoustic features.
        
        Returns:
            (valence, arousal, confidence)
        """
        # Calculate relative deviations from baseline
        pitch_ratio = pitch / max(self._baseline_pitch, 1)
        volume_ratio = volume / max(self._baseline_volume, 0.01)
        energy_ratio = energy / max(self._baseline_energy, 0.01)
        
        # Arousal is generally correlated with pitch, volume, energy
        arousal = np.clip(
            0.3 * (pitch_ratio - 1) + 
            0.3 * (volume_ratio - 1) + 
            0.2 * (energy_ratio - 1) + 
            0.2 * (speech_rate - 1),
            0, 1
        )
        
        # Valence is trickier - use heuristics
        valence = 0.0
        confidence = 0.5
        
        # High pitch + fast speech often positive
        if pitch_ratio > 1.1 and speech_rate > 1.1:
            valence = 0.3
            confidence = 0.6
        
        # Very high pitch + high volume can indicate stress (negative)
        if pitch_ratio > self.STRESS_PITCH_RATIO and volume_ratio > 1.2:
            valence = -0.4
            confidence = 0.7
        
        # Moderate values are neutral
        if 0.9 < pitch_ratio < 1.1 and 0.9 < volume_ratio < 1.1:
            valence = 0.0
            confidence = 0.5
        
        # Low energy, slow speech often negative
        if energy_ratio < 0.7 and speech_rate < 0.9:
            valence = -0.2
            confidence = 0.55
        
        return valence, arousal, confidence
    
    def analyze_visual(self,
                      facial_landmarks: Optional[np.ndarray] = None,
                      facial_au: Optional[Dict[str, float]] = None,
                      detected_emotion: Optional[str] = None) -> Tuple[Emotion, float]:
        """
        Analyze sentiment from visual features.
        
        Args:
            facial_landmarks: Facial landmark positions
            facial_au: Action unit activations
            detected_emotion: Pre-detected emotion label
            
        Returns:
            (emotion, confidence)
        """
        # If emotion already detected
        if detected_emotion:
            try:
                emotion = Emotion(detected_emotion.lower())
                return emotion, 0.7
            except ValueError:
                pass
        
        # Action unit based detection
        if facial_au:
            # AU6 (cheek raise) + AU12 (lip corner pull) = smile/happy
            if facial_au.get("AU6", 0) > 0.5 and facial_au.get("AU12", 0) > 0.5:
                return Emotion.HAPPY, 0.8
            
            # AU4 (brow lowerer) = frustrated/confused
            if facial_au.get("AU4", 0) > 0.6:
                if facial_au.get("AU9", 0) > 0.3:  # Nose wrinkler
                    return Emotion.FRUSTRATED, 0.7
                return Emotion.CONFUSED, 0.65
            
            # AU1+AU2 (inner/outer brow raise) = surprised/engaged
            if facial_au.get("AU1", 0) > 0.4 and facial_au.get("AU2", 0) > 0.4:
                return Emotion.ENGAGED, 0.6
        
        return Emotion.NEUTRAL, 0.4
    
    def fuse_sentiment(self, 
                       acoustic_valence: float,
                       acoustic_arousal: float,
                       acoustic_confidence: float,
                       visual_emotion: Emotion,
                       visual_confidence: float) -> Tuple[float, float]:
        """
        Fuse acoustic and visual sentiment.
        
        Returns:
            (combined_valence, combined_arousal)
        """
        # Map visual emotion to valence/arousal
        emotion_valence = {
            Emotion.NEUTRAL: 0.0,
            Emotion.HAPPY: 0.7,
            Emotion.ENTHUSIASTIC: 0.8,
            Emotion.CONFUSED: -0.2,
            Emotion.FRUSTRATED: -0.6,
            Emotion.STRESSED: -0.5,
            Emotion.BORED: -0.2,
            Emotion.ENGAGED: 0.3
        }
        
        emotion_arousal = {
            Emotion.NEUTRAL: 0.3,
            Emotion.HAPPY: 0.6,
            Emotion.ENTHUSIASTIC: 0.9,
            Emotion.CONFUSED: 0.4,
            Emotion.FRUSTRATED: 0.7,
            Emotion.STRESSED: 0.8,
            Emotion.BORED: 0.2,
            Emotion.ENGAGED: 0.6
        }
        
        visual_valence = emotion_valence.get(visual_emotion, 0.0)
        visual_arousal = emotion_arousal.get(visual_emotion, 0.3)
        
        # Weight by confidence
        total_confidence = acoustic_confidence + visual_confidence
        if total_confidence == 0:
            return 0.0, 0.3
        
        w_acoustic = acoustic_confidence / total_confidence
        w_visual = visual_confidence / total_confidence
        
        combined_valence = w_acoustic * acoustic_valence + w_visual * visual_valence
        combined_arousal = w_acoustic * acoustic_arousal + w_visual * visual_arousal
        
        return combined_valence, combined_arousal
    
    def update(self,
               speaker_id: Optional[str] = None,
               pitch: float = 150.0,
               volume: float = 0.1,
               energy: float = 0.5,
               speech_rate: float = 1.0,
               visual_emotion: Optional[str] = None,
               facial_au: Optional[Dict[str, float]] = None,
               timestamp: Optional[float] = None) -> SentimentPoint:
        """
        Update sentiment with new observations.
        
        Returns:
            Current sentiment point
        """
        timestamp = timestamp or time.time()
        
        # Acoustic analysis
        acoustic_valence, acoustic_arousal, acoustic_conf = self.analyze_acoustic(
            pitch, volume, energy, speech_rate
        )
        
        # Visual analysis
        emotion, visual_conf = self.analyze_visual(
            detected_emotion=visual_emotion,
            facial_au=facial_au
        )
        
        # Fusion
        combined_valence, combined_arousal = self.fuse_sentiment(
            acoustic_valence, acoustic_arousal, acoustic_conf,
            emotion, visual_conf
        )
        
        # Create sentiment point
        point = SentimentPoint(
            timestamp=timestamp,
            speaker_id=speaker_id,
            acoustic_valence=acoustic_valence,
            acoustic_arousal=acoustic_arousal,
            acoustic_confidence=acoustic_conf,
            visual_emotion=emotion,
            visual_confidence=visual_conf,
            combined_valence=combined_valence,
            combined_arousal=combined_arousal
        )
        
        # Store
        self._sentiment_history.append(point)
        self._mood_history.append((timestamp, point.get_mood()))
        
        # Per-speaker tracking
        if speaker_id:
            if speaker_id not in self._speaker_sentiment:
                self._speaker_sentiment[speaker_id] = deque(maxlen=100)
            self._speaker_sentiment[speaker_id].append(point)
        
        # Check for shift
        self._detect_shift(point)
        
        # Update baselines (slow adaptation)
        self._update_baselines(pitch, volume, energy)
        
        return point
    
    def _detect_shift(self, current: SentimentPoint):
        """Detect significant sentiment shifts."""
        if len(self._sentiment_history) < 5:
            return
        
        # Compare to recent average
        recent = list(self._sentiment_history)[-10:-1]
        avg_valence = np.mean([p.combined_valence for p in recent])
        
        delta = current.combined_valence - avg_valence
        
        if abs(delta) > self._shift_threshold:
            shift_type = "positive" if delta > 0 else "negative"
            if delta > 0.5:
                shift_type = "breakthrough"
            elif delta < -0.5:
                shift_type = "tension"
            
            shift = SentimentShift(
                timestamp=current.timestamp,
                speaker_id=current.speaker_id,
                from_valence=avg_valence,
                to_valence=current.combined_valence,
                shift_type=shift_type,
                magnitude=abs(delta)
            )
            
            self._recent_shifts.append(shift)
            logger.debug(f"Sentiment shift detected: {shift_type} ({delta:.2f})")
    
    def _update_baselines(self, pitch: float, volume: float, energy: float):
        """Slowly update baselines."""
        alpha = 0.01  # Slow adaptation
        self._baseline_pitch = (1 - alpha) * self._baseline_pitch + alpha * pitch
        self._baseline_volume = (1 - alpha) * self._baseline_volume + alpha * volume
        self._baseline_energy = (1 - alpha) * self._baseline_energy + alpha * energy
    
    def get_current_mood(self) -> MeetingMood:
        """Get current meeting mood."""
        if not self._sentiment_history:
            return MeetingMood.NEUTRAL
        
        # Average recent points
        recent = list(self._sentiment_history)[-self.window_size:]
        avg_valence = np.mean([p.combined_valence for p in recent])
        avg_arousal = np.mean([p.combined_arousal for p in recent])
        
        if avg_valence > 0.3:
            return MeetingMood.ENERGETIC if avg_arousal > 0.5 else MeetingMood.POSITIVE
        elif avg_valence < -0.3:
            return MeetingMood.TENSE
        else:
            return MeetingMood.FLAT if avg_arousal < 0.3 else MeetingMood.NEUTRAL
    
    def get_mood_timeline(self, window_seconds: float = 300) -> List[Tuple[float, str]]:
        """Get mood history for timeline."""
        cutoff = time.time() - window_seconds
        return [(t, m.value) for t, m in self._mood_history if t >= cutoff]
    
    def get_speaker_sentiment(self, speaker_id: str) -> dict:
        """Get average sentiment for a speaker."""
        if speaker_id not in self._speaker_sentiment:
            return {"valence": 0, "arousal": 0, "dominant_emotion": "neutral"}
        
        points = list(self._speaker_sentiment[speaker_id])
        if not points:
            return {"valence": 0, "arousal": 0, "dominant_emotion": "neutral"}
        
        avg_valence = np.mean([p.combined_valence for p in points])
        avg_arousal = np.mean([p.combined_arousal for p in points])
        
        # Find dominant emotion
        emotions = [p.visual_emotion for p in points]
        from collections import Counter
        emotion_counts = Counter(emotions)
        dominant = emotion_counts.most_common(1)[0][0]
        
        return {
            "valence": avg_valence,
            "arousal": avg_arousal,
            "dominant_emotion": dominant.value
        }
    
    def get_sentiment_timeline_data(self) -> dict:
        """Get data for sentiment timeline visualization."""
        return {
            "timestamps": [p.timestamp for p in self._sentiment_history],
            "valence": [p.combined_valence for p in self._sentiment_history],
            "arousal": [p.combined_arousal for p in self._sentiment_history],
            "shifts": [
                {
                    "timestamp": s.timestamp,
                    "type": s.shift_type,
                    "speaker": s.speaker_id,
                    "magnitude": s.magnitude
                }
                for s in self._recent_shifts
            ]
        }
