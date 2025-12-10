"""
Event Prediction - Anticipate events before they occur.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
from loguru import logger


class PredictedEvent(Enum):
    """Types of events that can be predicted."""
    MEETING_END = "meeting_end"
    TOPIC_CHANGE = "topic_change"
    ESCALATION = "escalation"
    BREAKTHROUGH = "breakthrough"
    DISENGAGEMENT = "disengagement"
    QUESTION_COMING = "question_coming"
    INTERRUPTION_LIKELY = "interruption_likely"


@dataclass 
class Prediction:
    """A predicted event."""
    event_type: PredictedEvent
    probability: float
    time_horizon: float  # Predicted seconds until event
    confidence: float
    contributing_factors: List[str]


class EventPredictor:
    """
    Predicts upcoming events using pattern recognition.
    
    Uses historical patterns and real-time signals to
    anticipate events before they occur.
    """
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        
        # Feature history
        self._feature_history: deque = deque(maxlen=history_window)
        
        # Pattern templates (learned or predefined)
        self._meeting_end_patterns = self._init_meeting_end_patterns()
        self._escalation_patterns = self._init_escalation_patterns()
        
        # Active predictions
        self._predictions: List[Prediction] = []
        
        logger.info("EventPredictor initialized")
    
    def _init_meeting_end_patterns(self) -> List[dict]:
        """Initialize patterns that predict meeting end."""
        return [
            {
                "name": "winding_down",
                "features": {
                    "speech_rate_declining": True,
                    "turn_frequency_declining": True,
                    "sentiment_stable": True
                },
                "weight": 0.6
            },
            {
                "name": "summary_phase",
                "features": {
                    "single_speaker_dominant": True,
                    "questions_decreasing": True
                },
                "weight": 0.5
            },
            {
                "name": "restlessness",
                "features": {
                    "movement_increasing": True,
                    "attention_declining": True
                },
                "weight": 0.4
            }
        ]
    
    def _init_escalation_patterns(self) -> List[dict]:
        """Initialize patterns that predict escalation."""
        return [
            {
                "name": "rising_tension",
                "features": {
                    "pitch_increasing": True,
                    "volume_increasing": True,
                    "interruptions_increasing": True
                },
                "weight": 0.7
            },
            {
                "name": "aggressive_body_language",
                "features": {
                    "forward_lean": True,
                    "pointing_gestures": True
                },
                "weight": 0.5
            }
        ]
    
    def update(self,
               speech_rate: float = 1.0,
               volume: float = 0.1,
               pitch: float = 150.0,
               interruption_rate: float = 0.0,
               sentiment: float = 0.0,
               num_active_speakers: int = 1,
               movement_level: float = 0.0,
               attention_score: float = 1.0,
               is_question: bool = False,
               timestamp: Optional[float] = None):
        """
        Update with new observations.
        
        Called frequently to build up feature history
        for pattern matching.
        """
        timestamp = timestamp or time.time()
        
        features = {
            "timestamp": timestamp,
            "speech_rate": speech_rate,
            "volume": volume,
            "pitch": pitch,
            "interruption_rate": interruption_rate,
            "sentiment": sentiment,
            "num_active_speakers": num_active_speakers,
            "movement_level": movement_level,
            "attention_score": attention_score,
            "is_question": is_question
        }
        
        self._feature_history.append(features)
        
        # Update predictions
        self._update_predictions()
    
    def _update_predictions(self):
        """Update current predictions based on feature history."""
        if len(self._feature_history) < 10:
            return
        
        self._predictions.clear()
        
        # Check meeting end
        meeting_end_prob = self._predict_meeting_end()
        if meeting_end_prob > 0.3:
            self._predictions.append(Prediction(
                event_type=PredictedEvent.MEETING_END,
                probability=meeting_end_prob,
                time_horizon=self._estimate_time_to_end(),
                confidence=min(len(self._feature_history) / self.history_window, 0.8),
                contributing_factors=self._get_meeting_end_factors()
            ))
        
        # Check escalation
        escalation_prob = self._predict_escalation()
        if escalation_prob > 0.4:
            self._predictions.append(Prediction(
                event_type=PredictedEvent.ESCALATION,
                probability=escalation_prob,
                time_horizon=30.0,  # Typically builds over ~30 seconds
                confidence=0.6,
                contributing_factors=self._get_escalation_factors()
            ))
        
        # Check disengagement
        disengage_prob = self._predict_disengagement()
        if disengage_prob > 0.4:
            self._predictions.append(Prediction(
                event_type=PredictedEvent.DISENGAGEMENT,
                probability=disengage_prob,
                time_horizon=60.0,
                confidence=0.5,
                contributing_factors=["declining_attention", "low_participation"]
            ))
    
    def _predict_meeting_end(self) -> float:
        """Predict probability of meeting ending soon."""
        recent = list(self._feature_history)[-20:]
        older = list(self._feature_history)[-40:-20] if len(self._feature_history) > 40 else recent
        
        if len(recent) < 10:
            return 0.0
        
        # Declining speech rate
        recent_speech_rate = np.mean([f["speech_rate"] for f in recent])
        older_speech_rate = np.mean([f["speech_rate"] for f in older])
        speech_declining = recent_speech_rate < older_speech_rate * 0.8
        
        # Declining movement
        recent_movement = np.mean([f["movement_level"] for f in recent])
        movement_low = recent_movement < 0.3
        
        # Single speaker (summary)
        recent_speakers = np.mean([f["num_active_speakers"] for f in recent])
        single_speaker = recent_speakers <= 1.2
        
        # Combine signals
        prob = 0.0
        if speech_declining:
            prob += 0.25
        if movement_low:
            prob += 0.15
        if single_speaker:
            prob += 0.2
        
        return min(prob, 0.9)
    
    def _predict_escalation(self) -> float:
        """Predict probability of escalation."""
        recent = list(self._feature_history)[-15:]
        older = list(self._feature_history)[-30:-15] if len(self._feature_history) > 30 else recent
        
        if len(recent) < 10:
            return 0.0
        
        # Rising volume
        recent_volume = np.mean([f["volume"] for f in recent])
        older_volume = np.mean([f["volume"] for f in older])
        volume_rising = recent_volume > older_volume * 1.3
        
        # Rising pitch
        recent_pitch = np.mean([f["pitch"] for f in recent])
        older_pitch = np.mean([f["pitch"] for f in older])
        pitch_rising = recent_pitch > older_pitch * 1.2
        
        # Increasing interruptions
        recent_interrupts = np.mean([f["interruption_rate"] for f in recent])
        interrupt_high = recent_interrupts > 0.3
        
        # Declining sentiment
        recent_sentiment = np.mean([f["sentiment"] for f in recent])
        sentiment_negative = recent_sentiment < -0.2
        
        prob = 0.0
        if volume_rising:
            prob += 0.25
        if pitch_rising:
            prob += 0.2
        if interrupt_high:
            prob += 0.25
        if sentiment_negative:
            prob += 0.2
        
        return min(prob, 0.95)
    
    def _predict_disengagement(self) -> float:
        """Predict participant disengagement."""
        recent = list(self._feature_history)[-20:]
        
        if len(recent) < 10:
            return 0.0
        
        avg_attention = np.mean([f["attention_score"] for f in recent])
        avg_movement = np.mean([f["movement_level"] for f in recent])
        avg_speakers = np.mean([f["num_active_speakers"] for f in recent])
        
        prob = 0.0
        if avg_attention < 0.5:
            prob += 0.4
        if avg_movement > 0.6:  # Restlessness
            prob += 0.2
        if avg_speakers < 1.5:  # Low participation
            prob += 0.2
        
        return min(prob, 0.9)
    
    def _estimate_time_to_end(self) -> float:
        """Estimate seconds until meeting ends."""
        # Heuristic based on declining rates
        recent = list(self._feature_history)[-20:]
        if len(recent) < 10:
            return 300.0  # Default 5 minutes
        
        speech_rates = [f["speech_rate"] for f in recent]
        decline_rate = (speech_rates[0] - speech_rates[-1]) / len(speech_rates)
        
        if decline_rate <= 0:
            return 300.0
        
        # Estimate time to reach zero activity
        remaining = speech_rates[-1] / decline_rate
        return min(max(remaining * 10, 30), 600)  # 30s to 10 min
    
    def _get_meeting_end_factors(self) -> List[str]:
        """Get factors contributing to meeting end prediction."""
        factors = []
        recent = list(self._feature_history)[-20:]
        
        if len(recent) >= 10:
            if np.mean([f["speech_rate"] for f in recent]) < 0.8:
                factors.append("declining_speech")
            if np.mean([f["movement_level"] for f in recent]) > 0.4:
                factors.append("restlessness")
            if np.mean([f["attention_score"] for f in recent]) < 0.6:
                factors.append("declining_attention")
        
        return factors
    
    def _get_escalation_factors(self) -> List[str]:
        """Get factors contributing to escalation prediction."""
        factors = []
        recent = list(self._feature_history)[-15:]
        
        if len(recent) >= 10:
            if np.mean([f["volume"] for f in recent]) > 0.3:
                factors.append("raised_voices")
            if np.mean([f["pitch"] for f in recent]) > 180:
                factors.append("high_pitch")
            if np.mean([f["interruption_rate"] for f in recent]) > 0.3:
                factors.append("frequent_interruptions")
            if np.mean([f["sentiment"] for f in recent]) < -0.2:
                factors.append("negative_sentiment")
        
        return factors
    
    def get_predictions(self, min_probability: float = 0.3) -> List[Prediction]:
        """Get current predictions above threshold."""
        return [p for p in self._predictions if p.probability >= min_probability]
    
    def get_prediction(self, event_type: PredictedEvent) -> Optional[Prediction]:
        """Get prediction for specific event type."""
        for p in self._predictions:
            if p.event_type == event_type:
                return p
        return None
    
    def get_alert_predictions(self) -> List[Prediction]:
        """Get predictions that should trigger alerts."""
        alert_types = {
            PredictedEvent.ESCALATION,
            PredictedEvent.DISENGAGEMENT
        }
        return [p for p in self._predictions 
                if p.event_type in alert_types and p.probability > 0.5]
