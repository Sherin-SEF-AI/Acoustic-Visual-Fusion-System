"""
Turn-Taking Analysis - Model conversation rhythm and flow.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
from loguru import logger


class PauseType(Enum):
    """Types of conversation pauses."""
    NATURAL = "natural"  # Normal turn-taking pause
    THOUGHTFUL = "thoughtful"  # Contemplative pause
    AWKWARD = "awkward"  # Uncomfortable silence
    DRAMATIC = "dramatic"  # Intentional for effect


class ConversationMomentum(Enum):
    """Conversation momentum states."""
    FLOWING = "flowing"
    BUILDING = "building"
    STALLING = "stalling"
    DEAD = "dead"


@dataclass
class TurnEvent:
    """A turn-taking event."""
    timestamp: float
    speaker_id: str
    event_type: str  # "start", "end", "question", "response"
    duration: float = 0.0
    is_question: bool = False
    response_to: Optional[str] = None


@dataclass
class ConversationState:
    """Current state of the conversation."""
    momentum: ConversationMomentum
    momentum_score: float  # 0-1
    current_speaker: Optional[str]
    last_speaker: Optional[str]
    avg_turn_duration: float
    avg_pause_duration: float
    question_pending: bool
    silence_duration: float


class TurnTakingAnalyzer:
    """
    Analyzes turn-taking patterns in conversations.
    
    Tracks conversation rhythm, detects pause types,
    measures momentum, and identifies question-response patterns.
    """
    
    # Pause duration thresholds (seconds)
    NATURAL_PAUSE_MAX = 1.0
    THOUGHTFUL_PAUSE_MAX = 3.0
    AWKWARD_PAUSE_MIN = 4.0
    
    # Momentum decay parameters
    MOMENTUM_DECAY_RATE = 0.1  # Per second of silence
    MOMENTUM_BOOST_TURN = 0.1
    MOMENTUM_BOOST_RESPONSE = 0.15
    
    def __init__(self):
        # State
        self._current_speaker: Optional[str] = None
        self._last_speaker: Optional[str] = None
        self._speech_start: float = 0.0
        self._silence_start: Optional[float] = None
        
        # History
        self._turn_events: deque = deque(maxlen=500)
        self._pause_history: deque = deque(maxlen=100)
        self._momentum_history: deque = deque(maxlen=100)
        
        # Question tracking
        self._pending_question: Optional[Tuple[str, float]] = None
        self._question_response_times: List[float] = []
        
        # Statistics
        self._turn_durations: Dict[str, List[float]] = {}
        self._response_patterns: Dict[str, Dict[str, int]] = {}
        
        # Momentum
        self._momentum_score = 0.5
        self._last_update = time.time()
        
        logger.info("TurnTakingAnalyzer initialized")
    
    def update(self, speaker_id: Optional[str], is_speaking: bool,
               is_question: bool = False,
               timestamp: Optional[float] = None) -> ConversationState:
        """
        Update with new speech state.
        
        Args:
            speaker_id: ID of the speaker (None if no one speaking)
            is_speaking: Whether the speaker is currently speaking
            is_question: Whether this is detected as a question
            timestamp: Optional timestamp
            
        Returns:
            Current conversation state
        """
        timestamp = timestamp or time.time()
        
        # Update momentum decay
        self._update_momentum(timestamp)
        
        if is_speaking and speaker_id:
            self._handle_speech_start(speaker_id, is_question, timestamp)
        elif not is_speaking and self._current_speaker:
            self._handle_speech_end(timestamp)
        
        return self.get_state(timestamp)
    
    def _handle_speech_start(self, speaker_id: str, 
                            is_question: bool, timestamp: float):
        """Handle speech starting."""
        # If this is a new speaker
        if speaker_id != self._current_speaker:
            # Calculate pause if there was silence
            if self._silence_start is not None:
                pause_duration = timestamp - self._silence_start
                pause_type = self._classify_pause(pause_duration)
                self._pause_history.append((pause_duration, pause_type))
            
            # Check if this is a response to a question
            response_to = None
            if self._pending_question:
                asker, ask_time = self._pending_question
                if asker != speaker_id:
                    response_time = timestamp - ask_time
                    self._question_response_times.append(response_time)
                    response_to = asker
                    
                    # Record response pattern
                    if asker not in self._response_patterns:
                        self._response_patterns[asker] = {}
                    if speaker_id not in self._response_patterns[asker]:
                        self._response_patterns[asker][speaker_id] = 0
                    self._response_patterns[asker][speaker_id] += 1
                    
                    # Boost momentum for response
                    self._momentum_score = min(1, self._momentum_score + 
                                               self.MOMENTUM_BOOST_RESPONSE)
                
                self._pending_question = None
            
            # Record turn event
            event = TurnEvent(
                timestamp=timestamp,
                speaker_id=speaker_id,
                event_type="start",
                is_question=is_question,
                response_to=response_to
            )
            self._turn_events.append(event)
            
            # Boost momentum for turn taking
            self._momentum_score = min(1, self._momentum_score + 
                                       self.MOMENTUM_BOOST_TURN)
            
            self._last_speaker = self._current_speaker
            self._current_speaker = speaker_id
            self._speech_start = timestamp
            self._silence_start = None
        
        # Track question
        if is_question:
            self._pending_question = (speaker_id, timestamp)
    
    def _handle_speech_end(self, timestamp: float):
        """Handle speech ending."""
        if self._current_speaker:
            # Calculate turn duration
            duration = timestamp - self._speech_start
            
            if self._current_speaker not in self._turn_durations:
                self._turn_durations[self._current_speaker] = []
            self._turn_durations[self._current_speaker].append(duration)
            
            # Record end event
            event = TurnEvent(
                timestamp=timestamp,
                speaker_id=self._current_speaker,
                event_type="end",
                duration=duration
            )
            self._turn_events.append(event)
            
            self._silence_start = timestamp
            self._current_speaker = None
    
    def _update_momentum(self, timestamp: float):
        """Update momentum based on time and activity."""
        elapsed = timestamp - self._last_update
        self._last_update = timestamp
        
        # Decay momentum during silence
        if self._silence_start is not None:
            silence_duration = timestamp - self._silence_start
            decay = self.MOMENTUM_DECAY_RATE * elapsed
            
            # Faster decay for longer silences
            if silence_duration > self.AWKWARD_PAUSE_MIN:
                decay *= 2
            
            self._momentum_score = max(0, self._momentum_score - decay)
        
        # Record momentum
        self._momentum_history.append((timestamp, self._momentum_score))
    
    def _classify_pause(self, duration: float) -> PauseType:
        """Classify a pause based on duration and context."""
        if duration <= self.NATURAL_PAUSE_MAX:
            return PauseType.NATURAL
        elif duration <= self.THOUGHTFUL_PAUSE_MAX:
            return PauseType.THOUGHTFUL
        else:
            return PauseType.AWKWARD
    
    def get_state(self, timestamp: Optional[float] = None) -> ConversationState:
        """Get current conversation state."""
        timestamp = timestamp or time.time()
        
        # Calculate average turn duration
        all_durations = []
        for durations in self._turn_durations.values():
            all_durations.extend(durations)
        avg_turn = np.mean(all_durations) if all_durations else 0.0
        
        # Calculate average pause duration
        pause_durations = [p[0] for p in self._pause_history]
        avg_pause = np.mean(pause_durations) if pause_durations else 0.0
        
        # Current silence
        silence_duration = 0.0
        if self._silence_start is not None:
            silence_duration = timestamp - self._silence_start
        
        # Determine momentum state
        if self._momentum_score > 0.7:
            momentum = ConversationMomentum.FLOWING
        elif self._momentum_score > 0.4:
            momentum = ConversationMomentum.BUILDING
        elif self._momentum_score > 0.15:
            momentum = ConversationMomentum.STALLING
        else:
            momentum = ConversationMomentum.DEAD
        
        return ConversationState(
            momentum=momentum,
            momentum_score=self._momentum_score,
            current_speaker=self._current_speaker,
            last_speaker=self._last_speaker,
            avg_turn_duration=avg_turn,
            avg_pause_duration=avg_pause,
            question_pending=self._pending_question is not None,
            silence_duration=silence_duration
        )
    
    def detect_question(self, pitch_contour: np.ndarray) -> bool:
        """
        Detect if speech is a question using pitch contour.
        
        Questions typically have rising intonation at the end.
        """
        if len(pitch_contour) < 10:
            return False
        
        # Look at the last 20% of the pitch contour
        end_portion = pitch_contour[-int(len(pitch_contour) * 0.2):]
        start_portion = pitch_contour[:int(len(pitch_contour) * 0.2)]
        
        # Check for rising intonation
        end_mean = np.mean(end_portion)
        start_mean = np.mean(start_portion)
        
        # Rising pitch at end suggests question
        return end_mean > start_mean * 1.1
    
    def get_response_patterns(self) -> Dict[str, Dict[str, int]]:
        """Get question-response patterns between speakers."""
        return self._response_patterns.copy()
    
    def get_avg_response_time(self) -> float:
        """Get average question response time."""
        if not self._question_response_times:
            return 0.0
        return np.mean(self._question_response_times)
    
    def get_pause_distribution(self) -> Dict[str, int]:
        """Get distribution of pause types."""
        distribution = {
            "natural": 0,
            "thoughtful": 0,
            "awkward": 0
        }
        
        for _, pause_type in self._pause_history:
            distribution[pause_type.value] += 1
        
        return distribution
    
    def get_momentum_history(self, window_seconds: float = 300) -> List[Tuple[float, float]]:
        """Get momentum history for plotting."""
        cutoff = time.time() - window_seconds
        return [(t, m) for t, m in self._momentum_history if t >= cutoff]
    
    def get_speaker_turn_stats(self, speaker_id: str) -> dict:
        """Get turn-taking statistics for a specific speaker."""
        durations = self._turn_durations.get(speaker_id, [])
        
        return {
            "total_turns": len(durations),
            "avg_duration": np.mean(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "total_time": sum(durations)
        }
