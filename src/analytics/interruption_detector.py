"""
Interruption Detection - Detect and classify speech interruptions.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
from loguru import logger


class InterruptionType(Enum):
    """Types of interruptions."""
    COLLABORATIVE = "collaborative"  # Supportive, building on speaker
    COMPETITIVE = "competitive"  # Takeover, dismissive
    CLARIFICATION = "clarification"  # Quick question
    BACKCHANNEL = "backchannel"  # "mm-hmm", "yeah", etc.


@dataclass
class Interruption:
    """An interruption event."""
    timestamp: float
    interrupter_id: str
    interrupted_id: str
    type: InterruptionType
    duration: float  # How long the overlap lasted
    confidence: float
    visual_cues: Dict[str, bool]  # forward_lean, hand_raise, etc.


class InterruptionDetector:
    """
    Detects and classifies interruptions in conversations.
    
    Uses acoustic features (overlap detection, prosody) and 
    visual cues (body language) to classify interruption types.
    """
    
    def __init__(self, 
                 overlap_threshold: float = 0.3,
                 backchannel_max_duration: float = 1.0):
        """
        Args:
            overlap_threshold: Minimum overlap duration (seconds) to count
            backchannel_max_duration: Max duration for backchannel classification
        """
        self.overlap_threshold = overlap_threshold
        self.backchannel_max_duration = backchannel_max_duration
        
        # State tracking
        self._active_speakers: Dict[str, float] = {}  # speaker_id -> start_time
        self._recent_interruptions: deque = deque(maxlen=100)
        
        # Body language state
        self._body_states: Dict[str, Dict[str, bool]] = {}
        
        # Statistics
        self._interruption_counts: Dict[str, Dict[str, int]] = {}
        
        logger.info("InterruptionDetector initialized")
    
    def update_speaker_state(self, speaker_id: str, is_speaking: bool,
                             timestamp: Optional[float] = None) -> Optional[Interruption]:
        """
        Update speaker state and detect interruptions.
        
        Returns:
            Interruption if one was detected, None otherwise
        """
        timestamp = timestamp or time.time()
        interruption = None
        
        if is_speaking:
            # Check if someone else is already speaking
            for other_id, start_time in self._active_speakers.items():
                if other_id != speaker_id:
                    # Potential interruption detected
                    interruption = self._classify_interruption(
                        interrupter_id=speaker_id,
                        interrupted_id=other_id,
                        timestamp=timestamp
                    )
                    self._record_interruption(interruption)
                    break
            
            # Mark as active speaker
            self._active_speakers[speaker_id] = timestamp
        else:
            # Speaker stopped
            if speaker_id in self._active_speakers:
                del self._active_speakers[speaker_id]
        
        return interruption
    
    def update_body_language(self, speaker_id: str, 
                             forward_lean: bool = False,
                             hand_raised: bool = False,
                             nodding: bool = False,
                             head_shake: bool = False,
                             aggressive_posture: bool = False):
        """Update body language state for a speaker."""
        self._body_states[speaker_id] = {
            "forward_lean": forward_lean,
            "hand_raised": hand_raised,
            "nodding": nodding,
            "head_shake": head_shake,
            "aggressive_posture": aggressive_posture
        }
    
    def _classify_interruption(self, interrupter_id: str, 
                               interrupted_id: str,
                               timestamp: float) -> Interruption:
        """
        Classify the type of interruption.
        
        Uses heuristics based on:
        - Duration of overlap
        - Body language cues
        - Speaking pattern
        """
        # Get body language cues
        body_cues = self._body_states.get(interrupter_id, {})
        
        # Default classification
        interrupt_type = InterruptionType.COMPETITIVE
        confidence = 0.5
        
        # Classification rules
        if body_cues.get("nodding", False):
            # Nodding while interrupting = collaborative
            interrupt_type = InterruptionType.COLLABORATIVE
            confidence = 0.7
        elif body_cues.get("hand_raised", False):
            # Hand raised = clarification attempt
            interrupt_type = InterruptionType.CLARIFICATION
            confidence = 0.75
        elif body_cues.get("aggressive_posture", False):
            # Aggressive posture = competitive
            interrupt_type = InterruptionType.COMPETITIVE
            confidence = 0.8
        elif body_cues.get("forward_lean", False):
            # Forward lean can be collaborative or competitive
            # Default to collaborative with lower confidence
            interrupt_type = InterruptionType.COLLABORATIVE
            confidence = 0.55
        
        return Interruption(
            timestamp=timestamp,
            interrupter_id=interrupter_id,
            interrupted_id=interrupted_id,
            type=interrupt_type,
            duration=0.0,  # Will be updated when overlap ends
            confidence=confidence,
            visual_cues=body_cues.copy()
        )
    
    def update_overlap_duration(self, interruption: Interruption, 
                                duration: float):
        """Update the overlap duration for an interruption."""
        interruption.duration = duration
        
        # Reclassify as backchannel if very short
        if duration < self.backchannel_max_duration:
            # Check if it seems like a backchannel
            if not interruption.visual_cues.get("aggressive_posture", False):
                interruption.type = InterruptionType.BACKCHANNEL
                interruption.confidence = 0.6
    
    def _record_interruption(self, interruption: Interruption):
        """Record an interruption for statistics."""
        self._recent_interruptions.append(interruption)
        
        # Update counts
        interrupter = interruption.interrupter_id
        if interrupter not in self._interruption_counts:
            self._interruption_counts[interrupter] = {
                "collaborative": 0,
                "competitive": 0,
                "clarification": 0,
                "backchannel": 0
            }
        
        self._interruption_counts[interrupter][interruption.type.value] += 1
    
    def get_interruption_stats(self, speaker_id: Optional[str] = None) -> dict:
        """
        Get interruption statistics.
        
        Args:
            speaker_id: If provided, get stats for specific speaker
            
        Returns:
            Dictionary of interruption statistics
        """
        if speaker_id:
            return self._interruption_counts.get(speaker_id, {})
        
        # Aggregate stats
        total = {
            "collaborative": 0,
            "competitive": 0,
            "clarification": 0,
            "backchannel": 0
        }
        
        for counts in self._interruption_counts.values():
            for type_name, count in counts.items():
                total[type_name] += count
        
        return total
    
    def get_interruption_patterns(self) -> Dict[str, Dict[str, int]]:
        """
        Get interruption patterns between speakers.
        
        Returns:
            Dict[interrupter_id, Dict[interrupted_id, count]]
        """
        patterns: Dict[str, Dict[str, int]] = {}
        
        for interruption in self._recent_interruptions:
            interrupter = interruption.interrupter_id
            interrupted = interruption.interrupted_id
            
            if interrupter not in patterns:
                patterns[interrupter] = {}
            
            if interrupted not in patterns[interrupter]:
                patterns[interrupter][interrupted] = 0
            
            patterns[interrupter][interrupted] += 1
        
        return patterns
    
    def get_recent_interruptions(self, count: int = 10) -> List[Interruption]:
        """Get most recent interruptions."""
        return list(self._recent_interruptions)[-count:]
    
    def get_timeline_data(self) -> List[dict]:
        """Get interruption data for timeline visualization."""
        return [
            {
                "timestamp": i.timestamp,
                "interrupter": i.interrupter_id,
                "interrupted": i.interrupted_id,
                "type": i.type.value,
                "duration": i.duration,
                "confidence": i.confidence
            }
            for i in self._recent_interruptions
        ]
