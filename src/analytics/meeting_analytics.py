"""
Meeting Analytics - Comprehensive conversation intelligence and speaker analysis.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from enum import Enum
from loguru import logger


@dataclass
class ParticipantStats:
    """Statistics for a meeting participant."""
    participant_id: str
    name: str = ""
    
    # Talk time
    total_talk_time: float = 0.0
    talk_segments: int = 0
    current_segment_start: Optional[float] = None
    
    # Interruptions
    interruptions_made: int = 0
    interruptions_received: int = 0
    
    # Turn taking
    turns_taken: int = 0
    avg_turn_duration: float = 0.0
    questions_asked: int = 0
    questions_answered: int = 0
    
    # Response patterns
    avg_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    
    # Engagement
    attention_score: float = 1.0
    sentiment_avg: float = 0.0
    
    def get_talk_percentage(self, total_meeting_time: float) -> float:
        """Get percentage of meeting time this participant spoke."""
        if total_meeting_time <= 0:
            return 0.0
        return (self.total_talk_time / total_meeting_time) * 100


@dataclass
class MeetingSegment:
    """A topic segment within the meeting."""
    segment_id: int
    start_time: float
    end_time: Optional[float] = None
    topic: str = "Unknown"
    primary_speaker: Optional[str] = None
    participant_talk_times: Dict[str, float] = field(default_factory=dict)
    avg_sentiment: float = 0.0
    key_moments: List[Tuple[float, str]] = field(default_factory=list)


class MeetingAnalytics:
    """
    Comprehensive meeting analytics engine.
    
    Tracks conversation dynamics, talk-time distribution,
    participation balance, and generates real-time insights.
    """
    
    def __init__(self, balance_threshold: float = 0.3):
        """
        Args:
            balance_threshold: Threshold for flagging dominated meetings
                              (0-1, lower = less tolerance for imbalance)
        """
        self.balance_threshold = balance_threshold
        
        # Meeting state
        self.meeting_start_time: Optional[float] = None
        self.meeting_active = False
        
        # Participants
        self.participants: Dict[str, ParticipantStats] = {}
        
        # Segments
        self.segments: List[MeetingSegment] = []
        self.current_segment: Optional[MeetingSegment] = None
        
        # Real-time tracking
        self._current_speaker: Optional[str] = None
        self._last_speaker: Optional[str] = None
        self._last_speech_end: float = 0.0
        self._speech_history: deque = deque(maxlen=1000)
        
        # Talk time history for trends
        self._talk_time_history: List[Dict[str, float]] = []
        self._history_interval = 30.0  # Record every 30 seconds
        self._last_history_time = 0.0
        
        # Silence tracking
        self._silence_start: Optional[float] = None
        self._silences: List[Tuple[float, float, str]] = []  # start, duration, type
        
        logger.info("MeetingAnalytics initialized")
    
    def start_meeting(self):
        """Start a new meeting session."""
        self.meeting_start_time = time.time()
        self.meeting_active = True
        self.participants.clear()
        self.segments.clear()
        
        # Start first segment
        self.current_segment = MeetingSegment(
            segment_id=0,
            start_time=self.meeting_start_time,
            topic="Opening"
        )
        self.segments.append(self.current_segment)
        
        logger.info("Meeting started")
    
    def end_meeting(self) -> dict:
        """End the meeting and generate summary."""
        if not self.meeting_active:
            return {}
        
        self.meeting_active = False
        end_time = time.time()
        
        # Close current segment
        if self.current_segment:
            self.current_segment.end_time = end_time
        
        # Close any active speaker
        if self._current_speaker:
            self._end_speech(self._current_speaker, end_time)
        
        duration = end_time - self.meeting_start_time
        
        summary = {
            "duration": duration,
            "participants": len(self.participants),
            "segments": len(self.segments),
            "balance_score": self.calculate_balance_score(),
            "talk_distribution": self.get_talk_distribution(),
            "top_speakers": self.get_top_speakers(3),
            "total_interruptions": sum(p.interruptions_made for p in self.participants.values()),
            "quiet_periods": len([s for s in self._silences if s[2] == "awkward"])
        }
        
        logger.info(f"Meeting ended: {duration:.1f}s, {len(self.participants)} participants")
        return summary
    
    def register_participant(self, participant_id: str, name: str = ""):
        """Register a participant in the meeting."""
        if participant_id not in self.participants:
            self.participants[participant_id] = ParticipantStats(
                participant_id=participant_id,
                name=name or f"Participant {len(self.participants) + 1}"
            )
            logger.debug(f"Registered participant: {participant_id}")
    
    def update_speech(self, participant_id: str, is_speaking: bool, 
                      timestamp: Optional[float] = None):
        """
        Update speech status for a participant.
        
        Args:
            participant_id: ID of the speaker
            is_speaking: Whether they are currently speaking
            timestamp: Optional timestamp (uses current time if not provided)
        """
        if not self.meeting_active:
            return
        
        timestamp = timestamp or time.time()
        
        # Ensure participant is registered
        self.register_participant(participant_id)
        participant = self.participants[participant_id]
        
        if is_speaking:
            self._start_speech(participant_id, timestamp)
        else:
            self._end_speech(participant_id, timestamp)
        
        # Update history periodically
        if timestamp - self._last_history_time >= self._history_interval:
            self._record_history(timestamp)
            self._last_history_time = timestamp
    
    def _start_speech(self, participant_id: str, timestamp: float):
        """Handle speech start."""
        participant = self.participants[participant_id]
        
        # Check for interruption
        if self._current_speaker and self._current_speaker != participant_id:
            self._record_interruption(participant_id, self._current_speaker, timestamp)
        
        # Start segment timing
        if participant.current_segment_start is None:
            participant.current_segment_start = timestamp
            participant.talk_segments += 1
            
            # Record turn
            if self._last_speaker != participant_id:
                participant.turns_taken += 1
                
                # Calculate response time
                if self._last_speech_end > 0:
                    response_time = timestamp - self._last_speech_end
                    if response_time < 5.0:  # Within 5 seconds = response
                        participant.response_times.append(response_time)
                        participant.avg_response_time = np.mean(participant.response_times)
        
        # Handle silence end
        if self._silence_start is not None:
            silence_duration = timestamp - self._silence_start
            silence_type = "thoughtful" if silence_duration < 3.0 else "awkward"
            self._silences.append((self._silence_start, silence_duration, silence_type))
            self._silence_start = None
        
        self._current_speaker = participant_id
        
        # Record in history
        self._speech_history.append({
            "participant": participant_id,
            "type": "start",
            "timestamp": timestamp
        })
    
    def _end_speech(self, participant_id: str, timestamp: float):
        """Handle speech end."""
        participant = self.participants.get(participant_id)
        if not participant:
            return
        
        if participant.current_segment_start is not None:
            duration = timestamp - participant.current_segment_start
            participant.total_talk_time += duration
            
            # Update average turn duration
            if participant.turns_taken > 0:
                participant.avg_turn_duration = (
                    participant.total_talk_time / participant.turns_taken
                )
            
            participant.current_segment_start = None
            
            # Update segment
            if self.current_segment:
                if participant_id not in self.current_segment.participant_talk_times:
                    self.current_segment.participant_talk_times[participant_id] = 0
                self.current_segment.participant_talk_times[participant_id] += duration
        
        self._last_speaker = participant_id
        self._last_speech_end = timestamp
        
        if self._current_speaker == participant_id:
            self._current_speaker = None
            self._silence_start = timestamp
        
        self._speech_history.append({
            "participant": participant_id,
            "type": "end",
            "timestamp": timestamp
        })
    
    def _record_interruption(self, interrupter_id: str, 
                             interrupted_id: str, timestamp: float):
        """Record an interruption event."""
        self.participants[interrupter_id].interruptions_made += 1
        self.participants[interrupted_id].interruptions_received += 1
        
        logger.debug(f"Interruption: {interrupter_id} -> {interrupted_id}")
    
    def _record_history(self, timestamp: float):
        """Record current state for historical trends."""
        distribution = {
            pid: p.total_talk_time 
            for pid, p in self.participants.items()
        }
        self._talk_time_history.append(distribution)
    
    def record_question(self, participant_id: str, timestamp: float = None):
        """Record that a participant asked a question."""
        if participant_id in self.participants:
            self.participants[participant_id].questions_asked += 1
    
    def record_answer(self, participant_id: str, question_asker: str = None):
        """Record that a participant answered a question."""
        if participant_id in self.participants:
            self.participants[participant_id].questions_answered += 1
    
    def start_new_segment(self, topic: str = "Unknown"):
        """Start a new topic segment."""
        timestamp = time.time()
        
        if self.current_segment:
            self.current_segment.end_time = timestamp
        
        self.current_segment = MeetingSegment(
            segment_id=len(self.segments),
            start_time=timestamp,
            topic=topic
        )
        self.segments.append(self.current_segment)
        
        logger.debug(f"New segment: {topic}")
    
    def calculate_balance_score(self) -> float:
        """
        Calculate how balanced the conversation is.
        
        Returns:
            Score from 0 (completely unbalanced) to 1 (perfectly balanced)
        """
        if len(self.participants) < 2:
            return 1.0
        
        talk_times = [p.total_talk_time for p in self.participants.values()]
        total_time = sum(talk_times)
        
        if total_time == 0:
            return 1.0
        
        # Calculate normalized distribution
        distribution = [t / total_time for t in talk_times]
        
        # Perfect balance would be 1/n for each participant
        n = len(self.participants)
        perfect = 1.0 / n
        
        # Calculate deviation from perfect
        deviations = [abs(d - perfect) for d in distribution]
        max_deviation = 1 - perfect  # Maximum possible deviation
        
        avg_deviation = np.mean(deviations)
        score = 1 - (avg_deviation / max_deviation)
        
        return max(0, min(1, score))
    
    def is_meeting_dominated(self) -> Tuple[bool, Optional[str]]:
        """
        Check if meeting is dominated by a single speaker.
        
        Returns:
            Tuple of (is_dominated, dominating_participant_id)
        """
        if len(self.participants) < 2:
            return False, None
        
        total_time = sum(p.total_talk_time for p in self.participants.values())
        if total_time == 0:
            return False, None
        
        for pid, participant in self.participants.items():
            percentage = participant.total_talk_time / total_time
            if percentage > (1 - self.balance_threshold):
                return True, pid
        
        return False, None
    
    def get_talk_distribution(self) -> Dict[str, float]:
        """Get talk time percentage for each participant."""
        total_time = sum(p.total_talk_time for p in self.participants.values())
        
        if total_time == 0:
            return {}
        
        return {
            pid: (p.total_talk_time / total_time) * 100
            for pid, p in self.participants.items()
        }
    
    def get_top_speakers(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N speakers by talk time."""
        sorted_participants = sorted(
            self.participants.items(),
            key=lambda x: x[1].total_talk_time,
            reverse=True
        )
        
        return [(pid, p.total_talk_time) for pid, p in sorted_participants[:n]]
    
    def get_talk_time_trends(self) -> Dict[str, List[float]]:
        """Get historical talk time trends for each participant."""
        trends: Dict[str, List[float]] = defaultdict(list)
        
        for snapshot in self._talk_time_history:
            for pid in self.participants:
                trends[pid].append(snapshot.get(pid, 0))
        
        return dict(trends)
    
    def get_realtime_data(self) -> dict:
        """Get current analytics data for real-time display."""
        meeting_duration = 0
        if self.meeting_start_time:
            meeting_duration = time.time() - self.meeting_start_time
        
        return {
            "meeting_duration": meeting_duration,
            "current_speaker": self._current_speaker,
            "participants": {
                pid: {
                    "name": p.name,
                    "talk_time": p.total_talk_time,
                    "talk_percentage": p.get_talk_percentage(meeting_duration),
                    "turns": p.turns_taken,
                    "interruptions_made": p.interruptions_made,
                    "interruptions_received": p.interruptions_received,
                    "avg_response_time": p.avg_response_time,
                    "is_speaking": p.current_segment_start is not None
                }
                for pid, p in self.participants.items()
            },
            "balance_score": self.calculate_balance_score(),
            "is_dominated": self.is_meeting_dominated()[0],
            "current_segment": self.current_segment.topic if self.current_segment else None,
            "segment_count": len(self.segments)
        }
    
    def get_pie_chart_data(self) -> List[dict]:
        """Get data formatted for pie chart display."""
        distribution = self.get_talk_distribution()
        
        colors = [
            "#58a6ff", "#3fb950", "#f778ba", "#d29922", 
            "#a371f7", "#f85149", "#56d4dd", "#8b949e"
        ]
        
        data = []
        for i, (pid, percentage) in enumerate(distribution.items()):
            participant = self.participants[pid]
            data.append({
                "id": pid,
                "name": participant.name,
                "value": percentage,
                "color": colors[i % len(colors)]
            })
        
        return data
