"""
Speaker Diarization - Who is speaking when.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from loguru import logger


@dataclass
class SpeakingSegment:
    """Represents a detected speaking segment."""
    track_id: int
    person_id: Optional[str]
    start_time: float
    end_time: float
    confidence: float
    audio_features: dict = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SpeakerStats:
    """Statistics for a detected speaker."""
    track_id: int
    person_id: Optional[str]
    total_speaking_time: float
    segment_count: int
    avg_confidence: float
    last_speaking: float


class SpeakerDiarization:
    """
    Speaker diarization: identify who is speaking when.
    Combines audio localization with visual tracking.
    """
    
    def __init__(self, 
                 distance_threshold: float = 1.5,
                 min_speaking_duration: float = 0.5,
                 energy_threshold: float = 0.1):
        """
        Initialize speaker diarization.
        
        Args:
            distance_threshold: Max distance to associate sound with track (m)
            min_speaking_duration: Minimum duration to count as speaking (s)
            energy_threshold: Minimum audio energy to consider as speech
        """
        self.distance_threshold = distance_threshold
        self.min_speaking_duration = min_speaking_duration
        self.energy_threshold = energy_threshold
        
        # Active speaking state
        self.active_speakers: dict[int, dict] = {}  # track_id -> state
        
        # Completed segments
        self.segments: list[SpeakingSegment] = []
        self.max_segments = 1000
        
        # Speaker statistics
        self.speaker_stats: dict[int, SpeakerStats] = {}
        
        # History for smoothing
        self.history_window = 10  # frames
        self.speaking_history: dict[int, deque] = {}
        
        logger.info("SpeakerDiarization initialized")
    
    def process(self, 
                tracks: list[dict],
                sound_position: np.ndarray = None,
                sound_confidence: float = 0,
                audio_energy: float = 0,
                timestamp: float = None) -> list[int]:
        """
        Process frame to identify speaking persons.
        
        Args:
            tracks: List of track dicts with 'id' and position info
            sound_position: Localized sound position [x, y, z]
            sound_confidence: Confidence of sound localization
            audio_energy: Current audio energy level
            timestamp: Current timestamp
            
        Returns:
            List of track IDs that are speaking
        """
        if timestamp is None:
            timestamp = time.time()
        
        speaking_tracks = []
        
        # Check if audio indicates speech
        has_speech = audio_energy > self.energy_threshold
        
        for track in tracks:
            track_id = track.get('id')
            if track_id is None:
                continue
            
            is_speaking = False
            confidence = 0.0
            
            # Calculate if this track is speaking
            if has_speech and sound_position is not None and sound_confidence > 0.3:
                # Get track position (from bbox center, simplified)
                bbox = track.get('bbox', [0, 0, 0, 0])
                track_x = (bbox[0] + bbox[2]) / 2
                track_y = (bbox[1] + bbox[3]) / 2
                
                # Map to world coordinates (simplified)
                world_x = (track_x / 640 - 0.5) * 10
                world_y = (track_y / 480 - 0.5) * 10
                
                # Calculate distance to sound
                distance = np.sqrt(
                    (world_x - sound_position[0])**2 +
                    (world_y - sound_position[1])**2
                )
                
                if distance < self.distance_threshold:
                    is_speaking = True
                    confidence = sound_confidence * (1 - distance / self.distance_threshold)
            
            # Apply temporal smoothing
            if track_id not in self.speaking_history:
                self.speaking_history[track_id] = deque(maxlen=self.history_window)
            
            self.speaking_history[track_id].append(is_speaking)
            
            # Majority vote for smoothing
            if len(self.speaking_history[track_id]) > 3:
                is_speaking = sum(self.speaking_history[track_id]) > len(self.speaking_history[track_id]) * 0.5
            
            # Update active speakers
            self._update_speaker_state(track_id, is_speaking, confidence, timestamp)
            
            if is_speaking:
                speaking_tracks.append(track_id)
        
        # Clean up old tracks
        self._cleanup_old_tracks(tracks, timestamp)
        
        return speaking_tracks
    
    def _update_speaker_state(self, track_id: int, is_speaking: bool,
                               confidence: float, timestamp: float):
        """Update speaker state and create segments."""
        if is_speaking:
            if track_id not in self.active_speakers:
                # Start new speaking segment
                self.active_speakers[track_id] = {
                    "start_time": timestamp,
                    "confidence_sum": confidence,
                    "confidence_count": 1
                }
            else:
                # Update ongoing segment
                self.active_speakers[track_id]["confidence_sum"] += confidence
                self.active_speakers[track_id]["confidence_count"] += 1
        else:
            if track_id in self.active_speakers:
                # End speaking segment
                state = self.active_speakers.pop(track_id)
                duration = timestamp - state["start_time"]
                
                if duration >= self.min_speaking_duration:
                    avg_conf = state["confidence_sum"] / max(state["confidence_count"], 1)
                    
                    segment = SpeakingSegment(
                        track_id=track_id,
                        person_id=None,  # Can be filled by face recognition
                        start_time=state["start_time"],
                        end_time=timestamp,
                        confidence=avg_conf
                    )
                    
                    self.segments.append(segment)
                    if len(self.segments) > self.max_segments:
                        self.segments = self.segments[-self.max_segments:]
                    
                    # Update stats
                    self._update_stats(track_id, segment)
    
    def _update_stats(self, track_id: int, segment: SpeakingSegment):
        """Update speaker statistics."""
        if track_id not in self.speaker_stats:
            self.speaker_stats[track_id] = SpeakerStats(
                track_id=track_id,
                person_id=None,
                total_speaking_time=0,
                segment_count=0,
                avg_confidence=0,
                last_speaking=0
            )
        
        stats = self.speaker_stats[track_id]
        stats.total_speaking_time += segment.duration
        stats.segment_count += 1
        stats.avg_confidence = (
            (stats.avg_confidence * (stats.segment_count - 1) + segment.confidence)
            / stats.segment_count
        )
        stats.last_speaking = segment.end_time
    
    def _cleanup_old_tracks(self, current_tracks: list[dict], timestamp: float):
        """Clean up state for tracks that are no longer visible."""
        current_ids = {t.get('id') for t in current_tracks}
        
        # End any active speaking for gone tracks
        gone_tracks = set(self.active_speakers.keys()) - current_ids
        for track_id in gone_tracks:
            self._update_speaker_state(track_id, False, 0, timestamp)
        
        # Clean up history
        for track_id in list(self.speaking_history.keys()):
            if track_id not in current_ids:
                del self.speaking_history[track_id]
    
    def get_speaking_tracks(self) -> list[int]:
        """Get currently speaking track IDs."""
        return list(self.active_speakers.keys())
    
    def get_recent_segments(self, count: int = 50) -> list[SpeakingSegment]:
        """Get recent speaking segments."""
        return self.segments[-count:]
    
    def get_speaker_stats(self) -> list[SpeakerStats]:
        """Get statistics for all speakers."""
        return list(self.speaker_stats.values())
    
    def get_top_speakers(self, n: int = 5) -> list[SpeakerStats]:
        """Get top n speakers by speaking time."""
        return sorted(
            self.speaker_stats.values(),
            key=lambda s: s.total_speaking_time,
            reverse=True
        )[:n]
    
    def get_speaking_timeline(self, start_time: float, 
                              end_time: float) -> dict[int, list[tuple]]:
        """
        Get speaking timeline for a time range.
        
        Returns: {track_id: [(start, end), ...]}
        """
        timeline = {}
        
        for segment in self.segments:
            if segment.end_time < start_time or segment.start_time > end_time:
                continue
            
            if segment.track_id not in timeline:
                timeline[segment.track_id] = []
            
            timeline[segment.track_id].append(
                (max(segment.start_time, start_time),
                 min(segment.end_time, end_time))
            )
        
        return timeline
