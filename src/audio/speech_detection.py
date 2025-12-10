"""
Speech Detection - Detect and classify speech using audio features.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from collections import deque
from loguru import logger

try:
    from scipy import signal
    from scipy.fft import fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class SpeechSegment:
    """Detected speech segment."""
    start_time: float
    end_time: float
    energy: float
    frequency: float
    is_male: bool
    confidence: float


class SpeechDetector:
    """
    Detects speech in audio streams using spectral analysis.
    """
    
    # Typical speech frequency ranges
    SPEECH_LOW_FREQ = 85  # Hz
    SPEECH_HIGH_FREQ = 255  # Hz
    MALE_RANGE = (85, 155)
    FEMALE_RANGE = (165, 255)
    
    def __init__(self, sample_rate: int = 48000,
                 frame_size: int = 2048,
                 energy_threshold: float = 0.01,
                 min_speech_duration: float = 0.2):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.energy_threshold = energy_threshold
        self.min_speech_duration = min_speech_duration
        
        # Buffers
        self._audio_buffer = deque(maxlen=int(sample_rate * 2))  # 2 sec
        self._speech_active = False
        self._speech_start = 0.0
        self._speech_history: List[SpeechSegment] = []
        
        # Smoothing
        self._energy_history = deque(maxlen=10)
        self._pitch_history = deque(maxlen=10)
        
        logger.info(f"SpeechDetector initialized: {sample_rate}Hz, {frame_size} samples")
    
    def process(self, audio: np.ndarray, timestamp: float = 0) -> dict:
        """
        Process audio frame for speech detection.
        
        Args:
            audio: Audio samples (mono)
            timestamp: Current timestamp
            
        Returns:
            Dict with:
                - has_speech: bool
                - energy: float
                - pitch: float  
                - is_male: bool
                - confidence: float
        """
        result = {
            'has_speech': False,
            'energy': 0.0,
            'pitch': 0.0,
            'is_male': True,
            'confidence': 0.0
        }
        
        if audio is None or len(audio) < self.frame_size:
            return result
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Calculate energy
        energy = np.sqrt(np.mean(audio ** 2))
        result['energy'] = float(energy)
        self._energy_history.append(energy)
        
        # Check if above threshold
        avg_energy = np.mean(list(self._energy_history))
        
        if avg_energy < self.energy_threshold:
            self._handle_speech_end(timestamp)
            return result
        
        # Spectral analysis
        if SCIPY_AVAILABLE:
            pitch, confidence = self._estimate_pitch(audio)
        else:
            pitch, confidence = self._estimate_pitch_simple(audio)
        
        result['pitch'] = float(pitch)
        result['confidence'] = float(confidence)
        self._pitch_history.append(pitch)
        
        # Check if in speech frequency range
        if self.SPEECH_LOW_FREQ <= pitch <= self.SPEECH_HIGH_FREQ:
            result['has_speech'] = True
            
            # Gender classification
            if self.MALE_RANGE[0] <= pitch <= self.MALE_RANGE[1]:
                result['is_male'] = True
            elif self.FEMALE_RANGE[0] <= pitch <= self.FEMALE_RANGE[1]:
                result['is_male'] = False
            
            self._handle_speech_start(timestamp)
        else:
            self._handle_speech_end(timestamp)
        
        return result
    
    def _estimate_pitch(self, audio: np.ndarray) -> tuple:
        """Estimate pitch using autocorrelation."""
        if not SCIPY_AVAILABLE:
            return self._estimate_pitch_simple(audio)
        
        # Autocorrelation
        corr = signal.correlate(audio, audio, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find first peak after zero crossing
        d = np.diff(corr)
        
        # Find zero crossings
        starts = np.where(d > 0)[0]
        if len(starts) == 0:
            return 0.0, 0.0
        
        start = starts[0]
        peak_idx = start + np.argmax(corr[start:start + self.sample_rate // 50])
        
        if peak_idx == 0:
            return 0.0, 0.0
        
        pitch = self.sample_rate / peak_idx
        confidence = corr[peak_idx] / corr[0] if corr[0] > 0 else 0
        
        return pitch, confidence
    
    def _estimate_pitch_simple(self, audio: np.ndarray) -> tuple:
        """Simple pitch estimation using zero crossings."""
        crossings = np.where(np.diff(np.sign(audio)))[0]
        
        if len(crossings) < 2:
            return 0.0, 0.0
        
        # Average period
        periods = np.diff(crossings)
        avg_period = np.mean(periods)
        
        if avg_period == 0:
            return 0.0, 0.0
        
        pitch = self.sample_rate / (2 * avg_period)
        confidence = 1.0 - (np.std(periods) / avg_period) if avg_period > 0 else 0
        
        return pitch, max(0, min(1, confidence))
    
    def _handle_speech_start(self, timestamp: float):
        """Handle speech start."""
        if not self._speech_active:
            self._speech_active = True
            self._speech_start = timestamp
    
    def _handle_speech_end(self, timestamp: float):
        """Handle speech end."""
        if self._speech_active:
            duration = timestamp - self._speech_start
            
            if duration >= self.min_speech_duration:
                avg_pitch = np.mean(list(self._pitch_history)) if self._pitch_history else 0
                avg_energy = np.mean(list(self._energy_history)) if self._energy_history else 0
                
                segment = SpeechSegment(
                    start_time=self._speech_start,
                    end_time=timestamp,
                    energy=avg_energy,
                    frequency=avg_pitch,
                    is_male=avg_pitch < 160,
                    confidence=0.7
                )
                
                self._speech_history.append(segment)
                if len(self._speech_history) > 100:
                    self._speech_history = self._speech_history[-100:]
            
            self._speech_active = False
    
    def get_recent_segments(self, count: int = 10) -> List[SpeechSegment]:
        """Get recent speech segments."""
        return self._speech_history[-count:]
    
    def get_total_speech_time(self) -> float:
        """Get total speech time from all segments."""
        return sum(s.end_time - s.start_time for s in self._speech_history)
    
    @property
    def is_speech_active(self) -> bool:
        return self._speech_active


class VoiceActivityDetector:
    """
    Simple Voice Activity Detection using energy and zero-crossing rate.
    """
    
    def __init__(self, sample_rate: int = 48000,
                 frame_duration_ms: int = 30):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Adaptive threshold
        self._energy_threshold = 0.01
        self._zcr_threshold = 0.1
        self._noise_floor = 0.001
        
        # History for adaptation
        self._energy_history = deque(maxlen=100)
        
        logger.info("VoiceActivityDetector initialized")
    
    def process(self, audio: np.ndarray) -> dict:
        """
        Detect voice activity in audio frame.
        
        Returns:
            Dict with:
                - is_voice: bool
                - energy: float
                - zcr: float
                - snr: float
        """
        if audio is None or len(audio) == 0:
            return {'is_voice': False, 'energy': 0, 'zcr': 0, 'snr': 0}
        
        # Calculate energy
        energy = np.sqrt(np.mean(audio ** 2))
        
        # Calculate zero-crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2
        
        # Update history
        self._energy_history.append(energy)
        
        # Adaptive threshold
        if len(self._energy_history) > 50:
            sorted_energy = sorted(self._energy_history)
            self._noise_floor = np.mean(sorted_energy[:10])
            self._energy_threshold = self._noise_floor * 3
        
        # SNR
        snr = 10 * np.log10(energy / max(self._noise_floor, 1e-10))
        
        # Voice detection
        is_voice = (energy > self._energy_threshold and
                   zcr < self._zcr_threshold and
                   snr > 3)
        
        return {
            'is_voice': is_voice,
            'energy': float(energy),
            'zcr': float(zcr),
            'snr': float(snr)
        }
