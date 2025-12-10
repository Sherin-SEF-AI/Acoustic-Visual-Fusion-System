"""
Audio Event Detection Module for the Acoustic-Visual Fusion System.

Implements neural network-based sound event detection and classification.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, event detection will be limited")


class AudioEventClass(Enum):
    """Audio event classification categories."""
    SPEECH = "speech"
    GLASS_BREAKING = "glass_breaking"
    DOOR = "door"
    FOOTSTEPS = "footsteps"
    ALARM = "alarm"
    APPLAUSE = "applause"
    MECHANICAL = "mechanical"
    MUSIC = "music"
    PHONE_RINGING = "phone_ringing"
    KNOCK = "knock"
    COUGH = "cough"
    SILENCE = "silence"
    UNKNOWN = "unknown"


@dataclass
class AudioEvent:
    """Represents a detected audio event."""
    event_class: AudioEventClass
    confidence: float  # 0.0 to 1.0
    timestamp: float  # Start timestamp
    duration: float  # Duration in seconds
    intensity_db: float  # RMS intensity in dB
    features: Optional[np.ndarray] = None  # Audio features
    class_probabilities: dict[str, float] = field(default_factory=dict)
    
    @property
    def end_timestamp(self) -> float:
        """End timestamp of the event."""
        return self.timestamp + self.duration
    
    def __repr__(self) -> str:
        return (
            f"AudioEvent({self.event_class.value}, "
            f"conf={self.confidence:.2f}, "
            f"dur={self.duration:.2f}s, "
            f"int={self.intensity_db:.1f}dB)"
        )


class MelSpectrogramExtractor:
    """Extracts mel spectrograms for audio classification."""
    
    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 20.0,
        f_max: float = 8000.0
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        
        if TORCH_AVAILABLE:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio.
        
        Args:
            audio: Audio samples (1D array)
            
        Returns:
            Mel spectrogram (n_mels x time_frames)
        """
        if TORCH_AVAILABLE:
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Mix to mono
            
            audio_tensor = torch.from_numpy(audio.astype(np.float32))
            mel_spec = self.mel_transform(audio_tensor)
            mel_spec_db = self.amplitude_to_db(mel_spec)
            return mel_spec_db.numpy()
        else:
            # Fallback: simple magnitude spectrogram
            from scipy import signal
            f, t, Sxx = signal.spectrogram(
                audio,
                fs=self.sample_rate,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length
            )
            return 10 * np.log10(Sxx + 1e-10)


if TORCH_AVAILABLE:
    class SimpleAudioClassifier(nn.Module):
        """Simple CNN-based audio classifier."""
        
        def __init__(self, num_classes: int = 12, n_mels: int = 128):
            super().__init__()
            
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                x = x.unsqueeze(1)  # Add channel dim
            x = self.conv_layers(x)
            x = self.classifier(x)
            return x
else:
    SimpleAudioClassifier = None


class AudioEventDetector:
    """
    Detects and classifies audio events from audio streams.
    
    Uses mel spectrogram features and a neural network classifier.
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        window_size: float = 1.0,
        hop_size: float = 0.5,
        confidence_threshold: float = 0.5,
        min_event_duration: float = 0.1,
        device: str = "auto"
    ):
        """
        Initialize audio event detector.
        
        Args:
            sample_rate: Audio sample rate
            window_size: Analysis window in seconds
            hop_size: Analysis hop in seconds
            confidence_threshold: Minimum confidence for detection
            min_event_duration: Minimum event duration in seconds
            device: Device for inference ("auto", "cuda", "cpu")
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.confidence_threshold = confidence_threshold
        self.min_event_duration = min_event_duration
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize components
        self.mel_extractor = MelSpectrogramExtractor(sample_rate=sample_rate)
        
        # Event class mapping
        self.class_names = [
            "speech", "glass_breaking", "door", "footsteps",
            "alarm", "applause", "mechanical", "music",
            "phone_ringing", "knock", "cough", "silence"
        ]
        self.class_to_enum = {
            name: AudioEventClass(name) if name in [e.value for e in AudioEventClass] 
            else AudioEventClass.UNKNOWN
            for name in self.class_names
        }
        
        # Initialize model
        self.model = None
        if TORCH_AVAILABLE:
            self._init_model()
        
        # Adaptive threshold state
        self.ambient_rms = 0.01
        self.ambient_alpha = 0.01
        
        # Event tracking
        self.current_event: Optional[AudioEvent] = None
        self.events_buffer: list[AudioEvent] = []
        
        logger.info(f"AudioEventDetector initialized on {self.device}")
    
    def _init_model(self) -> None:
        """Initialize the classification model."""
        if SimpleAudioClassifier is None:
            logger.warning("Audio classifier not available (PyTorch missing)")
            return
        
        self.model = SimpleAudioClassifier(num_classes=len(self.class_names))
        self.model.to(self.device)
        self.model.eval()
        
        # In production, load pretrained weights here
        logger.info("Audio classifier model initialized (random weights)")
    
    def compute_rms(self, audio: np.ndarray) -> float:
        """Compute RMS of audio signal."""
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def compute_db(self, rms: float, reference: float = 1.0) -> float:
        """Convert RMS to dB."""
        return 20 * np.log10(max(rms, 1e-10) / reference)
    
    def update_ambient(self, rms: float) -> None:
        """Update ambient noise estimate."""
        self.ambient_rms = (
            self.ambient_alpha * rms +
            (1 - self.ambient_alpha) * self.ambient_rms
        )
    
    def is_active(self, rms: float) -> bool:
        """Check if audio is above ambient threshold."""
        return rms > self.ambient_rms * 2.0
    
    def classify(self, audio: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """
        Classify audio segment.
        
        Args:
            audio: Audio samples
            
        Returns:
            Tuple of (class_name, confidence, class_probabilities)
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback: simple energy-based detection
            rms = self.compute_rms(audio)
            if rms > self.ambient_rms * 3:
                return "speech", 0.5, {"speech": 0.5, "unknown": 0.5}
            return "silence", 0.9, {"silence": 0.9}
        
        # Extract mel spectrogram
        mel_spec = self.mel_extractor.extract(audio)
        
        # Prepare input
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(mel_tensor)
            probs = torch.softmax(logits, dim=1)
        
        probs_np = probs.cpu().numpy()[0]
        class_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[class_idx])
        
        class_name = self.class_names[class_idx]
        class_probs = {
            name: float(probs_np[i])
            for i, name in enumerate(self.class_names)
        }
        
        return class_name, confidence, class_probs
    
    def detect(
        self,
        audio: np.ndarray,
        timestamp: float
    ) -> list[AudioEvent]:
        """
        Detect audio events in audio segment.
        
        Args:
            audio: Audio samples (mono or multi-channel)
            timestamp: Timestamp of audio start
            
        Returns:
            List of detected audio events
        """
        # Mix to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        detected_events = []
        window_samples = int(self.window_size * self.sample_rate)
        hop_samples = int(self.hop_size * self.sample_rate)
        
        # Process in windows
        for i in range(0, len(audio) - window_samples + 1, hop_samples):
            window = audio[i:i + window_samples]
            window_timestamp = timestamp + i / self.sample_rate
            
            rms = self.compute_rms(window)
            intensity_db = self.compute_db(rms)
            
            # Update ambient estimate with low-energy segments
            if rms < self.ambient_rms * 1.5:
                self.update_ambient(rms)
            
            # Check for activity
            if not self.is_active(rms):
                # End current event if any
                if self.current_event is not None:
                    if self.current_event.duration >= self.min_event_duration:
                        detected_events.append(self.current_event)
                    self.current_event = None
                continue
            
            # Classify the window
            class_name, confidence, class_probs = self.classify(window)
            
            if confidence < self.confidence_threshold:
                continue
            
            event_class = self.class_to_enum.get(class_name, AudioEventClass.UNKNOWN)
            
            # Check if continuing current event
            if (self.current_event is not None and
                self.current_event.event_class == event_class):
                # Extend current event
                self.current_event.duration = (
                    window_timestamp + self.window_size - 
                    self.current_event.timestamp
                )
                self.current_event.confidence = max(
                    self.current_event.confidence,
                    confidence
                )
            else:
                # End previous event
                if self.current_event is not None:
                    if self.current_event.duration >= self.min_event_duration:
                        detected_events.append(self.current_event)
                
                # Start new event
                self.current_event = AudioEvent(
                    event_class=event_class,
                    confidence=confidence,
                    timestamp=window_timestamp,
                    duration=self.window_size,
                    intensity_db=intensity_db,
                    class_probabilities=class_probs
                )
        
        return detected_events
    
    def detect_realtime(
        self,
        audio_frame: np.ndarray,
        timestamp: float
    ) -> Optional[AudioEvent]:
        """
        Real-time event detection for streaming audio.
        
        Args:
            audio_frame: Audio frame (chunk)
            timestamp: Frame timestamp
            
        Returns:
            Detected event or None
        """
        # Accumulate frames for window-based detection
        events = self.detect(audio_frame, timestamp)
        return events[0] if events else None
    
    def get_pending_event(self) -> Optional[AudioEvent]:
        """Get the current in-progress event."""
        return self.current_event
    
    def reset(self) -> None:
        """Reset detector state."""
        self.current_event = None
        self.events_buffer.clear()
        self.ambient_rms = 0.01


class VoiceActivityDetector:
    """
    Specialized voice activity detection (VAD).
    
    Uses energy and spectral features for robust speech detection.
    """
    
    def __init__(
        self,
        sample_rate: int = 48000,
        frame_duration_ms: int = 30,
        threshold: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.threshold = threshold
        
        # Energy-based VAD state
        self.energy_threshold = 0.001
        self.speech_energy = 0.0
        self.noise_energy = 0.0
        self.speech_count = 0
        self.noise_count = 0
        
        # Spectral features for robustness
        self.prev_spectrum = None
        
        logger.info("VoiceActivityDetector initialized")
    
    def compute_energy(self, frame: np.ndarray) -> float:
        """Compute frame energy."""
        return np.sum(frame ** 2) / len(frame)
    
    def compute_spectral_flux(self, frame: np.ndarray) -> float:
        """Compute spectral flux for speech detection."""
        spectrum = np.abs(np.fft.rfft(frame))
        spectrum = spectrum / (np.max(spectrum) + 1e-10)
        
        if self.prev_spectrum is None:
            self.prev_spectrum = spectrum
            return 0.0
        
        flux = np.sum((spectrum - self.prev_spectrum) ** 2)
        self.prev_spectrum = spectrum
        
        return flux
    
    def compute_zcr(self, frame: np.ndarray) -> float:
        """Compute zero-crossing rate."""
        signs = np.sign(frame)
        return np.sum(np.abs(np.diff(signs))) / (2 * len(frame))
    
    def detect(self, audio: np.ndarray) -> list[tuple[float, float]]:
        """
        Detect voice activity regions.
        
        Args:
            audio: Audio samples
            
        Returns:
            List of (start_ratio, end_ratio) for speech regions
        """
        regions = []
        in_speech = False
        speech_start = 0
        
        for i in range(0, len(audio) - self.frame_size, self.frame_size):
            frame = audio[i:i + self.frame_size]
            
            energy = self.compute_energy(frame)
            zcr = self.compute_zcr(frame)
            flux = self.compute_spectral_flux(frame)
            
            # Combine features
            is_speech = (
                energy > self.energy_threshold and
                zcr < 0.3 and  # Speech has moderate ZCR
                flux > 0.01
            )
            
            if is_speech and not in_speech:
                speech_start = i
                in_speech = True
            elif not is_speech and in_speech:
                speech_end = i
                start_ratio = speech_start / len(audio)
                end_ratio = speech_end / len(audio)
                if end_ratio - start_ratio > 0.05:  # Min 5% duration
                    regions.append((start_ratio, end_ratio))
                in_speech = False
            
            # Update thresholds
            if is_speech:
                self.speech_energy = 0.9 * self.speech_energy + 0.1 * energy
                self.speech_count += 1
            else:
                self.noise_energy = 0.99 * self.noise_energy + 0.01 * energy
                self.noise_count += 1
            
            # Adaptive threshold
            if self.noise_count > 10:
                self.energy_threshold = self.noise_energy * 2
        
        # Close any open region
        if in_speech:
            regions.append((speech_start / len(audio), 1.0))
        
        return regions
    
    def is_speech(self, frame: np.ndarray) -> bool:
        """Check if a single frame contains speech."""
        energy = self.compute_energy(frame)
        zcr = self.compute_zcr(frame)
        
        return energy > self.energy_threshold and zcr < 0.3


if __name__ == "__main__":
    # Test event detection
    print("Testing AudioEventDetector...")
    
    detector = AudioEventDetector(
        sample_rate=48000,
        confidence_threshold=0.3
    )
    
    # Generate test audio (speech-like burst)
    sample_rate = 48000
    duration = 2.0
    t = np.arange(int(duration * sample_rate)) / sample_rate
    
    # Mix of frequencies to simulate speech
    audio = np.zeros_like(t)
    audio += 0.3 * np.sin(2 * np.pi * 200 * t)  # Fundamental
    audio += 0.2 * np.sin(2 * np.pi * 400 * t)  # Harmonics
    audio += 0.1 * np.sin(2 * np.pi * 800 * t)
    audio *= np.exp(-t * 2)  # Decay envelope
    audio += 0.01 * np.random.randn(len(t))  # Noise
    
    timestamp = time.time()
    events = detector.detect(audio, timestamp)
    
    print(f"\nDetected {len(events)} events:")
    for event in events:
        print(f"  {event}")
    
    # Test VAD
    print("\nTesting VoiceActivityDetector...")
    vad = VoiceActivityDetector(sample_rate=48000)
    
    regions = vad.detect(audio)
    print(f"Detected {len(regions)} speech regions:")
    for start, end in regions:
        print(f"  {start:.2f} - {end:.2f}")
