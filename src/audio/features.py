"""
Audio Feature Extraction Module.

Extracts MFCC, spectral features, and other acoustic descriptors.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    mfcc: np.ndarray  # Mel-frequency cepstral coefficients
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_flux: float
    zero_crossing_rate: float
    rms_energy: float
    fundamental_frequency: Optional[float] = None
    timestamp: float = 0.0


class AudioFeatureExtractor:
    """Extracts acoustic features from audio signals."""
    
    def __init__(self, sample_rate: int = 48000, n_mfcc: int = 13,
                 n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.prev_spectrum = None
        logger.info(f"AudioFeatureExtractor initialized: {sample_rate}Hz")
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        if LIBROSA_AVAILABLE:
            return librosa.feature.mfcc(y=audio, sr=self.sample_rate,
                                        n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                        hop_length=self.hop_length)
        # Fallback: simple DCT of log mel spectrogram
        spectrum = np.abs(np.fft.rfft(audio, self.n_fft))
        log_spectrum = np.log(spectrum + 1e-10)
        from scipy.fftpack import dct
        return dct(log_spectrum, type=2, n=self.n_mfcc).reshape(-1, 1)
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> float:
        """Extract spectral centroid (brightness)."""
        spectrum = np.abs(np.fft.rfft(audio, self.n_fft))
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        return float(np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10))
    
    def extract_spectral_bandwidth(self, audio: np.ndarray) -> float:
        """Extract spectral bandwidth."""
        spectrum = np.abs(np.fft.rfft(audio, self.n_fft))
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
        return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / 
                            (np.sum(spectrum) + 1e-10)))
    
    def extract_spectral_flux(self, audio: np.ndarray) -> float:
        """Extract spectral flux (change between frames)."""
        spectrum = np.abs(np.fft.rfft(audio, self.n_fft))
        spectrum = spectrum / (np.max(spectrum) + 1e-10)
        if self.prev_spectrum is None:
            self.prev_spectrum = spectrum
            return 0.0
        flux = float(np.sum((spectrum - self.prev_spectrum) ** 2))
        self.prev_spectrum = spectrum
        return flux
    
    def extract_zcr(self, audio: np.ndarray) -> float:
        """Extract zero-crossing rate."""
        signs = np.sign(audio)
        return float(np.sum(np.abs(np.diff(signs))) / (2 * len(audio)))
    
    def extract_rms(self, audio: np.ndarray) -> float:
        """Extract RMS energy."""
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def extract_f0(self, audio: np.ndarray) -> Optional[float]:
        """Extract fundamental frequency (pitch)."""
        if LIBROSA_AVAILABLE:
            f0, _, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=self.sample_rate)
            valid_f0 = f0[~np.isnan(f0)]
            return float(np.median(valid_f0)) if len(valid_f0) > 0 else None
        # Simple autocorrelation-based F0
        corr = np.correlate(audio, audio, mode='full')
        corr = corr[len(corr)//2:]
        min_period = int(self.sample_rate / 500)
        max_period = int(self.sample_rate / 50)
        if max_period >= len(corr):
            return None
        peak_idx = np.argmax(corr[min_period:max_period]) + min_period
        return float(self.sample_rate / peak_idx) if peak_idx > 0 else None
    
    def extract(self, audio: np.ndarray, timestamp: float = 0.0) -> AudioFeatures:
        """Extract all features from audio segment."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return AudioFeatures(
            mfcc=self.extract_mfcc(audio),
            spectral_centroid=self.extract_spectral_centroid(audio),
            spectral_bandwidth=self.extract_spectral_bandwidth(audio),
            spectral_flux=self.extract_spectral_flux(audio),
            zero_crossing_rate=self.extract_zcr(audio),
            rms_energy=self.extract_rms(audio),
            fundamental_frequency=self.extract_f0(audio),
            timestamp=timestamp
        )
