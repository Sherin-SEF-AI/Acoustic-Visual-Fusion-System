"""
Beamforming Module for spatial audio filtering and source separation.
"""

import numpy as np
from abc import ABC, abstractmethod
from loguru import logger


class Beamformer(ABC):
    """Abstract base class for beamformers."""
    
    def __init__(self, mic_positions: np.ndarray, sample_rate: int = 48000,
                 speed_of_sound: float = 343.0):
        self.mic_positions = np.asarray(mic_positions)
        self.num_mics = len(self.mic_positions)
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound
        self.array_center = np.mean(self.mic_positions, axis=0)
    
    @abstractmethod
    def steer(self, audio: np.ndarray, direction: np.ndarray) -> np.ndarray:
        pass
    
    def compute_delays(self, direction: np.ndarray) -> np.ndarray:
        """Compute time delays for steering to direction."""
        delays = np.zeros(self.num_mics)
        for i in range(self.num_mics):
            mic_vec = self.mic_positions[i] - self.array_center
            proj = np.dot(mic_vec, direction)
            delays[i] = proj / self.speed_of_sound
        delays -= np.min(delays)
        return delays


class DelayAndSumBeamformer(Beamformer):
    """Delay-and-Sum Beamformer for simple spatial filtering."""
    
    def __init__(self, mic_positions: np.ndarray, sample_rate: int = 48000,
                 speed_of_sound: float = 343.0):
        super().__init__(mic_positions, sample_rate, speed_of_sound)
        logger.info(f"DelayAndSumBeamformer initialized: {self.num_mics} mics")
    
    def _apply_delay(self, signal: np.ndarray, delay_samples: float) -> np.ndarray:
        """Apply delay with linear interpolation."""
        n = len(signal)
        delay_int = int(np.floor(delay_samples))
        frac = delay_samples - delay_int
        output = np.zeros(n)
        
        for i in range(n):
            src_idx = i - delay_int
            if 0 <= src_idx < n - 1:
                output[i] = (1 - frac) * signal[src_idx] + frac * signal[src_idx + 1]
            elif src_idx == n - 1:
                output[i] = signal[src_idx]
        return output
    
    def steer(self, audio: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Steer beamformer to direction and return mono output."""
        direction = np.asarray(direction)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        delays = self.compute_delays(direction)
        delay_samples = delays * self.sample_rate
        
        output = np.zeros(len(audio))
        for i in range(self.num_mics):
            aligned = self._apply_delay(audio[:, i], delay_samples[i])
            output += aligned
        return output / self.num_mics
    
    def scan_directions(self, audio: np.ndarray, resolution_deg: float = 10.0):
        """Scan all directions and compute beamformer output power."""
        azimuths = np.arange(-180, 180, resolution_deg)
        elevations = np.arange(-90, 90, resolution_deg)
        power_map = np.zeros((len(azimuths), len(elevations)))
        
        for i, az in enumerate(azimuths):
            for j, el in enumerate(elevations):
                az_rad, el_rad = np.radians(az), np.radians(el)
                direction = np.array([
                    np.cos(el_rad) * np.cos(az_rad),
                    np.cos(el_rad) * np.sin(az_rad),
                    np.sin(el_rad)
                ])
                output = self.steer(audio, direction)
                power_map[i, j] = np.sum(output ** 2)
        return azimuths, elevations, power_map
    
    def find_peak_direction(self, audio: np.ndarray, resolution_deg: float = 5.0):
        """Find direction with maximum beamformer output."""
        azimuths, elevations, power_map = self.scan_directions(audio, resolution_deg)
        max_idx = np.unravel_index(np.argmax(power_map), power_map.shape)
        az = np.radians(azimuths[max_idx[0]])
        el = np.radians(elevations[max_idx[1]])
        return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


class MVDRBeamformer(Beamformer):
    """MVDR (Capon) Beamformer with better interference rejection."""
    
    def __init__(self, mic_positions: np.ndarray, sample_rate: int = 48000,
                 speed_of_sound: float = 343.0, fft_size: int = 1024):
        super().__init__(mic_positions, sample_rate, speed_of_sound)
        self.fft_size = fft_size
        self.frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        logger.info(f"MVDRBeamformer initialized: {self.num_mics} mics")
    
    def compute_steering_vector(self, direction: np.ndarray, freq: float):
        delays = self.compute_delays(direction)
        return np.exp(-2j * np.pi * freq * delays)
    
    def estimate_covariance(self, audio: np.ndarray):
        n_freqs = len(self.frequencies)
        hop = self.fft_size // 2
        n_frames = max(1, (len(audio) - self.fft_size) // hop + 1)
        R = np.zeros((self.num_mics, self.num_mics, n_freqs), dtype=complex)
        
        for frame_idx in range(n_frames):
            start = frame_idx * hop
            if start + self.fft_size > len(audio):
                break
            frame = audio[start:start + self.fft_size]
            X = np.array([np.fft.rfft(frame[:, i]) for i in range(self.num_mics)])
            for k in range(n_freqs):
                x = X[:, k:k+1]
                R[:, :, k] += x @ x.conj().T
        
        R /= max(n_frames, 1)
        for k in range(n_freqs):
            R[:, :, k] += np.eye(self.num_mics) * 1e-4
        return R
    
    def steer(self, audio: np.ndarray, direction: np.ndarray) -> np.ndarray:
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        R = self.estimate_covariance(audio)
        
        X = np.array([np.fft.rfft(audio[:, i], self.fft_size) for i in range(self.num_mics)])
        Y = np.zeros(len(self.frequencies), dtype=complex)
        
        for k, freq in enumerate(self.frequencies):
            a = self.compute_steering_vector(direction, freq)
            R_inv = np.linalg.inv(R[:, :, k])
            w = R_inv @ a / (a.conj() @ R_inv @ a + 1e-10)
            Y[k] = w.conj() @ X[:, k]
        
        output = np.fft.irfft(Y, self.fft_size)
        return output[:len(audio)] if len(output) >= len(audio) else np.pad(output, (0, len(audio)-len(output)))
