"""
Audio Capture Worker for real-time audio processing.
"""

import threading
import time
import numpy as np
from typing import Optional, Callable
from loguru import logger

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False


class AudioCaptureWorker:
    """Real-time audio capture with callbacks."""
    
    def __init__(self, device_index: int = None, sample_rate: int = 48000,
                 channels: int = 1, block_size: int = 1024):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        
        self.stream = None
        self._running = False
        
        # Callbacks
        self.on_audio: Optional[Callable[[np.ndarray, float], None]] = None
        self.on_level: Optional[Callable[[float, float], None]] = None
        
        # Buffer for analysis
        self.buffer = np.zeros((sample_rate * 2, channels))  # 2 seconds
        self.buffer_pos = 0
    
    def start(self) -> bool:
        """Start audio capture."""
        if not SD_AVAILABLE:
            logger.warning("sounddevice not available")
            return False
        
        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                callback=self._audio_callback
            )
            self.stream.start()
            self._running = True
            logger.info(f"Audio capture started: {self.sample_rate}Hz, {self.channels}ch")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info, status):
        """Audio stream callback."""
        if status:
            logger.warning(f"Audio status: {status}")
        
        timestamp = time.time()
        
        # Calculate level
        rms = np.sqrt(np.mean(indata ** 2))
        db = 20 * np.log10(max(rms, 1e-10))
        level = min(1.0, rms * 10)
        
        # Store in buffer
        samples = len(indata)
        if self.buffer_pos + samples <= len(self.buffer):
            self.buffer[self.buffer_pos:self.buffer_pos + samples] = indata
            self.buffer_pos += samples
        else:
            # Wrap around
            self.buffer_pos = 0
            self.buffer[:samples] = indata
            self.buffer_pos = samples
        
        # Callbacks
        if self.on_audio:
            self.on_audio(indata.flatten(), timestamp)
        
        if self.on_level:
            self.on_level(level, db)
    
    def get_recent_audio(self, duration: float = 0.5) -> np.ndarray:
        """Get recent audio data."""
        samples = int(duration * self.sample_rate)
        if self.buffer_pos >= samples:
            return self.buffer[self.buffer_pos - samples:self.buffer_pos].flatten()
        else:
            # Wrap around case
            return self.buffer[:self.buffer_pos].flatten()
    
    @property
    def is_running(self) -> bool:
        return self._running


class MultiMicCapture:
    """Capture from multiple microphones simultaneously."""
    
    def __init__(self, device_indices: list[int], sample_rate: int = 48000):
        self.device_indices = device_indices
        self.sample_rate = sample_rate
        
        self.workers: dict[int, AudioCaptureWorker] = {}
        self.levels: dict[int, tuple[float, float]] = {}
        self.buffers: dict[int, np.ndarray] = {}
        
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """Start all microphones."""
        success = True
        
        for idx in self.device_indices:
            worker = AudioCaptureWorker(
                device_index=idx,
                sample_rate=self.sample_rate,
                channels=1
            )
            
            # Set up callbacks
            worker.on_level = lambda l, d, i=idx: self._on_level(i, l, d)
            worker.on_audio = lambda a, t, i=idx: self._on_audio(i, a, t)
            
            if worker.start():
                self.workers[idx] = worker
            else:
                success = False
        
        logger.info(f"Started {len(self.workers)}/{len(self.device_indices)} mics")
        return success
    
    def stop(self):
        """Stop all microphones."""
        for worker in self.workers.values():
            worker.stop()
        self.workers.clear()
    
    def _on_level(self, device_idx: int, level: float, db: float):
        with self._lock:
            self.levels[device_idx] = (level, db)
    
    def _on_audio(self, device_idx: int, audio: np.ndarray, timestamp: float):
        with self._lock:
            self.buffers[device_idx] = audio
    
    def get_levels(self) -> dict[int, tuple[float, float]]:
        with self._lock:
            return self.levels.copy()
    
    def get_all_audio(self) -> dict[int, np.ndarray]:
        with self._lock:
            return self.buffers.copy()
