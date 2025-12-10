"""
Real-time Audio Capture with Hardware Integration.
"""

import threading
import time
import queue
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
from loguru import logger

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    logger.warning("sounddevice not available - install with: pip install sounddevice")


@dataclass
class AudioChunk:
    """Audio data chunk with metadata."""
    data: np.ndarray
    timestamp: float
    device_id: str
    channels: int
    sample_rate: int


class RealtimeAudioCapture:
    """
    Real-time multi-channel audio capture with callbacks.
    Supports multiple input devices simultaneously.
    """
    
    def __init__(self, sample_rate: int = 48000, 
                 chunk_size: int = 1024,
                 channels: int = 1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        
        self._running = False
        self._streams: dict[str, sd.InputStream] = {}
        self._queues: dict[str, queue.Queue] = {}
        
        # Callbacks
        self._on_audio: Optional[Callable[[AudioChunk], None]] = None
        self._on_level: Optional[Callable[[str, float, float], None]] = None
        
        # Level tracking
        self._levels: dict[str, float] = {}
        
        logger.info(f"RealtimeAudioCapture initialized: {sample_rate}Hz, {chunk_size} samples")
    
    @staticmethod
    def list_devices() -> list[dict]:
        """List available input devices."""
        if not SD_AVAILABLE:
            return []
        
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append({
                    'id': str(i),
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': int(dev['default_samplerate'])
                })
        return devices
    
    def add_device(self, device_id: str) -> bool:
        """Add an input device for capture."""
        if not SD_AVAILABLE:
            return False
        
        try:
            device_idx = int(device_id)
            device_info = sd.query_devices(device_idx)
            
            self._queues[device_id] = queue.Queue(maxsize=100)
            
            def callback(indata, frames, time_info, status):
                if status:
                    logger.debug(f"Audio status: {status}")
                
                # Calculate RMS level
                rms = np.sqrt(np.mean(indata ** 2))
                db = 20 * np.log10(max(rms, 1e-10))
                self._levels[device_id] = rms
                
                if self._on_level:
                    self._on_level(device_id, rms, db)
                
                # Create chunk
                chunk = AudioChunk(
                    data=indata.copy(),
                    timestamp=time.time(),
                    device_id=device_id,
                    channels=indata.shape[1] if len(indata.shape) > 1 else 1,
                    sample_rate=self.sample_rate
                )
                
                try:
                    self._queues[device_id].put_nowait(chunk)
                except queue.Full:
                    pass
                
                if self._on_audio:
                    self._on_audio(chunk)
            
            stream = sd.InputStream(
                device=device_idx,
                channels=min(self.channels, device_info['max_input_channels']),
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=callback
            )
            
            self._streams[device_id] = stream
            logger.info(f"Added audio device: {device_info['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add device {device_id}: {e}")
            return False
    
    def start(self):
        """Start all audio streams."""
        if not SD_AVAILABLE:
            logger.warning("Cannot start audio - sounddevice not available")
            return
        
        self._running = True
        for device_id, stream in self._streams.items():
            stream.start()
            logger.info(f"Started audio stream: {device_id}")
    
    def stop(self):
        """Stop all audio streams."""
        self._running = False
        for device_id, stream in self._streams.items():
            stream.stop()
            stream.close()
        self._streams.clear()
        self._queues.clear()
        logger.info("All audio streams stopped")
    
    def get_chunk(self, device_id: str, timeout: float = 0.1) -> Optional[AudioChunk]:
        """Get next audio chunk from device queue."""
        if device_id not in self._queues:
            return None
        
        try:
            return self._queues[device_id].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_combined_audio(self, window_ms: int = 100) -> Optional[np.ndarray]:
        """Get combined audio from all devices for localization."""
        if not self._queues:
            return None
        
        # Collect chunks from all devices
        chunks = []
        for device_id, q in self._queues.items():
            try:
                while not q.empty():
                    chunks.append(q.get_nowait())
            except:
                pass
        
        if not chunks:
            return None
        
        # Stack audio from different devices
        # For localization, we need synchronized multi-channel audio
        combined = np.column_stack([c.data.flatten()[:self.chunk_size] for c in chunks])
        return combined
    
    def set_audio_callback(self, callback: Callable[[AudioChunk], None]):
        """Set callback for audio chunks."""
        self._on_audio = callback
    
    def set_level_callback(self, callback: Callable[[str, float, float], None]):
        """Set callback for level updates (device_id, level, db)."""
        self._on_level = callback
    
    def get_level(self, device_id: str) -> float:
        """Get current audio level for device."""
        return self._levels.get(device_id, 0.0)
    
    @property
    def is_running(self) -> bool:
        return self._running
