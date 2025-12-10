"""
Audio Capture Module for the Acoustic-Visual Fusion System.

Provides synchronized multi-channel audio capture with circular buffering.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import numpy as np
from loguru import logger

try:
    import sounddevice as sd
except ImportError:
    sd = None
    logger.warning("sounddevice not available")


@dataclass
class AudioFrame:
    """Represents a captured audio frame."""
    data: np.ndarray  # Shape: (samples, channels)
    timestamp: float  # Unix timestamp
    sample_rate: int
    channels: int
    frame_index: int
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.data) / self.sample_rate
    
    @property
    def end_timestamp(self) -> float:
        """Timestamp of the end of this frame."""
        return self.timestamp + self.duration


class AudioBuffer:
    """
    Circular buffer for storing audio data with timestamp indexing.
    
    Maintains a rolling window of audio data from multiple channels
    for retrospective analysis.
    """
    
    def __init__(
        self,
        duration_sec: float,
        sample_rate: int,
        num_channels: int
    ):
        """
        Initialize audio buffer.
        
        Args:
            duration_sec: Buffer duration in seconds
            sample_rate: Audio sample rate
            num_channels: Number of audio channels
        """
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        
        # Pre-allocate buffer
        self.buffer_size = int(duration_sec * sample_rate)
        self.buffer = np.zeros((self.buffer_size, num_channels), dtype=np.float32)
        
        # Circular buffer state
        self.write_pos = 0
        self.samples_written = 0
        self.start_timestamp = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.debug(
            f"AudioBuffer initialized: {duration_sec}s, {sample_rate}Hz, "
            f"{num_channels}ch, {self.buffer_size} samples"
        )
    
    def write(self, data: np.ndarray, timestamp: float) -> None:
        """
        Write audio data to the buffer.
        
        Args:
            data: Audio samples (samples x channels)
            timestamp: Timestamp of first sample
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if data.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {data.shape[1]}"
            )
        
        with self._lock:
            num_samples = len(data)
            
            # Update start timestamp if buffer was empty
            if self.samples_written == 0:
                self.start_timestamp = timestamp
            
            # Handle wrap-around
            if self.write_pos + num_samples <= self.buffer_size:
                self.buffer[self.write_pos:self.write_pos + num_samples] = data
            else:
                # Split write across buffer boundary
                first_part = self.buffer_size - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:num_samples - first_part] = data[first_part:]
            
            # Update position
            self.write_pos = (self.write_pos + num_samples) % self.buffer_size
            self.samples_written += num_samples
            
            # Update start timestamp when buffer is full
            if self.samples_written > self.buffer_size:
                samples_from_start = self.samples_written - self.buffer_size
                self.start_timestamp = timestamp - (self.buffer_size - num_samples) / self.sample_rate
    
    def read(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None
    ) -> tuple[np.ndarray, float]:
        """
        Read audio data from the buffer.
        
        Args:
            start_time: Start timestamp (default: buffer start)
            end_time: End timestamp (default: buffer end)
            duration: Duration in seconds (alternative to end_time)
            
        Returns:
            Tuple of (audio data, start timestamp)
        """
        with self._lock:
            if self.samples_written == 0:
                return np.zeros((0, self.num_channels), dtype=np.float32), 0.0
            
            # Calculate available time range
            buffer_end_time = self.start_timestamp + min(
                self.samples_written, self.buffer_size
            ) / self.sample_rate
            buffer_start_time = buffer_end_time - min(
                self.samples_written, self.buffer_size
            ) / self.sample_rate
            
            # Set defaults
            if start_time is None:
                start_time = buffer_start_time
            if end_time is None:
                if duration is not None:
                    end_time = start_time + duration
                else:
                    end_time = buffer_end_time
            
            # Clamp to available range
            start_time = max(start_time, buffer_start_time)
            end_time = min(end_time, buffer_end_time)
            
            if end_time <= start_time:
                return np.zeros((0, self.num_channels), dtype=np.float32), start_time
            
            # Convert times to sample indices
            available_samples = min(self.samples_written, self.buffer_size)
            start_sample = int((start_time - buffer_start_time) * self.sample_rate)
            end_sample = int((end_time - buffer_start_time) * self.sample_rate)
            
            start_sample = max(0, min(start_sample, available_samples))
            end_sample = max(0, min(end_sample, available_samples))
            
            num_samples = end_sample - start_sample
            if num_samples <= 0:
                return np.zeros((0, self.num_channels), dtype=np.float32), start_time
            
            # Calculate buffer positions
            read_start = (self.write_pos - available_samples + start_sample) % self.buffer_size
            
            # Read data (handling wrap-around)
            result = np.zeros((num_samples, self.num_channels), dtype=np.float32)
            
            if read_start + num_samples <= self.buffer_size:
                result[:] = self.buffer[read_start:read_start + num_samples]
            else:
                first_part = self.buffer_size - read_start
                result[:first_part] = self.buffer[read_start:]
                result[first_part:] = self.buffer[:num_samples - first_part]
            
            return result, start_time
    
    def read_latest(self, duration: float) -> tuple[np.ndarray, float]:
        """
        Read the most recent audio data.
        
        Args:
            duration: Duration to read in seconds
            
        Returns:
            Tuple of (audio data, start timestamp)
        """
        with self._lock:
            if self.samples_written == 0:
                return np.zeros((0, self.num_channels), dtype=np.float32), 0.0
            
            buffer_end_time = self.start_timestamp + min(
                self.samples_written, self.buffer_size
            ) / self.sample_rate
            
        return self.read(start_time=buffer_end_time - duration, end_time=buffer_end_time)
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.samples_written = 0
            self.start_timestamp = 0.0
    
    @property
    def available_duration(self) -> float:
        """Available audio duration in seconds."""
        with self._lock:
            return min(self.samples_written, self.buffer_size) / self.sample_rate


class AudioCapture:
    """
    Multi-channel synchronized audio capture.
    
    Captures audio from multiple microphones with precise timestamping
    and stores in a circular buffer for retrospective analysis.
    """
    
    def __init__(
        self,
        device_indices: Optional[list[int]] = None,
        sample_rate: int = 48000,
        chunk_size: int = 1024,
        buffer_duration: float = 5.0,
        on_audio_callback: Optional[Callable[[AudioFrame], None]] = None
    ):
        """
        Initialize audio capture.
        
        Args:
            device_indices: List of input device indices (None for default)
            sample_rate: Sample rate in Hz
            chunk_size: Samples per callback
            buffer_duration: Circular buffer duration in seconds
            on_audio_callback: Callback for each captured frame
        """
        if sd is None:
            raise ImportError("sounddevice library required for audio capture")
        
        self.device_indices = device_indices
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_duration = buffer_duration
        self.on_audio_callback = on_audio_callback
        
        # Determine number of channels
        if device_indices:
            self.num_channels = len(device_indices)
        else:
            # Use default input device
            default_device = sd.default.device[0]
            device_info = sd.query_devices(default_device)
            self.num_channels = min(device_info['max_input_channels'], 4)
        
        # Initialize buffer
        self.buffer = AudioBuffer(
            duration_sec=buffer_duration,
            sample_rate=sample_rate,
            num_channels=self.num_channels
        )
        
        # Capture state
        self.stream: Optional[sd.InputStream] = None
        self.is_running = False
        self.frame_index = 0
        self._start_time = 0.0
        
        logger.info(
            f"AudioCapture initialized: {self.num_channels}ch @ {sample_rate}Hz, "
            f"chunk={chunk_size}, buffer={buffer_duration}s"
        )
    
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any
    ) -> None:
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio capture status: {status}")
        
        # Calculate timestamp
        timestamp = time.time()
        
        # Create audio frame
        frame = AudioFrame(
            data=indata.copy(),
            timestamp=timestamp,
            sample_rate=self.sample_rate,
            channels=self.num_channels,
            frame_index=self.frame_index
        )
        self.frame_index += 1
        
        # Write to buffer
        self.buffer.write(indata, timestamp)
        
        # Call user callback
        if self.on_audio_callback:
            try:
                self.on_audio_callback(frame)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
    
    def start(self) -> bool:
        """
        Start audio capture.
        
        Returns:
            True if started successfully.
        """
        if self.is_running:
            logger.warning("Audio capture already running")
            return True
        
        try:
            device = self.device_indices[0] if self.device_indices else None
            
            self.stream = sd.InputStream(
                device=device,
                channels=self.num_channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32,
                callback=self._audio_callback
            )
            
            self._start_time = time.time()
            self.stream.start()
            self.is_running = True
            
            logger.info("Audio capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def stop(self) -> None:
        """Stop audio capture."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
        
        self.is_running = False
        logger.info("Audio capture stopped")
    
    def get_latest_audio(self, duration: float) -> tuple[np.ndarray, float]:
        """
        Get the most recent audio data.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Tuple of (audio array, start timestamp)
        """
        return self.buffer.read_latest(duration)
    
    def get_audio_at(
        self,
        timestamp: float,
        duration: float
    ) -> tuple[np.ndarray, float]:
        """
        Get audio data at a specific timestamp.
        
        Args:
            timestamp: Start timestamp
            duration: Duration in seconds
            
        Returns:
            Tuple of (audio array, actual start timestamp)
        """
        return self.buffer.read(start_time=timestamp, duration=duration)
    
    def get_channels_separately(
        self,
        duration: float
    ) -> list[tuple[np.ndarray, float]]:
        """
        Get audio from each channel separately.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            List of (audio, timestamp) tuples, one per channel
        """
        audio, timestamp = self.buffer.read_latest(duration)
        return [(audio[:, i], timestamp) for i in range(self.num_channels)]
    
    @property
    def available_duration(self) -> float:
        """Get available audio duration in buffer."""
        return self.buffer.available_duration
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class MultiDeviceAudioCapture:
    """
    Captures audio from multiple separate microphones as synchronized channels.
    
    Useful when using multiple USB microphones rather than a single array.
    """
    
    def __init__(
        self,
        device_indices: list[int],
        sample_rate: int = 48000,
        chunk_size: int = 1024,
        buffer_duration: float = 5.0
    ):
        """
        Initialize multi-device capture.
        
        Args:
            device_indices: List of device indices to capture from
            sample_rate: Sample rate in Hz
            chunk_size: Samples per callback
            buffer_duration: Circular buffer duration in seconds
        """
        if sd is None:
            raise ImportError("sounddevice library required for audio capture")
        
        self.device_indices = device_indices
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_duration = buffer_duration
        self.num_channels = len(device_indices)
        
        # One buffer per device
        self.buffers = [
            AudioBuffer(
                duration_sec=buffer_duration,
                sample_rate=sample_rate,
                num_channels=1
            )
            for _ in device_indices
        ]
        
        # Combined buffer
        self.combined_buffer = AudioBuffer(
            duration_sec=buffer_duration,
            sample_rate=sample_rate,
            num_channels=self.num_channels
        )
        
        self.streams: list[sd.InputStream] = []
        self.is_running = False
        self._lock = threading.Lock()
        self._latest_samples: dict[int, np.ndarray] = {}
        self._latest_timestamps: dict[int, float] = {}
        
        logger.info(
            f"MultiDeviceAudioCapture initialized for devices: {device_indices}"
        )
    
    def _make_callback(self, device_idx: int):
        """Create callback for specific device."""
        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Device {device_idx} status: {status}")
            
            timestamp = time.time()
            
            # Store in individual buffer
            self.buffers[device_idx].write(indata, timestamp)
            
            # Store for combining
            with self._lock:
                self._latest_samples[device_idx] = indata.copy()
                self._latest_timestamps[device_idx] = timestamp
                
                # If we have data from all devices, combine them
                if len(self._latest_samples) == self.num_channels:
                    # Check timestamps are close enough
                    timestamps = list(self._latest_timestamps.values())
                    if max(timestamps) - min(timestamps) < 0.1:  # 100ms tolerance
                        combined = np.hstack([
                            self._latest_samples[i]
                            for i in range(self.num_channels)
                        ])
                        avg_timestamp = sum(timestamps) / len(timestamps)
                        self.combined_buffer.write(combined, avg_timestamp)
                    
                    self._latest_samples.clear()
                    self._latest_timestamps.clear()
        
        return callback
    
    def start(self) -> bool:
        """Start capture from all devices."""
        if self.is_running:
            return True
        
        try:
            for i, device_idx in enumerate(self.device_indices):
                stream = sd.InputStream(
                    device=device_idx,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    dtype=np.float32,
                    callback=self._make_callback(i)
                )
                stream.start()
                self.streams.append(stream)
            
            self.is_running = True
            logger.info("Multi-device audio capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start multi-device capture: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop all capture streams."""
        for stream in self.streams:
            try:
                stream.stop()
                stream.close()
            except:
                pass
        
        self.streams.clear()
        self.is_running = False
        logger.info("Multi-device audio capture stopped")
    
    def get_latest_audio(self, duration: float) -> tuple[np.ndarray, float]:
        """Get most recent combined audio data."""
        return self.combined_buffer.read_latest(duration)
    
    def get_channel_audio(
        self,
        channel: int,
        duration: float
    ) -> tuple[np.ndarray, float]:
        """Get audio from a specific channel."""
        if 0 <= channel < self.num_channels:
            return self.buffers[channel].read_latest(duration)
        return np.zeros((0, 1), dtype=np.float32), 0.0
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


if __name__ == "__main__":
    # Test audio capture
    import sys
    
    print("Testing AudioCapture...")
    
    if sd is None:
        print("sounddevice not available")
        sys.exit(1)
    
    # List available devices
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
    # Test capture
    def on_audio(frame):
        rms = np.sqrt(np.mean(frame.data ** 2))
        print(f"Frame {frame.frame_index}: RMS={rms:.4f}, samples={len(frame.data)}")
    
    with AudioCapture(
        sample_rate=48000,
        chunk_size=1024,
        buffer_duration=5.0,
        on_audio_callback=on_audio
    ) as capture:
        print("\nRecording for 3 seconds...")
        time.sleep(3)
        
        # Get latest audio
        audio, ts = capture.get_latest_audio(1.0)
        print(f"\nGot {len(audio)} samples from timestamp {ts:.3f}")
        print(f"Available duration: {capture.available_duration:.2f}s")
