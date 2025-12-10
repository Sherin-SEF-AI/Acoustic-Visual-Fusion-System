"""
Video Capture Module for synchronized multi-camera capture.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
import cv2
import numpy as np
from loguru import logger


@dataclass
class VideoFrame:
    """Represents a captured video frame."""
    image: np.ndarray
    timestamp: float
    camera_id: str
    frame_index: int
    resolution: tuple[int, int] = field(default=(0, 0))
    
    def __post_init__(self):
        if self.image is not None:
            self.resolution = (self.image.shape[1], self.image.shape[0])


class VideoCapture:
    """Single camera video capture with configuration options."""
    
    def __init__(self, device_index: int = 0, camera_id: str = None,
                 resolution: tuple[int, int] = (1280, 720), fps: int = 30):
        self.device_index = device_index
        self.camera_id = camera_id or f"camera_{device_index}"
        self.target_resolution = resolution
        self.target_fps = fps
        
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_index = 0
        self._lock = threading.Lock()
        
        logger.info(f"VideoCapture initialized: {self.camera_id}")
    
    def open(self) -> bool:
        """Open the camera device."""
        try:
            self.capture = cv2.VideoCapture(self.device_index)
            if not self.capture.isOpened():
                logger.error(f"Failed to open camera {self.device_index}")
                return False
            
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
            self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_running = True
            logger.info(f"Camera {self.camera_id} opened successfully")
            return True
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def read(self) -> Optional[VideoFrame]:
        """Read a frame from the camera."""
        if self.capture is None or not self.capture.isOpened():
            return None
        
        with self._lock:
            ret, frame = self.capture.read()
            if not ret:
                return None
            
            self.frame_index += 1
            return VideoFrame(
                image=frame,
                timestamp=time.time(),
                camera_id=self.camera_id,
                frame_index=self.frame_index
            )
    
    def close(self):
        """Release the camera."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.is_running = False
        logger.info(f"Camera {self.camera_id} closed")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()


class SynchronizedVideoCapture:
    """Synchronized capture from multiple cameras."""
    
    def __init__(self, device_indices: list[int], resolution: tuple[int, int] = (1280, 720),
                 fps: int = 30, sync_tolerance_ms: float = 50.0):
        self.device_indices = device_indices
        self.resolution = resolution
        self.fps = fps
        self.sync_tolerance = sync_tolerance_ms / 1000.0
        
        self.cameras: dict[str, VideoCapture] = {}
        self.is_running = False
        self._capture_threads: list[threading.Thread] = []
        self._latest_frames: dict[str, VideoFrame] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        logger.info(f"SynchronizedVideoCapture for {len(device_indices)} cameras")
    
    def open(self) -> bool:
        """Open all cameras."""
        success = True
        for idx in self.device_indices:
            camera_id = f"camera_{idx}"
            cam = VideoCapture(idx, camera_id, self.resolution, self.fps)
            if cam.open():
                self.cameras[camera_id] = cam
            else:
                success = False
                logger.warning(f"Failed to open camera {idx}")
        
        if self.cameras:
            self.is_running = True
            self._start_capture_threads()
        return success and len(self.cameras) > 0
    
    def _start_capture_threads(self):
        """Start background capture threads."""
        self._stop_event.clear()
        for camera_id, camera in self.cameras.items():
            thread = threading.Thread(
                target=self._capture_loop, args=(camera_id, camera), daemon=True
            )
            thread.start()
            self._capture_threads.append(thread)
    
    def _capture_loop(self, camera_id: str, camera: VideoCapture):
        """Continuous capture loop for one camera."""
        while not self._stop_event.is_set():
            frame = camera.read()
            if frame is not None:
                with self._lock:
                    self._latest_frames[camera_id] = frame
            time.sleep(0.001)
    
    def read_all(self) -> dict[str, Optional[VideoFrame]]:
        """Read latest frames from all cameras."""
        with self._lock:
            return {cam_id: self._latest_frames.get(cam_id) 
                    for cam_id in self.cameras}
    
    def read_synchronized(self) -> Optional[dict[str, VideoFrame]]:
        """Read synchronized frames from all cameras."""
        with self._lock:
            frames = dict(self._latest_frames)
        
        if len(frames) != len(self.cameras):
            return None
        
        timestamps = [f.timestamp for f in frames.values()]
        if max(timestamps) - min(timestamps) > self.sync_tolerance:
            return None
        
        return frames
    
    def close(self):
        """Close all cameras."""
        self._stop_event.set()
        for thread in self._capture_threads:
            thread.join(timeout=1.0)
        for camera in self.cameras.values():
            camera.close()
        self.cameras.clear()
        self.is_running = False
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
