"""
Hardware Manager for the Acoustic-Visual Fusion System.

Handles discovery, initialization, and management of cameras and microphones.
Provides hot-plug support and graceful degradation.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
import cv2
import numpy as np
from loguru import logger

try:
    import sounddevice as sd
except ImportError:
    sd = None
    logger.warning("sounddevice not available, microphone support disabled")

try:
    import pyaudio
except ImportError:
    pyaudio = None


class DeviceStatus(Enum):
    """Device connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    INITIALIZING = "initializing"


@dataclass
class CameraDevice:
    """Represents a camera device."""
    id: str
    index: int
    name: str
    status: DeviceStatus = DeviceStatus.DISCONNECTED
    resolution: tuple[int, int] = (1280, 720)
    fps: int = 30
    capture: Optional[cv2.VideoCapture] = None
    position: Optional[np.ndarray] = None  # 3D position in world frame
    orientation: Optional[np.ndarray] = None  # Rotation matrix
    intrinsic_matrix: Optional[np.ndarray] = None
    distortion_coeffs: Optional[np.ndarray] = None
    last_frame_time: float = 0.0
    frame_count: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.orientation is None:
            self.orientation = np.eye(3)


@dataclass
class MicrophoneDevice:
    """Represents a microphone device."""
    id: str
    index: int
    name: str
    status: DeviceStatus = DeviceStatus.DISCONNECTED
    sample_rate: int = 48000
    channels: int = 1
    position: Optional[np.ndarray] = None  # 3D position in world frame
    stream: Optional[Any] = None
    last_sample_time: float = 0.0
    sample_count: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)


class HardwareManager:
    """
    Manages hardware discovery, initialization, and monitoring.
    
    Provides:
    - Automatic device discovery
    - Hot-plug support
    - Graceful degradation on device failure
    - Device status monitoring
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        on_camera_change: Optional[Callable] = None,
        on_microphone_change: Optional[Callable] = None
    ):
        """
        Initialize the hardware manager.
        
        Args:
            config: Configuration object
            on_camera_change: Callback for camera status changes
            on_microphone_change: Callback for microphone status changes
        """
        from .config import get_config
        
        self.config = config or get_config()
        self.on_camera_change = on_camera_change
        self.on_microphone_change = on_microphone_change
        
        self.cameras: dict[str, CameraDevice] = {}
        self.microphones: dict[str, MicrophoneDevice] = {}
        
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        logger.info("HardwareManager initialized")
    
    def discover_cameras(self) -> list[CameraDevice]:
        """
        Discover all available cameras.
        
        Returns:
            List of discovered camera devices.
        """
        discovered = []
        max_cameras = 10  # Maximum cameras to check
        
        logger.info("Discovering cameras...")
        
        # Track which physical cameras we've found to avoid V4L2 duplicates
        found_resolutions = set()
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify it's a real camera
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    continue
                
                # Get camera info
                backend = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                
                # Check for V4L2 duplicates (same resolution at consecutive indices)
                # V4L2 creates /dev/video0, /dev/video1 for same camera
                camera_sig = (width, height, i // 2)  # Group consecutive pairs
                if camera_sig in found_resolutions and i > 0:
                    cap.release()
                    continue
                found_resolutions.add(camera_sig)
                
                # Generate unique ID
                device_id = f"camera_{i}_{backend}"
                
                # Determine name
                name = f"Camera {i} ({backend})"
                if len(discovered) == 0:
                    name = f"Integrated Camera ({backend})"
                else:
                    name = f"USB Camera {len(discovered)} ({backend})"
                
                device = CameraDevice(
                    id=device_id,
                    index=i,
                    name=name,
                    status=DeviceStatus.CONNECTED,
                    resolution=(width, height),
                    fps=fps,
                    capture=cap  # Keep capture open!
                )
                
                discovered.append(device)
                logger.info(f"Found camera: {name} at index {i} ({width}x{height}@{fps}fps)")
            else:
                cap.release()
        
        logger.info(f"Discovered {len(discovered)} cameras")
        return discovered
    
    def discover_microphones(self) -> list[MicrophoneDevice]:
        """
        Discover all available microphones.
        
        Returns:
            List of discovered microphone devices.
        """
        discovered = []
        
        logger.info("Discovering microphones...")
        
        if sd is not None:
            try:
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        device_id = f"mic_{i}_{device['name'][:20].replace(' ', '_')}"
                        
                        mic = MicrophoneDevice(
                            id=device_id,
                            index=i,
                            name=device['name'],
                            status=DeviceStatus.CONNECTED,
                            sample_rate=int(device['default_samplerate']),
                            channels=device['max_input_channels']
                        )
                        
                        discovered.append(mic)
                        logger.info(
                            f"Found microphone: {device['name']} "
                            f"({mic.channels}ch @ {mic.sample_rate}Hz)"
                        )
            except Exception as e:
                logger.error(f"Error discovering microphones via sounddevice: {e}")
        
        elif pyaudio is not None:
            try:
                pa = pyaudio.PyAudio()
                for i in range(pa.get_device_count()):
                    device = pa.get_device_info_by_index(i)
                    if device['maxInputChannels'] > 0:
                        device_id = f"mic_{i}_{device['name'][:20].replace(' ', '_')}"
                        
                        mic = MicrophoneDevice(
                            id=device_id,
                            index=i,
                            name=device['name'],
                            status=DeviceStatus.CONNECTED,
                            sample_rate=int(device['defaultSampleRate']),
                            channels=int(device['maxInputChannels'])
                        )
                        
                        discovered.append(mic)
                        logger.info(
                            f"Found microphone: {device['name']} "
                            f"({mic.channels}ch @ {mic.sample_rate}Hz)"
                        )
                pa.terminate()
            except Exception as e:
                logger.error(f"Error discovering microphones via pyaudio: {e}")
        else:
            logger.warning("No audio library available for microphone discovery")
        
        logger.info(f"Discovered {len(discovered)} microphones")
        return discovered
    
    def initialize_all(self) -> bool:
        """
        Discover and initialize all hardware devices.
        
        Returns:
            True if minimum required devices are available.
        """
        with self._lock:
            # Discover devices
            cameras = self.discover_cameras()
            microphones = self.discover_microphones()
            
            # Store devices
            for cam in cameras:
                self.cameras[cam.id] = cam
            
            for mic in microphones:
                self.microphones[mic.id] = mic
            
            # Check minimum requirements
            min_cameras = self.config.hardware.cameras.count
            min_mics = self.config.hardware.microphones.count
            
            if len(self.cameras) < min_cameras:
                logger.warning(
                    f"Found {len(self.cameras)} cameras, "
                    f"but {min_cameras} required. Continuing with degraded mode."
                )
            
            if len(self.microphones) < min_mics:
                logger.warning(
                    f"Found {len(self.microphones)} microphones, "
                    f"but {min_mics} required. Continuing with degraded mode."
                )
            
            # Initialize cameras with configured settings
            for cam in self.cameras.values():
                self._initialize_camera(cam)
            
            return len(self.cameras) > 0 or len(self.microphones) > 0
    
    def _initialize_camera(self, camera: CameraDevice) -> bool:
        """Initialize a single camera with configured settings."""
        try:
            # Skip if already initialized and open
            if camera.capture is not None and camera.capture.isOpened():
                camera.status = DeviceStatus.CONNECTED
                return True
            
            cap = cv2.VideoCapture(camera.index)
            if not cap.isOpened():
                camera.status = DeviceStatus.ERROR
                logger.error(f"Failed to open camera {camera.name}")
                return False
            
            # Configure camera
            resolution = self.config.hardware.cameras.default_resolution
            fps = self.config.hardware.cameras.default_fps
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Verify settings
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            camera.capture = cap
            camera.resolution = (actual_width, actual_height)
            camera.fps = actual_fps
            camera.status = DeviceStatus.CONNECTED
            
            logger.info(
                f"Initialized camera {camera.name}: "
                f"{actual_width}x{actual_height}@{actual_fps}fps"
            )
            return True
            
        except Exception as e:
            camera.status = DeviceStatus.ERROR
            logger.error(f"Error initializing camera {camera.name}: {e}")
            return False
    
    def open_camera(self, device_id: str) -> Optional[CameraDevice]:
        """
        Open a specific camera by ID.
        
        Args:
            device_id: Camera device ID.
            
        Returns:
            Camera device if successful, None otherwise.
        """
        with self._lock:
            if device_id not in self.cameras:
                logger.error(f"Camera {device_id} not found")
                return None
            
            camera = self.cameras[device_id]
            if camera.status == DeviceStatus.CONNECTED and camera.capture is not None:
                return camera
            
            if self._initialize_camera(camera):
                return camera
            return None
    
    def close_camera(self, device_id: str) -> None:
        """Close a specific camera."""
        with self._lock:
            if device_id in self.cameras:
                camera = self.cameras[device_id]
                if camera.capture is not None:
                    camera.capture.release()
                    camera.capture = None
                camera.status = DeviceStatus.DISCONNECTED
                logger.info(f"Closed camera {camera.name}")
    
    def read_camera(self, device_id: str) -> Optional[np.ndarray]:
        """
        Read a frame from a camera.
        
        Args:
            device_id: Camera device ID.
            
        Returns:
            Frame as numpy array if successful, None otherwise.
        """
        with self._lock:
            if device_id not in self.cameras:
                return None
            
            camera = self.cameras[device_id]
            if camera.capture is None or not camera.capture.isOpened():
                # Try to reconnect
                if not self._initialize_camera(camera):
                    return None
            
            try:
                ret, frame = camera.capture.read()
                if ret:
                    camera.last_frame_time = time.time()
                    camera.frame_count += 1
                    camera.error_count = 0
                    return frame
                else:
                    camera.error_count += 1
                    if camera.error_count > 5:
                        logger.warning(f"Camera {camera.name} read failures, reconnecting...")
                        self._initialize_camera(camera)
                    return None
            except Exception as e:
                camera.error_count += 1
                logger.error(f"Error reading camera {camera.name}: {e}")
                return None
    
    def read_all_cameras(self) -> dict[str, Optional[np.ndarray]]:
        """
        Read frames from all connected cameras.
        
        Returns:
            Dictionary mapping device IDs to frames.
        """
        frames = {}
        for device_id in list(self.cameras.keys()):
            frames[device_id] = self.read_camera(device_id)
        return frames
    
    def get_active_cameras(self) -> list[CameraDevice]:
        """Get list of currently active cameras."""
        with self._lock:
            return [
                cam for cam in self.cameras.values()
                if cam.status == DeviceStatus.CONNECTED
            ]
    
    def get_active_microphones(self) -> list[MicrophoneDevice]:
        """Get list of currently active microphones."""
        with self._lock:
            return [
                mic for mic in self.microphones.values()
                if mic.status == DeviceStatus.CONNECTED
            ]
    
    def start_monitoring(self, interval: float = 5.0) -> None:
        """
        Start background device monitoring for hot-plug detection.
        
        Args:
            interval: Monitoring interval in seconds.
        """
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_devices,
            args=(interval,),
            daemon=True,
            name="HardwareMonitor"
        )
        self._monitor_thread.start()
        logger.info("Started hardware monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background device monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Stopped hardware monitoring")
    
    def _monitor_devices(self, interval: float) -> None:
        """Background device monitoring loop."""
        while not self._stop_monitoring.wait(interval):
            try:
                self._check_camera_status()
                self._check_microphone_status()
            except Exception as e:
                logger.error(f"Error in device monitoring: {e}")
    
    def _check_camera_status(self) -> None:
        """Check and update camera connection status."""
        with self._lock:
            for camera in self.cameras.values():
                if camera.capture is not None:
                    if not camera.capture.isOpened():
                        old_status = camera.status
                        camera.status = DeviceStatus.DISCONNECTED
                        camera.capture = None
                        
                        if old_status != DeviceStatus.DISCONNECTED:
                            logger.warning(f"Camera {camera.name} disconnected")
                            if self.on_camera_change:
                                self.on_camera_change(camera, "disconnected")
                        
                        # Try to reconnect
                        if self._initialize_camera(camera):
                            logger.info(f"Camera {camera.name} reconnected")
                            if self.on_camera_change:
                                self.on_camera_change(camera, "reconnected")
    
    def _check_microphone_status(self) -> None:
        """Check and update microphone connection status."""
        # Re-discover microphones to detect changes
        current_mics = self.discover_microphones()
        current_ids = {m.id for m in current_mics}
        
        with self._lock:
            # Check for disconnections
            for mic_id, mic in list(self.microphones.items()):
                if mic_id not in current_ids:
                    old_status = mic.status
                    mic.status = DeviceStatus.DISCONNECTED
                    
                    if old_status == DeviceStatus.CONNECTED:
                        logger.warning(f"Microphone {mic.name} disconnected")
                        if self.on_microphone_change:
                            self.on_microphone_change(mic, "disconnected")
            
            # Check for new connections
            for mic in current_mics:
                if mic.id not in self.microphones:
                    self.microphones[mic.id] = mic
                    logger.info(f"New microphone detected: {mic.name}")
                    if self.on_microphone_change:
                        self.on_microphone_change(mic, "connected")
                elif self.microphones[mic.id].status == DeviceStatus.DISCONNECTED:
                    self.microphones[mic.id].status = DeviceStatus.CONNECTED
                    logger.info(f"Microphone {mic.name} reconnected")
                    if self.on_microphone_change:
                        self.on_microphone_change(self.microphones[mic.id], "reconnected")
    
    def set_camera_position(
        self,
        device_id: str,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None
    ) -> None:
        """Set camera position and orientation in world frame."""
        with self._lock:
            if device_id in self.cameras:
                self.cameras[device_id].position = position
                if orientation is not None:
                    self.cameras[device_id].orientation = orientation
    
    def set_microphone_position(self, device_id: str, position: np.ndarray) -> None:
        """Set microphone position in world frame."""
        with self._lock:
            if device_id in self.microphones:
                self.microphones[device_id].position = position
    
    def get_microphone_positions(self) -> np.ndarray:
        """
        Get positions of all active microphones.
        
        Returns:
            Nx3 array of microphone positions.
        """
        with self._lock:
            active_mics = self.get_active_microphones()
            if not active_mics:
                return np.zeros((0, 3))
            return np.array([mic.position for mic in active_mics])
    
    def get_status_summary(self) -> dict:
        """Get summary of hardware status."""
        with self._lock:
            return {
                "cameras": {
                    "total": len(self.cameras),
                    "connected": sum(
                        1 for c in self.cameras.values()
                        if c.status == DeviceStatus.CONNECTED
                    ),
                    "devices": [
                        {
                            "id": c.id,
                            "name": c.name,
                            "status": c.status.value,
                            "resolution": c.resolution,
                            "fps": c.fps,
                            "frame_count": c.frame_count
                        }
                        for c in self.cameras.values()
                    ]
                },
                "microphones": {
                    "total": len(self.microphones),
                    "connected": sum(
                        1 for m in self.microphones.values()
                        if m.status == DeviceStatus.CONNECTED
                    ),
                    "devices": [
                        {
                            "id": m.id,
                            "name": m.name,
                            "status": m.status.value,
                            "sample_rate": m.sample_rate,
                            "channels": m.channels,
                            "position": m.position.tolist() if m.position is not None else None
                        }
                        for m in self.microphones.values()
                    ]
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown hardware manager and release all devices."""
        logger.info("Shutting down hardware manager...")
        
        self.stop_monitoring()
        
        with self._lock:
            # Close all cameras
            for camera in self.cameras.values():
                if camera.capture is not None:
                    camera.capture.release()
                    camera.capture = None
                camera.status = DeviceStatus.DISCONNECTED
            
            # Close all microphone streams
            for mic in self.microphones.values():
                if mic.stream is not None:
                    try:
                        mic.stream.stop()
                        mic.stream.close()
                    except:
                        pass
                    mic.stream = None
                mic.status = DeviceStatus.DISCONNECTED
        
        logger.info("Hardware manager shutdown complete")
    
    def __enter__(self):
        self.initialize_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


if __name__ == "__main__":
    # Test hardware discovery
    import sys
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    console.print("\n[bold blue]Hardware Discovery Test[/bold blue]\n")
    
    with HardwareManager() as hw:
        status = hw.get_status_summary()
        
        # Display cameras
        cam_table = Table(title="Cameras")
        cam_table.add_column("ID")
        cam_table.add_column("Name")
        cam_table.add_column("Status")
        cam_table.add_column("Resolution")
        cam_table.add_column("FPS")
        
        for cam in status["cameras"]["devices"]:
            cam_table.add_row(
                cam["id"],
                cam["name"],
                cam["status"],
                f"{cam['resolution'][0]}x{cam['resolution'][1]}",
                str(cam["fps"])
            )
        
        console.print(cam_table)
        
        # Display microphones
        mic_table = Table(title="Microphones")
        mic_table.add_column("ID")
        mic_table.add_column("Name")
        mic_table.add_column("Status")
        mic_table.add_column("Sample Rate")
        mic_table.add_column("Channels")
        
        for mic in status["microphones"]["devices"]:
            mic_table.add_row(
                mic["id"],
                mic["name"],
                mic["status"],
                f"{mic['sample_rate']} Hz",
                str(mic["channels"])
            )
        
        console.print(mic_table)
        
        console.print(f"\n[green]Total cameras: {status['cameras']['total']}[/green]")
        console.print(f"[green]Total microphones: {status['microphones']['total']}[/green]")
