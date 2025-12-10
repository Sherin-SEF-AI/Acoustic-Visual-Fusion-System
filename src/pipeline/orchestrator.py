"""
System Orchestrator for the Acoustic-Visual Fusion System.

Coordinates all processing pipelines and manages real-time data flow.
"""

import threading
import queue
import time
from typing import Optional, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class ProcessingResult:
    """Generic processing result."""
    pipeline: str
    data: dict
    timestamp: float
    latency_ms: float


class SystemOrchestrator:
    """
    Coordinates audio, video, and fusion pipelines.
    
    Manages multi-threaded processing with lock-free queues
    for inter-stage communication.
    """
    
    def __init__(self, config=None):
        from ..core.config import get_config
        self.config = config or get_config()
        
        # Processing queues
        self.audio_queue = queue.Queue(maxsize=100)
        self.video_queue = queue.Queue(maxsize=100)
        self.fusion_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Thread management
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        
        # Components (initialized on start)
        self.hardware_manager = None
        self.audio_capture = None
        self.video_capture = None
        self.audio_localizer = None
        self.video_detector = None
        self.video_tracker = None
        self.fusion_engine = None
        
        # State tracking
        self.is_running = False
        self.start_time = 0.0
        self.frame_count = 0
        self.audio_event_count = 0
        
        # Callbacks
        self.on_result: Optional[Callable[[ProcessingResult], None]] = None
        
        logger.info("SystemOrchestrator initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            from ..core.hardware_manager import HardwareManager
            from ..audio.capture import AudioCapture
            from ..audio.localization import SoundLocalizer
            from ..video.capture import SynchronizedVideoCapture
            from ..video.detection import ObjectDetector
            from ..video.tracking import MultiObjectTracker
            from ..fusion.audio_visual_fusion import AudioVisualFusion
            
            # Hardware
            self.hardware_manager = HardwareManager(self.config)
            if not self.hardware_manager.initialize_all():
                logger.warning("Some hardware not available")
            
            # Audio
            active_mics = self.hardware_manager.get_active_microphones()
            if active_mics:
                mic_positions = self.hardware_manager.get_microphone_positions()
                self.audio_localizer = SoundLocalizer(
                    microphone_positions=mic_positions,
                    sample_rate=self.config.audio.capture.sample_rate
                )
                logger.info(f"Audio localizer ready with {len(active_mics)} mics")
            
            # Video
            active_cams = self.hardware_manager.get_active_cameras()
            if active_cams:
                device_indices = [cam.index for cam in active_cams]
                self.video_capture = SynchronizedVideoCapture(
                    device_indices=device_indices,
                    resolution=tuple(self.config.video.capture.resolution),
                    fps=self.config.video.capture.fps
                )
                self.video_detector = ObjectDetector(
                    model_name=self.config.video.detection.model,
                    confidence_threshold=self.config.video.detection.confidence_threshold
                )
                self.video_tracker = MultiObjectTracker(
                    track_buffer=self.config.video.tracking.track_buffer
                )
                logger.info(f"Video pipeline ready with {len(active_cams)} cameras")
            
            # Fusion
            self.fusion_engine = AudioVisualFusion(
                max_distance=self.config.fusion.spatial.max_distance_m,
                temporal_window=self.config.fusion.temporal.window_ms / 1000
            )
            
            logger.info("All components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start all processing pipelines."""
        if self.is_running:
            return True
        
        if not self.initialize_components():
            return False
        
        self._stop_event.clear()
        self.start_time = time.time()
        
        # Start video capture
        if self.video_capture:
            self.video_capture.open()
        
        # Start processing threads
        self._threads = [
            threading.Thread(target=self._video_loop, name="VideoProcessor", daemon=True),
            threading.Thread(target=self._audio_loop, name="AudioProcessor", daemon=True),
            threading.Thread(target=self._fusion_loop, name="FusionProcessor", daemon=True),
        ]
        
        for t in self._threads:
            t.start()
        
        self.is_running = True
        logger.info("System started")
        return True
    
    def stop(self):
        """Stop all processing."""
        self._stop_event.set()
        
        for t in self._threads:
            t.join(timeout=2.0)
        
        if self.video_capture:
            self.video_capture.close()
        
        if self.hardware_manager:
            self.hardware_manager.shutdown()
        
        self.is_running = False
        logger.info("System stopped")
    
    def _video_loop(self):
        """Video processing loop."""
        logger.info("Video processing started")
        
        while not self._stop_event.is_set():
            try:
                if self.video_capture is None:
                    time.sleep(0.1)
                    continue
                
                frames = self.video_capture.read_all()
                if not frames:
                    time.sleep(0.01)
                    continue
                
                start_time = time.time()
                all_detections = []
                
                for camera_id, frame in frames.items():
                    if frame is None:
                        continue
                    
                    # Detect
                    detections = self.video_detector.detect(
                        frame.image, camera_id, frame.timestamp
                    )
                    all_detections.extend(detections)
                
                # Track
                tracks = self.video_tracker.update(all_detections)
                
                # Queue for fusion
                self.video_queue.put({
                    'tracks': tracks,
                    'detections': all_detections,
                    'timestamp': time.time()
                })
                
                self.frame_count += 1
                latency = (time.time() - start_time) * 1000
                
                if self.on_result:
                    self.on_result(ProcessingResult(
                        pipeline="video",
                        data={'tracks': len(tracks), 'detections': len(all_detections)},
                        timestamp=time.time(),
                        latency_ms=latency
                    ))
                
            except Exception as e:
                logger.error(f"Video processing error: {e}")
                time.sleep(0.1)
    
    def _audio_loop(self):
        """Audio processing loop."""
        logger.info("Audio processing started")
        
        while not self._stop_event.is_set():
            try:
                if self.audio_localizer is None:
                    time.sleep(0.1)
                    continue
                
                # Simulate audio event (replace with actual capture)
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                time.sleep(0.1)
    
    def _fusion_loop(self):
        """Fusion processing loop."""
        logger.info("Fusion processing started")
        
        while not self._stop_event.is_set():
            try:
                # Get video data
                try:
                    video_data = self.video_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Get audio data
                audio_data = None
                try:
                    audio_data = self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # Fuse if we have both
                if audio_data and self.fusion_engine:
                    visual_tracks = [
                        {'track_id': t.track_id, 'position_3d': None,
                         'timestamp': video_data['timestamp'], 'is_speaking': False}
                        for t in video_data.get('tracks', [])
                    ]
                    # Would call fusion_engine.fuse() here
                
            except Exception as e:
                logger.error(f"Fusion error: {e}")
    
    def get_status(self) -> dict:
        """Get current system status."""
        uptime = time.time() - self.start_time if self.is_running else 0
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'frame_count': self.frame_count,
            'audio_events': self.audio_event_count,
            'fps': self.frame_count / max(uptime, 1)
        }
