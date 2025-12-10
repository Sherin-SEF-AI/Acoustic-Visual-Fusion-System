"""
Configuration loader and manager for the Acoustic-Visual Fusion System.

Provides centralized configuration management with validation, 
environment variable overrides, and persistent storage.
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict
import yaml
from pydantic import BaseModel, Field, field_validator
from loguru import logger


class CameraConfig(BaseModel):
    """Configuration for camera hardware."""
    count: int = 4
    default_resolution: list[int] = Field(default=[1280, 720])
    default_fps: int = 30
    auto_discover: bool = True
    devices: list[dict] = Field(default_factory=list)


class MicrophoneConfig(BaseModel):
    """Configuration for microphone hardware."""
    count: int = 4
    sample_rate: int = 48000
    channels_per_device: int = 1
    buffer_duration_sec: float = 5.0
    auto_discover: bool = True
    devices: list[dict] = Field(default_factory=list)
    positions: list[list[float]] = Field(default_factory=list)


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    cameras: CameraConfig = Field(default_factory=CameraConfig)
    microphones: MicrophoneConfig = Field(default_factory=MicrophoneConfig)


class CameraCalibrationConfig(BaseModel):
    """Camera calibration settings."""
    checkerboard_size: list[int] = Field(default=[9, 6])
    square_size_mm: float = 25.0
    min_samples: int = 20
    save_path: str = "./data/calibration/cameras"


class MicrophoneCalibrationConfig(BaseModel):
    """Microphone calibration settings."""
    speed_of_sound: float = 343.0
    temperature_celsius: float = 20.0
    calibration_chirp_freq: list[int] = Field(default=[500, 8000])
    save_path: str = "./data/calibration/microphones"


class CalibrationConfig(BaseModel):
    """Calibration configuration."""
    camera: CameraCalibrationConfig = Field(default_factory=CameraCalibrationConfig)
    microphone: MicrophoneCalibrationConfig = Field(default_factory=MicrophoneCalibrationConfig)


class AudioCaptureConfig(BaseModel):
    """Audio capture settings."""
    sample_rate: int = 48000
    chunk_size: int = 1024
    buffer_seconds: float = 5.0


class AudioEventDetectionConfig(BaseModel):
    """Audio event detection settings."""
    model: str = "ast"
    confidence_threshold: float = 0.5
    min_event_duration_ms: int = 100
    classes: list[str] = Field(default_factory=lambda: [
        "speech", "glass_breaking", "door", "footsteps",
        "alarm", "applause", "mechanical", "music"
    ])


class AudioLocalizationConfig(BaseModel):
    """Audio localization settings."""
    algorithm: str = "gcc-phat"
    fft_size: int = 4096
    hop_size: int = 512
    max_source_distance: float = 10.0
    uncertainty_threshold: float = 0.5


class BeamformingConfig(BaseModel):
    """Beamforming settings."""
    algorithm: str = "delay-sum"
    beam_resolution_deg: int = 5


class AudioTrackingConfig(BaseModel):
    """Audio source tracking settings."""
    algorithm: str = "kalman"
    max_sources: int = 5
    birth_threshold: float = 0.7
    death_threshold: float = 0.3
    process_noise: float = 0.1
    measurement_noise: float = 0.2


class AudioConfig(BaseModel):
    """Audio processing configuration."""
    capture: AudioCaptureConfig = Field(default_factory=AudioCaptureConfig)
    event_detection: AudioEventDetectionConfig = Field(default_factory=AudioEventDetectionConfig)
    localization: AudioLocalizationConfig = Field(default_factory=AudioLocalizationConfig)
    beamforming: BeamformingConfig = Field(default_factory=BeamformingConfig)
    tracking: AudioTrackingConfig = Field(default_factory=AudioTrackingConfig)


class VideoCaptureConfig(BaseModel):
    """Video capture settings."""
    resolution: list[int] = Field(default=[1280, 720])
    fps: int = 30
    sync_tolerance_ms: int = 33


class VideoDetectionConfig(BaseModel):
    """Video detection settings."""
    model: str = "yolov8m"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    classes: list[str] = Field(default_factory=lambda: ["person", "face"])


class VideoTrackingConfig(BaseModel):
    """Video tracking settings."""
    algorithm: str = "bytetrack"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8


class PoseConfig(BaseModel):
    """Pose estimation settings."""
    enabled: bool = True
    model: str = "mediapipe"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class FaceAnalysisConfig(BaseModel):
    """Face analysis settings."""
    enabled: bool = True
    detect_landmarks: bool = True
    detect_head_pose: bool = True
    detect_lip_movement: bool = True


class CrossCameraConfig(BaseModel):
    """Cross-camera tracking settings."""
    enabled: bool = True
    reid_model: str = "osnet_x1_0"
    appearance_weight: float = 0.7
    spatial_weight: float = 0.3
    match_threshold: float = 0.6


class DepthConfig(BaseModel):
    """Depth estimation settings."""
    enabled: bool = False
    model: str = "depth_anything_v2"


class ActivityConfig(BaseModel):
    """Activity detection settings."""
    speaking_threshold: float = 0.5
    gesture_detection: bool = True


class VideoConfig(BaseModel):
    """Video processing configuration."""
    capture: VideoCaptureConfig = Field(default_factory=VideoCaptureConfig)
    detection: VideoDetectionConfig = Field(default_factory=VideoDetectionConfig)
    tracking: VideoTrackingConfig = Field(default_factory=VideoTrackingConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    face_analysis: FaceAnalysisConfig = Field(default_factory=FaceAnalysisConfig)
    cross_camera: CrossCameraConfig = Field(default_factory=CrossCameraConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)


class FusionTemporalConfig(BaseModel):
    """Fusion temporal settings."""
    window_ms: int = 100
    max_delay_ms: int = 500


class FusionSpatialConfig(BaseModel):
    """Fusion spatial settings."""
    max_distance_m: float = 2.0
    confidence_decay: float = 0.5


class FusionAssociationConfig(BaseModel):
    """Fusion association settings."""
    min_confidence: float = 0.6
    use_contrastive: bool = True


class FusionConfig(BaseModel):
    """Multimodal fusion configuration."""
    enabled: bool = True
    model_path: str = "./models/fusion_model.pt"
    temporal: FusionTemporalConfig = Field(default_factory=FusionTemporalConfig)
    spatial: FusionSpatialConfig = Field(default_factory=FusionSpatialConfig)
    association: FusionAssociationConfig = Field(default_factory=FusionAssociationConfig)


class SpeakerDiarizationConfig(BaseModel):
    """Speaker diarization settings."""
    enabled: bool = True
    voice_embedding_model: str = "ecapa_tdnn"
    face_embedding_model: str = "arcface"


class AnomalyDetectionConfig(BaseModel):
    """Anomaly detection settings."""
    enabled: bool = True
    baseline_duration_sec: int = 300
    threshold_sigma: float = 3.0


class AttentionTrackingConfig(BaseModel):
    """Attention tracking settings."""
    enabled: bool = True
    gaze_model: str = "mediapipe"


class CorrelationConfig(BaseModel):
    """Correlation engine configuration."""
    speaker_diarization: SpeakerDiarizationConfig = Field(default_factory=SpeakerDiarizationConfig)
    anomaly_detection: AnomalyDetectionConfig = Field(default_factory=AnomalyDetectionConfig)
    attention_tracking: AttentionTrackingConfig = Field(default_factory=AttentionTrackingConfig)


class Scene3DConfig(BaseModel):
    """3D scene visualization settings."""
    enabled: bool = True
    camera_frustum_length: float = 1.0
    sound_sphere_duration: float = 1.0
    update_rate_hz: int = 30


class FloorplanConfig(BaseModel):
    """Floorplan visualization settings."""
    enabled: bool = True
    width_m: float = 10.0
    height_m: float = 10.0
    grid_size: float = 0.5
    heatmap_decay: float = 0.95


class OverlaysConfig(BaseModel):
    """Overlay visualization settings."""
    show_bounding_boxes: bool = True
    show_track_ids: bool = True
    show_pose: bool = True
    show_face_landmarks: bool = False
    show_speaking_indicator: bool = True
    show_audio_direction: bool = True


class VisualizationConfig(BaseModel):
    """Visualization configuration."""
    scene_3d: Scene3DConfig = Field(default_factory=Scene3DConfig)
    floorplan: FloorplanConfig = Field(default_factory=FloorplanConfig)
    overlays: OverlaysConfig = Field(default_factory=OverlaysConfig)


class WebSocketConfig(BaseModel):
    """WebSocket settings."""
    enabled: bool = True
    ping_interval: int = 30


class AuthConfig(BaseModel):
    """Authentication settings."""
    enabled: bool = False
    api_key: str = ""


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


class SQLiteConfig(BaseModel):
    """SQLite database settings."""
    path: str = "./data/avfusion.db"


class PostgreSQLConfig(BaseModel):
    """PostgreSQL database settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "avfusion"
    user: str = "avfusion"
    password: str = ""


class RetentionConfig(BaseModel):
    """Data retention settings."""
    tracks_days: int = 7
    events_days: int = 30
    recordings_days: int = 3


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = "sqlite"
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)


class AudioRecordingConfig(BaseModel):
    """Audio recording settings."""
    enabled: bool = True
    format: str = "opus"
    bitrate: int = 64000
    chunk_duration_sec: int = 60


class VideoRecordingConfig(BaseModel):
    """Video recording settings."""
    enabled: bool = True
    codec: str = "h264"
    quality: int = 23
    chunk_duration_sec: int = 60


class EvidenceConfig(BaseModel):
    """Evidence capture settings."""
    pre_roll_sec: int = 10
    post_roll_sec: int = 10
    save_path: str = "./data/evidence"


class RecordingConfig(BaseModel):
    """Recording configuration."""
    audio: AudioRecordingConfig = Field(default_factory=AudioRecordingConfig)
    video: VideoRecordingConfig = Field(default_factory=VideoRecordingConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)


class AlertRule(BaseModel):
    """Alert rule definition."""
    name: str
    audio_event: Optional[str] = None
    visual_event: Optional[str] = None
    audio_intensity_db: Optional[float] = None
    confidence: float = 0.8
    severity: str = "warning"
    zones: list[str] = Field(default_factory=list)


class WebhookNotificationConfig(BaseModel):
    """Webhook notification settings."""
    enabled: bool = False
    url: str = ""


class EmailNotificationConfig(BaseModel):
    """Email notification settings."""
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    recipients: list[str] = Field(default_factory=list)


class NotificationsConfig(BaseModel):
    """Notification settings."""
    webhook: WebhookNotificationConfig = Field(default_factory=WebhookNotificationConfig)
    email: EmailNotificationConfig = Field(default_factory=EmailNotificationConfig)


class AlertsConfig(BaseModel):
    """Alerts configuration."""
    enabled: bool = True
    rules: list[AlertRule] = Field(default_factory=list)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    target_latency_ms: int = 100
    gpu_enabled: bool = True
    gpu_device: int = 0
    batch_size: int = 1
    num_workers: int = 4
    queue_size: int = 10


class CaptionsConfig(BaseModel):
    """Captions accessibility settings."""
    enabled: bool = True
    language: str = "en"
    model: str = "whisper-small"


class SoundRadarConfig(BaseModel):
    """Sound radar accessibility settings."""
    enabled: bool = True
    update_rate_hz: int = 10


class HapticAlertsConfig(BaseModel):
    """Haptic alerts accessibility settings."""
    enabled: bool = False


class AccessibilityConfig(BaseModel):
    """Accessibility configuration."""
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    sound_radar: SoundRadarConfig = Field(default_factory=SoundRadarConfig)
    haptic_alerts: HapticAlertsConfig = Field(default_factory=HapticAlertsConfig)


class SystemConfig(BaseModel):
    """System-level configuration."""
    name: str = "AV-Fusion"
    version: str = "1.0.0"
    log_level: str = "INFO"
    data_dir: str = "./data"
    models_dir: str = "./models"


class Config(BaseModel):
    """Main configuration class for the Acoustic-Visual Fusion System."""
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    accessibility: AccessibilityConfig = Field(default_factory=AccessibilityConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)
        
        return cls(**data)
    
    @staticmethod
    def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            "AVFUSION_LOG_LEVEL": ("system", "log_level"),
            "AVFUSION_DATA_DIR": ("system", "data_dir"),
            "AVFUSION_MODELS_DIR": ("system", "models_dir"),
            "AVFUSION_API_HOST": ("api", "host"),
            "AVFUSION_API_PORT": ("api", "port"),
            "AVFUSION_GPU_ENABLED": ("performance", "gpu_enabled"),
            "AVFUSION_GPU_DEVICE": ("performance", "gpu_device"),
            "AVFUSION_DB_TYPE": ("database", "type"),
            "AVFUSION_DB_PATH": ("database", "sqlite", "path"),
            "AVFUSION_PG_HOST": ("database", "postgresql", "host"),
            "AVFUSION_PG_PORT": ("database", "postgresql", "port"),
            "AVFUSION_PG_DATABASE": ("database", "postgresql", "database"),
            "AVFUSION_PG_USER": ("database", "postgresql", "user"),
            "AVFUSION_PG_PASSWORD": ("database", "postgresql", "password"),
        }
        
        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                Config._set_nested(data, path, value)
        
        return data
    
    @staticmethod
    def _set_nested(data: Dict[str, Any], path: tuple, value: Any) -> None:
        """Set a nested dictionary value by path."""
        for key in path[:-1]:
            data = data.setdefault(key, {})
        
        # Convert value to appropriate type
        final_key = path[-1]
        if isinstance(value, str):
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
        
        data[final_key] = value
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {path}")
    
    def get_speed_of_sound(self) -> float:
        """Calculate speed of sound based on temperature."""
        temp_c = self.calibration.microphone.temperature_celsius
        # Speed of sound in air: c = 331.3 * sqrt(1 + T/273.15) m/s
        return 331.3 * ((1 + temp_c / 273.15) ** 0.5)


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str | Path] = None) -> Config:
    """Get the global configuration instance."""
    global _config
    
    if _config is None:
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        _config = Config.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    
    return _config


def reload_config(config_path: Optional[str | Path] = None) -> Config:
    """Reload the configuration from file."""
    global _config
    _config = None
    return get_config(config_path)
