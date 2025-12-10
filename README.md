# Acoustic-Visual Fusion System

**Real-Time 3D Sound Source Localization with Visual Tracking and Scene Understanding**

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/719c1778-330d-46fb-be8d-ca2623d8ef7a" />


A production-grade multi-modal sensing platform that fuses acoustic and visual data for comprehensive spatial awareness. Built for meeting analytics, security monitoring, smart spaces, and industrial IoT applications.

---

## Overview

The Acoustic-Visual Fusion System combines microphone array processing with multi-camera computer vision to achieve real-time sound source localization, person tracking, speaker identification, and conversation intelligence. The system correlates audio events with visual observations to provide a unified understanding of dynamic environments.

### Key Capabilities

- **3D Sound Localization**: GCC-PHAT beamforming with Kalman filtering for sub-meter accuracy
- **Multi-Camera Tracking**: YOLO-based detection with DeepSORT multi-object tracking
- **Audio-Visual Fusion**: Spatial and temporal correlation of sound sources with tracked persons
- **Meeting Analytics**: Talk-time distribution, interruption detection, sentiment analysis
- **Speaker Diarization**: Real-time speaker identification and turn-taking analysis
- **Gesture Recognition**: Hand and body gesture detection using pose estimation
- **Zone Management**: 3D spatial zone definition with occupancy alerts
- **Network Streaming**: HTTP MJPEG video streams and JSON API endpoints

### System Flow Diagram

```
                    INPUT LAYER
    ┌─────────────────────┬─────────────────────┐
    │   CAMERA ARRAY      │   MICROPHONE ARRAY  │
    │   (4 USB Cameras)   │   (8 Channel Array) │
    └─────────┬───────────┴──────────┬──────────┘
              │                      │
              ▼                      ▼
    ┌─────────────────────┐ ┌─────────────────────┐
    │   VIDEO PIPELINE    │ │   AUDIO PIPELINE    │
    │                     │ │                     │
    │  Frame Capture      │ │  Multi-Ch Capture   │
    │       ▼             │ │       ▼             │
    │  YOLO Detection     │ │  GCC-PHAT Beamform  │
    │       ▼             │ │       ▼             │
    │  DeepSORT Tracking  │ │  3D Localization    │
    │       ▼             │ │       ▼             │
    │  Pose Estimation    │ │  Speech Detection   │
    └─────────┬───────────┘ └──────────┬──────────┘
              │                        │
              └───────────┬────────────┘
                          ▼
            ┌─────────────────────────────┐
            │    AUDIO-VISUAL FUSION      │
            │                             │
            │  - Spatial Correlation      │
            │  - Temporal Matching        │
            │  - Speaker Attribution      │
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │     MEETING ANALYTICS       │
            │                             │
            │  Talk-Time │ Interruptions  │
            │  Sentiment │ Turn-Taking    │
            └──────────────┬──────────────┘
                           ▼
                    OUTPUT LAYER
    ┌─────────────────────┬─────────────────────┐
    │   PyQt6 GUI         │   HTTP API          │
    │   Dashboard         │   Streaming         │
    └─────────────────────┴─────────────────────┘
```

---

## Architecture

```
src/
├── audio/                 # Audio processing pipeline
│   ├── capture.py         # Multi-channel audio capture
│   ├── beamforming.py     # GCC-PHAT delay estimation
│   ├── localization.py    # 3D sound source localization
│   ├── speech_detection.py # Voice activity detection
│   ├── speaker_diarization.py # Speaker identification
│   └── scene_classifier.py # Acoustic environment classification
├── video/                 # Video processing pipeline
│   ├── capture.py         # Multi-camera frame capture
│   ├── detection.py       # YOLO object detection
│   ├── tracking.py        # DeepSORT multi-object tracking
│   ├── pose_estimation.py # MediaPipe pose landmarks
│   └── gesture_recognition.py # Hand and body gestures
├── fusion/                # Multi-modal fusion
│   └── audio_visual_fusion.py # Spatial-temporal correlation
├── analytics/             # Conversation intelligence
│   ├── meeting_analytics.py # Talk-time and participation
│   ├── interruption_detector.py # Interruption classification
│   ├── turn_taking.py     # Conversation flow analysis
│   ├── sentiment_analyzer.py # Acoustic-visual sentiment
│   └── event_predictor.py # Predictive analytics
├── core/                  # System infrastructure
│   ├── config.py          # Configuration management
│   ├── hardware_manager.py # Device discovery
│   ├── person_database.py # Person recognition
│   ├── zone_manager.py    # Spatial zone management
│   └── export_manager.py  # Data export utilities
├── gui/                   # PyQt6 desktop application
│   ├── main_window.py     # Main application window
│   ├── camera_panel.py    # Multi-camera display
│   ├── audio_panel.py     # Audio visualization
│   ├── meeting_panel.py   # Meeting analytics dashboard
│   ├── scene_3d_widget.py # 3D scene visualization
│   └── ...                # Additional GUI components
└── api/                   # Network interfaces
    └── streaming.py       # HTTP streaming server
```

---

## Installation

### System Requirements

- Python 3.10 or higher
- CUDA-capable GPU (recommended for real-time detection)
- USB cameras or V4L2-compatible devices
- USB microphone array or ALSA-compatible audio devices
- Linux (Ubuntu 20.04+ recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Sherin-SEF-AI/Acoustic-Visual-Fusion-System.git
cd Acoustic-Visual-Fusion-System

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install optional GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Dependencies

| Package | Purpose |
|---------|---------|
| PyQt6 | Desktop GUI framework |
| OpenCV | Video capture and processing |
| NumPy | Numerical computations |
| SciPy | Signal processing |
| ultralytics | YOLO object detection |
| mediapipe | Pose estimation |
| sounddevice | Audio capture |
| loguru | Logging |

---

## Usage

### Running the Application

```bash
source venv/bin/activate
python -m src.main
```

### Configuration

Edit `config/settings.yaml` to configure:

- Camera device indices and resolutions
- Microphone array geometry
- Detection confidence thresholds
- Fusion parameters
- Zone definitions

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Start/Stop processing |
| R | Toggle recording |
| F | Toggle fullscreen |
| 1-4 | Switch tabs |
| Ctrl+Q | Quit application |

---

## Features

### Meeting Analytics

The system provides comprehensive conversation intelligence:

- **Talk-Time Distribution**: Real-time pie chart showing speaking time per participant
- **Balance Score**: Quantifies participation equity and flags dominated meetings
- **Interruption Analysis**: Detects and classifies collaborative vs competitive interruptions
- **Turn-Taking Patterns**: Models conversation rhythm and identifies awkward silences
- **Sentiment Tracking**: Fuses acoustic prosody with visual emotion recognition

### Sound Localization

Multi-channel audio processing enables accurate sound source positioning:

- Microphone array geometry calibration
- GCC-PHAT time delay estimation
- Triangulation with Kalman filtering
- Real-time 3D position tracking

### Visual Tracking

Computer vision pipeline for person and object tracking:

- YOLO v8 detection with CUDA acceleration
- DeepSORT multi-object tracking with re-identification
- Pose estimation for gesture recognition
- Face detection for speaker correlation

---

## API Reference

### Streaming Endpoints

When the streaming server is enabled, access live data at:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web dashboard |
| `GET /video/<camera_id>` | MJPEG video stream |
| `GET /api/detections` | Current detections JSON |
| `GET /api/tracks` | Active tracks JSON |
| `GET /api/audio` | Audio data JSON |
| `GET /api/fusion` | Fusion results JSON |

---

## Performance

Tested hardware configurations:

| Configuration | Processing Rate |
|---------------|-----------------|
| 4 cameras, 8 microphones, RTX 3080 | 30 FPS |
| 2 cameras, 4 microphones, GTX 1660 | 25 FPS |
| 1 camera, 2 microphones, CPU only | 10 FPS |

---

## Applications

- **Meeting Rooms**: Automatic transcription, speaker attribution, engagement metrics
- **Security Monitoring**: Intruder detection with audio-visual correlation
- **Smart Spaces**: Occupancy sensing, zone-based automation
- **Industrial IoT**: Equipment monitoring, safety compliance
- **Accessibility**: Sound visualization for hearing-impaired users

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Author

**Sherin Joseph Roy**  
Co-Founder and Head of Products at DeepMost AI

Building enterprise AI systems that connect data, automation, and intelligence to solve real-world challenges.

- Website: [sherinjosephroy.link](https://sherinjosephroy.link)
- LinkedIn: [linkedin.com/in/sherin-roy-deepmost](https://www.linkedin.com/in/sherin-roy-deepmost)
- GitHub: [github.com/Sherin-SEF-AI](https://github.com/Sherin-SEF-AI)
- X: [x.com/SherinSEF](https://x.com/SherinSEF)

---

## Acknowledgments

- YOLO by Ultralytics for object detection
- MediaPipe by Google for pose estimation
- PyQt6 for the desktop application framework
- OpenCV for computer vision utilities

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{acoustic_visual_fusion,
  author = {Roy, Sherin Joseph},
  title = {Acoustic-Visual Fusion System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Sherin-SEF-AI/Acoustic-Visual-Fusion-System}
}
```
