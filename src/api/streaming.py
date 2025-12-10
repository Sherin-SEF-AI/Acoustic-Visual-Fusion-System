"""
Network Streaming Server - Stream video/audio over network.
"""

import json
import time
import threading
import base64
from typing import Optional, Callable
from loguru import logger
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from flask import Flask, Response, render_template_string, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not available - install with: pip install flask flask-cors")


class StreamingServer:
    """
    HTTP streaming server for video and data.
    Provides MJPEG streams and JSON API endpoints.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        
        self._app: Optional[Flask] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Frame buffers
        self._frames: dict[str, bytes] = {}
        self._frame_locks: dict[str, threading.Lock] = {}
        
        # Data buffers
        self._data: dict[str, dict] = {}
        
        # Stats
        self._clients = 0
        self._bytes_sent = 0
        
        if FLASK_AVAILABLE:
            self._setup_app()
        
        logger.info(f"StreamingServer initialized: {host}:{port}")
    
    def _setup_app(self):
        """Setup Flask application."""
        self._app = Flask(__name__)
        CORS(self._app)
        
        # Dashboard page
        @self._app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        # MJPEG stream for each camera
        @self._app.route('/stream/<camera_id>')
        def stream(camera_id):
            def generate():
                while self._running:
                    if camera_id in self._frames:
                        with self._frame_locks.get(camera_id, threading.Lock()):
                            frame = self._frames.get(camera_id)
                        
                        if frame:
                            self._bytes_sent += len(frame)
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
                    time.sleep(0.033)  # ~30 fps
            
            self._clients += 1
            response = Response(generate(),
                               mimetype='multipart/x-mixed-replace; boundary=frame')
            return response
        
        # API endpoints
        @self._app.route('/api/status')
        def api_status():
            return jsonify({
                'running': self._running,
                'cameras': list(self._frames.keys()),
                'clients': self._clients,
                'bytes_sent': self._bytes_sent
            })
        
        @self._app.route('/api/detections')
        def api_detections():
            return jsonify(self._data.get('detections', []))
        
        @self._app.route('/api/tracks')
        def api_tracks():
            return jsonify(self._data.get('tracks', []))
        
        @self._app.route('/api/audio')
        def api_audio():
            return jsonify(self._data.get('audio', {}))
        
        @self._app.route('/api/fusion')
        def api_fusion():
            return jsonify(self._data.get('fusion', {}))
    
    def start(self):
        """Start the streaming server."""
        if not FLASK_AVAILABLE:
            logger.warning("Cannot start streaming - Flask not available")
            return False
        
        self._running = True
        
        def run_server():
            self._app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                use_reloader=False
            )
        
        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()
        
        logger.info(f"Streaming server started: http://{self.host}:{self.port}")
        return True
    
    def stop(self):
        """Stop the streaming server."""
        self._running = False
        logger.info("Streaming server stopped")
    
    def update_frame(self, camera_id: str, frame: np.ndarray):
        """Update frame for a camera."""
        if not CV2_AVAILABLE:
            return
        
        if camera_id not in self._frame_locks:
            self._frame_locks[camera_id] = threading.Lock()
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        with self._frame_locks[camera_id]:
            self._frames[camera_id] = buffer.tobytes()
    
    def update_data(self, key: str, data: dict):
        """Update API data."""
        self._data[key] = data
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_running(self) -> bool:
        return self._running


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AV Fusion Stream</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: #0d1117; 
            color: #e6edf3; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .header {
            background: #161b22;
            padding: 16px 24px;
            border-bottom: 1px solid #30363d;
        }
        .header h1 { font-size: 20px; }
        .container { padding: 24px; }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
            gap: 16px;
        }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
        }
        .card-header {
            padding: 12px 16px;
            border-bottom: 1px solid #30363d;
            font-weight: 600;
        }
        .stream {
            width: 100%;
            aspect-ratio: 4/3;
            background: #000;
        }
        .stats {
            padding: 16px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 600;
            color: #58a6ff;
        }
        .stat-label {
            font-size: 12px;
            color: #8b949e;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎙️ Acoustic-Visual Fusion - Live Stream</h1>
    </div>
    <div class="container">
        <div class="grid" id="streams"></div>
        <div class="card" style="margin-top: 16px;">
            <div class="card-header">📊 System Status</div>
            <div class="stats" id="stats">
                <div class="stat">
                    <div class="stat-value" id="cameras">-</div>
                    <div class="stat-label">Cameras</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="clients">-</div>
                    <div class="stat-label">Clients</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="bandwidth">-</div>
                    <div class="stat-label">Bandwidth</div>
                </div>
            </div>
        </div>
    </div>
    <script>
        async function updateStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                
                document.getElementById('cameras').textContent = data.cameras.length;
                document.getElementById('clients').textContent = data.clients;
                document.getElementById('bandwidth').textContent = 
                    (data.bytes_sent / 1024 / 1024).toFixed(1) + ' MB';
                
                // Create stream cards
                const grid = document.getElementById('streams');
                if (grid.children.length === 0) {
                    data.cameras.forEach(cam => {
                        const card = document.createElement('div');
                        card.className = 'card';
                        card.innerHTML = `
                            <div class="card-header">📹 ${cam}</div>
                            <img class="stream" src="/stream/${cam}">
                        `;
                        grid.appendChild(card);
                    });
                }
            } catch (e) {}
        }
        
        updateStatus();
        setInterval(updateStatus, 2000);
    </script>
</body>
</html>
"""
