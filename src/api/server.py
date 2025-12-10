"""
FastAPI Server for the Acoustic-Visual Fusion System Dashboard.
"""

import asyncio
import json
from typing import Optional
from datetime import datetime
import numpy as np
from loguru import logger

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available")


def create_app() -> "FastAPI":
    """Create and configure FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required")
    
    app = FastAPI(
        title="AV-Fusion System",
        description="Acoustic-Visual Fusion for Source Localization",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # State storage
    app.state.system = None
    app.state.clients = set()
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve main dashboard."""
        return get_dashboard_html()
    
    @app.get("/api/status")
    async def get_status():
        """Get system status."""
        return {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "cameras": 4,
            "microphones": 4,
            "active_tracks": 0,
            "audio_events": 0
        }
    
    @app.get("/api/cameras")
    async def get_cameras():
        """Get camera status."""
        return {"cameras": [
            {"id": f"camera_{i}", "status": "connected", "fps": 30}
            for i in range(4)
        ]}
    
    @app.get("/api/microphones")
    async def get_microphones():
        """Get microphone status."""
        return {"microphones": [
            {"id": f"mic_{i}", "status": "connected", "sample_rate": 48000}
            for i in range(4)
        ]}
    
    @app.get("/api/tracks")
    async def get_tracks():
        """Get active visual tracks."""
        return {"tracks": []}
    
    @app.get("/api/audio_events")
    async def get_audio_events():
        """Get recent audio events."""
        return {"events": []}
    
    @app.get("/api/localizations")
    async def get_localizations():
        """Get sound source localizations."""
        return {"localizations": []}
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates."""
        await websocket.accept()
        app.state.clients.add(websocket)
        
        try:
            while True:
                data = await websocket.receive_text()
                # Handle incoming messages
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            app.state.clients.discard(websocket)
    
    return app


def get_dashboard_html() -> str:
    """Generate dashboard HTML."""
    return '''<!DOCTYPE html>
<html>
<head>
    <title>AV-Fusion Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff; min-height: 100vh;
        }
        .header {
            padding: 20px; background: rgba(255,255,255,0.05);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex; justify-content: space-between; align-items: center;
        }
        .header h1 { font-size: 24px; font-weight: 600; }
        .status { display: flex; gap: 20px; }
        .status-item { 
            padding: 8px 16px; background: rgba(0,255,136,0.2);
            border-radius: 20px; font-size: 14px;
        }
        .main { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; padding: 20px; }
        .camera-grid { 
            display: grid; grid-template-columns: 1fr 1fr; gap: 15px;
        }
        .camera-feed {
            aspect-ratio: 16/9; background: #0f0f23;
            border-radius: 12px; position: relative; overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .camera-label {
            position: absolute; top: 10px; left: 10px;
            background: rgba(0,0,0,0.7); padding: 5px 10px;
            border-radius: 6px; font-size: 12px;
        }
        .sidebar { display: flex; flex-direction: column; gap: 20px; }
        .panel {
            background: rgba(255,255,255,0.05); border-radius: 12px;
            padding: 20px; border: 1px solid rgba(255,255,255,0.1);
        }
        .panel h2 { font-size: 16px; margin-bottom: 15px; color: #88f; }
        .audio-viz { height: 100px; background: #0f0f23; border-radius: 8px; }
        .event-list { max-height: 200px; overflow-y: auto; }
        .event { 
            padding: 10px; margin-bottom: 8px;
            background: rgba(255,255,255,0.05); border-radius: 8px;
            font-size: 13px;
        }
        .event .time { color: #888; font-size: 11px; }
        .floorplan {
            aspect-ratio: 1; background: #0f0f23; border-radius: 8px;
            position: relative;
        }
        .sound-source {
            position: absolute; width: 20px; height: 20px;
            background: rgba(255,100,100,0.8); border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.3); }
        }
        .track {
            position: absolute; width: 12px; height: 12px;
            background: rgba(100,200,255,0.8); border-radius: 50%;
            transform: translate(-50%, -50%);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎙️ Acoustic-Visual Fusion System</h1>
        <div class="status">
            <div class="status-item">⚡ Online</div>
            <div class="status-item">📹 4 Cameras</div>
            <div class="status-item">🎤 4 Microphones</div>
        </div>
    </div>
    <div class="main">
        <div class="camera-grid">
            <div class="camera-feed"><div class="camera-label">Camera 0 - Front</div></div>
            <div class="camera-feed"><div class="camera-label">Camera 1 - Left</div></div>
            <div class="camera-feed"><div class="camera-label">Camera 2 - Right</div></div>
            <div class="camera-feed"><div class="camera-label">Camera 3 - Rear</div></div>
        </div>
        <div class="sidebar">
            <div class="panel">
                <h2>📊 Audio Visualization</h2>
                <div class="audio-viz" id="audio-viz"></div>
            </div>
            <div class="panel">
                <h2>🗺️ Spatial View</h2>
                <div class="floorplan" id="floorplan">
                    <div class="sound-source" style="left: 50%; top: 30%"></div>
                    <div class="track" style="left: 55%; top: 35%"></div>
                </div>
            </div>
            <div class="panel">
                <h2>📋 Recent Events</h2>
                <div class="event-list" id="events">
                    <div class="event">
                        <div>🗣️ Speech detected</div>
                        <div class="time">Just now</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onopen = () => console.log('Connected');
        ws.onmessage = (e) => {
            const data = JSON.parse(e.data);
            console.log('Received:', data);
        };
        setInterval(() => ws.send(JSON.stringify({type: 'ping'})), 30000);
    </script>
</body>
</html>'''


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available")
        return
    
    app = create_app()
    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
