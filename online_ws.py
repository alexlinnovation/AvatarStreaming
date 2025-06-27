"""avatar_streamer.py - WebSocket-based Avatar Streaming with Low Latency"""

import asyncio
import threading
import time
import uuid
import numpy as np
import cv2
import base64
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from kokoro import KPipeline
from stream_pipeline_online import StreamSDK

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
CFG_PKL = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
DATA_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
SOURCE_IMG = "static/avatar.png"
TARGET_FPS = 30
AUDIO_SAMPLE_RATE = 48000
VIDEO_QUALITY = 85  # JPEG quality (1-100)

# Initialize pipelines
sdk = StreamSDK(CFG_PKL, DATA_ROOT, chunk_size=(3, 5, 2))
sdk.online_mode = True
sdk.setup(SOURCE_IMG)
pipeline = KPipeline(lang_code="a")

# Global state
ACTIVE_CONNECTIONS = {}
AUDIO_PROCESSOR = None

class AudioProcessor:
    """Handles audio processing for StreamSDK"""
    def __init__(self):
        self.chunk_size = 640
        self.present = sdk.chunk_size[1] * self.chunk_size
        self.split_len = int(sum(sdk.chunk_size) * self.chunk_size) + 80
        self.lock = threading.Lock()

    def process_audio(self, samples: np.ndarray):
        """Process audio through StreamSDK"""
        with self.lock:
            sdk.start_processing_audio()
            pos = 0
            while pos < len(samples):
                chunk = samples[pos:pos+self.split_len]
                if len(chunk) < self.split_len:
                    chunk = np.pad(chunk, (0, self.split_len-len(chunk)), "constant")
                sdk.process_audio_chunk(chunk)
                pos += self.present
            sdk.end_processing_audio()

def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Fast linear resampling"""
    duration = len(samples) / orig_sr
    num_samples = int(duration * target_sr)
    indices = np.linspace(0, len(samples)-1, num_samples)
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

def generate_silence(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence audio for lip-sync animation"""
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)

@app.on_event("startup")
def startup_event():
    """Initialize global components on startup"""
    global AUDIO_PROCESSOR
    AUDIO_PROCESSOR = AudioProcessor()
    
    # Start with idle animation
    threading.Thread(
        target=AUDIO_PROCESSOR.process_audio, 
        args=(generate_silence(10.0),),
        daemon=True
    ).start()

async def video_stream_generator(websocket: WebSocket, connection_id: str):
    """Generate and send video frames to WebSocket client"""
    last_frame_time = time.time()
    frame_interval = 1.0 / TARGET_FPS
    
    try:
        while ACTIVE_CONNECTIONS.get(connection_id) == websocket:
            # Get frame from SDK
            jpg_bytes, _, _ = await asyncio.get_event_loop().run_in_executor(
                None, sdk.frame_queue.get
            )
            
            # Maintain frame rate
            current_time = time.time()
            time_since_last = current_time - last_frame_time
            if time_since_last < frame_interval:
                await asyncio.sleep(frame_interval - time_since_last)
            
            # Convert to base64 for WebSocket
            img_base64 = base64.b64encode(jpg_bytes).decode('utf-8')
            
            # Send to client
            await websocket.send_json({
                "type": "video",
                "data": img_base64,
                "timestamp": time.time()
            })
            
            last_frame_time = time.time()
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up connection
        if connection_id in ACTIVE_CONNECTIONS:
            del ACTIVE_CONNECTIONS[connection_id]

@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """Main WebSocket endpoint for audio and video streaming"""
    await websocket.accept()
    
    # Store connection
    ACTIVE_CONNECTIONS[connection_id] = websocket
    
    try:
        # Start video streaming task
        video_task = asyncio.create_task(video_stream_generator(websocket, connection_id))
        
        # Audio handling loop
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "audio":
                # Handle audio input from client
                audio_data = np.frombuffer(
                    base64.b64decode(data["data"]), 
                    dtype=np.float32
                )
                threading.Thread(
                    target=AUDIO_PROCESSOR.process_audio,
                    args=(audio_data,),
                    daemon=True
                ).start()
                
            elif data["type"] == "speak":
                # Handle TTS request
                text = data["text"]
                voice_style = data.get("voice_style", "af_heart")
                speed = data.get("speed", 1.1)
                
                # Generate TTS
                for _, _, wav24 in pipeline(text, voice=voice_style, speed=speed):
                    break
                wav24 = np.asarray(wav24, dtype=np.float32)
                
                # Process for animation
                wav16 = resample_audio(wav24, 24000, 16000)
                threading.Thread(
                    target=AUDIO_PROCESSOR.process_audio,
                    args=(wav16,),
                    daemon=True
                ).start()
                
                # Prepare audio for client
                wav48 = resample_audio(wav24, 24000, 48000)
                pcm48 = (np.clip(wav48, -1.0, 1.0) * 32767).astype(np.int16)
                audio_base64 = base64.b64encode(pcm48.tobytes()).decode('utf-8')
                
                # Send to client
                await websocket.send_json({
                    "type": "audio",
                    "data": audio_base64,
                    "sample_rate": 48000,
                    "timestamp": time.time()
                })
                
    except WebSocketDisconnect:
        pass
    finally:
        if connection_id in ACTIVE_CONNECTIONS:
            del ACTIVE_CONNECTIONS[connection_id]
        video_task.cancel()

@app.get("/create_connection")
async def create_connection():
    """Create a new connection ID for WebSocket"""
    connection_id = str(uuid.uuid4())
    return {"connection_id": connection_id}

@app.get("/")
async def serve_client():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Avatar Stream</title>
    </head>
    <body>
        <h2>Avatar Streaming</h2>
        <video id="video" autoplay muted playsinline></video>
        <input type="text" id="text" placeholder="Enter something to speak" />
        <button onclick="speak()">Speak</button>
        <script>
            let ws;
            let video = document.getElementById("video");
            let canvas = document.createElement("canvas");
            let ctx = canvas.getContext("2d");

            async function connect() {
                let res = await fetch("/create_connection");
                let { connection_id } = await res.json();
                ws = new WebSocket(`ws://${location.host}/ws/${connection_id}`);

                ws.onmessage = (event) => {
                    let msg = JSON.parse(event.data);
                    if (msg.type === "video") {
                        let img = new Image();
                        img.onload = () => {
                            canvas.width = img.width;
                            canvas.height = img.height;
                            ctx.drawImage(img, 0, 0);
                            video.srcObject = canvas.captureStream();
                        };
                        img.src = "data:image/jpeg;base64," + msg.data;
                    }
                };
            }

            async function speak() {
                let text = document.getElementById("text").value;
                ws.send(JSON.stringify({ type: "speak", text }));
            }

            connect();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8010)
