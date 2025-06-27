"""online.py – FastAPI + WebRTC offer endpoint with improved AV sync

✅ Streams Ditto avatar video via WebRTC with better synchronization
✅ Maintains consistent FPS for video stream
✅ Improved audio pacing with proper timestamping
✅ Cleaner architecture with separated concerns

Run:  uvicorn online:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import threading
import time
import uuid
from fractions import Fraction
from typing import Optional

import av
import cv2
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from kokoro import KPipeline
from stream_pipeline_online import StreamSDK

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ───────── Configuration ────────────────────────────────────────────────────
CFG_PKL = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
DATA_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
SOURCE_IMG = "static/avatar.png"
PROMPT = (
    "Yes, I came here five years ago, when I was just sixteen. At the time, I was.. "
    "I was still in the tenth grade, and I clearly remember doing my homework in the "
    "backseat of the car as we drove to our new home. Everything felt unfamiliar and "
    "uncertain, but I tried to stay focused on school. I didn't know what to expect, "
    "and it took a while to get used to the language, the people, and the new routines. "
)

# Target FPS for video stream
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Audio configuration
AUDIO_SAMPLE_RATE = 48000
AUDIO_FRAME_SIZE = 960  # 20ms frames at 48kHz

# Initialize pipelines
pipeline = KPipeline(lang_code="a")
sdk = StreamSDK(CFG_PKL, DATA_ROOT, chunk_size=(3, 5, 2))
sdk.online_mode = True
sdk.setup(SOURCE_IMG)

# ───────── Audio Processing Utilities ────────────────────────────────────────
class AudioProcessor:
    def __init__(self):
        self.bytes_per_frame = 640
        self.present = sdk.chunk_size[1] * self.bytes_per_frame
        self.split_len = int(sum(sdk.chunk_size) * self.bytes_per_frame) + 80
        self.idle_flag = threading.Event()
        self._start_idle_thread()

    def _start_idle_thread(self):
        threading.Thread(target=self._idle_loop, daemon=True).start()

    def _idle_loop(self):
        while True:
            self.idle_flag.wait()
            sdk.start_processing_audio()
            self.process_audio(np.zeros(int(13 * 16000), dtype=np.float32))
            time.sleep(0.5)

    def process_audio(self, samples_f32: np.ndarray):
        pos = 0
        while pos < len(samples_f32):
            chunk = samples_f32[pos: pos + self.split_len]
            if len(chunk) < self.split_len:
                chunk = np.pad(chunk, (0, self.split_len - len(chunk)), mode="constant")
            sdk.process_audio_chunk(chunk)
            pos += self.present
        sdk.end_processing_audio()

audio_processor = AudioProcessor()

# ───────── Video Track with Frame Rate Control ────────────────────────────────
class AvatarVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.start_time = time.monotonic()
        self.last_frame_time = 0
        self.loop = asyncio.get_event_loop()

    async def recv(self):
        # Get frame from SDK
        jpg_bytes, _, _ = await self.loop.run_in_executor(
            None, sdk.frame_queue.get
        )
        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")

        # Maintain consistent frame rate
        current_time = time.monotonic() - self.start_time
        time_since_last = current_time - self.last_frame_time
        
        if time_since_last < FRAME_INTERVAL:
            await asyncio.sleep(FRAME_INTERVAL - time_since_last)
            current_time = time.monotonic() - self.start_time

        # Set frame timestamps
        frame.pts = int(current_time * 90_000)
        frame.time_base = Fraction(1, 90_000)
        self.last_frame_time = current_time
        
        return frame

# ───────── Audio Track with Precise Timing ──────────────────────────────────
class PromptAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, samples_f32_48k: np.ndarray):
        super().__init__()
        self.samples = (np.clip(samples_f32_48k, -1.0, 1.0) * 32767).astype(np.int16)
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.frame_size = AUDIO_FRAME_SIZE
        self.position = 0
        self.start_time: Optional[float] = None
        self.sent_samples = 0

    async def recv(self):
        if self.start_time is None:
            self.start_time = time.monotonic()

        # Calculate expected time for next frame
        expected_time = self.start_time + (self.sent_samples / self.sample_rate)
        current_time = time.monotonic()
        
        # Sleep if we're ahead of schedule
        if expected_time > current_time:
            await asyncio.sleep(expected_time - current_time)

        # Get next audio frame
        if self.position < len(self.samples):
            chunk = self.samples[self.position:self.position + self.frame_size]
            if len(chunk) < self.frame_size:
                chunk = np.pad(chunk, (0, self.frame_size - len(chunk)), "constant")
            self.position += self.frame_size
        else:
            chunk = np.zeros(self.frame_size, dtype=np.int16)

        # Create and timestamp audio frame
        frame = av.AudioFrame.from_ndarray(
            chunk.reshape(1, -1), 
            format="s16", 
            layout="mono"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self.sent_samples
        frame.time_base = Fraction(1, self.sample_rate)
        self.sent_samples += self.frame_size

        return frame

# ───────── /offer endpoint ──────────────────────────────────────────────────
@app.post("/offer")
async def offer(offer: dict):
    # 1) Generate TTS
    for _, _, wav24 in pipeline(PROMPT, voice="af_heart", speed=1.1):
        break
    wav24 = np.asarray(wav24, dtype=np.float32)

    # 2) Resample for different purposes
    import librosa
    wav16 = librosa.resample(wav24, orig_sr=24_000, target_sr=16_000)
    wav48 = librosa.resample(wav24, orig_sr=24_000, target_sr=48_000)

    # 3) Set up WebRTC connection
    pc = RTCPeerConnection()
    peer_id = str(uuid.uuid4())

    pc.addTrack(AvatarVideoTrack())
    pc.addTrack(PromptAudioTrack(wav48))

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    )
    await pc.setLocalDescription(await pc.createAnswer())

    # 4) Feed audio to SDK in background
    def _feed_audio():
        audio_processor.idle_flag.clear()
        audio_processor.process_audio(wav16)
        audio_processor.idle_flag.set()

    threading.Thread(target=_feed_audio, daemon=True).start()

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "peer_id": peer_id,
    }