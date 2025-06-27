"""
online.py – FastAPI + WebRTC avatar streamer
• POST /offer → idle.mp3 loops continuously
• POST /speak → pauses idle, plays audio.mp3 once, resumes idle
Run: uvicorn online:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio, threading, time, uuid, queue
from fractions import Fraction

import cv2, av, numpy as np, soundfile as sf
from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

from stream_pipeline_online import StreamSDK

# ─── paths ────────────────────────────────────────────
CFG_PKL   = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
DATA_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
SRC_IMG   = "static/avatar.png"
IDLE_FILE = "static/idle.mp3"
TALK_FILE = "static/audio.mp3"

# ─── FastAPI ──────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── StreamSDK shared instance ───────────────────────
sdk = StreamSDK(CFG_PKL, DATA_ROOT, chunk_size=(3, 5, 2))
sdk.online_mode = True
sdk.setup(SRC_IMG)

FPS             = 25
FRAME_INTERVAL  = 1.0 / FPS
BYTES_PER_FRAME = 640
PRESENT         = sdk.chunk_size[1] * BYTES_PER_FRAME
SPLIT_LEN       = int(sum(sdk.chunk_size) * BYTES_PER_FRAME) + 80

# ─── audio helpers ───────────────────────────────────
def load_16k(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr == 16000:
        return data
    # linear resample to 16 kHz
    new_len = int(len(data) * 16000 / sr)
    idx     = np.linspace(0, len(data)-1, new_len, dtype=np.float32)
    base    = idx.astype(np.int32)
    frac    = idx - base
    nxt     = np.clip(base+1, 0, len(data)-1)
    return (data[base]*(1-frac) + data[nxt]*frac).astype(np.float32)

# ─── NEW: simple audio track for WebRTC ─────────────
class AudioTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self, samples: np.ndarray, sr: int = 16000):
        super().__init__()              # don't touch video logic
        self.samples = (samples * 32767).astype(np.int16).tobytes()
        self.sample_rate = sr
        self.pos = 0
        self.frame_size = 960           # 20 ms @48 kHz after browser resample
    async def recv(self):
        # slice out one packet of bytes
        start = self.pos
        end   = start + self.frame_size * 2
        chunk = self.samples[start*2:end*2]
        if not chunk:
            # once speech is done, return silence
            chunk = b"\x00" * (self.frame_size*2)
        self.pos += self.frame_size
        frame = av.AudioFrame(format="s16", layout="mono", samples=self.frame_size)
        frame.planes[0].update(chunk)
        frame.sample_rate = 48000
        frame.pts = self.pos
        frame.time_base = Fraction(1, 48000)
        return frame
# ────────────────────────────────────────────────────

# load both idle & speak into memory once
idle_audio  = load_16k(IDLE_FILE)[:3*16000]    # 3 s slice
speak_audio = load_16k(TALK_FILE)

# pre-slice idle chunk
idle_slice = idle_audio[:SPLIT_LEN] \
    if len(idle_audio) >= SPLIT_LEN else \
    np.pad(idle_audio, (0, SPLIT_LEN - len(idle_audio)), 'constant')

def push_chunk(chunk: np.ndarray):
    sdk.start_processing_audio()
    sdk.process_audio_chunk(chunk)
    sdk.end_processing_audio()

# ─── session state ───────────────────────────────────
sessions = {}   # sid → dict(video_q, idle_evt, kill_evt)

def idle_feeder(idle_evt: threading.Event, kill_evt: threading.Event):
    while not kill_evt.is_set():
        idle_evt.wait()
        if kill_evt.is_set():
            break
        sdk.interrupt()
        push_chunk(idle_slice)

def frame_collector(sid: str, kill_evt: threading.Event):
    q = sessions[sid]["video_q"]
    while not kill_evt.is_set():
        if sdk.has_pending_frames():
            buf, *_ = sdk.frame_queue.get()
            q.put(buf)
        else:
            time.sleep(0.004)

class AvatarVideo(MediaStreamTrack):
    kind = "video"
    def __init__(self, sid):
        super().__init__(); self.sid=sid; self.t0=time.monotonic(); self.last=0.0
    async def recv(self):
        q = sessions[self.sid]["video_q"]
        while q.empty():
            await asyncio.sleep(0.005)
        img_bgr = cv2.imdecode(np.frombuffer(q.get(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(img_rgb, format="rgb24")
        now = time.monotonic()-self.t0
        gap = FRAME_INTERVAL-(now-self.last)
        if gap>0:
            await asyncio.sleep(gap)
            now=time.monotonic()-self.t0
        frame.pts, frame.time_base = int(now*90000), Fraction(1,90000)
        self.last=now
        return frame

# ─── POST /offer ─────────────────────────────────────
@app.post("/offer")
async def offer(offer: dict):
    sid = str(uuid.uuid4())
    idle_evt, kill_evt = threading.Event(), threading.Event()
    idle_evt.set()
    sessions[sid] = {"video_q": queue.Queue(), "idle_evt": idle_evt, "kill_evt": kill_evt}

    threading.Thread(target=idle_feeder,   args=(idle_evt, kill_evt), daemon=True).start()
    threading.Thread(target=frame_collector, args=(sid,kill_evt), daemon=True).start()

    pc = RTCPeerConnection()
    pc.addTrack(AvatarVideo(sid))
    # ─── NEW: attach audio track here ───────────────
    pc.addTrack(AudioTrack(speak_audio, sr=16000))
    # ──────────────────────────────────────────────────

    await pc.setRemoteDescription(RTCSessionDescription(**offer))
    await pc.setLocalDescription(await pc.createAnswer())
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "sessionid": sid
    }

# ─── POST /speak ─────────────────────────────────────
@app.post("/speak")
async def speak(sessionid: str = Form(...)):
    sess = sessions.get(sessionid)
    if not sess:
        return {"error": "unknown session"}

    idle_evt = sess["idle_evt"]

    def speech():
        idle_evt.clear()
        # only clear these three queues—everything else unchanged
        sdk.hubert_features_queue.queue.clear()
        sdk.audio2motion_queue.queue.clear()
        sdk.motion_stitch_queue.queue.clear()

        pos = 0
        while pos < len(speak_audio):
            chunk = speak_audio[pos:pos+SPLIT_LEN]
            if len(chunk) < SPLIT_LEN:
                chunk = np.pad(chunk, (0, SPLIT_LEN-len(chunk)), "constant")
            push_chunk(chunk)
            pos += PRESENT

        idle_evt.set()

    threading.Thread(target=speech, daemon=True).start()
    return {"status": "playing"}
