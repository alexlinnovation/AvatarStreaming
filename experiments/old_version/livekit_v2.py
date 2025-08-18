#!/usr/bin/env python
# avatar_livekit_fixed.py  – StreamSDK avatar → LiveKit Cloud  (idle / speak API)
# ------------------------------------------------------------------------------

import asyncio, logging, uuid, time, sys, io
import numpy as np, torch, torchaudio, av
from PIL import Image
from livekit import api, rtc
import onnxruntime as ort
from stream_pipeline_online import StreamSDK
from kokoro_onnx import Kokoro
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ─── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(title="Avatar‑Server (idle + speak)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ─── LiveKit creds ────────────────────────────────────────────────
LIVEKIT_URL = "wss://wdqwd-i503xu0n.livekit.cloud"
API_KEY     = "APIewTmSoRk6rFS"
API_SECRET  = "XCWjdkZbDW2oj56f0eJjStwLEga2FRfMJzvfJ09WN7aB"

# ─── helpers ───────────────────────────────────────────────────────
_resampler = torchaudio.transforms.Resample(24_000, 16_000).cuda()
def resample(w: np.ndarray) -> np.ndarray:
    return _resampler(torch.from_numpy(w).cuda().unsqueeze(0)).squeeze(0).cpu().numpy()

def viewer_token(room: str) -> str:
    return (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(f"viewer-{uuid.uuid4().hex[:6]}")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )

# ─── per‑room session object ───────────────────────────────────────
class AvatarSession:
    def __init__(self, room: str, avatar_png: str):
        self.room_name  = room
        self.avatar_png = avatar_png
        self.loop       = asyncio.get_event_loop()
        self.sdk = None
        self.kokoro = None
        self.lk_room = None
        self.av_sync = None
        self.video_task = None
        self.silence_task = None
        self.started_at = None
        self.present = None
        self.SPL = None

    async def _silence_loop(self):
        zero_audio   = np.zeros(320, np.float32)
        zero_i16     = (zero_audio * 32767).astype(np.int16).tobytes()
        zero_visual  = np.zeros(self.SPL, np.float32)
        rate = 16_000
        pos = 0
        while True:
            if self.started_at is None:
                self.started_at = time.perf_counter()
            ts = time.perf_counter() - self.started_at
            frame = rtc.AudioFrame(zero_i16, rate, 1, 320)
            await self.av_sync.push(frame, ts)
            if pos % self.present == 0:
                self.sdk.process_audio_chunk(zero_visual)
            pos += 320
            await asyncio.sleep(0.02)

    async def offer(self):
        token = (
            api.AccessToken(API_KEY, API_SECRET)
            .with_identity(f"py-ava-{uuid.uuid4().hex[:6]}")
            .with_grants(api.VideoGrants(room_join=True, room=self.room_name, agent=True))
            .to_jwt()
        )
        self.lk_room = rtc.Room()
        await self.lk_room.connect(LIVEKIT_URL, token)

        vs   = rtc.VideoSource(1080, 800)
        asrc = rtc.AudioSource(16_000, 1, 1000)
        vtr  = rtc.LocalVideoTrack.create_video_track("v", vs)
        atr  = rtc.LocalAudioTrack.create_audio_track("a", asrc)
        await self.lk_room.local_participant.publish_track(vtr)
        await self.lk_room.local_participant.publish_track(atr)

        self.av_sync = rtc.AVSynchronizer(
            audio_source=asrc,
            video_source=vs,
            video_fps=40,
            video_queue_size_ms=0,
            _max_delay_tolerance_ms=0,
        )

        self.sdk = StreamSDK(
            "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
            "./checkpoints/ditto_trt_Ampere_Plus",
            chunk_size=(3, 5, 2),
        )
        self.sdk.online_mode = True
        self.sdk.setup(
            self.avatar_png,
            max_size=1080,
            sampling_timesteps=5,
            emo=4,
            drive_eye=True,
            overlap_v2=65,
        )

        sess = ort.InferenceSession(
            "checkpoints/kokoro-v1.0.onnx",
            providers=[("CUDAExecutionProvider", {"device_id": 0})],
        )
        self.kokoro = Kokoro.from_session(sess, "checkpoints/voices-v1.0.bin")
        self.FPS = 30
        self.present = self.sdk.chunk_size[1] * 640
        self.SPL     = int(sum(self.sdk.chunk_size) * 16_000 / self.FPS) + 80

        self.sdk.start_processing_audio()
        self.video_task   = self.loop.create_task(self._video_loop())
        self.silence_task = self.loop.create_task(self._silence_loop())

    async def _video_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            jpg, *_ = await loop.run_in_executor(None, self.sdk.frame_queue.get)
            rgba = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGBA"), np.uint8)
            if self.started_at is None:
                self.started_at = time.perf_counter()
            ts = time.perf_counter() - self.started_at
            vf = rtc.VideoFrame(rgba.shape[1], rgba.shape[0], rtc.VideoBufferType.RGBA, rgba.tobytes())
            await self.av_sync.push(vf, ts)

    async def speak(self, text: str, voice: str = "af_heart"):
        if self.silence_task:
            self.silence_task.cancel()
            self.silence_task = None
            
        text = "................" + text

        pos = 0
        buf = np.empty(0, np.float32)
        async for chunk24, _ in self.kokoro.create_stream(text, voice=voice, speed=1.0, lang="en-us"):
            buf = np.concatenate([buf, resample(chunk24)])
            while len(buf) >= 320:
                frame, buf = buf[:320], buf[320:]
                if self.started_at is None:
                    self.started_at = time.perf_counter()
                ts = time.perf_counter() - self.started_at
                af = rtc.AudioFrame((frame * 32767).astype(np.int16).tobytes(), 16_000, 1, 320)
                await self.av_sync.push(af, ts)

                if pos % self.present == 0:
                    vis = buf[: self.SPL]
                    if vis.size < self.SPL:
                        vis = np.pad(vis, (0, self.SPL - vis.size))
                    self.sdk.process_audio_chunk(vis)
                pos += 320
                await asyncio.sleep(0)

        if self.silence_task is None:
            self.silence_task = self.loop.create_task(self._silence_loop())

    async def stop(self):
        if self.video_task:
            self.video_task.cancel()
        if self.silence_task:
            self.silence_task.cancel()
        if self.lk_room:
            await self.lk_room.disconnect()

# ─── FastAPI endpoints ─────────────────────────────────────────────
loop = asyncio.get_event_loop()
_sessions: Dict[str, AvatarSession] = {}

class OfferReq(BaseModel):
    room: str
    input_image: Optional[str] = "static/avatar.png"

class OfferResp(BaseModel):
    room: str
    url: str
    token: str

class SpeakReq(BaseModel):
    room: str
    text: str
    voice: Optional[str] = "af_heart"

class StopReq(BaseModel):
    room: str

@app.post("/offer", response_model=OfferResp)
async def offer_endpoint(req: OfferReq):
    room = req.room.strip()
    if not room:
        raise HTTPException(400, "room must not be empty")
    if room in _sessions:
        raise HTTPException(409, "room already exists")
    image_path = req.input_image or "static/avatar.png"
    ses = AvatarSession(room, image_path)
    await ses.offer()
    _sessions[room] = ses
    return OfferResp(room=room, url=LIVEKIT_URL, token=viewer_token(room))

@app.post("/speak")
async def speak_endpoint(req: SpeakReq):
    ses = _sessions.get(req.room)
    if not ses:
        raise HTTPException(404, "room not found")
    await ses.speak(req.text, req.voice or "af_heart")
    return {"status": "ok"}

@app.post("/stop")
async def stop_endpoint(req: StopReq):
    ses = _sessions.pop(req.room, None)
    if not ses:
        raise HTTPException(404, "room not found")
    await ses.stop()
    return {"status": "stopped"}
# ---------------- models -----------------
class TokenReq(BaseModel):
    roomName: str | None = None
    participantId: str | None = None
    participantName: str | None = None
    agentName: str | None = None
    metadata: str | None = None

class TokenResp(BaseModel):
    accessToken: str


# ---------------- endpoint ----------------
@app.post("/token", response_model=TokenResp)
async def token_endpoint(req: TokenReq) -> TokenResp:
    room      = req.roomName or f"room-{uuid.uuid4().hex[:6]}"
    identity  = req.participantId or f"user-{uuid.uuid4().hex[:6]}"

    grant = api.VideoGrants(room_join=True, room=room)

    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_name(req.participantName or identity)
        .with_grants(grant)
    )

    if req.agentName:
        token.room_config = api.RoomConfiguration(
            agents=[api.RoomAgentDispatch(agent_name=req.agentName)]
        )

    return TokenResp(accessToken=token.to_jwt())

app.mount("/", StaticFiles(directory="out", html=True), name="frontend")
@app.get("/")
async def root():
    return FileResponse("out/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010, loop="uvloop", reload=False)
