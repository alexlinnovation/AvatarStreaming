#!/usr/bin/env python
# avatar_livekit_fixed.py  – StreamSDK avatar → LiveKit Cloud  (idle / speak API)
# ------------------------------------------------------------------------------
from gated_avsync import GatedAVSynchronizer
import collections
import asyncio, logging, uuid, time, io, queue, threading
import numpy as np, torch, torchaudio, av
from PIL import Image
from livekit import api, rtc
import onnxruntime as ort
from stream_pipeline_online import StreamSDK
from kokoro_onnx import Kokoro
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

log = logging.getLogger("speak_debug")
log.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Avatar-Server (idle + speak)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

LIVEKIT_URL = "wss://wdqwd-i503xu0n.livekit.cloud"
API_KEY = "APIewTmSoRk6rFS"
API_SECRET = "XCWjdkZbDW2oj56f0eJjStwLEga2FRfMJzvfJ09WN7aB"

FPS_1 = 25
FPS_2 = 25
CHUNK_SIZE = (2, 4, 2)
SAMPLING_TIMESTEP = 8
RESOLUTION = 1080
BUFFER = 160
SILENCE_BUFFER = 160

_resampler = None
_resampler_lock = threading.Lock()

def get_resampler():
    global _resampler
    with _resampler_lock:
        if _resampler is None:
            _resampler = torchaudio.transforms.Resample(24_000, 16_000).cuda()
        return _resampler

def resample(w: np.ndarray) -> np.ndarray:
    resampler = get_resampler()
    return resampler(torch.from_numpy(w).cuda().unsqueeze(0)).squeeze(0).cpu().numpy()

def viewer_token(room: str) -> str:
    return (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(f"viewer-{uuid.uuid4().hex[:6]}")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )

class AvatarSession:
    def __init__(self, room: str, avatar_png: str):
        self.room_name = room
        self.avatar_png = avatar_png
        self.loop = asyncio.get_event_loop()
        self.sdk = None
        self.kokoro = None
        self.lk_room = None
        self.av_sync = None
        self.video_task = None
        self.silence_task = None
        self.video_thread = None
        self.video_frame_queue = None
        self.video_thread_stop_event = None
        self.present = None
        self.SPL = None
        self.samples_pushed = 0
        self.buffered_audio: List[bytes] = []
        self.video_frames = 0
        self.FPS = FPS_2

    async def _silence_loop(self):
        zero_audio = np.zeros(SILENCE_BUFFER, np.float32)
        zero_i16 = (zero_audio * 32767).astype(np.int16).tobytes()
        zero_visual = np.zeros(self.SPL, np.float32)
        rate = 16_000
        pos = 0
        while True:
            ts = self.samples_pushed / rate
            frame = rtc.AudioFrame(zero_i16, rate, 1, SILENCE_BUFFER)
            await self.av_sync.push(frame, ts)
            self.samples_pushed += SILENCE_BUFFER
            if pos % self.present == 0:
                self.sdk.process_audio_chunk(zero_visual)
            pos += SILENCE_BUFFER
            try:
                await asyncio.sleep(SILENCE_BUFFER / rate)
            except asyncio.CancelledError:
                break

    async def offer(self):
        token = (
            api.AccessToken(API_KEY, API_SECRET)
            .with_identity(f"py-ava-{uuid.uuid4().hex[:6]}")
            .with_grants(api.VideoGrants(room_join=True, room=self.room_name, agent=True))
            .to_jwt()
        )
        self.lk_room = rtc.Room()
        await self.lk_room.connect(LIVEKIT_URL, token)
        vs = rtc.VideoSource(1080, 800)
        asrc = rtc.AudioSource(16_000, 1, 1000)
        vtr = rtc.LocalVideoTrack.create_video_track("v", vs)
        atr = rtc.LocalAudioTrack.create_audio_track("a", asrc)
        await self.lk_room.local_participant.publish_track(vtr)
        await self.lk_room.local_participant.publish_track(atr)
        self.av_sync = GatedAVSynchronizer(
            audio_source=asrc,
            video_source=vs,
            video_fps=FPS_1,
            video_queue_size_ms=100,
            _max_delay_tolerance_ms=20,
        )
        self.sdk = StreamSDK(
            "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
            "./checkpoints/ditto_trt_custom",
            chunk_size=CHUNK_SIZE,
        )
        self.sdk.online_mode = True
        self.sdk.setup(
            self.avatar_png,
            max_size=RESOLUTION,
            sampling_timesteps=SAMPLING_TIMESTEP,
            emo=4,
            drive_eye=True,
        )
        sess = ort.InferenceSession(
            "checkpoints/kokoro-v1.0.onnx",
            providers=[("CUDAExecutionProvider", {"device_id": 0})],
        )
        self.kokoro = Kokoro.from_session(sess, "checkpoints/voices-v1.0.bin")
        self.present = self.sdk.chunk_size[1] * 640
        self.SPL = int(sum(self.sdk.chunk_size) * 16_000 / self.FPS) + 80
        self.sdk.start_processing_audio()
        self._start_video_thread()
        self.video_task = self.loop.create_task(self._video_loop_push())
        self.silence_task = self.loop.create_task(self._silence_loop())
        
    async def _greet_when_ready(self, text: str, voice: str):
        while self.video_frames == 0:
            await asyncio.sleep(0.05)
        await self.speak(text, voice)

    def _start_video_thread(self):
        self.video_frame_queue = queue.Queue(maxsize=10)
        self.video_thread_stop_event = threading.Event()
        self.video_thread = threading.Thread(target=self._video_processing_worker, daemon=True)
        self.video_thread.start()

    def _video_processing_worker(self):
        while not self.video_thread_stop_event.is_set():
            frame_data_list = self.sdk.frame_queue.get()
            if frame_data_list and isinstance(frame_data_list[0], bytes):
                rgba = np.asarray(Image.open(io.BytesIO(frame_data_list[0])).convert("RGBA"), np.uint8)
                vf = rtc.VideoFrame(rgba.shape[1], rgba.shape[0], rtc.VideoBufferType.RGBA, rgba.tobytes())
                try:
                    self.video_frame_queue.put(vf, timeout=0.05)
                except queue.Full:
                    pass

    async def _video_loop_push(self):
        rate = 1 / self.FPS
        while True:
            try:
                vf = self.video_frame_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(rate)
                continue
            ts = self.video_frames / self.FPS
            await self.av_sync.push(vf, ts)
            self.video_frames += 1
            if self.video_frames == 1 and self.buffered_audio:
                for af_bytes in self.buffered_audio:
                    ts_flush = self.samples_pushed / 16_000
                    af = rtc.AudioFrame(af_bytes, 16_000, 1, BUFFER)
                    await self.av_sync.push(af, ts_flush)
                    self.samples_pushed += BUFFER
                self.buffered_audio.clear()
            await asyncio.sleep(rate)

    async def speak(self, text: str, voice: str = "af_heart"):
        if self.silence_task:
            self.silence_task.cancel()
            try:
                await self.silence_task
            except asyncio.CancelledError:
                pass
            self.silence_task = None
        rate = 16_000
        pos = 0
        buf = np.empty(0, np.float32)
        async for chunk24, _ in self.kokoro.create_stream(text, voice=voice, speed=1.0, lang="en-us"):
            buf = np.concatenate([buf, resample(chunk24)])
            while len(buf) >= BUFFER:
                frame, buf = buf[:BUFFER], buf[BUFFER:]
                af_bytes = (frame * 32767).astype(np.int16).tobytes()
                if self.video_frames == 0:
                    self.buffered_audio.append(af_bytes)
                    self.samples_pushed += BUFFER
                else:
                    ts = self.samples_pushed / rate
                    af = rtc.AudioFrame(af_bytes, rate, 1, BUFFER)
                    await self.av_sync.push(af, ts)
                    self.samples_pushed += BUFFER
                if pos % self.present == 0:
                    vis = buf[: self.SPL]
                    if vis.size < self.SPL:
                        vis = np.pad(vis, (0, self.SPL - vis.size))
                    self.sdk.process_audio_chunk(vis)
                pos += BUFFER
            await asyncio.sleep(0)
        if self.silence_task is None:
            self.silence_task = self.loop.create_task(self._silence_loop())


    async def stop(self):
        if self.video_thread_stop_event:
            self.video_thread_stop_event.set()
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)
        if self.video_task:
            self.video_task.cancel()
            try:
                await self.video_task
            except asyncio.CancelledError:
                pass
        if self.silence_task:
            self.silence_task.cancel()
            try:
                await self.silence_task
            except asyncio.CancelledError:
                pass
        if self.lk_room:
            await self.lk_room.disconnect()
        if self.sdk:
            try:
                self.sdk.close()
            except Exception:
                pass

_sessions: Dict[str, AvatarSession] = {}

class OfferReq(BaseModel):
    room: str
    input_image: Optional[str] = "static/avatar.png"
    voice: Optional[str] = "af_heart"

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

class TokenReq(BaseModel):
    roomName: str | None = None
    participantId: str | None = None
    participantName: str | None = None
    agentName: str | None = None
    metadata: str | None = None

class TokenResp(BaseModel):
    accessToken: str

class ConfigReq(BaseModel):
    FPS_1: Optional[int] = None
    FPS_2: Optional[int] = None
    CHUNK_SIZE: Optional[List[int]] = None
    SAMPLING_TIMESTEP: Optional[int] = None

@app.post("/offer", response_model=OfferResp)
async def offer_endpoint(req: OfferReq):
    room = req.room.strip()
    if not room:
        raise HTTPException(400, "room must not be empty")
    if room in _sessions:
        raise HTTPException(409, "room already exists")
    ses = AvatarSession(room, req.input_image or "static/avatar.png")
    await ses.offer()
    _sessions[room] = ses
    asyncio.create_task(
        ses._greet_when_ready(
            "...Hello, nice to meet you, how can I help you today?",
            req.voice or "af_heart",
        )
    )
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

@app.post("/token", response_model=TokenResp)
async def token_endpoint(req: TokenReq) -> TokenResp:
    room = req.roomName or f"room-{uuid.uuid4().hex[:6]}"
    identity = req.participantId or f"user-{uuid.uuid4().hex[:6]}"
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

@app.post("/config")
async def config_endpoint(req: ConfigReq):
    global FPS_1, FPS_2, CHUNK_SIZE, SAMPLING_TIMESTEP
    if req.FPS_1 is not None:
        FPS_1 = req.FPS_1
    if req.FPS_2 is not None:
        FPS_2 = req.FPS_2
    if req.CHUNK_SIZE is not None:
        if len(req.CHUNK_SIZE) != 3:
            raise HTTPException(400, "CHUNK_SIZE must have exactly 3 values")
        CHUNK_SIZE = tuple(req.CHUNK_SIZE)
    if req.SAMPLING_TIMESTEP is not None:
        SAMPLING_TIMESTEP = req.SAMPLING_TIMESTEP
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="out", html=True), name="frontend")

@app.get("/")
async def root():
    return FileResponse("out/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010, loop="uvloop", reload=False)
