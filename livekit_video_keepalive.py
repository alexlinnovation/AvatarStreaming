#!/usr/bin/env python
# avatar_livekit_fixed.py  – StreamSDK avatar → LiveKit Cloud  (idle / speak API)
# ------------------------------------------------------------------------------
import json
from src.gated_avsync import GatedAVSynchronizer
import collections
import asyncio, logging, uuid, time, io, queue, threading
import numpy as np, torch, torchaudio, av
from PIL import Image
from livekit import api, rtc
import onnxruntime as ort
from stream_pipeline_online import StreamSDK
from kokoro_onnx import Kokoro
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from livekit.agents.stt import SpeechEventType
from livekit.plugins import deepgram
import aiohttp
from openai import AsyncOpenAI

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
DEEPGRAM_API_KEY="f37043ae1b11b119212b8e75f7cc59b8ca722ac2"
OPEN_AI_KEY = "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"
FORWARD_INTERIM = True
Enable_STT = False

FPS_1 = 25
FPS_2 = 25
CHUNK_SIZE = (2, 4, 2)
SAMPLING_TIMESTEP = 12
RESOLUTION = 800
BUFFER = 320
SILENCE_BUFFER = 320

openai_client = AsyncOpenAI(api_key=OPEN_AI_KEY)

_GLOBAL_AVATAR = None
_chat_history: Dict[str, List[dict]] = {}

class AvatarSession:
    def __init__(self, room: str, avatar_png: str, voice: str = "af_heart",
                 name: Optional[str] = "Alice", desc: Optional[str] = "Interview assistant"):
        self.room_name = room
        self.avatar_png = avatar_png
        self.loop = asyncio.get_event_loop()
        self.sdk: Optional[StreamSDK] = None
        self.kokoro: Optional[Kokoro] = None
        self.lk_room: Optional[rtc.Room] = None
        self.av_sync: Optional[GatedAVSynchronizer] = None
        self.video_task: Optional[asyncio.Task] = None
        self.silence_task: Optional[asyncio.Task] = None
        self.video_thread: Optional[threading.Thread] = None
        self.video_frame_queue: Optional[queue.Queue] = None
        self.video_thread_stop_event: Optional[threading.Event] = None
        self.present: Optional[int] = None
        self.SPL: Optional[int] = None
        self.samples_pushed = 0
        self.buffered_audio: List[bytes] = []
        self.video_frames = 0
        self.FPS = FPS_2
        self.http_session: aiohttp.ClientSession | None = None
        self.speak_lock = asyncio.Lock()
        self.ready_event = asyncio.Event()
        self.voice = voice
        self.name = name
        self.desc = desc
        self._audio_ready_sent = False
        self._video_ready_sent = False
        self.resampler = torchaudio.transforms.Resample(24000, 16000).cuda()
        self.first_greet_done = False
        self.tts_queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self.tts_worker_task: Optional[asyncio.Task] = None
        self._connected: bool = False
        self._closing: bool = False

    def _reset_session_state(self):
        self.av_sync = None
        self.video_task = None
        self.silence_task = None
        self.video_thread = None
        self.video_frame_queue = None
        self.video_thread_stop_event = None
        self.samples_pushed = 0
        self.buffered_audio = []
        self.video_frames = 0
        self._audio_ready_sent = False
        self._video_ready_sent = False
        self.ready_event = asyncio.Event()

    def _resample(self, w: np.ndarray) -> np.ndarray:
        return self.resampler(torch.from_numpy(w).cuda().unsqueeze(0)).squeeze(0).cpu().numpy()

    def is_connected(self) -> bool:
        return bool(self.lk_room) and self._connected

    async def _cleanup_on_disconnect(self):
        if self._closing:
            return
        self._closing = True
        try:
            if self.video_thread_stop_event:
                self.video_thread_stop_event.set()
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=1.5)
            self.video_thread = None
            self.video_thread_stop_event = None
            self.video_frame_queue = None
            for t in (self.video_task, self.silence_task):
                if t:
                    t.cancel()
            for t in (self.video_task, self.silence_task):
                if t:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
            self.video_task = None
            self.silence_task = None
            self.av_sync = None
            self.samples_pushed = 0
            self.video_frames = 0
            self._audio_ready_sent = False
            self.first_greet_done = False
            self.buffered_audio.clear()
            self.ready_event = asyncio.Event()
            self._connected = False
            self.lk_room = None
        finally:
            self._closing = False
        
    async def _signal_audio_ready(self):
        if self._audio_ready_sent or not self.lk_room:
            return
        self._audio_ready_sent = True
        try:
            payload = json.dumps({"ok": True, "ts": time.time()}).encode("utf-8")
            await self.lk_room.local_participant.publish_data(
                payload, reliable=True, topic="agent_audio_ready"
            )
            self.first_greet_done = True
        except Exception as e:
            logging.warning(f"publish_data(agent_audio_ready) failed: {e}")

    async def _republish_ready_for_new_viewers(self):
        if not self.lk_room:
            return
        try:
            payload = json.dumps({"ok": True, "ts": time.time()}).encode("utf-8")
            await self.lk_room.local_participant.publish_data(
                payload, reliable=True, topic="agent_audio_ready"
            )
        except Exception as e:
            logging.warning(f"publish_data(agent_audio_ready ping) failed: {e}")
        
    async def _run_stt_for_track(self, track: rtc.RemoteAudioTrack):
        await self.ready_event.wait()
        stt = deepgram.STT(
            model="nova-3",
            api_key=DEEPGRAM_API_KEY,
            http_session=self.http_session,
            punctuate=False,
            smart_format=False,
        )
        stt_stream = stt.stream()
        audio_stream = rtc.AudioStream(track)
        stt_task = asyncio.create_task(_consume_stt_stream(self, stt_stream))
        try:
            async for evt in audio_stream:
                frame = evt.frame if hasattr(evt, "frame") else evt
                stt_stream.push_frame(frame)
        finally:
            stt_stream.end_input()
            await stt_task

    async def offer(self):
        self._reset_session_state()

        token = (
            api.AccessToken(API_KEY, API_SECRET)
            .with_identity(f"py-ava-{uuid.uuid4().hex[:6]}")
            .with_grants(api.VideoGrants(room_join=True, room=self.room_name, agent=True))
            .to_jwt()
        )
        self.lk_room = rtc.Room()

        @self.lk_room.on("disconnected")
        def _on_disconnected():
            log.info("LiveKit room disconnected")
            self._connected = False
            asyncio.create_task(self._cleanup_on_disconnect())

        await self.lk_room.connect(LIVEKIT_URL, token)
        self._connected = True
        if self.http_session is None:
            timeout = aiohttp.ClientTimeout(
                total=None,
                connect=10,
                sock_connect=10,
                sock_read=120,
            )
            connector = aiohttp.TCPConnector(
                limit=64,
                ttl_dns_cache=300,
                ssl=False,
            )
            self.http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            
        @self.lk_room.on("track_subscribed")
        def _on_track_subscribed(track: rtc.RemoteTrack):
            if isinstance(track, rtc.RemoteAudioTrack) and Enable_STT:
                asyncio.create_task(self._run_stt_for_track(track))

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
        )

        if self.sdk is None:
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
        else:
            self.sdk.reset()

        if self.kokoro is None:
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
        self.ready_event.set()

        if not self.tts_worker_task or self.tts_worker_task.done():
            self.tts_worker_task = self.loop.create_task(self._tts_worker())
        
    async def _greet_when_ready(self, text: str, voice: str):
        while self.video_frames == 0:
            await asyncio.sleep(0.05)
        if not self.first_greet_done:
            self.queue_speak(text, voice or self.voice)
            self.first_greet_done = True

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

    async def speak(self, text: str, voice: Optional[str] = None):
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
        v = (voice or self.voice)
        async for chunk24, _ in self.kokoro.create_stream(text, voice=v, speed=0.9, lang="en-us"):
            buf = np.concatenate([buf, self._resample(chunk24)])
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
                if not self._audio_ready_sent:
                    await self._signal_audio_ready()
                if pos % self.present == 0:
                    vis = buf[: self.SPL]
                    if vis.size < self.SPL:
                        vis = np.pad(vis, (0, self.SPL - vis.size))
                    self.sdk.process_audio_chunk(vis)
                pos += BUFFER
            await asyncio.sleep(0)
        
        self.sdk.audio2motion_queue.queue.clear()
        self.sdk.motion_stitch_queue.queue.clear()
        self.sdk.putback_queue.queue.clear()
        self.sdk.hubert_features_queue.queue.clear()
        self.sdk.motion_stitch_out_queue.queue.clear()
        self.sdk.decode_f3d_queue.queue.clear()
        
        if self.silence_task is None:
            self.silence_task = self.loop.create_task(self._silence_loop())

    def queue_speak(self, text: str, voice: Optional[str] = None):
        v = (voice or self.voice)
        try:
            if self.tts_queue.qsize() > 4:
                while not self.tts_queue.empty():
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
        except Exception:
            pass
        self.tts_queue.put_nowait((text, v))

    async def _tts_worker(self):
        while True:
            text, v = await self.tts_queue.get()
            try:
                async with self.speak_lock:
                    await self.speak(text, voice=v)
            except Exception as e:
                logging.exception(f"TTS worker error: {e}")
            finally:
                self.tts_queue.task_done()

    async def stop(self):
        return


async def _consume_stt_stream(session: AvatarSession, stream):
    try:
        async for ev in stream:
            if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                text = ev.alternatives[0].text.strip()
                if len(text) < 2:
                    continue
                if session.speak_lock.locked():
                    continue
                hist = _chat_history.setdefault(session.room_name, [])
                hist.append({"role": "user", "content": text})
                reply = await gpt4mini_reply(session.room_name, session.name, session.desc)
                if not reply:
                    continue
                session.queue_speak(reply, session.voice)
                hist.append({"role": "assistant", "content": reply})
    finally:
        await stream.aclose()

async def gpt4mini_reply(room: str, name: str, desc: str, max_tokens: int = 200) -> str | None:
    try:
        hist = _chat_history.get(room, [])
        if "interview" in (desc or "").lower():
            role_instructions = (
                f"Your name is {name}, you are simulating an {desc}. "
                "Instead of asking the candidate what they applied for, "
                "assume a job title (e.g., 'Software Engineer') in Sakura System (Company). "
                "Lead the interview with concise questions and 1–2 sentence responses."
            )
        elif "customer service" in (desc or "").lower():
            role_instructions = (
                f"Your name is {name}, you are simulating an {desc}. "
                "Your product is SPISy (HR information system) "
                "engage and lead the conversation. "
                "Stay concise in 1–2 sentences and keep the conversation flowing."
            )
        else:
            role_instructions = (
                f"Your name is {name}, your role is {desc}. "
                "You are designed for realtime STT. "
                "Always lead the conversation politely, "
                "ask engaging questions, and reply concisely in 1–2 sentences."
            )

        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": role_instructions},
                *hist[-8:],
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT error: {e}")
        return None

def viewer_token(room: str) -> str:
    return (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(f"viewer-{uuid.uuid4().hex[:6]}")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )

class OfferReq(BaseModel):
    room: str
    input_image: Optional[str] = "static/avatar.png"
    voice: Optional[str] = "af_heart"
    name: Optional[str] = None
    desc: Optional[str] = None

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

class StatusResp(BaseModel):
    loaded: bool
    room: Optional[str] = None

@app.post("/offer", response_model=OfferResp)
async def offer_endpoint(req: OfferReq):
    global _GLOBAL_AVATAR, _chat_history
    req_room = req.room.strip()
    if not req_room:
        raise HTTPException(400, "room must not be empty")

    if _GLOBAL_AVATAR is None:
        _GLOBAL_AVATAR = AvatarSession(
            req_room,
            req.input_image or "static/avatar.png",
            req.voice or "af_heart",
            req.name or "Assistant",
            req.desc or "Realtime assistant"
        )
        await _GLOBAL_AVATAR.offer()
        asyncio.create_task(
            _GLOBAL_AVATAR._greet_when_ready(
                f"...Hello, nice to meet you, my name is {_GLOBAL_AVATAR.name}, I am your {_GLOBAL_AVATAR.desc}, and how can I help you today?",
                _GLOBAL_AVATAR.voice,
            )
        )
    else:
        _GLOBAL_AVATAR.room_name = req_room or _GLOBAL_AVATAR.room_name
        if _GLOBAL_AVATAR.lk_room is None or not _GLOBAL_AVATAR.is_connected():
            await _GLOBAL_AVATAR.offer()
            if not _GLOBAL_AVATAR.first_greet_done:
                asyncio.create_task(
                    _GLOBAL_AVATAR._greet_when_ready(
                        f"...Hello, nice to meet you, my name is {_GLOBAL_AVATAR.name}, I am your {_GLOBAL_AVATAR.desc}, and how can I help you today?",
                        _GLOBAL_AVATAR.voice,
                    )
                )
        else:
            await _GLOBAL_AVATAR._republish_ready_for_new_viewers()

    if req_room not in _chat_history:
        _chat_history[req_room] = []

    active_room = _GLOBAL_AVATAR.room_name
    return OfferResp(room=active_room, url=LIVEKIT_URL, token=viewer_token(active_room))

@app.get("/status", response_model=StatusResp)
async def status_endpoint():
    ses = _GLOBAL_AVATAR
    if not ses:
        return StatusResp(loaded=False, room=None)
    return StatusResp(loaded=bool(ses.first_greet_done), room=ses.room_name if ses else None)

@app.post("/speak")
async def speak_endpoint(req: SpeakReq):
    ses = _GLOBAL_AVATAR
    if not ses or (ses.room_name != req.room and ses.lk_room is None):
        raise HTTPException(404, "room not found")
    ses.queue_speak(req.text, (req.voice or ses.voice))
    return {"status": "ok"}

@app.post("/stop")
async def stop_endpoint(req: StopReq):
    ses = _GLOBAL_AVATAR
    if not ses or (ses.room_name != req.room and ses.lk_room is None):
        raise HTTPException(404, "room not found")
    await ses.stop()
    return {"status": "stopped"}

@app.post("/token", response_model=TokenResp)
async def token_endpoint(req: TokenReq) -> TokenResp:
    room = req.roomName or (_GLOBAL_AVATAR.room_name if _GLOBAL_AVATAR else f"room-{uuid.uuid4().hex[:6]}")
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
