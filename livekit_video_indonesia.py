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
# NOTE: Kokoro/ORT removed because Indonesian TTS now uses OpenAI TTS (wav → PCM)
# import onnxruntime as ort
# from kokoro_onnx import Kokoro
from stream_pipeline_online import StreamSDK
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from livekit.agents.stt import SpeechEventType, SpeechEvent
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
ENABLE_STT = True  # set True to enable Indonesian STT from mic

FPS_1 = 25
FPS_2 = 25
CHUNK_SIZE = (2, 4, 2)
SAMPLING_TIMESTEP = 12
RESOLUTION = 1080
BUFFER = 320
SILENCE_BUFFER = 320

openai_client = AsyncOpenAI(api_key=OPEN_AI_KEY)

# --- helpers (resample + drain) ------------------------------------------------
def _drain(q: queue.Queue):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return

def _resample_1d_float32(wave: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    if src_hz == dst_hz:
        return wave.astype(np.float32, copy=False)
    # CPU resample is fine here; GPU not required
    rs = torchaudio.transforms.Resample(src_hz, dst_hz)
    with torch.inference_mode():
        out = rs(torch.from_numpy(wave).unsqueeze(0)).squeeze(0).cpu().numpy()
    return out.astype(np.float32, copy=False)

# --- TTS (Bahasa Indonesia via OpenAI TTS → wav → float32 16k) -----------------
async def tts_id_to_float16k(http: aiohttp.ClientSession, text: str, voice: str = "alloy") -> np.ndarray:
    """
    Returns mono float32 PCM at 16kHz for LiveKit/StreamSDK visemes.
    Uses OpenAI TTS (no Azure). Language is inferred from Indonesian text.
    """
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPEN_AI_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini-tts",
        "voice": "coral",         # choose any voice supported (e.g., alloy, verse, aria, etc.)
        "input": text,
        "format": "wav",        # request WAV for reliable decode
    }
    async with http.post(url, headers=headers, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"TTS HTTP {resp.status}: {body}")

        wav_bytes = await resp.read()

    # Decode WAV to float32 using PyAV, resample → 16k mono
    container = av.open(io.BytesIO(wav_bytes))
    samples: List[np.ndarray] = []
    src_rate = None
    num_channels = None
    for frame in container.decode(audio=0):
        src_rate = frame.sample_rate
        pcm = frame.to_ndarray()  # (channels, n)
        if pcm.ndim == 2:
            num_channels = pcm.shape[0]
            pcm = pcm.mean(axis=0)  # mixdown to mono
        samples.append(pcm.astype(np.float32, copy=False))
    container.close()

    if not samples:
        return np.zeros(0, dtype=np.float32)

    pcm_all = np.concatenate(samples)
    if src_rate is None:
        src_rate = 24000  # safe default if header missing

    pcm_16k = _resample_1d_float32(pcm_all, src_rate, 16000)
    # clamp to [-1,1] just in case
    np.clip(pcm_16k, -1.0, 1.0, out=pcm_16k)
    return pcm_16k

# --- viewer token ---------------------------------------------------------------
def viewer_token(room: str) -> str:
    return (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(f"viewer-{uuid.uuid4().hex[:6]}")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )

# --- Avatar session -------------------------------------------------------------
class AvatarSession:
    def __init__(self, room: str, avatar_png: str, voice: str = "alloy", name: Optional[str] = "Alice", desc: Optional[str] = "Asisten wawancara"):
        self.room_name = room
        self.avatar_png = avatar_png
        self.loop = asyncio.get_event_loop()
        self.sdk = None
        # self.kokoro = None  # removed
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
        self.http_session: aiohttp.ClientSession | None = None
        self.speak_lock = asyncio.Lock()
        self.ready_event = asyncio.Event()
        self.voice = voice          # OpenAI TTS voice name
        self.name = name
        self.desc = desc

    async def _run_stt_for_track(self, track: rtc.RemoteAudioTrack):
        await self.ready_event.wait()
        stt = deepgram.STT(
            model="nova-2",                 # Indonesian supported here
            api_key=DEEPGRAM_API_KEY,
            http_session=self.http_session,
            punctuate=True,
            smart_format=True,
            language="id",                 # Bahasa Indonesia
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
        token = (
            api.AccessToken(API_KEY, API_SECRET)
            .with_identity(f"py-ava-{uuid.uuid4().hex[:6]}")
            .with_grants(api.VideoGrants(room_join=True, room=self.room_name, agent=True))
            .to_jwt()
        )
        self.lk_room = rtc.Room()
        await self.lk_room.connect(LIVEKIT_URL, token)
        if self.http_session is None:
            timeout = aiohttp.ClientTimeout(
                total=None,
                connect=10,
                sock_connect=10,
                sock_read=180,
            )
            connector = aiohttp.TCPConnector(
                limit=64,
                ttl_dns_cache=300,
                ssl=False,
            )
            self.http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)

        @self.lk_room.on("track_subscribed")
        def _on_track_subscribed(track: rtc.RemoteTrack):
            if ENABLE_STT and isinstance(track, rtc.RemoteAudioTrack):
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

        # Indonesian flow: TTS is OpenAI; no Kokoro session
        # sess = ort.InferenceSession(...)  # removed
        # self.kokoro = Kokoro.from_session(...)

        self.present = self.sdk.chunk_size[1] * 640
        self.SPL = int(sum(self.sdk.chunk_size) * 16_000 / self.FPS) + 80
        self.sdk.start_processing_audio()
        self._start_video_thread()
        self.video_task = self.loop.create_task(self._video_loop_push())
        self.silence_task = self.loop.create_task(self._silence_loop())
        self.ready_event.set()

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

    async def speak(self, text: str, voice: str = "alloy"):
        # Bahasa Indonesia TTS path (OpenAI TTS → 16k PCM)
        if self.silence_task:
            self.silence_task.cancel()
            try:
                await self.silence_task
            except asyncio.CancelledError:
                pass
            self.silence_task = None

        rate = 16_000
        pos = 0
        # Generate full utterance, then stream by BUFFER frames to preserve gating
        pcm_16k = await tts_id_to_float16k(self.http_session, text, voice=voice)
        buf = pcm_16k.astype(np.float32, copy=False)

        idx = 0
        while idx + BUFFER <= buf.size:
            frame = buf[idx: idx + BUFFER]
            af_bytes = (frame * 32767.0).astype(np.int16).tobytes()
            if self.video_frames == 0:
                self.buffered_audio.append(af_bytes)
                self.samples_pushed += BUFFER
            else:
                ts = self.samples_pushed / rate
                af = rtc.AudioFrame(af_bytes, rate, 1, BUFFER)
                await self.av_sync.push(af, ts)
                self.samples_pushed += BUFFER

            if (pos % self.present) == 0:
                # feed viseme driver
                vis = buf[idx: idx + self.SPL]
                if vis.size < self.SPL:
                    vis = np.pad(vis, (0, self.SPL - vis.size))
                self.sdk.process_audio_chunk(vis)
            pos += BUFFER
            idx += BUFFER
            await asyncio.sleep(0)  # yield to loop

        # cleanup queues to avoid drift between utterances
        self.sdk.audio2motion_queue.queue.clear()
        self.sdk.motion_stitch_queue.queue.clear()
        self.sdk.putback_queue.queue.clear()
        self.sdk.hubert_features_queue.queue.clear()
        self.sdk.motion_stitch_out_queue.queue.clear()
        self.sdk.decode_f3d_queue.queue.clear()

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

# --- STT consumer → GPT (Bahasa Indonesia) ------------------------------------
async def _consume_stt_stream(session: AvatarSession, stream):
    try:
        async for ev in stream:
            if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                text = ev.alternatives[0].text.strip()
                if len(text) < 2:
                    continue

                hist = _chat_history.setdefault(session.room_name, [])
                hist.append({"role": "user", "content": text})

                async def handle_reply(user_text: str):
                    reply = await gpt4mini_reply(session.room_name, session.name, session.desc)
                    if reply:
                        async with session.speak_lock:
                            await session.speak(reply, session.voice)
                        hist.append({"role": "assistant", "content": reply})

                if not session.speak_lock.locked():
                    asyncio.create_task(handle_reply(text))

    finally:
        await stream.aclose()

async def gpt4mini_reply(room: str, name: str, desc: str, max_tokens: int = 200) -> str | None:
    """Force Indonesian replies from GPT."""
    try:
        hist = _chat_history.get(room, [])

        # Bahasa Indonesia instructions (branch on desc but keep language consistent)
        if "wawancara" in (desc or "").lower() or "interview" in (desc or "").lower():
            role_instructions = (
                f"Namamu {name}. Kamu mensimulasikan {desc}. "
                "Jangan tanya lowongan apa; anggap peran 'Software Engineer'. "
                "Bertanyalah singkat dan jawab dalam 1–2 kalimat. Gunakan Bahasa Indonesia."
            )
        elif "customer service" in (desc or "").lower():
            role_instructions = (
                f"Namamu {name}. Kamu mensimulasikan {desc}. "
                "Produkmu adalah avatar digital (punya API). "
                "Pimpin percakapan, jawab ringkas 1–2 kalimat memakai Bahasa Indonesia."
            )
        else:
            role_instructions = (
                f"Namamu {name}. Peranmu: {desc}. "
                "Sistem dirancang untuk STT realtime. "
                "Selalu sopan, pancing percakapan, dan jawab ringkas 1–2 kalimat dalam Bahasa Indonesia."
            )

        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            temperature=0.6,
            messages=[
                {"role": "system", "content": role_instructions},
                *hist[-8:],
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"GPT error: {e}")
        return None

# --- API models ----------------------------------------------------------------
_sessions: Dict[str, AvatarSession] = {}
_chat_history: Dict[str, List[dict]] = {}

class OfferReq(BaseModel):
    room: str
    input_image: Optional[str] = "static/avatar.png"
    voice: Optional[str] = "alloy"  # OpenAI TTS voice (Bahasa content)
    name: Optional[str] = None
    desc: Optional[str] = None

class OfferResp(BaseModel):
    room: str
    url: str
    token: str

class SpeakReq(BaseModel):
    room: str
    text: str
    voice: Optional[str] = "alloy"

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

# --- Endpoints -----------------------------------------------------------------
@app.post("/offer", response_model=OfferResp)
async def offer_endpoint(req: OfferReq):
    room = req.room.strip()
    if not room:
        raise HTTPException(400, "room must not be empty")
    if room in _sessions:
        raise HTTPException(409, "room already exists")

    ses = AvatarSession(
        room,
        req.input_image or "static/avatar.png",
        req.voice or "alloy",
        req.name or "Asisten",
        req.desc or "Asisten Realtime"
    )
    await ses.offer()
    _sessions[room] = ses

    # Mulai percakapan dalam Bahasa Indonesia
    _chat_history[room] = [
        {
            "role": "system",
            "content": "Kamu asisten untuk STT realtime. "
                       "Mulailah dengan perkenalan singkat dalam Bahasa Indonesia "
                       "dan ajukan satu pertanyaan pembuka."
        }
    ]

    async def _intro_task():
        reply = await gpt4mini_reply(room, max_tokens=40, name=ses.name, desc=ses.desc)
        if reply:
            await ses._greet_when_ready(reply, ses.voice)
            _chat_history[room].append({"role": "assistant", "content": reply})

    asyncio.create_task(_intro_task())
    return OfferResp(room=room, url=LIVEKIT_URL, token=viewer_token(room))

@app.post("/speak")
async def speak_endpoint(req: SpeakReq):
    ses = _sessions.get(req.room)
    if not ses:
        raise HTTPException(404, "room not found")
    await ses.speak(req.text, req.voice or "alloy")
    return {"status": "ok"}

@app.post("/stop")
async def stop_endpoint(req: StopReq):
    ses = _sessions.pop(req.room, None)
    _chat_history.pop(req.room, None)
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
