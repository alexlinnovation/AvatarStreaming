"""
online.py – FastAPI + WebRTC avatar streamer
• POST /offer → idle.mp3 loops continuously
• POST /speak → pauses idle, plays audio.mp3 once, resumes idle
"""
import asyncio
import threading, time, uuid, queue
from fractions import Fraction
import librosa
import requests
import torch, torchaudio
import cv2, numpy as np, soundfile as sf
from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription
from stream_pipeline_online import StreamSDK
from src.webrtc import HumanPlayer, AUDIO_FRAME_SAMP
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
from kokoro_onnx import Kokoro
from onnxruntime.capi._pybind_state import set_default_logger_severity
set_default_logger_severity(3)

# ─── config & paths ───────────────────────────────────────────────────
CFG_PKL   = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
DATA_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
SRC_IMG   = "static/avatar.png"
IDLE_FILE = "static/audio_with_silence.wav"
BYTES_PER_FRAME = 640 
IDLE_AUDIO = np.zeros(16000, dtype=np.float32)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ONNX Session Options for Performance ─────────────────────────────
def initialize_kokoro():
    sess_opts = ort.SessionOptions()
    sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.add_session_config_entry("arena_extend_strategy", "kNextPowerOfTwo")

    cuda_provider = (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": True,
        },
    )
    sess = ort.InferenceSession("checkpoints/kokoro-v1.0.onnx", sess_options=sess_opts, providers=[cuda_provider])
    kokoro = Kokoro.from_session(session=sess, voices_path="checkpoints/voices-v1.0.bin")
    audio, sr = kokoro.create("Dummy initialization text", voice="af_heart",speed=1.0,lang="en-us")
    sf.write("test.wav", audio, sr)
    return kokoro


# ─── helpers ──────────────────────────────────────────────────────────
resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000).cuda()
def resample_torch(wav24: np.ndarray) -> np.ndarray:
    wav_tensor = torch.tensor(wav24, dtype=torch.float32, device='cuda').unsqueeze(0)
    wav16 = resampler(wav_tensor)
    return wav16.squeeze(0).cpu().numpy()

def load_16k(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        new_len = int(len(data) * 16000 / sr)
        idx  = np.linspace(0, len(data) - 1, new_len, dtype=np.float32)
        base = idx.astype(np.int32)
        frac = idx - base
        nxt  = np.clip(base + 1, 0, len(data) - 1)
        data = (data[base] * (1 - frac) + data[nxt] * frac).astype(np.float32)
    return data

def new_sdk(src_img: str = SRC_IMG) -> StreamSDK:
    sdk = StreamSDK(CFG_PKL, DATA_ROOT, chunk_size=(2, 4, 2))
    sdk.online_mode = True
    sdk.setup(
        src_img,
        max_size=1980,
        sampling_timesteps=15,
        emo=4,
        drive_eye=True,
        fix_kp_cond=0, 
        v_min_max_for_clip=None, 
        # overlap_v2=68,
    )
    return sdk

def _drain(q: queue.Queue):
    while True:
        try: q.get_nowait()
        except queue.Empty: return

sessions: dict[str, dict] = {}

# ─── background workers ──────────────────────────────────────────────
def idle_feeder(sdk: StreamSDK, idle_evt: threading.Event, stop_evt: threading.Event, idle_slice: np.ndarray):
    while not stop_evt.is_set():
        idle_evt.wait()
        if stop_evt.is_set():
            break
        sdk.interrupt()                # reset for idle
        sdk.start_processing_audio()
        sdk.process_audio_chunk(np.zeros(sdk.split_len, dtype=np.float32))
        sdk.end_processing_audio()
        torch.cuda.synchronize()

def frame_collector(sdk: StreamSDK, player: HumanPlayer, stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            buf, *_ = sdk.frame_queue.get(timeout=0.02)
            player.push_video(buf)
        except queue.Empty:
            continue

# ─── Endpoints ─────────────────────────────────────────────────────
class OfferModel(BaseModel):
    sdp: str
    type: str
    src_img: str | None = None
    
@app.post("/offer")
async def offer(offer: OfferModel):
    sid = str(uuid.uuid4())
    idle_evt = threading.Event();  idle_evt.set()
    kill_evt = threading.Event()
    sdk = new_sdk(offer.src_img or SRC_IMG)
    kokoro = initialize_kokoro()

    present    = sdk.chunk_size[1] * BYTES_PER_FRAME
    # split_len  = int(sum(sdk.chunk_size) * BYTES_PER_FRAME) + 80
    TARGET_FPS = 35
    AUDIO_PER_FRAME =  640 #int(16000 / TARGET_FPS)
    split_len  = (sdk.chunk_size[0] + sdk.chunk_size[1] + sdk.chunk_size[2]) \
             * AUDIO_PER_FRAME  + 80
    idle_slice = np.pad(IDLE_AUDIO[:split_len], (0, max(0, split_len - len(IDLE_AUDIO))), 'constant')

    player = HumanPlayer()
    sessions[sid] = {
        "sdk": sdk,
        "kokoro": kokoro,
        "idle_evt": idle_evt,
        "kill_evt": kill_evt,
        "player": player,
        "speech_stop": None,
        "speech_thread": None,
        "present": present,
        "split_len": split_len
    }

    threading.Thread(target=idle_feeder, args=(sdk, idle_evt, kill_evt, idle_slice), daemon=True).start()
    threading.Thread(target=frame_collector, args=(sdk, player, kill_evt), daemon=True).start()

    pc = RTCPeerConnection()
    pc.addTrack(player.video)
    pc.addTrack(player.audio)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
    await pc.setLocalDescription(await pc.createAnswer())
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sid}

# put this near your other constants
SILENCE_SEC      = 1.05
SILENCE_SAMPLES  = int(SILENCE_SEC * 16000)          # 1-second, 16 kHz
SILENCE_FRAME_I16 = (np.zeros(AUDIO_FRAME_SAMP).astype(np.float32) * 32767).astype(np.int16)
@app.post("/speak", response_model=dict)
async def speak(
    sessionid: str = Form(...),
    text: str = Form(None),
    voice_style: str = Form(None),
    speed: float = Form(None),
):
    sess = sessions.get(sessionid)
    if not sess:
        return {"error": "unknown session"}

    sdk        = sess["sdk"]
    idle_evt   = sess["idle_evt"]
    player     = sess["player"]
    present    = sess["present"]
    split_len  = sess["split_len"]
    kokoro     = sess["kokoro"]

    text = text or (
        "Yes, I came here five years ago, when I was just sixteen. "
        "At the time, I was.. I was still in the tenth grade, and I clearly "
        "remember doing my homework in the backseat of the car as we drove to our new home. "
        "Everything felt unfamiliar and uncertain, but I tried to stay focused on school. "
        "I didn’t know what to expect, and it took a while to get used to the language, the people, and the new routines."
    )
    voice_style = (voice_style or "af_heart").strip() or "af_heart"
    speed       = speed or 1.1

    # Stop any running speech worker
    prev_stop = sess.get("speech_stop")
    prev_thr  = sess.get("speech_thread")
    if prev_thr and prev_thr.is_alive():
        prev_stop.set()
        prev_thr.join(timeout=1.0)
    stop_evt = threading.Event()
    sess["speech_stop"]   = stop_evt
    sess["speech_thread"] = None

    # Flush (but do NOT touch timestamps)
    def hard_flush_player_audio(player):
        try:
            while True:
                player._aud_q.get_nowait()
        except Exception:
            pass
        if hasattr(player.audio, "_queue"):
            try:
                while True:
                    player.audio._queue.get_nowait()
            except Exception:
                pass

    def speech_worker():
        idle_evt.clear()
        for q in (
            sdk.hubert_features_queue, sdk.audio2motion_queue, 
            sdk.motion_stitch_queue, sdk.frame_queue, sdk.decode_f3d_queue, 
            sdk.warp_f3d_queue, sdk.motion_stitch_out_queue
        ):
            _drain(q)
        hard_flush_player_audio(player)
        sdk.start_processing_audio()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fut = loop.create_task(
            _tts_and_stream(
                kokoro, text, voice_style, speed, player, sdk, present, split_len, stop_evt
            )
        )
        loop.run_until_complete(fut)
        sdk.end_processing_audio()
        idle_evt.set()

    async def _tts_and_stream(kokoro, text, voice, speed, player, sdk, present, split_len, stop_evt):
        pos = 0
        chunk_buf = np.array([], dtype=np.float32)
        async for audio_chunk, sr in kokoro.create_stream(text, voice=voice, speed=speed, lang="en-us"):
            chunk_16k = resample_torch(audio_chunk)
            chunk_buf = np.concatenate([chunk_buf, chunk_16k])
            while len(chunk_buf) >= AUDIO_FRAME_SAMP:
                frame = chunk_buf[:AUDIO_FRAME_SAMP]
                chunk_buf = chunk_buf[AUDIO_FRAME_SAMP:]
                player.push_audio((frame * 32767).astype(np.int16))

                # SDK visual chunk processing at proper boundary
                if pos % present == 0:
                    vis_chunk = chunk_buf[:split_len]
                    if len(vis_chunk) < split_len:
                        vis_chunk = np.pad(vis_chunk, (0, split_len - len(vis_chunk)))
                    sdk.process_audio_chunk(vis_chunk)
                pos += AUDIO_FRAME_SAMP
                await asyncio.sleep(0.02)  # Stream in real-time
        # Push any trailing frames
        while len(chunk_buf) > 0:
            frame = chunk_buf[:AUDIO_FRAME_SAMP]
            chunk_buf = chunk_buf[AUDIO_FRAME_SAMP:]
            player.push_audio((frame * 32767).astype(np.int16))
            pos += AUDIO_FRAME_SAMP
            await asyncio.sleep(0.02)

    t = threading.Thread(target=speech_worker, daemon=True)
    sess["speech_thread"] = t
    t.start()
    return {"status": "playing"}


TURNIX_API_KEY = "b750a7ff80271bf7ac63536f1f5f0b2d"

@app.post("/api/iceservers")
def get_ice_servers():
    headers = {
        "Authorization": f"Bearer {TURNIX_API_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.post("https://turnix.io/api/v1/credentials/ice", headers=headers)
    return r.json()