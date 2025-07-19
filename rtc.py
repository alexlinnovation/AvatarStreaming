import asyncio
import threading
import time
import uuid
from fractions import Fraction
import numpy as np
import torch, cv2, librosa
import soundfile as sf

from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import AudioFrame, VideoFrame
from pydantic import BaseModel
import onnxruntime as ort

from stream_pipeline_online import StreamSDK
from kokoro_onnx import Kokoro

# --- your config ---
CFG_PKL   = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
DATA_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
SRC_IMG   = "static/avatar.png"
BYTES_PER_FRAME = 640
IDLE_AUDIO = np.zeros(16000, dtype=np.float32)
CHUNK_SIZE = (4, 8, 4)  # or as you wish
FPS = 25
SAMPLE_RATE = 16000
AUDIO_FRAME_SAMP = int(SAMPLE_RATE / FPS)

# --- FastAPI ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# --- KOKORO (your function, uncut) ---
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
    audio, sr = kokoro.create("Dummy initialization text", voice="af_heart", speed=1.0, lang="en-us")
    sf.write("test.wav", audio, sr)
    return kokoro

# --- RESAMPLER ---
import torchaudio
resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000).cuda()
def resample_torch(wav24: np.ndarray) -> np.ndarray:
    wav_tensor = torch.tensor(wav24, dtype=torch.float32, device='cuda').unsqueeze(0)
    wav16 = resampler(wav_tensor)
    return wav16.squeeze(0).cpu().numpy()

# --- CUSTOM AIORTC TRACKS (sync push) ---
class SyncAudioTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self._ts = 0

    async def recv(self):
        samples = await self.queue.get()
        af = AudioFrame(samples=samples.reshape(1, -1), layout="mono", format="s16")
        af.sample_rate = SAMPLE_RATE
        af.pts = self._ts
        af.time_base = Fraction(1, SAMPLE_RATE)
        self._ts += AUDIO_FRAME_SAMP
        return af

class SyncVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self._ts = 0

    async def recv(self):
        jpg_buf = await self.queue.get()
        img_bgr = cv2.imdecode(np.frombuffer(jpg_buf, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vf = VideoFrame.from_ndarray(img_rgb, format="rgb24")
        vf.pts = self._ts
        vf.time_base = Fraction(1, FPS*3600//144)
        self._ts += int(90000 / FPS)
        return vf

# --- SDK HELPER ---
def new_sdk(src_img: str = SRC_IMG) -> StreamSDK:
    sdk = StreamSDK(CFG_PKL, DATA_ROOT, chunk_size=CHUNK_SIZE)
    sdk.online_mode = True
    sdk.setup(
        src_img,
        max_size=1980,
        sampling_timesteps=13,
        emo=4,
        drive_eye=True,
    )
    return sdk

def _drain(q):
    while True:
        try: q.get_nowait()
        except queue.Empty: return

sessions = {}

# --- API MODELS ---
class OfferModel(BaseModel):
    sdp: str
    type: str
    src_img: str | None = None

@app.post("/offer")
async def offer(offer: OfferModel):
    sid = str(uuid.uuid4())
    sdk = new_sdk(offer.src_img or SRC_IMG)
    kokoro = initialize_kokoro()
    audio_track = SyncAudioTrack()
    video_track = SyncVideoTrack()

    # Preload idle: push silence & idle frame for a smooth join (optional)
    idle_evt = threading.Event()
    idle_evt.set()
    kill_evt = threading.Event()
    sessions[sid] = {
        "sdk": sdk,
        "kokoro": kokoro,
        "audio": audio_track,
        "video": video_track,
        "idle_evt": idle_evt,
        "kill_evt": kill_evt,
        "speech_thread": None
    }
    pc = RTCPeerConnection()
    pc.addTrack(video_track)
    pc.addTrack(audio_track)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
    await pc.setLocalDescription(await pc.createAnswer())
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sid}

@app.post("/speak")
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
    kokoro     = sess["kokoro"]
    audio_track= sess["audio"]
    video_track= sess["video"]

    # Kill old speech thread if any
    thr = sess.get("speech_thread")
    if thr and thr.is_alive():
        sess["kill_evt"].set()
        thr.join(timeout=2.0)
        sess["kill_evt"].clear()

    text = text or (
        "Yes, I came here five years ago, when I was just sixteen. "
        "At the time, I was.. I was still in the tenth grade, and I clearly "
        "remember doing my homework in the backseat of the car as we drove to our new home. "
        "Everything felt unfamiliar and uncertain, but I tried to stay focused on school. "
        "I didn’t know what to expect, and it took a while to get used to the language, the people, and the new routines."
    )
    voice_style = (voice_style or "af_heart").strip() or "af_heart"
    speed = speed or 1.1

    def worker():
        sess["idle_evt"].clear()
        sdk.interrupt()
        _drain(sdk.frame_queue)
        sdk.start_processing_audio()
        chunk_buf = np.array([], dtype=np.float32)
        kill_evt = sess["kill_evt"]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def stream_audio_video():
            pos = 0
            async for audio_chunk, sr in kokoro.create_stream(
                text, voice=voice_style, speed=speed, lang="en-us"
            ):
                # Resample to 16kHz if not already
                if sr == 24000:
                    chunk_16k = resample_torch(audio_chunk)
                else:
                    chunk_16k = audio_chunk
                chunk_buf = np.concatenate([chunk_buf, chunk_16k])
                while len(chunk_buf) >= AUDIO_FRAME_SAMP:
                    frame = chunk_buf[:AUDIO_FRAME_SAMP]
                    chunk_buf = chunk_buf[AUDIO_FRAME_SAMP:]

                    # Video chunk logic: match offline, one process_audio_chunk per frame
                    sdk.process_audio_chunk(frame)
                    # Wait for frame
                    for _ in range(1000):
                        if not sdk.frame_queue.empty():
                            buf, *_ = sdk.frame_queue.get()
                            video_track.queue.put_nowait(buf)
                            break
                        time.sleep(0.001)

                    # Send audio
                    audio_track.queue.put_nowait((frame * 32767).astype(np.int16))
                    pos += AUDIO_FRAME_SAMP
                    if kill_evt.is_set():
                        break
                    await asyncio.sleep(0.015)

            # Push remaining
            while len(chunk_buf) > 0:
                frame = chunk_buf[:AUDIO_FRAME_SAMP]
                chunk_buf = chunk_buf[AUDIO_FRAME_SAMP:]
                sdk.process_audio_chunk(frame)
                for _ in range(1000):
                    if not sdk.frame_queue.empty():
                        buf, *_ = sdk.frame_queue.get()
                        video_track.queue.put_nowait(buf)
                        break
                    time.sleep(0.001)
                audio_track.queue.put_nowait((frame * 32767).astype(np.int16))
                pos += AUDIO_FRAME_SAMP
                if kill_evt.is_set():
                    break
                await asyncio.sleep(0.015)

            sdk.end_processing_audio()
            sess["idle_evt"].set()

        loop.run_until_complete(stream_audio_video())

    t = threading.Thread(target=worker, daemon=True)
    sess["speech_thread"] = t
    t.start()
    return {"status": "playing"}
