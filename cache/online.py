"""
online.py – FastAPI + WebRTC avatar streamer
• POST /offer → idle.mp3 loops continuously
• POST /speak → pauses idle, plays audio.mp3 once, resumes idle
"""
import asyncio
import threading, time, uuid, queue
from fractions import Fraction
import librosa
import torch
import cv2, numpy as np, soundfile as sf
from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription
from stream_pipeline_online import StreamSDK
from webrtc import HumanPlayer, AUDIO_FRAME_SAMP
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
BYTES_PER_FRAME = 640      # fixed for this model
IDLE_AUDIO = np.zeros(16000, dtype=np.float32)  # 1-s silence

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
    sess = ort.InferenceSession("kokoro-v1.0.onnx", sess_options=sess_opts, providers=[cuda_provider])
    kokoro = Kokoro.from_session(session=sess, voices_path="voices-v1.0.bin")
    audio, sr = kokoro.create("Dummy initialization text", voice="af_heart",speed=1.0,lang="en-us")
    sf.write("test.wav", audio, sr)
    return kokoro


# ─── helpers ──────────────────────────────────────────────────────────
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


def new_sdk() -> StreamSDK:
    sdk = StreamSDK(CFG_PKL, DATA_ROOT, chunk_size=(3, 5, 2))
    sdk.online_mode = True
    sdk.setup(
        SRC_IMG,
        max_size=1920,
        sampling_timesteps=15,
        emo=4,
        drive_eye=True,
        # smo_k_s=5,
        # smo_k_d=2,
        # overlap_v2=20,
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
        # torch.cuda.synchronize()

def frame_collector(sdk: StreamSDK, player: HumanPlayer, stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            buf, *_ = sdk.frame_queue.get(timeout=0.005)
            player.push_video(buf)
        except queue.Empty:
            time.sleep(0.001)

# ─── Endpoints ─────────────────────────────────────────────────────
class OfferModel(BaseModel):
    sdp: str
    type: str
    
@app.post("/offer")
async def offer(offer: OfferModel):
    sid = str(uuid.uuid4())
    idle_evt = threading.Event();  idle_evt.set()
    kill_evt = threading.Event()
    sdk = new_sdk()
    kokoro = initialize_kokoro()

    present    = sdk.chunk_size[1] * BYTES_PER_FRAME
    # split_len  = int(sum(sdk.chunk_size) * BYTES_PER_FRAME) + 80
    TARGET_FPS = 25 
    AUDIO_PER_FRAME = int(16000 / TARGET_FPS)
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
SILENCE_SEC      = 1.0
SILENCE_SAMPLES  = int(SILENCE_SEC * 16000)          # 1-second, 16 kHz
SILENCE_FRAME_I16 = (np.zeros(AUDIO_FRAME_SAMP).astype(np.float32) * 32767).astype(np.int16)


@app.post("/speak")
async def speak(sessionid: str = Form(...)):
    sess = sessions.get(sessionid)
    if not sess: 
        return {"error": "unknown session"}
    sdk          = sess["sdk"]
    idle_evt     = sess["idle_evt"]
    player       = sess["player"]
    present      = sess["present"]
    split_len    = sess["split_len"]
    kokoro       = sess["kokoro"]
    loop         = asyncio.get_event_loop()
    
    speed = 1.0
    voice_style = "af_heart"
    dummy_text = "Yes, I came here five years ago, when I was just sixteen. At the time, I was.. I was still in the tenth grade, and I clearly remember doing my homework in the backseat of the car as we drove to our new home. Everything felt unfamiliar and uncertain, but I tried to stay focused on school. I didn’t know what to expect, and it took a while to get used to the language, the people, and the new routines."
    # dummy_text = "Yes, I came here five years ago, when I was just sixteen."

    def run_tts() -> np.ndarray:
        wav24, _ = kokoro.create(
            dummy_text,
            voice=voice_style,
            speed=speed,
            lang="en-us",
        )
        return librosa.resample(wav24.astype(np.float32), orig_sr=24000, target_sr=16000)

    SPEECH: np.ndarray = await loop.run_in_executor(None, run_tts)
    
    # stop any running speech
    prev_stop = sess.get("speech_stop")
    prev_thr  = sess.get("speech_thread")
    if prev_thr and prev_thr.is_alive():
        prev_stop.set()
        prev_thr.join(timeout=0.2)

    stop_evt = threading.Event()
    sess["speech_stop"]   = stop_evt
    sess["speech_thread"] = None

    def speech_worker():
        idle_evt.clear()
        # Clear all processing queues to ensure fresh start
        for q in (sdk.hubert_features_queue, sdk.audio2motion_queue, 
                sdk.motion_stitch_queue, sdk.frame_queue):
            _drain(q)

        # Start audio processing pipeline
        sdk.start_processing_audio()
        
        # First push some silence to create buffer
        initial_silence = np.zeros(SILENCE_SAMPLES, dtype=np.float32)
        for i in range(0, len(initial_silence), AUDIO_FRAME_SAMP):
            silence_frame = initial_silence[i:i+AUDIO_FRAME_SAMP]
            if len(silence_frame) < AUDIO_FRAME_SAMP:
                silence_frame = np.pad(silence_frame, (0, AUDIO_FRAME_SAMP - len(silence_frame)))
            player.push_audio((silence_frame * 32767).astype(np.int16))
        
        # Process speech audio in chunks
        pos = 0
        while pos < len(SPEECH) and not stop_evt.is_set():
            # Push audio frame to player (20ms chunks)
            slice_f32 = SPEECH[pos:pos + AUDIO_FRAME_SAMP]
            if len(slice_f32) < AUDIO_FRAME_SAMP:
                slice_f32 = np.pad(slice_f32, (0, AUDIO_FRAME_SAMP - len(slice_f32)))
            player.push_audio((slice_f32 * 32767).astype(np.int16))

            # Process visual chunk when at presentation boundary
            if pos % present == 0:
                chunk = SPEECH[pos:pos + split_len]
                if len(chunk) < split_len:
                    chunk = np.pad(chunk, (0, split_len - len(chunk)))
                sdk.process_audio_chunk(chunk)
            
            pos += AUDIO_FRAME_SAMP
            time.sleep(0.020)  # 20ms per frame

        # End processing and return to idle
        sdk.end_processing_audio()
        idle_evt.set()

    # def speech_worker():
    #     idle_evt.clear()
    #     for q in (sdk.hubert_features_queue, sdk.audio2motion_queue, sdk.motion_stitch_queue):
    #         _drain(q)
    #     _drain(sdk.frame_queue)

    #     sdk.start_processing_audio()
    #     audio_started = threading.Event()
    #     audio_queue = queue.Queue()

    #     # Start the audio pusher thread
    #     def audio_pusher():
    #         pos = 0
    #         while not stop_evt.is_set():
    #             try:
    #                 chunk = audio_queue.get(timeout=0.1)
    #             except queue.Empty:
    #                 continue
    #             chunk = librosa.resample(np.asarray(chunk, dtype=np.float32), orig_sr=24000, target_sr=16000)

    #             if not audio_started.is_set() and sdk.frame_queue.qsize() > 0:
    #                 audio_started.set()

    #             # slice into 20ms frames and push
    #             local_pos = 0
    #             while local_pos < len(chunk) and not stop_evt.is_set():
    #                 frame = chunk[local_pos:local_pos + AUDIO_FRAME_SAMP]
    #                 if len(frame) < AUDIO_FRAME_SAMP:
    #                     frame = np.pad(frame, (0, AUDIO_FRAME_SAMP - len(frame)))
    #                 player.push_audio((frame * 32767).astype(np.int16))
    #                 local_pos += AUDIO_FRAME_SAMP
    #                 time.sleep(0.020)

    #     threading.Thread(target=audio_pusher, daemon=True).start()

    #     # Start TTS and push chunks to both audio & motion
    #     pos = 0
    #     for _, _, wav24 in pipeline(dummy_text, voice=voice_style, speed=float(speed)):
    #         audio_queue.put(wav24)
    #         chunk = librosa.resample(np.asarray(wav24, dtype=np.float32), orig_sr=24000, target_sr=16000)

    #         # same chunking logic for visual processing
    #         while pos < len(chunk) and not stop_evt.is_set():
    #             if pos % present == 0:
    #                 vis_chunk = chunk[pos:pos + split_len]
    #                 if len(vis_chunk) < split_len:
    #                     vis_chunk = np.pad(vis_chunk, (0, split_len - len(vis_chunk)))
    #                 sdk.process_audio_chunk(vis_chunk)
    #             pos += AUDIO_FRAME_SAMP

    #     sdk.end_processing_audio()
    #     idle_evt.set()

    t = threading.Thread(target=speech_worker, daemon=True)
    sess["speech_thread"] = t
    t.start()
    return {"status": "playing"}
