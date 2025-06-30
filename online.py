"""
online.py – FastAPI + WebRTC avatar streamer
• POST /offer → idle.mp3 loops continuously
• POST /speak → pauses idle, plays audio.mp3 once, resumes idle
"""
import threading, time, uuid, queue
from fractions import Fraction
import torch
import cv2, numpy as np, soundfile as sf
from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from aiortc import RTCPeerConnection, RTCSessionDescription
from stream_pipeline_online import StreamSDK
from webrtc import HumanPlayer, AUDIO_FRAME_SAMP

# ─── config & paths ───────────────────────────────────────────────────
CFG_PKL   = "./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
DATA_ROOT = "./checkpoints/ditto_trt_Ampere_Plus"
SRC_IMG   = "static/avatar.png"
IDLE_FILE = "static/idle.mp3"
BYTES_PER_FRAME = 640      # fixed for this model
FPS   = 25
IDLE_AUDIO = np.zeros(16000, dtype=np.float32)  # 1-s silence

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    sdk = StreamSDK(CFG_PKL, DATA_ROOT, chunk_size=(2, 4, 2))
    sdk.online_mode = True
    sdk.setup(
        SRC_IMG,
        max_size=800,
        sampling_timesteps=20,
        emo=4,
        drive_eye=True,
        smo_k_s=20,
        # smo_k_d=5,
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
        torch.cuda.synchronize()

def frame_collector(sdk: StreamSDK, player: HumanPlayer, stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            buf, *_ = sdk.frame_queue.get(timeout=0.25)
            player.push_video(buf)
        except queue.Empty:
            continue

# ─── Endpoints ─────────────────────────────────────────────────────
@app.post("/offer")
async def offer(offer: dict):
    sid = str(uuid.uuid4())
    idle_evt = threading.Event();  idle_evt.set()
    kill_evt = threading.Event()
    sdk = new_sdk()

    present    = sdk.chunk_size[1] * BYTES_PER_FRAME
    split_len  = int(sum(sdk.chunk_size) * BYTES_PER_FRAME) + 80
    idle_slice = np.pad(IDLE_AUDIO[:split_len], (0, max(0, split_len - len(IDLE_AUDIO))), 'constant')

    player = HumanPlayer()
    sessions[sid] = {
        "sdk": sdk,
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

    await pc.setRemoteDescription(RTCSessionDescription(**offer))
    await pc.setLocalDescription(await pc.createAnswer())
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sid}


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
    
    TALK_FILE = "static/audio.mp3"
    SPEECH = load_16k(TALK_FILE) # dummy audio

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
        for q in (sdk.hubert_features_queue, sdk.audio2motion_queue, sdk.motion_stitch_queue):
            _drain(q)
        _drain(sdk.frame_queue)

        sdk.start_processing_audio()
        
        pos = 0
        while pos < len(SPEECH) and not stop_evt.is_set():
            slice_f32 = SPEECH[pos:pos + AUDIO_FRAME_SAMP]
            if len(slice_f32) < AUDIO_FRAME_SAMP:
                slice_f32 = np.pad(slice_f32, (0, AUDIO_FRAME_SAMP - len(slice_f32)))
            player.push_audio((slice_f32 * 32767).astype(np.int16))

            if pos % present == 0:
                chunk = SPEECH[pos:pos + split_len]
                if len(chunk) < split_len:
                    chunk = np.pad(chunk, (0, split_len - len(chunk)))
                sdk.process_audio_chunk(chunk)
            pos += AUDIO_FRAME_SAMP
            time.sleep(0.020)

        sdk.end_processing_audio()
        idle_evt.set()

    t = threading.Thread(target=speech_worker, daemon=True)
    sess["speech_thread"] = t
    t.start()
    return {"status": "playing"}
