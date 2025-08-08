#!/usr/bin/env python
# avatar_livekit_fixed.py  – StreamSDK avatar → LiveKit Cloud
# ------------------------------------------------------------
import asyncio, logging, uuid, time, sys, io
from pathlib import Path
import numpy as np, torch, torchaudio, av
from PIL import Image
from livekit import api, rtc
import onnxruntime as ort
from stream_pipeline_online import StreamSDK
from kokoro_onnx import Kokoro

LIVEKIT_URL = "wss://wdqwd-i503xu0n.livekit.cloud"
API_KEY     = "APIewTmSoRk6rFS"
API_SECRET  = "XCWjdkZbDW2oj56f0eJjStwLEga2FRfMJzvfJ09WN7aB"

# ── 24 k → 16 k lightweight resampler ───────────────────────────────
_resampler = torchaudio.transforms.Resample(24_000, 16_000).cuda()
def resample(w):
    return _resampler(torch.from_numpy(w).cuda().unsqueeze(0))\
             .squeeze(0).cpu().numpy()

# ────────────────────────────────────────────────────────────────────
async def avatar_stream(room_name: str, text: str, avatar_png: str):

    token = (api.AccessToken(api_key=API_KEY, api_secret=API_SECRET)
             .with_identity(f"py-ava-{uuid.uuid4().hex[:6]}")
             .with_grants(api.VideoGrants(room_join=True,
                                           room=room_name,
                                           agent=True))
             .to_jwt())

    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    print("🔗 connected →", room.name)

    video_src = rtc.VideoSource(1080, 800)
    audio_src = rtc.AudioSource(16_000, 1, 1000)
    v_tr = rtc.LocalVideoTrack.create_video_track("v", video_src)
    a_tr = rtc.LocalAudioTrack.create_audio_track("a", audio_src)
    await room.local_participant.publish_track(v_tr)
    await room.local_participant.publish_track(a_tr)
    print("📡  publishing tracks … done")

    av_sync = rtc.AVSynchronizer(
        audio_source        = audio_src,
        video_source        = video_src,
        video_fps           = 26,
        video_queue_size_ms = 100
    )

    # epoch is set by the very first **audio** packet -------------- FIX
    start_time: float | None = None

    # ----- init SDK + TTS ---------------------------------------
    sdk = StreamSDK("./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
                    "./checkpoints/ditto_trt_Ampere_Plus",
                    chunk_size=(3, 5, 2))
    sdk.online_mode = True
    sdk.setup(avatar_png,
              max_size=1080,
              sampling_timesteps=15,
              emo=4,
              drive_eye=True,
              overlap_v2=70)

    sess   = ort.InferenceSession("checkpoints/kokoro-v1.0.onnx",
                                  providers=[("CUDAExecutionProvider",
                                              {"device_id": 0})])
    kokoro = Kokoro.from_session(sess, "checkpoints/voices-v1.0.bin")

    BYTES_PER_FRAME = 640
    present = sdk.chunk_size[1] * BYTES_PER_FRAME
    SPL = int(sum(sdk.chunk_size) * 16_000 / 25) + 80

    # ------------------------------------------------------------
    async def audio_task():
        nonlocal start_time
        pos, buf = 0, np.empty(0, np.float32)

        async for chunk, _ in kokoro.create_stream(
                text, voice="af_heart", speed=1.0, lang="en-us"):

            buf = np.concatenate([buf, resample(chunk)])
            while len(buf) >= 320:
                frame, buf = buf[:320], buf[320:]
                if start_time is None:                 # FIX
                    start_time = time.perf_counter()
                ts = time.perf_counter() - start_time  # FIX

                af = rtc.AudioFrame(
                        (frame * 32767).astype(np.int16).tobytes(),
                        16_000, 1, 320)
                await av_sync.push(af, ts)

                if pos % present == 0:
                    vis = buf[:SPL]
                    if len(vis) < SPL:
                        vis = np.pad(vis, (0, SPL - len(vis)))
                    sdk.process_audio_chunk(vis)
                pos += 320
                await asyncio.sleep(0)

    # ------------------------------------------------------------
    async def video_task():
        nonlocal start_time
        loop = asyncio.get_running_loop()

        def jpeg_rgba(jpg: bytes) -> np.ndarray:
            img = Image.open(io.BytesIO(jpg)).convert("RGBA")
            return np.asarray(img, np.uint8)

        while True:
            jpg, *_ = await loop.run_in_executor(None, sdk.frame_queue.get)
            rgba = jpeg_rgba(jpg)

            if start_time is None:                 # video might still win the race
                start_time = time.perf_counter()   # earlier than first audio pkt
            ts = time.perf_counter() - start_time  # FIX

            vf = rtc.VideoFrame(rgba.shape[1], rgba.shape[0],
                                rtc.VideoBufferType.RGBA,
                                rgba.tobytes())
            await av_sync.push(vf, ts)
            await asyncio.sleep(0)

    sdk.start_processing_audio()
    await asyncio.gather(video_task(), audio_task())

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage:  python avatar_livekit_fixed.py <room> <text> <avatar.png>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    asyncio.run(avatar_stream(sys.argv[1], sys.argv[2], sys.argv[3]))
