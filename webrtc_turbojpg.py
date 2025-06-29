# webrtc.py (updated)
import queue, asyncio, threading, time, fractions
from typing import Optional, Set, Tuple, Union

import numpy as np
import av
from av import AudioFrame
from aiortc import MediaStreamTrack
from turbojpeg import TurboJPEG, TJPF_RGB

# Initialize once
jpeg = TurboJPEG()

# Timing constants
FPS = 25
VIDEO_PTIME = 1 / FPS
VIDEO_CLOCK_RATE = 90_000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16_000
AUDIO_PTIME = 0.020
AUDIO_FRAME_SAMP = int(AUDIO_PTIME * SAMPLE_RATE)
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

class PlayerStreamTrack(MediaStreamTrack):
    def __init__(self, player: "HumanPlayer", kind: str):
        super().__init__()
        self.kind = kind
        self._player = player
        self._queue: asyncio.Queue = asyncio.Queue()
        self._start_time: float | None = None
        self._timestamp = 0

    async def _paced_timestamp(self):
        if self._start_time is None:
            self._start_time = time.time()
            return 0, (VIDEO_TIME_BASE if self.kind == "video" else AUDIO_TIME_BASE)
        if self.kind == "video":
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start_time + self._timestamp / VIDEO_CLOCK_RATE - time.time()
        else:
            self._timestamp += AUDIO_FRAME_SAMP
            wait = self._start_time + self._timestamp / SAMPLE_RATE - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        return self._timestamp, (VIDEO_TIME_BASE if self.kind == "video" else AUDIO_TIME_BASE)

    async def recv(self):
        self._player._ensure_worker_running()
        frame, _ = await self._queue.get()
        pts, tb = await self._paced_timestamp()
        frame.pts, frame.time_base = pts, tb
        return frame

    def stop(self):
        super().stop()
        if self._player:
            self._player._track_stopped(self)

class HumanPlayer:
    def __init__(self):
        self.audio = PlayerStreamTrack(self, "audio")
        self.video = PlayerStreamTrack(self, "video")
        self._started: Set[PlayerStreamTrack] = set()
        self._quit_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._aud_q: queue.Queue = queue.Queue()
        self._vid_q: queue.Queue = queue.Queue()

    def push_audio(self, samples: np.ndarray) -> None:
        if samples.dtype != np.int16:
            samples = (samples * 32767).astype(np.int16)
        self._aud_q.put(samples, block=False)

    def push_video(self, jpeg_bytes: bytes) -> None:
        self._vid_q.put(jpeg_bytes, block=False)

    def _ensure_worker_running(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="webrtc-worker", daemon=True)
            self._thread.start()

    def _track_stopped(self, track: PlayerStreamTrack):
        self._started.discard(track)
        if not self._started:
            self._quit_evt.set()

    def _worker(self):
        while not self._quit_evt.is_set():
            try:
                aud = self._aud_q.get(timeout=0.001)
                frame = AudioFrame.from_ndarray(aud.reshape(1, -1), layout="mono", format="s16")
                frame.sample_rate = SAMPLE_RATE
                asyncio.run(self.audio._queue.put((frame, None)))
            except queue.Empty:
                pass

            try:
                jpg = self._vid_q.get_nowait()
                # Use TurboJPEG
                img_rgb = jpeg.decode(jpg, pixel_format=TJPF_RGB)
                vf = av.VideoFrame.from_ndarray(img_rgb, format="rgb24")
                asyncio.run(self.video._queue.put((vf, None)))
            except queue.Empty:
                pass

        self.audio.stop()
        self.video.stop()
