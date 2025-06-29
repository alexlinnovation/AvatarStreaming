import queue
import asyncio, threading, time, fractions
from typing import Optional, Set, Tuple, Union

import numpy as np
import av
from av import AudioFrame
from aiortc import MediaStreamTrack
import cv2
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

class DaliJPEGDecoder(Pipeline):
    def __init__(self, batch_size=1, threads=1, device_id=0):
        super().__init__(batch_size, threads, device_id, exec_pipelined=False)
        self.input = fn.external_source(name="images", device="cpu")
        self.decode = fn.decoders.image(
            self.input, device="mixed", output_type=types.RGB
        )

    def define_graph(self):
        self.images = self.decode
        return self.images

# --- timing constants --------------------------------------------------------
FPS               = 25
VIDEO_PTIME       = 1 / FPS
VIDEO_CLOCK_RATE  = 90_000
VIDEO_TIME_BASE   = fractions.Fraction(1, VIDEO_CLOCK_RATE)

SAMPLE_RATE       = 16_000
AUDIO_PTIME       = 0.020                      # 20 ms
AUDIO_FRAME_SAMP  = int(AUDIO_PTIME * SAMPLE_RATE)  # 320
AUDIO_TIME_BASE   = fractions.Fraction(1, SAMPLE_RATE)

# -----------------------------------------------------------------------------


class PlayerStreamTrack(MediaStreamTrack):
    """
    A MediaStreamTrack whose frames are pushed by HumanPlayer.
    Timing is generated locally so the caller never worries about PTS.
    """

    def __init__(self, player: "HumanPlayer", kind: str):
        super().__init__()
        self.kind         = kind
        self._player      = player
        self._queue       = asyncio.Queue()
        self._start_time: float | None = None
        self._timestamp   = 0            # running samples or 90 kHz ticks

    # -------------------------------------------------------------------------
    async def _paced_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self._start_time is None:
            self._start_time = time.time()
            return 0, (VIDEO_TIME_BASE if self.kind == "video"
                        else AUDIO_TIME_BASE)

        if self.kind == "video":
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start_time + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            return self._timestamp, VIDEO_TIME_BASE
        else:
            self._timestamp += AUDIO_FRAME_SAMP
            wait = self._start_time + (self._timestamp / SAMPLE_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            return self._timestamp, AUDIO_TIME_BASE

    # -------------------------------------------------------------------------
    async def recv(self) -> Union[av.VideoFrame, av.AudioFrame]:
        self._player._ensure_worker_running()

        frame, _ = await self._queue.get()
        pts, tb  = await self._paced_timestamp()
        frame.pts, frame.time_base = pts, tb
        return frame

    # -------------------------------------------------------------------------
    def stop(self) -> None:
        super().stop()
        if self._player:
            self._player._track_stopped(self)


# -----------------------------------------------------------------------------


class HumanPlayer:
    """
    Owns two PlayerStreamTracks (audio + video) and a worker thread.
    The application feeds raw frames via push_video() / push_audio().
    """

    def __init__(self):
        self.audio  = PlayerStreamTrack(self, "audio")
        self.video  = PlayerStreamTrack(self, "video")

        self._dali_decoder = DaliJPEGDecoder()
        self._dali_decoder.build()

        self._started: Set[PlayerStreamTrack] = set()
        self._quit_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # simple internal queues – producer (app) → worker → tracks
        self._aud_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=0)
        self._vid_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=0)

    # ---------------------------------------------------------------------
    # public feed API
    def push_audio(self, samples: np.ndarray) -> None:
        """samples: int16 mono, length == 320 (20 ms)"""
        if samples.dtype != np.int16:
            samples = (samples * 32767).astype(np.int16)
        self._aud_q.put(samples, block=False)

    def push_video(self, jpeg_bytes: bytes) -> None:
        """jpeg‐encoded BGR frame from SDK"""
        self._vid_q.put(jpeg_bytes, block=False)

    # ---------------------------------------------------------------------
    # internals
    def _ensure_worker_running(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._worker, name="webrtc-worker", daemon=True
            )
            self._thread.start()

    def _track_stopped(self, track: PlayerStreamTrack):
        self._started.discard(track)
        if not self._started:
            self._quit_evt.set()

    def _worker(self):
        """pull from app queues; push to track queues; keep pacing short"""
        import queue
        while not self._quit_evt.is_set():
            try:
                aud = self._aud_q.get(timeout=0.001)
                frame = AudioFrame.from_ndarray(aud.reshape(1, -1),
                                                layout="mono", format="s16")
                frame.sample_rate = SAMPLE_RATE
                asyncio.run(self.audio._queue.put((frame, None)))
            except queue.Empty:
                pass

            try:
                jpg = self._vid_q.get_nowait()

                # Decode JPEG using DALI (GPU accelerated)
                self._dali_decoder.feed_input("images", [np.frombuffer(jpg, dtype=np.uint8)])
                out = self._dali_decoder.run()
                img = out[0].as_cpu().as_array()[0]  # shape: (H, W, 3) RGB

                vf = av.VideoFrame.from_ndarray(img, format="rgb24")
                asyncio.run(self.video._queue.put((vf, None)))

            except queue.Empty:
                pass

        # drain → stop
        self.audio.stop()
        self.video.stop()


