# gated_avsync.py
import asyncio
import time
from collections import deque
from typing import Optional, Union
from livekit.rtc import AudioSource, VideoSource
from livekit.rtc import VideoFrame as LKVideoFrame
from livekit.rtc import AudioFrame as LKAudioFrame
import numpy as np  # <-- added


class GatedAVSynchronizer:
    """
    Video is the master clock.
    - Audio never leads video (allowed_audio_lead_ms controls any small allowance).
    - When backlog accumulates (typically silence), quickly skip audio frames whose
      timestamps are already behind the video clock to prevent stalls before speech.
      While skipping, periodically yield to avoid starving the video task.
    """

    def __init__(
        self,
        *,
        audio_source: AudioSource,
        video_source: VideoSource,
        video_fps: float,
        video_queue_size_ms: float = 100,
        allowed_audio_lead_ms: float = 0.0,
        _max_delay_tolerance_ms: float = 100,
    ):
        self._audio_source = audio_source
        self._video_source = video_source
        self._video_fps = float(video_fps)
        self._frame_interval = 1.0 / self._video_fps
        self._max_delay_tolerance_secs = _max_delay_tolerance_ms / 1000.0
        self._allowed_lead = allowed_audio_lead_ms / 1000.0
        self._video_queue_size_secs = float(video_queue_size_ms) / 1000.0

        self._stopped = False

        # Timelines (media time, seconds)
        self._last_video_time: float = 0.0
        self._last_audio_time_reported: float = 0.0

        # Queues
        self._video_queue_max = max(1, int(self._video_fps * self._video_queue_size_secs))
        self._video_queue: asyncio.Queue[tuple[LKVideoFrame, Optional[float]]] = asyncio.Queue(
            maxsize=60
        )
        self._audio_queue: asyncio.Queue[tuple[LKAudioFrame, Optional[float]]] = asyncio.Queue(
            maxsize=60
        )

        # Backlog / catch-up policy (structural, not tuning)
        self._drop_margin = max(0.10, 2 * self._frame_interval)       # >=100ms or ~2 frames
        self._fast_forward_threshold = max(10, int(self._video_fps))   # ~1s audio backlog
        self._drop_batch_yield = 256                                   # yield after N drops

        # Video pacing
        self._next_frame_time: Optional[float] = None
        self._send_timestamps: deque[float] = deque(maxlen=max(2, int(1.0 * self._video_fps)))

        # Tasks
        self._t_video = asyncio.create_task(self._drain_video())
        self._t_audio = asyncio.create_task(self._drain_audio())
        self._audio_wall_next = None
        self._audio_dt_default = 0.02

        # ---- NEW: zero-latency de-click state ----
        self._prev_tail: Optional[np.ndarray] = None  # last sample per channel
        self._xfade_ms: float = 1.5                  # boundary crossfade length
        self._softclip_strength: float = 1.2         # 1.0=no clip, 1.2–1.6 gentle


    async def push(
        self,
        frame: Union[LKVideoFrame, LKAudioFrame],
        timestamp: Optional[float] = None,
    ) -> None:
        if isinstance(frame, LKAudioFrame):
            await self._audio_queue.put((frame, timestamp))
            return
        await self._video_queue.put((frame, timestamp))

    async def clear_queue(self) -> None:
        while not self._audio_queue.empty():
            await self._audio_queue.get()
            self._audio_queue.task_done()
        while not self._video_queue.empty():
            await self._video_queue.get()
            self._video_queue.task_done()

    async def wait_for_playout(self) -> None:
        await asyncio.gather(
            self._video_queue.join(),
            self._audio_queue.join(),
        )

    def reset(self) -> None:
        self._next_frame_time = None
        self._send_timestamps.clear()
        self._prev_tail = None  # reset de-click state

    async def _drain_video(self) -> None:
        while not self._stopped:
            vf, ts = await self._video_queue.get()
            await self._wait_next_frame()
            self._video_source.capture_frame(vf)
            if ts is not None:
                self._last_video_time = float(ts)
            self._after_frame()
            self._video_queue.task_done()

    async def _drain_audio(self) -> None:
        QueueEmpty = asyncio.QueueEmpty

        while not self._stopped:
            af, ts = await self._audio_queue.get()

            # Effective timestamp: when None, align to current video edge (no lead)
            ts_eff = (self._last_video_time + self._allowed_lead) if ts is None else float(ts)

            # Fast-forward stale backlog without starving the loop
            if self._audio_queue.qsize() > self._fast_forward_threshold:
                dropped = 0
                while (self._audio_queue.qsize() > 0) and ((self._last_video_time - ts_eff) > self._drop_margin):
                    # drop this stale frame
                    self._audio_queue.task_done()
                    dropped += 1
                    try:
                        af, ts = self._audio_queue.get_nowait()
                        ts_eff = (self._last_video_time + self._allowed_lead) if ts is None else float(ts)
                    except QueueEmpty:
                        break
                    # yield periodically so video task can run
                    if (dropped % self._drop_batch_yield) == 0:
                        await asyncio.sleep(0)

            # Normal gating: audio never leads video
            while (self._last_video_time + self._allowed_lead) < ts_eff and not self._stopped:
                await asyncio.sleep(0.001)

            self._last_audio_time_reported = ts_eff

            # ---- NEW: zero-latency de-click / soft-clip (no timing changes) ----
            try:
                self._declick_inplace(af)
            except Exception:
                # Fail-safe: never block playout if processing fails
                pass

            # Release to LiveKit
            await self._audio_source.capture_frame(af)
            self._audio_queue.task_done()

    async def aclose(self) -> None:
        self._stopped = True
        for t in (self._t_video, self._t_audio):
            if t:
                t.cancel()

    # ---- FPS helpers (video pacing) ----
    async def _wait_next_frame(self) -> None:
        now = time.perf_counter()
        if self._next_frame_time is None:
            self._next_frame_time = now
        sleep_for = self._next_frame_time - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            if -sleep_for > self._max_delay_tolerance_secs:
                self._next_frame_time = time.perf_counter()

    def _after_frame(self) -> None:
        assert self._next_frame_time is not None
        self._send_timestamps.append(time.perf_counter())
        self._next_frame_time += self._frame_interval

    @property
    def actual_fps(self) -> float:
        if len(self._send_timestamps) < 2:
            return 0.0
        return (len(self._send_timestamps) - 1) / (self._send_timestamps[-1] - self._send_timestamps[0])

    @property
    def last_video_time(self) -> float:
        return self._last_video_time

    @property
    def last_audio_time(self) -> float:
        return self._last_audio_time_reported

    # ---- NEW: helpers for in-place transient suppression ----
    def _declick_inplace(self, af: LKAudioFrame) -> None:
        """
        Causal, zero-latency smoothing:
          - short crossfade from previous tail sample(s)
          - gentle soft-clip for rare spikes
        Works per frame; no extra buffering or delay.
        """
        # Try to access PCM as int16
        sr = int(getattr(af, "sample_rate", 16000))
        ch = int(getattr(af, "num_channels", 1))

        buf = getattr(af, "data", None)
        if buf is None:
            # try alternate attribute names if SDK differs
            for alt in ("buffer", "buf", "pcm", "samples_bytes"):
                buf = getattr(af, alt, None)
                if buf is not None:
                    break
        if buf is None:
            return  # nothing we can do safely

        # Make a writable copy for processing
        arr = np.frombuffer(buf, dtype=np.int16).copy()
        if arr.size == 0:
            return

        # Shape [num_frames, ch]
        if ch > 1:
            if arr.size % ch != 0:
                # fallback: treat as mono
                ch = 1
            frames = arr.size // ch
            x = arr.reshape(frames, ch).astype(np.float32)
        else:
            x = arr.astype(np.float32).reshape(-1, 1)
            frames = x.shape[0]

        # 1) Boundary crossfade to remove clicks between frames
        xfade_len = max(1, min(int(sr * self._xfade_ms / 1000.0), frames))
        if self._prev_tail is None or self._prev_tail.shape[0] != x.shape[1]:
            # initialize prev tail as first sample to avoid ramp from zero
            self._prev_tail = x[0, :].copy()

        if xfade_len > 1:
            alphas = np.linspace(0.0, 1.0, xfade_len, dtype=np.float32).reshape(-1, 1)
            head = x[:xfade_len, :]
            x[:xfade_len, :] = (1.0 - alphas) * self._prev_tail[None, :] + alphas * head

        # 2) Gentle soft-clip to shave transients (very mild)
        # scale to [-1,1], soft-clip, back to int16 range
        y = x / 32768.0
        s = float(self._softclip_strength)
        y = np.tanh(s * y) / np.tanh(s)

        # update tail state for next frame
        self._prev_tail = y[-1, :].copy()

        # write back
        y = (y * 32767.0).astype(np.int16)
        if ch > 1:
            y = y.reshape(-1)
        out_bytes = y.tobytes()

        # Try to replace in-place; if not possible, set attribute
        try:
            # attempt mutable view
            mv = memoryview(buf)
            if not mv.readonly and len(mv) == len(out_bytes):
                mv[:] = out_bytes  # type: ignore[index]
                return
        except Exception:
            pass

        # Replace data attribute (common for dataclass-style frames)
        try:
            setattr(af, "data", out_bytes)
        except Exception:
            # last resort: try common alt names
            for alt in ("buffer", "buf", "pcm", "samples_bytes"):
                try:
                    setattr(af, alt, out_bytes)
                    break
                except Exception:
                    continue