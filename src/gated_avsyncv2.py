# gated_avsyncv2.py  (drop-in replacement with minimal changes)
import asyncio
import time
from collections import deque
from typing import Optional, Union
from livekit.rtc import AudioSource, VideoSource
from livekit.rtc import VideoFrame as LKVideoFrame
from livekit.rtc import AudioFrame as LKAudioFrame
import numpy as np


class GatedAVSynchronizerV2:
    """
    Video is the master clock.
    - Audio never leads video (allowed_audio_lead_ms controls any small allowance).
    - Backlog control: skip stale audio when far behind to avoid stalls before speech.
    - Event-driven gating (no 1ms polling), plus start barrier so audio can't run before video.
    - Small, bounded catch-up burst when video sprints ahead.
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
        self._video_queue: asyncio.Queue[tuple[LKVideoFrame, Optional[float]]] = asyncio.Queue(maxsize=60)
        self._audio_queue: asyncio.Queue[tuple[LKAudioFrame, Optional[float]]] = asyncio.Queue(maxsize=60)

        # Backlog / catch-up policy (structural, not tuning)
        self._drop_margin = max(0.10, 2 * self._frame_interval)       # >=100ms or ~2 frames
        self._fast_forward_threshold = max(10, int(self._video_fps))   # ~1s audio backlog
        self._drop_batch_yield = 256                                   # yield after N drops

        # Video pacing
        self._next_frame_time: Optional[float] = None
        self._send_timestamps: deque[float] = deque(maxlen=max(2, int(1.0 * self._video_fps)))

        # ★ NEW: video tick & start barrier (event-driven gating)
        self._video_tick = asyncio.Event()
        self._video_started = asyncio.Event()

        # ★ NEW: small catch-up burst knobs (keep tiny to avoid rush)
        self._catchup_high = 0.080   # start bursting if video leads audio ts by >80 ms
        self._catchup_target = 0.020 # stop bursting once inside 20 ms
        self._catchup_max = 2        # at most 2 extra frames in one go

        # Tasks
        self._t_video = asyncio.create_task(self._drain_video())
        self._t_audio = asyncio.create_task(self._drain_audio())
        self._audio_wall_next = None
        self._audio_dt_default = 0.02

        # De-click state
        self._prev_tail: Optional[np.ndarray] = None
        self._xfade_ms: float = 1.5
        self._softclip_strength: float = 1.2

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
        self._prev_tail = None
        # ★ NEW: reset events
        self._video_tick.clear()
        # _video_started auto-resets naturally on new run when first frame comes

    async def _drain_video(self) -> None:
        while not self._stopped:
            vf, ts = await self._video_queue.get()
            await self._wait_next_frame()
            self._video_source.capture_frame(vf)
            if ts is not None:
                self._last_video_time = float(ts)
            # ★ NEW: signal "a video frame advanced"
            if not self._video_started.is_set():
                self._video_started.set()
            self._video_tick.set()
            self._video_tick.clear()
            self._after_frame()
            self._video_queue.task_done()

    async def _drain_audio(self) -> None:
        QueueEmpty = asyncio.QueueEmpty

        while not self._stopped:
            af, ts = await self._audio_queue.get()

            # Effective timestamp: if None, align to current video edge (no lead)
            ts_eff = (self._last_video_time + self._allowed_lead) if ts is None else float(ts)

            # ★ NEW: start barrier — don't let audio out before first video frame
            if not self._video_started.is_set():
                try:
                    await asyncio.wait_for(self._video_started.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

            # Fast-forward stale backlog without starving the loop (unchanged)
            if self._audio_queue.qsize() > self._fast_forward_threshold:
                dropped = 0
                while (self._audio_queue.qsize() > 0) and ((self._last_video_time - ts_eff) > self._drop_margin):
                    self._audio_queue.task_done()
                    dropped += 1
                    try:
                        af, ts = self._audio_queue.get_nowait()
                        ts_eff = (self._last_video_time + self._allowed_lead) if ts is None else float(ts)
                    except QueueEmpty:
                        break
                    if (dropped % self._drop_batch_yield) == 0:
                        await asyncio.sleep(0)

            # ★ NEW: event-driven gate instead of 1ms polling
            # If audio ts would lead video+lead, wait to be poked by next video tick
            # Use short timeouts to stay responsive, but no sub-ms sleeps.
            while (self._last_video_time + self._allowed_lead) < ts_eff and not self._stopped:
                try:
                    await asyncio.wait_for(self._video_tick.wait(), timeout=0.001)
                except asyncio.TimeoutError:
                    pass
                finally:
                    self._video_tick.clear()

            # ★ NEW: soft catch-up burst if video ran far ahead of this frame's ts
            # (Process current + a couple more frames immediately to close the gap)
            def lead_amount() -> float:
                return (self._last_video_time + self._allowed_lead) - ts_eff  # >0 means video ahead

            burst = 0
            while lead_amount() > self._catchup_high and burst < self._catchup_max:
                # play current 'af' now
                try:
                    self._declick_inplace(af)
                except Exception:
                    pass
                await self._audio_source.capture_frame(af)
                self._last_audio_time_reported = ts_eff
                self._audio_queue.task_done()
                burst += 1

                # pull next audio frame if available and update ts_eff
                try:
                    af, ts = self._audio_queue.get_nowait()
                    ts_eff = (self._last_video_time + self._allowed_lead) if ts is None else float(ts)
                except QueueEmpty:
                    af = None
                    break

                if (self._last_video_time + self._allowed_lead) - ts_eff <= self._catchup_target:
                    break

            if af is None:
                # nothing left after the burst
                continue

            # De-click / soft-clip (no timing change)
            try:
                self._declick_inplace(af)
            except Exception:
                pass

            # Release to LiveKit (await required)
            await self._audio_source.capture_frame(af)
            self._last_audio_time_reported = ts_eff
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

    # ---- in-place transient suppression (unchanged) ----
    def _declick_inplace(self, af: LKAudioFrame) -> None:
        sr = int(getattr(af, "sample_rate", 16000))
        ch = int(getattr(af, "num_channels", 1))

        buf = getattr(af, "data", None)
        if buf is None:
            for alt in ("buffer", "buf", "pcm", "samples_bytes"):
                buf = getattr(af, alt, None)
                if buf is not None:
                    break
        if buf is None:
            return

        arr = np.frombuffer(buf, dtype=np.int16).copy()
        if arr.size == 0:
            return

        if ch > 1:
            if arr.size % ch != 0:
                ch = 1
            frames = arr.size // ch
            x = arr.reshape(frames, ch).astype(np.float32)
        else:
            x = arr.astype(np.float32).reshape(-1, 1)
            frames = x.shape[0]

        xfade_len = max(1, min(int(sr * self._xfade_ms / 1000.0), frames))
        if self._prev_tail is None or self._prev_tail.shape[0] != x.shape[1]:
            self._prev_tail = x[0, :].copy()

        if xfade_len > 1:
            alphas = np.linspace(0.0, 1.0, xfade_len, dtype=np.float32).reshape(-1, 1)
            head = x[:xfade_len, :]
            x[:xfade_len, :] = (1.0 - alphas) * self._prev_tail[None, :] + alphas * head

        y = x / 32768.0
        s = float(self._softclip_strength)
        y = np.tanh(s * y) / np.tanh(s)

        self._prev_tail = y[-1, :].copy()

        y = (y * 32767.0).astype(np.int16)
        if ch > 1:
            y = y.reshape(-1)
        out_bytes = y.tobytes()

        try:
            mv = memoryview(buf)
            if not mv.readonly and len(mv) == len(out_bytes):
                mv[:] = out_bytes
                return
        except Exception:
            pass

        try:
            setattr(af, "data", out_bytes)
        except Exception:
            for alt in ("buffer", "buf", "pcm", "samples_bytes"):
                try:
                    setattr(af, alt, out_bytes)
                    break
                except Exception:
                    continue
