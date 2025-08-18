# gated_avsync.py
import asyncio
import time
from collections import deque
from typing import Optional, Union
from livekit.rtc import AudioSource, VideoSource
from livekit.rtc import VideoFrame as LKVideoFrame
from livekit.rtc import AudioFrame as LKAudioFrame


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
            maxsize=50
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
