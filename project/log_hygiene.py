"""Keep the terminal readable: drop known-harmless library noise, keep real
problems, and give the app its own short lifecycle lines.

Dropped (each one checked, none of them is an error in this app):

- "The fragment with id ... does not exist anymore" (INFO,
  streamlit.runtime.app_session): a fragment's run_every timer fired just
  after a full rerun replaced it. Streamlit simply skips that run.
- "AudioSourceTrack: Audio frame callback is too slow (125 ms > 20 ms)" for
  the first few frames of a WebRTC playback connection: a one-off warm-up
  stall. If it keeps happening, one summary line is printed instead.
- "Task was destroyed but it is pending!" for aiortc / aioice tasks: WebRTC
  internals left behind when a browser tab closes or reloads. Pending tasks
  of our own are fixed at the source (LiveCallSession / TtsStreamer close
  their loops cleanly), not hidden here.

`install()` is idempotent: run_app.py calls it before Streamlit starts, the
app calls it again on every rerun.
"""
from __future__ import annotations

import logging
import sys
import time

_SENTINEL = "_qivora_log_hygiene"
STARTUP_STALLS_ALLOWED = 3
STALL_SUMMARY_WINDOW_S = 60.0
STALL_SUMMARY_MIN = 5


class _DropMessage(logging.Filter):
    def __init__(self, *needles: str) -> None:
        super().__init__()
        self.needles = needles

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(n in msg for n in self.needles)


class _PendingWebRtcTasks(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Task was destroyed but it is pending" not in msg:
            return True
        return not ("aiortc" in msg or "aioice" in msg)


class _PlaybackStalls(logging.Filter):
    """The first few "too slow" warnings are warm-up; after that, summarize."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0
        self.window: list[float] = []
        self.last_summary = 0.0

    def filter(self, record: logging.LogRecord) -> bool:
        if "too slow" not in record.getMessage():
            return True
        self.seen += 1
        if self.seen <= STARTUP_STALLS_ALLOWED:
            return False
        now = time.time()
        self.window = [t for t in self.window if now - t < STALL_SUMMARY_WINDOW_S] + [now]
        if len(self.window) >= STALL_SUMMARY_MIN and now - self.last_summary > STALL_SUMMARY_WINDOW_S:
            self.last_summary = now
            logging.getLogger("qivora").warning(
                "audio  agent-voice playback stalled %d times in the last minute (CPU busy?)", len(self.window)
            )
        return False


def install() -> None:
    root = logging.getLogger()
    if getattr(root, _SENTINEL, False):
        return
    setattr(root, _SENTINEL, True)

    logging.getLogger("streamlit.runtime.app_session").addFilter(_DropMessage("does not exist anymore"))
    logging.getLogger("streamlit_webrtc.source").addFilter(_PlaybackStalls())
    logging.getLogger("asyncio").addFilter(_PendingWebRtcTasks())

    # The app's own lifecycle lines: "01:27:18  call   started  ...".
    qlog = logging.getLogger("qivora")
    if not qlog.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        qlog.addHandler(handler)
        qlog.setLevel(logging.INFO)
        qlog.propagate = False
