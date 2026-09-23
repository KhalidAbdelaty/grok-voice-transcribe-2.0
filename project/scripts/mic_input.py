"""Caller microphone: audio frames in, 100 ms PCM16 chunks out.

Frames arrive from the streamlit-webrtc sink callback (aiortc's thread) or
from a local sounddevice stream (scripts/local_audio.py), and the call
engine reads the state from another thread, so everything here is guarded
by one lock.

Three gate modes decide where captured audio goes:

- "closed"  - nothing is forwarded; the last PREROLL_MAX_S is kept in a ring
              so opening the gate can include the syllable that was already
              being spoken (the first word used to get clipped while the app
              reran to open the gate).
- "open"    - 100 ms chunks go to the STT session as they fill.
- "hold"    - everything is buffered, nothing forwarded. Used right after a
              barge-in or while an agent turn is still finalizing: the
              caller's words must not reach the socket while the agent's
              utterance is still the active one, or they would be
              transcribed into it (experiment_log.md entry 14's bleed).

Speech detection (experiment_log.md entry 16). The previous version kept a
noise floor that started at a fixed 0.004 and only learned from frames it
already considered quiet. With browser auto-gain on, ordinary room noise
sat above the first threshold, so the floor never moved, every frame
counted as speech, and the engine believed the caller never stopped
talking - it cancelled every reply as "you were still talking". Now the
floor is the 10th percentile of the last FLOOR_WINDOW_S of frame levels:
it follows the room up and down on its own, and because even continuous
speech has gaps between syllables, one long sentence does not drag it up
to the voice (the failure GPT Live Transcribe's SpeechGate documents).
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
CHUNK_BYTES = BYTES_PER_SECOND // 10
# Long enough to hold an interruption from its first syllable until the
# listener transcription confirms it (can take a second or two).
PREROLL_MAX_S = 3.0

FLOOR_WINDOW_S = 8.0
FLOOR_PERCENTILE = 10
FLOOR_MIN, FLOOR_MAX = 0.0005, 0.08
SPEECH_MIN_RMS = 0.006
SPEECH_FLOOR_RATIO = 3.0
BARGE_MIN_RMS = 0.015
BARGE_MIN_RMS_HEADSET = 0.010
HEADSET_COUPLING = 10 ** (-30 / 20)  # -30 dB
BARGE_FLOOR_RATIO = 4.0
# How much of the agent's playback actually reaches the mic ("coupling") is
# measured during the call, not guessed (entry 19): a fixed 0.8 x playback
# bar on the local path meant a headset caller at -28 dBFS could never
# clear it. Until enough frames are measured, start from these per path
# (the browser's echo canceller removes most of the leak).
COUPLING_SEED = {"browser": 0.1, "local": 0.3}
COUPLING_WINDOW_S = 6.0
# 60th percentile of mic/playback while the agent plays: sits at the echo's
# typical level (the syllable peaks are covered by the margins below), and
# only moves if the caller talks over more than 40% of the agent's time.
COUPLING_PERCENTILE = 60
COUPLING_MIN_FRAMES = 50
COUPLING_MIN, COUPLING_MAX = 0.002, 2.0
BARGE_ECHO_MARGIN = 3.0
OVER_ECHO_MARGIN = 2.0
# Gaps shorter than this don't reset a "continuous speech" run.
RUN_GAP_S = 0.12
# Gaps between words inside one phrase: an over-echo stretch that pauses
# for less than this is still the same utterance.
PHRASE_GAP_S = 0.7
# Barge-in candidates are measured as loud time within this window, not one
# unbroken run: syllables of quick words ("Sure,") stay above the bar for
# ~150 ms with gaps longer than RUN_GAP_S between them.
LOUD_WINDOW_S = 0.5
# Noise-gate detection: over this window of non-speech frames, the share at
# digital silence. A live mic always carries some noise; Windows' Voice
# Clarity / audio enhancements gate it away. Measured on a GM301 in a quiet
# room: frame RMS ~0.000015 through MME (77% zero samples, the rest +-1 LSB),
# exactly 0 through WASAPI shared, ~0.00007 through WASAPI exclusive.
GATE_SILENCE_RMS = 2.5e-5  # -92 dBFS
GATE_WINDOW_S = 10.0
GATE_MIN_FRAMES = 250  # 5 s of 20 ms frames before judging
GATE_ZERO_SHARE = 0.5


def _soft_limit(x: np.ndarray) -> np.ndarray:
    """Linear up to 0.8 of full scale, then a tanh knee instead of clipping."""
    mag = np.abs(x)
    knee = 0.8
    over = mag > knee
    if over.any():
        x = x.copy()
        x[over] = np.sign(x[over]) * (knee + (1 - knee) * np.tanh((mag[over] - knee) / (1 - knee)))
    return x


class MicInput:
    def __init__(self, boost: float = 1.0, source: str = "browser", line=None) -> None:
        self._lock = threading.Lock()
        self.line = line  # a phone_line.PhoneLine, or None for the mic as-is
        self._resampler = None
        self._resampler_key: tuple | None = None
        self.session = None
        self.mode = "closed"
        # A second consumer that gets every chunk regardless of the gate: the
        # barge-in listener transcription while an agent is speaking.
        self.tap = None
        self._tap_chunk = bytearray()
        self.boost = max(1.0, float(boost))
        self.source = source
        self.coupling = COUPLING_SEED.get(source, 0.3)
        self._coupling_hist: deque[tuple[float, float]] = deque()  # (time, mic_rms / playback_rms)
        self._coupling_since = 0
        self.echo_playing = False
        self._chunk = bytearray()
        self._ring = bytearray()
        self._hold = bytearray()
        self.level = 0.0
        self.rms = 0.0
        self.noise_floor = 0.004
        self.echo_ref = 0.0
        self._hist: deque[tuple[float, float]] = deque()  # (time, rms)
        self._frames_since_floor = 0
        self._frame_times: deque[float] = deque()
        self.last_frame_at = 0.0
        self.last_speech_at = 0.0
        self.last_any_speech_at = 0.0
        self.heard_speech = False
        # Speech louder than a leaked copy of the agent's recent audio: the
        # caller for sure, not the speakers (entry 17).
        self.last_over_echo_at = 0.0
        self.heard_over_echo = False
        self._phrase_start: float | None = None
        self._run_start: float | None = None
        self._run_last: float = 0.0
        self._loud_start: float | None = None
        self._loud_last: float = 0.0
        self._loud_frames: deque[tuple[float, float]] = deque()  # (time, duration) above the barge-in bar
        self._quiet_frames: deque[tuple[float, bool]] = deque()  # (time, exact digital zero) below speech
        self.frames_total = 0
        self.forwarded_bytes = 0
        self.first_error: str | None = None
        self.errors = 0

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def on_frame(self, frame) -> None:
        """streamlit-webrtc audio sink callback (an av.AudioFrame)."""
        try:
            import av

            key = (frame.format.name, frame.layout.name, frame.sample_rate)
            if self._resampler is None or key != self._resampler_key:
                # A STOP/START or a device switch can change the browser's
                # frame format; the old resampler would then raise on every
                # frame and the mic would go silent without a word.
                self._resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
                self._resampler_key = key
            for chunk in self._resampler.resample(frame):
                samples = chunk.to_ndarray()
                if samples.size:
                    self.feed_pcm(samples.astype(np.int16).tobytes())
        except Exception as exc:  # noqa: BLE001 - never let a bad frame kill the media callback
            self.errors += 1
            if self.first_error is None:
                self.first_error = f"{type(exc).__name__}: {exc}"
            self._resampler = None

    def feed_pcm(self, pcm: bytes) -> None:
        """16 kHz mono PCM16 in (local audio and the headless tests call this
        directly)."""
        if not pcm:
            return
        samples = np.frombuffer(pcm[: len(pcm) & ~1], dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return
        if self.boost > 1.0:
            samples = _soft_limit(samples * self.boost)
            pcm = (samples * 32767.0).astype(np.int16).tobytes()
        # Levels come from the mic as captured. A frame at digital silence is
        # what a Windows noise gate (Voice Clarity / audio enhancements) leaves
        # behind; a live mic always has some noise.
        rms = float(np.sqrt(np.mean(samples * samples)))
        zero_frame = rms < GATE_SILENCE_RMS
        dur = samples.size / SAMPLE_RATE
        if self.line is not None:
            # Simulated phone line (phone_line.py): only what is forwarded to
            # Transcribe is degraded. Measuring after the 300-3400 Hz filter
            # made the caller read quieter and barge-in stop working.
            pcm = self.line.process(pcm)
            if not pcm:
                return
        now = time.time()
        forward: list[bytes] = []
        with self._lock:
            self.frames_total += 1
            self.last_frame_at = now
            self._frame_times.append(now)
            while self._frame_times and now - self._frame_times[0] > 1.0:
                self._frame_times.popleft()
            self._update_levels(rms, now, dur, zero_frame)
            if self.mode == "open":
                self._chunk.extend(pcm)
                while len(self._chunk) >= CHUNK_BYTES:
                    forward.append(bytes(self._chunk[:CHUNK_BYTES]))
                    del self._chunk[:CHUNK_BYTES]
                session = self.session
            else:
                session = None
                if self.mode == "hold":
                    self._hold.extend(pcm)
                self._ring.extend(pcm)
                excess = len(self._ring) - int(PREROLL_MAX_S * BYTES_PER_SECOND)
                if excess > 0:
                    del self._ring[:excess]
            if session is not None:
                # Inside the lock so a concurrent open_gate() flush can never
                # be overtaken by a later live chunk.
                for piece in forward:
                    session.feed_live_audio(piece)
                    self.forwarded_bytes += len(piece)
            if self.tap is not None:
                self._tap_chunk.extend(pcm)
                while len(self._tap_chunk) >= CHUNK_BYTES:
                    self.tap.feed_live_audio(bytes(self._tap_chunk[:CHUNK_BYTES]))
                    del self._tap_chunk[:CHUNK_BYTES]

    def _update_floor(self, now: float) -> None:
        levels = np.fromiter((r for _, r in self._hist), dtype=np.float32, count=len(self._hist))
        if levels.size < 10:
            # First ~200 ms: seed from the quietest frame heard so far.
            floor = float(levels.min())
        else:
            floor = float(np.percentile(levels, FLOOR_PERCENTILE))
        self.noise_floor = min(FLOOR_MAX, max(FLOOR_MIN, floor))

    def _thresholds(self) -> tuple[float, float]:
        speech = max(SPEECH_MIN_RMS, SPEECH_FLOOR_RATIO * self.noise_floor)
        leak = self.coupling * self.echo_ref  # expected agent echo in the mic right now
        # On a headset (measured coupling under -30 dB) the agent barely
        # reaches the mic, so a quieter voice can pause it; a false pause only
        # costs a 1.5 s hiccup before it resumes.
        headset = len(self._coupling_hist) >= COUPLING_MIN_FRAMES and self.coupling < HEADSET_COUPLING
        floor = BARGE_MIN_RMS_HEADSET if headset else BARGE_MIN_RMS
        barge = max(floor, BARGE_FLOOR_RATIO * self.noise_floor, BARGE_ECHO_MARGIN * leak)
        return speech, barge

    def _over_echo_threshold(self, speech_thr: float) -> float:
        return max(speech_thr, OVER_ECHO_MARGIN * self.coupling * self.echo_ref)

    def _update_coupling(self, rms: float, now: float) -> None:
        """Learn how much of the agent's playback reaches this mic, from the
        frames heard while it plays (see COUPLING_PERCENTILE)."""
        if not self.echo_playing or self.echo_ref < 0.02:
            return
        self._coupling_hist.append((now, rms / self.echo_ref))
        while self._coupling_hist and now - self._coupling_hist[0][0] > COUPLING_WINDOW_S:
            self._coupling_hist.popleft()
        self._coupling_since += 1
        if len(self._coupling_hist) >= COUPLING_MIN_FRAMES and self._coupling_since >= 5:
            self._coupling_since = 0
            ratios = np.fromiter((r for _, r in self._coupling_hist), dtype=np.float32, count=len(self._coupling_hist))
            self.coupling = float(min(COUPLING_MAX, max(COUPLING_MIN, np.percentile(ratios, COUPLING_PERCENTILE))))

    def _update_levels(self, rms: float, now: float, dur: float = 0.02, zero_frame: bool = False) -> None:
        self.rms = rms
        self._hist.append((now, rms))
        while self._hist and now - self._hist[0][0] > FLOOR_WINDOW_S:
            self._hist.popleft()
        self._frames_since_floor += 1
        if self._frames_since_floor >= 5 or len(self._hist) < 10:
            self._frames_since_floor = 0
            self._update_floor(now)

        self._update_coupling(rms, now)
        speech_thr, barge_thr = self._thresholds()
        if len(self._hist) >= 10 and rms > speech_thr:
            self.heard_speech = True
            self.last_speech_at = now
            self.last_any_speech_at = now
            if self._run_start is None or now - self._run_last > RUN_GAP_S:
                self._run_start = now
            self._run_last = now
        if len(self._hist) >= 10 and rms > self._over_echo_threshold(speech_thr):
            # Only where the leaked agent sits below speech level (a headset):
            # with open speakers its louder syllables clear the over-echo bar
            # too, and the "phrase" would reach back into the agent's voice.
            quiet_leak = self.coupling * self.echo_ref < speech_thr
            if quiet_leak and (self._phrase_start is None or now - self.last_over_echo_at > PHRASE_GAP_S):
                self._phrase_start = now
            elif not quiet_leak:
                self._phrase_start = None
            self.last_over_echo_at = now
            self.heard_over_echo = True
        if len(self._hist) >= 10 and rms > barge_thr:
            if self._loud_start is None or now - self._loud_last > RUN_GAP_S:
                self._loud_start = now
            self._loud_last = now
            self._loud_frames.append((now, dur))
        while self._loud_frames and now - self._loud_frames[0][0] > LOUD_WINDOW_S:
            self._loud_frames.popleft()
        if rms <= speech_thr:
            self._quiet_frames.append((now, zero_frame))
        while self._quiet_frames and now - self._quiet_frames[0][0] > GATE_WINDOW_S:
            self._quiet_frames.popleft()
        target = min(1.0, rms * 6.0)
        # Fast attack, slow release, so the meter reads as a level, not flicker.
        self.level = target if target > self.level else self.level * 0.85 + target * 0.15

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def set_echo_reference(self, playback_rms: float, playing: bool = False) -> None:
        """What the agent is currently playing (0 when silent). Barge-in has
        to be louder than a leaked copy of it. `playing` means the agent is
        audibly speaking right now (not paused, not just its echo tail), so
        these frames can be used to measure the coupling."""
        with self._lock:
            self.echo_ref = max(0.0, float(playback_rms))
            self.echo_playing = bool(playing)

    def quiet_for_s(self) -> float:
        """Time since the last frame above the speech threshold, whatever the
        gate mode - including the agent's own voice leaking in. Unlike
        silence_s() it is not reset when the gate opens."""
        with self._lock:
            if not self.last_any_speech_at:
                return float("inf")
            return time.time() - self.last_any_speech_at

    def silence_s(self) -> float:
        with self._lock:
            return time.time() - self.last_speech_at if self.last_speech_at else float("inf")

    def speech_run_s(self) -> float:
        with self._lock:
            if self._run_start is None or time.time() - self._run_last > RUN_GAP_S:
                return 0.0
            return self._run_last - self._run_start

    def speech_run_started_at(self) -> float:
        with self._lock:
            if self._run_start is None or time.time() - self._run_last > RUN_GAP_S:
                return 0.0
            return self._run_start

    def loud_run_s(self) -> float:
        """Continuous speech above the (higher, echo-aware) barge-in threshold."""
        with self._lock:
            if self._loud_start is None or time.time() - self._loud_last > RUN_GAP_S:
                return 0.0
            return self._loud_last - self._loud_start

    def loud_recent_s(self) -> float:
        """Time above the barge-in bar within the last LOUD_WINDOW_S."""
        with self._lock:
            cutoff = time.time() - LOUD_WINDOW_S
            return sum(d for t, d in self._loud_frames if t >= cutoff)

    def phrase_started_at(self) -> float:
        """Start of the caller's current phrase, heard over the echo: quieter
        than the barge-in bar, so a soft first word ("Sure, ...") isn't lost
        when only the louder words that follow trigger the pause."""
        with self._lock:
            if self._phrase_start is None or time.time() - self.last_over_echo_at > PHRASE_GAP_S:
                return 0.0
            return self._phrase_start

    def loud_run_started_at(self) -> float:
        with self._lock:
            if self._loud_start is None or time.time() - self._loud_last > RUN_GAP_S:
                return 0.0
            return self._loud_start

    def frames_recent(self, within_s: float = 2.0) -> bool:
        return self.last_frame_at > 0 and time.time() - self.last_frame_at < within_s

    def reset_runs(self) -> None:
        with self._lock:
            self._run_start = None
            self._loud_start = None
            self.heard_speech = False
            self.last_speech_at = 0.0

    def stats(self) -> dict:
        """Everything the diagnostics panel and the call log show."""
        with self._lock:
            speech_thr, barge_thr = self._thresholds()
            now = time.time()
            quiet = len(self._quiet_frames)
            zero_share = sum(1 for _, z in self._quiet_frames if z) / quiet if quiet else 0.0
            return {
                "source": self.source,
                "gated_pct": round(100 * zero_share),
                "gated": quiet >= GATE_MIN_FRAMES and zero_share >= GATE_ZERO_SHARE,
                "frames_per_s": len(self._frame_times),
                "rms": round(self.rms, 5),
                "dbfs": round(20 * np.log10(max(self.rms, 1e-6)), 1),
                "floor": round(self.noise_floor, 5),
                "floor_dbfs": round(20 * np.log10(max(self.noise_floor, 1e-6)), 1),
                "speech_thr": round(speech_thr, 5),
                "barge_thr": round(barge_thr, 5),
                "over_echo_thr": round(self._over_echo_threshold(speech_thr), 5),
                "echo_ref": round(self.echo_ref, 4),
                "coupling": round(self.coupling, 4),
                "coupling_db": round(20 * np.log10(max(self.coupling, 1e-6)), 1),
                "coupling_measured": len(self._coupling_hist) >= COUPLING_MIN_FRAMES,
                "speaking_now": bool(self.last_any_speech_at and now - self.last_any_speech_at < 0.25),
                "last_speech_ago_s": round(now - self.last_any_speech_at, 1) if self.last_any_speech_at else None,
                "mode": self.mode,
                "forwarded_s": round(self.forwarded_bytes / BYTES_PER_SECOND, 1),
                "boost": self.boost,
                "errors": self.errors,
                "first_error": self.first_error,
            }

    # ------------------------------------------------------------------
    # gate
    # ------------------------------------------------------------------
    def open_gate(self, session, preroll_s: float = 0.3, prefix: bytes = b"") -> None:
        """Start forwarding to `session`: `prefix` (caller audio saved from
        earlier), then held audio (if the gate was holding) or the last
        `preroll_s` of the ring, then live chunks."""
        with self._lock:
            if self.mode == "hold" and self._hold:
                head = bytes(prefix) + bytes(self._hold)
            else:
                n = int(preroll_s * BYTES_PER_SECOND) & ~1
                head = bytes(prefix) + (bytes(self._ring[-n:]) if n else b"")
            self._hold = bytearray()
            self._ring = bytearray()
            self._chunk = bytearray()
            self.session = session
            self.mode = "open"
            self.heard_speech = False
            self.heard_over_echo = False
            self.last_speech_at = 0.0
            self._run_start = None
            for i in range(0, len(head), CHUNK_BYTES):
                piece = head[i : i + CHUNK_BYTES]
                session.feed_live_audio(piece)
                self.forwarded_bytes += len(piece)

    def hold(self, include_s: float = 0.0) -> None:
        """Buffer everything from now on (plus the last `include_s` already
        captured) without forwarding."""
        with self._lock:
            n = int(include_s * BYTES_PER_SECOND) & ~1
            self._hold = bytearray(self._ring[-n:]) if n else bytearray()
            self._chunk = bytearray()
            self.mode = "hold"

    def set_tap(self, session, preroll_s: float = 0.0) -> None:
        """Copy every captured chunk to `session` (None stops it), starting
        with the last `preroll_s` already in the ring."""
        with self._lock:
            self._tap_chunk = bytearray()
            self.tap = session
            if session is None:
                return
            n = int(preroll_s * BYTES_PER_SECOND) & ~1
            head = bytes(self._ring[-n:]) if n else b""
            for i in range(0, len(head), CHUNK_BYTES):
                session.feed_live_audio(head[i : i + CHUNK_BYTES])

    def take_hold(self) -> bytes:
        """Hand over (and clear) what the gate has been holding."""
        with self._lock:
            data = bytes(self._hold)
            self._hold = bytearray()
            return data

    def close_gate(self) -> None:
        with self._lock:
            self.mode = "closed"
            self._chunk = bytearray()
            self._hold = bytearray()
            self._ring = bytearray()

    def held_seconds(self) -> float:
        with self._lock:
            return len(self._hold) / BYTES_PER_SECOND
