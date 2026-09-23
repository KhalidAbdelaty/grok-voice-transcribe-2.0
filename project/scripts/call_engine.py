"""The live call's turn-taking, off the Streamlit script thread.

Streamlit can only redraw between script runs, so the old app blocked its
own script for the whole agent turn (think, synthesize, play, wait) and
then reran to open the mic - which is where the "still thinking" pause and
the clipped first word came from. `CallEngine` runs the call on its own
thread instead; the page just polls `snapshot()`.

    listening --(speech_final)--> thinking --(first TTS audio)--> speaking
        ^                            |                               |
        |                (caller keeps talking)          (playback ends, or barge-in)
        +----------------------------+-------------------------------+

Latency tricks (experiment_log.md entry 15):

- Speculative start: once the caller has been quiet ~350 ms with text on
  screen, the reply is already being written from that text while Smart
  Turn is still deciding. If the final transcript matches, the head start
  is kept; if not (or the caller resumes), it is thrown away unheard.
- The reply streams straight into a TTS socket, and each PCM chunk goes
  to the caller's speakers (WebRTC) and into the same Transcribe session
  at the same moment.
- The engine opens the mic itself when playback ends; no rerun needed.

Barge-in: during playback the mic is still measured (not forwarded). The
playback now runs through the same WebRTC connection as the mic, so the
browser's echo canceller has the agent's audio as its reference. Sustained
speech above a raised threshold cuts playback, cancels the TTS utterance,
finalizes the agent's utterance on the socket, and holds the caller's
audio until that utterance is closed - then opens a live turn with the
held audio first. One mono stream can't carry two voices at once, so the
caller's words wait a few hundred ms rather than landing inside the
agent's utterance.
"""
from __future__ import annotations

import difflib
import json
import logging
import queue
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.agent_brain import FAST_MODEL, AgentBrain, AgentReply, ReplyCancelled
from scripts.live_call_session import BYTES_PER_SECOND, LiveCallSession, TurnResult, stitch_partials, stitch_parts
from scripts.mic_input import PREROLL_MAX_S, MicInput
from scripts.phone_line import PhoneLine
from scripts.tts_stream import TtsStreamer, TtsUtterance, synthesize_pcm_rest

# Short, readable lifecycle lines for the terminal (log_hygiene.py formats it).
QLOG = logging.getLogger("qivora")

SPECULATE_AFTER_SILENCE_S = 0.25
MAX_SPECULATIONS = 4  # per caller turn; each discarded guess costs one short model call
# Our own end of turn, when the server's Smart Turn never closes it: once the
# mic has been quiet for smart_turn_timeout plus this margin. On the 8 kHz
# phone leg the server's timeout did not fire at all (call_20260923_051305:
# 4 of 6 phone turns closed only by the old fixed 6 s watchdog, and the
# caller spoke up into the silence and cut the late reply).
SILENCE_WATCHDOG_MARGIN_S = 0.5
BARGE_MIN_PLAYBACK_S = 0.4
# Voice barge-in (see _step_barge_in).
STRONG_CANDIDATE_S = 0.15  # a short "Wait." is ~0.28 s of loud frames spread over 0.6 s
FALSE_PAUSE_S = 1.5
FALSE_PAUSE_QUIET_S = 0.6
MAX_PAUSE_S = 4.0
LISTENER_WINDOW_WORDS = 8
FALLBACK_CUT_S = 0.6
ABANDONED_AFTER_S = 30.0
# Speech with short gaps ("Sure, one second.") can take a while to count as
# 0.35 s of *continuous* speech, so keep a generous stretch from before the
# detection point.
BARGE_INCLUDE_S = 1.2
# Transcribe's voice-activity threshold on the 8 kHz mu-law phone leg:
# narrowband audio scores lower (Grok LiveOps, config.TELEPHONY_VAD_THRESHOLD).
PHONE_VAD_THRESHOLD = 0.04
# How far before the pause point a soft first word can still be claimed
# (MicInput.phrase_started_at, headset-level leak only).
SOFT_START_MAX_S = 1.5
# "You were still talking" while the reply is being prepared: the caller must
# still be talking this long after the turn closed, and it may fire at most
# MAX_CONTINUES times between two agent replies (entry 16: with a stuck noise
# floor it fired on every turn and the agent never answered; a dictated
# number can legitimately need two or three).
CONTINUE_SPEECH_S = 0.6
MAX_CONTINUES = 3
# The agent's first audio waits until the caller has been quiet this long
# (or MAX_WAIT_FOR_QUIET_S has passed, so steady background noise can never
# block a reply forever).
CALLER_QUIET_S = 0.3
MAX_WAIT_FOR_QUIET_S = 2.5
# A caller turn that ends on fewer digits than a phone number: wait this long
# after the turn closed before answering, in case they are still dictating.
NUMBER_HOLD_S = 1.8
FULL_NUMBER_DIGITS = 7
ECHO_TAIL_S = 0.25
# The mic reopens once it has been this quiet after the agent's audio ends,
# or after ECHO_SETTLE_MAX_S at the latest (see _echo_settled).
# Longer than the gaps between words, so a pause inside the leaked sentence
# doesn't count as the echo having died away.
ECHO_QUIET_S = 0.35
ECHO_SETTLE_MAX_S = 1.2
ECHO_WINDOW_S = 2.0
# Paced playback (see _advance_playback).
PLAYBACK_LEAD_S = 0.25
STT_LEAD_S = 0.1
CHUNK_PUSH_BYTES = 3200
# Playback level the echo reference looks back over (speaker-to-mic latency).
ECHO_LOOKBACK_S = 0.3
LOG_DIR = Path(__file__).resolve().parent.parent / "results" / "live_logs"


@dataclass
class EngineSettings:
    diarize: bool = True
    format_: bool = False
    # 0.8, not 0.7: a real hesitation ("My ...") closed at 0.704 in
    # call_20260923_031600, while finished turns came in at 0.79-0.99.
    smart_turn: float = 0.8
    smart_turn_timeout: int = 3000
    vad_threshold: float | None = None
    model: str = FAST_MODEL
    priority: bool = True
    speculative: bool = True
    barge_in: bool = True
    language: str = "en"
    audio_path: str = "browser"
    mic_boost: float = 1.0
    filler_words: bool = True
    phone_line: bool = False


class CallLog:
    """One JSONL file per call in results/live_logs: phase changes, turns,
    system notes, and one stats line per second - so a call that "doesn't
    hear me" can be read back afterwards instead of guessed at."""

    def __init__(self, enabled: bool = True) -> None:
        self.path: Path | None = None
        self._lock = threading.Lock()
        self._fh = None
        if enabled:
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                self.path = LOG_DIR / time.strftime("call_%Y%m%d_%H%M%S.jsonl")
                self._fh = open(self.path, "a", encoding="utf-8")
            except OSError:
                self._fh = None

    def write(self, kind: str, **fields: Any) -> None:
        if self._fh is None:
            return
        line = json.dumps({"t": round(time.time(), 3), "kind": kind, **fields}, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None


class NullPlayback:
    """Stand-in for the WebRTC playback track (headless runs)."""

    def push(self, pcm: bytes) -> None:
        pass

    def clear(self) -> None:
        pass


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def _same_utterance(guess: str, final: str) -> bool:
    """A reply written from `guess` still fits `final`: identical once
    case and punctuation are ignored, or a near-identical re-recognition
    of the same words (the utterance final is re-stitched server-side and
    can differ by a word or a hyphen)."""
    a, b = _norm(guess), _norm(final)
    if a == b:
        return True
    return len(b) > 0 and abs(len(a) - len(b)) <= max(6, len(b) // 12) and difflib.SequenceMatcher(None, a, b).ratio() >= 0.93


# Acknowledgements that mean "go on", not "stop" (LiveKit calls filtering
# them adaptive interruption handling).
BACKCHANNEL = {
    "mm", "mmm", "mhm", "hmm", "hm", "uh", "um", "umm", "er", "erm", "huh", "uhhuh", "uh-huh", "ah", "aha", "oh",
    "yeah", "yes", "yep", "yup", "okay", "ok", "right", "sure", "alright", "cool",
}
# Words that mean "stop talking" on their own: one of these from the listener
# cuts the agent, where any other single word needs a pause or a second word.
INTERRUPT_WORDS = {"wait", "stop", "hold", "no", "nope", "sorry", "excuse", "what", "hey", "pardon",
                   "actually", "hang", "listen"}
# The mic heard the caller over the echo this recently: a single new word is
# their own, not a mis-heard echo.
RECENT_OVER_ECHO_S = 1.0
# Hesitations Transcribe returns with filler_words=true.
FILLERS = {"um", "umm", "uh", "uhm", "er", "erm", "hmm", "hm", "mm", "mmm", "ah"}


NUMBER_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
SPOKEN_SYMBOLS = {"dot", "at", "dash", "hyphen", "underscore", "slash", "point"}


def _tokens(text: str) -> list[str]:
    return [NUMBER_WORDS.get(t, t) for t in re.findall(r"[\w'-]+", text.lower())]


def _trailing_digits(text: str) -> int:
    """How many digits the text ends on ("my number is 0 2 1 5." -> 4,
    "zero one zero" -> 3, "one second" -> 0)."""
    count = 0
    for tok in reversed(_tokens(text)):
        if not tok.isdigit():
            break
        count += len(tok)
    return count


def _spoken_vocab(text: str) -> set[str]:
    """Every token the listener could hear when this text is spoken aloud.
    The written reply says "0105551234" and "khalid.demo@qivorasync.com"; TTS
    reads them as "0 1 0 5 5 5 ..." and "khalid dot demo at q i v o r a
    sync dot com", and without these forms the agent's own read-back,
    leaking through speakers, counted as the caller's words (entry 18)."""
    vocab: set[str] = set()
    for tok in _tokens(text):
        vocab.add(tok)
        if any(c.isdigit() for c in tok):
            vocab.update(c for c in tok if c.isdigit())
        if len(tok) <= 24:
            vocab.update(tok)  # spelled-out letters
    if re.search(r"\w[.@]\w", text):
        vocab.update(SPOKEN_SYMBOLS)
    return vocab


def _is_echo_word(word: str, vocab: set[str]) -> bool:
    """A word the agent itself is saying (heard back through the speakers),
    allowing for the listener mis-hearing it slightly."""
    if word in vocab:
        return True
    if len(word) < 4:
        return False
    return any(len(v) >= 4 and difflib.SequenceMatcher(None, word, v).ratio() >= 0.8 for v in vocab)


def _rms(pcm: bytes) -> float:
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) & ~1], dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0


class AgentJob:
    """One agent reply: model stream -> TTS utterance -> PCM queue, on its
    own threads. Nothing here plays anything; the engine decides whether
    (and when) the audio is used, which is what makes speculation free."""

    def __init__(self, brain: AgentBrain, tts: TtsStreamer, voices: dict[str, str], pending_caller: str | None) -> None:
        self.stream = brain.reply_stream(pending_caller)
        self.pending_caller = pending_caller
        self.tts = tts
        self.voices = voices
        self.started_wall = time.time()
        self.speaker: str | None = None
        self.language = "en"
        self.said = ""
        self.reply: AgentReply | None = None
        self.error: str | None = None
        self.utt: TtsUtterance | None = None
        self.audio: "queue.Queue[bytes | None]" = queue.Queue()
        self.llm_done = threading.Event()
        self.cancelled = False
        self.used_rest_fallback = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        forwarder_started = False
        try:
            for kind, value in self.stream:
                if self.cancelled:
                    break
                if kind == "speaker":
                    self.speaker = value
                    self.language = self.stream.language
                    self.utt = self.tts.start(self.voices[value], self.language)
                    threading.Thread(target=self._forward_audio, daemon=True).start()
                    forwarder_started = True
                elif kind == "text":
                    self.said += value
                    if self.utt is not None:
                        self.utt.push_text(value)
            self.reply = self.stream.reply
            if self.utt is not None:
                self.utt.done_text()
        except ReplyCancelled:
            pass
        except Exception as exc:  # noqa: BLE001 - surfaced through .error
            self.error = f"{type(exc).__name__}: {exc}"
            if self.utt is not None:
                self.utt.cancel()
        finally:
            self.llm_done.set()
            if not forwarder_started:
                self.audio.put(None)

    def _forward_audio(self) -> None:
        utt = self.utt
        assert utt is not None
        try:
            while True:
                pcm = utt.audio.get()
                if pcm is None:
                    break
                if not self.cancelled:
                    self.audio.put(pcm)
            if utt.error and not utt.got_audio and not self.cancelled:
                # The socket failed before any audio: synthesize the whole
                # reply over REST instead of going silent.
                self.llm_done.wait(timeout=30)
                if self.reply is not None and not self.cancelled:
                    self.used_rest_fallback = True
                    self.audio.put(synthesize_pcm_rest(self.reply.text, self.voices[self.reply.speaker],
                                                       language=self.reply.language))
        except Exception as exc:  # noqa: BLE001
            self.error = self.error or f"TTS failed: {exc}"
        finally:
            self.audio.put(None)

    def cancel(self) -> None:
        self.cancelled = True
        self.stream.cancel()
        if self.utt is not None:
            self.utt.cancel()


class CallEngine:
    def __init__(
        self,
        *,
        settings: EngineSettings,
        voices: dict[str, str],
        keyterms: list[str],
        mic: MicInput | None = None,
        playback: Any = None,
        brain: AgentBrain | None = None,
        log: bool = True,
    ) -> None:
        self.settings = settings
        self.voices = voices
        self.keyterms = keyterms
        self.mic = mic or MicInput(boost=settings.mic_boost, source=settings.audio_path,
                                   line=PhoneLine() if settings.phone_line else None)
        self._ear_line = PhoneLine(dropout=0.0) if settings.phone_line else None
        self._pending_swap: dict[str, Any] | None = None
        self.log = CallLog(enabled=log)
        self.describe = ""  # e.g. "mic=Microphone (4- GM301) -> Speakers (4- GM301)", for the terminal line
        self._last_ui_touch = time.time()
        self._logged_phase = ""
        self._last_stats_log = 0.0
        self.continues = 0
        self.pending_caller_audio = b""
        self.agent_speaker_ids: set[int] = set()
        self.caller_speaker_ids: set[int] = set()
        self.phantoms_ignored: list[str] = []
        self._turn_audio_from = 0.0
        self._turn_had_prefix = False
        self._number_hold = False
        self.last_play_end = 0.0
        self.last_play_rms = 0.0
        self.prev_agent_text = ""
        self.listener: LiveCallSession | None = None
        self._reset_playback()
        self._unowned_seen = 0
        self.playback = playback or NullPlayback()
        self.brain = brain or AgentBrain(model=settings.model, priority=settings.priority)
        # The agents always answer in English, so no Arabic sockets up front.
        self.tts = TtsStreamer([voices["maya"], voices["nadia"]], language=settings.language, preopen=())
        self.session: LiveCallSession | None = None
        # Which Transcribe session a turn came from: speaker ids restart in
        # each one (phone line switch, reconnect), so they are only comparable
        # within a session.
        self.session_no = 0
        self.lock = threading.RLock()

        self.phase = "connecting"
        self.error: str | None = None
        self.turns: list[dict] = []
        self.record: list[dict] = []
        self.version = 0
        self.turn_no = 0
        self.all_events_archive: list[dict] = []
        self.interrupt_requested = False

        self.live = None
        self.live_started_wall = 0.0
        self.watchdog_fired = False
        self.job: AgentJob | None = None
        self.spec_text: str | None = None
        self.spec_hit = False
        self.spec_attempts = 0
        self.agent_turn = None
        self.audio_ended = False
        self.finish_sent = False
        self.interrupted = False
        self.play_end = 0.0
        self.play_segments: deque = deque(maxlen=400)
        self.first_audio_wall: float | None = None
        self.thinking_since = 0.0
        self.speaking_since = 0.0
        self.caller_speech_end = 0.0
        self.caller_turn_closed_at = 0.0
        self.agent_holding = False
        self.last_latency_ms: int | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _new_session(self, phone: bool | None = None) -> LiveCallSession:
        s = self.settings
        phone = s.phone_line if phone is None else phone
        return LiveCallSession(
            diarize=s.diarize,
            language=s.language,
            format_=s.format_,
            keyterms=self.keyterms,
            smart_turn=s.smart_turn,
            smart_turn_timeout=s.smart_turn_timeout,
            vad_threshold=PHONE_VAD_THRESHOLD if phone and s.vad_threshold is None else s.vad_threshold,
            filler_words=s.filler_words,
            wire="mulaw8k" if phone else "pcm16k",
        )

    def _new_listener(self, phone: bool | None = None) -> LiveCallSession:
        phone = self.settings.phone_line if phone is None else phone
        return LiveCallSession(
            diarize=False,
            language=self.settings.language,
            format_=False,
            keyterms=[],
            smart_turn=None,
            smart_turn_timeout=None,
            vad_threshold=PHONE_VAD_THRESHOLD if phone else None,
            wire="mulaw8k" if phone else "pcm16k",
        )

    # ------------------------------------------------------------------
    # phone line on/off mid-call
    # ------------------------------------------------------------------
    def set_phone_line(self, on: bool) -> None:
        """Switch the call to (or off) the 8 kHz mu-law phone leg. A socket's
        encoding is fixed when it connects, so new sessions are opened now, in
        the background, and swapped in at the next safe point: the caller's
        turn is open, they are quiet, and nothing has been transcribed yet."""
        with self.lock:
            pending = self._pending_swap
            if pending is not None and pending["on"] == on:
                return
            if pending is None and self.settings.phone_line == on:
                return
            if pending is not None:
                # Toggled back before the swap happened: drop the prepared sessions.
                self._pending_swap = None
                for s in (pending.get("session"), pending.get("listener")):
                    if s is not None:
                        threading.Thread(target=s.close, daemon=True).start()
                if self.settings.phone_line == on:
                    return
            swap: dict[str, Any] = {"on": on, "ready": False, "error": None, "session": None, "listener": None}
            self._pending_swap = swap

        def _prepare() -> None:
            try:
                session = self._new_session(phone=on)
                session.connect()
                listener = None
                if self.settings.barge_in:
                    listener = self._new_listener(phone=on)
                    listener.connect()
                swap.update(session=session, listener=listener, ready=True)
            except Exception as exc:  # noqa: BLE001
                swap["error"] = f"{type(exc).__name__}: {exc}"
                with self.lock:
                    if self._pending_swap is swap:
                        self._pending_swap = None
                    self._add_system(f"could not switch the phone line: {swap['error']}")

        threading.Thread(target=_prepare, daemon=True).start()

    def _swap_ready(self) -> bool:
        swap = self._pending_swap
        if swap is None or not swap["ready"] or self.job is not None:
            return False
        live = self.live
        text = stitch_partials(live.result.events)[0] if live is not None else ""
        return not text and self.mic.silence_s() > 0.5

    def _swap_wire(self) -> None:
        swap, self._pending_swap = self._pending_swap, None
        on = swap["on"]
        self.mic.close_gate()
        old_session, old_listener = self.session, self.listener
        if old_session is not None:
            self.all_events_archive.extend(old_session.all_events)
        for s in (old_session, old_listener):
            if s is not None:
                threading.Thread(target=s.close, daemon=True).start()
        self.mic.set_tap(None)
        self.mic.line = PhoneLine() if on else None
        self._ear_line = PhoneLine(dropout=0.0) if on else None
        self.settings.phone_line = on
        self._reset_turn_state()
        self._unowned_seen = 0
        # A new session numbers its diarized speakers from scratch.
        self.agent_speaker_ids.clear()
        self.caller_speaker_ids.clear()
        self.session = swap["session"]
        self.session_no += 1
        self.listener = swap["listener"]
        self.log.write("phone_line", on=on, wire=self.session.wire)
        self._add_system("phone line on - Transcribe now hears the call as 8 kHz \u03bc-law" if on
                         else "phone line off - back to 16 kHz PCM")
        self._start_listening()

    def start(self) -> None:
        """Connect (blocking, so a bad key fails on the Start button) and
        start the engine thread."""
        self.log.write("start", settings=asdict(self.settings), model=self.brain.model)
        self.tts.connect_all()
        threading.Thread(target=self.brain.warm_up, daemon=True).start()
        session = self._new_session()
        session.connect()
        with self.lock:
            self.session = session
            self.session_no += 1
            self.phase = "waiting_mic"
        QLOG.info("call   started  path=%s  %s  brain=%s", self.settings.audio_path,
                  self.describe or "", self.brain.model.replace("-0309", ""))
        if self.settings.barge_in:
            threading.Thread(target=self._connect_listener, daemon=True).start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _connect_listener(self) -> None:
        """The barge-in listener: a second, caller-only transcription that
        only hears the mic while an agent is talking. The main session can't
        do this job - the agent's own audio is streaming into it at that
        moment - and words, not loudness, are what tell a real interruption
        from echo, a cough or an "mm-hmm". No diarization, no keyterms (the
        keyterm is what turned a garbled echo into "Qivora Sync", entry 17),
        no Smart Turn."""
        try:
            listener = self._new_listener()
            listener.connect()
            with self.lock:
                self.listener = listener
            self.log.write("listener", status="connected")
        except Exception as exc:  # noqa: BLE001 - barge-in degrades to loudness only
            self.log.write("error", error=f"barge-in listener: {type(exc).__name__}: {exc}")

    def reconnect(self) -> None:
        old = self.session
        if old is not None:
            self.all_events_archive.extend(old.all_events)
            threading.Thread(target=old.close, daemon=True).start()
        session = self._new_session()
        session.connect()
        with self.lock:
            self._reset_turn_state()
            self._unowned_seen = 0
            # A new session numbers its diarized speakers from scratch.
            self.agent_speaker_ids.clear()
            self.caller_speaker_ids.clear()
            self.session = session
            self.session_no += 1
            self.error = None
            self.phase = "waiting_mic"
            self._add_system("reconnected - the transcript so far is kept")
        if self.settings.barge_in and (self.listener is None or not self.listener.alive):
            threading.Thread(target=self._connect_listener, daemon=True).start()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def end(self, reason: str = "call ended", wait: bool = False) -> None:
        """Stop the call now; the sockets close in the background (the STT
        socket waits for transcript.done) unless `wait` is set."""
        with self.lock:
            if self.phase == "ended":
                return
            self.phase = "ended"
            if self.job is not None:
                self.job.cancel()
            self.playback.clear()
            self.mic.close_gate()
            if reason:
                self._add_system(reason)
        self._stop.set()

        self.log.write("end", reason=reason, mic=self.mic.stats())
        latencies = [r["response_latency_ms"] for r in self.record if r.get("response_latency_ms")]
        avg = f"{sum(latencies) / len(latencies) / 1000:.1f}s" if latencies else "-"
        log_path = self.log.path
        try:
            shown = log_path.relative_to(LOG_DIR.parent.parent) if log_path else None
        except ValueError:
            shown = log_path
        QLOG.info("call   ended    %s  %d turns  avg reply %s  log=%s", reason, self.turn_no, avg, shown or "-")

        def _close() -> None:
            self.mic.set_tap(None)
            stop_audio = getattr(self.playback, "stop", None)  # LocalAudio owns real devices
            if callable(stop_audio):
                try:
                    stop_audio()
                except Exception:  # noqa: BLE001
                    pass
            for session in (self.session, self.listener):
                if session is not None:
                    session.close()
            self.tts.close()
            self.log.close()

        if wait:
            _close()
        else:
            threading.Thread(target=_close, daemon=True).start()

    def touch(self) -> None:
        self._last_ui_touch = time.time()
        for session in (self.session, self.listener):
            if session is not None:
                session.touch()

    def set_playback(self, playback: Any) -> None:
        with self.lock:
            self.playback = playback or NullPlayback()

    def request_interrupt(self) -> None:
        self.interrupt_requested = True

    @property
    def all_events(self) -> list[dict]:
        session = self.session
        return self.all_events_archive + (session.all_events if session else [])

    # ------------------------------------------------------------------
    # loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            if time.time() - self._last_ui_touch > ABANDONED_AFTER_S and self.phase not in ("ended", "ending"):
                # The browser tab was closed or reloaded: nobody will ever end
                # this call, and on the local path it holds the mic open.
                self.phase = "ending"
                threading.Thread(target=self.end, args=("no browser tab for 30s",), daemon=True).start()
            try:
                with self.lock:
                    self._step()
                    self._log_tick()
            except Exception as exc:  # noqa: BLE001 - a bug here must not freeze the call silently
                with self.lock:
                    self.phase = "error"
                    self.error = f"engine error: {type(exc).__name__}: {exc}"
                    self.log.write("error", error=self.error)
            time.sleep(0.02)

    def _log_tick(self) -> None:
        if self.phase != self._logged_phase:
            self.log.write("phase", phase=self.phase, previous=self._logged_phase, error=self.error)
            self._logged_phase = self.phase
        now = time.time()
        if now - self._last_stats_log >= 1.0:
            self._last_stats_log = now
            session = self.session
            job = self.job
            playback = None
            if self.phase == "speaking":
                playback = {
                    "buffered_s": round(len(self.out_buf) / BYTES_PER_SECOND, 2),
                    "pushed_s": round(self.dev_cursor / BYTES_PER_SECOND, 2),
                    "played_s": round(self.played_pos / BYTES_PER_SECOND, 2),
                    "to_stt_s": round(self.stt_cursor / BYTES_PER_SECOND, 2),
                    "paused": self.paused,
                    "tts_done": self.audio_ended,
                    "done": self.playback_done,
                    "llm_done": job.llm_done.is_set() if job else None,
                    "job_error": job.error if job else None,
                    "tts_error": job.utt.error if job and job.utt else None,
                    "agent_turn_done": self.agent_turn.done if self.agent_turn else None,
                    # What the barge-in listener heard, and which of it counted
                    # as the caller's own words - so a missed interruption can
                    # be read back from the log.
                    "listener_heard": " ".join(self.listener_tokens),
                    "listener_new_words": list(self.new_words),
                }
            self.log.write(
                "stats",
                phase=self.phase,
                mic=self.mic.stats(),
                playback=playback,
                stt_sent_s=round(session.audio_seconds, 1) if session else None,
                stt_partials=session.partials_total if session else None,
                stt_last_partial_age_s=self._last_partial_age(),
            )

    def _last_partial_age(self) -> float | None:
        session = self.session
        if session is None or not session.last_partial_at:
            return None
        return round(time.time() - session.last_partial_at, 1)

    def _is_agent_echo(self, result: TurnResult, heard_caller: bool) -> bool:
        """A caller turn that is really an agent's voice: every word is
        diarized as a speaker we have only ever seen on agent turns, and the
        mic never heard anything louder than a leaked copy of the agent's
        recent audio (`heard_over_echo`). Needs both, so a real caller who
        happens to be mis-diarized is never dropped."""
        ids = set(result.speaker_ids)
        text = _norm(result.final_text)
        keyterm_only = bool(text) and text in {_norm(k) for k in self.keyterms}
        if keyterm_only and not ids:
            # Nothing but the keyterm, on no diarized voice at all: the bias
            # filling in near-silence (turn 1 of call_20260923_051305 opened
            # the caller's first bubble with a "Qivora Sync" nobody said).
            return True
        agent_voice = bool(ids) and ids <= self.agent_speaker_ids and not ids & self.caller_speaker_ids
        if not agent_voice:
            return False
        if not heard_caller:
            return True
        # Even with the caller audible around it: a turn that is nothing but
        # the keyterm, in an agent's voice, is the keyterm bias filling in a
        # fragment (entries 17 and 18), not the caller.
        if keyterm_only:
            return True
        # ...or nothing but words from the agent's own last line ("Email
        # address?" right after "...your full email address?").
        tokens = _tokens(result.final_text)
        return bool(tokens) and all(_is_echo_word(t, _spoken_vocab(self.prev_agent_text)) for t in tokens)

    def _output_latency(self) -> float:
        """How long after push() the agent is actually audible: the local
        output stream reports its own; the browser path adds the WebRTC
        jitter buffer, taken as ~150 ms; headless playback has none."""
        latency = getattr(self.playback, "latency", None)
        if latency is not None:
            return float(latency)
        return 0.15 if self.settings.audio_path == "browser" and not isinstance(self.playback, NullPlayback) else 0.0

    def _echo_tail(self) -> float:
        return ECHO_TAIL_S + self._output_latency()

    def _echo_settled(self, now: float) -> tuple[bool, float]:
        """Whether the agent's voice has actually died away in the mic, so it
        is safe to start capturing the caller's next turn - and how much of
        the ring to keep when it is.

        A fixed tail after the estimated playback end wasn't enough: speakers
        plus FxSound-style processing delay the sound by an unknown amount,
        and the last syllable leaking in was transcribed as a caller turn
        ("Qivora Sync" via keyterm biasing, diarized as Maya - entry 17). So
        after the known latency, wait until the mic has been below the speech
        threshold for ECHO_QUIET_S. If it never goes quiet (the caller started
        talking straight away, or loud room noise), give up after
        ECHO_SETTLE_MAX_S and keep everything since the known playback end so
        a caller who jumped in isn't clipped; the phantom-turn filter covers
        any echo that rides along."""
        known_end = self.play_end + self._echo_tail()
        if now < known_end:
            return False, 0.0
        if self.mic.quiet_for_s() >= ECHO_QUIET_S:
            return True, 0.0
        waited = now - known_end
        if waited >= ECHO_SETTLE_MAX_S:
            return True, min(ECHO_SETTLE_MAX_S, now - (self.play_end + self._output_latency()))
        return False, 0.0

    def _playback_level(self, now: float) -> float:
        """Loudest agent audio playing over the last ECHO_LOOKBACK_S, i.e. the
        level a speaker-to-mic leak could be arriving at right now. For
        ECHO_WINDOW_S after the agent stops, the level of its last second
        stays in force: the leak can arrive later than any latency we know
        about (speakers + FxSound), and only sound louder than it should
        count as the caller."""
        if self.phase != "speaking" or self.interrupted:
            if now < self.last_play_end + self._output_latency() + ECHO_WINDOW_S:
                return self.last_play_rms
            return 0.0
        level = 0.0
        latency = self._output_latency()
        for start, end, rms in reversed(self.play_segments):
            if end + latency < now - ECHO_LOOKBACK_S:
                break
            if start + latency <= now:
                level = max(level, rms)
        return level

    def _step(self) -> None:
        session = self.session
        if self.phase in ("ended", "error", "connecting") or session is None:
            return
        if not session.alive:
            self.phase = "error"
            self.error = session.close_reason or "the transcription session closed"
            QLOG.warning("call   transcription session dropped: %s", self.error)
            if self.job is not None:
                self.job.cancel()
            self.playback.clear()
            self.mic.close_gate()
            return
        playing = self.phase == "speaking" and not self.paused and not self.interrupted and not self.playback_done
        self.mic.set_echo_reference(self._playback_level(time.time()), playing=playing)
        if self.phase == "waiting_mic":
            if self.mic.frames_recent(1.0):
                self._start_listening()
        elif self.phase == "listening":
            if self._swap_ready():
                self._swap_wire()
                return
            self._step_listening()
        elif self.phase in ("thinking", "speaking"):
            self._step_agent()

    def _reset_turn_state(self) -> None:
        if self.job is not None:
            self.job.cancel()
        self.live = None
        self.job = None
        self.spec_text = None
        self.spec_hit = False
        self.agent_turn = None
        self.audio_ended = False
        self.finish_sent = False
        self.interrupted = False
        self.agent_holding = False
        self.first_audio_wall = None
        self.interrupt_requested = False
        self.play_segments.clear()
        self._reset_playback()

    def _reset_playback(self) -> None:
        self.out_buf = bytearray()
        self.dev_cursor = 0
        self.stt_cursor = 0
        self.played_pos = 0.0
        self.last_tick = time.time()
        self.paused = False
        self.paused_at = 0.0
        self.pause_count = 0
        self.playback_done = False
        # Barge-in bookkeeping (see _step_barge_in).
        self.listen_from = 0
        self._listener_seen = 0
        self.new_words: list[str] = []
        self.listener_tokens: list[str] = []
        self.barge_seen_at = 0.0
        self.barge_reason = ""

    def _start_listening(self) -> None:
        self.spec_attempts = 0
        self.live = self.session.start_live_turn()
        self.live_started_wall = time.time()
        self.watchdog_fired = False
        # The oldest audio this turn will be sent (held audio + the gate's
        # preroll), and whether it opens with caller audio saved from earlier:
        # together they say whether the caller could be in it at all.
        held = self.mic.held_seconds() if self.mic.mode == "hold" else 0.0
        self._turn_audio_from = self.live_started_wall - held - 0.35
        self._turn_had_prefix = bool(self.pending_caller_audio)
        self.mic.open_gate(self.session, prefix=self.pending_caller_audio)
        self.pending_caller_audio = b""
        self.phase = "listening"

    # ------------------------------------------------------------------
    # caller turn
    # ------------------------------------------------------------------
    def _step_listening(self) -> None:
        live = self.live
        mic = self.mic
        text, _ = stitch_partials(live.result.events)

        if self.job is not None and mic.last_speech_at > self.job.started_wall + 0.05 and mic.speech_run_s() > 0.15:
            # The caller picked up again: that guess is stale, and the next
            # pause gets a fresh budget (a dictated number has several).
            self.job.cancel()
            self.job = None
            self.spec_text = None
            self.spec_attempts = 0
        elif (
            self.job is not None
            and self.spec_text is not None
            and text
            and not _same_utterance(self.spec_text, text)
            and self.spec_attempts < MAX_SPECULATIONS
        ):
            # The transcript caught up with the last words (interim text lags
            # the audio by up to a second): guess again from the fuller text.
            self.job.cancel()
            self.job = None
        if (
            self.settings.speculative
            and self.job is None
            and text
            and mic.heard_speech
            and mic.silence_s() >= SPECULATE_AFTER_SILENCE_S
            and self.spec_attempts < MAX_SPECULATIONS
        ):
            self.job = AgentJob(self.brain, self.tts, self.voices, pending_caller=text)
            self.spec_text = text
            self.spec_attempts += 1

        if (
            mic.heard_speech
            and not live.done
            and not self.watchdog_fired
            and mic.silence_s() > self.settings.smart_turn_timeout / 1000 + SILENCE_WATCHDOG_MARGIN_S
        ):
            self.session.end_live_turn()
            self.watchdog_fired = True

        if not live.done:
            return

        try:
            result: TurnResult = live.join(timeout=1.0)
        except Exception as exc:  # noqa: BLE001
            self.error = f"your turn failed: {exc}"
            self.phase = "error"
            return
        speech_end = mic.last_speech_at or time.time()
        heard_caller = mic.heard_over_echo
        mic.hold()
        self.live = None
        final = result.final_text.strip()
        # Text with no caller audio under it: the mic never rose above the
        # speech threshold anywhere in the audio this turn was sent (turns 10,
        # 15, 18 of call_20260923_031600: mic RMS 0.0000, a faint leak of the
        # agent's last words turned into "Qivora Sync" by the keyterm bias).
        no_caller_audio = not self._turn_had_prefix and mic.last_any_speech_at < self._turn_audio_from
        phantom = bool(final) and (no_caller_audio or self._is_agent_echo(result, heard_caller))
        if phantom:
            # Transcribe diarized this "caller" turn as one of the agents and
            # our own mic heard no caller speech during it: it is the agent's
            # voice picked up from the speakers, not the caller (entry 17).
            self.log.write("phantom_turn", text=final, speaker_ids=result.speaker_ids, no_caller_audio=no_caller_audio,
                           agent_ids=sorted(self.agent_speaker_ids), caller_ids=sorted(self.caller_speaker_ids))
            # Counted for the diagnostics panel, not shown in the conversation:
            # it said nothing about the call and cluttered the Live view.
            self.phantoms_ignored.append(final)
        if not final or phantom:
            # A cough, a click, or an echo: nothing to answer. Listen again,
            # keeping anything said since.
            if self.job is not None:
                self.job.cancel()
                self.job = None
            self._start_listening()
            return

        now = time.time()
        if all(t in FILLERS for t in _tokens(final)):
            # "Um..." on its own is the caller thinking, not a question: show
            # it (filler words are on) but let them finish before answering.
            self._record_turn("khalid", result, smart_turn_ms=int(max(0.0, now - speech_end) * 1000))
            if self.job is not None:
                self.job.cancel()
                self.job = None
            self._start_listening()
            return
        self.caller_speech_end = speech_end
        self.caller_turn_closed_at = now
        self._record_turn("khalid", result, smart_turn_ms=int(max(0.0, now - speech_end) * 1000))
        # Judged on the whole bubble, so a number split across a merge counts once.
        self._number_hold = 0 < _trailing_digits(self.record[-1]["heard_by_transcribe"]) < FULL_NUMBER_DIGITS

        if (
            self.job is not None
            and self.spec_text is not None
            and self.job.error is None
            and _same_utterance(self.spec_text, final)
        ):
            self.spec_hit = True
        else:
            if self.job is not None:
                self.job.cancel()
            self.spec_hit = False
            self.job = None
        self.brain.add_caller(final)
        if self.job is None:
            self.job = AgentJob(self.brain, self.tts, self.voices, pending_caller=None)
        self.thinking_since = now
        self.phase = "thinking"

    # ------------------------------------------------------------------
    # agent turn
    # ------------------------------------------------------------------
    def _step_agent(self) -> None:
        job = self.job
        mic = self.mic
        now = time.time()

        if (
            self.phase == "thinking"
            and self.continues < MAX_CONTINUES
            and mic.speech_run_s() > 0.2
            and mic.last_over_echo_at - self.caller_turn_closed_at >= CONTINUE_SPEECH_S
        ):
            # Smart Turn closed the turn but the caller wasn't done: drop the
            # reply before anyone hears it and keep listening (held audio first).
            job.cancel()
            self._reset_turn_state()
            self.continues += 1
            # Logged, not shown: the caller's next words join their bubble.
            self.log.write("system", text="kept listening - you were still talking", mic=self.mic.stats())
            self._start_listening()
            return

        if (
            self.phase == "thinking"
            and mic.silence_s() < CALLER_QUIET_S
            and now - self.thinking_since < MAX_WAIT_FOR_QUIET_S
        ):
            # Don't talk over the caller: Smart Turn can close a turn at a
            # tiny pause mid-sentence, and with the reply already drafted the
            # agent used to start 0.2 s later, on top of them. The reply keeps
            # generating; its audio just waits for a breath.
            return

        if self.phase == "thinking" and self._number_hold and now - self.caller_turn_closed_at < NUMBER_HOLD_S:
            # The turn closed mid-number ("my phone number is 0 2 1 5."):
            # Smart Turn scored that 0.87, as confident as a finished sentence.
            # Give the caller a moment to carry on before answering half a
            # number; if they do, the continue path above joins it up.
            return

        # Collect whatever TTS audio has arrived; playback is paced from here.
        while True:
            try:
                pcm = job.audio.get_nowait()
            except queue.Empty:
                break
            if pcm is None:
                self.audio_ended = True
                break
            if not self.interrupted:
                self.out_buf.extend(pcm)

        if self.agent_turn is None:
            if self.out_buf:
                self.agent_turn = self.session.start_stream_turn()
                self.first_audio_wall = now
                self.speaking_since = now
                self.last_tick = now
                self.play_end = now
                self.phase = "speaking"
                self.log.write("speaking", speaker=job.speaker)
                # Anything the caller said while the reply was being prepared
                # is theirs: keep it for the start of their next turn instead
                # of throwing it away with the gate.
                if mic.last_speech_at > self.caller_turn_closed_at + 0.1:
                    self.pending_caller_audio += mic.take_hold()
                # Measure (don't forward) the mic while the agent talks, and
                # let the barge-in listener hear it.
                mic.close_gate()
                mic.reset_runs()
                if self.listener is not None and self.listener.alive and self.settings.barge_in:
                    self.listen_from = len(self.listener.all_events)
                    self._listener_seen = self.listen_from
                    mic.set_tap(self.listener)
            elif self.audio_ended:
                # The reply failed before producing any audio.
                reason = job.error or "no audio came back"
                self._add_system(f"the reply failed ({reason}) - say that again?")
                self._reset_turn_state()
                self._start_listening()
                return
            else:
                return

        if self.phase != "speaking":
            return

        if not self.interrupted and not self.playback_done:
            # Not after playback is done: it would keep moving play_end to
            # "now" and the echo-settle wait would never end.
            self._advance_playback(now)

        if (
            not self.playback_done
            and not self.interrupted
            and self.audio_ended
            and self.dev_cursor >= len(self.out_buf)
            and self.played_pos >= len(self.out_buf)
        ):
            # Everything has been played: hand Transcribe the rest and close
            # the agent's utterance.
            if self.stt_cursor < len(self.out_buf):
                self.agent_turn.push(bytes(self.out_buf[self.stt_cursor :]))
                self.stt_cursor = len(self.out_buf)
            self.agent_turn.finish()
            self.finish_sent = True
            self.playback_done = True
            self.play_end = now
            self.last_play_end = now

        if not self.interrupted:
            if self.interrupt_requested:
                self.barge_reason = "interrupt requested"
                self._barge_in(now)
            elif self.settings.barge_in and not self.playback_done:
                self._step_barge_in(now)

        if not self.agent_holding and self.playback_done and not self.interrupted:
            settled, include_s = self._echo_settled(now)
            if settled:
                mic.hold(include_s=include_s)
                self.agent_holding = True

        if self.agent_holding and self.agent_turn.done:
            self._finish_agent_turn()

    # ------------------------------------------------------------------
    # paced playback
    # ------------------------------------------------------------------
    def _advance_playback(self, now: float) -> None:
        """Keep the device at most PLAYBACK_LEAD_S ahead of what has been
        played, and feed Transcribe only up to the played point.

        Pushing a whole reply into the playback track at once (as fast as TTS
        produced it) meant nothing could be paused or resumed - it was already
        queued in the browser - and Transcribe heard audio seconds before the
        caller did. Paced, a pause loses at most the lead, resume rewinds to
        the exact played point, and Transcribe hears what the caller heard."""
        dt = max(0.0, now - self.last_tick)
        self.last_tick = now
        if self.paused:
            return
        # The device plays in real time but can't play what it hasn't got
        # (an underrun is padded with silence and playback carries on from
        # the next push), so the played point never passes what was pushed.
        self.played_pos = min(self.played_pos + dt * BYTES_PER_SECOND, float(self.dev_cursor))
        total = len(self.out_buf)
        lead = PLAYBACK_LEAD_S * BYTES_PER_SECOND
        while self.dev_cursor < total and self.dev_cursor - self.played_pos < lead:
            n = min(total - self.dev_cursor, CHUNK_PUSH_BYTES) & ~1
            if n <= 0:
                break
            chunk = bytes(self.out_buf[self.dev_cursor : self.dev_cursor + n])
            start = now + (self.dev_cursor - self.played_pos) / BYTES_PER_SECOND
            # On a phone line the caller hears the agents narrowband too.
            self.playback.push(self._ear_line.process(chunk) if self._ear_line is not None else chunk)
            self.dev_cursor += n
            end = start + n / BYTES_PER_SECOND
            self.play_segments.append((start, end, _rms(chunk)))
            self.last_play_end = end
            self.last_play_rms = max((rms for s, e, rms in self.play_segments if e >= end - 1.0), default=0.0)
        # A small lead keeps the session's 100 ms paced sender from underrunning.
        target = int(min(self.dev_cursor, self.played_pos + STT_LEAD_S * BYTES_PER_SECOND)) & ~1
        if target > self.stt_cursor:
            self.agent_turn.push(bytes(self.out_buf[self.stt_cursor : target]))
            self.stt_cursor = target
        self.play_end = now + max(0.0, total - self.played_pos) / BYTES_PER_SECOND

    def _pause_playback(self, now: float) -> None:
        """Stop the agent mid-word (a possible interruption): drop what the
        device still had queued and rewind to the played point."""
        if self.paused:
            return
        self.paused = True
        self.paused_at = now
        self.pause_count += 1
        self.playback.clear()
        self.dev_cursor = max(0, int(self.played_pos) & ~1)
        self.play_segments = deque((s for s in self.play_segments if s[0] <= now), maxlen=400)
        self.last_play_end = now

    def _resume_playback(self, now: float) -> None:
        """False alarm: carry on from exactly where the agent was paused."""
        if not self.paused:
            return
        self.paused = False
        self.last_tick = now

    # ------------------------------------------------------------------
    # barge-in: pause on speech, confirm with words, resume on false alarms
    # ------------------------------------------------------------------
    def _agent_vocab(self) -> set[str]:
        """Words the caller could be hearing from the speakers right now."""
        said = self.job.said if self.job is not None else ""
        return _spoken_vocab(said) | _spoken_vocab(self.prev_agent_text)

    def _update_new_words(self, now: float) -> None:
        listener = self.listener
        if listener is None or len(listener.all_events) == self._listener_seen:
            return
        self._listener_seen = len(listener.all_events)
        events = [e for e in listener.all_events[self.listen_from :] if e.get("type") == "transcript.partial"]
        text = stitch_partials(events)[0]
        # The most recent words: when the caller talks over leaked echo, the
        # tail is theirs even if the start of the utterance was the agent.
        self.listener_tokens = _tokens(text)[-LISTENER_WINDOW_WORDS:]
        vocab = self._agent_vocab()
        self.new_words = [w for w in self.listener_tokens if w not in BACKCHANNEL and not _is_echo_word(w, vocab)]
        if self.new_words and not self.barge_seen_at:
            # Partials trail the audio by roughly a second.
            self.barge_seen_at = now - 1.0

    def _step_barge_in(self, now: float) -> None:
        """Voice-only interruption, the way LiveKit Agents does it:

        - pause the agent the moment the caller clearly starts talking over
          it (speech above the echo-aware bar for STRONG_CANDIDATE_S);
        - confirm with words from the listener transcription - new words, not
          the agent's own (echo) and not a backchannel like "mm-hmm" - and cut
          the agent: two new words, or one if it is already paused;
        - resume from the pause point if no such words come (a cough, a door,
          "yeah") within FALSE_PAUSE_S of the caller going quiet."""
        if now - self.speaking_since < BARGE_MIN_PLAYBACK_S:
            return
        mic = self.mic
        self._update_new_words(now)

        loud_s = max(mic.loud_run_s(), mic.loud_recent_s())
        if not self.paused and loud_s >= STRONG_CANDIDATE_S:
            self._pause_playback(now)
            self.barge_seen_at = self.barge_seen_at or (now - loud_s)
            self.log.write("barge_pause", loud_s=round(loud_s, 2), mic=mic.stats())

        if (self.listener is None or not self.listener.alive) and mic.loud_run_s() >= FALLBACK_CUT_S:
            # No listener transcription to confirm with: fall back to loudness.
            self.barge_reason = "sustained loud speech (listener unavailable)"
            self._barge_in(now)
            return

        n_new = len(self.new_words)
        mostly_new = n_new >= 0.5 * max(1, len(self.listener_tokens))
        # One word is enough (LiveKit's min_words=1) when it is an interrupt
        # word ("wait", "stop", "what"), when the agent is already paused, or
        # when the mic heard the caller over the echo just now. Otherwise two
        # new words that make up most of what the listener heard. The agent's
        # own words (`_spoken_vocab`) and backchannels never count.
        interrupt_word = any(w in INTERRUPT_WORDS for w in self.new_words)
        single_ok = self.paused or interrupt_word or now - mic.last_over_echo_at < RECENT_OVER_ECHO_S
        need = 1 if single_ok else 2
        # An interrupt word doesn't have to be most of what the listener heard:
        # it also transcribes the agent's faint echo ("hi thank you for calling
        # keep wait"), which the echo filter removes from the new words but not
        # from the count.
        confirmed = self.paused or interrupt_word or mostly_new
        if n_new >= need and confirmed:
            self.barge_reason = f"{n_new} new word{'s' if n_new != 1 else ''}: \u201c{' '.join(self.new_words[-6:])}\u201d"
            self._barge_in(now)
            return

        if self.paused:
            quiet_for = now - max(mic.last_over_echo_at, self.paused_at)
            if (now - self.paused_at >= FALSE_PAUSE_S and quiet_for >= FALSE_PAUSE_QUIET_S) or now - self.paused_at >= MAX_PAUSE_S:
                heard = " ".join(self.listener_tokens[-6:])
                self.log.write("barge_resume", paused_s=round(now - self.paused_at, 2), heard=heard, new_words=self.new_words)
                QLOG.info("barge  resumed %s (false alarm%s)", self.job.speaker if self.job else "agent",
                          f', heard only "{heard}"' if heard else "")
                self._resume_playback(now)
                self.barge_seen_at = 0.0

    def _barge_in(self, now: float) -> None:
        self.interrupt_requested = False
        self.interrupted = True
        self.playback.clear()
        self.job.cancel()
        self.agent_turn.cut()
        self.audio_ended = True
        self.finish_sent = True
        self.play_end = now
        self.last_play_end = now
        heard_so_far = stitch_partials(self.agent_turn.result.events)[0]
        self.log.write("barge_cut", reason=self.barge_reason, heard=heard_so_far, paused=self.paused)
        QLOG.info("barge  cut %s after \u201c%s\u201d (%s)", (self.job.speaker if self.job else "agent").capitalize(),
                  heard_so_far[-60:] or "...", self.barge_reason or "voice")
        # Keep the caller's speech from where it started (confirmation by words
        # can take a second or two after the first syllable, and a soft first
        # word can sit below the pause bar).
        started = self.barge_seen_at or (now - BARGE_INCLUDE_S)
        phrase = self.mic.phrase_started_at()
        if phrase:
            started = min(started, max(phrase, started - SOFT_START_MAX_S))
        self.mic.hold(include_s=min(PREROLL_MAX_S, max(BARGE_INCLUDE_S, now - started + 0.6)))
        self.mic.set_tap(None)
        self.agent_holding = True

    def _finish_agent_turn(self) -> None:
        job = self.job
        try:
            result = self.agent_turn.join(timeout=1.0)
        except Exception:  # noqa: BLE001
            result = self.agent_turn.result
        speaker = job.speaker or (job.reply.speaker if job.reply else "maya")
        heard = result.final_text.strip() or stitch_partials(result.events)[0]
        self.mic.set_tap(None)
        if self.listener is not None and self.listener.alive:
            # Close whatever the listener was in the middle of, so the next
            # agent turn starts from a clean utterance.
            self.listener.end_live_turn()
        self.prev_agent_text = job.said
        response_ms = None
        if self.first_audio_wall is not None and self.caller_speech_end:
            response_ms = int((self.first_audio_wall - self.caller_speech_end) * 1000)
            self.last_latency_ms = response_ms
        if self.interrupted:
            self.brain.add_agent_partial(speaker, heard or job.said)
        elif job.reply is not None:
            self.brain.add_agent(job.reply)
        self._record_turn(
            speaker,
            result,
            generated=job.said if self.interrupted else (job.reply.text if job.reply else job.said),
            interrupted=self.interrupted,
            response_ms=response_ms,
            job=job,
        )
        session = self.session
        while session is not None and self._unowned_seen < len(session.unowned_texts):
            text = session.unowned_texts[self._unowned_seen]
            self._unowned_seen += 1
            if text:
                # Words the caller added after Smart Turn closed their turn; kept
                # out of the agent's transcript, shown and passed on instead.
                self._add_system(f"also heard from you, after your turn closed: \u201c{text}\u201d")
                self.brain.add_caller(text)
        end_call = bool(job.reply and job.reply.end_call and not self.interrupted)
        self._reset_turn_state()
        self.continues = 0
        if end_call:
            self.phase = "ending"
            threading.Thread(target=self.end, args=("call ended by the agent",), daemon=True).start()
            return
        self._start_listening()

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------
    def _add_system(self, text: str) -> None:
        self.turns.append({"speaker": "system", "text": text})
        self.version += 1
        self.log.write("system", text=text, mic=self.mic.stats())

    def _record_turn(
        self,
        speaker: str,
        result: TurnResult,
        *,
        generated: str | None = None,
        interrupted: bool = False,
        smart_turn_ms: int | None = None,
        response_ms: int | None = None,
        job: AgentJob | None = None,
    ) -> None:
        self.turn_no += 1
        heard = result.final_text.strip() or stitch_partials(result.events)[0] or "(nothing transcribed)"
        values = [w["confidence"] for w in result.words if isinstance(w.get("confidence"), (int, float)) and w["confidence"] > 0]
        avg_conf = round(sum(values) / len(values), 3) if values else None
        human = speaker == "khalid"
        eotc = result.end_of_turn_confidence if (human and result.got_utterance_final) else None
        ids = {i for i in result.speaker_ids if i is not None}
        if human:
            # Never an id already heard on an agent turn: one echo turn that
            # slipped through used to make the agent's id "the caller's", and
            # every later echo then passed the filter (turns 15 and 18).
            self.caller_speaker_ids |= ids - self.agent_speaker_ids
        else:
            # An agent turn can carry a caller id when a word bled in; only ids
            # never heard from the caller count as agent voices.
            self.agent_speaker_ids |= ids - self.caller_speaker_ids

        # What Transcribe actually received for this turn (the LiveOps-style
        # source label): the 8 kHz mu-law phone leg or 16 kHz PCM.
        source = "phone" if self.session is not None and self.session.wire == "mulaw8k" else "mic"
        meta = [f"turn {self.turn_no}"]
        if ids:
            meta.append("diarized as speaker " + "+".join(str(i) for i in sorted(ids)))
        if eotc is not None:
            meta.append(f"end-of-turn confidence {eotc:.2f}")
        if avg_conf is not None:
            meta.append(f"word confidence {avg_conf:.2f}")
        if smart_turn_ms is not None:
            meta.append(f"Smart Turn closed it {smart_turn_ms / 1000:.1f}s after you stopped")
        stream = job.stream if job else None
        if not human:
            meta.append(f"written by {stream.model if stream else self.brain.model}, voiced by Grok TTS")
        previous = next((t for t in reversed(self.turns) if t["speaker"] != "system"), None)
        if human and previous is not None and previous["speaker"] == "khalid" and self.record and \
                self.record[-1]["speaker"] == "khalid":
            # The caller carried on before any agent reply (Smart Turn closed
            # at a pause, the "kept listening" path, a filler-only turn, or a
            # number split on the phone line): one bubble, not three.
            self.turn_no -= 1
            previous["text"] = f"{previous['text']} {heard}".strip()
            previous["eotc"] = eotc
            previous["speaker_ids"] = sorted(set(previous.get("speaker_ids") or []) | ids)
            previous["meta"] = " \u00b7 ".join([f"turn {self.turn_no}", *meta[1:]])
            rec = self.record[-1]
            rec["heard_by_transcribe"] = f"{rec['heard_by_transcribe']} {heard}".strip()
            rec["end_of_turn_confidence"] = eotc
            rec["diarized_speaker_ids"] = sorted(set(rec["diarized_speaker_ids"]) | set(result.speaker_ids))
            rec["word_count"] += len(result.words)
            rec["smart_turn_ms"] = smart_turn_ms
            rec["merged_parts"] = rec.get("merged_parts", 1) + 1
            self.version += 1
            self.log.write("turn_merged", **rec)
            return
        self.turns.append({
            "speaker": speaker,
            "text": heard,
            "said": generated,
            "meta": " \u00b7 ".join(meta),
            "latency_ms": response_ms,
            "interrupted": interrupted,
            "source": source,
            "speaker_ids": sorted(ids),
            "eotc": eotc,
            "language": job.language if job else ("ar" if any("\u0600" <= ch <= "\u06ff" for ch in heard) else "en"),
        })
        self.record.append({
            "turn": self.turn_no,
            "speaker": speaker,
            "heard_by_transcribe": heard,
            "generated_text": generated,
            "interrupted": interrupted,
            "diarized_speaker_ids": result.speaker_ids,
            "end_of_turn_confidence": eotc,
            "got_utterance_final": result.got_utterance_final,
            "word_count": len(result.words),
            "avg_word_confidence": avg_conf,
            "smart_turn_ms": smart_turn_ms,
            "response_latency_ms": response_ms,
            "llm_first_text_ms": stream.ttft_ms if stream else None,
            "llm_total_ms": stream.total_ms if stream else None,
            "tts_first_audio_ms": job.utt.tts_first_audio_ms if job and job.utt else None,
            "speculative_hit": self.spec_hit if job else None,
            "speculative_attempts": self.spec_attempts if job else None,
            "model": stream.model if stream else None,
            "reply_language": job.language if job else None,
            "source": source,
            "session": self.session_no,
            "service_tier": stream.service_tier if stream else None,
            "tts_rest_fallback": job.used_rest_fallback if job else None,
            "barge_pauses": self.pause_count if job else None,
            "barge_reason": self.barge_reason if interrupted else None,
        })
        self.version += 1
        self.log.write("turn", **self.record[-1])

    # ------------------------------------------------------------------
    # UI view
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            mic = self.mic
            live_done, live_tail, live_tag = ("", "", "interim")
            if self.live is not None:
                live_done, live_tail, live_tag = stitch_parts(self.live.result.events)
            live_text = " ".join(p for p in (live_done, live_tail) if p)
            heard_done, heard_tail, heard_tag = ("", "", "interim")
            if self.agent_turn is not None:
                heard_done, heard_tail, heard_tag = stitch_parts(self.agent_turn.result.events)
            heard_text = " ".join(p for p in (heard_done, heard_tail) if p)
            play_level = 0.0
            if self.phase == "speaking" and not self.interrupted:
                for start, end, rms in reversed(self.play_segments):
                    if start <= now < end:
                        play_level = min(1.0, rms * 6.0)
                        break
            job = self.job
            session = self.session
            mic_stats = mic.stats()
            partial_age = self._last_partial_age()
            listening_for = now - self.live_started_wall if self.phase == "listening" else 0.0
            speech_ago = mic_stats["last_speech_ago_s"]
            # Frames arrive but nothing has risen above the room since this turn
            # opened: most likely the wrong input device (a virtual mic, a muted
            # headset) rather than the caller being quiet on purpose.
            only_silence = (
                self.phase == "listening"
                and mic_stats["frames_per_s"] > 0
                and listening_for > 6.0
                and not live_text
                and (speech_ago is None or speech_ago > listening_for)
            )
            # The mic hears speech but Transcribe has returned nothing for a
            # while: audio isn't reaching the model in a usable form.
            since_text = min(listening_for, partial_age if partial_age is not None else listening_for)
            stt_silent = (
                self.phase == "listening"
                and speech_ago is not None
                and speech_ago < 1.0
                and since_text > 8.0
            )
            return {
                "mic_stats": mic_stats,
                "stt_partials": session.partials_total if session else 0,
                "stt_last_partial_age_s": partial_age,
                "only_silence": only_silence,
                "stt_silent": stt_silent,
                "log_path": str(self.log.path) if self.log.path else None,
                "audio_path": self.settings.audio_path,
                "phase": self.phase,
                "error": self.error,
                "speaker": job.speaker if job is not None and self.phase in ("thinking", "speaking") else None,
                "said": job.said if job is not None and self.phase == "speaking" else "",
                "heard": heard_text,
                "heard_done": heard_done,
                "heard_tail": heard_tail,
                "heard_tag": heard_tag,
                "live_text": live_text,
                "live_done": live_done,
                "live_tail": live_tail,
                "live_tag": live_tag,
                "mic_level": mic.level,
                "mic_ok": mic.frames_recent(2.0),
                "mic_error": mic.first_error,
                "holding": mic.mode == "hold",
                "play_level": play_level,
                "thinking_s": now - self.thinking_since if self.phase == "thinking" else 0.0,
                "speculating": job is not None and self.phase == "listening",
                "spec_hit": self.spec_hit,
                "interrupted": self.interrupted,
                "paused": self.paused and self.phase == "speaking",
                "listener_ok": bool(self.listener is not None and self.listener.alive),
                "phantoms_ignored": list(self.phantoms_ignored),
                "phone_line": self.mic.line.stats() if getattr(self.mic, "line", None) is not None else None,
                "smart_turn": self.settings.smart_turn,
                "phone_on": self.settings.phone_line,
                "phone_switch_to": self._pending_swap["on"] if self._pending_swap is not None else None,
                "last_eotc": next((r["end_of_turn_confidence"] for r in reversed(self.record)
                                   if r["speaker"] == "khalid" and r["end_of_turn_confidence"]), None),
                "turns": list(self.turns),
                "record": list(self.record),
                "version": self.version,
                "turn_no": self.turn_no,
                "audio_seconds": session.audio_seconds if session else 0.0,
                "last_latency_ms": self.last_latency_ms,
                "model": self.brain.model,
            }
