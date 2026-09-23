"""
Persistent streaming STT session for the Streamlit live demo.

Streamlit reruns the whole script on every interaction, so the WebSocket
cannot live inside a normal function call: it would open and close on
every button click. Instead this module runs one background thread with
its own asyncio event loop for the lifetime of the browser session, and
Streamlit hands it audio and reads back events through thread-safe calls
(`asyncio.run_coroutine_threadsafe`). This is the same
`grok-voice-transcribe-2.0` streaming protocol exercised for real in
scripts/stt_stream_client.py (see results/08_streaming_smartturn_events.json) -
this module just keeps the same session open across many short turns
instead of one long unattended stream.

Design choice driven by real experiment 08: left to its own devices,
Smart Turn on a fast multi-speaker exchange with short gaps merged four
ground-truth turns into a single utterance before `speech_final` fired
(see results/experiment_log.md, section 08). Because this app already
knows exactly when an agent turn ends (its TTS audio runs out, or the
caller barges in), it sends the documented `{"type": "finalize"}` client
message at the end of every agent turn instead of waiting on Smart Turn
to guess. That gives one clean utterance-final per turn, which is both a
better live demo and a more honest use of a feature meant for
push-to-talk / known-turn-boundary scenarios.

Single-reader design (see results/experiment_log.md for the write-up): an
earlier version of this module opened a fresh `receiver()` coroutine per
turn to call `self._ws.recv()`. Streamlit aborts a running script on a
new widget interaction ("rerun") but does not cancel background asyncio
tasks that script already started, so a turn interrupted mid-flight
could leave an orphaned `receiver()` still calling `recv()` in the
background; a retried turn's own fresh `receiver()` then collided with
it. The `websockets` library forbids two coroutines from calling
`.recv()` on the same connection at once and raises a concurrency error
the moment that happens - documented, along with the "one reader,
dispatch to consumers" fix, at
https://websockets.readthedocs.io/en/stable/topics/design.html.

The fix mirrors a sibling project's `SttSession` / `VoiceAgentSession`
(D:\\2026\\Grok LiveOps\\backend\\app\\xai\\stt.py and realtime.py, same
`websockets` library, same streaming-STT-per-turn shape): one `_read_loop`
task, started once in `connect()`, is the only code that ever calls
`.recv()` for the life of the connection. Turns only ever `send()` and
then wait on an `asyncio.Event` the reader sets, serialized by a lock so
two turns can never both be "active" at once - which also means a stray
duplicate `start_turn()` call now queues safely instead of crashing.

Continuous audio (experiment_log.md entries 13-15): the server's VAD,
Smart Turn and `smart_turn_timeout` can only measure silence from frames
they actually receive, and a stream that goes quiet can be closed for
inactivity. So a `_pump_loop` fills every gap - the agent thinking, the
mic not started yet - with real-time-paced digital silence, as long as
the UI is still polling (`touch()`); an abandoned browser tab stops
sending (and costing) within HEARTBEAT_TIMEOUT_S.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import websockets

from scripts.phone_line import PhoneLine

WS_BASE = "wss://api.x.ai/v1/stt"
BYTES_PER_SECOND = 16000 * 2
CHUNK_BYTES = BYTES_PER_SECOND // 10  # 100 ms, the chunk size the STT docs recommend

# How long the reader waits for speech_final after a turn's finalize is
# sent before giving up and handing back whatever text arrived anyway.
#
# Real-API finding (see results/experiment_log.md): at the original 8.0s,
# a turn whose speech_final took longer than that to arrive would time
# out, the next turn would start and claim `_active`, and the first
# turn's own speech_final - still in flight - would land AFTER that and
# get misattributed to the turn that was active by the time it arrived.
# Raising this reduces how often that race gets a chance to happen at
# all; the `_finalize_sent` guard on the completion check below is what
# actually closes the misattribution itself; a stray late event can no
# longer complete a turn that has not yet asked to finalize.
FINALIZE_GRACE_S = 12.0
# How often the reader re-checks its own state (whether finalize was just
# sent, whether the call is closing) when no server message is arriving.
# Bounding every recv() wait to this, rather than blocking forever, is what
# lets the grace deadline above arm promptly even through total silence.
POLL_S = 0.5
# Digital silence the server must have received ahead of every agent turn.
# The server can only close an utterance on silence it actually receives;
# without this gap, a caller utterance still open when the mic gate shut ran
# straight into the agent's audio and came back merged into the agent's
# turn (a real bleed seen in live testing, results/experiment_log.md entry
# 14). The pump usually covers it already; only the shortfall is sent.
SILENCE_PREFIX_S = 0.6
SILENCE_CHUNK = bytes(CHUNK_BYTES)  # 100 ms of PCM16 silence at 16 kHz
# Second chance for a live turn that timed out without speech_final: stream
# this much silence, finalize again, and wait this long before falling back.
LIVE_RETRY_SILENCE_S = 1.0
LIVE_RETRY_WAIT_S = 5.0
# The pump only fills a gap once nothing real has been sent for this long,
# so it never interleaves with a live mic feeding 100 ms chunks.
PUMP_IDLE_S = 0.2
# Stop pumping silence when the UI has not polled for this long.
HEARTBEAT_TIMEOUT_S = 30.0


def _api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("XAI_API_KEY not set in environment")
    return key


@dataclass
class TurnResult:
    """Everything the UI needs to render one turn's real transcription."""
    events: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    words: list[dict[str, Any]] = field(default_factory=list)
    speaker_ids: list[int] = field(default_factory=list)
    end_of_turn_confidence: float | None = None
    got_utterance_final: bool = False


def stitch_parts(events: list[dict[str, Any]]) -> tuple[str, str, str]:
    """(settled text, still-changing tail, tag): the same stitching as
    stitch_partials, split so the UI can dim only the interim tail."""
    text, tag = stitch_partials(events)
    if tag != "interim":
        return text, "", tag
    current = ""
    for evt in reversed(events):
        if evt.get("type") == "transcript.partial":
            if not evt.get("is_final") and not evt.get("speech_final"):
                current = _event_text(evt)
            break
    if current and text.endswith(current):
        return text[: len(text) - len(current)].rstrip(), current, tag
    return "", text, tag


def _event_text(evt: dict[str, Any]) -> str:
    """An event's text without the leading punctuation the server sometimes
    carries over from the previous utterance (", okay, uh" or "? Sorry",
    call_20260923_051305)."""
    return (evt.get("text") or "").strip().lstrip(",.;:!?،؟ ").strip()


def stitch_partials(events: list[dict[str, Any]]) -> tuple[str, str]:
    """The text to show for a turn still in progress, and its state tag
    ("interim", "locked" or "final").

    Interim and chunk-final text only covers the current ~3 s chunk: after
    each chunk final the next interim starts over from the new chunk
    (results/08_streaming_smartturn_events.json, t=31.3s -> 31.9s). Showing
    only the latest event makes everything said earlier vanish, so locked
    chunks are kept and the current interim is appended to them; the
    utterance final is already stitched by the server and replaces both.
    """
    # A turn can span several server utterances (an agent line paused for a
    # possible interruption and then resumed, entry 18): each utterance final
    # is kept, not replaced by the next one.
    done: list[str] = []
    locked: list[str] = []
    current = ""
    tag = "interim"
    for evt in events:
        if evt.get("type") != "transcript.partial":
            continue
        text = _event_text(evt)
        if evt.get("speech_final"):
            utterance = text or " ".join([*locked, current]).strip()
            if utterance:
                done.append(utterance)
            locked = []
            current = ""
            tag = "final"
        elif evt.get("is_final"):
            if text:
                if locked and text.startswith(locked[-1]):
                    locked[-1] = text
                elif not locked or locked[-1] != text:
                    locked.append(text)
            current = ""
            tag = "locked"
        else:
            current = text
            tag = "interim"
    if current and locked and current.startswith(locked[-1]):
        locked = locked[:-1]
    if (locked or current) and tag == "final":
        tag = "interim" if current else "locked"
    return " ".join([*done, *locked, current] if current else [*done, *locked]).strip(), tag


def _fallback_text(result: TurnResult) -> str:
    """Best text for a turn that never got speech_final."""
    return stitch_partials(result.events)[0]


@dataclass
class TurnHandle:
    """A turn in flight, for a caller that wants to watch it happen instead
    of blocking until it is done.

    `result` is the exact same object the session's single persistent
    reader keeps appending real events to as they arrive over the socket,
    so a Streamlit script can poll `handle.result.events` in a plain loop
    and render genuinely live interim/locked/final text - not a replay of
    events that already finished arriving, which is all a fully blocking
    call can offer after the fact.
    """
    result: TurnResult
    future: Any  # concurrent.futures.Future, from run_coroutine_threadsafe

    @property
    def done(self) -> bool:
        return self.future.done()

    def join(self, timeout: float = 60.0) -> TurnResult:
        """Wait for the turn to finish and re-raise any error it hit."""
        self.future.result(timeout=timeout)
        return self.result


class _StreamFeed:
    """Thread-safe PCM buffer between a producer (TTS audio arriving on
    another thread) and the session's paced sender."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._finished = False
        self._cut = False

    def push(self, pcm: bytes) -> None:
        with self._lock:
            if not self._finished and not self._cut:
                self._buf.extend(pcm)

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def cut(self) -> None:
        with self._lock:
            self._cut = True
            self._buf = bytearray()

    def take(self, n: int) -> tuple[bytes | None, bool]:
        """(chunk, done). An empty chunk means "underrun, send silence"."""
        with self._lock:
            if self._cut:
                return None, True
            if len(self._buf) >= n:
                chunk = bytes(self._buf[:n])
                del self._buf[:n]
                return chunk, False
            if self._finished:
                if self._buf:
                    chunk = bytes(self._buf)
                    self._buf = bytearray()
                    return chunk, False
                return None, True
            return b"", False


@dataclass
class StreamTurnHandle(TurnHandle):
    """An agent turn whose audio is pushed as it is synthesized, and sent to
    the socket at real-time pace - in step with what the caller hears."""
    feed: _StreamFeed = field(default_factory=_StreamFeed)

    def push(self, pcm16_bytes: bytes) -> None:
        self.feed.push(pcm16_bytes)

    def finish(self) -> None:
        """No more audio is coming; finalize once what is queued is sent."""
        self.feed.finish()

    def cut(self) -> None:
        """Barge-in: drop whatever is still queued and finalize now."""
        self.feed.cut()


class LiveCallSession:
    """One shared grok-voice-transcribe-2.0 streaming session for the whole call."""

    def __init__(
        self,
        *,
        model: str = "grok-voice-transcribe-2.0",
        sample_rate: int = 16000,
        diarize: bool = True,
        language: str | None = "en",
        format_: bool = False,
        keyterms: list[str] | None = None,
        smart_turn: float | None = 0.7,
        smart_turn_timeout: int | None = 3000,
        vad_threshold: float | None = None,
        endpointing: int | None = None,
        filler_words: bool = False,
        wire: str = "pcm16k",
        keepalive: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        # "mulaw8k": a real phone leg, as a telephony provider would stream it
        # (Grok LiveOps' TelephonyTap). Everything in this class still works
        # in 16 kHz PCM16; each chunk is encoded to 8 kHz G.711 mu-law by one
        # stateful encoder at the moment it goes on the socket.
        self.wire = wire
        self._encoder = PhoneLine(dropout=0.0) if wire == "mulaw8k" else None
        params: list[tuple[str, str]] = [
            ("model", model),
            ("sample_rate", "8000" if self._encoder else str(sample_rate)),
            ("encoding", "mulaw" if self._encoder else "pcm"),
            ("interim_results", "true"),
            ("diarize", "true" if diarize else "false"),
        ]
        # Off by default in the API: "um", "uh", "er" are stripped from the
        # transcript unless this is set.
        if filler_words:
            params.append(("filler_words", "true"))
        if language:
            params.append(("language", language))
        if format_:
            params.append(("format", "true"))
        if smart_turn is not None:
            params.append(("smart_turn", str(smart_turn)))
        if smart_turn_timeout is not None:
            params.append(("smart_turn_timeout", str(smart_turn_timeout)))
        if vad_threshold is not None:
            params.append(("vad_threshold", str(vad_threshold)))
        if endpointing is not None:
            params.append(("endpointing", str(endpointing)))
        for term in keyterms or []:
            params.append(("keyterm", term))
        self.url = f"{WS_BASE}?{urlencode(params)}"
        self.keepalive = keepalive

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        _OPEN_LOOPS.add(self)
        self._ws: websockets.ClientConnection | None = None
        self.all_events: list[dict[str, Any]] = []
        self.connected = False
        self.close_reason: str | None = None
        self.last_error: str | None = None
        self.bytes_sent = 0
        self.partials_total = 0
        self.last_partial_at = 0.0
        self._unowned_text = ""
        self.unowned_texts: list[str] = []

        # Single-reader state (see module docstring). `_active`/`_finalize_sent`
        # are written by the turn coroutines and read by `_read_loop`; all of
        # them run on the same event loop thread, so plain attributes are
        # enough - the `_turn_lock` below is what actually prevents two turns
        # from being "active" at the same time, not these reads/writes.
        self._reader_task: asyncio.Task | None = None
        self._pump_task: asyncio.Task | None = None
        self._mic_writer_task: asyncio.Task | None = None
        self._turn_lock = asyncio.Lock()
        self._active: TurnResult | None = None
        self._finalize_sent: bool = False
        # True only for a live (continuously-fed microphone) turn that has
        # not asked to finalize yet - see start_live_turn(). Lets `_read_loop`
        # accept a genuine Smart Turn `speech_final` as real completion
        # without requiring a client finalize, while still gating out a
        # stray late event from a *previous*, already-finalized turn (the
        # bug FINALIZE_GRACE_S's comment describes) via `_finalize_sent`.
        self._live_turn_active: bool = False
        # While an agent turn is sending its own paced audio, the pump stays out.
        self._streaming_turn: bool = False
        self._last_send = 0.0
        self._last_real_send = 0.0
        self._last_touch = time.monotonic()
        # Mic chunks go through one queue and one writer so they reach the
        # socket in exactly the order they were captured (a barge-in flushes
        # a held buffer first, then live chunks follow).
        self._mic_queue: asyncio.Queue[bytes] | None = None
        self._turn_done = asyncio.Event()
        self._call_done = asyncio.Event()
        self._call_done_event: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro, timeout: float = 30.0):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    @property
    def alive(self) -> bool:
        return self.connected and self._ws is not None and self.close_reason is None

    @property
    def audio_seconds(self) -> float:
        """Audio actually streamed to the server so far - what streaming STT bills."""
        return self.bytes_sent / BYTES_PER_SECOND

    def touch(self) -> None:
        """Heartbeat from the UI; keeps the silence pump running."""
        self._last_touch = time.monotonic()

    def _mark_dead(self, reason: str) -> None:
        if self.close_reason is None:
            self.close_reason = reason
        self._turn_done.set()
        self._call_done.set()

    async def _send(self, data: bytes | str, *, real: bool = True) -> None:
        if self._ws is None or self.close_reason is not None:
            raise websockets.exceptions.ConnectionClosedError(None, None)
        payload = data
        if self._encoder is not None and isinstance(data, (bytes, bytearray)):
            payload = self._encoder.to_mulaw_8k(bytes(data))
        try:
            if payload:
                await self._ws.send(payload)
        except websockets.exceptions.ConnectionClosed as exc:
            self._mark_dead(self.last_error or f"connection closed ({exc})")
            raise
        now = time.monotonic()
        self._last_send = now
        if isinstance(data, (bytes, bytearray)):
            self.bytes_sent += len(data)
            if real:
                self._last_real_send = now

    async def _send_finalize(self) -> None:
        # Sent in both casings on purpose: xAI's own docs spell this message
        # both "Finalize" and "finalize", and a sibling project's STT client
        # (D:\2026\Grok LiveOps\backend\app\xai\stt.py) sends both for exactly
        # that reason, on the assumption the server ignores whichever one it
        # does not recognise.
        await self._send(json.dumps({"type": "Finalize"}))
        await self._send(json.dumps({"type": "finalize"}))

    async def _connect(self) -> None:
        headers = {"Authorization": f"Bearer {_api_key()}"}
        self._ws = await websockets.connect(self.url, additional_headers=headers, max_size=None)
        msg = json.loads(await self._ws.recv())
        if msg.get("type") != "transcript.created":
            raise RuntimeError(f"unexpected first STT event: {msg}")
        self.all_events.append(msg)
        self._mic_queue = asyncio.Queue()
        # The one and only recv() caller for the rest of this connection's
        # life (see module docstring).
        self._reader_task = asyncio.create_task(self._read_loop())
        self._mic_writer_task = asyncio.create_task(self._mic_writer())
        if self.keepalive:
            self._pump_task = asyncio.create_task(self._pump_loop())

    def connect(self) -> None:
        self._submit(self._connect())
        self.connected = True

    async def _pump_loop(self) -> None:
        try:
            while self._ws is not None and self.close_reason is None:
                await asyncio.sleep(0.05)
                now = time.monotonic()
                if self._streaming_turn or now - self._last_touch > HEARTBEAT_TIMEOUT_S:
                    continue
                if now - self._last_real_send >= PUMP_IDLE_S and now - self._last_send >= 0.1:
                    await self._send(SILENCE_CHUNK, real=False)
        except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
            pass

    async def _mic_writer(self) -> None:
        assert self._mic_queue is not None
        try:
            while True:
                chunk = await self._mic_queue.get()
                if self.close_reason is not None:
                    continue
                await self._send(chunk)
        except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
            pass

    async def _read_loop(self) -> None:
        """The only coroutine that ever calls `self._ws.recv()`.

        Dispatches every event to whichever turn is currently active
        (`self._active`), applying the same speech_final / grace-period
        logic a per-turn `receiver()` used to apply locally, and resolves
        `close()`'s wait for `transcript.done` too - so nothing else in
        this class ever needs its own recv loop.
        """
        grace_deadline: float | None = None
        deadline_owner: TurnResult | None = None
        try:
            while True:
                # A deadline belongs to the turn that armed it. Without this,
                # a deadline left over from one turn could fire during the
                # next and close it early.
                if grace_deadline is not None and self._active is not deadline_owner:
                    grace_deadline = None
                if self._active is not None and self._finalize_sent and grace_deadline is None:
                    grace_deadline = time.time() + FINALIZE_GRACE_S
                    deadline_owner = self._active

                if grace_deadline is None:
                    timeout = POLL_S
                else:
                    timeout = min(POLL_S, max(0.0, grace_deadline - time.time()))

                try:
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if grace_deadline is not None and time.time() >= grace_deadline:
                        # Finalize was sent and speech_final never arrived
                        # within the grace window - hand back the best text
                        # that did arrive rather than an empty turn.
                        if self._active is not None:
                            if not self._active.got_utterance_final:
                                self._active.final_text = _fallback_text(self._active)
                            self._turn_done.set()
                        grace_deadline = None
                    continue

                evt = json.loads(raw)
                evt["_t"] = round(time.time(), 3)
                self.all_events.append(evt)

                if evt.get("type") == "error":
                    # Most STT errors close the connection right after (docs:
                    # "Only client message parse errors keep the connection
                    # open"); keep the message so the UI can say why.
                    self.last_error = str(evt.get("message") or evt)
                    if self._active is not None:
                        self._active.events.append(evt)
                    continue

                if evt.get("type") == "transcript.partial" and evt.get("text"):
                    self.partials_total += 1
                    self.last_partial_at = time.time()

                if evt.get("type") == "transcript.done":
                    self._call_done_event = evt
                    self._call_done.set()
                    continue

                result = self._active
                if result is None:
                    if evt.get("type") == "transcript.partial" and evt.get("text") and not evt.get("speech_final"):
                        # The caller kept talking after their turn closed and the
                        # server opened an utterance nobody owns (see
                        # _flush_unowned_utterance).
                        self._unowned_text = evt.get("text", "")
                    continue
                result.events.append(evt)
                # `self._finalize_sent` gates this: without it, a stray event
                # that actually belongs to the turn just before this one (its
                # own speech_final, delayed past that turn's grace window)
                # could complete THIS turn before it has even finished
                # sending its own audio - the exact misattribution found
                # during real testing (see FINALIZE_GRACE_S's comment above).
                if (
                    evt.get("type") == "transcript.partial"
                    and evt.get("is_final")
                    and evt.get("speech_final")
                    and (self._finalize_sent or self._live_turn_active)
                ):
                    # Every utterance final in this turn, not just the last one
                    # (a paused-and-resumed agent line is two utterances).
                    result.final_text = stitch_partials(result.events)[0]
                    result.words = [
                        w for e in result.events
                        if e.get("type") == "transcript.partial" and e.get("speech_final")
                        for w in (e.get("words") or [])
                    ]
                    result.speaker_ids = sorted({w.get("speaker") for w in result.words if w.get("speaker") is not None})
                    result.end_of_turn_confidence = evt.get("end_of_turn_confidence")
                    result.got_utterance_final = True
                    self._turn_done.set()
                    grace_deadline = None
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed as exc:
            # Wake up anyone waiting rather than leaving them hanging until
            # their own outer timeout, and let the task end quietly - the
            # connection is already gone, there is nothing left to read.
            if self._active is not None and not self._active.got_utterance_final:
                self._active.final_text = _fallback_text(self._active)
            self._mark_dead(self.last_error or f"connection closed ({exc.rcvd.code if exc.rcvd else 'no close frame'})")

    # ------------------------------------------------------------------
    # buffered and streamed agent turns
    # ------------------------------------------------------------------
    async def _flush_unowned_utterance(self) -> None:
        """Close an utterance the caller started after their turn closed.

        Smart Turn can close a turn at a mid-sentence pause while the caller
        keeps going; those words reach the server with no turn active, and
        without this they came back merged into the start of the agent's
        own transcript ("It keep... Hello, thank you for calling", entry 16).
        Finalizing first gives them their own utterance, recorded in
        `unowned_texts` instead of being attributed to the agent."""
        if not self._unowned_text:
            return
        orphan = TurnResult()
        self._active = orphan
        self._turn_done.clear()
        await self._send_finalize()
        self._finalize_sent = True
        try:
            await asyncio.wait_for(self._turn_done.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            pass
        text = orphan.final_text.strip() or _fallback_text(orphan) or self._unowned_text
        self.unowned_texts.append(text)
        self._unowned_text = ""

    async def _silence_prefix(self, realtime: bool) -> None:
        """Make sure the server has heard SILENCE_PREFIX_S of silence since
        the last real audio, sending only the shortfall."""
        quiet_for = time.monotonic() - self._last_real_send
        missing = max(0.0, SILENCE_PREFIX_S - quiet_for)
        for _ in range(int(round(missing * 10))):
            await self._send(SILENCE_CHUNK, real=False)
            if realtime:
                await asyncio.sleep(0.1)

    async def _send_turn(self, pcm16_bytes: bytes, chunk_ms: int, realtime: bool, result: TurnResult) -> TurnResult:
        """Send one turn's audio, then explicitly `finalize` it (see module
        docstring), and wait for `_read_loop` to resolve it.

        The `_turn_lock` is what makes a stray duplicate call for the same
        or a different turn safe: instead of a second coroutine racing the
        first one's `recv()` (the bug this module now avoids by design),
        it simply waits its turn here, sending only after the previous
        turn's audio and finalize are fully sent and resolved.
        """
        assert self._ws is not None
        bytes_per_chunk = int(self.sample_rate * 2 * chunk_ms / 1000)

        async with self._turn_lock:
            self._active = result
            self._finalize_sent = False
            self._turn_done.clear()
            self._streaming_turn = True
            try:
                await self._silence_prefix(realtime)
                for i in range(0, len(pcm16_bytes), bytes_per_chunk):
                    await self._send(pcm16_bytes[i : i + bytes_per_chunk])
                    if realtime:
                        await asyncio.sleep(chunk_ms / 1000)
                await self._send_finalize()
                self._finalize_sent = True
                self._streaming_turn = False
                await self._turn_done.wait()
            finally:
                self._active = None
                self._finalize_sent = False
                self._streaming_turn = False

        return result

    def start_turn(self, pcm16_bytes: bytes, *, chunk_ms: int = 100, realtime: bool = True) -> TurnHandle:
        """Kick off one pre-recorded turn's audio without blocking."""
        assert self._ws is not None
        result = TurnResult()
        future = asyncio.run_coroutine_threadsafe(
            self._send_turn(pcm16_bytes, chunk_ms, realtime, result), self._loop
        )
        return TurnHandle(result=result, future=future)

    def send_turn(self, pcm16_bytes: bytes, *, chunk_ms: int = 100, realtime: bool = True) -> TurnResult:
        """Stream one turn's raw PCM16 mono audio into the shared session
        and return the finalized result once `finalize` closes it out."""
        return self.start_turn(pcm16_bytes, chunk_ms=chunk_ms, realtime=realtime).join(timeout=60.0)

    def start_stream_turn(self) -> StreamTurnHandle:
        """Begin an agent turn whose audio is pushed while it is still being
        synthesized. Chunks go out every 100 ms against an absolute clock,
        the same pace the caller's speakers play them; an underrun (TTS
        slower than real time) sends silence, exactly like the playback
        track pads it, so what Transcribe hears stays in step with what the
        caller hears. `finish()` finalizes once the queue drains; `cut()`
        (barge-in) finalizes immediately."""
        assert self._ws is not None
        result = TurnResult()
        feed = _StreamFeed()

        async def _run() -> TurnResult:
            async with self._turn_lock:
                self._active = result
                self._finalize_sent = False
                self._turn_done.clear()
                self._streaming_turn = True
                try:
                    await self._flush_unowned_utterance()
                    self._active = result
                    self._finalize_sent = False
                    self._turn_done.clear()
                    await self._silence_prefix(realtime=True)
                    next_t = time.monotonic()
                    while True:
                        chunk, done = feed.take(CHUNK_BYTES)
                        if done:
                            break
                        await self._send(chunk or SILENCE_CHUNK, real=bool(chunk))
                        next_t += 0.1
                        await asyncio.sleep(max(0.0, next_t - time.monotonic()))
                    await self._send_finalize()
                    self._finalize_sent = True
                    # From here the pump streams silence, which is what lets
                    # the server close the utterance cleanly.
                    self._streaming_turn = False
                    await self._turn_done.wait()
                except websockets.exceptions.ConnectionClosed:
                    if not result.got_utterance_final:
                        result.final_text = _fallback_text(result)
                finally:
                    self._active = None
                    self._finalize_sent = False
                    self._streaming_turn = False
            return result

        future = asyncio.run_coroutine_threadsafe(_run(), self._loop)
        return StreamTurnHandle(result=result, future=future, feed=feed)

    # ------------------------------------------------------------------
    # live (microphone) turns
    # ------------------------------------------------------------------
    def start_live_turn(self) -> TurnHandle:
        """Begin a turn whose audio arrives continuously from a live
        microphone callback (`feed_live_audio`) instead of one
        pre-recorded buffer sent all at once.

        No client `finalize` is sent here. This is the single-speaker,
        own-thinking-pauses case Smart Turn's silence detection is
        documented for - the opposite of the fast multi-speaker exchange
        that made an explicit `finalize` necessary for agent turns (see the
        top-of-file docstring). `smart_turn_timeout` (configured on
        `connect()`) is still the safety net if a pause never gets
        classified confidently; `end_live_turn()` is a manual override.

        Holds `_turn_lock` for the live turn's entire duration, exactly
        like an agent turn holds it - so nothing else can become `_active`
        mid-conversation.
        """
        assert self._ws is not None
        result = TurnResult()

        async def _run() -> TurnResult:
            async with self._turn_lock:
                self._active = result
                self._finalize_sent = False
                self._live_turn_active = True
                self._unowned_text = ""  # anything still open is this turn's now
                self._turn_done.clear()
                try:
                    await self._turn_done.wait()
                    if not result.got_utterance_final and self.close_reason is None:
                        # Closed by the grace timeout, not by speech_final.
                        # Don't hand the turn back yet: an utterance still open
                        # on the server would otherwise land inside the next
                        # (agent) turn. Give it real silence and one more
                        # finalize, then accept the fallback text.
                        self._turn_done.clear()
                        try:
                            for _ in range(int(LIVE_RETRY_SILENCE_S * 10)):
                                await self._send(SILENCE_CHUNK, real=False)
                                await asyncio.sleep(0.1)
                            await self._send_finalize()
                            self._finalize_sent = True
                            await asyncio.wait_for(self._turn_done.wait(), timeout=LIVE_RETRY_WAIT_S)
                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                            pass
                    if not result.got_utterance_final:
                        result.final_text = _fallback_text(result)
                finally:
                    self._active = None
                    self._finalize_sent = False
                    self._live_turn_active = False
            return result

        future = asyncio.run_coroutine_threadsafe(_run(), self._loop)
        return TurnHandle(result=result, future=future)

    def feed_live_audio(self, pcm16_bytes: bytes) -> None:
        """Queue captured microphone audio for the socket. Safe from any
        thread; chunks are sent in call order by a single writer."""
        if not self.alive or self._mic_queue is None or not pcm16_bytes:
            return
        try:
            self._loop.call_soon_threadsafe(self._mic_queue.put_nowait, bytes(pcm16_bytes))
        except RuntimeError:  # loop already stopped
            pass

    def end_live_turn(self) -> None:
        """Manual 'I'm done talking' fallback for a live turn: ask the
        server to finalize now instead of waiting on Smart Turn alone.
        From this point the live turn behaves like a buffered one, with
        the same FINALIZE_GRACE_S safety net if speech_final is slow."""
        if not self.alive:
            return

        async def _finalize_now() -> None:
            try:
                await self._send_finalize()
                self._finalize_sent = True
            except websockets.exceptions.ConnectionClosed:
                pass

        asyncio.run_coroutine_threadsafe(_finalize_now(), self._loop)

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    async def _close(self) -> dict[str, Any] | None:
        if self._ws is None:
            await self._cancel_remaining_tasks()
            return None
        # Nothing may follow audio.done - a pump still sending silence keeps
        # the server from ever wrapping up with transcript.done.
        for task in (self._pump_task, self._mic_writer_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
                    pass
        self._pump_task = self._mic_writer_task = None
        if self.close_reason is None:
            self._call_done.clear()
            self._call_done_event = None
            try:
                await self._send(json.dumps({"type": "audio.done"}))
                await asyncio.wait_for(self._call_done.wait(), timeout=10)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass
        for task in (self._reader_task,):
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
                    pass
        self._reader_task = None
        try:
            await self._ws.close()
        except Exception:  # noqa: BLE001 - already closing, nothing to salvage
            pass
        self._ws = None
        self.close_reason = self.close_reason or "closed"
        await self._cancel_remaining_tasks()
        return self._call_done_event

    async def _cancel_remaining_tasks(self) -> None:
        """Cancel and await every other task on this loop (a live turn still
        waiting for speech_final, a stream turn, a finalize) so none is left
        pending when the loop stops - Python otherwise prints "Task was
        destroyed but it is pending!" for each one at shutdown."""
        self._turn_done.set()
        self._call_done.set()
        me = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks(self._loop) if t is not me and not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _shutdown_now(self) -> None:
        """Fast close for process exit: no audio.done / transcript.done wait."""
        self.close_reason = self.close_reason or "shutdown"
        if self._ws is not None:
            try:
                await asyncio.wait_for(self._ws.close(), timeout=1.0)
            except Exception:  # noqa: BLE001
                pass
            self._ws = None
        await self._cancel_remaining_tasks()

    def _stop_loop(self) -> None:
        _OPEN_LOOPS.discard(self)
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:  # loop already closed
            pass

    def close(self) -> dict[str, Any] | None:
        """Close the session. Never raises: a call that is already dead
        just gets cleaned up."""
        if not self.connected:
            self._stop_loop()
            return None
        try:
            return self._submit(self._close(), timeout=15.0)
        except Exception:  # noqa: BLE001
            return None
        finally:
            self.connected = False
            self._stop_loop()

    def shutdown_now(self, timeout: float = 2.0) -> None:
        """Used at interpreter exit for sessions nobody closed (an abandoned
        browser tab, Ctrl+C mid-call)."""
        if self._loop.is_running():
            try:
                self._submit(self._shutdown_now(), timeout=timeout)
            except Exception:  # noqa: BLE001
                pass
        self.connected = False
        self._stop_loop()


# Every session whose loop is still running, so an atexit hook can close the
# ones nobody closed (Ctrl+C mid-call, an abandoned tab) instead of leaving
# their tasks pending when the interpreter tears down.
_OPEN_LOOPS: "weakref.WeakSet[Any]" = weakref.WeakSet()


def _shutdown_open_sessions() -> None:
    for owner in list(_OPEN_LOOPS):
        try:
            owner.shutdown_now()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_shutdown_open_sessions)
