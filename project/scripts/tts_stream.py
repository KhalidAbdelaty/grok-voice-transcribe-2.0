"""Streaming Grok TTS for the live call's agent voices.

The REST `/v1/tts` call the app used to make returns the whole MP3 only
once synthesis is finished, and it then had to go through ffmpeg before a
single sample could play. The bidirectional socket (`wss://api.x.ai/v1/tts`,
docs "Streaming TTS (WebSocket)") takes text as it is written and sends
audio back as it is synthesized, so a reply can start playing while the
model is still writing it.

- `codec=pcm&sample_rate=16000`: raw PCM16 mono at the STT socket's rate,
  so the same bytes go to the caller's speakers and into Transcribe with no
  decoding step.
- `optimize_streaming_latency=2`: the smallest first chunk, documented as
  "lowest time-to-first-audio, with more noticeable quality tradeoff at
  chunk boundaries" - on a phone-style call the first syllable matters more.
- One connection per voice, opened when the call starts and reused for
  every utterance: the docs say the connection stays open after
  `audio.done`, and a fresh handshake costs ~600 ms for distant clients.
  A dropped connection is reopened on the next utterance.

`synthesize_pcm_rest` is the fallback when the socket fails before any
audio has arrived.
"""
from __future__ import annotations

import asyncio
import atexit
import base64
import json
import os
import queue
import threading
import time
import weakref
from urllib.parse import urlencode

import requests
import websockets
from websockets.protocol import State

TTS_WS = "wss://api.x.ai/v1/tts"
TTS_REST = "https://api.x.ai/v1/tts"
SAMPLE_RATE = 16000


def _api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("XAI_API_KEY not set in environment")
    return key


def synthesize_pcm_rest(text: str, voice_id: str, language: str = "en", timeout: float = 60.0) -> bytes:
    """One-shot REST synthesis straight to PCM16 16 kHz (no ffmpeg)."""
    resp = requests.post(
        TTS_REST,
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
        json={
            "text": text,
            "voice_id": voice_id,
            "language": language,
            "output_format": {"codec": "pcm", "sample_rate": SAMPLE_RATE},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.content


class TtsUtterance:
    """One utterance on a voice's socket. Text goes in with `push_text` /
    `done_text` from any thread; PCM comes out of `audio` (a queue that
    ends with None). `error` is set if the socket failed."""

    def __init__(self, streamer: "TtsStreamer", voice_id: str, language: str = "en") -> None:
        self._streamer = streamer
        self.voice_id = voice_id
        self.language = language
        self.audio: "queue.Queue[bytes | None]" = queue.Queue()
        self.error: str | None = None
        self.got_audio = False
        self.first_text_at: float | None = None
        self.first_audio_at: float | None = None
        self.text = ""
        self._text_q: asyncio.Queue | None = None
        self._pending: list[tuple[str, str | None]] = []
        self._lock = threading.Lock()
        self.cancelled = False

    def _enqueue(self, item: tuple[str, str | None]) -> None:
        with self._lock:
            if self._text_q is None:
                self._pending.append(item)
                return
        self._streamer._loop.call_soon_threadsafe(self._text_q.put_nowait, item)

    def _attach(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._text_q = q
            for item in self._pending:
                q.put_nowait(item)
            self._pending.clear()

    def push_text(self, delta: str) -> None:
        if not delta or self.cancelled:
            return
        if self.first_text_at is None:
            self.first_text_at = time.monotonic()
        self.text += delta
        self._enqueue(("delta", delta))

    def done_text(self) -> None:
        self._enqueue(("done", None))

    def cancel(self) -> None:
        """Barge-in or a discarded speculative reply: stop synthesis."""
        if self.cancelled:
            return
        self.cancelled = True
        self._enqueue(("clear", None))

    @property
    def tts_first_audio_ms(self) -> int | None:
        if self.first_text_at is None or self.first_audio_at is None:
            return None
        return int((self.first_audio_at - self.first_text_at) * 1000)


class TtsStreamer:
    """Persistent TTS sockets, one per (voice, language), on a private event
    loop. `language` is a socket URL parameter, so an agent answering in
    Egyptian Arabic needs its own socket; `language` and `preopen` are opened
    up front (a cold socket costs ~0.7 s of first audio), anything else on
    first use."""

    def __init__(self, voice_ids: list[str], language: str = "en", latency: int = 2,
                 preopen: tuple[str, ...] = ("ar-EG",)) -> None:
        self.voice_ids = list(dict.fromkeys(voice_ids))
        self.language = language
        self.preopen = preopen
        self.latency = latency
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._sockets: dict[tuple[str, str], websockets.ClientConnection] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.closed = False
        _OPEN_STREAMERS.add(self)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _url(self, voice_id: str, language: str) -> str:
        return f"{TTS_WS}?" + urlencode({
            "voice": voice_id,
            "language": language,
            "codec": "pcm",
            "sample_rate": str(SAMPLE_RATE),
            "optimize_streaming_latency": str(self.latency),
        })

    async def _socket(self, voice_id: str, language: str | None = None) -> websockets.ClientConnection:
        key = (voice_id, language or self.language)
        ws = self._sockets.get(key)
        if ws is not None and ws.state is State.OPEN:
            return ws
        ws = await websockets.connect(
            self._url(*key),
            additional_headers={"Authorization": f"Bearer {_api_key()}"},
            max_size=None,
        )
        self._sockets[key] = ws
        return ws

    def connect_all(self) -> None:
        """Pre-open every voice's socket; failures are left for the first
        utterance to retry."""
        async def _open_all():
            for language in (self.language, *self.preopen):
                for voice_id in self.voice_ids:
                    try:
                        await self._socket(voice_id, language)
                    except Exception:  # noqa: BLE001
                        pass

        asyncio.run_coroutine_threadsafe(_open_all(), self._loop)

    def start(self, voice_id: str, language: str | None = None) -> TtsUtterance:
        utt = TtsUtterance(self, voice_id, language or self.language)
        asyncio.run_coroutine_threadsafe(self._run_utterance(utt), self._loop)
        return utt

    async def _run_utterance(self, utt: TtsUtterance) -> None:
        key = (utt.voice_id, utt.language)
        lock = self._locks.setdefault(key, asyncio.Lock())
        text_q: asyncio.Queue = asyncio.Queue()
        utt._attach(text_q)
        async with lock:
            try:
                await self._speak(utt, text_q)
            except Exception as exc:  # noqa: BLE001 - reported through utt.error
                utt.error = f"{type(exc).__name__}: {exc}"
                stale = self._sockets.pop(key, None)
                if stale is not None:
                    try:
                        await stale.close()
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                utt.audio.put(None)

    async def _speak(self, utt: TtsUtterance, text_q: asyncio.Queue) -> None:
        ws = await self._socket(utt.voice_id, utt.language)
        finished = asyncio.Event()

        async def sender() -> None:
            sent_any = False
            while True:
                kind, payload = await text_q.get()
                if kind == "delta":
                    await ws.send(json.dumps({"type": "text.delta", "delta": payload}))
                    sent_any = True
                elif kind == "done":
                    if not sent_any:
                        finished.set()
                        return
                    await ws.send(json.dumps({"type": "text.done"}))
                    return
                elif kind == "clear":
                    if sent_any:
                        await ws.send(json.dumps({"type": "text.clear"}))
                    else:
                        finished.set()
                    return

        send_task = asyncio.create_task(sender())
        finished_task = asyncio.create_task(finished.wait())
        try:
            while not finished.is_set():
                recv = asyncio.create_task(ws.recv())
                done, _ = await asyncio.wait({recv, finished_task}, return_when=asyncio.FIRST_COMPLETED)
                if recv not in done:
                    recv.cancel()
                    break
                evt = json.loads(recv.result())
                etype = evt.get("type")
                if etype == "audio.delta":
                    if utt.cancelled:
                        continue
                    pcm = base64.b64decode(evt.get("delta") or "")
                    if pcm:
                        if utt.first_audio_at is None:
                            utt.first_audio_at = time.monotonic()
                        utt.got_audio = True
                        utt.audio.put(pcm)
                elif etype in ("audio.done", "audio.clear"):
                    break
                elif etype == "error":
                    raise RuntimeError(evt.get("message") or str(evt))
            if send_task.done() and not send_task.cancelled() and send_task.exception():
                raise send_task.exception()
        finally:
            finished_task.cancel()
            if not send_task.done():
                send_task.cancel()

    def close(self, timeout: float = 5.0) -> None:
        if self.closed:
            return
        self.closed = True
        _OPEN_STREAMERS.discard(self)

        async def _close_all():
            for ws in list(self._sockets.values()):
                try:
                    await asyncio.wait_for(ws.close(), timeout=1.0)
                except Exception:  # noqa: BLE001
                    pass
            self._sockets.clear()
            # Cancel and await anything still running (an utterance waiting on
            # text, a reader) so no task is left pending when the loop stops.
            me = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks(self._loop) if t is not me and not t.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_close_all(), self._loop).result(timeout=timeout)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass


_OPEN_STREAMERS: "weakref.WeakSet[TtsStreamer]" = weakref.WeakSet()


def _close_open_streamers() -> None:
    for streamer in list(_OPEN_STREAMERS):
        streamer.close(timeout=2.0)


atexit.register(_close_open_streamers)
