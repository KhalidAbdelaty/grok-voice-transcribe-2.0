"""
Decides what Maya or Nadia says next in the Streamlit live call, from what
Khalid actually said (as transcribed by Grok Voice Transcribe 2.0).

One streamed call per agent turn to the Responses API (`POST /v1/responses`,
`stream: true`), constrained by a loose Qivora Sync support scenario - who
the agents are and roughly how a call like this goes - but no scripted
wording. The article's measured results still come from the locked
fixture in ground_truth/qivora_call.py; this module only drives the
interactive demo.

Latency is the whole game for a voice agent, so (experiment_log.md entry 15):

- The default model is `grok-4.20-0309-non-reasoning`. `grok-4.7` cannot
  switch reasoning off (`low` is its floor, docs.x.ai "Reasoning"), and its
  thinking was 5-10 s of dead air per reply in entry 14. It stays selectable.
- The reply streams. The schema lists `speaker` first, so the TTS voice can
  be picked from the first few tokens, and `text` is decoded out of the
  partial JSON as it arrives so speech can start before the reply is done.
- `service_tier: "priority"` (2x token price, lower time to first token)
  and a per-call `prompt_cache_key` with the static instructions first, the
  same cache routing the sibling Grok LiveOps brain uses.
- A persistent `requests.Session`, so TLS setup is paid once per call, not
  once per reply. HTTP streaming rather than the Responses WebSocket: LiveOps
  measured that a Responses socket answers exactly one request, so holding
  one open gains nothing.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

import requests

from scripts.tutorial_checks import longest_digit_run

RESPONSES_URL = "https://api.x.ai/v1/responses"
FAST_MODEL = "grok-4.20-0309-non-reasoning"
REASONING_MODEL = "grok-4.7"
MODEL = FAST_MODEL
MODELS = (FAST_MODEL, REASONING_MODEL)

INSTRUCTIONS = """\
You voice two support agents on a live phone call for Qivora Sync, a file-sync
app that keeps files in step between a customer's laptop and phone.

- Maya is first-line support. She answers the call.
- Nadia is the escalation engineer. She only speaks after Maya brings her in.

How a call like this usually goes (a guide, not a script - react to what the
caller actually says, in whatever order they say it):
- Maya greets the caller, introduces herself, and asks how she can help.
- To look up the account she asks for the caller's phone number and email.
- If the problem sounds like it is on Qivora's side (sync failing with no
  obvious cause on the caller's devices), Maya says she is bringing in Nadia
  from the escalation team, and Nadia takes over.
- Nadia investigates, explains what she found in plain language, fixes it,
  and asks if there is anything else.
- When the caller is done, the agent who is speaking says goodbye.

Rules:
- Pick exactly one speaker per turn: "maya" or "nadia". Nadia never speaks
  before Maya has handed over to her.
- This is spoken audio: one to three short sentences, no lists, no markdown,
  no emoji, no stage directions.
- Always reply in English, set `language` to "en", even when the caller
  speaks or mixes in Arabic: understand the Arabic and answer it in English.
  Never ask the caller to switch to English; if the Arabic is unclear, ask
  about the problem itself. Keep the product name "Qivora Sync" as it is.
- When the caller gives a phone number or email, read it back once so they
  can confirm it.
- Reading back a phone number: a caller line may end with a note like
  "[digits heard: 01455502 (8 digits)]". Use exactly those digits, in order,
  never adding, dropping or repeating one, written digit by digit in spaced
  groups ("0 1 4 5, 5 5 0 2"). Only treat the number as incomplete if the
  caller says so.
- Reading back an email address: keep every part the caller said, in order,
  including parts after a dot ("jane dot doe"), and write it the way it is
  spoken ("jane dot doe at example dot com"), never as an @ address and
  never letter by letter unless the caller spelled it. If a part sounds
  unclear, read your best guess back and ask the caller to confirm it.
- The transcript you see comes from speech recognition and may contain small
  errors; go with the most plausible meaning instead of asking about typos.
- A line marked "(interrupted)" was cut off by the caller mid-sentence; the
  caller only heard the part shown. Respond to what the caller said next.
- Never invent the caller's details; use only what they actually said.
- Set end_call to true only on the turn where the agent says goodbye after
  the caller has indicated they are done.
"""

# Property order matters for streaming: `speaker` and `language` arrive first
# so the voice (and its TTS language) can be chosen before any text, `text`
# last so it can be spoken as it grows.
LANGUAGES = ("en",)
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker": {"type": "string", "enum": ["maya", "nadia"]},
        "language": {"type": "string", "enum": list(LANGUAGES)},
        "end_call": {"type": "boolean"},
        "text": {"type": "string"},
    },
    "required": ["speaker", "language", "end_call", "text"],
    "additionalProperties": False,
}

NAMES = {"khalid": "Caller", "maya": "Maya", "nadia": "Nadia"}


def _api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("XAI_API_KEY not set in environment")
    return key


@dataclass
class AgentReply:
    speaker: str
    text: str
    end_call: bool
    language: str = "en"


class ReplyCancelled(Exception):
    pass


class _PartialJsonReader:
    """Pulls `speaker`, `end_call` and the growing `text` value out of a
    JSON object that is still arriving token by token."""

    _SPEAKER = re.compile(r'"speaker"\s*:\s*"(maya|nadia)"')
    _LANGUAGE = re.compile(r'"language"\s*:\s*"(en)"')
    _END = re.compile(r'"end_call"\s*:\s*(true|false)')
    _TEXT = re.compile(r'"text"\s*:\s*"')
    _ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

    def __init__(self) -> None:
        self.raw = ""
        self.speaker: str | None = None
        self.language: str | None = None
        self.end_call: bool | None = None
        self._text_pos: int | None = None
        self.text_closed = False

    def feed(self, delta: str) -> str:
        """Add raw JSON, return newly decoded characters of `text`."""
        self.raw += delta
        if self.speaker is None:
            m = self._SPEAKER.search(self.raw)
            if m:
                self.speaker = m.group(1)
        if self.language is None:
            m = self._LANGUAGE.search(self.raw)
            if m:
                self.language = m.group(1)
        if self.end_call is None:
            m = self._END.search(self.raw)
            if m:
                self.end_call = m.group(1) == "true"
        if self._text_pos is None:
            m = self._TEXT.search(self.raw)
            if not m:
                return ""
            self._text_pos = m.end()
        if self.text_closed:
            return ""
        out: list[str] = []
        i = self._text_pos
        raw = self.raw
        while i < len(raw):
            ch = raw[i]
            if ch == '"':
                self.text_closed = True
                i += 1
                break
            if ch == "\\":
                if i + 1 >= len(raw):
                    break  # escape split across deltas; wait for the rest
                nxt = raw[i + 1]
                if nxt == "u":
                    if i + 6 > len(raw):
                        break
                    try:
                        out.append(chr(int(raw[i + 2 : i + 6], 16)))
                    except ValueError:
                        pass
                    i += 6
                    continue
                out.append(self._ESCAPES.get(nxt, nxt))
                i += 2
                continue
            out.append(ch)
            i += 1
        self._text_pos = i
        return "".join(out)


class ReplyStream:
    """One streamed agent reply. Iterate it for ("speaker", name) once and
    ("text", delta) as the reply is written; afterwards `reply`, the
    timings and `service_tier` are set. `cancel()` works from any thread."""

    def __init__(self, brain: "AgentBrain", pending_caller: str | None, timeout: float) -> None:
        self._brain = brain
        self.pending_caller = pending_caller
        # Built now, not when iteration starts on another thread: by then the
        # caller line may already have been committed to the history, and a
        # speculative prompt would carry it twice.
        self._body = brain._request_body(pending_caller)
        self._timeout = timeout
        self._cancel = threading.Event()
        self._resp: requests.Response | None = None
        self.reply: AgentReply | None = None
        self.started_at = time.monotonic()
        self.first_text_at: float | None = None
        self.done_at: float | None = None
        self.service_tier: str | None = None
        self.model = brain.model
        # Set just before the ("speaker", ...) event, so the voice's TTS
        # socket can be opened in the right language.
        self.language = "en"

    @property
    def ttft_ms(self) -> int | None:
        return None if self.first_text_at is None else int((self.first_text_at - self.started_at) * 1000)

    @property
    def total_ms(self) -> int | None:
        return None if self.done_at is None else int((self.done_at - self.started_at) * 1000)

    def cancel(self) -> None:
        self._cancel.set()
        resp = self._resp
        if resp is not None:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def __iter__(self) -> Iterator[tuple[str, str]]:
        brain = self._brain
        body = dict(self._body)
        body["stream"] = True
        reader = _PartialJsonReader()
        final_payload: dict | None = None
        try:
            resp = brain.http.post(
                RESPONSES_URL,
                headers=brain._headers(),
                json=body,
                stream=True,
                timeout=(10, self._timeout),
            )
            self._resp = resp
            # text/event-stream comes without a charset, and requests then
            # decodes as ISO-8859-1: Arabic replies arrived as mojibake.
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                raise RuntimeError(f"grok responses call failed ({resp.status_code}): {resp.text[:400]}")
            said_speaker = False
            held = ""  # text decoded before `speaker` was seen
            for line in resp.iter_lines(decode_unicode=True):
                if self._cancel.is_set():
                    raise ReplyCancelled()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                etype = evt.get("type", "")
                if etype == "response.output_text.delta":
                    new_text = held + reader.feed(evt.get("delta") or "")
                    held = ""
                    # `language` follows `speaker` in the schema; wait for it
                    # (or for the text to start, if it never comes).
                    if reader.speaker and not said_speaker and (reader.language or reader._text_pos is not None):
                        said_speaker = True
                        self.language = reader.language or "en"
                        yield "speaker", reader.speaker
                    if new_text and not said_speaker:
                        held = new_text
                    elif new_text:
                        if self.first_text_at is None:
                            self.first_text_at = time.monotonic()
                        yield "text", new_text
                elif etype == "response.completed":
                    final_payload = evt.get("response") or {}
                    break
                elif etype in ("response.failed", "error", "response.incomplete"):
                    detail = evt.get("response", {}).get("error") if "response" in evt else evt.get("error", evt)
                    raise RuntimeError(f"grok reply failed: {detail}")
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, AttributeError):
            if self._cancel.is_set():
                raise ReplyCancelled()
            raise
        finally:
            if self._resp is not None:
                self._resp.close()

        if self._cancel.is_set():
            raise ReplyCancelled()
        if final_payload is not None:
            self.service_tier = final_payload.get("service_tier")
            full = _output_text(final_payload)
            if full and len(full) > len(reader.raw):
                # Some deltas were missed; the completed payload is authoritative.
                rest = _PartialJsonReader()
                rest.feed(full)
                reader = rest
        self.reply = _parse_reply(reader.raw)
        self.done_at = time.monotonic()
        if self.first_text_at is None:
            # Nothing was streamed (no deltas, or no speaker until the end):
            # hand over the whole reply at once.
            self.first_text_at = self.done_at
            if not said_speaker:
                self.language = self.reply.language
                yield "speaker", self.reply.speaker
            yield "text", self.reply.text


@dataclass
class AgentBrain:
    """Conversation history plus one streamed model call per agent turn.

    Caller turns go in as the transcribed text; agent turns go in as the
    text the model generated (not the re-transcription of the TTS audio),
    so a recognition slip on an agent line can't derail the next reply - an
    interrupted line goes in as what Transcribe heard of it, since that is
    all the caller got to hear.

    Nothing is added to the history by `reply_stream()` itself: a
    speculative reply that ends up discarded, or a retry, must not leave a
    line behind. The caller commits with `add_caller` / `add_agent`.
    """
    history: list[tuple[str, str]] = field(default_factory=list)
    model: str = MODEL
    reasoning_effort: str = "low"  # only sent to reasoning models
    priority: bool = True
    cache_key: str = field(default_factory=lambda: f"qivora-live-{uuid.uuid4().hex[:12]}")
    http: requests.Session = field(default_factory=requests.Session, repr=False)
    last_service_tier: str | None = None

    def add_caller(self, text: str) -> None:
        text = text.strip() or "(inaudible)"
        if self.history and self.history[-1][0] == "khalid":
            # The caller carried on before any agent reply: one line, not two.
            self.history[-1] = ("khalid", f"{self.history[-1][1]} {text}")
            return
        self.history.append(("khalid", text))

    def add_agent(self, reply: AgentReply) -> None:
        self.history.append((reply.speaker, reply.text))

    def add_agent_partial(self, speaker: str, heard_text: str) -> None:
        self.history.append((speaker, f"{heard_text.strip() or '...'} (interrupted)"))

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}

    def _prompt(self, pending_caller: str | None = None) -> str:
        turns = list(self.history)
        if pending_caller is not None:
            turns.append(("khalid", pending_caller.strip() or "(inaudible)"))
        lines = [f"{NAMES[s]}: {t}{_digit_note(t) if s == 'khalid' else ''}" for s, t in turns]
        transcript = "\n".join(lines) if lines else "(the call just connected)"
        return (
            "Conversation so far:\n"
            f"{transcript}\n\n"
            "Write the next agent turn, in English (language en)."
        )

    def _request_body(self, pending_caller: str | None) -> dict:
        body: dict = {
            "model": self.model,
            "instructions": INSTRUCTIONS,
            "input": self._prompt(pending_caller),
            "prompt_cache_key": self.cache_key,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "agent_reply",
                    "schema": REPLY_SCHEMA,
                    "strict": True,
                }
            },
        }
        if self.model != FAST_MODEL:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if self.priority:
            body["service_tier"] = "priority"
        return body

    def warm_up(self) -> None:
        """Best effort: open the keep-alive connection and prime the prompt
        cache before the first real reply is needed. Failures are ignored."""
        try:
            body = self._request_body(None)
            body["generate"] = False
            body["max_output_tokens"] = 16
            self.http.post(RESPONSES_URL, headers=self._headers(), json=body, timeout=15).close()
        except Exception:  # noqa: BLE001
            pass

    def reply_stream(self, pending_caller: str | None = None, timeout: float = 30.0) -> ReplyStream:
        """A streamed reply to the history plus, optionally, a caller line
        that has not been committed yet (the speculative start)."""
        return ReplyStream(self, pending_caller, timeout)

    def reply(self, timeout: float = 30.0) -> AgentReply:
        """Blocking convenience wrapper; commits the reply to the history."""
        stream = self.reply_stream(timeout=timeout)
        for _ in stream:
            pass
        assert stream.reply is not None
        self.last_service_tier = stream.service_tier
        self.add_agent(stream.reply)
        return stream.reply


def _digit_note(caller_text: str) -> str:
    """The digits in a caller line, counted in code. Transcribe writes a
    dictated number as "0 1 4 5. 5 5 0 2.", and the model read that back as
    "01455" and then "014555502" (call_20260923_051305)."""
    digits = longest_digit_run(caller_text)
    return f" [digits heard: {digits} ({len(digits)} digits)]" if len(digits) >= 4 else ""


def _output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def _parse_reply(raw: str) -> AgentReply:
    raw = raw.strip()
    # Structured output should make this plain JSON; tolerate a fenced or
    # prefixed answer rather than failing a live call over formatting.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"grok reply had no JSON object: {raw[:200]!r}")
    data = json.loads(raw[start : end + 1])
    speaker = data.get("speaker") if data.get("speaker") in ("maya", "nadia") else "maya"
    text = str(data.get("text", "")).strip()
    if not text:
        raise RuntimeError(f"grok reply had empty text: {raw[:200]!r}")
    language = data.get("language") if data.get("language") in LANGUAGES else "en"
    return AgentReply(speaker=speaker, text=text, end_call=bool(data.get("end_call", False)), language=language)
