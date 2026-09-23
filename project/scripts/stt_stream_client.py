"""
Reusable streaming client for Grok Voice Transcribe 2.0 (wss://api.x.ai/v1/stt).

Streams a 16kHz mono PCM16 WAV file in real-time-paced 100ms chunks
(matching how a live microphone would arrive), and logs every server
event (transcript.created / transcript.partial / transcript.done /
error) to results/<run_name>_events.json so interim vs. final vs.
Smart-Turn behavior can be inspected and quoted exactly as observed.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import wave
from pathlib import Path
from urllib.parse import urlencode

import websockets

WS_BASE = "wss://api.x.ai/v1/stt"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("XAI_API_KEY not set in environment")
    return key


async def stream_transcribe(
    wav_path: str,
    run_name: str,
    *,
    model: str = "grok-voice-transcribe-2.0",
    sample_rate: int = 16000,
    interim_results: bool = True,
    diarize: bool = False,
    language: str | None = None,
    keyterms: list[str] | None = None,
    filler_words: bool | None = None,
    smart_turn: float | None = None,
    smart_turn_timeout: int | None = None,
    endpointing: int | None = None,
    realtime_pace: bool = True,
    chunk_ms: int = 100,
) -> list[dict]:
    params: list[tuple[str, str]] = [
        ("model", model),
        ("sample_rate", str(sample_rate)),
        ("encoding", "pcm"),
        ("interim_results", "true" if interim_results else "false"),
    ]
    if diarize:
        params.append(("diarize", "true"))
    if language:
        params.append(("language", language))
    if filler_words is not None:
        params.append(("filler_words", "true" if filler_words else "false"))
    if smart_turn is not None:
        params.append(("smart_turn", str(smart_turn)))
    if smart_turn_timeout is not None:
        params.append(("smart_turn_timeout", str(smart_turn_timeout)))
    if endpointing is not None:
        params.append(("endpointing", str(endpointing)))
    for term in keyterms or []:
        params.append(("keyterm", term))

    url = f"{WS_BASE}?{urlencode(params)}"
    headers = {"Authorization": f"Bearer {_api_key()}"}

    wf = wave.open(wav_path, "rb")
    assert wf.getframerate() == sample_rate, f"wav is {wf.getframerate()}Hz, expected {sample_rate}Hz"
    assert wf.getnchannels() == 1, "expected mono wav"
    assert wf.getsampwidth() == 2, "expected 16-bit PCM"

    bytes_per_chunk = int(sample_rate * 2 * chunk_ms / 1000)

    events: list[dict] = []
    t_start = time.time()

    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        msg = json.loads(await ws.recv())
        events.append({"t": round(time.time() - t_start, 3), **msg})
        assert msg["type"] == "transcript.created", msg

        async def sender():
            while True:
                chunk = wf.readframes(bytes_per_chunk // 2)
                if not chunk:
                    break
                await ws.send(chunk)
                if realtime_pace:
                    await asyncio.sleep(chunk_ms / 1000)
            await ws.send(json.dumps({"type": "audio.done"}))

        async def receiver():
            async for raw in ws:
                evt = json.loads(raw)
                evt["_t"] = round(time.time() - t_start, 3)
                events.append({"t": evt["_t"], **{k: v for k, v in evt.items() if k != "_t"}})
                if evt["type"] == "transcript.done":
                    break
                if evt["type"] == "error":
                    break

        await asyncio.gather(sender(), receiver())

    wf.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_name}_events.json"
    out_path.write_text(json.dumps({
        "run_name": run_name,
        "wav_path": wav_path,
        "params": dict(params),
        "event_count": len(events),
        "events": events,
    }, indent=2, ensure_ascii=False))
    print(f"saved {len(events)} events to {out_path}")
    return events


if __name__ == "__main__":
    import sys
    wav_arg = sys.argv[1]
    run_name_arg = sys.argv[2]
    asyncio.run(stream_transcribe(wav_arg, run_name_arg, diarize=True, keyterms=["Qivora Sync"], smart_turn=0.7, smart_turn_timeout=3000, language="en"))
