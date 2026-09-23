"""
Reusable REST client for Grok Voice Transcribe 2.0 (POST /v1/stt).

Every call this script makes is logged to results/stt_call_log.json
(append mode) and the raw JSON response is saved to results/<run_name>.json
so every number quoted in the article traces back to a real, reproducible
API response.

Usage (as a library):
    from stt_client import transcribe
    result, meta = transcribe(
        audio_path="audio/mixed/clean_master.wav",
        run_name="01_baseline",
        model="grok-voice-transcribe-2.0",
    )

Usage (CLI, for the baseline run):
    python3 scripts/stt_client.py audio/mixed/clean_master.wav 01_baseline
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

STT_URL = "https://api.x.ai/v1/stt"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CALL_LOG = RESULTS_DIR / "stt_call_log.json"


def _api_key() -> str:
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("XAI_API_KEY not set in environment")
    return key


def _append_log(entry: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    if CALL_LOG.exists():
        log = json.loads(CALL_LOG.read_text())
    log.append(entry)
    CALL_LOG.write_text(json.dumps(log, indent=2))


def transcribe(
    audio_path: str,
    run_name: str,
    *,
    model: str = "grok-voice-transcribe-2.0",
    language: str | None = None,
    format_: bool | None = None,
    diarize: bool | None = None,
    keyterms: list[str] | None = None,
    filler_words: bool | None = None,
    vad_threshold: float | None = None,
    multichannel: bool | None = None,
    channels: int | None = None,
    audio_format: str | None = None,
    sample_rate: int | None = None,
    save_raw: bool = True,
) -> tuple[dict, dict]:
    """Make one real POST /v1/stt call and return (response_json, meta).

    Fields are sent in the multipart form BEFORE `file`, per the docs
    ("Option fields should precede `file`... for streamable uploads,
    fields sent after `file` may be ignored"), and `file` is added last.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(audio_file)

    data: list[tuple[str, str]] = [("model", model)]
    if language is not None:
        data.append(("language", language))
    if format_ is not None:
        data.append(("format", "true" if format_ else "false"))
    if diarize is not None:
        data.append(("diarize", "true" if diarize else "false"))
    if filler_words is not None:
        data.append(("filler_words", "true" if filler_words else "false"))
    if vad_threshold is not None:
        data.append(("vad_threshold", str(vad_threshold)))
    if multichannel is not None:
        data.append(("multichannel", "true" if multichannel else "false"))
    if channels is not None:
        data.append(("channels", str(channels)))
    if audio_format is not None:
        data.append(("audio_format", audio_format))
    if sample_rate is not None:
        data.append(("sample_rate", str(sample_rate)))
    for term in keyterms or []:
        data.append(("keyterm", term))

    mime = "audio/wav" if audio_file.suffix.lower() == ".wav" else "application/octet-stream"

    t0 = time.time()
    with open(audio_file, "rb") as f:
        response = requests.post(
            STT_URL,
            headers={"Authorization": f"Bearer {_api_key()}"},
            data=data,
            files={"file": (audio_file.name, f, mime)},
            timeout=180,
        )
    elapsed = time.time() - t0

    meta = {
        "run_name": run_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audio_path": str(audio_file),
        "audio_bytes": audio_file.stat().st_size,
        "request_fields": dict(data),
        "status_code": response.status_code,
        "elapsed_s": round(elapsed, 3),
    }

    if response.status_code != 200:
        meta["error_body"] = response.text[:2000]
        _append_log(meta)
        response.raise_for_status()

    result = response.json()
    meta["response_text_preview"] = result.get("text", "")[:200]
    meta["response_duration_s"] = result.get("duration")
    meta["response_word_count"] = len(result.get("words", []))
    _append_log(meta)

    if save_raw:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{run_name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        meta["saved_to"] = str(out_path)

    return result, meta


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: stt_client.py <audio_path> <run_name> [model]")
        sys.exit(1)
    audio_arg = sys.argv[1]
    run_name_arg = sys.argv[2]
    model_arg = sys.argv[3] if len(sys.argv) > 3 else "grok-voice-transcribe-2.0"

    result, meta = transcribe(audio_arg, run_name_arg, model=model_arg)
    print(f"status: {meta['status_code']}  elapsed: {meta['elapsed_s']}s")
    print(f"duration: {result.get('duration')}s  words: {len(result.get('words', []))}")
    print(f"text: {result.get('text')}")
    print(f"saved to: {meta.get('saved_to')}")
