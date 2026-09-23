"""The finished support-call transcriber: one config, batch or streaming.

Everything the article adds one option at a time, behind a single
`SupportConfig`: diarization (or multichannel), keyterm biasing, formatting,
filler words, language, VAD, and Smart Turn for streaming. Each run saves the
transcript, the readable speaker turns, the word metadata and the exact
configuration, so any result can be reproduced from its JSON alone.

    # the final run in the article: the 8 kHz mu-law phone call, batch
    python project/scripts/support_transcriber.py audio/phone/call_8k_mulaw.raw --phone

    # the same client, streaming the clean 16 kHz call with Smart Turn
    python project/scripts/support_transcriber.py audio/mixed/clean_master.wav --stream
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from run_app import load_env  # noqa: E402
from ground_truth.qivora_call import PRODUCT_NAME  # noqa: E402
from scripts.speaker_turns import format_turns, group_turns, names_by_first_appearance  # noqa: E402
from scripts.stt_client import transcribe  # noqa: E402
from scripts.stt_stream_client import stream_transcribe  # noqa: E402

MODEL = "grok-voice-transcribe-2.0"


@dataclass
class SupportConfig:
    diarize: bool = True
    multichannel: bool = False
    keyterms: list[str] = field(default_factory=lambda: [PRODUCT_NAME])
    format: bool = True
    language: str | None = "en"  # formatting needs a language
    filler_words: bool = False  # clean support notes; set True for a verbatim record
    vad_threshold: float | None = None
    smart_turn: float = 0.8  # streaming only
    smart_turn_timeout: int = 3000  # streaming only
    audio_format: str | None = None  # "mulaw" / "alaw" for raw telephony audio
    sample_rate: int | None = None


def run_batch(audio: Path, cfg: SupportConfig, run_name: str) -> dict:
    result, meta = transcribe(
        str(audio), run_name, model=MODEL,
        language=cfg.language, format_=cfg.format, diarize=cfg.diarize and not cfg.multichannel,
        keyterms=cfg.keyterms, filler_words=cfg.filler_words, vad_threshold=cfg.vad_threshold,
        multichannel=cfg.multichannel or None, audio_format=cfg.audio_format, sample_rate=cfg.sample_rate,
    )
    return {"mode": "batch", "text": result.get("text", ""), "words": result.get("words", []),
            "duration_s": result.get("duration"), "elapsed_s": meta["elapsed_s"]}


def run_stream(audio: Path, cfg: SupportConfig, run_name: str) -> dict:
    events = asyncio.run(stream_transcribe(
        str(audio), run_name, model=MODEL, diarize=cfg.diarize, language=cfg.language,
        keyterms=cfg.keyterms, filler_words=cfg.filler_words,
        smart_turn=cfg.smart_turn, smart_turn_timeout=cfg.smart_turn_timeout,
    ))
    finals = [e for e in events if e.get("type") == "transcript.partial" and e.get("speech_final")]
    words = [w for e in finals for w in e.get("words", [])]
    first = next((e["t"] for e in events if e.get("type") == "transcript.partial"), None)
    return {"mode": "stream", "text": " ".join(e.get("text", "") for e in finals), "words": words,
            "turn_closes": [{"t": e["t"], "end_of_turn_confidence": e.get("end_of_turn_confidence")} for e in finals],
            "first_text_after_s": first}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="path relative to project/ (or absolute)")
    parser.add_argument("--phone", action="store_true", help="raw 8 kHz G.711 mu-law input")
    parser.add_argument("--stream", action="store_true", help="stream over the WebSocket instead of batch")
    parser.add_argument("--fillers", action="store_true", help="keep um/uh")
    parser.add_argument("--name", default=None, help="results/<name>.json")
    args = parser.parse_args()

    load_env(ROOT.parent / ".env")
    audio = Path(args.audio) if Path(args.audio).is_absolute() else ROOT / args.audio
    cfg = SupportConfig(filler_words=args.fillers)
    if args.phone:
        cfg.audio_format, cfg.sample_rate = "mulaw", 8000
    name = args.name or ("20_final_phone_run" if args.phone else "20_final_run")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = run_stream(audio, cfg, name) if args.stream else run_batch(audio, cfg, name)
    turns = group_turns(out["words"]) if out["words"] and "speaker" in out["words"][0] else []
    names = names_by_first_appearance(turns)
    record = {
        "model": MODEL,
        "audio_fixture": str(audio.relative_to(ROOT)) if audio.is_relative_to(ROOT) else str(audio),
        "run_at_utc": started,
        "config": asdict(cfg),
        "speaker_names": {str(k): v for k, v in names.items()},
        "turns": [{**t, "name": names.get(t["speaker"])} for t in turns],
        **out,
    }
    path = ROOT / "results" / f"{name}_transcript.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_turns(turns, names) if turns else out["text"])
    print(f"\nsaved {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
