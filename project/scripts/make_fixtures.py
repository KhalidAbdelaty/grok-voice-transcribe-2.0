"""Rebuild every audio fixture the article and the checks use, from the
locked script in ground_truth/qivora_call.py.

    python project/scripts/make_fixtures.py

Steps (each is its own script, run in order):
  1. generate_tts.py        13 Grok TTS calls -> audio/raw_lines/turnNN_<speaker>.wav
                            (Khalid's turn 4 is four segments: en, ar-EG, en, en)
  2. mix_call.py            -> audio/mixed/clean_master.wav + ground_truth/timeline.json
  3. build_multichannel.py  -> audio/multichannel/call_3channel.wav (one speaker per channel)
  4. ffmpeg                 -> audio/phone/call_8k_mulaw.raw (8 kHz G.711 mu-law)
                            -> audio/tmp/turn04_khalid_16k.wav (the Smart Turn clip)

Needs XAI_API_KEY (read from .env) and ffmpeg/ffprobe on PATH. The TTS
calls cost a few cents; everything after step 1 is local.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from run_app import load_env  # noqa: E402

AUDIO = ROOT / "audio"


def run(step: str, cmd: list[str]) -> None:
    print(f"\n== {step}")
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT, env=os.environ.copy())
    print(f"   done in {time.time() - t0:.1f}s")


def main() -> None:
    load_env(ROOT.parent / ".env")
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        sys.exit(f"{' and '.join(missing)} not found on PATH - install ffmpeg first (https://ffmpeg.org/download.html)")
    if not os.environ.get("XAI_API_KEY"):
        sys.exit("XAI_API_KEY is not set - copy .env.example to .env and add your key")
    for sub in ("raw_lines", "mixed", "multichannel", "phone", "tmp"):
        (AUDIO / sub).mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    py = sys.executable
    run("1/4 Grok TTS lines", [py, "scripts/generate_tts.py"])
    run("2/4 mix the call", [py, "scripts/mix_call.py"])
    run("3/4 three-channel version", [py, "scripts/build_multichannel.py"])
    run("4/4 phone version and Smart Turn clip", [
        "ffmpeg", "-y", "-loglevel", "error", "-i", "audio/mixed/clean_master.wav",
        "-ar", "8000", "-ac", "1", "-f", "mulaw", "audio/phone/call_8k_mulaw.raw",
    ])
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", "audio/raw_lines/turn04_khalid.wav",
        "-ar", "16000", "-ac", "1", "audio/tmp/turn04_khalid_16k.wav",
    ], check=True, cwd=ROOT)

    print("\nFixtures ready:")
    for path in sorted(AUDIO.rglob("*")):
        if path.is_file() and path.suffix in (".wav", ".raw"):
            print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
