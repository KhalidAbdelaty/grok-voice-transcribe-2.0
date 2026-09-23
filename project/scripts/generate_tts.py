"""
Generate the Khalid/Maya/Nadia audio lines for the Qivora Sync fixture
from the real Grok Text to Speech API. Every file here is a live API
result, not a cached sample. Run once; re-run is safe (overwrites).
"""
import json
import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ground_truth.qivora_call import TURNS, VOICES

XAI_API_KEY = os.environ["XAI_API_KEY"]
TTS_URL = "https://api.x.ai/v1/tts"
HEADERS = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "audio", "raw_lines")
os.makedirs(RAW_DIR, exist_ok=True)

RUN_LOG = []


def synthesize(text: str, voice_id: str, language: str) -> bytes:
    resp = requests.post(
        TTS_URL,
        headers=HEADERS,
        json={"text": text, "voice_id": voice_id, "language": language},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def main():
    for turn in TURNS:
        speaker = turn["speaker"]
        voice_id = VOICES[speaker]
        seg_paths = []
        for i, (lang, text) in enumerate(turn["segments"]):
            out_mp3 = os.path.join(RAW_DIR, f"turn{turn['turn_id']:02d}_{speaker}_seg{i}.mp3")
            t0 = time.time()
            audio = synthesize(text, voice_id, lang)
            elapsed = time.time() - t0
            with open(out_mp3, "wb") as f:
                f.write(audio)
            seg_paths.append(out_mp3)
            RUN_LOG.append({
                "turn_id": turn["turn_id"], "speaker": speaker, "voice_id": voice_id,
                "segment": i, "language": lang, "text": text,
                "bytes": len(audio), "tts_latency_s": round(elapsed, 3),
            })
            print(f"turn{turn['turn_id']:02d} seg{i} [{lang}] {voice_id}: {len(audio)} bytes in {elapsed:.2f}s")

        # concatenate this turn's segments (if >1) into one continuous line
        turn_wav = os.path.join(RAW_DIR, f"turn{turn['turn_id']:02d}_{speaker}.wav")
        if len(seg_paths) == 1:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", seg_paths[0],
                 "-ar", "48000", "-ac", "1", turn_wav],
                check=True,
            )
        else:
            list_file = os.path.join(RAW_DIR, f"turn{turn['turn_id']:02d}_concat.txt")
            wavs = []
            for j, sp in enumerate(seg_paths):
                w = sp.replace(".mp3", ".wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", sp,
                     "-ar", "48000", "-ac", "1", w],
                    check=True,
                )
                wavs.append(w)
            with open(list_file, "w") as f:
                for w in wavs:
                    f.write(f"file '{os.path.abspath(w)}'\n")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                 "-i", list_file, "-ar", "48000", "-ac", "1", turn_wav],
                check=True,
            )

    log_path = os.path.join(BASE, "results", "tts_generation_log.json")
    with open(log_path, "w") as f:
        json.dump(RUN_LOG, f, indent=2, ensure_ascii=False)
    total_bytes = sum(r["bytes"] for r in RUN_LOG)
    total_calls = len(RUN_LOG)
    total_chars = sum(len(r["text"]) for r in RUN_LOG)
    print(f"\n{total_calls} real TTS calls, {total_bytes:,} bytes audio, {total_chars} chars billed")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
