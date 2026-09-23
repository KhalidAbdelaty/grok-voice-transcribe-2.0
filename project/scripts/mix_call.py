"""
Assemble the ten synthesized/recorded turns into one clean master
conversation, preserving realistic timing and one brief controlled
overlap (Maya starts turn 3 while Khalid is still finishing turn 2).
Writes the timeline to ground_truth/timeline.json so every later
comparison (diarization, Smart Turn) can be checked against known
start/end times, not guessed.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ground_truth.qivora_call import TURNS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "audio", "raw_lines")
MIXED_DIR = os.path.join(BASE, "audio", "mixed")
os.makedirs(MIXED_DIR, exist_ok=True)

DEFAULT_GAP_MS = 400


def probe_duration_ms(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return round(float(out) * 1000)


def main():
    cursor = 0
    timeline = []
    inputs = []
    for turn in TURNS:
        wav = os.path.join(RAW_DIR, f"turn{turn['turn_id']:02d}_{turn['speaker']}.wav")
        dur = probe_duration_ms(wav)
        overlap = turn.get("overlap_with_previous_ms", 0)
        start = max(0, cursor - overlap) if overlap else cursor
        end = start + dur
        timeline.append({
            "turn_id": turn["turn_id"], "speaker": turn["speaker"],
            "start_ms": start, "end_ms": end, "duration_ms": dur,
            "notes": turn["notes"], "plain_text": turn["plain_text"],
        })
        inputs.append((wav, start))
        cursor = end + DEFAULT_GAP_MS

    total_ms = max(t["end_ms"] for t in timeline)

    # build ffmpeg filter_complex: delay each input to its start offset, then mix
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for wav, _ in inputs:
        cmd += ["-i", wav]
    filter_parts = []
    mix_labels = []
    for i, (_, start) in enumerate(inputs):
        filter_parts.append(f"[{i}:a]adelay={start}|{start}[a{i}]")
        mix_labels.append(f"[a{i}]")
    filter_complex = ";".join(filter_parts) + ";" + "".join(mix_labels) + f"amix=inputs={len(inputs)}:duration=longest:normalize=0[mixed]"
    out_wav = os.path.join(MIXED_DIR, "clean_master.wav")
    cmd += ["-filter_complex", filter_complex, "-map", "[mixed]", "-ar", "16000", "-ac", "1", out_wav]
    subprocess.run(cmd, check=True)

    with open(os.path.join(os.path.dirname(BASE) if False else os.path.join(BASE, "ground_truth"), "timeline.json"), "w") as f:
        json.dump({"total_duration_ms": total_ms, "turns": timeline}, f, indent=2, ensure_ascii=False)

    actual_dur = probe_duration_ms(out_wav)
    print(f"clean_master.wav written: {actual_dur/1000:.1f}s (planned {total_ms/1000:.1f}s)")
    print("timeline: ground_truth/timeline.json")


if __name__ == "__main__":
    main()
