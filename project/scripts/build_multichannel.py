"""
Build a real 3-channel WAV from the same locked turns used for
clean_master.wav: channel 0 = Khalid, channel 1 = Maya, channel 2 = Nadia,
each at the exact start times from ground_truth/timeline.json, silent
elsewhere. This mirrors a call-center recording setup where each party
has a dedicated line, and lets multichannel=true be tested against known
per-speaker ground truth without relying on diarization.
"""
import json
import os
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "audio", "raw_lines")
MC_DIR = os.path.join(BASE, "audio", "multichannel")
os.makedirs(MC_DIR, exist_ok=True)

SPEAKERS = ["khalid", "maya", "nadia"]  # channel order 0,1,2


def main():
    timeline = json.load(open(os.path.join(BASE, "ground_truth", "timeline.json")))
    total_ms = timeline["total_duration_ms"]

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    input_index = 0
    per_speaker_filter = []
    channel_labels = []

    for speaker in SPEAKERS:
        turns = [t for t in timeline["turns"] if t["speaker"] == speaker]
        delayed = []
        for t in turns:
            wav = os.path.join(RAW_DIR, f"turn{t['turn_id']:02d}_{t['speaker']}.wav")
            cmd += ["-i", wav]
            per_speaker_filter.append(f"[{input_index}:a]adelay={t['start_ms']}|{t['start_ms']}[d{input_index}]")
            delayed.append(f"[d{input_index}]")
            input_index += 1
        # mix this speaker's own turns into one full-length mono channel track
        label = f"[ch_{speaker}]"
        per_speaker_filter.append(
            "".join(delayed) + f"amix=inputs={len(delayed)}:duration=longest:normalize=0,apad=whole_dur={total_ms/1000},aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono{label}"
        )
        channel_labels.append(label)

    # merge the 3 mono channel tracks into one 3-channel interleaved stream
    merge_filter = "".join(channel_labels) + f"amerge=inputs={len(channel_labels)}[merged]"
    filter_complex = ";".join(per_speaker_filter) + ";" + merge_filter

    out_wav = os.path.join(MC_DIR, "call_3channel.wav")
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[merged]",
        "-ar", "16000", "-ac", "3",
        out_wav,
    ]
    subprocess.run(cmd, check=True)

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=channels", "-of", "default=noprint_wrappers=1", out_wav],
        capture_output=True, text=True, check=True,
    ).stdout
    print(f"wrote {out_wav}")
    print(out)


if __name__ == "__main__":
    main()
