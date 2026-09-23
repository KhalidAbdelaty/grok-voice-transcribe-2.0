"""Turn a diarized `words` array into readable, timed speaker turns.

Grok Voice Transcribe 2.0 puts a numeric `speaker` on every word when
`diarize=true`; it has no idea who "Maya" is. This groups consecutive words
from the same speaker into turns and maps the ids to names by order of first
appearance, because we know the script: Maya answers, Khalid speaks second,
Nadia joins last.

    python project/scripts/speaker_turns.py results/03_diarize.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CALL_ORDER = ["Maya", "Khalid", "Nadia"]


def group_turns(words: list[dict]) -> list[dict]:
    """Consecutive words with the same speaker id become one turn."""
    turns: list[dict] = []
    for word in words:
        speaker = word.get("speaker")
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["words"].append(word["text"])
            turns[-1]["end"] = word["end"]
        else:
            turns.append({"speaker": speaker, "start": word["start"], "end": word["end"], "words": [word["text"]]})
    for turn in turns:
        turn["text"] = " ".join(turn.pop("words"))
    return turns


def names_by_first_appearance(turns: list[dict], names: list[str] = CALL_ORDER) -> dict:
    """{speaker_id: name}, in the order the ids first speak."""
    mapping: dict = {}
    for turn in turns:
        if turn["speaker"] not in mapping and len(mapping) < len(names):
            mapping[turn["speaker"]] = names[len(mapping)]
    return mapping


def format_turns(turns: list[dict], names: dict) -> str:
    lines = []
    for turn in turns:
        who = names.get(turn["speaker"], f"Speaker {turn['speaker']}")
        lines.append(f"[{turn['start']:6.2f}-{turn['end']:6.2f}] {who:<7} {turn['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "03_diarize.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    turns = group_turns(result["words"])
    print(format_turns(turns, names_by_first_appearance(turns)))
    print(f"\n{len(turns)} turns from {len(result['words'])} words")
