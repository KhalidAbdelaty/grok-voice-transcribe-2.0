"""Offline check for live_call_session.stitch_partials, no API calls.

Replays the real events from results/08_streaming_smartturn_events.json
one at a time, the way the UI sees them arrive, and asserts the displayed
text never loses what was already said: before each utterance final it
may only grow (an interim can revise its own tail, so the check is that
every locked chunk is still present), and the final shows the server's
stitched utterance.

    python project/scripts/test_stitch_partials.py

The events file is not shipped; record it once (a real streaming run, about
80 seconds, needs the fixtures and an API key):

    python project/scripts/stt_stream_client.py project/audio/mixed/clean_master.wav 08_streaming_smartturn
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.live_call_session import _event_text, stitch_partials  # noqa: E402

EVENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "08_streaming_smartturn_events.json")


def check_leading_punctuation() -> int:
    """The server can open an utterance with the previous one's closing mark
    (call_20260923_051305: ", okay, uh, 0 1 4 5." and "? Sorry about that")."""
    cases = [
        ([{"type": "transcript.partial", "text": ", okay, uh, 0 1 4 5.", "is_final": True, "speech_final": True}],
         "okay, uh, 0 1 4 5."),
        ([{"type": "transcript.partial", "text": "? Sorry about that", "is_final": False}], "Sorry about that"),
        ([{"type": "transcript.partial", "text": "؟ طيب", "is_final": True, "speech_final": True}],
         "طيب"),
    ]
    failures = 0
    for events, want in cases:
        got = stitch_partials(events)[0]
        if got != want:
            failures += 1
            print(f"FAIL leading punctuation: {ascii(got)} != {ascii(want)}")
    return failures


def main() -> int:
    if check_leading_punctuation():
        print("FAIL")
        return 1
    if not os.path.isfile(EVENTS):
        print("SKIP: no recorded streaming events yet. Record them once with\n"
              "  python project/scripts/stt_stream_client.py project/audio/mixed/clean_master.wav 08_streaming_smartturn")
        return 0
    events = json.load(open(EVENTS, encoding="utf-8"))["events"]
    seen: list[dict] = []
    locked_so_far: list[str] = []
    failures = 0
    for evt in events:
        seen.append(evt)
        if evt.get("type") != "transcript.partial":
            continue
        text, tag = stitch_partials(seen)
        if evt.get("speech_final"):
            # The display now keeps every finished utterance of the turn, so
            # it must end with the server's stitched utterance.
            if not text.endswith(_event_text(evt)):
                failures += 1
                print(f"FAIL t={evt['t']}: final text does not end with the server's stitched utterance")
            print(f"{evt['t']:7.2f} final   {len(text):4d} chars  ...{ascii(text[-70:])}")
            locked_so_far = []
            continue
        if evt.get("is_final"):
            locked_so_far.append(_event_text(evt))
        for chunk in locked_so_far:
            if chunk and chunk not in text:
                failures += 1
                print(f"FAIL t={evt['t']}: locked chunk vanished from the display: {chunk[:60]!r}")
        print(f"{evt['t']:7.2f} {tag:7s} {len(text):4d} chars  ...{ascii(text[-70:])}")

    naive_drop = sum(
        1 for a, b in zip(events, events[1:])
        if a.get("type") == b.get("type") == "transcript.partial"
        and a.get("is_final") and not a.get("speech_final") and not b.get("is_final")
    )
    print(f"\nchunk-final -> interim resets in this run (where the old UI dropped text): {naive_drop}")
    print("PASS" if failures == 0 else f"{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
