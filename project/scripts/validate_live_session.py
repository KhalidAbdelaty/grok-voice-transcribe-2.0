"""Non-mic validation of LiveCallSession: replays all 10 cached article
lines (real TTS audio, including the sirius stand-in for Khalid) through
one persistent streaming session with per-turn explicit finalize, and
prints whether each turn cleanly produced an utterance-final event.
Run for real before trusting the Streamlit app's mic path."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path  # noqa: E402

from run_app import load_env  # noqa: E402
from ground_truth.qivora_call import TURNS, PRODUCT_NAME
from scripts.audio_utils import wav_file_to_pcm16_16k
from scripts.live_call_session import LiveCallSession

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "raw_lines")


def main():
    load_env(Path(__file__).resolve().parents[2] / ".env")
    session = LiveCallSession(diarize=True, language="en", format_=False, keyterms=[PRODUCT_NAME])
    t0 = time.time()
    session.connect()
    print(f"connected in {time.time()-t0:.2f}s")

    clean = 0
    for turn in TURNS:
        wav = os.path.join(RAW_DIR, f"turn{turn['turn_id']:02d}_{turn['speaker']}.wav")
        pcm = wav_file_to_pcm16_16k(wav)
        t1 = time.time()
        result = session.send_turn(pcm, realtime=True)
        dt = time.time() - t1
        status = "UTTERANCE_FINAL" if result.got_utterance_final else "NO utterance_final (check events)"
        if result.got_utterance_final:
            clean += 1
        print(f"turn {turn['turn_id']:02d} [{turn['speaker']:6s}] {dt:5.2f}s  {status}  speakers={result.speaker_ids}  eotc={result.end_of_turn_confidence}")
        print(f"   text: {result.final_text!r}")

    done = session.close()
    print(f"\nclosed. transcript.done={bool(done)}  total raw events={len(session.all_events)}")
    print(f"clean per-turn utterance-final: {clean}/{len(TURNS)}")


if __name__ == "__main__":
    main()
