"""
Hero experiment: replay Khalid's turn 4 (code switch + phone number with a
mid-dictation pause + spoken email) through the streaming API three times,
changing only smart_turn (0.5 / 0.7 / 0.9). Same audio, one variable.
Saves every raw event per threshold for the article.
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.stt_stream_client import stream_transcribe

WAV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "tmp", "turn04_khalid_16k.wav")


async def run_all():
    for threshold in (0.5, 0.7, 0.9):
        run_name = f"10_smartturn_{str(threshold).replace('.', '')}"
        print(f"\n=== smart_turn={threshold} ===")
        events = await stream_transcribe(
            WAV, run_name,
            diarize=False, keyterms=["Qivora Sync"],
            smart_turn=threshold, smart_turn_timeout=3000,
            language="en",
        )
        finals = [e for e in events if e.get("type") == "transcript.partial" and e.get("is_final")]
        for e in finals:
            kind = "UTTERANCE_FINAL" if e.get("speech_final") else "chunk_final"
            print(f"  t={e['t']:5.2f}s [{kind}] eotc={e.get('end_of_turn_confidence')}  text={e.get('text','')!r}")
        time.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_all())
