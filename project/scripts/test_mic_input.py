"""Offline checks for scripts/mic_input.py speech detection, no API calls.

The bug this guards against (experiment_log.md entry 16): with browser
auto-gain on, room noise sat above the old fixed first threshold, the floor
never learned, and every frame counted as speech - so the engine thought
the caller never stopped talking and cancelled every reply.

    python project/scripts/test_mic_input.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.audio_utils import wav_file_to_pcm16_16k  # noqa: E402
from scripts.mic_input import MicInput  # noqa: E402

FRAME = 320  # 20 ms at 16 kHz
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "raw_lines")
rng = np.random.default_rng(3)
failures = 0


class Clock:
    """Drive MicInput's time.time() so 30 s of audio checks in milliseconds."""

    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


clock = Clock()
time.time = clock  # MicInput only reads time.time()


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        failures += 1


def frames(signal: np.ndarray):
    for i in range(0, len(signal) - FRAME + 1, FRAME):
        yield signal[i : i + FRAME]


def feed(mic: MicInput, signal: np.ndarray) -> list[bool]:
    """Feed float audio; return, per frame, whether it was classified speech."""
    flags = []
    for f in frames(signal):
        before = mic.last_speech_at
        mic.feed_pcm((np.clip(f, -1, 1) * 32767).astype(np.int16).tobytes())
        clock.t += 0.02
        flags.append(mic.last_speech_at != before)
    return flags


def noise(rms: float, seconds: float) -> np.ndarray:
    return rng.normal(0, rms, int(16000 * seconds)).astype(np.float32)


def sentence(noise_rms: float, speech_rms: float, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Syllables (180 ms, shaped) with 40 ms dips back to the room and no
    pauses. The second array is the syllable number per frame (-1 in dips)."""
    out, syllable = [], []
    t, k = 0.0, 0
    while t < seconds:
        syl = rng.normal(0, speech_rms, int(16000 * 0.18)) * np.hanning(int(16000 * 0.18)) * 1.6
        out.append(syl + noise(noise_rms, 0.18))
        syllable += [k] * (len(syl) // FRAME)
        out.append(noise(noise_rms, 0.04))
        syllable += [-1] * 2
        t += 0.22
        k += 1
    return np.concatenate(out).astype(np.float32), np.array(syllable)


def longest_gap_s(flags: np.ndarray) -> float:
    """Longest run of non-speech frames between the first and last speech
    frame - what the engine would read as the caller pausing."""
    idx = np.flatnonzero(flags)
    if idx.size < 2:
        return float("inf")
    return float(np.diff(idx).max() - 1) * 0.02


for room in (0.005, 0.02, 0.05):
    mic = MicInput()
    flags = feed(mic, noise(room, 4.0))
    floor = mic.noise_floor
    check(f"room {room}: floor tracks the room", 0.6 * room < floor < 1.5 * room, f"floor={floor:.4f}")
    check(f"room {room}: silence is not speech", sum(flags[20:]) == 0, f"{sum(flags[20:])} speech frames")
    check(f"room {room}: caller reads as silent", mic.silence_s() > 3.0 or mic.last_speech_at == 0.0)

    speech_rms = max(0.08, room * 6)
    audio, syllable = sentence(room, speech_rms, 10.0)
    flags = np.array(feed(mic, audio))
    n = min(len(flags), len(syllable))
    flags, syllable = flags[:n], syllable[:n]
    heard = {s for s, f in zip(syllable, flags) if f and s >= 0}
    total = set(syllable[syllable >= 0])
    late = {s for s in total if s >= max(total) - 9}
    check(f"room {room}: every syllable heard", len(heard) / len(total) > 0.97, f"{len(heard)}/{len(total)}")
    check(f"room {room}: still heard 10 s into a sentence", late <= heard, f"floor={mic.noise_floor:.4f}")
    gap = longest_gap_s(flags)
    check(f"room {room}: a sentence never looks like a pause", gap < 0.35, f"longest gap {gap:.2f}s")
    feed(mic, noise(room, 1.0))
    check(f"room {room}: silence after the sentence", mic.silence_s() > 0.8, f"silence_s={mic.silence_s():.2f}")

# Echo: the agent's own voice leaking in must not count as barging in; the
# caller talking over it must. The coupling (how much of the playback reaches
# the mic) is measured while the agent plays, so the bar fits the setup:
# open speakers (30% leak), the browser after echo cancellation (8%), and a
# headset (2%) where the caller speaks at -28 dBFS (RMS 0.037) - the real
# GM301 call in which the old fixed 0.8 x playback bar never let them in.
agent, _ = sentence(0.0, 0.12, 6.0)
cases = (
    ("local", "open speakers", 0.3, 0.2),
    ("browser", "after echo cancellation", 0.08, 0.08),
    ("local", "headset", 0.02, 0.037),
)
for source, setup, leak, caller_rms in cases:
    mic = MicInput(source=source)
    feed(mic, noise(0.002, 2.0))
    mic.set_echo_reference(0.12, playing=True)
    feed(mic, agent * leak + noise(0.002, len(agent) / 16000)[: len(agent)])
    stats = mic.stats()
    check(f"{setup}: coupling measured near the real leak", stats["coupling_measured"] and stats["coupling"] < leak * 2.5 + 0.01,
          f"coupling={stats['coupling']} leak={leak}")
    check(f"{setup}: {leak:.0%} echo leak is not barge-in", mic._loud_start is None, f"barge_thr={stats['barge_thr']}")
    caller, _ = sentence(0.0, caller_rms, 1.0)
    feed(mic, agent[: len(caller)] * leak + caller)
    check(f"{setup}: caller at {caller_rms} RMS over the echo is barge-in", mic._loud_start is not None,
          f"barge_thr={mic.stats()['barge_thr']}")

# Quick words ("Sure, one second" at a headset level): each syllable is above
# the barge bar for ~160 ms with ~160 ms gaps, so no single loud run reaches
# 0.2 s - the loud time within the last 0.5 s does, and the agent pauses.
mic = MicInput(source="local")
feed(mic, noise(0.002, 2.0))
mic.set_echo_reference(0.1, playing=True)
feed(mic, agent[: 16000 * 2] * 0.02 + noise(0.002, 2.0))  # headset leak: coupling learned first
burst = rng.normal(0, 0.04, int(16000 * 0.16)).astype(np.float32)
quick = np.concatenate([burst, noise(0.002, 0.16), burst, noise(0.002, 0.16)])
best_run = best_recent = 0.0
for f in frames(quick):
    mic.feed_pcm((np.clip(f, -1, 1) * 32767).astype(np.int16).tobytes())
    clock.t += 0.02
    best_run, best_recent = max(best_run, mic.loud_run_s()), max(best_recent, mic.loud_recent_s())
check("quick words: loud time in the window reaches the 0.2 s pause bar", best_recent >= 0.2 and best_run < 0.2,
      f"longest run {best_run:.2f}s, in window {best_recent:.2f}s")

# The old failure, directly: AGC-level room noise from the very first frame.
mic = MicInput()
flags = feed(mic, noise(0.02, 3.0))
check("AGC-level room noise from frame one is not 'always talking'", sum(flags) == 0 and not mic.heard_speech,
      f"{sum(flags)} speech frames")

# A real recorded line plus room noise: heard while talking, silent after.
line = wav_file_to_pcm16_16k(os.path.join(RAW, "turn04_khalid.wav"))
speech = np.frombuffer(line, dtype=np.int16).astype(np.float32) / 32768.0
mic = MicInput()
feed(mic, noise(0.02, 2.0))
flags = feed(mic, speech + noise(0.02, len(speech) / 16000)[: len(speech)])
check("real line: speech detected", mic.heard_speech and sum(flags) > len(flags) * 0.3, f"{sum(flags)}/{len(flags)} frames")
feed(mic, noise(0.02, 1.0))
check("real line: silence detected after it", mic.silence_s() > 0.6, f"silence_s={mic.silence_s():.2f}")

# Echo tail: quiet_for_s() counts from the last above-threshold frame in any
# gate mode (the agent's leaking voice included), so the engine can reopen
# the mic only once a decaying tail has actually died away.
mic = MicInput(source="local")
feed(mic, noise(0.005, 2.0))
tail = sentence(0.005, 0.1, 0.6)[0] * np.linspace(1.0, 0.0, int(16000 * 0.66))[: len(sentence(0.005, 0.1, 0.6)[0])]
feed(mic, tail)
check("quiet_for_s is ~0 right after the echo tail", mic.quiet_for_s() < 0.25, f"{mic.quiet_for_s():.2f}s")
feed(mic, noise(0.005, 0.2))
check("quiet_for_s passes 0.15 s once the tail has died away", mic.quiet_for_s() >= 0.15, f"{mic.quiet_for_s():.2f}s")
mic.open_gate(type("S", (), {"feed_live_audio": lambda self, b: None})())
check("opening the gate does not reset quiet_for_s", mic.quiet_for_s() >= 0.15, f"{mic.quiet_for_s():.2f}s")

# Boost: a quiet mic gets louder without clipping.
mic = MicInput(boost=3.0)
feed(mic, noise(0.3, 0.5))
check("boost soft-limits instead of clipping", mic.rms < 0.9, f"rms={mic.rms:.3f}")

print("\nALL PASS" if failures == 0 else f"\n{failures} FAILURES")
sys.exit(1 if failures else 0)
