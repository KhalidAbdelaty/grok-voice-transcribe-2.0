"""Headless, real-API run of the whole live call engine (no browser).

Cached caller lines (the sirius stand-in for Khalid) play into the engine's
microphone input at real-time pace, with silence in between, exactly as a
WebRTC mic would deliver them. Everything else is live: Transcribe 2.0,
the streamed brain, streaming TTS, and the agent audio going back through
the same STT session.

    python project/scripts/validate_engine.py latency   [model] [priority 0|1]
    python project/scripts/validate_engine.py barge-in
    python project/scripts/validate_engine.py backchannel   ("Yeah." under Maya: must not cut)
    python project/scripts/validate_engine.py noise-burst   (loud noise, no words: pause, then resume)
    ... add --room-noise 0.02 (AGC-like room noise on every mic frame)
    ... add --echo 0.3 (the agent's playback leaks into the mic, laptop speakers)
    ... add --path local (the "This computer" echo rules instead of the browser's)
    ... add --echo-delay 0.35 (the leak arrives late, like speakers + FxSound)
    ... add --caller-wait 3 (seconds the caller stays quiet before each line)

`latency` prints, per agent reply, how long after the caller stopped
talking the first agent audio started, and where that time went.
`barge-in` talks over Maya 1.5 s into her first reply and checks that her
playback stops, her turn closes with only the part that was heard, and the
caller's words land in a turn of their own.
"""
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(ROOT))

from run_app import load_env  # noqa: E402
from ground_truth.qivora_call import PRODUCT_NAME, VOICES  # noqa: E402
from scripts.agent_brain import FAST_MODEL  # noqa: E402
from scripts.audio_utils import wav_file_to_pcm16_16k  # noqa: E402
from scripts.call_engine import CallEngine, EngineSettings  # noqa: E402

RAW = ROOT / "project" / "audio" / "raw_lines"
FRAME = 640  # 20 ms at 16 kHz mono PCM16, what WebRTC delivers


class EchoPlayback:
    """Stands in for the speakers: keeps what the agent "plays" so the fake
    mic can leak it back in at real-time pace, like laptop speakers do.
    `delay_s` is a constant delay line on top (output buffers, FxSound-style
    audio processing): what leaks into the mic lags what was pushed. It has
    no `latency` attribute on purpose - the engine doesn't know this delay,
    just as it can't know FxSound's."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.lock = threading.Lock()
        self.buf = bytearray()
        self.line = bytearray(int(16000 * delay_s) * 2)

    def push(self, pcm: bytes) -> None:
        with self.lock:
            self.buf.extend(pcm)

    def clear(self) -> None:
        with self.lock:
            self.buf = bytearray()

    def take(self, n: int) -> bytes:
        with self.lock:
            fresh = bytes(self.buf[:n]).ljust(n, b"\0")
            del self.buf[:n]
            self.line.extend(fresh)
            out = bytes(self.line[:n])
            del self.line[:n]
        return out


class FakeMic(threading.Thread):
    """Real-time 20 ms frames into the engine's mic input. `room_noise` adds
    Gaussian noise at that RMS to every frame (what a browser with automatic
    gain control delivers in a normal room); `echo` mixes in the agent's own
    playback at that gain."""

    def __init__(self, engine: CallEngine, room_noise: float = 0.0, echo: float = 0.0,
                 playback: EchoPlayback | None = None) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        self.pending: deque[bytes] = deque()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.room_noise = room_noise
        self.echo = echo
        self.playback = playback
        self.rng = np.random.default_rng(7)

    def _mix(self, frame: bytes) -> bytes:
        if not self.room_noise and not (self.echo and self.playback):
            return frame
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if self.room_noise:
            x += self.rng.normal(0.0, self.room_noise * 32768.0, x.size)
        if self.echo and self.playback is not None:
            x += np.frombuffer(self.playback.take(FRAME), dtype=np.int16).astype(np.float32) * self.echo
        return np.clip(x, -32768, 32767).astype(np.int16).tobytes()

    def say(self, pcm: bytes) -> None:
        if CALLER_VOLUME != 1.0:
            pcm = (np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * CALLER_VOLUME).astype(np.int16).tobytes()
        with self.lock:
            for i in range(0, len(pcm), FRAME):
                self.pending.append(pcm[i : i + FRAME].ljust(FRAME, b"\0"))

    @property
    def talking(self) -> bool:
        with self.lock:
            return bool(self.pending)

    def run(self) -> None:
        silence = bytes(FRAME)
        next_t = time.monotonic()
        while not self.stop.is_set():
            with self.lock:
                frame = self.pending.popleft() if self.pending else silence
            self.engine.mic.feed_pcm(self._mix(frame))
            self.engine.touch()
            next_t += 0.02
            time.sleep(max(0.0, next_t - time.monotonic()))


def line(name: str) -> bytes:
    return wav_file_to_pcm16_16k(str(RAW / name))


def wait_for(pred, timeout: float, what: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for {what}")


ROOM_NOISE = 0.0
ECHO = 0.0
ECHO_DELAY = 0.0
CALLER_WAIT = 0.6
CALLER_VOLUME = 1.0
AUDIO_PATH = "browser"
PHONE = False


def make_engine(model: str, priority: bool) -> tuple[CallEngine, FakeMic]:
    playback = EchoPlayback(ECHO_DELAY) if ECHO else None
    engine = CallEngine(
        settings=EngineSettings(model=model, priority=priority, audio_path=AUDIO_PATH, phone_line=PHONE),
        voices=VOICES,
        keyterms=[PRODUCT_NAME],
        playback=playback,
    )
    mic = FakeMic(engine, room_noise=ROOM_NOISE, echo=ECHO, playback=playback)
    engine.start()
    mic.start()
    return engine, mic


def phase(engine: CallEngine) -> str:
    return engine.snapshot()["phase"]


def print_report(engine: CallEngine) -> None:
    snap = engine.snapshot()
    print("\n--- turns ---")
    for t in snap["turns"]:
        if t["speaker"] == "system":
            print(f"  [system] {t['text']}")
            continue
        flag = " (INTERRUPTED)" if t.get("interrupted") else ""
        print(f"  {t['speaker']:6s}{flag}: heard={ascii(t['text'])}")
        if t.get("said"):
            print(f"          said ={ascii(t['said'])}")
    print("\n--- latency per agent reply ---")
    for r in snap["record"]:
        if r["speaker"] == "khalid":
            print(f"  caller turn {r['turn']}: Smart Turn closed it {r['smart_turn_ms']}ms after speech, eotc={r['end_of_turn_confidence']}, speaker ids={r['diarized_speaker_ids']}")
            continue
        print(
            f"  {r['speaker']:6s} turn {r['turn']}: first audio {r['response_latency_ms']}ms after the caller stopped | "
            f"llm first text {r['llm_first_text_ms']}ms, tts first audio {r['tts_first_audio_ms']}ms, "
            f"speculative hit={r['speculative_hit']} ({r['speculative_attempts']} guesses), tier={r['service_tier']}, model={r['model']}, "
            f"interrupted={r['interrupted']}, speaker ids={r['diarized_speaker_ids']}"
        )
    print(f"\nstreamed audio: {snap['audio_seconds']:.1f}s, phase at end: {snap['phase']}, error: {snap['error']}")
    agent_ids = {i for r in snap["record"] if r["speaker"] != "khalid" for i in r["diarized_speaker_ids"]}
    phantoms = [
        r for r in snap["record"]
        if r["speaker"] == "khalid" and r["diarized_speaker_ids"] and set(r["diarized_speaker_ids"]) <= agent_ids
    ]
    ignored = snap.get("phantoms_ignored") or []
    print(f"caller turns diarized only as an agent voice (phantoms): {len(phantoms)}"
          + "".join(f"\n  turn {r['turn']} ids={r['diarized_speaker_ids']} heard={ascii(r['heard_by_transcribe'])}" for r in phantoms))
    print(f"agent-voice turns the engine ignored: {len(ignored)}" + "".join(f"\n  {ascii(t)}" for t in ignored))


def agent_turns(engine: CallEngine) -> int:
    return sum(1 for r in engine.snapshot()["record"] if r["speaker"] != "khalid")


def run_latency(model: str, priority: bool) -> None:
    engine, mic = make_engine(model, priority)
    done_phases = ("ended", "ending", "error")
    try:
        for name in ("turn02_khalid.wav", "turn04_khalid.wav", "turn09_khalid.wav"):
            # Like a real caller: wait until the agent has answered and it is
            # your turn again, then speak the next line.
            wait_for(lambda: phase(engine) in ("listening", *done_phases) and not mic.talking, 90, "the caller's turn")
            if phase(engine) != "listening":
                break
            n = agent_turns(engine)
            time.sleep(CALLER_WAIT)
            mic.say(line(name))
            wait_for(lambda: agent_turns(engine) > n or phase(engine) in done_phases, 120, "the agent's reply")
        time.sleep(1.0)
    finally:
        mic.stop.set()
        print_report(engine)
        engine.end()


def run_barge_in() -> None:
    engine, mic = make_engine(FAST_MODEL, True)
    try:
        wait_for(lambda: phase(engine) == "listening", 30, "listening")
        time.sleep(0.6)
        mic.say(line("turn02_khalid.wav"))
        wait_for(lambda: phase(engine) == "speaking", 60, "Maya to start speaking")
        time.sleep(1.5)
        print(">>> caller talks over Maya now")
        talk_at = time.time()
        mic.say(line("turn04_khalid.wav"))
        wait_for(lambda: engine.snapshot()["paused"] or engine.snapshot()["interrupted"] or phase(engine) != "speaking",
                 10, "Maya to go quiet")
        quiet_at = time.time()
        wait_for(lambda: engine.snapshot()["interrupted"] or phase(engine) != "speaking", 10, "the barge-in")
        cut_at = time.time()
        print(f">>> Maya went quiet {quiet_at - talk_at:.2f}s after the caller started talking, turn cut at "
              f"{cut_at - talk_at:.2f}s (reason: {engine.barge_reason}, coupling {engine.mic.stats()['coupling_db']} dB)")
        wait_for(lambda: phase(engine) == "listening", 30, "the caller's turn after the barge-in")
        print(f">>> Maya's turn closed, caller's turn opened {time.time() - cut_at:.2f}s after the cut")
        wait_for(lambda: phase(engine) in ("thinking", "speaking"), 60, "the reply to the interruption")
        wait_for(lambda: phase(engine) in ("listening", "ended", "ending", "error"), 90, "that reply to finish")
        time.sleep(1.0)
    finally:
        mic.stop.set()
        print_report(engine)
        snap = engine.snapshot()
        interrupted = [r for r in snap["record"] if r.get("interrupted")]
        callers = [r for r in snap["record"] if r["speaker"] == "khalid"]
        # The interrupting line dictates 010 555 1234; Transcribe spaces it
        # either "0 1 0 5 5 5" or "010 555", so check the digits only.
        digits = "".join(ch for ch in (callers[1]["heard_by_transcribe"] if len(callers) > 1 else "") if ch.isdigit())
        ok = bool(interrupted) and len(callers) >= 2 and "0105551234" in digits
        print("\nBARGE-IN", "PASS" if ok else "CHECK OUTPUT")
        engine.end()


def run_overlay(kind: str) -> None:
    """Something that must NOT cut the agent, played 1.5 s into Maya's first
    reply: a backchannel ("Yeah." in the caller's voice) or a burst of loud
    noise with no words. Expect at most a short pause, then the reply plays to
    the end and is not marked interrupted."""
    from scripts.tts_stream import synthesize_pcm_rest

    if kind == "backchannel":
        overlay = synthesize_pcm_rest("Yeah.", VOICES["khalid"])
    else:
        rng = np.random.default_rng(11)
        overlay = (rng.normal(0, 0.15, int(16000 * 0.7)) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
    engine, mic = make_engine(FAST_MODEL, True)
    try:
        wait_for(lambda: phase(engine) == "listening", 30, "listening")
        time.sleep(CALLER_WAIT)
        mic.say(line("turn02_khalid.wav"))
        wait_for(lambda: phase(engine) == "speaking", 60, "Maya to start speaking")
        time.sleep(1.5)
        print(f">>> {kind} over Maya now")
        mic.say(overlay)
        wait_for(lambda: phase(engine) in ("listening", "ended", "ending", "error"), 90, "Maya's reply to finish")
        time.sleep(1.0)
    finally:
        mic.stop.set()
        print_report(engine)
        snap = engine.snapshot()
        agent = [r for r in snap["record"] if r["speaker"] != "khalid"]
        ok = bool(agent) and not agent[0]["interrupted"]
        print(f"\n{kind.upper()}: agent pauses={agent[0]['barge_pauses'] if agent else '?'}, "
              f"interrupted={agent[0]['interrupted'] if agent else '?'} ->", "PASS" if ok else "FAIL")
        engine.end()


def run_phantom() -> None:
    """The GM301 failure (call_20260923_031600, turns 10/15/18): right after
    Maya finishes, a faint leak of her last words reaches an otherwise silent
    mic. Played at 2% (below the speech threshold, like the headset leak) and
    at 10% (above it). Neither may become a caller turn."""
    from scripts.tts_stream import synthesize_pcm_rest

    engine, mic = make_engine(FAST_MODEL, True)
    try:
        wait_for(lambda: phase(engine) == "listening", 30, "listening")
        time.sleep(CALLER_WAIT)
        mic.say(line("turn02_khalid.wav"))
        results = []
        for level in (0.02, 0.10):
            n = agent_turns(engine)
            wait_for(lambda: agent_turns(engine) > n and phase(engine) == "listening", 90, "Maya's reply to finish")
            said = engine.prev_agent_text
            tail = " ".join(said.split()[-5:])
            leak = synthesize_pcm_rest(tail, VOICES["maya"])
            callers_before = sum(1 for r in engine.snapshot()["record"] if r["speaker"] == "khalid")
            ignored_before = len(engine.snapshot()["phantoms_ignored"])
            pcm = (np.frombuffer(leak, dtype=np.int16).astype(np.float32) * level).astype(np.int16).tobytes()
            print(f">>> leaking {ascii(tail)} at {level:.0%} into a silent mic")
            time.sleep(0.2)
            with mic.lock:
                for i in range(0, len(pcm), FRAME):
                    mic.pending.append(pcm[i : i + FRAME].ljust(FRAME, b"\0"))
            time.sleep(7.0)
            snap = engine.snapshot()
            new_callers = [r for r in snap["record"] if r["speaker"] == "khalid"][callers_before:]
            results.append((level, new_callers, snap["phantoms_ignored"][ignored_before:]))
            if level == 0.02:
                # Hand the floor back with a real line so Maya replies again.
                mic.say(line("turn04_khalid_seg2.wav"))
        time.sleep(1.0)
    finally:
        mic.stop.set()
        print_report(engine)
        ok = len(results) == 2
        for level, new_callers, ignored in results:
            print(f"\n{level:.0%} leak: caller turns {[ascii(r['heard_by_transcribe']) for r in new_callers]}, "
                  f"ignored {[ascii(t) for t in ignored]}")
            ok &= not new_callers
        print("\nPHANTOM", "PASS" if ok else "FAIL")
        engine.end()


def run_arabic() -> None:
    """The caller speaks Egyptian Arabic; the agent should answer in it."""
    engine, mic = make_engine(FAST_MODEL, True)
    try:
        wait_for(lambda: phase(engine) == "listening", 30, "listening")
        time.sleep(CALLER_WAIT)
        mic.say(line("turn02_khalid.wav"))
        wait_for(lambda: agent_turns(engine) >= 1 and phase(engine) == "listening", 90, "Maya's first reply")
        time.sleep(CALLER_WAIT)
        mic.say(line("turn04_khalid_seg1.wav"))
        wait_for(lambda: agent_turns(engine) >= 2 and phase(engine) in ("listening", "ended", "ending"), 90,
                 "the reply to the Arabic line")
        time.sleep(1.0)
    finally:
        mic.stop.set()
        print_report(engine)
        agents = [r for r in engine.snapshot()["record"] if r["speaker"] != "khalid"]
        last = agents[-1] if len(agents) >= 2 else None
        arabic = bool(last) and any("\u0600" <= ch <= "\u06ff" for ch in (last["heard_by_transcribe"] or ""))
        print(f"\nreply language={last['reply_language'] if last else None}, Arabic script in what was heard={arabic}")
        print("ARABIC", "PASS" if last and last["reply_language"] == "ar-EG" and arabic else "FAIL")
        engine.end()


def run_toggle() -> None:
    """Phone line switched on and back off in the middle of one call. Each
    caller line must land on the wire that was active when it was spoken."""
    engine, mic = make_engine(FAST_MODEL, True)
    plan = [("turn02_khalid.wav", None), ("turn04_khalid_seg2.wav", True), ("turn09_khalid.wav", False)]
    try:
        for name, phone in plan:
            wait_for(lambda: phase(engine) == "listening" and not mic.talking, 90, "the caller's turn")
            if phone is not None:
                engine.set_phone_line(phone)
                t0 = time.time()
                wait_for(lambda: engine.snapshot()["phone_on"] == phone and engine.snapshot()["phone_switch_to"] is None,
                         20, "the phone line switch")
                print(f">>> phone line {'on' if phone else 'off'} after {time.time() - t0:.2f}s "
                      f"(wire {engine.session.wire})")
            n = agent_turns(engine)
            time.sleep(CALLER_WAIT)
            mic.say(line(name))
            wait_for(lambda: agent_turns(engine) > n or phase(engine) in ("ended", "ending", "error"), 90,
                     "the agent's reply")
        time.sleep(1.0)
    finally:
        mic.stop.set()
        print_report(engine)
        callers = [r for r in engine.snapshot()["record"] if r["speaker"] == "khalid"]
        sources = [r["source"] for r in callers]
        print(f"\ncaller turn sources: {sources}")
        ok = sources[:1] == ["mic"] and "phone" in sources and sources[-1] == "mic" and not engine.snapshot()["error"]
        print("TOGGLE", "PASS" if ok else "FAIL")
        engine.end()


def _pop_option(args: list[str], name: str) -> float:
    if name in args:
        i = args.index(name)
        value = float(args[i + 1])
        del args[i : i + 2]
        return value
    return 0.0


if __name__ == "__main__":
    load_env(ROOT / ".env")
    ROOM_NOISE = _pop_option(sys.argv, "--room-noise")
    ECHO = _pop_option(sys.argv, "--echo")
    ECHO_DELAY = _pop_option(sys.argv, "--echo-delay")
    CALLER_WAIT = _pop_option(sys.argv, "--caller-wait") or 0.6
    CALLER_VOLUME = _pop_option(sys.argv, "--caller-volume") or 1.0
    if "--phone" in sys.argv:
        PHONE = True
        sys.argv.remove("--phone")
        print("(phone line: Transcribe on encoding=mulaw&sample_rate=8000, narrowband playback)")
    if "--path" in sys.argv:
        i = sys.argv.index("--path")
        AUDIO_PATH = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if ROOM_NOISE or ECHO or AUDIO_PATH != "browser" or CALLER_VOLUME != 1.0:
        print(f"(fake mic: room noise RMS {ROOM_NOISE}, speaker echo gain {ECHO}, echo delay {ECHO_DELAY}s, "
              f"audio path {AUDIO_PATH}, caller volume {CALLER_VOLUME}x)")
    mode = sys.argv[1] if len(sys.argv) > 1 else "latency"
    if mode == "barge-in":
        run_barge_in()
    elif mode in ("backchannel", "noise-burst"):
        run_overlay(mode)
    elif mode == "phantom":
        run_phantom()
    elif mode == "arabic":
        run_arabic()
    elif mode == "toggle":
        run_toggle()
    else:
        model = sys.argv[2] if len(sys.argv) > 2 else FAST_MODEL
        priority = (sys.argv[3] != "0") if len(sys.argv) > 3 else True
        run_latency(model, priority)
    sys.stdout.flush()
    # Daemon threads (socket loops) would otherwise log teardown noise.
    os._exit(0)
