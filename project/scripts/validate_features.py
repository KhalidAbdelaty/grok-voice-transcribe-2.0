"""Real-API checks of the Transcribe 2.0 features the tutorial wants to show,
each graded PASS/FAIL from what the streaming socket actually returned.

    python project/scripts/validate_features.py

Every check streams the article's fixture audio (real Grok TTS lines, the
same ones the article's REST experiments used) to wss://api.x.ai/v1/stt at
real-time pace, the way a live call would, and scores the events:

  speakers      the full 3-voice call mix, diarize=true: each ground-truth
                speaker maps to one distinct speaker id
  keyterm       an invented product name ("Qivora Sync"), keyterm off vs on,
                clean and over a phone line
  smart_turn    a phone number with a real 1 s mid-number pause, at
                smart_turn 0.5 / 0.7 / 0.9: where the turn gets closed, and
                the end-of-turn confidence the server reported
  language      English -> Egyptian Arabic -> English inside one turn
  phone line    8 kHz G.711 mu-law, 300-3400 Hz, ~3% lost packets, streamed
                as encoding=mulaw&sample_rate=8000 (no resampling on our side)
  phone+email   digits and a spoken email read out loud, format off vs on
  fillers       filler_words false vs true on a line with "uh" and "um"

Writes results/19_feature_checks.json (summary + every raw event) and
results/19_feature_checks.md (the report).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from run_app import load_env  # noqa: E402
from ground_truth.qivora_call import EMAIL, PHONE_DIGITS, PRODUCT_NAME, TURNS  # noqa: E402
from scripts.audio_utils import wav_file_to_pcm16_16k  # noqa: E402
from scripts.live_call_session import stitch_partials  # noqa: E402
from scripts.phone_line import MULAW_SILENCE, PhoneLine  # noqa: E402
from scripts.tutorial_checks import (  # noqa: E402
    ARABIC, FILLERS, NUMBER_WORDS, digits_in, email_found, name_hits, spoken_email, tokens,
)

WS_BASE = "wss://api.x.ai/v1/stt"
MODEL = "grok-voice-transcribe-2.0"
RAW = ROOT / "audio" / "raw_lines"
MIX = ROOT / "audio" / "mixed" / "clean_master.wav"
TIMELINE = ROOT / "ground_truth" / "timeline.json"
RESULTS = ROOT / "results"
PARALLEL = 4


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------
async def stream(audio: bytes, *, encoding: str = "pcm", sample_rate: int = 16000, pace: float = 1.0,
                 chunk_ms: int = 100, tail_s: float = 2.0, **options) -> dict:
    """Stream `audio` like a live mic and return every event plus the
    utterance finals (text, words, end-of-turn confidence, arrival time)."""
    params = [("model", MODEL), ("sample_rate", str(sample_rate)), ("encoding", encoding),
              ("interim_results", "true")]
    for key, value in options.items():
        if value is None:
            continue
        for v in (value if isinstance(value, list) else [value]):
            params.append((key, "true" if v is True else "false" if v is False else str(v)))
    url = f"{WS_BASE}?{urlencode(params)}"
    bytes_per_sample = 1 if encoding == "mulaw" else 2
    silence = bytes([MULAW_SILENCE]) if encoding == "mulaw" else b"\x00\x00"
    step = int(sample_rate * bytes_per_sample * chunk_ms / 1000)
    payload = audio + silence * int(sample_rate * tail_s)
    events: list[dict] = []
    t0 = time.time()
    headers = {"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"}
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        first = json.loads(await ws.recv())
        events.append({"t": 0.0, **first})
        t0 = time.time()  # event times from the first audio byte, so they line up with word times

        async def send() -> None:
            for i in range(0, len(payload), step):
                await ws.send(payload[i:i + step])
                await asyncio.sleep(chunk_ms / 1000 / pace)
            await ws.send(json.dumps({"type": "audio.done"}))

        async def receive() -> None:
            async for raw in ws:
                evt = json.loads(raw)
                events.append({"t": round(time.time() - t0, 3), **evt})
                if evt.get("type") in ("transcript.done", "error"):
                    return

        sender = asyncio.create_task(send())
        await asyncio.wait_for(receive(), timeout=len(payload) / (sample_rate * bytes_per_sample) / pace + 30)
        sender.cancel()
    utterances = []
    for e in events:
        if e.get("type") == "transcript.partial" and e.get("speech_final"):
            words = e.get("words") or []
            last_end = max((w.get("end") or 0 for w in words), default=None)
            utterances.append({
                "t": e["t"], "text": (e.get("text") or "").strip(), "eotc": e.get("end_of_turn_confidence"),
                "words": words,
                # Silence between the last word and the server closing the turn.
                "closed_after_s": round(e["t"] - last_end, 2) if last_end else None,
            })
    text = " ".join(u["text"] for u in utterances if u["text"]) or stitch_partials(events)[0]
    errors = [e for e in events if e.get("type") == "error"]
    return {"params": dict(params) | {"keyterm": options.get("keyterm")}, "events": events,
            "utterances": utterances, "text": text, "words": [w for u in utterances for w in u["words"]],
            "error": errors[0].get("message", str(errors[0])) if errors else None}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def line(turn_id: int) -> bytes:
    turn = next(t for t in TURNS if t["turn_id"] == turn_id)
    return wav_file_to_pcm16_16k(str(RAW / f"turn{turn_id:02d}_{turn['speaker']}.wav"))


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(16000 * seconds)


def with_pause(pcm: bytes, pause_s: float) -> tuple[bytes, dict]:
    """Stretch the longest gap in the second half of `pcm` (the TTS [pause]
    between "five five five" and "one two three four") to `pause_s`."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    frame = 320  # 20 ms
    rms = np.sqrt(np.mean(x[: len(x) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    quiet = rms < max(0.004, 0.08 * float(np.median(rms[rms > 0.01])) if np.any(rms > 0.01) else 0.004)
    runs, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start, i))
            start = None
    speech = np.flatnonzero(~quiet)
    inner = [(a, b) for a, b in runs if a > speech[0] and b < speech[-1] and a > len(quiet) * 0.4]
    a, b = max(inner, key=lambda r: r[1] - r[0])
    mid = (a + b) // 2 * frame
    old_gap = (b - a) * frame / 16000
    extra = max(0.0, pause_s - old_gap)
    out = pcm[: mid * 2] + silence(extra) + pcm[mid * 2:]
    return out, {"pause_at_s": round(a * frame / 16000, 2), "tts_gap_s": round(old_gap, 2), "pause_s": pause_s}


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------
def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def normalize_for_wer(text: str) -> list[str]:
    toks = [NUMBER_WORDS.get(t, t) for t in tokens(ARABIC.sub(" ", text))]
    out = []
    for t in toks:
        out.extend(list(t) if t.isdigit() else [t])
    return out


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
async def check_speakers(run) -> dict:
    timeline = json.loads(TIMELINE.read_text(encoding="utf-8"))["turns"]
    res = await run("speakers", wav_file_to_pcm16_16k(str(MIX)), diarize=True, keyterm=PRODUCT_NAME,
                    smart_turn=0.7, smart_turn_timeout=3000)
    per_turn = []
    tally: dict[str, dict[int, int]] = {}
    for turn in timeline:
        lo, hi = turn["start_ms"] / 1000, turn["end_ms"] / 1000
        ids = [w.get("speaker") for w in res["words"]
               if w.get("speaker") is not None and lo <= (w["start"] + w["end"]) / 2 <= hi]
        counts = {i: ids.count(i) for i in set(ids)}
        for i, c in counts.items():
            tally.setdefault(turn["speaker"], {}).setdefault(i, 0)
            tally[turn["speaker"]][i] += c
        per_turn.append({"turn": turn["turn_id"], "speaker": turn["speaker"], "ids": counts,
                         "majority": max(counts, key=counts.get) if counts else None})
    mapping = {spk: max(c, key=c.get) for spk, c in tally.items() if c}
    total = sum(sum(c.values()) for c in tally.values())
    agree = sum(tally[spk].get(mapping[spk], 0) for spk in mapping)
    purity = agree / total if total else 0.0
    distinct = len(set(mapping.values())) == len(mapping) == 3
    turns_ok = sum(1 for t in per_turn if t["majority"] is not None and t["majority"] == mapping.get(t["speaker"]))
    ok = distinct and purity >= 0.9 and turns_ok == len(per_turn)
    return {"name": "Multiple voices, each labeled", "verdict": verdict(ok), "run": res,
            "summary": f"speaker ids {mapping}; {turns_ok}/{len(per_turn)} turns carry their speaker's id; "
                       f"{purity:.0%} of {total} words agree",
            "detail": {"mapping": mapping, "purity": round(purity, 3), "per_turn": per_turn}}


async def check_keyterm(run) -> dict:
    clean = line(2) + silence(0.6) + line(8)
    phone = PhoneLine(seed=11).to_mulaw_8k(clean)
    runs = await asyncio.gather(
        run("keyterm_off_clean", clean, language="en", smart_turn=0.7),
        run("keyterm_on_clean", clean, language="en", smart_turn=0.7, keyterm=PRODUCT_NAME),
        run("keyterm_off_phone", phone, encoding="mulaw", sample_rate=8000, language="en", smart_turn=0.7),
        run("keyterm_on_phone", phone, encoding="mulaw", sample_rate=8000, language="en", smart_turn=0.7,
            keyterm=PRODUCT_NAME),
    )
    hits = {r["label"]: name_hits(r["text"], PRODUCT_NAME) for r in runs}
    spelled = {}
    for r in runs:
        m = re.findall(r"\b(\w+\s+sync)\b", r["text"], re.I)
        spelled[r["label"]] = sorted(set(m))
    # The line says the name twice (turn 2 and turn 8).
    ok = hits["keyterm_on_clean"] == 2 and hits["keyterm_on_phone"] == 2
    return {"name": "Keyterm for an invented product name", "verdict": verdict(ok), "runs": runs,
            "summary": "\"Qivora Sync\" found (of 2): " + ", ".join(f"{k.replace('keyterm_', '')} {v}" for k, v in hits.items()),
            "detail": {"hits": hits, "spelled_as": spelled}}


SMART_TURN_TIMEOUT_MS = 3000


async def check_smart_turn(run) -> dict:
    """Two pauses, three thresholds (0.5 / 0.7 / 0.9), timeout 3 s:

      number      "010 555 [pause] 1234" - the pause at 1.0 s (a breath) and
                  3.5 s (past the timeout, so the turn must close there)
      mid-phrase  "failing to sync my files between [pause] my laptop" at
                  3.5 s - cut mid-phrase at Transcribe's own timestamp for
                  "between", where the model should be unsure

    Every close is graded against the documented rule: a turn closes either
    because the confidence reached the threshold (reported as that
    confidence) or because the timeout ran out (reported as 0.0, entry 08).
    Close delays are wall-clock and only shown, not graded."""
    thresholds = (0.5, 0.7, 0.9)
    number = wav_file_to_pcm16_16k(str(RAW / "turn04_khalid_seg2.wav"))
    probe = await run("smart_turn_word_times", line(2), language="en", smart_turn=0.7)
    cut = next((w["end"] for w in probe["words"] if w["text"].strip(",.").lower() == "between"), None)
    if cut is None:
        raise RuntimeError(f"no 'between' in the probe transcript: {probe['text']!r}")
    turn2 = line(2)
    at = int(cut * 16000) * 2 + 1600  # 50 ms past the word's end, so its tail isn't clipped
    clips = {
        "number_1.0s": with_pause(number, 1.0),
        "number_3.5s": with_pause(number, 3.5),
        "mid-phrase_3.5s": (turn2[:at] + silence(3.5) + turn2[at:], {"pause_at_s": round(cut, 2), "pause_s": 3.5}),
    }
    jobs = [(name, t) for name in clips for t in thresholds]
    runs = await asyncio.gather(*[
        run(f"smart_turn_{str(t).replace('.', '')}_{name}", clips[name][0], language="en", smart_turn=t,
            smart_turn_timeout=SMART_TURN_TIMEOUT_MS, keyterm=PRODUCT_NAME, filler_words=True, tail_s=4.0)
        for name, t in jobs
    ])
    rows, broken = [], []
    for (name, t), r in zip(jobs, runs):
        closes = []
        for u in r["utterances"]:
            eotc = u["eotc"] or 0.0
            by = "timeout" if eotc == 0.0 else "confidence" if eotc >= t else None
            closes.append({"text": u["text"], "eotc": u["eotc"], "closed_after_s": u["closed_after_s"], "by": by})
            if by is None:
                broken.append(f"{name} @ {t}: closed at eotc {eotc}, below the threshold and not a timeout")
        rows.append({"clip": name, "threshold": t, "turns": len(closes), "closes": closes})
    errors = [r["error"] for r in runs if r["error"]]
    breath_ok = all(row["turns"] == 1 for row in rows if row["clip"] == "number_1.0s")
    all_digits = all(PHONE_DIGITS in digits_in(r["text"]) for (name, _), r in zip(jobs, runs) if name.startswith("number"))
    ok = not errors and not broken and breath_ok and all_digits

    # Did the threshold change anything? Same clip, different threshold ->
    # different number of turns or different split points.
    outcomes = {}
    for row in rows:
        outcomes.setdefault(row["clip"], set()).add(tuple(c["text"] for c in row["closes"]))
    threshold_mattered = [clip for clip, seen in outcomes.items() if len(seen) > 1]
    confidences = sorted({c["eotc"] for row in rows for c in row["closes"] if c["eotc"]})
    timeouts = sum(1 for row in rows for c in row["closes"] if c["by"] == "timeout")
    split = {row["clip"]: row["turns"] for row in rows}
    summary = (f"1.0 s breath mid-number {'never' if breath_ok else 'sometimes'} ended the turn; a 3.5 s pause "
               f"mid-number split it into {split['number_3.5s']}, mid-phrase into {split['mid-phrase_3.5s']}; "
               f"every close was {'a timeout (0.0) or ' if timeouts else ''}a confidence close at "
               f"{confidences[0] if confidences else '-'}-{confidences[-1] if confidences else '-'}, so "
               + (f"the threshold changed the outcome on {', '.join(threshold_mattered)}" if threshold_mattered
                  else "0.5 / 0.7 / 0.9 gave identical turns on every clip")
               + (f"; rule broken: {'; '.join(broken)}" if broken else ""))
    return {"name": "End-of-turn confidence threshold", "verdict": verdict(ok), "runs": runs, "summary": summary,
            "detail": {"clips": {name: info for name, (_, info) in clips.items()}, "rows": rows, "rule_broken": broken,
                       "threshold_mattered": threshold_mattered}}


async def check_language(run) -> dict:
    pcm = line(4)
    runs = await asyncio.gather(
        run("language_auto", pcm, smart_turn=0.7, keyterm=PRODUCT_NAME),
        run("language_en", pcm, language="en", smart_turn=0.7, keyterm=PRODUCT_NAME),
    )
    rows = []
    for r in runs:
        text = r["text"]
        first_ar = ARABIC.search(text)
        last_ar = None
        for m in ARABIC.finditer(text):
            last_ar = m
        before = text[: first_ar.start()] if first_ar else text
        after = text[last_ar.end():] if last_ar else ""
        rows.append({"label": r["label"], "arabic_chars": len(ARABIC.findall(text)),
                     "english_before": "second" in before.lower(), "english_after": "email" in after.lower(),
                     "language_field": next((e.get("language") for e in r["events"] if e.get("language")), None)})
    ok = all(row["arabic_chars"] >= 10 and row["english_before"] and row["english_after"] for row in rows)
    return {"name": "Language switch halfway", "verdict": verdict(ok), "runs": runs,
            "summary": "; ".join(f"{row['label']}: {row['arabic_chars']} Arabic letters between English on both sides"
                                 if row["english_before"] and row["english_after"] else f"{row['label']}: switch not kept"
                                 for row in rows),
            "detail": {"rows": rows}}


async def check_phone_line(run) -> dict:
    clean = line(2) + silence(0.6) + line(4) + silence(0.6) + line(8)
    line_sim = PhoneLine(seed=7)
    phone = line_sim.to_mulaw_8k(clean)
    runs = await asyncio.gather(
        run("clean_16k", clean, language="en", smart_turn=0.7, keyterm=PRODUCT_NAME),
        run("phone_8k_mulaw", phone, encoding="mulaw", sample_rate=8000, language="en", smart_turn=0.7,
            keyterm=PRODUCT_NAME),
    )
    t4 = next(t for t in TURNS if t["turn_id"] == 4)
    ref_text = " ".join([next(t for t in TURNS if t["turn_id"] == 2)["plain_text"],
                         t4["segments"][0][1], t4["segments"][2][1].replace("[pause]", ""), t4["segments"][3][1],
                         next(t for t in TURNS if t["turn_id"] == 8)["plain_text"]])
    ref = [t for t in normalize_for_wer(ref_text) if t not in FILLERS]
    rates = {}
    for r in runs:
        hyp = [t for t in normalize_for_wer(r["text"]) if t not in FILLERS]
        rates[r["label"]] = round(wer(ref, hyp), 3)
    phone_run = runs[1]
    digits_ok = PHONE_DIGITS in digits_in(phone_run["text"])
    email_ok, _ = email_found(phone_run["text"])
    ok = rates["phone_8k_mulaw"] <= 0.15 and digits_ok and email_ok and phone_run["error"] is None
    stats = line_sim.stats()
    return {"name": "Flaky, phone-like audio", "verdict": verdict(ok), "runs": runs,
            "summary": f"word error rate {rates['clean_16k']:.0%} clean vs {rates['phone_8k_mulaw']:.0%} over the phone line "
                       f"({stats['dropout_pct']}% of {stats['packets']} packets lost); number "
                       f"{'kept' if digits_ok else 'lost'}, email {'kept' if email_ok else 'lost'}",
            "detail": {"wer": rates, "line": stats, "english_reference_words": len(ref)}}


async def check_phone_email(run) -> dict:
    tail = b"".join(wav_file_to_pcm16_16k(str(RAW / f"turn04_khalid_seg{i}.wav")) for i in (2, 3))
    runs = await asyncio.gather(
        run("readout_plain", tail, language="en", smart_turn=0.9, smart_turn_timeout=3000, keyterm=PRODUCT_NAME),
        run("readout_format", tail, language="en", smart_turn=0.9, smart_turn_timeout=3000, keyterm=PRODUCT_NAME,
            format=True),
    )
    rows = []
    for r in runs:
        found, missing = email_found(r["text"])
        rebuilt = spoken_email(r["text"])
        rows.append({"label": r["label"], "digits": digits_in(r["text"]), "digits_ok": PHONE_DIGITS in digits_in(r["text"]),
                     "email_ok": found and rebuilt == EMAIL, "email_missing": missing, "email_rebuilt": rebuilt,
                     "text": r["text"]})
    ok = all(row["digits_ok"] and row["email_ok"] for row in rows)
    plain, fmt = rows
    return {"name": "Phone number and email read out loud", "verdict": verdict(ok), "runs": runs,
            "summary": f"digits {PHONE_DIGITS} in order: plain {verdict(plain['digits_ok'])}, format {verdict(fmt['digits_ok'])}; "
                       f"email rebuilt as {plain['email_rebuilt']} (plain) / {fmt['email_rebuilt']} (format), "
                       f"expected {EMAIL}",
            "detail": {"rows": rows, "expected_email": EMAIL}}


async def check_fillers(run) -> dict:
    pcm = line(2)
    runs = await asyncio.gather(
        run("fillers_off", pcm, language="en", smart_turn=0.7, keyterm=PRODUCT_NAME, filler_words=False),
        run("fillers_on", pcm, language="en", smart_turn=0.7, keyterm=PRODUCT_NAME, filler_words=True),
    )
    counts = {r["label"]: [t for t in tokens(r["text"]) if t in FILLERS] for r in runs}
    ok = len(counts["fillers_on"]) >= 2 and len(counts["fillers_off"]) < len(counts["fillers_on"])
    return {"name": "Filler words (um, uh)", "verdict": verdict(ok), "runs": runs,
            "summary": f"filler_words=false: {counts['fillers_off'] or 'none'}; filler_words=true: {counts['fillers_on'] or 'none'}",
            "detail": {"fillers": counts}}


# ---------------------------------------------------------------------------
async def main() -> None:
    load_env(ROOT.parent / ".env")
    if not os.environ.get("XAI_API_KEY"):
        sys.exit("XAI_API_KEY is not set (put it in .env)")
    gate = asyncio.Semaphore(PARALLEL)

    async def run(label: str, audio: bytes, **kw) -> dict:
        async with gate:
            print(f"  streaming {label} ...", flush=True)
            for attempt in (1, 2):
                try:
                    res = await stream(audio, **kw)
                    break
                except Exception as exc:  # noqa: BLE001 - one retry on a dropped socket
                    if attempt == 2:
                        res = {"params": kw, "events": [], "utterances": [], "text": "", "words": [], "error": repr(exc)}
            res["label"] = label
            return res

    t0 = time.time()
    checks = await asyncio.gather(
        check_speakers(run), check_keyterm(run), check_smart_turn(run), check_language(run),
        check_phone_line(run), check_phone_email(run), check_fillers(run),
    )
    elapsed = time.time() - t0
    write_report(checks, elapsed)
    print()
    for c in checks:
        print(f"{c['verdict']}  {c['name']}: {ascii(c['summary'])[1:-1]}")
    print(f"\n{sum(c['verdict'] == 'PASS' for c in checks)}/{len(checks)} PASS in {elapsed:.0f}s "
          f"-> results/19_feature_checks.md")
    sys.stdout.flush()


def write_report(checks: list[dict], elapsed: float) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "19_feature_checks.json").write_text(json.dumps({
        "model": MODEL, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed, 1), "checks": checks,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = ["# 19 - Tutorial feature checks (real API, streaming)", "",
          f"`{MODEL}` over `wss://api.x.ai/v1/stt`, fixture audio streamed at real-time pace. "
          f"Run {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, {elapsed:.0f}s. "
          "Raw events: `19_feature_checks.json`. Script: `scripts/validate_features.py`.", "",
          "| Check | Result | What came back |", "|---|---|---|"]
    for c in checks:
        md.append(f"| {c['name']} | **{c['verdict']}** | {c['summary']} |")
    for c in checks:
        md += ["", f"## {c['name']} - {c['verdict']}", ""]
        for r in c.get("runs") or [c["run"]]:
            opts = {k: v for k, v in r["params"].items() if k not in ("model", "interim_results") and v is not None}
            md.append(f"- `{r['label']}` ({', '.join(f'{k}={v}' for k, v in opts.items())})"
                      + (f" - error: {r['error']}" if r["error"] else ""))
            if c["name"].startswith("End-of-turn"):
                for u in r["utterances"]:
                    md.append(f"  - closed {u['closed_after_s']}s after the last word, eotc {u['eotc']}: "
                              f"\u201c{u['text']}\u201d")
            else:
                md.append(f"  - \u201c{r['text']}\u201d")
        if c["name"].startswith("Multiple voices"):
            md += ["", "| Turn | Speaker | Ids heard (words) | Majority |", "|---|---|---|---|"]
            for t in c["detail"]["per_turn"]:
                md.append(f"| {t['turn']} | {t['speaker']} | {t['ids']} | {t['majority']} |")
    (RESULTS / "19_feature_checks.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
