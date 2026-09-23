"""What a live call has shown so far of the tutorial's feature list, read off
the call record. The same text checks score the offline fixture runs in
validate_features.py, so the app and the report agree on what counts.
"""
from __future__ import annotations

import re

ARABIC = re.compile(r"[\u0600-\u06FF]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
NUMBER_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
FILLERS = {"um", "uh", "umm", "uhm", "er", "erm", "hmm"}
SPEAKER_LABELS = {"khalid": "You", "maya": "Maya", "nadia": "Nadia"}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def digits_in(text: str) -> str:
    out = []
    for tok in re.findall(r"[a-z]+|\d+", text.lower()):
        if tok.isdigit():
            out.append(tok)
        elif tok in NUMBER_WORDS:
            out.append(NUMBER_WORDS[tok])
    return "".join(out)


def longest_digit_run(text: str) -> str:
    """The longest stretch of digits read out back to back ("zero one zero,
    five five five" -> 010555), ignoring a lone "one" in "one second"."""
    best, cur = "", []
    for tok in re.findall(r"[a-z]+|\d+", text.lower()):
        d = tok if tok.isdigit() else NUMBER_WORDS.get(tok)
        if d is not None:
            cur.append(d)
        elif tok not in ("and", "is", "it's", "its", "s"):
            best = max(best, "".join(cur), key=len)
            cur = []
    return max(best, "".join(cur), key=len)


def email_found(text: str, parts: tuple[str, ...] = ("khalid", "demo", "qivorasync", "com")) -> tuple[bool, str]:
    """True when the email's parts come out in order, however it was written
    ("khalid dot demo at q i v o r a sync dot com" or khalid.demo@qivorasync.com)."""
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    pos = 0
    for part in parts:
        pos = compact.find(part, pos)
        if pos < 0:
            return False, part
        pos += len(part)
    return True, ""


def spoken_email(text: str) -> str | None:
    """An email address in the transcript, written (a@b.com) or spoken
    ("name dot x at domain dot com"), returned as written."""
    m = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
    if m:
        return m.group(0)
    dot = r"(?:\s*\.\s*|\s+dot\s+)"
    m = re.search(rf"(\w+(?:{dot}\w+)*)\s+at\s+((?:[a-z] )*\w+(?:{dot}\w+)+)", text.lower())
    if not m:
        return None

    def squash(s: str) -> str:
        s = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", s)  # "q i v o r a" -> "qivora"
        return re.sub(dot, ".", s).replace(" ", "")

    return f"{squash(m.group(1))}@{squash(m.group(2))}"


def name_hits(text: str, name: str) -> int:
    first, *rest = name.split()
    pattern = re.escape(first) + "".join(r"\s*" + re.escape(r) for r in rest)
    return len(re.findall(pattern, text, re.I))


def near_misses(text: str, name: str) -> list[str]:
    """Spellings that sound like the product name but aren't it ("Kivora Sync")."""
    first, *rest = name.split()
    tail = r"\s+" + r"\s+".join(re.escape(r) for r in rest) if rest else ""
    found = re.findall(r"\b(\w*[iy]v[oa]r\w*" + tail + r")\b", text, re.I)
    return [f for f in found if f.lower() != name.lower()]


def summarize(snap: dict, settings, product_name: str) -> tuple[list[dict], list[dict]]:
    """(feature rows, per-turn rows) for the Tutorial checks tab."""
    record = snap.get("record") or []
    caller = [r for r in record if r["speaker"] == "khalid"]
    features: list[dict] = []

    ids: dict[str, dict[int, int]] = {}
    for r in record:
        for i in r.get("diarized_speaker_ids") or []:
            ids.setdefault(r["speaker"], {}).setdefault(i, 0)
            ids[r["speaker"]][i] += 1
    if not settings.diarize:
        features.append(dict(feature="Multiple voices, each labeled", status="off", evidence="diarize is off in Call settings"))
    elif ids:
        main = {s: max(c, key=c.get) for s, c in ids.items()}
        distinct = len(set(main.values())) == len(main)
        mixed = [SPEAKER_LABELS.get(s, s) for s, c in ids.items() if len(c) > 1]
        evidence = ", ".join(f"{SPEAKER_LABELS.get(s, s)} \u2192 speaker {i}" for s, i in main.items())
        if mixed:
            evidence += f" (more than one id heard for {', '.join(mixed)})"
        features.append(dict(feature="Multiple voices, each labeled", status="seen" if distinct and len(main) >= 2 else "partly",
                             evidence=evidence))
    else:
        features.append(dict(feature="Multiple voices, each labeled", status="not yet", evidence="no diarized turns yet"))

    heard = [r["heard_by_transcribe"] or "" for r in record]
    hits = sum(name_hits(t, product_name) for t in heard)
    misses = sorted({m for t in heard for m in near_misses(t, product_name)})
    features.append(dict(
        feature=f"Keyterm \u201c{product_name}\u201d", status="seen" if hits else "not yet",
        evidence=f"spelled right {hits}\u00d7 across all voices" + (f"; also heard as {', '.join(misses)}" if misses else ""),
    ))

    closes = [r["end_of_turn_confidence"] for r in caller if r.get("end_of_turn_confidence")]
    features.append(dict(
        feature="End-of-turn confidence threshold", status="seen" if closes else "not yet",
        evidence=f"threshold {settings.smart_turn:.2f}; your turns closed at " + ", ".join(f"{c:.2f}" for c in closes)
        if closes else f"threshold {settings.smart_turn:.2f}; no turn closed by Smart Turn yet",
    ))

    switched = [r["turn"] for r in caller if ARABIC.search(r["heard_by_transcribe"] or "")
                and LATIN_WORD.search(r["heard_by_transcribe"] or "")]
    features.append(dict(
        feature="Language switch halfway", status="seen" if switched else "not yet",
        evidence=f"turn {', '.join(map(str, switched))}: English and Arabic script in one turn" if switched
        else "switch language mid-sentence (the session is language=en; Transcribe keeps the Arabic in Arabic script)",
    ))

    line = snap.get("phone_line")
    features.append(dict(
        feature="Flaky, phone-like audio", status="on" if line else "off",
        evidence=(f"Transcribe streams encoding=mulaw&sample_rate=8000 for every voice; your mic lost "
                  f"{line['dropout_pct']}% of {line['packets']} packets" if line
                  else "flip the \u201cPhone line\u201d switch above the conversation"),
    ))

    numbers = [(r["turn"], longest_digit_run(r["heard_by_transcribe"] or "")) for r in caller]
    numbers = [(t, d) for t, d in numbers if len(d) >= 7]
    emails = [(r["turn"], spoken_email(r["heard_by_transcribe"] or "")) for r in caller]
    emails = [(t, e) for t, e in emails if e]
    parts = [f"turn {t}: {d}" for t, d in numbers] + [f"turn {t}: {e}" for t, e in emails]
    features.append(dict(
        feature="Phone number / email read out loud", status="seen" if parts else "not yet",
        evidence="; ".join(parts) if parts else "read a phone number or an email address to the agent",
    ))

    fillers = [t for r in caller for t in tokens(r["heard_by_transcribe"] or "") if t in FILLERS]
    features.append(dict(
        feature="Filler words (um, uh)",
        status="off" if not getattr(settings, "filler_words", False) else ("seen" if fillers else "not yet"),
        evidence=(f"{len(fillers)} heard: {', '.join(fillers[:8])}" if fillers else "say um or uh")
        if getattr(settings, "filler_words", False) else "filler_words is off - Transcribe strips them",
    ))

    turns = []
    for r in record:
        text = r["heard_by_transcribe"] or ""
        turns.append({
            "turn": r["turn"],
            "speaker": SPEAKER_LABELS.get(r["speaker"], r["speaker"]),
            "diarized id": ", ".join(map(str, r.get("diarized_speaker_ids") or [])),
            "end-of-turn conf.": r.get("end_of_turn_confidence"),
            "keyterm hits": name_hits(text, product_name),
            "fillers": " ".join(t for t in tokens(text) if t in FILLERS),
            "language switch": bool(ARABIC.search(text) and LATIN_WORD.search(text)),
            "digits": longest_digit_run(text) if len(longest_digit_run(text)) >= 4 else "",
        })
    return features, turns
