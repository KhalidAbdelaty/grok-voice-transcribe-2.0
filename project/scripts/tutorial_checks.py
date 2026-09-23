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


def spoken_email(text: str, partial_ok: bool = False) -> str | None:
    """An email address in the transcript, written (a@b.com) or spoken
    ("name dot x at domain dot com"), returned as written.

    Live callers don't say it as cleanly as the fixture does: fillers land in
    the middle ("khalid dot demo at, uh, q i v o r a sync"), and the ending
    can be left off or said later ("Khaled dot at Qivora Sync", then "Com"
    as the next turn, call_20260923_053326). With `partial_ok` (the live
    checks), an address with no ".com" still counts when the line talks
    about an email, so "look at the app" never does. Without it (the fixture
    report), the ending is required: interim text cut off mid-address
    ("... at q i v") is not an address."""
    m = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
    if m:
        return m.group(0)
    low = re.sub(r"\b(?:" + "|".join(FILLERS) + r")\b", " ", text.lower())
    low = re.sub(r"[,;:?!]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    # In the name, Transcribe sometimes writes a spoken "dot" as a full stop
    # ("Khalid. demo"); in the domain a full stop followed by a space ends
    # the sentence instead ("... dot com. Could you ...").
    name_dot = r"(?:\s*\.\s*|\s+dot\s+)"
    domain_dot = r"(?:\s+dot\s+|\.(?=[a-z0-9]))"
    word = r"(?!(?:and|my|is|the|a|to|so|at|dot)\b)[a-z0-9]+"
    spelled = rf"(?:\b[a-z] )*\b{word}"  # "q i v o r a sync", "c h a l i d"
    m = re.search(rf"({spelled}(?:{name_dot}{word})*)(?:\s+dot)?\s+at\s+({spelled}(?:\s+{word})?(?:{domain_dot}{word})*)", low)
    if not m:
        return None
    name, domain = m.group(1), m.group(2)
    if not re.search(domain_dot, domain):
        if not partial_ok or "mail" not in low or len(domain.replace(" ", "")) < 4:
            return None

    def squash(s: str, dot: str) -> str:
        s = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", s)  # "q i v o r a" -> "qivora"
        return re.sub(dot, ".", s).replace(" ", "")

    return f"{squash(name, name_dot)}@{squash(domain, domain_dot)}"


def number_read_out(texts: list[tuple[int, str]]) -> tuple[list[int], str] | None:
    """A phone number the caller read out: seven or more digits in one turn,
    or pieces over the next few turns (the agent asks for "the rest": "0 1 0.
    2 5. 6." in one turn, "five five" two turns later). Returns the turns and
    the digits heard, or None."""
    runs = [(t, longest_digit_run(text)) for t, text in texts]
    runs = [(t, d) for t, d in runs if len(d) >= 2]
    for i, (t, d) in enumerate(runs):
        if len(d) >= 7:
            return [t], d
        turns, digits = [t], d
        for t2, d2 in runs[i + 1 : i + 3]:
            if t2 - turns[-1] > 4:  # a caller turn, an agent reply, and back
                break
            turns.append(t2)
            digits += d2
            if len(digits) >= 7:
                return turns, digits
    return None


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

    # Speaker ids restart in every Transcribe session (a phone line switch or
    # a reconnect opens a new one), so they are only compared within one:
    # pooled, the phone leg's "0" for everyone read as Maya and Nadia each
    # having several ids (call_20260923_051305).
    sessions: dict[int, dict[str, dict[int, int]]] = {}
    phone_sessions: set[int] = set()
    for r in record:
        sess = r.get("session", 1)
        if r.get("source") == "phone":
            phone_sessions.add(sess)
        for i in r.get("diarized_speaker_ids") or []:
            counts = sessions.setdefault(sess, {}).setdefault(r["speaker"], {})
            counts[i] = counts.get(i, 0) + 1

    def labels(sess: int) -> tuple[str, bool]:
        """'You -> 0, Maya -> 1' for one session, and whether every speaker
        heard in it got an id of their own."""
        main = {s: max(c, key=c.get) for s, c in sessions[sess].items()}
        text = ", ".join(f"{SPEAKER_LABELS.get(s, s)} \u2192 {i}" for s, i in main.items())
        return text, len(set(main.values())) == len(main)

    clean = [s for s in sorted(sessions) if s not in phone_sessions]
    phone = [s for s in sorted(sessions) if s in phone_sessions]
    multi = len(sessions) > 1

    def name(sess: int) -> str:
        return f"session {sess}: " if multi else ""

    if not settings.diarize:
        features.append(dict(feature="Multiple voices, each labeled", status="off", evidence="diarize is off in Call settings"))
    elif clean:
        judged = [(s, *labels(s)) for s in clean]
        voices = {sp for s in clean for sp in sessions[s]}
        clashes = [s for s, _, ok in judged if not ok]
        evidence = " \u00b7 ".join(f"{name(s)}{text}" for s, text, _ in judged)
        if clashes:
            evidence += f" (two voices share an id in session {', '.join(map(str, clashes))})" if multi \
                else " (two voices share an id)"
        features.append(dict(feature="Multiple voices, each labeled",
                             status="seen" if len(voices) >= 2 and not clashes else ("partly" if clashes else "not yet"),
                             evidence=evidence))
    elif phone:
        features.append(dict(feature="Multiple voices, each labeled", status="not yet",
                             evidence="only phone-line turns so far; see the phone row for how the 8 kHz leg was diarized"))
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
    phone_ids = []
    for s in phone:
        text, ok = labels(s)
        phone_ids.append(f"{name(s)}{text}" + ("" if ok else " (voices merged)"))
    phone_note = "; diarization on the 8 kHz leg: " + " \u00b7 ".join(phone_ids) if phone_ids and settings.diarize else ""
    features.append(dict(
        feature="Flaky, phone-like audio", status="on" if line else ("seen" if phone else "off"),
        evidence=(f"Transcribe streams encoding=mulaw&sample_rate=8000 for every voice; your mic lost "
                  f"{line['dropout_pct']}% of {line['packets']} packets{phone_note}" if line
                  else f"used earlier in this call{phone_note}" if phone
                  else "flip the \u201cPhone line\u201d switch above the conversation"),
    ))

    caller_texts = [(r["turn"], r["heard_by_transcribe"] or "") for r in caller]
    number = number_read_out(caller_texts)
    email = None
    for i, (t, text) in enumerate(caller_texts):
        # The ending can come as the caller's next turn ("... at Qivora Sync" / "Com").
        nxt = caller_texts[i + 1][1] if i + 1 < len(caller_texts) else ""
        found = spoken_email(text, partial_ok=True)
        if found and "." not in found.split("@")[1] and re.match(r"\s*(?:dot\s+)?com\b", nxt, re.I):
            found += ".com"
        if found:
            email = (t, found)
            break
    parts = []
    if number:
        parts.append(f"turn{'s' if len(number[0]) > 1 else ''} {'+'.join(map(str, number[0]))}: {number[1]}")
    if email:
        parts.append(f"turn {email[0]}: {email[1]}")
    # What the agent read back, as Transcribe heard it in the agent's voice.
    readback = []
    for r in record:
        if r["speaker"] == "khalid":
            continue
        heard = r["heard_by_transcribe"] or ""
        if number and not any("number" in p for p in readback) and len(longest_digit_run(heard)) >= 7:
            readback.append(f"number {longest_digit_run(heard)} (turn {r['turn']})")
        if email and not any("email" in p for p in readback) and spoken_email(heard, partial_ok=True):
            readback.append(f"email {spoken_email(heard, partial_ok=True)} (turn {r['turn']})")
    evidence = "; ".join(parts) if parts else "read a phone number or an email address to the agent"
    if readback:
        evidence += "; read back as " + ", ".join(readback)
    features.append(dict(
        feature="Phone number / email read out loud",
        status="seen" if parts else "not yet",
        evidence=evidence + ("" if not parts or (number and email) else
                             f"; {'an email address' if number else 'a phone number'} not heard yet"),
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
