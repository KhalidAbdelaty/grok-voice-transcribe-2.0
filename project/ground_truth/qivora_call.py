"""
Ground truth for the Qivora Sync support call fixture.
Locked BEFORE any transcription test is run. Every derivative audio file
(clean mix, stream replay, 8 kHz phone, multichannel) comes from these
same synthesized/recorded lines.

Speakers:
  khalid - customer (stand-in TTS voice "sirius" for pipeline testing;
           the Streamlit companion app swaps this for a real microphone)
  maya   - first-line Qivora Sync support agent (TTS voice "celeste")
  nadia  - escalation engineer (TTS voice "iris")

Each line may be split into sub-segments so a single conversational
turn can switch language mid-utterance (the TTS API takes one
`language` per request, so a code-switched turn is built by
concatenating separately synthesized segments back-to-back before
it ever reaches the transcriber).
"""

VOICES = {"khalid": "sirius", "maya": "celeste", "nadia": "iris"}

PRODUCT_NAME = "Qivora Sync"
PHONE_DIGITS = "0105551234"          # ground-truth digits, no formatting
EMAIL = "khalid.demo@qivorasync.com"  # ground-truth email, no formatting

# turn_id, speaker, list of (language, tts_text) segments, plain_text (no speech tags,
# for ground truth comparison), notes (what this turn is planted to test)
TURNS = [
    dict(
        turn_id=1, speaker="maya",
        segments=[("en", "Thank you for calling Qivora Sync support, this is Maya. How can I help you today?")],
        plain_text="Thank you for calling Qivora Sync support, this is Maya. How can I help you today?",
        notes="opening, product name #1",
    ),
    dict(
        turn_id=2, speaker="khalid",
        segments=[("en", "Hi, [pause] uh, yeah, I'm having a problem with Qivora Sync. It keeps failing to sync my files between my laptop and my phone, and, [pause] um, it's been happening since yesterday.")],
        plain_text="Hi, uh, yeah, I'm having a problem with Qivora Sync. It keeps failing to sync my files between my laptop and my phone, and, um, it's been happening since yesterday.",
        notes="filler words (uh, um), product name #2",
    ),
    dict(
        turn_id=3, speaker="maya",
        segments=[("en", "I'm sorry about that. Let me pull up your account. Can I get your phone number and email address, please?")],
        plain_text="I'm sorry about that. Let me pull up your account. Can I get your phone number and email address, please?",
        notes="overlap target: starts slightly before turn 2 fully ends at mix time",
        overlap_with_previous_ms=350,
    ),
    dict(
        turn_id=4, speaker="khalid",
        segments=[
            ("en", "Sure, one second."),
            ("ar-EG", "طب خليني بس أدور على الموبايل, تمام معايا دلوقتي."),
            ("en", "Okay, got it. So it's zero one zero, five five five, [pause] one two three four."),
            ("en", "And the email is khalid dot demo at qivorasync dot com."),
        ],
        plain_text=(
            "Sure, one second. Tayyib khalliini bass adawwar 'ala el mobile, tamaam ma3aaya dilwa'ti. "
            "Okay, got it. So it's zero one zero, five five five, one two three four. "
            "And the email is khalid dot demo at qivorasync dot com."
        ),
        notes="English->Egyptian Arabic->English code switch, phone number with mid-dictation pause, spoken email",
    ),
    dict(
        turn_id=5, speaker="maya",
        segments=[("en", "Got it, thank you. Let me check the sync logs real quick. [pause] Uh, okay, I can see a few failed sync attempts on Qivora Sync. Let me bring in Nadia from our escalation team to take a closer look.")],
        plain_text="Got it, thank you. Let me check the sync logs real quick. Uh, okay, I can see a few failed sync attempts on Qivora Sync. Let me bring in Nadia from our escalation team to take a closer look.",
        notes="filler word (uh), product name #3, introduces third speaker",
    ),
    dict(
        turn_id=6, speaker="nadia",
        segments=[("en", "Hi Khalid, this is Nadia from the escalation team. I can see the failed syncs on your account. Give me just a second to check something on our end.")],
        plain_text="Hi Khalid, this is Nadia from the escalation team. I can see the failed syncs on your account. Give me just a second to check something on our end.",
        notes="third speaker's first turn",
    ),
    dict(
        turn_id=7, speaker="khalid",
        segments=[("en", "Okay, thank you so much.")],
        plain_text="Okay, thank you so much.",
        notes="short acknowledgement",
    ),
    dict(
        turn_id=8, speaker="nadia",
        segments=[("en", "Found it. There was a stale sync token on our backend for your account, and that's what was blocking Qivora Sync. I've cleared it, so it should sync properly now. Is there anything else I can help with?")],
        plain_text="Found it. There was a stale sync token on our backend for your account, and that's what was blocking Qivora Sync. I've cleared it, so it should sync properly now. Is there anything else I can help with?",
        notes="product name #4, resolution",
    ),
    dict(
        turn_id=9, speaker="khalid",
        segments=[("en", "No, that's everything. Thank you both.")],
        plain_text="No, that's everything. Thank you both.",
        notes="closing",
    ),
    dict(
        turn_id=10, speaker="maya",
        segments=[("en", "You're very welcome. Have a great day!")],
        plain_text="You're very welcome. Have a great day!",
        notes="closing",
    ),
]

if __name__ == "__main__":
    total_words = sum(len(t["plain_text"].split()) for t in TURNS)
    print(f"{len(TURNS)} turns, ~{total_words} words across 3 speakers")
