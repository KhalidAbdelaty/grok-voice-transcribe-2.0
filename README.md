# Grok Voice Transcribe 2.0: one support call, every feature

The code behind the DataCamp tutorial *Grok Voice Transcribe 2.0 API Tutorial: Build a Real-Time Support Call Transcriber*. One fictional support call (Khalid calls Qivora Sync support, Maya answers, Nadia joins) goes through `grok-voice-transcribe-2.0` batch and streaming, and gets harder one option at a time: diarization, keyterms, an English-to-Egyptian-Arabic switch, a spoken phone number and email, Smart Turn, 8 kHz phone audio, and multichannel.

It comes in two parts:

- **The experiments:** small scripts you run against the same locked fixture, each saving the raw API response so any number in the article can be checked.
- **A live companion app:** you talk as Khalid into your own mic; Maya and Nadia answer with `grok-4.20-0309-non-reasoning` and Grok TTS, and Transcribe 2.0 transcribes all three voices in one session.

## Quickstart

You need Python 3.10+, an [xAI API key](https://console.x.ai), and [ffmpeg](https://ffmpeg.org/download.html) on your PATH (only for building the audio fixtures).

```bash
git clone https://github.com/KhalidAbdelaty/grok-voice-transcribe-2.0.git
cd grok-voice-transcribe-2.0
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then paste your key into .env
python project/scripts/make_fixtures.py             # 13 TTS calls (a few cents), then local mixing
```

`make_fixtures.py` writes `project/audio/`: each voiced line, the mixed 79.5-second call (`clean_master.wav`), a three-channel version, an 8 kHz mu-law phone version, and the 16 kHz clip of Khalid's number used for Smart Turn. Your fixture is freshly synthesized, so expect small wording differences from the article's run; the checks below grade behavior, not exact strings.

Every script reads `XAI_API_KEY` from `.env` (a Windows line ending on the key is stripped for you). Results land in `project/results/`.

## Reproduce the article, section by section

Run these from the repository root.

| Article section | Command | What to look at |
|---|---|---|
| Baseline | `python project/scripts/stt_client.py project/audio/mixed/clean_master.wav 01_baseline` | `results/01_baseline.json`: 231 words, "Kivora Sync" |
| Diarization, keyterm, format, fillers | the snippet below | one option per run, same audio |
| Speaker turns | `python project/scripts/speaker_turns.py project/results/03_diarize.json` | 10 named, timed turns |
| Streaming + Smart Turn (full call) | `python project/scripts/stt_stream_client.py project/audio/mixed/clean_master.wav 08_streaming_smartturn` | how rarely `speech_final` fires with three speakers |
| Smart Turn threshold sweep (TTS pause) | `python project/scripts/smart_turn_experiment.py` | 0.5, 0.7 and 0.9 behave identically |
| All six tutorial checks, graded | `python project/scripts/validate_features.py` | `results/19_feature_checks.md` (PASS/FAIL per feature) |
| Final phone-quality run | `python project/scripts/support_transcriber.py audio/phone/call_8k_mulaw.raw --phone` | timed turns plus a saved run record |

The batch experiments are the same client with one more argument:

```python
import sys; sys.path.insert(0, "project")
from scripts.stt_client import transcribe

clip = "project/audio/mixed/clean_master.wav"
transcribe(clip, "02_keyterm", keyterms=["Qivora Sync"])
transcribe(clip, "03_diarize", diarize=True)
transcribe(clip, "04_format_language", language="en", format_=True)
transcribe(clip, "05_filler_words", filler_words=True)
transcribe("project/audio/phone/call_8k_mulaw.raw", "06_phone_8k_mulaw",
           keyterms=["Qivora Sync"], diarize=True, audio_format="mulaw", sample_rate=8000)
transcribe("project/audio/multichannel/call_3channel.wav", "07_multichannel",
           keyterms=["Qivora Sync"], multichannel=True)
```

`validate_features.py` streams real audio at real-time pace, so it takes about 2.5 minutes. It checks:

- the three voices get three consistent speaker ids;
- the keyterm, with and without it, on clean and phone audio;
- Smart Turn at 0.5, 0.7 and 0.9 on a 1.0 s and a 3.5 s pause;
- the language switch;
- a flaky 8 kHz mu-law line with dropped packets;
- the phone number and email;
- filler words.

### Things the article found that you should see too

- **Word timings vs formatting:** with `format=true`, only the top-level `text` is formatted (`0105551234`). The `words` array keeps the spoken form, so turns built from words show "0 1 0 5 5 5 ...".
- **Diarization on phone audio:** on the 8 kHz fixture, diarization finds every boundary but uses four ids for three voices. If you have separate legs, `multichannel=true` avoids that entirely.
- **Word `confidence`:** the REST reference documents a per-word `confidence` ("omitted when 0"). It never came back in our runs; diarized streaming words carry `speaker_confidence` instead.

## The live app

```bash
python run_app.py
```

Open `http://localhost:8501`, pick an audio path, and press **Start call**. Say hello and Maya picks up.

- **Audio path:**
  - **Browser** (WebRTC, with echo cancellation): use SELECT DEVICE to pick the mic you actually talk into.
  - **This computer**: opens the mic and speakers you choose directly from Python. Use this if the browser path doesn't hear you.
- **Use headphones.** Open speakers feed Maya's voice back into the mic. The app filters its own voice out of your turns and measures how much leaks in (Mic diagnostics shows it), but headphones make interruptions and turn-taking far cleaner.
- **Turn-taking:** no buttons. Smart Turn ends your turn when you finish a thought (threshold 0.8 by default, in Call settings). Talk over an agent and she stops to listen; "mm-hmm" or a cough won't cut her off.
- **Arabic:** switch to Egyptian Arabic and the agents answer in it.
- **Phone line:** flip the "Phone line · 8 kHz μ-law" switch during the call. At your next pause, the app reopens Transcribe on `encoding=mulaw&sample_rate=8000`, your mic starts losing ~3% of its packets, and you hear the agents narrowband. Every bubble is labeled `PHONE · 8 kHz` or `MIC · 16 kHz` by what Transcribe actually received.
- **Tutorial checks tab:** ticks off the article's features as they happen in your call: speaker ids, keyterm hits, end-of-turn confidence per turn, language switch, digits and emails, fillers.

Each call writes a JSON-lines log to `project/results/live_logs/`, with one stats line per second (mic level, thresholds, what the barge-in listener heard). Start there when something feels off.

### Headless checks (no mic needed)

`validate_engine.py` plays the fixture's Khalid lines into the app's engine as a fake microphone, with everything else live:

```bash
python project/scripts/validate_engine.py latency          # time from "caller stopped" to first agent audio
python project/scripts/validate_engine.py barge-in --path local --echo 0.02 --caller-volume 0.3
python project/scripts/validate_engine.py backchannel      # "Yeah." under Maya must not cut her off
python project/scripts/validate_engine.py phantom --path local --echo 0.02   # agent echo must not become your turn
python project/scripts/validate_engine.py arabic           # reply comes back in Egyptian Arabic
python project/scripts/validate_engine.py toggle           # phone line on and off mid-call
python project/scripts/validate_engine.py latency --phone  # a whole call over the mu-law wire
```

Options: `--echo 0.3` simulates open speakers, `--room-noise 0.02` simulates a noisy room, and `--path local` applies the "This computer" echo rules.

Offline tests, no API calls:

```bash
python project/scripts/test_mic_input.py
python project/scripts/test_stitch_partials.py   # replays the events from the streaming run in the table above
```

`test_stitch_partials.py` needs the events file from the "Streaming + Smart Turn" command in the table above; without it, the test skips and prints that command.

## Troubleshooting

- **`InvalidHeader ... return character(s) in header value`:** your key has a Windows `\r` on the end. Put it in `.env` and use `run_app.py` or the scripts, which strip it. If you export it yourself: `export $(grep -v '^#' .env | tr -d '\r' | xargs)`.
- **`ConnectionResetError: [WinError 10054]` spam on Windows:** harmless socket resets from browser tabs closing. `run_app.py` installs a fix before Streamlit starts, so launch with it rather than `streamlit run`.
- **Words go missing, quiet syllables get chopped (Windows):** Windows audio effects (Voice Clarity, audio enhancements, noise suppression) can gate the mic to digital silence in shared mode, which is what browsers use. Check yours with `python project/scripts/mic_check.py` (add part of the mic's name, e.g. `GM301`), staying quiet while it records. If the shared paths show "gated," either turn it off (Settings > System > Sound > your mic > Audio enhancements: Off) or use the app's "This computer" path. That path records the mic raw in WASAPI exclusive mode, which bypasses those effects but holds the mic for the length of the call. The app also warns you on the stage when it detects the gate.
- **The app never hears you:** open Mic diagnostics under the stage. "Frames arriving" at 0 means the wrong device (a virtual mic such as WO Mic stays silent); a level that never crosses the speech bar means raise Mic boost or pick another input. The "This computer" path skips the browser's device choice entirely.
- **Agent voices get transcribed as you:** that is speaker echo, usually from open speakers or a sound enhancer such as FxSound. Use headphones; the app drops the turns it can prove are echo and counts them in Mic diagnostics.
- **`ffmpeg not found` from `make_fixtures.py`:** install ffmpeg and make sure both `ffmpeg` and `ffprobe` are on your PATH. The app itself decodes audio with PyAV and doesn't need it.

## Cost

- **Transcription:** $0.10 per hour of audio for batch and $0.20 per hour for streaming, [per SpaceXAI](https://x.ai/news/grok-voice-transcribe-2), with diarization, timestamps and keyterms included. The full batch experiment set costs about 2 cents; `validate_features.py` is about 3 cents per run.
- **The live app:** it streams two sessions (your turns and the barge-in listener) for as long as a call is open, about 7 cents per 10-minute call for transcription, plus the language model and TTS for the agents' replies.

## Layout

```
run_app.py                      # starts the app (loads .env, Windows socket fix)
app_streamlit.py                # the app UI
project/
  ground_truth/qivora_call.py   # the locked call script, voices, plain text
  ground_truth/timeline.json    # where each turn sits in the mixed call
  scripts/
    make_fixtures.py            # rebuilds every audio fixture
    stt_client.py               # batch REST client
    stt_stream_client.py        # streaming WebSocket client
    speaker_turns.py            # diarized words into named, timed turns
    support_transcriber.py      # the finished client, batch or streaming
    validate_features.py        # the tutorial checks, graded against the real API
    call_engine.py              # the live call: turns, barge-in, phone line
    live_call_session.py        # one shared Transcribe session per call
    agent_brain.py, tts_stream.py, mic_input.py, phone_line.py, ...
```

Code comments sometimes cite "entry N": that is the lab notebook the article was written from, kept with the article rather than in this repo.
