"""
Qivora Sync live call demo - the practical companion to the Grok Voice
Transcribe 2.0 article.

A real conversation, not a replay. You talk as Khalid on your own
microphone; grok-voice-transcribe-2.0 transcribes you live; a Grok model
(project/scripts/agent_brain.py, `grok-4.20-0309-non-reasoning` by default)
decides what Maya or Nadia says back from what you actually said, inside a
loose Qivora Sync support scenario with no scripted lines; Grok TTS voices
that reply in the same "celeste" (Maya) and "iris" (Nadia) voices the
article uses, streamed as it is written; and that same reply audio goes
back through the one open Transcribe 2.0 session, so all three speakers
land in one continuous, diarized transcript.

One WebRTC connection (`streamlit-webrtc`) carries both directions: your
microphone in, and the agents' voices out. Playing the agents through it
(instead of an <audio> element) is what lets them start speaking while the
reply is still being written, and it gives the browser's echo canceller
the agent audio as a reference, which is what makes talking over an agent
(barge-in) work.

The turn-taking itself runs on a background thread
(project/scripts/call_engine.py); this page only draws it. There is no
button to end your turn: the server's Smart Turn decides you've finished
from the silence your mic keeps streaming (experiment_log.md, entries 13
to 15).

The article's measured results come from the locked fixture in
project/ground_truth/qivora_call.py and its batch runs, not from this
app; the app is the interactive version of the same scenario.

Run (any shell - loads .env and fixes the Windows WinError 10054 noise
before Streamlit starts):
    python run_app.py
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_sink_track,
    create_pcm_audio_source_track,
    webrtc_streamer,
)

PROJECT_DIR = Path(__file__).resolve().parent / "project"
sys.path.insert(0, str(PROJECT_DIR))

import log_hygiene  # noqa: E402
import win_asyncio_fix  # noqa: E402

# Idempotent; run_app.py already did this before the server started, this
# covers a plain `streamlit run`.
win_asyncio_fix.install()
log_hygiene.install()

from ground_truth.qivora_call import PRODUCT_NAME, VOICES  # noqa: E402
from scripts.agent_brain import FAST_MODEL, MODELS, REASONING_MODEL  # noqa: E402
from scripts.call_engine import CallEngine, EngineSettings  # noqa: E402
from scripts.local_audio import LocalAudio, list_devices  # noqa: E402
from scripts.tutorial_checks import summarize as summarize_checks  # noqa: E402
import ui_theme  # noqa: E402

st.set_page_config(page_title="Qivora Sync - Live Call Demo", page_icon="\U0001F3A7", layout="wide")

MODEL_LABELS = {
    FAST_MODEL: f"{FAST_MODEL} - fast, no reasoning (recommended for voice)",
    REASONING_MODEL: f"{REASONING_MODEL} - flagship, reasons first (slower replies)",
}


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("engine", None)
    ss.setdefault("call_started", False)
    ss.setdefault("call_ended", False)
    # Auto-gain off, as in the sibling Grok Voice Think Fast demo and in the
    # setup that last worked here (experiment_log.md entries 14 and 16): it
    # flattens loudness (talking louder barely registers) and lifts room
    # noise and speaker bleed. The mic boost setting covers quiet mics.
    ss.setdefault("mic_agc", False)
    ss.setdefault("mic_ns", True)
    ss.setdefault("local_audio", None)


def api_key_ok() -> bool:
    return bool(os.environ.get("XAI_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Media: one WebRTC connection, mic in + agent voices out
# ---------------------------------------------------------------------------
def setup_media(engine: CallEngine):
    """Render the persistent audio connection. Called on every full rerun,
    in the same place, so the component keeps its connection; the browser
    permission prompt happens once for the whole call."""
    ss = st.session_state
    try:
        mic_sink = create_audio_sink_track(engine.mic.on_frame, key="caller-mic")
        voice = create_pcm_audio_source_track(key="agent-voice", sample_rate=16000)
        engine.set_playback(voice)
        ctx = webrtc_streamer(
            key="call-media",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={
                "video": False,
                "audio": {
                    "channelCount": 1,
                    "echoCancellation": True,
                    "noiseSuppression": bool(ss.mic_ns),
                    "autoGainControl": bool(ss.mic_agc),
                },
            },
            sink_audio_track=mic_sink,
            source_audio_track=voice.track,
            audio_html_attrs={"autoPlay": True, "controls": False, "style": {"display": "none"}},
            video_html_attrs={"hidden": True, "style": {"display": "none"}},
            sendback_video=False,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            translations={"start": "Connect microphone", "stop": "Disconnect"},
        )
        return ctx
    except Exception as exc:  # noqa: BLE001 - reported to the page, not fatal
        st.warning(
            f"Live audio is unavailable here ({type(exc).__name__}: {exc}). "
            "Run with `python run_app.py` and open it on localhost."
        )
        return None


# ---------------------------------------------------------------------------
# Call lifecycle
# ---------------------------------------------------------------------------
def start_call(settings: EngineSettings, input_device: int | None = None, output_device: int | None = None) -> None:
    ss = st.session_state
    engine = CallEngine(settings=settings, voices=VOICES, keyterms=[PRODUCT_NAME])
    local = None
    if settings.audio_path == "local":
        local = LocalAudio(engine.mic, input_device, output_device)
        engine.set_playback(local)
        devs = list_devices()
        engine.describe = (f"mic={device_name(devs['inputs'], input_device)} -> "
                           f"{device_name(devs['outputs'], output_device)}")
    else:
        engine.describe = "browser audio (WebRTC)"
    engine.start()
    if local is not None:
        try:
            local.start()
        except Exception:
            engine.end("could not open the audio devices", wait=False)
            raise
    ss.engine = engine
    ss.local_audio = local
    if local is not None:
        devs = list_devices()
        ss.local_device_names = (device_name(devs["inputs"], input_device), device_name(devs["outputs"], output_device))
    ss.call_started = True
    ss.call_ended = False
    ss["phone-toggle"] = settings.phone_line


def stop_local_audio() -> None:
    local = st.session_state.get("local_audio")
    if local is not None:
        local.stop()
        st.session_state.local_audio = None


def end_call(engine: CallEngine) -> None:
    engine.end("call ended")
    stop_local_audio()
    st.session_state.call_ended = True


def device_name(devices: list[tuple[int, str]], index: int | None) -> str:
    return next((name for i, name in devices if i == index), "system default")


def paired_output(devs: dict, input_device: int | None) -> int | None:
    """The output that belongs to the same physical device as the mic:
    Windows names both halves of a headset with the same tag in brackets,
    e.g. "Microphone (4- GM301)" and "Speakers (4- GM301)"."""
    name = device_name(devs["inputs"], input_device)
    m = re.search(r"\(([^)]+)\)?\s*$", name)
    if not m:
        return None
    tag = m.group(1).strip().lower()
    if len(tag) < 3:
        return None
    for i, out_name in devs["outputs"]:
        if tag in out_name.lower():
            return i
    return None


def mic_hint() -> str:
    """Name the real microphones, so the browser's SELECT DEVICE isn't left on
    a virtual or muted one."""
    try:
        names = [n for _, n in list_devices()["inputs"] if "sound mapper" not in n.lower()]
    except Exception:  # noqa: BLE001 - sounddevice missing or no devices
        return ""
    if not names:
        return ""
    return " Microphones on this computer: " + ", ".join(names) + "."


# ---------------------------------------------------------------------------
# The live stage: redrawn every 0.15 s from the engine's snapshot
# ---------------------------------------------------------------------------
@st.fragment(run_every=0.15)
def call_stage(engine: CallEngine) -> None:
    ss = st.session_state
    engine.touch()
    snap = engine.snapshot()

    if snap["phase"] == "ended" and not ss.call_ended:
        ss.call_ended = True
        stop_local_audio()
        st.rerun(scope="app")

    speaking = {
        "listening": "Khalid (you)",
        "thinking": ui_theme.SPEAKER_STYLE.get(snap["speaker"] or "", {}).get("label", "Agent"),
        "speaking": ui_theme.SPEAKER_STYLE.get(snap["speaker"] or "", {}).get("label", "Agent"),
    }.get(snap["phase"], "\u2014")
    ui_theme.status_chips(
        snap["turn_no"], speaking, snap["model"], snap["last_latency_ms"],
        engine.settings.diarize, engine.settings.format_,
        phone_line=snap.get("phone_line"),
    )
    st.markdown(
        ui_theme.render_statebar(snap["phase"], snap["speaker"], snap["audio_seconds"], snap["last_latency_ms"]),
        unsafe_allow_html=True,
    )
    st.markdown(ui_theme.render_stage(snap, first_turn=snap["turn_no"] == 0), unsafe_allow_html=True)

    # A fixed slot per button, so the row never shifts when one appears.
    cols = st.columns([1.1, 1, 2.6, 2.3])
    with cols[0]:
        # No interrupt button: just start talking (see CallEngine._step_barge_in).
        if snap["phase"] == "error":
            if st.button("\U0001F504 Reconnect", key="reconnect", type="primary"):
                with st.spinner("Reconnecting to grok-voice-transcribe-2.0 ..."):
                    try:
                        engine.reconnect()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Reconnect failed: {exc}")
    with cols[1]:
        if snap["phase"] not in ("ended", "ending"):
            if st.button("\u23F9 End call", key="end-call"):
                end_call(engine)
                st.rerun(scope="app")
    with cols[2]:
        if snap["phase"] not in ("ended", "ending"):
            if "phone-toggle" not in ss:
                ss["phone-toggle"] = snap["phone_on"]
            st.toggle(
                "Phone line \u00b7 8 kHz \u03bc-law", key="phone-toggle",
                on_change=lambda: engine.set_phone_line(bool(ss["phone-toggle"])),
                help="Switch the call onto a simulated phone line and back, mid-call. Transcribe then gets every "
                     "voice as `encoding=mulaw&sample_rate=8000`, your mic loses ~3% of its packets, and you hear "
                     "the agents narrowband. Each bubble is labelled PHONE or MIC by what Transcribe received.",
            )
            if snap["phone_switch_to"] is not None:
                st.caption(f"Switching {'onto' if snap['phone_switch_to'] else 'off'} the phone line at your next pause\u2026")

    gated = bool(snap["mic_stats"].get("gated"))
    with st.expander("\U0001F50E Mic diagnostics", expanded=bool(snap["only_silence"] or snap["stt_silent"] or gated)):
        if engine.settings.audio_path == "local":
            label = ss.get("local_device_names", ("system default", ""))[0]
            local = ss.get("local_audio")
            if local is not None:
                label += f" \u00b7 {local.capture}"
        else:
            label = "the device chosen in SELECT DEVICE"
        st.markdown(ui_theme.render_diagnostics(snap, label), unsafe_allow_html=True)

    st.markdown("##### Conversation")
    st.markdown(ui_theme.render_conversation(snap["turns"], highlight_last=True), unsafe_allow_html=True)


@st.fragment(run_every=1.5)
def transcript_tab(engine: CallEngine) -> None:
    snap = engine.snapshot()
    st.caption("What Transcribe 2.0 heard for every speaker; for agent turns the line the model actually wrote is shown underneath when the two differ.")
    st.markdown(ui_theme.render_conversation(snap["turns"], full=True, show_said=True), unsafe_allow_html=True)


@st.fragment(run_every=1.5)
def checks_tab(engine: CallEngine) -> None:
    snap = engine.snapshot()
    features, turns = summarize_checks(snap, engine.settings, PRODUCT_NAME)
    st.caption(
        "The tutorial's feature list, checked against this call as it happens. The same checks run offline "
        "on the article's fixture audio in `scripts/validate_features.py` (report: `results/19_feature_checks.md`)."
    )
    st.markdown(ui_theme.render_checks(features), unsafe_allow_html=True)
    if turns:
        st.markdown("##### Per turn")
        st.dataframe(turns, width="stretch", hide_index=True)


@st.fragment(run_every=1.5)
def record_tab(engine: CallEngine) -> None:
    snap = engine.snapshot()
    if not snap["record"]:
        st.caption("No turns yet.")
        return
    st.dataframe(snap["record"], width="stretch")
    record = {
        "product_name": PRODUCT_NAME,
        "agent_model": snap["model"],
        "settings": vars(engine.settings),
        "turns": snap["record"],
        "raw_event_count": len(engine.all_events),
    }
    st.download_button(
        "\u2B07 Download full transcript JSON",
        data=json.dumps(record, indent=2, ensure_ascii=False),
        file_name="qivora_sync_live_call.json",
        mime="application/json",
        key=f"dl-{snap['version']}",
    )


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------
def before_start_screen() -> None:
    ss = st.session_state
    ui_theme.hero(FAST_MODEL)
    st.write("")
    st.markdown("#### What this demo actually uses")
    ui_theme.feature_cards()
    st.write("")

    with st.container(border=True):
        st.subheader("Before you start")
        st.markdown(
            "- Nothing is scripted. Say whatever you like; Maya and Nadia answer what you actually said.\n"
            "- Replies are written live inside a loose Qivora Sync support scenario, streamed into Grok TTS "
            "as they are written, played through the same connection as your mic, and fed back into the same "
            "Transcribe 2.0 session - so all three voices are transcribed and diarized together.\n"
            "- Your mic stays live for the whole call, one permission prompt up front. Talk, pause, and the call "
            "moves on by itself: Smart Turn decides you're done from the silence your mic keeps streaming.\n"
            "- Talk over Maya or Nadia and they stop to listen. **Headphones recommended**, so the agents "
            "don't hear themselves through your speakers."
        )

        st.markdown("**Audio path**")
        audio_path = st.radio(
            "Audio path", ["browser", "local"], horizontal=True, label_visibility="collapsed",
            format_func=lambda p: {
                "browser": "Browser (WebRTC, with echo cancellation)",
                "local": "This computer (pick the mic and speakers directly)",
            }[p],
            help="If the browser path doesn't hear you, use 'This computer': it opens the device you pick "
                 "straight from Python, with no browser device choice or gain processing in between.",
        )
        input_device = output_device = None
        if audio_path == "local":
            try:
                devs = list_devices()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not list audio devices ({exc}). Is `sounddevice` installed?")
                return
            in_ids = [i for i, _ in devs["inputs"]]
            out_ids = [i for i, _ in devs["outputs"]]
            d1, d2 = st.columns(2)
            input_device = d1.selectbox(
                "Microphone", in_ids, format_func=lambda i: device_name(devs["inputs"], i),
                index=in_ids.index(devs["default_in"]) if devs["default_in"] in in_ids else 0,
                help="Virtual devices (e.g. 'WO Mic') are silent unless their app is running.",
            )
            paired = paired_output(devs, input_device)
            default_out = paired if paired is not None else devs["default_out"]
            output_device = d2.selectbox(
                "Speakers / headphones", out_ids, format_func=lambda i: device_name(devs["outputs"], i),
                index=out_ids.index(default_out) if default_out in out_ids else 0,
                # Keyed on the mic, so picking another mic re-pairs the output.
                key=f"out-dev-{input_device}",
            )
            if output_device != paired and paired is not None:
                st.caption(
                    f"Tip: play the agents on **{device_name(devs['outputs'], paired)}** - the headset that goes with "
                    "this mic. Open speakers (or FxSound) feed Maya's voice back into the mic, where it can be "
                    "transcribed as you."
                )
            elif paired is None:
                st.caption("Tip: headphones keep Maya's voice out of the mic; open speakers feed it back in.")
            st.caption(
                "Headphones give the cleanest calls on this path (there is no echo canceller). Talking over "
                "the agents works either way: a second transcription checks for your own words, so Maya's "
                "voice coming back through open speakers doesn't count as you. The mic is recorded raw "
                "(WASAPI exclusive), past Windows' voice effects, so other apps can't use it during the call."
            )

        with st.expander("Call settings", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Agents**")
                model = st.selectbox("Brain model", MODELS, format_func=lambda m: MODEL_LABELS.get(m, m))
                priority = st.checkbox(
                    "Priority processing (2x token price, lower time to first token)", value=True,
                    help="xAI `service_tier: priority`. A reply here is ~1.5k tokens, so this is well under a cent per turn.",
                )
                speculative = st.checkbox(
                    "Start the reply while Smart Turn is still deciding", value=True,
                    help="Drafts the reply from what you said as soon as you pause; thrown away unheard if you keep talking.",
                )
                barge_in = st.checkbox("Let me interrupt the agents by talking", value=True)
            with c2:
                st.markdown("**Transcription & microphone**")
                diarize = st.checkbox("diarize", value=True)
                format_ = st.checkbox("format (ITN) - can overfire on words like 'second'", value=False)
                filler_words = st.checkbox(
                    "Show filler words (um, uh)", value=True,
                    help="`filler_words=true`. Off by default in the API, which strips hesitations from the transcript.",
                )
                phone_line = st.checkbox(
                    "Start on a phone line (switchable during the call)", value=False,
                    help="Makes the whole call a phone call. Transcribe gets every voice as a phone line "
                         "delivers it (`encoding=mulaw&sample_rate=8000`, 8 kHz G.711), your mic also loses ~3% "
                         "of its 20 ms packets, and you hear Maya and Nadia narrowband (300-3400 Hz).",
                )
                smart_turn = st.slider("Smart Turn threshold", 0.5, 0.9, 0.8, 0.05,
                                       help="Higher waits for a more confident end of turn (better for dictating numbers).")
                smart_turn_timeout = st.slider("Smart Turn timeout (ms)", 1000, 5000, 3000, 250,
                                               help="Maximum silence before your turn is closed anyway.")
                vad_threshold = st.selectbox(
                    "Transcribe voice-activity sensitivity", [None, 0.04, 0.02],
                    format_func=lambda v: {None: "server default (0.08)", 0.04: "0.04 - quiet mic",
                                           0.02: "0.02 - very quiet mic"}[v],
                    help="`vad_threshold`: audio scoring below it is skipped as non-speech. Lower it if quiet "
                         "speech is being ignored (may add stray words from background noise).",
                )
                mic_boost = st.slider("Mic boost", 1.0, 4.0, 1.0, 0.5,
                                      help="Software gain with a soft limiter, applied before Transcribe. "
                                           "Raise it if the diagnostics show your voice below the speech threshold.")
                if audio_path == "browser":
                    ss.mic_agc = st.checkbox("Browser automatic gain control", value=ss.mic_agc,
                                             help="Off by default: it flattens loudness and lifts room noise and speaker bleed.")
                    ss.mic_ns = st.checkbox("Browser noise suppression", value=ss.mic_ns)

        if not api_key_ok():
            st.error("XAI_API_KEY is not set. Start the app with `python run_app.py` (it reads .env) or export it first.")
            return
        if st.button("\u25B6 Start call", type="primary"):
            settings = EngineSettings(
                diarize=diarize, format_=format_, smart_turn=smart_turn,
                smart_turn_timeout=smart_turn_timeout, vad_threshold=vad_threshold,
                model=model, priority=priority, speculative=speculative,
                barge_in=barge_in, audio_path=audio_path, mic_boost=mic_boost,
                filler_words=filler_words, phone_line=phone_line,
            )
            with st.spinner("Connecting to grok-voice-transcribe-2.0 and Grok TTS ..."):
                try:
                    start_call(settings, input_device, output_device)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not start the call: {exc}")
                    return
            st.rerun()


def main() -> None:
    init_state()
    ui_theme.apply_theme()
    ui_theme.topbar()
    ss = st.session_state

    if not ss.call_started or ss.engine is None:
        before_start_screen()
        return

    engine: CallEngine = ss.engine

    if not ss.call_ended:
        with st.container(border=True):
            if engine.settings.audio_path == "local":
                in_name, out_name = ss.get("local_device_names", ("system default", "system default"))
                st.markdown(
                    '<div class="mic-note">\U0001F399\uFE0F <b>This computer</b> - listening on '
                    f"<b>{html.escape(in_name)}</b>, agents play on "
                    f"<b>{html.escape(out_name)}</b>. No browser permission needed."
                    + (" Talk over the agents any time to interrupt." if engine.settings.barge_in else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="mic-note">\U0001F399\uFE0F <b>Audio connection</b> - one click, live for the whole call. '
                    "Your mic goes in, Maya's and Nadia's voices come back on the same line. "
                    "Use <b>SELECT DEVICE</b> to pick the mic you actually talk into."
                    f"{html.escape(mic_hint())}</div>",
                    unsafe_allow_html=True,
                )
                setup_media(engine)

    tabs = st.tabs(["Live call", "Transcript", "Tutorial checks", "Structured record"])
    with tabs[0]:
        if ss.call_ended:
            snap = engine.snapshot()
            st.markdown(
                ui_theme.render_statebar("ended", None, snap["audio_seconds"], snap["last_latency_ms"]),
                unsafe_allow_html=True,
            )
            st.markdown(ui_theme.render_stage({**snap, "phase": "ended"}, first_turn=False), unsafe_allow_html=True)
            if st.button("Start a new call", type="primary"):
                stop_local_audio()
                ss.call_started = False
                ss.engine = None
                st.rerun()
            st.markdown(ui_theme.render_conversation(snap["turns"]), unsafe_allow_html=True)
        else:
            call_stage(engine)
    with tabs[1]:
        transcript_tab(engine)
    with tabs[2]:
        checks_tab(engine)
    with tabs[3]:
        record_tab(engine)


if __name__ == "__main__":
    main()
