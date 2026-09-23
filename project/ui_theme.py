"""Look and feel for the Qivora Sync live call demo.

A "master combo" theme, pulling the pieces that worked best across three
sibling DataCamp voice-API demos and adapting them to this app's own
three-speaker shape:

- the pulsing state dot + level-style call bar, the flash highlight on a
  changed record, and the pill-style tabs, from the Grok Voice Think Fast
  demo's ``ui_theme.py``;
- the feature-card grid and the live running-cost pill, from the DeepSeek
  V4.1 Flash visual-repair-agent app;
- the dual-logo top bar, the hero badge, and growing chat bubbles per
  speaker, from the GPT-Live-1 call widget - extended here from two
  speakers to three (Khalid, Maya, Nadia each get their own bubble color
  instead of a plain user/assistant split).

The "stage" card is this app's own: one fixed-height card that always
shows who has the floor right now (you listening-and-talking, an agent
thinking, an agent speaking with what it wrote next to what Transcribe
heard), so the page never jumps when the turn changes hands.

Kept out of app_streamlit.py on purpose, same reasoning as the sibling
projects: the app file should read as call logic, not markup.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DATACAMP_LOGO = ASSETS_DIR / "datacamp-logo.png"
GROK_LOGO = ASSETS_DIR / "grok-logo.png"

# Per-speaker color so a three-way call reads at a glance, not just
# "caller vs. agent" - Khalid is the customer, Maya first-line support,
# Nadia the escalation engineer who joins later.
SPEAKER_STYLE = {
    "khalid": {"label": "Khalid", "role": "Customer (you)", "dot": "#05192D", "initial": "K"},
    "maya": {"label": "Maya", "role": "Support agent", "dot": "#04603A", "initial": "M"},
    "nadia": {"label": "Nadia", "role": "Escalation engineer", "dot": "#6B3FA0", "initial": "N"},
}

STREAMING_RATE_PER_HOUR = 0.20  # published SpaceXAI streaming STT rate

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Spectral:wght@500;600;700&display=swap');

:root {
  --bg: #FFFFFF;
  --paper: #F5F7F9;
  --accent: #03EF62;
  --accent-ink: #05192D;
  --ink: #05192D;
  --muted: #5A6872;
  --border: #E1E5E9;
  --warn: #C4820E;
  --danger: #D64550;
  --live: #00A67E;
  --maya: #04603A;
  --nadia: #6B3FA0;
}

.stApp { background: var(--bg); }
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'Inter', sans-serif; color: var(--ink); }
h1, h2, h3, h4 { font-family: 'Spectral', Georgia, serif; color: var(--ink); letter-spacing: -0.015em; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] { visibility: visible; }
.block-container { padding-top: 1.4rem; max-width: 1180px; }
/* Fragment reruns every few hundred ms; don't fade the stage while they do. */
[data-stale="true"] { opacity: 1 !important; transition: none !important; }

/* ---- top bar ---- */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding-bottom: .9rem; border-bottom: 1px solid var(--border); margin-bottom: 1.1rem;
}
.topbar-brand { display: flex; align-items: center; gap: .7rem; font-weight: 800; }
.topbar-brand img { height: 22px; width: auto; display: block; }
.topbar-divider { width: 1px; height: 22px; background: var(--border); }
.pill-link {
  display: inline-flex; align-items: center; white-space: nowrap;
  font-size: .8rem; font-weight: 600; color: var(--ink);
  border: 1px solid var(--border); border-radius: 999px; padding: .35rem .85rem;
  text-decoration: none !important; background: #fff;
}
.pill-link:hover { border-color: var(--accent-ink); color: var(--accent-ink); }

/* ---- hero ---- */
.hero-badge {
  display: inline-flex; align-items: center; gap: .4rem; font-size: .78rem; font-weight: 700;
  color: var(--accent-ink); background: rgba(3,239,98,.14); border-radius: 999px;
  padding: .3rem .8rem; margin-bottom: .7rem;
}
.hero-title { font-size: 1.9rem; font-weight: 800; color: var(--ink); margin: 0 0 .35rem; line-height: 1.15; }
.hero-sub { color: var(--muted); font-size: .98rem; line-height: 1.55; max-width: 800px; margin: 0; }

/* ---- feature cards ---- */
.feature-card {
  background: var(--paper); border: 1px solid var(--border); border-radius: 14px;
  padding: 1rem 1.1rem; height: 100%; margin-bottom: .8rem; transition: all .15s ease;
}
.feature-card:hover { border-color: var(--accent); box-shadow: 0 4px 14px rgba(3,239,98,.18); transform: translateY(-2px); }
.feature-icon { font-size: 1.35rem; }
.feature-name { font-weight: 700; color: var(--ink); margin: .3rem 0 .2rem; font-size: .95rem; }
.feature-desc { color: var(--muted); font-size: .84rem; line-height: 1.45; margin: 0; }

/* ---- buttons ---- */
.stButton > button {
  background: #fff; color: var(--ink); border: 1px solid var(--border);
  border-radius: 11px; padding: .5rem 1rem; font-weight: 600; font-size: .9rem;
  transition: all .15s ease;
}
.stButton > button:hover { border-color: var(--accent-ink); transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: var(--accent); color: var(--accent-ink); border: none; }

/* ---- state bar ---- */
.state {
  display: flex; align-items: center; gap: 12px; padding: 10px 15px;
  border: 1px solid var(--border); border-radius: 13px; background: #fff; margin-bottom: 10px;
  flex-wrap: wrap;
}
.state-label { font-weight: 700; font-size: .92rem; white-space: nowrap; }
.state-sub { color: var(--muted); font-size: .8rem; }
.livedot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.live-off { background: var(--border); }
.live-listen { background: var(--live); animation: pulse 1.6s infinite; }
.live-speak { background: var(--accent); animation: pulse .8s infinite; }
.live-think { background: var(--warn); animation: pulse 1s infinite; }
.live-err { background: var(--danger); }
@keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: .3 } }
.state-pills { margin-left: auto; display: inline-flex; gap: .4rem; flex-wrap: wrap; }
.spill {
  display: inline-flex; align-items: center; gap: .35rem; border-radius: 999px;
  padding: .28rem .7rem; font-size: .76rem; font-weight: 600; white-space: nowrap;
  background: var(--paper); border: 1px solid var(--border); color: var(--ink);
}
.spill b { font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.spill.dark { background: var(--accent-ink); color: #fff; border-color: var(--accent-ink); }
.spill.dark b { color: var(--accent); }
.spill.fast b { color: #04603A; }

/* ---- the stage: who has the floor right now ---- */
.stage {
  display: flex; gap: 18px; align-items: flex-start; min-height: 150px; box-sizing: border-box;
  padding: 18px 20px; border-radius: 18px; border: 1px solid var(--border);
  background: linear-gradient(180deg, #fff, var(--paper)); margin-bottom: 12px;
  transition: border-color .25s ease, box-shadow .25s ease;
}
.stage.st-listening { border-color: rgba(0,166,126,.45); box-shadow: 0 0 0 4px rgba(0,166,126,.07); }
.stage.st-thinking { border-color: rgba(196,130,14,.40); }
.stage.st-speaking.sp-maya { border-color: rgba(4,96,58,.45); box-shadow: 0 0 0 4px rgba(3,239,98,.10); }
.stage.st-speaking.sp-nadia { border-color: rgba(107,63,160,.45); box-shadow: 0 0 0 4px rgba(107,63,160,.08); }
.stage.st-error { border-color: rgba(214,69,80,.5); background: #FFF7F8; }
.avatar-wrap { position: relative; width: 68px; height: 68px; flex: none; }
.avatar {
  position: absolute; inset: 0; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 1.5rem; color: #fff; background: var(--ink); z-index: 2;
}
.avatar.av-maya { background: var(--maya); }
.avatar.av-nadia { background: var(--nadia); }
.avatar.av-idle { background: #C9D1D8; }
.halo {
  position: absolute; inset: -8px; border-radius: 50%; z-index: 1;
  background: rgba(0,166,126,.20); transition: transform .12s ease-out, opacity .12s ease-out;
}
.halo.h-maya { background: rgba(3,239,98,.28); }
.halo.h-nadia { background: rgba(107,63,160,.22); }
.halo.h-think { background: rgba(196,130,14,.18); animation: breathe 1.4s ease-in-out infinite; }
@keyframes breathe { 0%,100% { transform: scale(.92); opacity: .5 } 50% { transform: scale(1.08); opacity: 1 } }
.stage-body { flex: 1; min-width: 0; }
.stage-title { display: flex; align-items: center; gap: .55rem; flex-wrap: wrap; font-weight: 800; font-size: 1.02rem; }
.stage-role { color: var(--muted); font-weight: 600; font-size: .8rem; }
.tag {
  display: inline-flex; align-items: center; gap: .3rem; font-size: .68rem; font-weight: 800; letter-spacing: .05em;
  text-transform: uppercase; border-radius: 999px; padding: .18rem .55rem; background: var(--paper); color: var(--muted);
  border: 1px solid var(--border);
}
.tag.t-live { background: rgba(0,166,126,.12); color: #04603A; border-color: rgba(0,166,126,.3); }
.tag.t-speak { background: rgba(3,239,98,.16); color: #04603A; border-color: rgba(3,239,98,.4); }
.tag.t-think { background: rgba(196,130,14,.12); color: #8A5B06; border-color: rgba(196,130,14,.3); }
.tag.t-final { background: rgba(3,239,98,.16); color: #04603A; }
.tag.t-err { background: rgba(214,69,80,.12); color: var(--danger); border-color: rgba(214,69,80,.3); }
.stage-text { margin-top: 8px; font-size: 1.02rem; line-height: 1.55; white-space: pre-wrap; word-break: break-word; }
.stage-text .interim { opacity: .5; }
.stage-src { display: block; margin-top: 8px; font-size: .66rem; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }
.stage-hint { margin-top: 8px; color: var(--muted); font-size: .9rem; line-height: 1.5; }
.stage-heard {
  margin-top: 10px; padding: 8px 11px; border-radius: 10px; background: #fff; border: 1px dashed var(--border);
  font-size: .86rem; line-height: 1.45; color: var(--ink);
}
.stage-heard .lbl { font-size: .66rem; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); margin-right: .4rem; }
.stage-foot { margin-top: 8px; color: var(--muted); font-size: .78rem; }
.eotc-bar { position: relative; display: inline-block; width: 90px; height: 7px; margin: 0 6px; vertical-align: middle;
            background: rgba(127,127,127,.18); border-radius: 4px; }
.eotc-bar span { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent, #00a67e); border-radius: 4px; }
.eotc-bar i { position: absolute; top: -3px; bottom: -3px; width: 2px; background: currentColor; opacity: .7; }
.dots { display: inline-flex; gap: 4px; margin-left: 2px; }
.dots span { width: 6px; height: 6px; border-radius: 50%; background: currentColor; opacity: .3; animation: dot 1.2s infinite both; }
.dots span:nth-child(2) { animation-delay: .18s; }
.dots span:nth-child(3) { animation-delay: .36s; }
@keyframes dot { 0%, 80%, 100% { opacity: .25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-2px); } }
.eq { display: inline-flex; gap: 4px; align-items: flex-end; height: 34px; flex: none; align-self: center; }
.eq i { width: 6px; border-radius: 3px; background: var(--border); display: block; transition: height .12s ease-out; }
.eq.e-live i { background: var(--live); }
.eq.e-maya i { background: var(--accent); }
.eq.e-nadia i { background: var(--nadia); }

/* ---- status chips ---- */
.status-row { display: flex; gap: .6rem; flex-wrap: wrap; margin-bottom: .8rem; }
.status-chip { border-radius: 12px; padding: .45rem .8rem; background: var(--paper); border: 1px solid var(--border); flex: 1 1 0; min-width: 96px; }
.status-chip .label { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.status-chip .value { font-size: .9rem; font-weight: 700; color: var(--ink); margin-top: .1rem; }
.status-chip.on .value { color: #04603A; }
.status-chip.off .value { color: var(--muted); }

/* ---- tabs ---- */
[data-baseweb="tab-list"], [role="tablist"] {
  gap: 4px; background: var(--paper); padding: 4px; border-radius: 12px; border: 1px solid var(--border);
}
[data-baseweb="tab-list"] button[data-baseweb="tab"], [data-testid="stTab"] {
  border-radius: 9px; padding: 6px 14px; color: var(--muted); font-weight: 600; font-size: .85rem;
  transition: all .15s ease; cursor: pointer;
}
[data-testid="stTab"] p { margin: 0; }
[data-baseweb="tab-list"] button[data-baseweb="tab"]:hover,
[data-testid="stTab"]:hover { background: rgba(5,25,45,.05); color: var(--ink); }
[data-baseweb="tab-list"] button[aria-selected="true"],
[data-testid="stTab"][aria-selected="true"], [data-testid="stTab"][data-selected="true"] {
  background: #fff; color: var(--ink); box-shadow: 0 1px 3px rgba(5,25,45,.12); border: 1px solid var(--border);
}
[data-testid="stTab"][aria-selected="true"] p, [data-testid="stTab"][data-selected="true"] p { color: var(--ink); }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
[data-testid="stTab"] .react-aria-SelectionIndicator { display: none; }

/* ---- conversation: newest at the bottom, pinned there without JS ----
   column-reverse makes the browser anchor the scroll position at the
   bottom, so every redraw lands on the latest turn instead of the top. */
.convo { display: flex; flex-direction: column-reverse; max-height: 46vh; overflow-y: auto; padding-right: 4px; }
.convo.full { max-height: 70vh; }
.bubble { margin-bottom: 10px; padding: 9px 13px; border-radius: 13px; max-width: 88%; border: 1px solid var(--border); }
.bubble-who { display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; font-size: .68rem; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 2px; }
.bubble-text { font-size: .92rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
.bubble-said { font-size: .8rem; color: var(--muted); margin-top: 4px; line-height: 1.4; }
.bubble-said b { font-weight: 700; }
.bubble-meta { font-size: .68rem; color: var(--muted); margin-top: 4px; }
.tchips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
.bubble-khalid .tchips { justify-content: flex-end; }
.tchip { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .62rem; letter-spacing: .03em;
         padding: 1px 6px; border-radius: 4px; border: 1px solid var(--border); color: var(--muted); background: rgba(255,255,255,.6); }
.tchip.phone { border-color: var(--warn); color: var(--warn); }
.bubble-khalid { background: var(--paper); margin-left: auto; border-right: 3px solid #05192D; }
.bubble-khalid .bubble-who { color: #05192D; justify-content: flex-end; }
.bubble-maya { background: rgba(3,239,98,.10); margin-right: auto; border-left: 3px solid var(--maya); }
.bubble-maya .bubble-who { color: var(--maya); }
.bubble-nadia { background: rgba(107,63,160,.09); margin-right: auto; border-left: 3px solid var(--nadia); }
.bubble-nadia .bubble-who { color: var(--nadia); }
.bubble-system { background: transparent; border: none; text-align: center; color: var(--muted); font-size: .8rem; max-width: 100%; margin: 2px auto 10px; }
.badge { font-size: .62rem; font-weight: 800; border-radius: 999px; padding: .1rem .45rem; letter-spacing: .03em; }
.badge.fast { background: rgba(3,239,98,.18); color: #04603A; }
.badge.slow { background: rgba(196,130,14,.14); color: #8A5B06; }
.badge.cut { background: rgba(214,69,80,.12); color: var(--danger); }
.just-finalized { animation: flash 1.4s ease-out; }
@keyframes flash { from { box-shadow: 0 0 0 3px rgba(3,239,98,.55); } to { box-shadow: 0 0 0 0 rgba(3,239,98,0); } }

/* ---- mic diagnostics ---- */
.diag-bar { position: relative; height: 14px; border-radius: 7px; background: var(--paper); border: 1px solid var(--border); overflow: hidden; }
.diag-fill { position: absolute; left: 0; top: 0; bottom: 0; background: #C9D1D8; transition: width .12s ease-out; }
.diag-fill.on { background: var(--live); }
.diag-mark { position: absolute; top: -2px; bottom: -2px; width: 2px; }
.diag-mark.floor { background: var(--muted); }
.diag-mark.thr { background: var(--warn); }
.diag-legend { display: flex; justify-content: space-between; font-size: .72rem; color: var(--muted); margin: 4px 0 10px; }
.diag-grid { display: grid; grid-template-columns: 170px 1fr; gap: 4px 12px; font-size: .82rem; }
.diag-k { color: var(--muted); }
.diag-v { font-family: ui-monospace, "SF Mono", Menlo, monospace; word-break: break-all; }
.checks { display: grid; grid-template-columns: 250px 90px 1fr; gap: 0; font-size: .86rem;
          border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.checks > div { padding: 8px 12px; border-bottom: 1px solid var(--border); }
.checks > div:nth-last-child(-n+3) { border-bottom: none; }
.checks .ck-f { font-weight: 600; }
.checks .ck-e { color: var(--muted); }
.ck-s { display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .74rem; font-weight: 600; }
.ck-seen, .ck-on { background: rgba(3,239,98,.18); color: var(--accent-ink); }
.ck-partly { background: rgba(196,130,14,.16); color: var(--warn); }
.ck-not-yet, .ck-off { background: rgba(90,104,114,.12); color: var(--muted); }

.hint { background: #fff; border: 1px dashed var(--border); border-radius: 14px; padding: 16px 18px; color: var(--muted); }
.mic-note { color: var(--muted); font-size: .84rem; margin: .1rem 0 .5rem; }
[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--border); }
hr { border-color: var(--border); margin: .7rem 0; }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def _data_uri(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def topbar() -> None:
    """DataCamp + Grok logos side by side, same shape as the GPT-Live-1 demo."""
    dc = _data_uri(str(DATACAMP_LOGO))
    grok = _data_uri(str(GROK_LOGO))
    dc_html = f"<img src='{dc}' alt='DataCamp'/>" if dc else "<span>DataCamp</span>"
    grok_html = f"<img src='{grok}' alt='Grok'/>" if grok else "<span>Grok</span>"
    st.markdown(
        f"""
        <div class="topbar">
          <div class="topbar-brand">
            <a href="https://www.datacamp.com/blog" target="_blank" rel="noopener">{dc_html}</a>
            <div class="topbar-divider"></div>
            {grok_html}
          </div>
          <a class="pill-link" href="https://docs.x.ai/developers/model-capabilities/audio/speech-to-text"
             target="_blank" rel="noopener">Grok Voice Transcribe 2.0 docs</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(brain_model: str) -> None:
    st.markdown(
        f"""
        <div class="hero-badge">\U0001F3A7 Companion demo &middot; Grok Voice Transcribe 2.0 article</div>
        <div class="hero-title">Qivora Sync &mdash; live call demo</div>
        <p class="hero-sub">You are Khalid, calling Qivora Sync support on your own microphone. Nothing
        is scripted: <code>{html.escape(brain_model)}</code> writes what Maya or Nadia says back to what you
        actually said, Grok TTS speaks it as it is written, and one live <code>grok-voice-transcribe-2.0</code>
        session transcribes and diarizes all three of you as you talk.</p>
        """,
        unsafe_allow_html=True,
    )


FEATURES = [
    ("\U0001F4AC", "Real replies, no script", "Maya and Nadia answer what you actually said; a Grok model writes each line inside a loose support scenario, streamed straight into Grok TTS."),
    ("\U0001F5E3\uFE0F", "Speaker diarization", "You, Maya, and Nadia are all transcribed by the same Transcribe 2.0 session and told apart by voice as words arrive."),
    ("\U0001F3F7\uFE0F", "Keyterm biasing", "\u201cQivora Sync\u201d is pinned as a keyterm so the invented brand name comes back spelled right instead of \u201cKivora Sync.\u201d"),
    ("\u23F8\uFE0F", "Smart Turn ends your turn", "No button: pause and Smart Turn closes your turn from the silence your mic keeps streaming. Talk over an agent and it stops to listen."),
]


def feature_cards() -> None:
    cols = st.columns(2)
    for i, (icon, name, desc) in enumerate(FEATURES):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="feature-card">
                  <div class="feature-icon">{icon}</div>
                  <div class="feature-name">{name}</div>
                  <p class="feature-desc">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def status_chips(turns_done: int, speaking: str, model: str, last_latency_ms: int | None, diarize: bool, format_: bool,
                 phone_line: dict | None = None) -> None:
    latency = f"{last_latency_ms / 1000:.1f}s" if last_latency_ms is not None else "\u2014"
    chips = [
        ("Turn", str(turns_done + 1), None),
        ("Floor", speaking, None),
        ("Brain", model.replace("-0309", ""), None),
        ("Last reply", latency, None),
        ("Diarize \u00b7 ITN", f"{'on' if diarize else 'off'} \u00b7 {'on' if format_ else 'off'}", None),
    ]
    if phone_line is not None:
        chips.append(("Phone line", f"Transcribe on 8 kHz \u03bc-law \u00b7 {phone_line['dropout_pct']:.1f}% lost", True))
    parts = ['<div class="status-row">']
    for label, value, state in chips:
        cls = "status-chip" + (f" {'on' if state else 'off'}" if state is not None else "")
        parts.append(f'<div class="{cls}"><div class="label">{label}</div><div class="value">{html.escape(value)}</div></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


PHASE_LABEL = {
    "connecting": ("Connecting...", "live-think"),
    "waiting_mic": ("Waiting for your microphone", "live-think"),
    "listening": ("Listening to you", "live-listen"),
    "thinking": ("Agent is replying", "live-think"),
    "speaking": ("Agent is speaking", "live-speak"),
    "ending": ("Wrapping up", "live-off"),
    "ended": ("Call ended", "live-off"),
    "error": ("Connection problem", "live-err"),
}


def render_statebar(phase: str, speaker: str | None, audio_seconds: float, last_latency_ms: int | None) -> str:
    label, cls = PHASE_LABEL.get(phase, (phase, "live-off"))
    if phase in ("thinking", "speaking") and speaker:
        name = SPEAKER_STYLE.get(speaker, {}).get("label", "Agent")
        label = f"{name} is {'speaking' if phase == 'speaking' else 'replying'}"
    cost = audio_seconds / 3600 * STREAMING_RATE_PER_HOUR
    pills = []
    if last_latency_ms is not None:
        pills.append(f'<span class="spill fast">last reply <b>{last_latency_ms / 1000:.1f}s</b></span>')
    pills.append(f'<span class="spill">audio streamed <b>{audio_seconds:.0f}s</b></span>')
    pills.append(f'<span class="spill dark">streaming cost <b>${cost:.5f}</b></span>')
    return (
        f'<div class="state"><span class="livedot {cls}"></span>'
        f'<span class="state-label">{label}</span>'
        f'<span class="state-pills">{"".join(pills)}</span></div>'
    )


def _eq(level: float, cls: str) -> str:
    shape = (0.55, 0.85, 1.0, 0.75, 0.5)
    bars = "".join(
        f'<i style="height:{max(4, int(4 + 30 * min(1.0, (level ** 0.6) * s)))}px"></i>' for s in shape
    )
    return f'<span class="eq {cls}">{bars}</span>'


def _halo(level: float, cls: str = "") -> str:
    scale = 1.0 + 0.35 * min(1.0, level ** 0.6)
    opacity = 0.35 + 0.65 * min(1.0, level ** 0.6)
    return f'<div class="halo {cls}" style="transform:scale({scale:.2f});opacity:{opacity:.2f}"></div>'


def _tag_html(tag: str) -> str:
    return {
        "interim": '<span class="tag t-live">\u23FA interim</span>',
        "locked": '<span class="tag">\U0001F512 locked</span>',
        "final": '<span class="tag t-final">\u2705 final</span>',
    }.get(tag, "")


def _growing_text(done: str, tail: str) -> str:
    """Settled words in full ink, the still-changing interim tail dimmed."""
    parts = []
    if done:
        parts.append(html.escape(done))
    if tail:
        parts.append(f'<span class="interim">{html.escape(tail)}</span>')
    return f'<div class="stage-text">{" ".join(parts)}</div>'


def _eotc_readout(threshold: float, last: float | None) -> str:
    """Smart Turn's rule, with the confidence the last turn actually closed at.
    Interim events always report 0.0; the server only fills the confidence in
    on the event that ends the turn, so there is no live meter to show."""
    rule = f"Smart Turn ends your turn once end-of-turn confidence reaches <b>{threshold:.2f}</b>"
    if last is None:
        return rule + "."
    pct = max(0.0, min(1.0, last)) * 100
    bar = (f'<span class="eotc-bar"><span style="width:{pct:.0f}%"></span>'
           f'<i style="left:{threshold * 100:.0f}%"></i></span>')
    return f"{rule} {bar} your last turn closed at <b>{last:.2f}</b>."


def render_stage(snap: dict, first_turn: bool) -> str:
    """The one card that always shows who has the floor."""
    phase = snap["phase"]
    speaker = snap.get("speaker")

    if phase in ("connecting", "waiting_mic"):
        if snap.get("audio_path") == "local":
            body = (
                '<div class="stage-title">Opening your microphone<span class="dots"><span></span><span></span><span></span></span></div>'
                '<div class="stage-hint">If this doesn\u2019t change within a couple of seconds, the selected device '
                "isn\u2019t delivering audio - end the call and pick another microphone.</div>"
            )
        else:
            body = (
                '<div class="stage-title">Connect your microphone <span class="tag t-think">step 1</span></div>'
                '<div class="stage-hint">Click <b>Connect microphone</b> above and allow access in the browser, '
                "and use <b>SELECT DEVICE</b> to pick the mic you talk into. Maya and Nadia's voices play through "
                "the same connection. Headphones are best: they stop the agents hearing themselves.</div>"
            )
        return f'<div class="stage st-thinking"><div class="avatar-wrap"><div class="avatar av-idle">\U0001F399</div></div><div class="stage-body">{body}</div></div>'

    if phase == "listening":
        text, tag = snap["live_text"], snap["live_tag"]
        level = snap["mic_level"]
        if text:
            content = _growing_text(snap.get("live_done", ""), snap.get("live_tail", "")) if "live_done" in snap \
                else _growing_text("", text)
        elif first_turn:
            content = '<div class="stage-hint">Say hello and tell Maya what\u2019s wrong - she picks up the moment you pause.</div>'
        else:
            content = '<div class="stage-hint">Go ahead - pause when you\u2019re done and the call moves on by itself.</div>'
        foot = ""
        if snap.get("speculating"):
            foot = '<div class="stage-foot">\u26A1 A reply is already being drafted while Smart Turn confirms you\u2019re done.</div>'
        elif not snap.get("mic_ok"):
            foot = '<div class="stage-foot" style="color:var(--danger)">No audio is arriving from your microphone - check the device, or reconnect it above.</div>'
        elif (snap.get("mic_stats") or {}).get("gated"):
            foot = ('<div class="stage-foot" style="color:var(--warn)">Windows is noise-gating your mic (Voice Clarity / '
                    "audio enhancements): quiet syllables arrive as digital silence and words go missing. Turn it off in "
                    "Settings &gt; System &gt; Sound &gt; your mic &gt; Audio enhancements: Off, or use the "
                    "\u201cThis computer\u201d audio path, which captures the mic raw.</div>")
        elif snap.get("stt_silent"):
            foot = ('<div class="stage-foot" style="color:var(--danger)">Your mic hears you, but Transcribe hasn\u2019t '
                    "returned any words for a while - try a higher Mic boost or a lower voice-activity sensitivity, "
                    "or switch to the \u201cThis computer\u201d audio path.</div>")
        elif snap.get("only_silence"):
            where = ("pick another microphone in the call settings" if snap.get("audio_path") == "local"
                     else "check SELECT DEVICE above - a virtual mic such as WO Mic stays silent")
            foot = ('<div class="stage-foot" style="color:var(--warn)">Not hearing any speech yet. If you are talking, '
                    f"the wrong input is probably selected: {where}.</div>")
        elif snap.get("smart_turn") is not None:
            foot = f'<div class="stage-foot">{_eotc_readout(snap["smart_turn"], snap.get("last_eotc"))}</div>'
        title = (
            '<div class="stage-title">Your turn <span class="tag t-live">\u25CF speak now</span>'
            f'{_tag_html(tag) if text else ""}</div>'
        )
        return (
            '<div class="stage st-listening">'
            f'<div class="avatar-wrap">{_halo(level)}<div class="avatar">K</div></div>'
            f'<div class="stage-body">{title}{content}{foot}</div>{_eq(level, "e-live")}</div>'
        )

    if phase == "thinking":
        style = SPEAKER_STYLE.get(speaker or "", {})
        name = style.get("label", "Maya")
        initial = style.get("initial", "\u2026")
        av = f"av-{speaker}" if speaker else "av-idle"
        return (
            '<div class="stage st-thinking">'
            f'<div class="avatar-wrap"><div class="halo h-think"></div><div class="avatar {av}">{initial}</div></div>'
            f'<div class="stage-body"><div class="stage-title">{name} is replying '
            f'<span class="tag t-think">{snap["thinking_s"]:.1f}s</span></div>'
            '<div class="stage-text" style="color:var(--muted)">Writing the reply<span class="dots"><span></span><span></span><span></span></span></div>'
            "</div></div>"
        )

    if phase == "speaking":
        # Only what Transcribe 2.0 has heard so far, growing like the caller's
        # bubble - not the written reply, which used to appear in full ahead of
        # the audio and then again underneath as the transcript caught up.
        style = SPEAKER_STYLE.get(speaker or "maya", SPEAKER_STYLE["maya"])
        level = snap["play_level"]
        if snap.get("heard"):
            body = _growing_text(snap.get("heard_done", ""), snap.get("heard_tail", ""))
        else:
            body = (f'<div class="stage-text" style="color:var(--muted)">listening to {style["label"]}'
                    '<span class="dots"><span></span><span></span><span></span></span></div>')
        if snap.get("interrupted"):
            status = '<span class="tag t-err">interrupted</span>'
        elif snap.get("paused"):
            status = '<span class="tag t-think">\u23F8 paused - listening to you</span>'
        else:
            status = '<span class="tag t-speak">\U0001F50A speaking</span>'
        source = '<span class="stage-src">live transcript \u00b7 Transcribe 2.0</span>'
        return (
            f'<div class="stage st-speaking sp-{speaker}">'
            f'<div class="avatar-wrap">{_halo(level, "h-" + (speaker or "maya"))}<div class="avatar av-{speaker}">{style["initial"]}</div></div>'
            f'<div class="stage-body"><div class="stage-title">{style["label"]} <span class="stage-role">{style["role"]}</span>'
            f'{status}{_tag_html(snap["heard_tag"]) if snap.get("heard") else ""}</div>'
            f'{body}{source}</div>{_eq(level, "e-" + (speaker or "maya"))}</div>'
        )

    if phase == "error":
        return (
            '<div class="stage st-error"><div class="avatar-wrap"><div class="avatar" style="background:var(--danger)">!</div></div>'
            '<div class="stage-body"><div class="stage-title">The transcription session dropped <span class="tag t-err">error</span></div>'
            f'<div class="stage-hint">{html.escape(snap.get("error") or "unknown error")}. Reconnect to carry on - '
            "the conversation so far is kept.</div></div></div>"
        )

    return (
        '<div class="stage"><div class="avatar-wrap"><div class="avatar av-idle">\u2713</div></div>'
        '<div class="stage-body"><div class="stage-title">Call ended</div>'
        '<div class="stage-hint">See the Transcript and Structured record tabs for the full result.</div></div></div>'
    )


def _db_pos(dbfs: float) -> float:
    """Position on a -80..0 dBFS bar, in percent."""
    return max(0.0, min(100.0, (dbfs + 80.0) / 80.0 * 100.0))


def render_checks(features: list[dict]) -> str:
    """The Tutorial checks table: feature, status pill, what the call showed."""
    cells = []
    for f in features:
        status = f["status"]
        cells.append(
            f'<div class="ck-f">{html.escape(f["feature"])}</div>'
            f'<div><span class="ck-s ck-{status.replace(" ", "-")}">{html.escape(status)}</span></div>'
            f'<div class="ck-e">{html.escape(f["evidence"])}</div>'
        )
    return f'<div class="checks">{"".join(cells)}</div>'


def render_diagnostics(snap: dict, device_label: str) -> str:
    """What the mic is doing right now: level against the room and the
    speech threshold, where the audio is going, and whether Transcribe is
    answering. Everything here also lands in the per-call JSONL log."""
    import math

    m = snap.get("mic_stats") or {}
    level_db = m.get("dbfs", -120.0)
    floor_db = m.get("floor_dbfs", -120.0)
    thr_db = 20 * math.log10(max(m.get("speech_thr", 1e-6), 1e-6))
    speaking = m.get("speaking_now")
    bar = (
        '<div class="diag-bar">'
        f'<div class="diag-fill{" on" if speaking else ""}" style="width:{_db_pos(level_db):.1f}%"></div>'
        f'<div class="diag-mark floor" style="left:{_db_pos(floor_db):.1f}%" title="room noise"></div>'
        f'<div class="diag-mark thr" style="left:{_db_pos(thr_db):.1f}%" title="speech threshold"></div>'
        "</div>"
        '<div class="diag-legend"><span>-80 dBFS</span><span>'
        f'now {level_db:.0f} dB \u00b7 room {floor_db:.0f} dB \u00b7 speech above {thr_db:.0f} dB'
        "</span><span>0</span></div>"
    )
    partial_age = snap.get("stt_last_partial_age_s")
    rows = [
        ("Input", f"{'This computer' if snap.get('audio_path') == 'local' else 'Browser (WebRTC)'} \u00b7 {device_label}"),
        ("Frames arriving", f"{m.get('frames_per_s', 0)}/s (expect ~50)"),
        ("You right now", "speaking" if speaking else "quiet"),
        ("Mic gate", {"open": "open - sent to Transcribe", "hold": "holding (agent turn closing)",
                      "closed": "closed (agent speaking; measured only)"}.get(m.get("mode"), m.get("mode", "?"))),
        ("Sent to Transcribe", f"{m.get('forwarded_s', 0):.0f}s of your audio"),
        ("Transcript events", f"{snap.get('stt_partials', 0)} \u00b7 last {partial_age:.0f}s ago" if partial_age is not None else "none yet"),
        ("Mic boost", f"{m.get('boost', 1.0):.1f}x"),
        ("Noise gate", f"{m.get('gated_pct', 0)}% of quiet frames are digital silence"
                       + (" - Windows audio enhancements are gating this mic" if m.get("gated") else " (a live mic is near 0%)")),
        ("Echo reaching the mic", f"{m.get('coupling_db', 0):.0f} dB of the agent's playback"
                                  + ("" if m.get("coupling_measured") else " (estimate - measured once an agent speaks)")),
        ("Barge-in", f"pauses above {20 * math.log10(max(m.get('barge_thr', 1e-6), 1e-6)):.0f} dB, cuts on your words"
                     + ("" if snap.get("listener_ok") else " (listener not connected - loudness only)")),
    ]
    phantoms = snap.get("phantoms_ignored") or []
    if phantoms:
        rows.append(("Agent echo ignored", f"{len(phantoms)} turn(s), last: \u201c{phantoms[-1]}\u201d"))
    if m.get("errors"):
        rows.append(("Mic errors", f"{m['errors']} - {m.get('first_error')}"))
    if snap.get("log_path"):
        rows.append(("Call log", snap["log_path"]))
    table = "".join(f"<div class='diag-k'>{html.escape(k)}</div><div class='diag-v'>{html.escape(str(v))}</div>" for k, v in rows)
    return f'<div class="diag">{bar}<div class="diag-grid">{table}</div></div>'


def _latency_badge(ms: int | None) -> str:
    if ms is None:
        return ""
    cls = "fast" if ms < 3000 else "slow"
    return f'<span class="badge {cls}">replied in {ms / 1000:.1f}s</span>'


def _turn_chips(turn: dict) -> str:
    chips = []
    if turn.get("source") == "phone":
        chips.append('<span class="tchip phone">PHONE &middot; 8 kHz</span>')
    elif turn.get("source") == "mic":
        chips.append('<span class="tchip">MIC &middot; 16 kHz</span>')
    ids = turn.get("speaker_ids") or []
    if ids:
        chips.append(f'<span class="tchip">SPK {"+".join(str(i) for i in ids)}</span>')
    if turn.get("eotc") is not None:
        chips.append(f'<span class="tchip">END-OF-TURN {turn["eotc"]:.2f}</span>')
    if turn.get("language", "").startswith("ar"):
        chips.append('<span class="tchip">AR</span>')
    return f'<div class="tchips">{"".join(chips)}</div>' if chips else ""


def render_conversation(turns: list[dict], *, full: bool = False, highlight_last: bool = False, show_said: bool = False) -> str:
    """Finalized turns as chat bubbles, one color per speaker, newest at
    the bottom. Every bubble shows what Transcribe 2.0 heard - that is the
    point of the demo; `show_said` adds what the model actually wrote."""
    if not turns:
        return '<div class="hint">Nothing yet - say hello and Maya picks up.</div>'
    out = []
    for i, turn in enumerate(turns):
        speaker = turn.get("speaker", "system")
        text = html.escape(turn["text"])  # recognition and model output go into raw HTML
        if speaker == "system":
            out.append(f'<div class="bubble bubble-system">{text}</div>')
            continue
        style = SPEAKER_STYLE.get(speaker, {"label": speaker.capitalize(), "role": ""})
        flash = " just-finalized" if (highlight_last and i == len(turns) - 1) else ""
        badges = _latency_badge(turn.get("latency_ms"))
        if turn.get("interrupted"):
            badges += '<span class="badge cut">interrupted</span>'
        said = ""
        if show_said and turn.get("said") and turn["said"].strip() != turn["text"].strip():
            said = f'<div class="bubble-said"><b>Written:</b> {html.escape(turn["said"])}</div>'
        # Compact chips in the live view (LiveOps-style provenance: where the
        # words came from, who Transcribe thought it was, how sure it was the
        # turn had ended); the full sentence stays in the Transcript tab.
        chips = _turn_chips(turn)
        meta = f'<div class="bubble-meta">{html.escape(turn["meta"])}</div>' if full and turn.get("meta") else ""
        out.append(
            f'<div class="bubble bubble-{speaker}{flash}">'
            f'<div class="bubble-who">{style["label"]} &middot; {style.get("role", "")} {badges}</div>'
            f'<div class="bubble-text" dir="auto">{text}</div>{said}{chips}{meta}</div>'
        )
    cls = "convo full" if full else "convo"
    return f'<div class="{cls}"><div>{"".join(out)}</div></div>'
