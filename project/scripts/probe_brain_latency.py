"""Real-API latency probe for scripts/agent_brain.py.

Streams the same short agent reply from each brain model, with and without
priority processing, and prints time to first text, total time, the tier
the API actually granted, and the reply itself.

    python run_app.py is not needed; export XAI_API_KEY (or keep it in .env)
    python project/scripts/probe_brain_latency.py [runs_per_config] [model,model,...]

The second argument probes other model IDs (list them with GET /v1/models),
with priority processing only.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from run_app import load_env  # noqa: E402
from scripts.agent_brain import MODELS, AgentBrain  # noqa: E402

HISTORY = [
    ("maya", "Thank you for calling Qivora Sync support, this is Maya. How can I help you today?"),
    ("khalid", "Hi, yeah, my files stopped syncing between my laptop and my phone since yesterday."),
]


def main() -> None:
    load_env(__import__("pathlib").Path(__file__).resolve().parents[2] / ".env")
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else MODELS
    for model in models:
        for priority in ((True,) if len(sys.argv) > 2 else (True, False)):
            brain = AgentBrain(model=model, priority=priority, history=list(HISTORY))
            t = time.monotonic()
            brain.warm_up()
            warm_ms = int((time.monotonic() - t) * 1000)
            for i in range(runs):
                stream = brain.reply_stream()
                first_speaker_ms = None
                for kind, _ in stream:
                    if kind == "speaker" and first_speaker_ms is None:
                        first_speaker_ms = int((time.monotonic() - stream.started_at) * 1000)
                r = stream.reply
                print(
                    f"{model:32s} priority={str(priority):5s} run {i + 1}: speaker@{first_speaker_ms}ms "
                    f"first-text@{stream.ttft_ms}ms total={stream.total_ms}ms tier={stream.service_tier} "
                    f"(warm-up {warm_ms}ms) -> {r.speaker}: {ascii(r.text)}"
                )


if __name__ == "__main__":
    main()
