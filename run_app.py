"""Launch the Qivora Sync live call demo.

    python run_app.py            (from anywhere)

Replaces the shell-specific .env export dance (PowerShell and bash need
different one-liners, and a Windows-written .env carries a trailing \\r
that breaks the Authorization header - experiment_log.md, "Known real
bug"). It also installs the WinError 10054 fix before Streamlit's server
starts, so resets that happen before the first page load stay quiet too.
Extra arguments are passed through to `streamlit run`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "project"


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip().replace("\r", "")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        # A variable already set in the shell wins over the file.
        os.environ.setdefault(key, value)


def main() -> int:
    load_env(ROOT / ".env")
    if not os.environ.get("XAI_API_KEY", "").strip():
        print("XAI_API_KEY is not set (looked in the shell and in .env).", file=sys.stderr)
        return 1

    sys.path.insert(0, str(PROJECT_DIR))
    import log_hygiene
    import win_asyncio_fix

    win_asyncio_fix.install()
    log_hygiene.install()

    # Streamlit reads .streamlit/config.toml from the working directory.
    os.chdir(PROJECT_DIR)
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(ROOT / "app_streamlit.py"), *sys.argv[1:]]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
