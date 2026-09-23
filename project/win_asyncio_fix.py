"""Silence the harmless WinError 10054 asyncio logs on Windows.

When a peer resets a socket (a browser tab reconnecting to Streamlit, an
aiortc or websocket teardown), asyncio's Proactor transport raises
ConnectionResetError inside `_call_connection_lost` and the default
exception handler prints a full traceback. Nothing is actually wrong.

`install()` must be idempotent: Streamlit re-executes the app script on
every rerun, and wrapping the method again each time would nest wrappers
until a connection loss hit the recursion limit. It also has to run before
the server starts (see run_app.py), or resets that happen before the first
page load still print.
"""
from __future__ import annotations

import sys
from functools import wraps

_SENTINEL = "_qivora_conn_reset_patched"


def install() -> None:
    if sys.platform != "win32":
        return
    from asyncio.proactor_events import _ProactorBasePipeTransport

    original = _ProactorBasePipeTransport._call_connection_lost
    if getattr(original, _SENTINEL, False):
        return

    @wraps(original)
    def _call_connection_lost(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except ConnectionResetError:
            return None

    setattr(_call_connection_lost, _SENTINEL, True)
    _ProactorBasePipeTransport._call_connection_lost = _call_connection_lost
