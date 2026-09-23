"""Is Windows noise-gating your microphone?

Records 3 seconds of silence from the same physical mic through each
Windows audio path and reports how much of it is exact digital zero. A live
mic always carries a little noise; a path that returns mostly zeros is
running an audio effect (Voice Clarity / audio enhancements / noise
suppression) that also chops quiet syllables off your speech.

    python project/scripts/mic_check.py            # the default mic
    python project/scripts/mic_check.py GM301      # a mic whose name contains "GM301"

Stay quiet while it records. Shared paths (MME, WASAPI shared) are what the
browser and most apps use; WASAPI exclusive is what the app's "This computer"
path uses, and bypasses those effects.
"""
from __future__ import annotations

import sys

import numpy as np
import sounddevice as sd

SECONDS = 3


def find_inputs(fragment: str | None) -> dict[str, int]:
    """{host API name: device index} for one physical mic."""
    if fragment is None:
        fragment = sd.query_devices(sd.default.device[0])["name"][:24]
    found: dict[str, int] = {}
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and fragment.lower() in d["name"].lower():
            api = sd.query_hostapis(d["hostapi"])["name"]
            found.setdefault(api, i)
    return found


SILENCE_RMS = 2.5e-5  # -92 dBFS: below any live mic's noise, what a gate leaves


def measure(device: int, exclusive: bool = False) -> tuple[float, float, float]:
    """(% exact-zero samples, % 20 ms frames at digital silence, rms)."""
    info = sd.query_devices(device)
    rate = int(info["default_samplerate"])
    extra = sd.WasapiSettings(exclusive=True) if exclusive else None
    x = sd.rec(rate * SECONDS, samplerate=rate, channels=1, dtype="int16", device=device, extra_settings=extra)
    sd.wait()
    x = x[:, 0].astype(np.float32) / 32768
    block = int(rate * 0.02)
    frames = x[: len(x) // block * block].reshape(-1, block)
    silent = np.sqrt(np.mean(frames * frames, axis=1)) < SILENCE_RMS
    return 100 * float(np.mean(x == 0)), 100 * float(np.mean(silent)), float(np.sqrt(np.mean(x * x)))


def main() -> None:
    paths = find_inputs(sys.argv[1] if len(sys.argv) > 1 else None)
    if not paths:
        sys.exit("no matching microphone; pass part of its name, e.g. python mic_check.py GM301")
    runs = [(api, idx, False) for api, idx in paths.items() if "WDM-KS" not in api]
    runs += [("Windows WASAPI exclusive", idx, True) for api, idx in paths.items() if "WASAPI" in api]
    print(f"Recording {SECONDS} s of silence per path from: {sd.query_devices(next(iter(paths.values())))['name']}")
    print("Stay quiet.\n")
    print(f"{'path':28s} {'zero samples':>13s} {'silent frames':>14s} {'rms':>10s}")
    gated_shared = False
    for label, idx, exclusive in runs:
        try:
            zeros, zero_frames, rms = measure(idx, exclusive)
        except Exception as exc:  # noqa: BLE001 - e.g. exclusive refused while another app holds the mic
            print(f"{label:28s} failed: {exc}")
            continue
        flag = "  <- gated" if zero_frames > 50 else ""
        gated_shared |= bool(flag) and not exclusive
        print(f"{label:28s} {zeros:12.0f}% {zero_frames:13.0f}% {rms:10.6f}{flag}")
    if gated_shared:
        print("\nWindows is gating this mic in shared mode. Turn it off in Settings > System > Sound > "
              "(your mic) > Audio enhancements: Off (or Voice Clarity: Off), or use the app's "
              "\"This computer\" audio path, which records in exclusive mode.")
    else:
        print("\nNo gating found on the shared paths.")


if __name__ == "__main__":
    main()
