"""Direct microphone and speaker access for the live call ("This computer").

The browser path (streamlit-webrtc) depends on which input the browser
picked, on its gain and noise processing, and on a WebRTC connection; when
any of those goes wrong the call simply "doesn't hear you". The demo runs on
localhost anyway, so this path opens the devices straight from Python with
sounddevice - the same pattern as the GPT Live Transcribe project's
mic_stream.py - and lets you pick the exact mic and speakers.

- Input: 16 kHz mono int16 in 20 ms blocks, straight into MicInput.feed_pcm.
- Output: the agent voice, with the same push()/clear() interface as the
  WebRTC PcmAudioSource, so CallEngine doesn't know which path it is on.
- If a device refuses 16 kHz, it is opened at its own rate and resampled.

There is no echo canceller on this path, so the agent's voice from open
speakers reaches the mic; the app turns barge-in off here unless you say
you are on headphones.
"""
from __future__ import annotations

import threading

import numpy as np

SAMPLE_RATE = 16000
BLOCK_S = 0.02


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = max(1, int(round(x.size * dst / src)))
    return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x.astype(np.float32)).astype(np.float32)


def list_devices() -> dict:
    """Input and output devices on the default host API (one entry per
    physical device instead of the same mic listed under MME, DirectSound
    and WASAPI), plus the system defaults."""
    import sounddevice as sd

    devices = sd.query_devices()
    try:
        hostapi = sd.default.hostapi
    except Exception:  # noqa: BLE001
        hostapi = 0
    default_in, default_out = sd.default.device
    inputs, outputs = [], []
    for i, d in enumerate(devices):
        if d["hostapi"] != hostapi:
            continue
        if d["max_input_channels"] > 0:
            inputs.append((i, d["name"]))
        if d["max_output_channels"] > 0:
            outputs.append((i, d["name"]))
    return {"inputs": inputs, "outputs": outputs, "default_in": default_in, "default_out": default_out}


class LocalAudio:
    def __init__(self, mic, input_device: int | None = None, output_device: int | None = None) -> None:
        self.mic = mic
        self.input_device = input_device
        self.output_device = output_device
        self._lock = threading.Lock()
        self._out = bytearray()
        self._in_stream = None
        self._out_stream = None
        self.in_rate = SAMPLE_RATE
        self.out_rate = SAMPLE_RATE
        self.errors: list[str] = []
        self.status_flags = 0
        # Seconds between push() and the sound leaving the speakers; the
        # engine adds it to its echo tail before reopening the mic.
        self.latency = 0.0

    # ------------------------------------------------------------------
    # playback interface (same as streamlit_webrtc.PcmAudioSource)
    # ------------------------------------------------------------------
    def push(self, pcm: bytes) -> None:
        with self._lock:
            self._out.extend(pcm)

    def clear(self) -> None:
        with self._lock:
            self._out = bytearray()

    # ------------------------------------------------------------------
    # streams
    # ------------------------------------------------------------------
    def _open(self, factory, device, callback):
        """Open at 16 kHz, or at the device's own rate if it refuses."""
        import sounddevice as sd

        try:
            stream = factory(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                             blocksize=int(SAMPLE_RATE * BLOCK_S), device=device, callback=callback)
            return stream, SAMPLE_RATE
        except Exception as exc:  # noqa: BLE001 - retry at the native rate
            self.errors.append(f"16 kHz refused ({exc}); using the device rate")
        kind = "input" if factory is sd.InputStream else "output"
        rate = int(sd.query_devices(device, kind)["default_samplerate"])
        stream = factory(samplerate=rate, channels=1, dtype="int16",
                         blocksize=int(rate * BLOCK_S), device=device, callback=callback)
        return stream, rate

    def _on_input(self, indata, frames, time_info, status) -> None:
        if status:
            self.status_flags += 1
        try:
            x = indata[:, 0]
            if self.in_rate != SAMPLE_RATE:
                x = np.clip(_resample(x, self.in_rate, SAMPLE_RATE), -32768, 32767).astype(np.int16)
            self.mic.feed_pcm(x.astype(np.int16).tobytes())
        except Exception as exc:  # noqa: BLE001 - never raise inside the audio callback
            if len(self.errors) < 20:
                self.errors.append(f"input: {exc}")

    def _on_output(self, outdata, frames, time_info, status) -> None:
        try:
            need = frames if self.out_rate == SAMPLE_RATE else int(np.ceil(frames * SAMPLE_RATE / self.out_rate)) + 1
            with self._lock:
                chunk = bytes(self._out[: need * 2])
                del self._out[: need * 2]
            x = np.frombuffer(chunk.ljust(need * 2, b"\0"), dtype=np.int16)
            if self.out_rate != SAMPLE_RATE:
                x = np.clip(_resample(x, SAMPLE_RATE, self.out_rate), -32768, 32767)
            y = np.zeros(frames, dtype=np.int16)
            y[: min(frames, x.size)] = x[:frames]
            outdata[:, 0] = y
        except Exception as exc:  # noqa: BLE001
            outdata.fill(0)
            if len(self.errors) < 20:
                self.errors.append(f"output: {exc}")

    def start(self) -> None:
        import sounddevice as sd

        self._in_stream, self.in_rate = self._open(sd.InputStream, self.input_device, self._on_input)
        self._out_stream, self.out_rate = self._open(sd.OutputStream, self.output_device, self._on_output)
        self._in_stream.start()
        self._out_stream.start()
        try:
            self.latency = float(self._out_stream.latency) + BLOCK_S
        except Exception:  # noqa: BLE001
            self.latency = 0.1

    def stop(self) -> None:
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
        self._in_stream = self._out_stream = None
        self.clear()
