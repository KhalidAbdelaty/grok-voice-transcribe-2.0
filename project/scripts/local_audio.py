"""Direct microphone and speaker access for the live call ("This computer").

The browser path (streamlit-webrtc) depends on which input the browser
picked, on its gain and noise processing, and on a WebRTC connection; when
any of those goes wrong the call simply "doesn't hear you". The demo runs on
localhost anyway, so this path opens the devices straight from Python with
sounddevice - the same pattern as the GPT Live Transcribe project's
mic_stream.py - and lets you pick the exact mic and speakers.

- Input: raw capture first - the chosen mic's WASAPI twin in exclusive
  mode, at its native rate, resampled to 16 kHz with PyAV. Shared capture
  (MME, WASAPI shared, and every browser) runs through the Windows audio
  engine's effects (Voice Clarity, noise suppression), and on a GM301
  headset those gate silence to exact digital zero and chop quiet
  syllables; exclusive mode bypasses them. If exclusive is refused (another
  app holds the mic), it falls back to shared capture and says so.
- Output: the agent voice, with the same push()/clear() interface as the
  WebRTC PcmAudioSource, so CallEngine doesn't know which path it is on.
- If an output device refuses 16 kHz, it is opened at its own rate.

There is no echo canceller on this path, so the agent's voice from open
speakers reaches the mic; the app turns barge-in off here unless you say
you are on headphones.
"""
from __future__ import annotations

import threading

import numpy as np

SAMPLE_RATE = 16000
BLOCK_S = 0.02

# Every LocalAudio with open streams. Exclusive capture locks the mic: while
# one is open, any other open of that mic fails with "Device unavailable
# [PaErrorCode -9985]". A call left behind by a reloaded tab keeps its
# streams until the engine notices (30 s), so a new call releases it first.
_active: set["LocalAudio"] = set()
_active_lock = threading.Lock()


def _explain(exc: Exception) -> str:
    text = str(exc)
    if "-9985" in text or "Device unavailable" in text:
        return (f"the microphone is in use by another program that has exclusive control of it ({text}). "
                "Close that program or pick another mic, or turn off 'Allow applications to take exclusive "
                "control of this device' in the mic's Sound properties (Advanced tab).")
    return text


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


def wasapi_twin(device: int | None) -> int | None:
    """The WASAPI index of the same physical mic as `device` (an index on
    the default host API, usually MME). MME cuts names at 31 characters, so
    the WASAPI name only has to start with the MME one."""
    import sounddevice as sd

    try:
        name = sd.query_devices(device if device is not None else sd.default.device[0])["name"]
        wasapi = next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])
    except Exception:  # noqa: BLE001 - not Windows, or no WASAPI
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == wasapi and d["max_input_channels"] > 0 and d["name"].startswith(name):
            return i
    return None


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
        self.capture = "not started"  # "raw (WASAPI exclusive)" or "shared (...)", for the diagnostics
        self._resampler = None
        self._in_channels = 1
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

    def _open_raw_input(self):
        """The mic's WASAPI twin in exclusive mode (bypasses Windows audio
        effects), at its native rate; None if there is no twin or it is
        refused."""
        import sounddevice as sd

        twin = wasapi_twin(self.input_device)
        if twin is None:
            self.errors.append("no WASAPI device for this mic; using shared capture")
            return None
        info = sd.query_devices(twin)
        rate = int(info["default_samplerate"])
        for channels in dict.fromkeys((1, int(info["max_input_channels"]))):
            try:
                stream = sd.InputStream(
                    samplerate=rate, channels=channels, dtype="int16", blocksize=int(rate * BLOCK_S),
                    device=twin, callback=self._on_input, extra_settings=sd.WasapiSettings(exclusive=True),
                )
                self._in_channels = channels
                return stream, rate
            except Exception as exc:  # noqa: BLE001 - try the device's own channel count, then give up
                last = exc
        self.errors.append(f"exclusive capture refused ({last}); using shared capture")
        return None

    def _to_16k(self, x: np.ndarray) -> np.ndarray:
        """Native-rate int16 -> 16 kHz int16. PyAV's resampler is stateful
        and low-passes, unlike per-block interpolation, which clicks at block
        edges and aliases when dropping 48 kHz to 16 kHz."""
        if self.in_rate == SAMPLE_RATE:
            return x
        try:
            import av

            if self._resampler is None:
                self._resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            frame = av.AudioFrame.from_ndarray(x.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = self.in_rate
            out = [f.to_ndarray().reshape(-1) for f in self._resampler.resample(frame)]
            return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)
        except Exception:  # noqa: BLE001 - fall back to plain interpolation
            return np.clip(_resample(x, self.in_rate, SAMPLE_RATE), -32768, 32767).astype(np.int16)

    def _on_input(self, indata, frames, time_info, status) -> None:
        if status:
            self.status_flags += 1
        try:
            x = np.ascontiguousarray(indata[:, 0], dtype=np.int16)
            x = self._to_16k(x)
            if x.size:
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

        with _active_lock:
            others = [a for a in _active if a is not self]
        for other in others:
            other.stop()
        with _active_lock:
            _active.add(self)
        try:
            raw = self._open_raw_input()
            if raw is not None:
                self._in_stream, self.in_rate = raw
                self.capture = f"raw (WASAPI exclusive, {self.in_rate // 1000} kHz)"
            else:
                self._in_stream, self.in_rate = self._open(sd.InputStream, self.input_device, self._on_input)
                api = sd.query_hostapis(sd.query_devices(self.input_device if self.input_device is not None
                                                         else sd.default.device[0])["hostapi"])["name"]
                self.capture = f"shared ({api}) - Windows audio effects apply"
            self._out_stream, self.out_rate = self._open(sd.OutputStream, self.output_device, self._on_output)
            self._in_stream.start()
            self._out_stream.start()
        except Exception as exc:
            # Close what did open, or the mic stays locked for the next call.
            self.stop()
            raise RuntimeError(_explain(exc)) from exc
        try:
            self.latency = float(self._out_stream.latency) + BLOCK_S
        except Exception:  # noqa: BLE001
            self.latency = 0.1

    def stop(self) -> None:
        for stream in (self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    try:
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass
        self._in_stream = self._out_stream = None
        with _active_lock:
            _active.discard(self)
        self._resampler = None
        self.clear()
