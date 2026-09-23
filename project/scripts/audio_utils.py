"""Small audio helpers shared by the Streamlit app and the CLI experiment
scripts: convert whatever container/sample-rate audio comes in (browser
mic recordings, cached TTS mp3/wav lines) into the raw 16kHz mono PCM16
little-endian bytes the streaming STT socket expects.

Decoding goes through PyAV (already a dependency of the live app, and it
bundles its own FFmpeg libraries), so no `ffmpeg` binary has to be on
PATH; the ffmpeg CLI is only a fallback for formats PyAV refuses.
"""
from __future__ import annotations

import io
import shutil
import subprocess


def _decode_with_av(audio_bytes: bytes) -> bytes:
    import av

    out = bytearray()
    with av.open(io.BytesIO(audio_bytes)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for chunk in resampler.resample(frame):
                out.extend(chunk.to_ndarray().tobytes())
        for chunk in resampler.resample(None):
            out.extend(chunk.to_ndarray().tobytes())
    return bytes(out)


def to_pcm16_16k(audio_bytes: bytes) -> bytes:
    """Decode any container (wav, webm, mp3, ogg, ...) in memory and return
    raw s16le mono 16kHz PCM bytes (no header)."""
    try:
        return _decode_with_av(audio_bytes)
    except Exception:
        if shutil.which("ffmpeg") is None:
            raise
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", "pipe:0",
            "-ar", "16000", "-ac", "1", "-f", "s16le",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def wav_file_to_pcm16_16k(path: str) -> bytes:
    with open(path, "rb") as f:
        return to_pcm16_16k(f.read())
