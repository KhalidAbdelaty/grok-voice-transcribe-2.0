"""A simulated phone line: what a caller sounds like after a real telephone
network, so Transcribe 2.0's 8 kHz mu-law path can be tried without a SIP
trunk.

The chain is the one a PSTN call goes through: band-limit to 300-3400 Hz,
resample to 8 kHz, G.711 mu-law encode (8 bits per sample), and lose ~3% of
the 20 ms packets. `PhoneLine.process` runs it on a live 16 kHz PCM16 stream
and hands back 16 kHz PCM16 (so the call keeps one session format);
`to_mulaw_8k` stops after the encoder and returns the raw bytes a telephony
provider would send, for streaming with `encoding=mulaw&sample_rate=8000`.

numpy only (no scipy): a windowed-sinc FIR and the G.711 bit layout.
"""

from __future__ import annotations

import numpy as np

IN_RATE = 16000
LINE_RATE = 8000
PACKET_S = 0.02
PACKET_SAMPLES = int(LINE_RATE * PACKET_S)  # 160 samples at 8 kHz
DEFAULT_DROPOUT = 0.03

_BIAS = 0x84
_CLIP = 32635


def bandpass_fir(lo: float = 300.0, hi: float = 3400.0, fs: int = IN_RATE, taps: int = 129) -> np.ndarray:
    """Linear-phase band-pass: difference of two Hamming-windowed sincs."""
    n = np.arange(taps) - (taps - 1) / 2
    def lowpass(fc: float) -> np.ndarray:
        return 2 * fc / fs * np.sinc(2 * fc / fs * n)
    h = (lowpass(hi) - lowpass(lo)) * np.hamming(taps)
    return h.astype(np.float32)


def mulaw_encode(pcm: np.ndarray) -> np.ndarray:
    """int16 samples -> G.711 mu-law bytes (uint8)."""
    s = pcm.astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    mag = np.minimum(np.abs(s), _CLIP) + _BIAS
    exponent = np.clip(np.floor(np.log2(mag)).astype(np.int32) - 7, 0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """G.711 mu-law bytes -> int16 samples."""
    u = ~codes.astype(np.int32) & 0xFF
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(u & 0x80, -mag, mag).astype(np.int16)


MULAW_SILENCE = int(mulaw_encode(np.zeros(1, dtype=np.int16))[0])


class PhoneLine:
    """Streaming 16 kHz PCM16 -> phone line -> 16 kHz PCM16. Stateful, so
    feeding 10 ms browser frames gives the same audio as one long buffer."""

    def __init__(self, dropout: float = DEFAULT_DROPOUT, seed: int | None = None) -> None:
        self.dropout = float(dropout)
        self._rng = np.random.default_rng(seed)
        self._fir = bandpass_fir()
        self._tail = np.zeros(len(self._fir) - 1, dtype=np.float32)
        self._carry = np.zeros(0, dtype=np.float32)  # odd sample left over before decimating
        self._line_pos = 0  # samples at 8 kHz sent so far
        self._packet_dropped = False
        self._last_line = 0.0
        self.packets = 0
        self.dropped = 0

    def _band_limit(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self._tail, x])
        y = np.convolve(buf, self._fir, mode="valid")
        self._tail = buf[-(len(self._fir) - 1):]
        return y

    def _drop_mask(self, n: int) -> np.ndarray:
        """True for 8 kHz samples inside a lost 20 ms packet."""
        mask = np.zeros(n, dtype=bool)
        for i in range(n):
            if (self._line_pos + i) % PACKET_SAMPLES == 0:
                self._packet_dropped = bool(self._rng.random() < self.dropout)
                self.packets += 1
                self.dropped += self._packet_dropped
            mask[i] = self._packet_dropped
        self._line_pos += n
        return mask

    def _to_line(self, pcm16k: bytes) -> np.ndarray:
        """Band-limited, decimated, dropout-marked 8 kHz int16 samples."""
        x = np.frombuffer(pcm16k[: len(pcm16k) & ~1], dtype=np.int16).astype(np.float32)
        y = np.concatenate([self._carry, self._band_limit(x)])
        even = len(y) & ~1
        self._carry = y[even:]
        # Band-limited below 3.4 kHz already, so every other sample is enough.
        line = np.clip(y[:even:2], -32768, 32767).astype(np.int16)
        self._mask = self._drop_mask(len(line))
        return line

    def to_mulaw_8k(self, pcm16k: bytes) -> bytes:
        """What a telephony provider would stream: 8 kHz mu-law bytes."""
        codes = mulaw_encode(self._to_line(pcm16k))
        codes[self._mask] = MULAW_SILENCE
        return codes.tobytes()

    def process(self, pcm16k: bytes) -> bytes:
        """The caller's audio as it arrives after the phone line, back at
        16 kHz PCM16 so it can go into the call's existing session."""
        line = mulaw_decode(mulaw_encode(self._to_line(pcm16k))).astype(np.float32)
        line[self._mask] = 0.0
        if line.size == 0:
            return b""
        # Linear interpolation back to 16 kHz, continuing from the last sample.
        prev = np.concatenate([[self._last_line], line[:-1]])
        up = np.empty(line.size * 2, dtype=np.float32)
        up[0::2] = (prev + line) / 2
        up[1::2] = line
        self._last_line = float(line[-1])
        return np.clip(up, -32768, 32767).astype(np.int16).tobytes()

    def stats(self) -> dict:
        return {"packets": self.packets, "dropped": self.dropped,
                "dropout_pct": round(100 * self.dropped / self.packets, 1) if self.packets else 0.0}
