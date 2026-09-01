"""Sample-rate conversion for the local audio backends.

The macOS voice-processing unit picks its own capture and render rates and does
not always pick the same ones twice, so the client has to convert between whatever
CoreAudio offers and the 16 kHz PCM the Realtime protocol carries. Conversion runs
per callback, which means it has to be continuous across chunk boundaries: a
stateless resampler rings at every seam and the artefacts land in the recognizer.
"""

from __future__ import annotations

from math import ceil, gcd

import numpy as np
from scipy.signal import firwin, resample_poly

_PCM16_SCALE = 32768.0


def pcm16_to_float32(pcm: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Decode mono ``int16`` PCM (raw bytes or an array) to ``float32`` in [-1, 1)."""

    samples = pcm if isinstance(pcm, np.ndarray) else np.frombuffer(pcm, dtype=np.int16)
    return (samples.astype(np.float32) / _PCM16_SCALE).astype(np.float32)


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Encode ``float32`` samples as mono ``int16`` PCM, clipping rather than wrapping."""

    scaled = np.clip(samples.astype(np.float32) * _PCM16_SCALE, -_PCM16_SCALE, _PCM16_SCALE - 1)
    return scaled.astype(np.int16).tobytes()


class StreamResampler:
    """Rational resampler that stays continuous across successive chunks.

    ``resample_poly`` zero-pads *both* ends of whatever it is given, so a chunk
    resampled on its own rings at its start and decays at its finish. Feeding it
    real audio on both sides is the whole trick: each conversion runs over a
    window laid out as context, the new samples, then lookahead, and only the middle
    is emitted. That costs one lookahead window of latency -- 1.4 ms converting 48 kHz
    down to 16 kHz, 4 ms going back up, and 10 ms for a 44.1 kHz device -- and removes
    the seam entirely.
    """

    #: Input samples of context kept on each side. The filter's transient half-width is
    #: ``10 * max(up, down) / up`` input samples: 10 for the ratios CoreAudio reports
    #: when upsampling, 30 when decimating 48 kHz to 16 kHz. This covers both.
    _CONTEXT = 64

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError(f"Sample rates must be positive, got {src_rate} -> {dst_rate}")
        divisor = gcd(int(src_rate), int(dst_rate))
        self._up = int(dst_rate) // divisor
        self._down = int(src_rate) // divisor
        # Rounding the window to a whole number of input periods keeps every
        # conversion boundary on an exact output sample, so nothing drifts.
        self._pad = ceil(self._CONTEXT / self._down) * self._down
        self._buffer = np.zeros(self._pad, dtype=np.float32)
        # resample_poly designs its filter on every call. That is cheap for the small
        # ratios CoreAudio usually reports, but a 44.1 kHz device gives up=441 and an
        # ~8800-tap design -- several hundred times a second, on the feeder thread.
        # Reproduce its default Kaiser design once and hand the taps over instead.
        max_rate = max(self._up, self._down)
        self._taps = (
            None if max_rate == 1 else firwin(2 * 10 * max_rate + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * self._up
        )

    @property
    def passthrough(self) -> bool:
        return self._up == self._down

    @property
    def lookahead_frames(self) -> int:
        """Output frames withheld at any moment so the next chunk has real context."""

        return 0 if self.passthrough else (self._pad * self._up) // self._down

    def process(self, block: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return block

        buffer = np.concatenate((self._buffer, block)) if len(block) else self._buffer
        # Leave `_pad` samples of context behind and `_pad` of lookahead ahead.
        consume = ((len(buffer) - 2 * self._pad) // self._down) * self._down
        if consume <= 0:
            self._buffer = buffer
            return np.zeros(0, dtype=np.float32)

        converted = resample_poly(buffer, self._up, self._down, window=self._taps).astype(np.float32)
        start = (self._pad * self._up) // self._down
        count = (consume * self._up) // self._down
        self._buffer = buffer[consume:]
        return converted[start : start + count]
