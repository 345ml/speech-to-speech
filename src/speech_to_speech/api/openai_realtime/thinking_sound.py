"""The looping cue the local client plays while a turn is being generated.

Between the end of the user's speech and the first byte of response audio the
speaker is silent for as long as STT, the LLM and the first TTS chunk take.
This module supplies the waveform that fills that gap; ``PlaybackBuffer`` owns
when it starts, when it fades and how it is mixed with the response.
"""

from __future__ import annotations

import os

import numpy as np

DEFAULT_GAIN = 0.12
DEFAULT_DELAY_S = 0.3

# A three-note rise, then silence for the rest of the period. The period is set
# against the companion's p50 time to first audio (~2.1s): long enough that a
# normal wait hears the phrase once or twice rather than a stream of beeps,
# short enough that the first repeat lands before the wait feels unanswered.
_PULSE_PERIOD_S = 1.4
# (start second, frequency, duration second). C6, E6, G6.
_PULSES = (
    (0.00, 1046.5, 0.07),
    (0.09, 1318.5, 0.07),
    (0.18, 1568.0, 0.09),
)
_ATTACK_S = 0.004
# Above 1 the note drops away fast and reads as a blip rather than a tone.
_DECAY_SHAPE = 1.5


def _square(frequency: float, frames: int, sample_rate: int) -> np.ndarray:
    """A square wave summed harmonic by harmonic, so nothing folds back as aliasing hash."""

    t = np.arange(frames, dtype=np.float32) / float(sample_rate)
    wave = np.zeros(frames, dtype=np.float32)
    harmonic = 1
    while frequency * harmonic < sample_rate * 0.475:
        wave += np.sin(2.0 * np.pi * frequency * harmonic * t).astype(np.float32) / harmonic
        harmonic += 2
    peak = float(np.max(np.abs(wave)))
    return wave / peak if peak > 0.0 else wave


def _note(frequency: float, duration_s: float, sample_rate: int) -> np.ndarray:
    """One plucked blip that starts and ends at exactly zero, so it never clicks."""

    frames = max(1, int(round(duration_s * sample_rate)))
    envelope = np.linspace(1.0, 0.0, frames, dtype=np.float32) ** _DECAY_SHAPE
    attack = min(frames, max(1, int(round(_ATTACK_S * sample_rate))))
    envelope[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
    return _square(frequency, frames, sample_rate) * envelope


class ThinkingSound:
    """A gain-scaled mono loop with a cursor that survives across blocks."""

    def __init__(self, loop: np.ndarray, *, gain: float = DEFAULT_GAIN) -> None:
        samples = np.asarray(loop, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise ValueError("Thinking sound loop must contain at least one sample")
        self._loop = samples * float(gain)
        self._cursor = 0

    @classmethod
    def default(cls, sample_rate: int, *, gain: float = DEFAULT_GAIN) -> ThinkingSound:
        """Synthesize the built-in cue so no audio file has to ship with the package."""

        period = max(1, int(round(_PULSE_PERIOD_S * sample_rate)))
        loop = np.zeros(period, dtype=np.float32)
        for start_s, frequency, duration_s in _PULSES:
            start = int(round(start_s * sample_rate))
            note = _note(frequency, duration_s, sample_rate)
            end = min(period, start + note.size)
            if end > start:
                loop[start:end] += note[: end - start]
        peak = float(np.max(np.abs(loop)))
        if peak > 0.0:
            loop /= peak
        return cls(loop, gain=gain)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], sample_rate: int, *, gain: float = DEFAULT_GAIN) -> ThinkingSound:
        """Load a user-supplied cue, resampled and downmixed to the playback format."""

        import librosa

        loop, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        return cls(loop, gain=gain)

    def __len__(self) -> int:
        return int(self._loop.size)

    def reset(self) -> None:
        self._cursor = 0

    def next_block(self, frames: int) -> np.ndarray:
        """Return the next ``frames`` samples, wrapping around the loop as needed."""

        if frames <= 0:
            return np.zeros(0, dtype=np.float32)
        indices = (self._cursor + np.arange(frames)) % self._loop.size
        self._cursor = int((self._cursor + frames) % self._loop.size)
        return self._loop[indices]
