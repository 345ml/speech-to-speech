import numpy as np
import pytest
import soundfile as sf

from speech_to_speech.api.openai_realtime.thinking_sound import ThinkingSound


def test_default_thinking_sound_loops_without_a_seam():
    sound = ThinkingSound.default(16000)

    first = sound.next_block(len(sound))
    second = sound.next_block(len(sound))

    assert first.dtype == np.float32
    assert np.allclose(first, second)


def test_default_thinking_sound_begins_and_ends_at_silence():
    """A loop that wraps from a non-zero sample to another clicks on every repeat."""

    loop = ThinkingSound.default(16000).next_block(len(ThinkingSound.default(16000)))

    assert loop[0] == pytest.approx(0.0, abs=1e-6)
    assert loop[-1] == pytest.approx(0.0, abs=1e-6)


def test_default_thinking_sound_leaves_audible_gaps_between_pulses():
    sound = ThinkingSound.default(16000)

    loop = sound.next_block(len(sound))

    assert np.max(np.abs(loop)) > 0.0
    assert float(np.mean(np.abs(loop) < 1e-6)) > 0.5


def pulse_segments(loop, frame=64, threshold=1e-3):
    """Slice one loop into its runs of audible samples.

    Framed peaks, not raw samples: any oscillator passes through zero every
    half cycle, so a per-sample threshold counts zero crossings, not pulses.
    """

    frames = loop[: len(loop) // frame * frame].reshape(-1, frame)
    loud = np.max(np.abs(frames), axis=1) > threshold
    edges = np.diff(loud.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if loud[0]:
        starts.insert(0, 0)
    if loud[-1]:
        ends.append(len(loud))
    return [(start * frame, end * frame) for start, end in zip(starts, ends)]


def dominant_frequency(segment, sample_rate=16000):
    spectrum = np.abs(np.fft.rfft(segment))
    return float(np.fft.rfftfreq(len(segment), 1.0 / sample_rate)[int(np.argmax(spectrum))])


def test_default_thinking_sound_is_a_three_note_ascending_arpeggio():
    sound = ThinkingSound.default(16000)
    loop = sound.next_block(len(sound))

    segments = pulse_segments(loop)

    assert len(segments) == 3
    pitches = [dominant_frequency(loop[start:end]) for start, end in segments]
    assert pitches[0] < pitches[1] < pitches[2]


def test_default_thinking_sound_period_leaves_room_for_a_two_second_wait():
    """Roughly one and a half repeats at the companion's p50 time to first audio."""

    assert len(ThinkingSound.default(16000)) == pytest.approx(int(1.4 * 16000), abs=1)


def test_thinking_sound_reset_rewinds_the_loop():
    sound = ThinkingSound.default(16000)
    first = sound.next_block(512)
    sound.next_block(512)

    sound.reset()

    assert np.allclose(sound.next_block(512), first)


def test_thinking_sound_applies_its_gain():
    sound = ThinkingSound(np.ones(8, dtype=np.float32), gain=0.25)

    assert np.allclose(sound.next_block(8), 0.25)


def test_thinking_sound_from_file_is_resampled_to_the_playback_rate(tmp_path):
    path = tmp_path / "tick.wav"
    sf.write(path, np.zeros(4000, dtype=np.float32), 8000)

    sound = ThinkingSound.from_file(path, 16000)

    assert len(sound) == pytest.approx(8000, abs=8)


def test_thinking_sound_rejects_an_empty_loop():
    with pytest.raises(ValueError):
        ThinkingSound(np.zeros(0, dtype=np.float32))
