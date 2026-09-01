from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.api.openai_realtime.audio_resample import (
    StreamResampler,
    float32_to_pcm16,
    pcm16_to_float32,
)


def sine(freq, seconds, rate):
    t = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_frequency(x, rate):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1.0 / rate)[int(np.argmax(spectrum))])


@pytest.mark.parametrize("rate", [16000, 24000, 48000])
def test_resampler_is_a_passthrough_when_the_rates_match(rate):
    resampler = StreamResampler(rate, rate)
    block = sine(440.0, 0.1, rate)

    assert np.array_equal(resampler.process(block), block)


@pytest.mark.parametrize(("src", "dst"), [(48000, 16000), (24000, 16000), (16000, 48000), (16000, 24000)])
def test_resampler_does_not_drift_against_the_input_clock(src, dst):
    resampler = StreamResampler(src, dst)
    block = sine(440.0, 0.1, src)

    emitted = sum(len(resampler.process(block)) for _ in range(50))

    # Everything but the lookahead window still held back has been emitted.
    assert emitted == len(block) * 50 * dst // src - resampler.lookahead_frames


@pytest.mark.parametrize(("src", "dst"), [(48000, 16000), (24000, 16000), (16000, 48000), (16000, 24000)])
def test_resampler_emits_a_steady_frame_count_once_primed(src, dst):
    resampler = StreamResampler(src, dst)
    block = sine(440.0, 0.1, src)

    counts = [len(resampler.process(block)) for _ in range(10)]

    assert counts[1:] == [len(block) * dst // src] * 9


@pytest.mark.parametrize(("src", "dst"), [(48000, 16000), (24000, 16000), (16000, 48000)])
def test_resampler_preserves_the_tone(src, dst):
    resampler = StreamResampler(src, dst)

    out = resampler.process(sine(440.0, 0.5, src))

    assert dominant_frequency(out, dst) == pytest.approx(440.0, abs=5.0)


def spurious_energy_ratio(x, rate, freq):
    """Fraction of the spectrum that is not the expected tone.

    Discontinuities at chunk seams are broadband, so they show up here no matter
    where in the signal they land, and unlike a sample-wise comparison this does
    not care about the resampler's group delay.
    """
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    tone = np.abs(freqs - freq) <= 30.0
    return float(spectrum[~tone].sum() / spectrum[tone].sum())


@pytest.mark.parametrize(("src", "dst"), [(48000, 16000), (16000, 48000)])
def test_resampler_is_continuous_across_chunk_boundaries(src, dst):
    signal = sine(440.0, 0.5, src)
    chunk = 480
    blocks = [signal[i : i + chunk] for i in range(0, len(signal), chunk)]

    streaming = StreamResampler(src, dst)
    streamed = np.concatenate([streaming.process(block) for block in blocks])
    # A fresh resampler per chunk is the stateless baseline: it rings at every seam.
    stateless = np.concatenate([StreamResampler(src, dst).process(block) for block in blocks])

    streamed_spurious = spurious_energy_ratio(streamed, dst, 440.0)
    stateless_spurious = spurious_energy_ratio(stateless, dst, 440.0)

    assert streamed_spurious < 0.05
    assert stateless_spurious > 5.0 * streamed_spurious


def test_resampler_downsampling_rejects_content_above_the_new_nyquist():
    # 12 kHz is above the 8 kHz Nyquist of the new rate; without a lowpass it would
    # fold back down to an audible 4 kHz instead of disappearing.
    resampler = StreamResampler(48000, 16000)

    out = resampler.process(sine(12000.0, 0.5, 48000))

    assert float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) < 0.02


def test_pcm16_round_trip_preserves_the_waveform():
    block = sine(440.0, 0.05, 16000)

    restored = pcm16_to_float32(float32_to_pcm16(block))

    assert np.max(np.abs(restored - block)) < 1e-3


def test_float32_to_pcm16_clips_instead_of_wrapping():
    loud = np.array([2.0, -2.0], dtype=np.float32)

    assert np.frombuffer(float32_to_pcm16(loud), dtype=np.int16).tolist() == [32767, -32768]


def test_pcm16_to_float32_accepts_raw_bytes():
    assert pcm16_to_float32(b"\x00\x40\x00\xc0").tolist() == pytest.approx([0.5, -0.5], abs=1e-4)


def test_resampler_designs_its_filter_once_not_per_chunk(monkeypatch):
    # A 44.1 kHz device gives up=441 and an ~8800-tap design; doing that on every
    # pump would burn the feeder thread that playback continuity depends on.
    import speech_to_speech.api.openai_realtime.audio_resample as module

    designs = []
    real_firwin = module.firwin
    monkeypatch.setattr(module, "firwin", lambda *a, **k: designs.append(1) or real_firwin(*a, **k))

    resampler = StreamResampler(48000, 16000)
    for _ in range(5):
        resampler.process(sine(440.0, 0.1, 48000))

    assert len(designs) == 1


def test_cached_filter_matches_the_library_default():
    from scipy.signal import resample_poly

    block = sine(440.0, 0.5, 48000)
    resampler = StreamResampler(48000, 16000)

    ours = resampler.process(block)
    reference = resample_poly(block, 1, 3).astype(np.float32)

    # Same length modulo the withheld lookahead, and the same samples where they overlap.
    overlap = len(ours)
    assert np.max(np.abs(ours[100:overlap] - reference[100:overlap])) < 1e-5
