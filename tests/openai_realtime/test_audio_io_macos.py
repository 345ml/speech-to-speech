from __future__ import annotations

import numpy as np
import pytest

import speech_to_speech.api.openai_realtime.audio_io_macos as audio_io_macos
from speech_to_speech.api.openai_realtime.audio_io import AudioStreamRequest
from speech_to_speech.api.openai_realtime.audio_io_macos import VoiceProcessingAudioIO
from speech_to_speech.api.openai_realtime.audio_resample import pcm16_to_float32

from .fake_avfoundation import FakeAVFoundation, FakeTime

REQUEST = AudioStreamRequest(send_rate=16000, recv_rate=16000, chunk_size=1024)


@pytest.fixture
def av():
    return FakeAVFoundation()


def open_session(av, *, on_capture=None, fill_playback=None, request=REQUEST):
    return VoiceProcessingAudioIO(av_module=av).open(
        request,
        on_capture=on_capture or (lambda _chunk: None),
        fill_playback=fill_playback or (lambda _outdata: None),
    )


def audible(outdata):
    """A filler that always supplies real audio.

    The pump deliberately refuses to queue silence while idle, so a test that wants
    to exercise scheduling has to hand it something to schedule.
    """
    outdata[:] = b"\x10\x20" * (len(outdata) // 2)


def sine(freq, seconds, rate):
    return (0.5 * np.sin(2 * np.pi * freq * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


def dominant_frequency(samples, rate):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1.0 / rate)[int(np.argmax(spectrum))])


def test_open_enables_voice_processing_and_turns_the_automatic_gain_control_off(av):
    open_session(av)

    assert av.engine.input.isVoiceProcessingEnabled()
    # AGC rides the ambient noise floor up between phrases, which the recognizer reads as speech.
    assert not av.engine.input.isVoiceProcessingAGCEnabled()


def test_open_reports_a_refusal_to_enable_voice_processing(av):
    av.engine.input.voice_processing_error = "device is busy"

    with pytest.raises(RuntimeError, match="device is busy"):
        open_session(av)


def test_open_rebuilds_the_mixer_connection_at_the_output_nodes_own_format(av):
    # mainMixerNode stays at its default rate while the voice-processing unit forces
    # the output node onto its own; leaving that connection alone fails engine start
    # with CoreAudio error -10875.
    open_session(av)

    mixer_connections = [c for c in av.engine.connections if c[0] is av.engine.mixer]
    assert len(mixer_connections) == 1
    _source, destination, fmt = mixer_connections[0]
    assert destination is av.engine.output
    assert fmt is av.engine.output.format


def test_open_connects_the_player_as_mono_at_the_output_rate(av):
    open_session(av)

    player_connections = [c for c in av.engine.connections if c[0] is av.player]
    assert len(player_connections) == 1
    _source, destination, fmt = player_connections[0]
    assert destination is av.engine.mixer
    assert fmt.sampleRate() == av.engine.output.format.sampleRate()
    assert fmt.channelCount() == 1
    assert av.player in av.engine.attached


def test_open_taps_the_capture_format_published_after_voice_processing_is_enabled(av):
    # Enabling voice processing republishes the input format, so a format read
    # beforehand describes a device layout that no longer exists.
    open_session(av)

    assert av.engine.input.tap_bus == 0
    assert av.engine.input.tap_format is av.engine.input.format
    assert av.engine.input.tap_format.sampleRate() == 24000


def test_open_does_not_start_the_engine(av):
    open_session(av)

    assert not av.engine.running


def test_start_runs_the_engine_and_the_player(av):
    session = open_session(av)
    try:
        session.start()

        assert av.engine.running
        assert av.player.playing
    finally:
        session.close()


def test_start_reports_a_refusal_from_core_audio(av):
    av.engine.start_error = "-10875"
    session = open_session(av)

    with pytest.raises(RuntimeError, match="-10875"):
        session.start()


def test_capture_is_resampled_to_the_send_rate_and_handed_over_as_pcm16(av):
    captured: list[bytes] = []
    open_session(av, on_capture=captured.append)
    tone = sine(440.0, 0.5, 24000)

    for start in range(0, len(tone), 2400):
        av.engine.input.emit(tone[start : start + 2400])

    assert captured
    joined = b"".join(captured)
    assert len(joined) % 2 == 0
    samples = pcm16_to_float32(joined)
    # 24 kHz in, 16 kHz out: two thirds as many frames, and still a 440 Hz tone.
    assert len(samples) == pytest.approx(len(tone) * 2 / 3, rel=0.02)
    assert dominant_frequency(samples, 16000) == pytest.approx(440.0, abs=5.0)


def test_capture_reads_the_first_channel_only(av):
    captured: list[bytes] = []
    open_session(av, on_capture=captured.append)
    frames = 4800
    voice = sine(440.0, frames / 24000, 24000)

    av.engine.input.emit_channels([voice, np.zeros(frames, np.float32), np.zeros(frames, np.float32)])

    samples = pcm16_to_float32(b"".join(captured))
    assert float(np.sqrt(np.mean(samples**2))) > 0.2


def test_capture_ignores_an_empty_buffer(av):
    captured: list[bytes] = []
    open_session(av, on_capture=captured.append)

    av.engine.input.emit(np.zeros(0, np.float32))

    assert captured == []


def test_playback_schedules_the_audio_the_filler_provides(av):
    tone = sine(440.0, 0.5, 16000)
    cursor = {"at": 0}

    def fill(outdata):
        wanted = len(outdata) // 2
        block = tone[cursor["at"] : cursor["at"] + wanted]
        cursor["at"] += wanted
        padded = np.zeros(wanted, np.float32)
        padded[: len(block)] = block
        outdata[:] = (padded * 32767).astype(np.int16).tobytes()

    session = open_session(av, fill_playback=fill)

    while session._pump_playback():
        pass

    assert av.player.scheduled
    rendered = np.concatenate([buffer.samples() for buffer in av.player.scheduled])
    # 16 kHz in, 24 kHz out at the format the voice-processing unit chose.
    assert av.player.scheduled[0].format().sampleRate() == 24000
    assert dominant_frequency(rendered, 24000) == pytest.approx(440.0, abs=10.0)


def test_playback_stops_scheduling_once_it_is_far_enough_ahead(av):
    session = open_session(av, fill_playback=audible)

    while session._pump_playback():
        pass

    # Enough to ride out a late callback, but short enough that a barge-in cuts off promptly.
    lead_seconds = av.player.scheduled_frames() / 24000
    assert 0.05 <= lead_seconds <= 0.25
    assert not session._pump_playback()


def test_playback_resumes_scheduling_as_the_player_consumes_audio(av):
    session = open_session(av, fill_playback=audible)
    while session._pump_playback():
        pass
    av.player.render_time = FakeTime(0)
    av.player.consumed_frames = av.player.scheduled_frames()

    assert session._pump_playback()


def test_flush_playback_drops_audio_that_was_already_handed_to_the_device(av):
    session = open_session(av, fill_playback=audible)
    session.start()
    try:
        while session._pump_playback():
            pass
        assert av.player.scheduled

        session.flush_playback()

        assert av.player.scheduled == []
        # The node has to be running again, or the next turn is never heard.
        assert av.player.playing
        assert session._pump_playback()
    finally:
        session.close()


def test_close_removes_the_tap_and_stops_the_devices(av):
    session = open_session(av)
    session.start()

    session.close()

    assert av.engine.input.tap is None
    assert not av.player.playing
    assert not av.engine.running


def test_close_is_idempotent(av):
    session = open_session(av)
    session.start()
    session.close()
    av.log.clear()

    session.close()

    assert av.log == []


def pump_until_full(session):
    """Fill the playback lead and report how many frames that took."""
    before = session._scheduled_frames
    while session._pump_playback():
        pass
    return session._scheduled_frames - before


def test_playback_recovers_the_lead_after_the_player_runs_dry(av):
    # AVAudioPlayerNode's timeline keeps advancing on an empty queue, so after a
    # starved feeder it reads *past* everything scheduled. Treating that deficit as
    # audio still to come would make every underrun permanently lengthen the queue.
    session = open_session(av, fill_playback=audible)
    pump_until_full(session)
    starved_frames = 48000
    av.player.render_time = FakeTime(0)
    av.player.consumed_frames = session._scheduled_frames + starved_frames

    assert session._frames_in_flight() == 0
    requeued = pump_until_full(session)

    # The lead is rebuilt, not the whole starvation gap.
    assert requeued < starved_frames / 2
    assert session._frames_in_flight() == pytest.approx(session._target_lead_frames, rel=0.5)


def test_playback_does_not_build_a_lead_out_of_silence(av):
    # Queueing silence while idle only puts a delay in front of the next response.
    session = open_session(av, fill_playback=lambda outdata: outdata.__setitem__(slice(None), bytes(len(outdata))))

    assert not session._pump_playback()
    assert av.player.scheduled == []


def test_playback_starts_queueing_as_soon_as_there_is_audio(av):
    pending = {"audio": False}

    def fill(outdata):
        outdata[:] = (b"\x10\x20" if pending["audio"] else b"\x00\x00") * (len(outdata) // 2)

    session = open_session(av, fill_playback=fill)
    assert not session._pump_playback()

    pending["audio"] = True

    assert session._pump_playback()
    assert av.player.scheduled


def test_flush_discards_a_chunk_that_was_prepared_before_the_barge_in(av):
    # The feeder fills and resamples outside the lock, so a barge-in can land while a
    # chunk of the outgoing response is in flight. Scheduling it anyway would let the
    # assistant talk over the user, which is the thing the flush exists to stop.
    session = open_session(av)
    session.start()
    try:

        def fill_then_barge_in(outdata):
            outdata[:] = b"\x10\x20" * (len(outdata) // 2)
            session.flush_playback()

        session._fill_playback = fill_then_barge_in

        assert not session._pump_playback()
        assert av.player.scheduled == []
        assert session._scheduled_frames == 0
    finally:
        session.close()


def test_open_leaves_no_engine_running_when_construction_fails(av):
    def explode(*_args):
        raise RuntimeError("tap refused")

    av.engine.input.installTapOnBus_bufferSize_format_block_ = explode

    with pytest.raises(RuntimeError, match="tap refused"):
        open_session(av)

    # Otherwise the unit keeps the microphone while the caller falls back to PortAudio.
    assert not av.engine.input.isVoiceProcessingEnabled()
    assert not av.engine.running


def test_open_warns_that_device_selection_does_not_reach_the_voice_processing_unit(av, caplog):
    with caplog.at_level("WARNING"):
        open_session(av, request=AudioStreamRequest(send_rate=16000, recv_rate=16000, chunk_size=1024, input_device=3))

    assert "input_device" in caplog.text or "device" in caplog.text.lower()
    assert "echo-cancellation" in caplog.text


def test_feeder_gives_up_after_repeated_failures_instead_of_logging_forever(av, monkeypatch, caplog):
    monkeypatch.setattr(audio_io_macos, "_FEED_INTERVAL_SECONDS", 0.0)
    session = open_session(av)
    # Mark the engine live without start(), which would race a real feeder thread
    # against the one this test drives.
    av.engine.running = True
    attempts = {"count": 0}

    def always_fails():
        attempts["count"] += 1
        raise RuntimeError("scheduling is broken")

    session._pump_playback = always_fails
    with caplog.at_level("ERROR"):
        session._feed_playback()

    assert attempts["count"] <= audio_io_macos._MAX_CONSECUTIVE_FAILURES
    assert "giving up" in caplog.text.lower() or "stopping" in caplog.text.lower()


def test_feeder_reports_an_engine_that_stopped_on_its_own(av, monkeypatch, caplog):
    # Plugging in headphones makes AVAudioEngine tear its own graph down. Left
    # undetected, capture dies silently and the playback buffer grows forever.
    monkeypatch.setattr(audio_io_macos, "_FEED_INTERVAL_SECONDS", 0.0)
    session = open_session(av)
    session.start()
    av.engine.running = False

    with caplog.at_level("ERROR"):
        session._feed_playback()

    assert "audio device" in caplog.text.lower() or "configuration" in caplog.text.lower()
    session.close()


def test_playback_ignores_a_player_clock_that_has_not_started_yet(av):
    # Right after play(), lastRenderTime() can still be a node time from before the
    # player started, so playerTimeForNodeTime_ reports a negative sample time. Read
    # as a lead, that phantom hides the idle state and the pump queues silence forever.
    session = open_session(av, fill_playback=lambda outdata: None)
    av.player.render_time = FakeTime(0)
    av.player.consumed_frames = -960

    assert session._frames_in_flight() == 0
    assert not session._pump_playback()
    assert av.player.scheduled == []
