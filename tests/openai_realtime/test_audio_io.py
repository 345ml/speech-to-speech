from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from speech_to_speech.api.openai_realtime.audio_io import (
    AudioStreamRequest,
    SoundDeviceAudioIO,
    create_audio_io,
    open_audio_session,
)

REQUEST = AudioStreamRequest(
    send_rate=16000,
    recv_rate=16000,
    chunk_size=1024,
    input_device=3,
    output_device=4,
)


class FakeStream:
    def __init__(self, kind, log, **kwargs):
        self.kind = kind
        self.kwargs = kwargs
        self.log = log
        self.log.append((kind, "open"))

    def start(self):
        self.log.append((self.kind, "start"))

    def stop(self):
        self.log.append((self.kind, "stop"))

    def close(self):
        self.log.append((self.kind, "close"))


class FakeSoundDevice:
    def __init__(self):
        self.log: list[tuple[str, str]] = []
        self.streams: dict[str, FakeStream] = {}

    def RawInputStream(self, **kwargs):  # noqa: N802 - mirrors the sounddevice API
        stream = FakeStream("input", self.log, **kwargs)
        self.streams["input"] = stream
        return stream

    def RawOutputStream(self, **kwargs):  # noqa: N802 - mirrors the sounddevice API
        stream = FakeStream("output", self.log, **kwargs)
        self.streams["output"] = stream
        return stream


@pytest.fixture
def fake_sd(monkeypatch):
    module = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


def open_session(fake_sd, *, on_capture=None, fill_playback=None):
    return SoundDeviceAudioIO().open(
        REQUEST,
        on_capture=on_capture or (lambda _chunk: None),
        fill_playback=fill_playback or (lambda _outdata: None),
    )


def test_sounddevice_io_opens_streams_with_the_requested_parameters(fake_sd):
    open_session(fake_sd)

    input_kwargs = fake_sd.streams["input"].kwargs
    assert input_kwargs["samplerate"] == 16000
    assert input_kwargs["channels"] == 1
    assert input_kwargs["dtype"] == "int16"
    assert input_kwargs["blocksize"] == 1024
    assert input_kwargs["device"] == 3

    output_kwargs = fake_sd.streams["output"].kwargs
    assert output_kwargs["samplerate"] == 16000
    assert output_kwargs["device"] == 4


def test_sounddevice_io_does_not_start_streams_until_start_is_called(fake_sd):
    session = open_session(fake_sd)

    assert [entry for entry in fake_sd.log if entry[1] == "start"] == []

    session.start()

    assert [entry for entry in fake_sd.log if entry[1] == "start"] == [
        ("input", "start"),
        ("output", "start"),
    ]


def test_sounddevice_io_stops_streams_before_closing_them_in_reverse_order(fake_sd):
    session = open_session(fake_sd)
    session.start()
    fake_sd.log.clear()

    session.close()

    assert fake_sd.log == [
        ("output", "stop"),
        ("input", "stop"),
        ("output", "close"),
        ("input", "close"),
    ]


def test_sounddevice_io_close_is_idempotent(fake_sd):
    session = open_session(fake_sd)
    session.start()
    session.close()
    fake_sd.log.clear()

    session.close()

    assert fake_sd.log == []


def test_sounddevice_io_closes_the_first_stream_when_the_second_fails_to_open(fake_sd, monkeypatch):
    def explode(**_kwargs):
        raise RuntimeError("no such output device")

    monkeypatch.setattr(fake_sd, "RawOutputStream", explode)

    with pytest.raises(RuntimeError, match="no such output device"):
        open_session(fake_sd)

    assert fake_sd.log == [("input", "open"), ("input", "close")]


def test_sounddevice_io_closes_started_streams_when_a_later_start_fails(fake_sd, monkeypatch):
    session = open_session(fake_sd)
    monkeypatch.setattr(fake_sd.streams["output"], "start", lambda: (_ for _ in ()).throw(RuntimeError("busy")))
    fake_sd.log.clear()

    with pytest.raises(RuntimeError, match="busy"):
        session.start()

    assert fake_sd.log == [
        ("input", "start"),
        ("input", "stop"),
        ("output", "close"),
        ("input", "close"),
    ]


def test_sounddevice_io_forwards_captured_frames_as_bytes(fake_sd):
    captured: list[bytes] = []
    open_session(fake_sd, on_capture=captured.append)

    callback = fake_sd.streams["input"].kwargs["callback"]
    callback(memoryview(b"\x01\x02\x03\x04"), 2, None, None)

    assert captured == [b"\x01\x02\x03\x04"]


def test_sounddevice_io_lets_the_playback_filler_write_into_the_device_buffer(fake_sd):
    def fill(outdata):
        outdata[:] = b"\x09" * len(outdata)

    open_session(fake_sd, fill_playback=fill)

    callback = fake_sd.streams["output"].kwargs["callback"]
    outdata = bytearray(4)
    callback(memoryview(outdata), 2, None, None)

    assert bytes(outdata) == b"\x09\x09\x09\x09"


def test_create_audio_io_defaults_to_sounddevice():
    assert isinstance(create_audio_io("off"), SoundDeviceAudioIO)


def test_create_audio_io_falls_back_to_sounddevice_off_darwin(monkeypatch, caplog):
    monkeypatch.setattr(sys, "platform", "linux")

    with caplog.at_level("WARNING"):
        audio_io = create_audio_io("os")

    assert isinstance(audio_io, SoundDeviceAudioIO)
    assert "only available on macOS" in caplog.text


def test_create_audio_io_falls_back_when_the_macos_backend_cannot_be_imported(monkeypatch, caplog):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "AVFoundation", None)

    with caplog.at_level("WARNING"):
        audio_io = create_audio_io("os")

    assert isinstance(audio_io, SoundDeviceAudioIO)
    assert "pyobjc-framework-AVFoundation" in caplog.text


def test_create_audio_io_returns_the_voice_processing_backend_on_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "AVFoundation", SimpleNamespace())

    from speech_to_speech.api.openai_realtime.audio_io_macos import VoiceProcessingAudioIO

    assert isinstance(create_audio_io("os"), VoiceProcessingAudioIO)


def test_create_audio_io_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="Unknown echo cancellation mode"):
        create_audio_io("magic")


class RefusingAudioIO:
    def __init__(self, error="the device will not take voice processing"):
        self.error = error

    def open(self, _request, *, on_capture, fill_playback):
        raise RuntimeError(self.error)


class RecordingAudioIO:
    def __init__(self):
        self.session = None

    def open(self, request, *, on_capture, fill_playback):
        self.session = SimpleNamespace(started=False, closed=False)
        self.session.start = lambda: setattr(self.session, "started", True)
        self.session.close = lambda: setattr(self.session, "closed", True)
        return self.session


def test_open_audio_session_starts_the_configured_backend():
    audio_io = RecordingAudioIO()

    session = open_audio_session(
        audio_io, REQUEST, on_capture=lambda _c: None, fill_playback=lambda _o: None, mode="os"
    )

    assert session is audio_io.session
    assert session.started


def test_open_audio_session_falls_back_when_the_backend_refuses_to_open(monkeypatch, caplog):
    # The documented promise is that --echo-cancellation degrades rather than refuses.
    # A device that will not take the voice-processing unit fails here, not at import,
    # and the failure would otherwise end the client with no message and exit code 0.
    fallback = RecordingAudioIO()
    monkeypatch.setattr("speech_to_speech.api.openai_realtime.audio_io.SoundDeviceAudioIO", lambda: fallback)

    with caplog.at_level("WARNING"):
        session = open_audio_session(
            RefusingAudioIO(), REQUEST, on_capture=lambda _c: None, fill_playback=lambda _o: None, mode="os"
        )

    assert session is fallback.session
    assert session.started
    assert "without echo cancellation" in caplog.text


def test_open_audio_session_reports_a_failure_of_the_fallback_backend_itself(fake_sd, monkeypatch):
    def explode(**_kwargs):
        raise RuntimeError("no audio devices at all")

    monkeypatch.setattr(fake_sd, "RawInputStream", explode)

    with pytest.raises(RuntimeError, match="no audio devices at all"):
        open_audio_session(
            SoundDeviceAudioIO(), REQUEST, on_capture=lambda _c: None, fill_playback=lambda _o: None, mode="off"
        )
