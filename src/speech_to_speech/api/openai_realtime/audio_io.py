"""Pluggable microphone/speaker transport for the packaged Realtime audio client.

The client needs two things from a local audio device: mono ``int16`` capture
frames pushed at it, and a callback that fills the speaker buffer on demand.
``sounddevice`` is the portable default. macOS can instead route both directions
through the system voice-processing unit, which cancels the speaker signal out of
the microphone (see :mod:`.audio_io_macos`); that only works when capture *and*
playback share one engine, which is why both directions live behind one seam.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

EchoCancellation = Literal["off", "os"]

#: Receives one chunk of mono ``int16`` capture frames at ``AudioStreamRequest.send_rate``.
CaptureCallback = Callable[[bytes], None]
#: Fills a writable buffer with mono ``int16`` playback frames at ``AudioStreamRequest.recv_rate``.
PlaybackFiller = Callable[[Any], None]


@dataclass(frozen=True)
class AudioStreamRequest:
    """Device-independent description of the audio the client wants to move."""

    send_rate: int
    recv_rate: int
    chunk_size: int
    input_device: int | None = None
    output_device: int | None = None


class AudioSession(Protocol):
    """One open pair of capture and playback streams."""

    def start(self) -> None:
        """Begin moving audio. Cleans up after itself if any stream fails to start."""

    def close(self) -> None:
        """Stop and release every device. Safe to call more than once."""


class AudioIO(Protocol):
    """Opens :class:`AudioSession` instances for one audio backend."""

    def open(
        self,
        request: AudioStreamRequest,
        *,
        on_capture: CaptureCallback,
        fill_playback: PlaybackFiller,
    ) -> AudioSession: ...


class SoundDeviceAudioSession:
    """Capture and playback over a pair of PortAudio streams."""

    def __init__(self, streams: list[Any]) -> None:
        self._opened = streams
        self._started: list[Any] = []
        self._closed = False

    def start(self) -> None:
        try:
            for stream in self._opened:
                stream.start()
                self._started.append(stream)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for stream in reversed(self._started):
            try:
                stream.stop()
            except Exception:
                logger.exception("Failed to stop local audio stream")
        self._started.clear()
        for stream in reversed(self._opened):
            try:
                stream.close()
            except Exception:
                logger.exception("Failed to close local audio stream")
        self._opened.clear()


class SoundDeviceAudioIO:
    """Portable backend. No echo cancellation: the speaker is audible to the microphone."""

    def open(
        self,
        request: AudioStreamRequest,
        *,
        on_capture: CaptureCallback,
        fill_playback: PlaybackFiller,
    ) -> AudioSession:
        import sounddevice as sd

        def callback_send(indata: Any, _frames: int, _time_info: Any, status: Any) -> None:
            if status:
                logger.warning("Microphone status: %s", status)
            on_capture(bytes(indata))

        def callback_recv(outdata: Any, _frames: int, _time_info: Any, status: Any) -> None:
            if status:
                logger.warning("Speaker status: %s", status)
            fill_playback(outdata)

        opened: list[Any] = []
        try:
            opened.append(
                sd.RawInputStream(
                    samplerate=request.send_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=request.chunk_size,
                    callback=callback_send,
                    device=request.input_device,
                )
            )
            opened.append(
                sd.RawOutputStream(
                    samplerate=request.recv_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=request.chunk_size,
                    callback=callback_recv,
                    device=request.output_device,
                )
            )
        except BaseException:
            for stream in reversed(opened):
                try:
                    stream.close()
                except Exception:
                    logger.exception("Failed to close local audio stream")
            raise
        return SoundDeviceAudioSession(opened)


def open_audio_session(
    audio_io: AudioIO,
    request: AudioStreamRequest,
    *,
    on_capture: CaptureCallback,
    fill_playback: PlaybackFiller,
    mode: str,
) -> AudioSession:
    """Open and start ``audio_io``, degrading to ``sounddevice`` if it will not run.

    ``create_audio_io`` can only rule out the static failures -- wrong platform, missing
    bindings. Whether a device will actually accept the voice-processing unit is only
    known here, and that refusal must not be fatal: the promise attached to
    ``--echo-cancellation`` is that it falls back rather than stopping the client.
    """

    def start(io: AudioIO) -> AudioSession:
        session = io.open(request, on_capture=on_capture, fill_playback=fill_playback)
        try:
            session.start()
        except BaseException:
            session.close()
            raise
        return session

    try:
        return start(audio_io)
    except Exception:
        if mode == "off":
            # Already the fallback: there is nothing further to degrade to.
            raise
        logger.warning(
            "Could not start the %r echo cancellation backend; continuing without echo cancellation.",
            mode,
            exc_info=True,
        )
    return start(SoundDeviceAudioIO())


def create_audio_io(mode: str) -> AudioIO:
    """Pick an audio backend, degrading to ``sounddevice`` rather than refusing to run."""

    if mode == "off":
        return SoundDeviceAudioIO()
    if mode != "os":
        raise ValueError(f"Unknown echo cancellation mode: {mode!r}")
    if sys.platform != "darwin":
        logger.warning(
            "Echo cancellation mode 'os' is only available on macOS; continuing without it. "
            "On Linux, select a PulseAudio/PipeWire echo-cancel source as the input device instead."
        )
        return SoundDeviceAudioIO()
    try:
        # Import the framework separately so a missing PyObjC is reported as such
        # rather than as a missing backend module.
        import_module("AVFoundation")
        from .audio_io_macos import VoiceProcessingAudioIO
    except ImportError:
        logger.warning(
            "Echo cancellation mode 'os' needs PyObjC; continuing without it. "
            "Install it with: pip install 'speech-to-speech[macos-aec]' "
            "(or pip install pyobjc-framework-AVFoundation)."
        )
        return SoundDeviceAudioIO()
    return VoiceProcessingAudioIO()
