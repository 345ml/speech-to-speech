"""Microphone and speaker routed through the macOS voice-processing unit.

CoreAudio's voice-processing unit cancels the render signal out of the capture
signal, which is what keeps the assistant's own voice from being transcribed as
user speech. It is reachable only through ``AVAudioEngine`` — PortAudio, and so
``sounddevice``, talks to AUHAL and cannot get at it — and it cancels only what it
renders itself, so playback has to run through the same engine as capture.

Three details of that unit drive the shape of this module:

* Enabling voice processing republishes the input format, at a rate and channel
  count chosen by the unit rather than by us, and not necessarily the same ones
  twice. Every format is therefore read back at runtime and resampled.
* ``mainMixerNode`` keeps its default rate while the unit forces the output node
  onto its own, so the mixer-to-output connection has to be rebuilt explicitly.
  Without that, starting the engine fails with CoreAudio error -10875.
* ``AVAudioPlayerNode`` is scheduled rather than pulled, so playback is fed by a
  small thread that keeps a short lead of audio queued.
"""

from __future__ import annotations

import logging
from importlib import import_module
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

from .audio_io import AudioStreamRequest, CaptureCallback, PlaybackFiller
from .audio_resample import StreamResampler, float32_to_pcm16, pcm16_to_float32

logger = logging.getLogger(__name__)

#: Seconds of audio kept queued on the player. Long enough to survive a late
#: feeder wake-up, short enough that a barge-in silences the assistant promptly:
#: audio already handed to the device plays out even after the buffer is cleared.
_TARGET_LEAD_SECONDS = 0.12
#: Feeder poll interval. Fine enough not to eat into the lead above, coarse enough
#: not to spin a core while the assistant is silent.
_FEED_INTERVAL_SECONDS = 0.005
#: Consecutive pump failures tolerated before the feeder gives up. Without a ceiling
#: a deterministic failure logs a traceback every interval for the rest of the session.
_MAX_CONSECUTIVE_FAILURES = 20


class VoiceProcessingAudioSession:
    """Capture and playback sharing one voice-processing ``AVAudioEngine``."""

    def __init__(
        self,
        request: AudioStreamRequest,
        *,
        on_capture: CaptureCallback,
        fill_playback: PlaybackFiller,
        av_module: Any,
    ) -> None:
        self._av = av_module
        self._request = request
        self._on_capture = on_capture
        self._fill_playback = fill_playback

        self._closed = False
        self._stopping = Event()
        self._feeder: Thread | None = None
        self._schedule_lock = Lock()
        self._scheduled_frames = 0

        self._generation = 0
        self._engine = self._av.AVAudioEngine.alloc().init()
        self._input_node = self._engine.inputNode()
        # Assigned by _build_graph; declared here so close() works on a partial build.
        self._player: Any = None

        if request.input_device is not None or request.output_device is not None:
            # The unit binds to the system default devices; AVAudioEngine offers no
            # equivalent of a PortAudio device index to map these onto.
            logger.warning(
                "--input-device/--output-device are ignored under --echo-cancellation os: "
                "the macOS voice-processing unit uses the system default input and output. "
                "Select the devices in System Settings, or drop --echo-cancellation to use the indices."
            )

        try:
            self._build_graph(request)
        except BaseException:
            self.close()
            raise

    def _build_graph(self, request: AudioStreamRequest) -> None:
        enabled, error = self._input_node.setVoiceProcessingEnabled_error_(True, None)
        if not enabled:
            raise RuntimeError(f"Could not enable macOS voice processing on the input device: {error}")
        # The unit's gain control chases the noise floor up between phrases, which
        # both the recognizer and the turn detector read as speech.
        self._input_node.setVoiceProcessingAGCEnabled_(False)

        # Read every format back only now: enabling voice processing replaced them.
        capture_format = self._input_node.outputFormatForBus_(0)
        output_node = self._engine.outputNode()
        render_format = output_node.inputFormatForBus_(0)
        self._render_rate = int(render_format.sampleRate())

        mixer = self._engine.mainMixerNode()
        self._engine.connect_to_format_(mixer, output_node, render_format)

        self._player = self._av.AVAudioPlayerNode.alloc().init()
        self._engine.attachNode_(self._player)
        self._play_format = self._av.AVAudioFormat.alloc().initStandardFormatWithSampleRate_channels_(
            render_format.sampleRate(), 1
        )
        self._engine.connect_to_format_(self._player, mixer, self._play_format)

        self._capture_resampler = StreamResampler(int(capture_format.sampleRate()), request.send_rate)
        self._playback_resampler = StreamResampler(request.recv_rate, self._render_rate)
        self._playback_chunk_frames = request.chunk_size
        self._target_lead_frames = int(self._render_rate * _TARGET_LEAD_SECONDS)
        # One poll's worth of playback frames; see the note in _pump_playback.
        self._idle_fill_frames = max(1, int(round(_FEED_INTERVAL_SECONDS * request.recv_rate)))

        self._input_node.installTapOnBus_bufferSize_format_block_(0, request.chunk_size, capture_format, self._on_tap)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._engine.prepare()
        started, error = self._engine.startAndReturnError_(None)
        if not started:
            self.close()
            raise RuntimeError(f"Could not start the macOS voice-processing audio engine: {error}")
        self._player.play()
        self._feeder = Thread(target=self._feed_playback, name="vpio-playback", daemon=True)
        self._feeder.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stopping.set()
        feeder, self._feeder = self._feeder, None
        if feeder is not None:
            feeder.join(timeout=1.0)
        for description, action in (
            ("remove the capture tap", lambda: self._input_node.removeTapOnBus_(0)),
            ("stop the playback node", lambda: self._player and self._player.stop()),
            ("stop the audio engine", self._engine.stop),
            # Release the microphone on close rather than whenever the engine is
            # collected, so the recording indicator tracks the session.
            (
                "release the voice-processing unit",
                lambda: self._input_node.setVoiceProcessingEnabled_error_(False, None),
            ),
        ):
            try:
                action()
            except Exception:
                logger.exception("Failed to %s", description)

    # -- capture -----------------------------------------------------------

    def _on_tap(self, buffer: Any, _when: Any) -> None:
        try:
            frames = int(buffer.frameLength())
            if frames <= 0:
                return
            # Voice processing can publish several channels carrying the same
            # processed signal; the first one is the microphone.
            channel = buffer.floatChannelData()[0]
            samples = np.frombuffer(channel.as_buffer(frames), dtype=np.float32)
            resampled = self._capture_resampler.process(samples)
            if len(resampled):
                self._on_capture(float32_to_pcm16(resampled))
        except Exception:
            # This runs on a CoreAudio thread; an escaping exception kills capture silently.
            logger.exception("Failed to forward a captured audio buffer")

    # -- playback ----------------------------------------------------------

    def _frames_in_flight(self) -> int:
        """Frames handed to the player that it has not rendered yet."""

        render_time = self._player.lastRenderTime()
        player_time = self._player.playerTimeForNodeTime_(render_time) if render_time is not None else None
        with self._schedule_lock:
            if player_time is None:
                return self._scheduled_frames
            # A negative player time means the node has not begun rendering yet -- the
            # render time still predates play(). Reading it as a lead would mask the
            # idle state below and start a queue of silence that never drains.
            consumed = max(0, int(player_time.sampleTime()))
            in_flight = self._scheduled_frames - consumed
            if in_flight < 0:
                # The player ran dry. Its timeline keeps advancing on an empty queue,
                # so it has read past everything scheduled; the deficit is silence
                # already gone, not audio still to come. Re-anchor, or every underrun
                # would permanently add its own length to the queue.
                self._scheduled_frames = consumed
                in_flight = 0
            return in_flight

    def _pump_playback(self) -> bool:
        """Queue one chunk if the player is running low. Returns whether it did."""

        in_flight = self._frames_in_flight()
        if in_flight >= self._target_lead_frames:
            return False
        with self._schedule_lock:
            generation = self._generation
            resampler = self._playback_resampler

        # While the player is idle, ask for one poll's worth rather than a whole chunk.
        # Filling is not free of consequence: the filler advances real-time state on
        # every call -- the thinking cue's start delay counts down in frames asked for,
        # on the assumption that asking for them costs the time they take to play. A
        # sound card keeps that true by asking only as fast as it plays. This feeder
        # polls far faster than that and throws away what it gets while idle, so asking
        # for a chunk each poll would run the cue ahead of the clock. Asking for exactly
        # one interval's worth keeps it honest without delaying the start of a response.
        frames = self._playback_chunk_frames if in_flight > 0 else self._idle_fill_frames

        # Filling and resampling stay outside the lock: the filler reaches into the
        # playback buffer, which the event loop also holds a lock on.
        raw = bytearray(frames * 2)
        self._fill_playback(memoryview(raw))
        if in_flight == 0 and not any(raw):
            # Idle. Building a lead out of silence would only push the first audio of
            # the next response back by the length of that lead.
            return False
        rendered = resampler.process(pcm16_to_float32(np.frombuffer(raw, dtype=np.int16)))
        if not len(rendered):
            return False

        buffer = self._av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(self._play_format, len(rendered))
        buffer.setFrameLength_(len(rendered))
        np.frombuffer(buffer.floatChannelData()[0].as_buffer(len(rendered)), dtype=np.float32)[:] = rendered
        with self._schedule_lock:
            if generation != self._generation:
                # A barge-in landed while this chunk was being prepared. Scheduling it
                # now would let the interrupted response talk over the user.
                return False
            self._scheduled_frames += len(rendered)
            self._player.scheduleBuffer_completionHandler_(buffer, None)
        return True

    def flush_playback(self) -> None:
        """Drop audio already queued on the device so a barge-in takes effect now."""

        with self._schedule_lock:
            # Bumping the generation under the same lock that guards scheduling is what
            # makes this atomic against a chunk the feeder is already preparing.
            self._generation += 1
            self._player.stop()
            self._scheduled_frames = 0
            self._playback_resampler = StreamResampler(self._request.recv_rate, self._render_rate)
            if not self._stopping.is_set():
                self._player.play()

    def _feed_playback(self) -> None:
        failures = 0
        while not self._stopping.is_set():
            if not self._engine.isRunning():
                # AVAudioEngine stops itself and tears the graph down when the audio
                # device configuration changes -- headphones, Bluetooth, a new default
                # device. Capture is already dead at this point; say so rather than
                # stalling silently with the playback buffer growing behind us.
                logger.error(
                    "The macOS audio engine stopped, most likely because the audio device "
                    "configuration changed. Local audio has stopped; restart the session."
                )
                return
            try:
                if not self._pump_playback():
                    self._stopping.wait(_FEED_INTERVAL_SECONDS)
                failures = 0
            except Exception:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.exception("Local playback failed %d times in a row; stopping the feeder", failures)
                    return
                logger.exception("Failed to queue local playback audio")
                self._stopping.wait(_FEED_INTERVAL_SECONDS * failures)


class VoiceProcessingAudioIO:
    """Backend that hands both audio directions to the macOS voice-processing unit."""

    def __init__(self, av_module: Any = None) -> None:
        self._av_module = av_module

    def open(
        self,
        request: AudioStreamRequest,
        *,
        on_capture: CaptureCallback,
        fill_playback: PlaybackFiller,
    ) -> VoiceProcessingAudioSession:
        av_module = self._av_module if self._av_module is not None else import_module("AVFoundation")
        return VoiceProcessingAudioSession(
            request,
            on_capture=on_capture,
            fill_playback=fill_playback,
            av_module=av_module,
        )
