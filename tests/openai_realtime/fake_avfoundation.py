"""A stand-in for the AVFoundation bindings, shaped like the parts we call.

The real framework is only present on macOS with PyObjC installed, and even there
the graph it hands back depends on the machine's audio hardware. This fake keeps
the wiring assertions honest and platform-independent.
"""

from __future__ import annotations

import numpy as np


class FakeFormat:
    def __init__(self, sample_rate, channels):
        self._sample_rate = float(sample_rate)
        self._channels = int(channels)

    def sampleRate(self):  # noqa: N802 - mirrors the Objective-C selector
        return self._sample_rate

    def channelCount(self):  # noqa: N802
        return self._channels

    def __repr__(self):
        return f"FakeFormat({self._sample_rate:.0f}Hz, {self._channels}ch)"


class FakeChannelData:
    """Mimics the ``float **`` PyObjC exposes, where ``as_buffer`` counts items."""

    def __init__(self, channels):
        self._channels = channels

    def __getitem__(self, index):
        return _FakeChannel(self._channels[index])


class _FakeChannel:
    def __init__(self, samples):
        self._samples = samples

    def as_buffer(self, count):
        return memoryview(self._samples[:count])


class FakePCMBuffer:
    def __init__(self, fmt, capacity):
        self._format = fmt
        self._capacity = int(capacity)
        self._frame_length = 0
        self.channels = [np.zeros(self._capacity, dtype=np.float32) for _ in range(fmt.channelCount())]

    @classmethod
    def alloc(cls):
        return cls

    @classmethod
    def initWithPCMFormat_frameCapacity_(cls, fmt, capacity):  # noqa: N802
        return cls(fmt, capacity)

    def format(self):
        return self._format

    def frameLength(self):  # noqa: N802
        return self._frame_length

    def setFrameLength_(self, length):  # noqa: N802
        self._frame_length = int(length)

    def floatChannelData(self):  # noqa: N802
        return FakeChannelData(self.channels)

    def samples(self):
        return self.channels[0][: self._frame_length]


class FakeTime:
    def __init__(self, sample_time):
        self._sample_time = sample_time

    def sampleTime(self):  # noqa: N802
        return self._sample_time


class FakeInputNode:
    def __init__(self, log):
        self.log = log
        self.voice_processing = False
        self.agc = True
        self.format = FakeFormat(48000, 1)
        self.tap = None
        self.tap_bus = None
        self.tap_buffer_size = None
        self.tap_format = None
        self.voice_processing_error = None

    def setVoiceProcessingEnabled_error_(self, enabled, _error):  # noqa: N802
        if self.voice_processing_error is not None:
            return False, self.voice_processing_error
        self.voice_processing = bool(enabled)
        self.log.append(("input", "voice_processing", bool(enabled)))
        # The unit takes the device over and republishes its own format.
        self.format = FakeFormat(24000, 3)
        return True, None

    def isVoiceProcessingEnabled(self):  # noqa: N802
        return self.voice_processing

    def setVoiceProcessingAGCEnabled_(self, enabled):  # noqa: N802
        self.agc = bool(enabled)
        self.log.append(("input", "agc", bool(enabled)))

    def isVoiceProcessingAGCEnabled(self):  # noqa: N802
        return self.agc

    def outputFormatForBus_(self, _bus):  # noqa: N802
        return self.format

    def installTapOnBus_bufferSize_format_block_(self, bus, buffer_size, fmt, block):  # noqa: N802
        self.tap_bus, self.tap_buffer_size, self.tap_format, self.tap = bus, buffer_size, fmt, block
        self.log.append(("input", "install_tap", bus))

    def removeTapOnBus_(self, bus):  # noqa: N802
        self.tap = None
        self.log.append(("input", "remove_tap", bus))

    def emit(self, samples):
        """Deliver one capture buffer through the installed tap."""
        self.emit_channels([samples] * self.format.channelCount())

    def emit_channels(self, channels):
        """Deliver one capture buffer whose channels differ from each other."""
        frames = len(channels[0])
        buffer = FakePCMBuffer(FakeFormat(self.format.sampleRate(), len(channels)), frames)
        buffer.setFrameLength_(frames)
        for destination, source in zip(buffer.channels, channels):
            destination[:frames] = source
        self.tap(buffer, FakeTime(0))


class FakeOutputNode:
    def __init__(self):
        self.format = FakeFormat(24000, 2)

    def inputFormatForBus_(self, _bus):  # noqa: N802
        return self.format


class FakeMixerNode:
    pass


class FakePlayerNode:
    def __init__(self, log):
        self.log = log
        self.scheduled = []
        self.playing = False
        self.render_time = None
        self.consumed_frames = 0

    @staticmethod
    def alloc():
        raise NotImplementedError

    def play(self):
        self.playing = True
        self.log.append(("player", "play", None))

    def stop(self):
        self.playing = False
        self.scheduled.clear()
        self.consumed_frames = 0
        self.log.append(("player", "stop", None))

    def scheduleBuffer_completionHandler_(self, buffer, _handler):  # noqa: N802
        self.scheduled.append(buffer)

    def lastRenderTime(self):  # noqa: N802
        return self.render_time

    def playerTimeForNodeTime_(self, _node_time):  # noqa: N802
        # Real AVAudioPlayerNode keeps advancing this while playing, whether or not
        # anything is scheduled, so it can run past everything handed to it.
        return FakeTime(self.consumed_frames)

    def scheduled_frames(self):
        return sum(buffer.frameLength() for buffer in self.scheduled)


class FakeEngine:
    def __init__(self, log):
        self.log = log
        self.input = FakeInputNode(log)
        self.output = FakeOutputNode()
        self.mixer = FakeMixerNode()
        self.attached = []
        self.connections = []
        self.running = False
        self.start_error = None

    def inputNode(self):  # noqa: N802
        return self.input

    def outputNode(self):  # noqa: N802
        return self.output

    def mainMixerNode(self):  # noqa: N802
        return self.mixer

    def attachNode_(self, node):  # noqa: N802
        self.attached.append(node)

    def connect_to_format_(self, source, destination, fmt):  # noqa: N802
        self.connections.append((source, destination, fmt))

    def prepare(self):
        self.log.append(("engine", "prepare", None))

    def startAndReturnError_(self, _error):  # noqa: N802
        if self.start_error is not None:
            return False, self.start_error
        self.running = True
        self.log.append(("engine", "start", None))
        return True, None

    def isRunning(self):  # noqa: N802
        return self.running

    def stop(self):
        self.running = False
        self.log.append(("engine", "stop", None))


class FakeAVFoundation:
    """Namespace object substituted for the ``AVFoundation`` module."""

    def __init__(self):
        self.log: list[tuple[str, str, object]] = []
        self.engine = FakeEngine(self.log)
        self.player = FakePlayerNode(self.log)
        outer = self

        class AVAudioEngine:
            @staticmethod
            def alloc():
                return AVAudioEngine

            @staticmethod
            def init():
                return outer.engine

        class AVAudioPlayerNode:
            @staticmethod
            def alloc():
                return AVAudioPlayerNode

            @staticmethod
            def init():
                return outer.player

        class AVAudioFormat:
            @staticmethod
            def alloc():
                return AVAudioFormat

            @staticmethod
            def initStandardFormatWithSampleRate_channels_(rate, channels):  # noqa: N802
                return FakeFormat(rate, channels)

        self.AVAudioEngine = AVAudioEngine
        self.AVAudioPlayerNode = AVAudioPlayerNode
        self.AVAudioFormat = AVAudioFormat
        self.AVAudioPCMBuffer = FakePCMBuffer
