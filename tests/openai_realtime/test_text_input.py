"""Tests for typed text input in the packaged microphone/speaker client."""

from __future__ import annotations

import logging
import sys
from io import StringIO
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.api.openai_realtime.audio_client import PlaybackBuffer
from speech_to_speech.api.openai_realtime.console import display_width
from speech_to_speech.api.openai_realtime.text_input import (
    PinnedConsole,
    TextInputCoordinator,
    TextInputUnavailable,
    TextTurnSubmitter,
    _bind_stream_handlers,
    check_text_input_support,
    create_text_terminal,
    terminal_is_interactive,
)
from speech_to_speech.api.openai_realtime.thinking_sound import ThinkingSound


class RecordingConnection:
    def __init__(self, on_send=None) -> None:
        self.sent: list[dict] = []
        self._on_send = on_send

    async def send(self, event: dict) -> None:
        if self._on_send is not None:
            self._on_send(event)
        self.sent.append(event)


class RecordingConsole:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def line(self, text: str) -> None:
        self.calls.append(("line", text))

    def stream(self, text: str) -> None:
        self.calls.append(("stream", text))

    def end_stream(self) -> None:
        self.calls.append(("end_stream",))

    def transient(self, text: str, *, final: bool = False) -> None:
        self.calls.append(("transient", text, final))

    def clear_transient(self) -> None:
        self.calls.append(("clear_transient",))


class FakeTerminal:
    def __init__(self, lines=(), raises=None) -> None:
        self.console = RecordingConsole()
        self._lines = list(lines)
        self._raises = raises
        self.activated = False
        self.deactivated = False

    def activate(self) -> None:
        self.activated = True

    def deactivate(self) -> None:
        self.deactivated = True

    async def read_line(self) -> str:
        if self._lines:
            return self._lines.pop(0)
        if self._raises is not None:
            raise self._raises
        raise EOFError


# ── TextTurnSubmitter ─────────────────────────────


async def test_typed_turn_cancels_before_creating_the_item() -> None:
    conn = RecordingConnection()
    submitter = TextTurnSubmitter(conn, playback=PlaybackBuffer(16000), console=RecordingConsole())

    await submitter.submit("いま何時?")

    assert [event["type"] for event in conn.sent] == [
        "response.cancel",
        "conversation.item.create",
        "response.create",
    ]
    assert conn.sent[1]["item"] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "いま何時?"}],
    }


async def test_typed_turn_cancels_even_with_no_visible_response() -> None:
    """A speech-triggered turn is queued before the client is told about it."""

    conn = RecordingConnection()
    submitter = TextTurnSubmitter(conn, playback=PlaybackBuffer(16000), console=RecordingConsole())

    await submitter.submit("hello")

    assert conn.sent[0]["type"] == "response.cancel"


async def test_typed_turn_stops_local_playback_before_the_first_send() -> None:
    playback = PlaybackBuffer(16000)
    playback.append(b"\x00\x01" * 4000)
    buffered_at_send: list[int] = []
    conn = RecordingConnection(on_send=lambda _event: buffered_at_send.append(playback.buffered_bytes))
    submitter = TextTurnSubmitter(conn, playback=playback, console=RecordingConsole())

    await submitter.submit("stop")

    assert buffered_at_send[0] == 0


async def test_typed_turn_starts_the_thinking_sound() -> None:
    """A typed turn has the same generation gap as a spoken one."""

    playback = PlaybackBuffer(
        16000,
        thinking_sound=ThinkingSound(np.full(1024, 0.1, dtype=np.float32), gain=1.0),
        thinking_delay_s=0.0,
    )
    submitter = TextTurnSubmitter(RecordingConnection(), playback=playback, console=RecordingConsole())

    await submitter.submit("hello")

    outdata = bytearray(2048)
    playback.write(outdata)
    assert np.all(np.frombuffer(bytes(outdata), dtype=np.int16) != 0)


async def test_typed_turn_never_sends_output_audio_buffer_clear() -> None:
    """The WebSocket router rejects that event; playback is cleared locally."""

    conn = RecordingConnection()
    submitter = TextTurnSubmitter(conn, playback=PlaybackBuffer(16000), console=RecordingConsole())

    await submitter.submit("hi")

    assert "output_audio_buffer.clear" not in [event["type"] for event in conn.sent]


async def test_typed_turn_drops_transient_text() -> None:
    console = RecordingConsole()
    submitter = TextTurnSubmitter(RecordingConnection(), playback=PlaybackBuffer(16000), console=console)

    await submitter.submit("hi")

    assert ("clear_transient",) in console.calls


# ── TextInputCoordinator ──────────────────────────


async def test_submitted_lines_are_sent() -> None:
    conn = RecordingConnection()
    terminal = FakeTerminal(lines=["one", "two"])
    stop_event = Event()
    coordinator = TextInputCoordinator(
        terminal=terminal,
        submitter=TextTurnSubmitter(conn, playback=PlaybackBuffer(16000), console=terminal.console),
        stop_event=stop_event,
    )

    await coordinator.run()

    texts = [event["item"]["content"][0]["text"] for event in conn.sent if event["type"] == "conversation.item.create"]
    assert texts == ["one", "two"]


async def test_blank_lines_are_ignored() -> None:
    conn = RecordingConnection()
    terminal = FakeTerminal(lines=["   ", "\t"])
    coordinator = TextInputCoordinator(
        terminal=terminal,
        submitter=TextTurnSubmitter(conn, playback=PlaybackBuffer(16000), console=terminal.console),
        stop_event=Event(),
    )

    await coordinator.run()

    assert conn.sent == []


@pytest.mark.parametrize("quit_exception", [EOFError(), KeyboardInterrupt()])
async def test_quitting_the_prompt_stops_the_session(quit_exception: BaseException) -> None:
    terminal = FakeTerminal(raises=quit_exception)
    stop_event = Event()
    coordinator = TextInputCoordinator(
        terminal=terminal,
        submitter=TextTurnSubmitter(RecordingConnection(), playback=PlaybackBuffer(16000), console=terminal.console),
        stop_event=stop_event,
    )

    await coordinator.run()

    assert stop_event.is_set()


async def test_send_failures_reach_the_session() -> None:
    def boom(_event: dict) -> None:
        raise RuntimeError("socket closed")

    terminal = FakeTerminal(lines=["hi"])
    coordinator = TextInputCoordinator(
        terminal=terminal,
        submitter=TextTurnSubmitter(
            RecordingConnection(on_send=boom), playback=PlaybackBuffer(16000), console=terminal.console
        ),
        stop_event=Event(),
    )

    with pytest.raises(RuntimeError, match="socket closed"):
        await coordinator.run()


# ── PinnedConsole ─────────────────────────────────


def test_partial_text_stays_out_of_the_scrollback(capsys) -> None:
    refreshes: list[int] = []
    console = PinnedConsole(refresh=lambda: refreshes.append(1), columns=lambda: 80)

    console.stream("ASSISTANT: ")
    console.stream("うん、")
    assert capsys.readouterr().out == ""
    assert console.status_text == "ASSISTANT: うん、"

    console.end_stream()
    assert capsys.readouterr().out == "ASSISTANT: うん、\n"
    assert console.status_text is None
    assert refreshes


def test_live_transcripts_never_use_carriage_returns(capsys) -> None:
    console = PinnedConsole(refresh=lambda: None, columns=lambda: 80)

    console.transient("USER: こん")
    console.transient("USER: こんにちは")
    assert capsys.readouterr().out == ""
    assert console.status_text == "USER: こんにちは"

    console.transient("USER: こんにちは", final=True)
    out = capsys.readouterr().out
    assert out == "USER: こんにちは\n"
    assert "\r" not in out
    assert console.status_text is None


def test_clearing_transient_text_discards_it() -> None:
    console = PinnedConsole(refresh=lambda: None, columns=lambda: 80)
    console.transient("USER: あ")
    console.clear_transient()
    assert console.status_text is None


def test_status_text_is_truncated_to_the_terminal_width() -> None:
    console = PinnedConsole(refresh=lambda: None, columns=lambda: 10)
    console.stream("あ" * 20)
    status = console.status_text
    assert status is not None
    assert status.startswith("…")
    assert display_width(status) <= 10


# ── availability and logging ──────────────────────


def test_log_handlers_are_rebound_and_restored() -> None:
    original, replacement = StringIO(), StringIO()
    handler = logging.StreamHandler(original)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with _bind_stream_handlers({original: replacement}):
            assert handler.stream is replacement
        assert handler.stream is original
    finally:
        root.removeHandler(handler)


def test_text_input_is_disabled_without_a_terminal(monkeypatch, caplog) -> None:
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with caplog.at_level(logging.WARNING):
        assert create_text_terminal(SimpleNamespace(text_input=True)) is None
    assert "needs a terminal" in caplog.text


def test_text_input_is_skipped_when_not_requested(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    assert create_text_terminal(SimpleNamespace(text_input=False)) is None


def test_terminal_is_interactive_requires_both_streams(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: False))
    assert terminal_is_interactive() is False


def test_missing_dependency_reports_how_to_install_it(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "prompt_toolkit", None)
    with pytest.raises(TextInputUnavailable, match="pip install prompt_toolkit"):
        check_text_input_support()


def test_prompt_toolkit_terminal_restores_the_streams_on_deactivate() -> None:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from speech_to_speech.api.openai_realtime.text_input import PromptToolkitTerminal

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        terminal = PromptToolkitTerminal()
        original_stdout, original_stderr = sys.stdout, sys.stderr

        terminal.activate()
        assert sys.stdout is not original_stdout

        terminal.deactivate()
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


def test_failed_activation_leaves_no_patched_streams_behind(monkeypatch) -> None:
    """A partial activate() must unwind, or nothing can restore stdout later."""

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from speech_to_speech.api.openai_realtime import text_input as module

    def explode(_replacements):
        raise RuntimeError("handler rebinding failed")

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        terminal = module.PromptToolkitTerminal()
        monkeypatch.setattr(module, "_bind_stream_handlers", explode)
        original_stdout, original_stderr = sys.stdout, sys.stderr

        with pytest.raises(RuntimeError, match="handler rebinding failed"):
            terminal.activate()

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


async def test_prompt_toolkit_terminal_reads_a_typed_line() -> None:
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from speech_to_speech.api.openai_realtime.text_input import PromptToolkitTerminal

    with create_pipe_input() as pipe:
        pipe.send_text("こんにちは\r")
        with create_app_session(input=pipe, output=DummyOutput()):
            terminal = PromptToolkitTerminal()
            assert await terminal.read_line() == "こんにちは"


async def test_read_line_disables_prompt_toolkits_own_sigint_handling() -> None:
    """The default installs a SIGINT handler that fails off the main thread.

    ``local`` runs this client on a ThreadManager thread, where
    ``loop.add_signal_handler`` raises RuntimeError, and on the main thread the
    teardown resets the handler that ``talk`` installed for shutdown.
    """

    from speech_to_speech.api.openai_realtime.text_input import PromptToolkitTerminal

    recorded: dict = {}

    class FakeSession:
        async def prompt_async(self, **kwargs):
            recorded.update(kwargs)
            return "hi"

    terminal = PromptToolkitTerminal.__new__(PromptToolkitTerminal)
    terminal._session = FakeSession()

    assert await terminal.read_line() == "hi"
    assert recorded["handle_sigint"] is False
