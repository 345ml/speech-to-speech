"""Typed text input for the packaged microphone/speaker client.

Lets the user type a turn instead of speaking it, while the microphone stays
live. A typed turn reaches the server as the standard Realtime triple
``response.cancel`` -> ``conversation.item.create`` -> ``response.create``, so
the server needs no special handling: the text is appended to the same chat
history a transcript would be, and the reply is spoken exactly as it would be
for a spoken turn.

The module is import-safe without ``prompt_toolkit``; the dependency is only
imported when a terminal is actually created.

Layering, weakest dependency first:

* :class:`TextTurnSubmitter` -- protocol only. No terminal, no websocket type,
  no audio hardware, so the wire sequence is unit-testable on its own.
* :class:`PinnedConsole` -- console output for a pinned prompt. Needs a redraw
  callback, not ``prompt_toolkit`` itself.
* :class:`PromptToolkitTerminal` -- the only code that imports
  ``prompt_toolkit``.
* :class:`TextInputCoordinator` -- joins the read loop to the client session.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from threading import Event
from typing import IO, TYPE_CHECKING, Any, Protocol

from .console import ConsoleSink, fit

if TYPE_CHECKING:
    from .audio_client import RealtimeAudioClientConfig

logger = logging.getLogger(__name__)

_INSTALL_HINT = "Typed text input needs prompt_toolkit. Install it with: pip install prompt_toolkit"


class TextInputUnavailable(RuntimeError):
    """Typed input was requested but cannot be provided."""


def check_text_input_support() -> None:
    """Raise :class:`TextInputUnavailable` if ``prompt_toolkit`` is missing.

    Called from the CLI so a stale environment fails before models load,
    rather than after a multi-minute warmup.
    """

    try:
        import prompt_toolkit  # noqa: F401
    except ImportError as exc:
        raise TextInputUnavailable(_INSTALL_HINT) from exc


def terminal_is_interactive() -> bool:
    """Whether both stdin and stdout are terminals."""

    for stream in (sys.stdin, sys.stdout):
        isatty = getattr(stream, "isatty", None)
        if not callable(isatty):
            return False
        try:
            if not isatty():
                return False
        except ValueError:  # closed stream
            return False
    return True


def _stream_handlers() -> Iterator[logging.StreamHandler]:
    """Every ``StreamHandler`` reachable in the logging tree.

    Walking only the root logger is not enough: uvicorn installs its own
    handlers on ``uvicorn``/``uvicorn.access`` with ``propagate=False``, bound
    to the real stderr before this client thread starts, so under ``local``
    they would write straight through the pinned prompt.
    """

    loggers: list[Any] = [logging.getLogger()]
    loggers.extend(
        candidate for candidate in logging.root.manager.loggerDict.values() if isinstance(candidate, logging.Logger)
    )
    seen: set[int] = set()
    for log in loggers:
        for handler in log.handlers:
            if isinstance(handler, logging.StreamHandler) and id(handler) not in seen:
                seen.add(id(handler))
                yield handler


@contextmanager
def _bind_stream_handlers(replacements: Mapping[IO[str], IO[str]]) -> Iterator[None]:
    """Point root ``StreamHandler``s at replacement streams for the duration.

    ``logging.basicConfig`` runs before the client thread starts, so its
    handler holds a direct reference to the original ``sys.stderr`` and would
    write straight through a pinned prompt. Rebinding routes those records
    through the patched stream instead, where they scroll above the prompt.
    """

    rebound: list[tuple[logging.StreamHandler, IO[str]]] = []
    for handler in _stream_handlers():
        replacement = replacements.get(handler.stream)
        if replacement is None or replacement is handler.stream:
            continue
        rebound.append((handler, handler.stream))
        handler.setStream(replacement)
    try:
        yield
    finally:
        for handler, original in rebound:
            handler.setStream(original)


class PinnedConsole:
    """Console output that keeps partial text off the scrollback.

    A pinned prompt is redrawn after every write, and a redraw erases the row
    the prompt occupies. Text written without a trailing newline shares that
    row, so it is erased by the next write. Only completed lines can safely
    reach the scrollback; anything still in progress is shown in the prompt's
    bottom toolbar instead.
    """

    def __init__(self, refresh: Callable[[], None], columns: Callable[[], int]) -> None:
        self._refresh = refresh
        self._columns = columns
        self._stream = ""
        self._status = ""

    def line(self, text: str) -> None:
        print(text, flush=True)

    def stream(self, text: str) -> None:
        self._stream += text
        self._refresh()

    def end_stream(self) -> None:
        pending = self._stream
        self._stream = ""
        if pending:
            self.line(pending)
        self._refresh()

    def transient(self, text: str, *, final: bool = False) -> None:
        self._status = ""
        if final:
            self.line(text)
        else:
            self._status = text
        self._refresh()

    def clear_transient(self) -> None:
        if not self._status:
            return
        self._status = ""
        self._refresh()

    @property
    def status_text(self) -> str | None:
        """Text for the bottom toolbar, or ``None`` to hide it."""

        pending = self._stream or self._status
        if not pending:
            return None
        return fit(pending, self._columns())


class TextTerminal(Protocol):
    """A terminal that can read typed lines while other output scrolls past."""

    @property
    def console(self) -> ConsoleSink: ...

    def activate(self) -> None: ...

    def deactivate(self) -> None: ...

    async def read_line(self) -> str: ...


class PromptToolkitTerminal:
    """A ``prompt_toolkit`` prompt pinned to the bottom of the terminal."""

    def __init__(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout

        self._patch_stdout = patch_stdout
        # The same prefix a spoken turn is rendered with, so an accepted line
        # is left in the scrollback looking like any other user turn.
        self._session: Any = PromptSession(message="USER: ", bottom_toolbar=self._toolbar)
        self._console = PinnedConsole(refresh=self._invalidate, columns=self._terminal_columns)
        self._stack: ExitStack | None = None

    @property
    def console(self) -> ConsoleSink:
        return self._console

    def _toolbar(self) -> str | None:
        return self._console.status_text

    def _invalidate(self) -> None:
        # A no-op while the application is not running, so this is safe to call
        # from event handling before the first prompt is drawn.
        self._session.app.invalidate()

    def _terminal_columns(self) -> int:
        try:
            return int(self._session.output.get_size().columns)
        except Exception:  # pragma: no cover - depends on the terminal
            return 80

    def activate(self) -> None:
        if self._stack is not None:
            return
        with ExitStack() as stack:
            # Registered first so it unwinds last: whatever else fails, the
            # terminal is handed back in the mode we found it in.
            self._push_termios_restore(stack)
            original_stdout, original_stderr = sys.stdout, sys.stderr
            stack.enter_context(self._patch_stdout(raw=False))
            stack.enter_context(_bind_stream_handlers({original_stdout: sys.stdout, original_stderr: sys.stderr}))
            # Only now is the stack complete; until pop_all() succeeds the
            # `with` unwinds whatever was entered, so a partial activation
            # cannot strand a patched stdout that nothing can restore.
            self._stack = stack.pop_all()

    def deactivate(self) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.close()

    async def read_line(self) -> str:
        # handle_sigint=False is required, not cosmetic. The default installs a
        # SIGINT handler through loop.add_signal_handler, which raises
        # RuntimeError off the main thread (the `local` command runs this
        # client on a ThreadManager thread) and, on the main thread, resets the
        # handler on teardown -- silently replacing `talk`'s own shutdown
        # handler. Ctrl-C still arrives as KeyboardInterrupt either way,
        # because raw mode delivers it as a keypress rather than a signal.
        return await self._session.prompt_async(handle_sigint=False)

    @staticmethod
    def _push_termios_restore(stack: ExitStack) -> None:
        try:
            import termios

            fileno = sys.stdin.fileno()
            saved = termios.tcgetattr(fileno)
        except Exception:  # pragma: no cover - not a POSIX terminal
            return

        def restore() -> None:
            try:
                termios.tcsetattr(fileno, termios.TCSADRAIN, saved)
            except Exception:  # pragma: no cover - terminal already gone
                logger.debug("Could not restore terminal attributes", exc_info=True)

        stack.callback(restore)


def create_text_terminal(config: RealtimeAudioClientConfig) -> TextTerminal | None:
    """Build a terminal for typed input, or ``None`` to stay voice-only.

    A missing dependency is an error the caller asked for; a non-interactive
    stdin is a property of the environment and must not kill a session that
    would otherwise work.
    """

    if not config.text_input:
        return None
    if not terminal_is_interactive():
        logger.warning("Typed text input needs a terminal; continuing with voice input only")
        return None
    check_text_input_support()
    return PromptToolkitTerminal()


class TextTurnSubmitter:
    """Send one typed turn, interrupting whatever the assistant is saying."""

    def __init__(self, conn: Any, *, playback: Any, console: ConsoleSink) -> None:
        self._conn = conn
        self._playback = playback
        self._console = console

    async def submit(self, text: str) -> None:
        # Locally first: the speaker goes quiet the instant Enter is pressed,
        # rather than one websocket round trip later.
        self._playback.clear()
        self._console.clear_transient()
        # Unconditional, not gated on having seen `response.created`. A turn
        # triggered by speech is queued (`response_pending`) before the client
        # is told about it, and creating a response during that window is
        # rejected as `conversation_already_has_active_response`. Cancelling
        # with nothing in flight is a no-op that emits no events.
        await self._conn.send({"type": "response.cancel"})
        # Must follow the cancel: `conversation.item.create` defers the item
        # while a response is in progress, and cancelling clears that flag
        # synchronously, so the text is in the chat before generation starts.
        await self._conn.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        await self._conn.send({"type": "response.create"})
        # After the sends, not before: `clear()` above stops whatever was already
        # armed, and a typed turn waits on the same generation gap as a spoken one.
        self._playback.start_thinking()


class TextInputCoordinator:
    """Read typed lines until the user quits or the session stops."""

    def __init__(
        self,
        *,
        terminal: TextTerminal,
        submitter: TextTurnSubmitter,
        stop_event: Event,
    ) -> None:
        self._terminal = terminal
        self._submitter = submitter
        self._stop_event = stop_event

    async def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    text = await self._terminal.read_line()
                except (EOFError, KeyboardInterrupt):
                    # Ctrl-C reaches us as a keypress rather than a signal,
                    # because the prompt puts the terminal in raw mode. Letting
                    # it escape the task would bypass the session's shutdown.
                    return
                text = text.strip()
                if text:
                    await self._submitter.submit(text)
        finally:
            self._stop_event.set()
