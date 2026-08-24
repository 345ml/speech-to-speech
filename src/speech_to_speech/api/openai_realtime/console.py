"""Console output ports for the packaged microphone/speaker client.

The client writes three kinds of output: permanent lines, text that streams
into a line still being written, and a transient line that is overwritten in
place (live speech transcripts). Plain stdout renders the last two with
carriage returns, which is why they are separate operations here rather than
one ``write``: a pinned input prompt cannot use carriage returns at all, and
needs to route both of them somewhere else entirely.

``StdoutConsole`` is the default and reproduces the original behaviour exactly.
``PinnedConsole`` in :mod:`.text_input` is the alternative used when typed
input pins a prompt to the bottom of the terminal.
"""

from __future__ import annotations

import unicodedata
from typing import Protocol


class ConsoleSink(Protocol):
    """Where the client's console output goes."""

    def line(self, text: str) -> None:
        """Write one permanent line."""

    def stream(self, text: str) -> None:
        """Append to the line currently being written."""

    def end_stream(self) -> None:
        """Terminate the line currently being written."""

    def transient(self, text: str, *, final: bool = False) -> None:
        """Show text that later calls may overwrite, or commit it when final."""

    def clear_transient(self) -> None:
        """Drop any transient text without committing it."""


class StdoutConsole:
    """Print straight to stdout, overwriting transient text with carriage returns."""

    def __init__(self) -> None:
        self._transient_width = 0

    def line(self, text: str) -> None:
        print(text, flush=True)

    def stream(self, text: str) -> None:
        print(text, end="", flush=True)

    def end_stream(self) -> None:
        print("", flush=True)

    def transient(self, text: str, *, final: bool = False) -> None:
        # Padding and erasing are measured in terminal columns, not characters:
        # a Japanese transcript is twice as wide as len() reports, and erasing
        # by character count leaves the tail of the previous line on screen.
        width = display_width(text)
        padded = text + (" " * max(0, self._transient_width - width))
        if final:
            print(f"\r{padded}", flush=True)
            self._transient_width = 0
            return
        print(f"\r{padded}", end="", flush=True)
        self._transient_width = width

    def clear_transient(self) -> None:
        if self._transient_width == 0:
            return
        print("\r" + (" " * self._transient_width) + "\r", end="", flush=True)
        self._transient_width = 0


def _char_width(char: str) -> int:
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def display_width(text: str) -> int:
    """Terminal columns ``text`` occupies, counting CJK characters as two."""

    return sum(_char_width(char) for char in text)


def fit(text: str, columns: int) -> str:
    """Truncate ``text`` from the left so its tail fits within ``columns``."""

    if columns <= 0:
        # A terminal that reports no width gets no status text, rather than all
        # of it: an unbounded string here would wrap and push the prompt away.
        return ""
    if display_width(text) <= columns:
        return text
    budget = columns - 1
    kept: list[str] = []
    used = 0
    for char in reversed(text):
        width = _char_width(char)
        if used + width > budget:
            break
        kept.append(char)
        used += width
    return "…" + "".join(reversed(kept))
