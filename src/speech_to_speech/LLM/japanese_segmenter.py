"""Streaming Japanese clause segmentation for TTS.

NLTK's punkt tokenizer does not treat 。 as a sentence terminator, so a Japanese
reply reaches the TTS as a single sentence and nothing is spoken until the language
model has finished generating. The reply's length lands directly on time-to-first-audio.

The core asymmetry: **the first clause of a turn is cut aggressively, every later one
conservatively.** Only the first clause sits on the TTFA critical path. Splitting later
clauses at 、 buys no latency and costs prosody, because a neural TTS needs the fuller
clause to place accent and intonation.

Ported from the companion phase 0 harness, which measured 166 ms of TTFA coming from
releasing a 2-4 character opening interjection (「え、」「うん、」「そっか、」) before
the rest of the reply exists.
"""

from __future__ import annotations

from dataclasses import dataclass

HARD = "。．.！!？?…‥\n"
SOFT = "、，,・：:；;"
CLOSERS = "」』）)"
_BLANK = " \t　"


@dataclass(frozen=True)
class JapaneseSegmenterConfig:
    """Cut thresholds, in characters."""

    # The first clause cuts at the FIRST boundary of any kind. Japanese opening
    # interjections are 2-4 characters and releasing one immediately IS the latency win.
    first_min_chars: int = 2
    # A 。 is a real sentence end: cutting there is always prosodically correct, so it
    # needs only a floor that rejects debris, not a length quota.
    hard_min_chars: int = 4
    # A 、 mid-reply is NOT a good cut. Only accept one once the clause is already long.
    soft_after_chars: int = 24
    # Forced flush so audio never stalls behind generation.
    max_chars: int = 60


def _is_speakable(text: str) -> bool:
    """False when the fragment is only punctuation or whitespace."""
    return any(ch not in HARD + SOFT + CLOSERS + _BLANK for ch in text)


def _protected(buf: str, i: int) -> bool:
    """True when the punctuation at ``i`` sits inside an atomic run.

    Guards decimals (``3.14``) and ASCII abbreviations (``e.g``). Note that the
    neighbours are what matter: ``buf[i]`` is the punctuation itself, so inspecting
    it decides nothing.
    """
    if i == 0 or i + 1 >= len(buf):
        return False
    before, after = buf[i - 1], buf[i + 1]
    if before.isdigit() and after.isdigit():
        return True
    if before.isascii() and before.isalnum() and after.isascii() and after.isalnum():
        return True
    return False


class JapaneseClauseTokenizer:
    """Sentence tokenizer for ONE assistant turn.

    Stateful on purpose: it has to know whether the turn's first clause has already
    been released, because that clause is cut by a different rule than the rest.
    Construct a new instance per turn.

    Shaped to drop into ``sent_tokenize_preserving_markdown_code(text, tokenizer)``:
    the caller re-invokes it with the un-flushed remainder, flushes ``parts[:-1]``,
    and keeps ``parts[-1]`` as the new buffer.
    """

    def __init__(self, config: JapaneseSegmenterConfig | None = None) -> None:
        self.config = config or JapaneseSegmenterConfig()
        self._emitted = 0

    def __call__(self, text: str) -> list[str]:
        parts: list[str] = []
        buf = text
        emitted = self._emitted
        while True:
            cut = self._find_cut(buf, first=emitted == 0)
            if cut is None:
                break
            parts.append(buf[:cut])
            buf = buf[cut:].lstrip(_BLANK)
            emitted += 1
        parts.append(buf)
        self._emitted = emitted
        return parts

    def _find_cut(self, buf: str, *, first: bool) -> int | None:
        """Return the exclusive index of the earliest legal cut, or None."""
        config = self.config
        last_soft: int | None = None

        for i, ch in enumerate(buf):
            if ch not in HARD and ch not in SOFT:
                continue
            if _protected(buf, i):
                continue
            end = i + 1
            # Absorb closing brackets that immediately follow the terminator.
            while end < len(buf) and buf[end] in CLOSERS:
                end += 1
            if not _is_speakable(buf[:end]):
                continue
            if ch in SOFT:
                # Remember every legal SOFT position in every branch: a forced flush
                # falls back to the last one so it never cuts mid-phrase.
                last_soft = end
            if first:
                if end >= config.first_min_chars:
                    return end
            elif ch in HARD:
                if end >= config.hard_min_chars:
                    return end
            elif end >= config.soft_after_chars:
                return end

        if len(buf) > config.max_chars:
            end = last_soft if last_soft else config.max_chars
            if _is_speakable(buf[:end]):
                return end
        return None
