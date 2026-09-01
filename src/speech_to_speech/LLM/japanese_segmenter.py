"""Streaming Japanese clause segmentation for TTS.

NLTK's punkt tokenizer does not treat 。 as a sentence terminator, so a Japanese
reply reaches the TTS as a single sentence and nothing is spoken until the language
model has finished generating. The reply's length lands directly on time-to-first-audio.

The core asymmetry: **the first clause of a turn is cut by a different rule than every
later one.** Only the first clause sits on the TTFA critical path. Splitting later
clauses at 、 buys no latency and costs prosody, because a neural TTS needs the fuller
clause to place accent and intonation.

The first-clause floor sits ABOVE interjection length, which reverses the rule this
module shipped with. Cutting at the very first boundary released the 2-3 character
interjection the prompt asks the model to open on, so segment 0 -- the only part the
listener hears first -- was almost always the same two words. Measured over 90 turns
(Qwen3-4B-Instruct-2507 Q4_K_M, the RUN.md prompt): 「うん、」 94.4%, 「えっ、」 5.6%,
3 distinct openers in total. Raising the floor to 6 changes nothing about what the
model generates and lets the rest of the reaction into segment 0 instead: 60 distinct
openers, most frequent 15.6%, for 106 ms of time-to-first-audio at p50.

Two rationales the original floor of 2 rested on were re-measured and did not hold:
the TTS instability on short input is a sampling-temperature effect, not a length one
(greedy is bit-identical across runs at every length), and the playback underrun it
was meant to avoid does not occur -- the gap between consecutive clauses leaving the
language model is ~136 ms at p50, well inside even a 3-character clause's audio.
"""

from __future__ import annotations

from dataclasses import dataclass

from speech_to_speech.LLM.utils import MARKDOWN_SENTINEL

HARD = "。．.！!？?…‥\n"
SOFT = "、，,・：:；;"
CLOSERS = "」』）)"
_BLANK = " \t　"
# Hoisted: `_is_speakable` runs on the per-delta streaming path, and building this
# inside the generator body concatenated four strings once per character.
_UNSPEAKABLE = frozenset(HARD + SOFT + CLOSERS + _BLANK)
# `\n` is a HARD terminator, so it stays with the clause that ends on it. Stripping it
# from the remainder as well keeps a source newline from being emitted twice.
_STRIP_AFTER_CUT = _BLANK + "\n"


@dataclass(frozen=True)
class JapaneseSegmenterConfig:
    """Cut thresholds, in characters."""

    # The first clause cuts at the first boundary of any kind AT OR AFTER this floor.
    # Set above interjection length on purpose: cutting at the very first boundary made
    # segment 0 a bare 2-3 character interjection on 100% of measured turns, so every
    # turn opened on the same word. See the module docstring for the measurements.
    first_min_chars: int = 6
    # A 。 is a real sentence end: cutting there is always prosodically correct, so it
    # needs only a floor that rejects debris, not a length quota.
    hard_min_chars: int = 4
    # A 、 mid-reply is NOT a good cut. Only accept one once the clause is already long.
    soft_after_chars: int = 24
    # Forced flush so audio never stalls behind generation.
    max_chars: int = 60


def _is_speakable(text: str) -> bool:
    """False when the fragment is only punctuation or whitespace."""
    return any(ch not in _UNSPEAKABLE for ch in text)


def _outside_markdown_sentinel(buf: str, end: int) -> int:
    """Move ``end`` back so a cut never lands inside a Markdown sentinel.

    ``sent_tokenize_preserving_markdown_code`` replaces every complete code span and
    matched emphasis run with ``\x00markdown-<kind>-N\x00`` before the tokenizer runs
    and restores it afterwards by exact string match. punkt never split one; a forced
    flush cuts at an arbitrary character index and can. Neither fragment would then
    restore, so a raw NUL byte and the literal text ``markdown-code-`` would reach the
    TTS and the code span itself would be lost.

    An odd number of sentinel characters before ``end`` means a sentinel is still open
    there, so cut in front of the one that opened it.
    """
    if buf.count(MARKDOWN_SENTINEL, 0, end) % 2 == 0:
        return end
    return buf.rfind(MARKDOWN_SENTINEL, 0, end)


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
            buf = buf[cut:].lstrip(_STRIP_AFTER_CUT)
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
            end = last_soft if last_soft is not None else config.max_chars
            end = _outside_markdown_sentinel(buf, end)
            if end > 0 and _is_speakable(buf[:end]):
                return end
        return None
