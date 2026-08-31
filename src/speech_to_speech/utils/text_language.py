"""Language-dependent text conventions shared by more than one pipeline stage.

These started in ``LLM/utils.py`` because the language model was the only stage that
needed them. It is not the only one: the TTS stage re-joins the clauses it coalesced out
of its input queue, and joining Japanese clauses with a half-width space produces a
character the TTS reads aloud -- the same bug, in the same words, one stage later.

Importing ``LLM.utils`` from ``TTS`` would have fixed that and created a TTS -> LLM
package dependency; ``STT/hallucinations.py`` had faced the same choice and kept a
private copy of the predicate rather than take it. Two copies is how they drift apart,
and the bug this module exists to close is precisely two stages disagreeing about one
rule -- so the rule lives here, under ``utils``, which every stage may already import,
and all three stages ask it rather than answering it themselves.
"""

from __future__ import annotations

from typing import Optional

# Whisper reports "ja"; some callers pass an ISO 639-2 code or a region tag.
_JAPANESE_LANGUAGE_CODES = frozenset({"ja", "jpn"})


def is_japanese_language(language_code: Optional[str]) -> bool:
    """True when this turn's language needs the Japanese-specific text handling."""
    if not language_code:
        return False
    return language_code.strip().lower().replace("_", "-").split("-")[0] in _JAPANESE_LANGUAGE_CODES


def sentence_join_separator(language_code: Optional[str]) -> str:
    """The string that joins a batch of sentences before it is spoken.

    Japanese gets an empty separator; every other language keeps the single space it
    has always had.
    """
    return "" if is_japanese_language(language_code) else " "
