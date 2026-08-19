"""Drop Japanese Whisper hallucinations before they become a language-model turn.

On near-silence, Japanese Whisper emits YouTube outro boilerplate from its training
data. Without this filter a cough produces a full LLM turn: phase 0 measured 2 of 12
real-voice turns as hallucinations.

Japanese only. "Thanks for watching!" is something an English speaker actually says.
"""

from __future__ import annotations

from typing import Optional

# Whisper reports "ja"; some callers pass ISO 639-2 or a region tag.
_JAPANESE_LANGUAGE_CODES = frozenset({"ja", "jpn"})

_HALLUCINATIONS = frozenset(
    {
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございました。",
        "ご覧いただきありがとうございます",
        "チャンネル登録お願いします",
        "チャンネル登録よろしくお願いします",
        "おわり",
        "終わり",
        "字幕視聴者",
        "最後までご視聴いただきありがとうございました",
        "エンディング",
        "Thanks for watching!",
        "Thank you for watching",
    }
)

# Stripped before the length check so a transcript of pure punctuation counts as empty.
_PUNCTUATION = "。、．，！？!?…「」『』 \t　\n"

# 「うん」「へえ」are real backchannels and must survive.
_MIN_MEANINGFUL_CHARS = 2


def is_meaningful_transcription(text: str, language_code: Optional[str]) -> bool:
    """False when the transcript is a known hallucination or too short to be speech.

    Deliberate hole: a bare 「ありがとうございました」 passes. It is both a Whisper
    hallucination on near-silence and a phrase people genuinely say, and adding it to
    the set would throw away the real utterances along with the false ones. Separating
    the two needs utterance length and VAD confidence, neither of which reaches this
    stage yet, so the phrase is left through on purpose.
    """
    if not language_code:
        return True
    if language_code.strip().lower().replace("_", "-").split("-")[0] not in _JAPANESE_LANGUAGE_CODES:
        return True

    stripped = text.strip()
    if stripped in _HALLUCINATIONS:
        return False
    if len(stripped.strip(_PUNCTUATION)) < _MIN_MEANINGFUL_CHARS:
        return False
    return True
