"""Emotion tags in the assistant's spoken output.

The companion prompt asks the model to open a reply with one bracketed emotion tag
(``[平]``, ``[喜]``, ...). The tag selects which reference audio the voice-cloning TTS
clones from, so a delighted reply and a consoling one no longer share one prosody.

The tag must never reach the TTS. ``remove_unspeechable`` keeps square brackets and
``remove_markdown`` does not touch them, so nothing downstream strips it: the synthesiser
would read "かくかっこ ひらき たいら" aloud. Stripping happens here instead, upstream of
both, which also keeps the tag out of the text sent to Realtime clients.

The tag DOES survive in the chat history, which is written from the provider's raw text
and never passes through here. That is deliberate: seeing its own tagged replies is what
keeps a small model emitting the format on the next turn.
"""

from __future__ import annotations

from typing import Optional

from speech_to_speech.LLM.japanese_segmenter import SOFT
from speech_to_speech.pipeline.voice_slots import (
    AIZUCHI_SLOT,
    ALL_SLOTS,
    DEFAULT_SLOT,
    EMOTION_SLOTS,
)

__all__ = [
    "AIZUCHI_MAX_CHARS",
    "AIZUCHI_SLOT",
    "ALL_SLOTS",
    "DEFAULT_SLOT",
    "EMOTION_SLOTS",
    "EmotionTagStripper",
    "is_opening_backchannel",
    "strip_emotion_tags",
]

#: Longest opening clause still treated as a backchannel rather than the reply proper.
#: The interjections the companion prompt asks for are 2-4 characters; the headroom
#: covers "なるほどね、" without swallowing a real first sentence.
AIZUCHI_MAX_CHARS = 8

# A model told to write "[喜]" in Japanese will sometimes reach for the full-width or
# lenticular brackets its keyboard offers. Those are worse than the ASCII pair, not
# better: remove_unspeechable keeps [ and ] but DELETES ［］【】, so an unhandled
# "【喜】" loses its brackets and leaves a bare 喜 for the TTS to read aloud.
_OPENERS = "[［【"
_CLOSERS = "]］】"
# One opener + one tag character + one closer. A fourth character proves the run is not
# a tag, so the hold buffer never needs to grow past it.
_MAX_HOLD = 3


def is_opening_backchannel(clause: str) -> bool:
    """True for the short interjection that opens a turn ("うん、", "そっか、").

    Length alone is not enough. "ふざけないで。" is seven characters and is a whole reply
    with real emotional content -- speaking it in the backchannel voice would be wrong.
    The second signal is the soft terminator: the Japanese segmenter cuts a turn's first
    clause at a 、 only when the reply carries on past it, so a clause ending in 、 is an
    opener by construction while one ending in 。 is a finished sentence.

    Callers must restrict this to Japanese turns. SOFT contains ASCII "," and ":", which
    a short English clause could end on without the first-clause rule ever applying.
    """
    stripped = clause.strip()
    return bool(stripped) and len(stripped) <= AIZUCHI_MAX_CHARS and stripped[-1] in SOFT


class EmotionTagStripper:
    """Removes ``[emotion]`` tags from a token stream, remembering the last one seen.

    One instance per turn: the hold buffer spans deltas, because a provider is free to
    split ``[喜]`` into ``[`` and ``喜]``. Text is released as soon as it is known not to
    be part of a tag, so holding costs at most three characters of latency and only when
    an opening bracket actually arrives.

    Tags are honoured anywhere in the stream, not just at the head, so a model that tags
    every sentence degrades to "the last tag wins" instead of speaking the extras.
    """

    def __init__(self) -> None:
        self._hold = ""
        self.emotion: Optional[str] = None

    @property
    def slot(self) -> str:
        """The emotion slot for this turn, falling back when the model tagged nothing."""
        return self.emotion or DEFAULT_SLOT

    def feed(self, delta: str) -> str:
        """Consume one streaming delta, returning it without any tag characters."""
        out: list[str] = []
        for ch in delta:
            if self._hold:
                self._hold += ch
                if ch in _CLOSERS:
                    tag = self._hold[1:-1]
                    if tag in EMOTION_SLOTS:
                        self.emotion = tag
                    else:
                        # Not an emotion tag -- a real bracketed aside. Speak it.
                        out.append(self._hold)
                    self._hold = ""
                elif ch in _OPENERS:
                    # Two openers in a row -- the earlier run cannot close any more.
                    # Release it and restart the hold on this bracket.
                    out.append(self._hold[:-1])
                    self._hold = ch
                elif len(self._hold) > _MAX_HOLD:
                    out.append(self._hold)
                    self._hold = ""
                continue
            if ch in _OPENERS:
                self._hold = ch
                continue
            out.append(ch)
        return "".join(out)

    def flush(self) -> str:
        """Release an unterminated hold at end of turn, so no character is swallowed."""
        held, self._hold = self._hold, ""
        return held


def strip_emotion_tags(text: str) -> tuple[str, str]:
    """Strip tags from complete text, returning the slot to speak it in.

    For non-streaming paths. Unlike :attr:`EmotionTagStripper.emotion` the slot is never
    ``None``: an untagged reply resolves to :data:`DEFAULT_SLOT` here rather than leaving
    every caller to spell that fallback out again.
    """
    stripper = EmotionTagStripper()
    clean = stripper.feed(text) + stripper.flush()
    return clean, stripper.slot
