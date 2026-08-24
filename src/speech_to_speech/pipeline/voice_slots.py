"""The vocabulary of the ``voice_slot`` field on pipeline messages.

A voice slot names which reference audio the voice-cloning TTS should clone for one
utterance. The LLM stage picks an emotion slot from the tag the model emits; the TTS
stage resolves that slot to a reference clip. Both stages need these names, and neither
should import the other, so they live here beside the message field itself
(:class:`speech_to_speech.pipeline.messages.TTSInput`).
"""

from __future__ import annotations

#: Slots the model may name. ``相槌`` is not one of them -- it is assigned by position
#: (the opening interjection of a turn), never by the model.
EMOTION_SLOTS: tuple[str, ...] = ("平", "喜", "哀", "怒", "驚", "優")

#: Slot for the short reaction that opens a turn ("うん、" "そっか、" "へえ、").
AIZUCHI_SLOT = "相槌"

#: Slot used when the model emits no tag, or one that is not in EMOTION_SLOTS.
DEFAULT_SLOT = "平"

#: Every slot a reference manifest may define.
ALL_SLOTS: tuple[str, ...] = EMOTION_SLOTS + (AIZUCHI_SLOT,)
