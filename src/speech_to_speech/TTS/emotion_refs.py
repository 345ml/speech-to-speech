"""Per-emotion reference audio for Qwen3-TTS voice cloning.

A manifest maps each voice slot to the reference clip the TTS clones from and that
clip's transcription::

    {
      "平":   {"audio": "neutral.wav", "text": "お疲れー。まだ残ってたの?"},
      "相槌": {"audio": "aizuchi.wav", "text": "そりゃあるわよ。何度もね。"}
    }

``audio`` may be relative to the manifest, which keeps the clips and the manifest
together in one directory outside the repository -- reference clips are third-party
voice recordings and are not committed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from speech_to_speech.pipeline.voice_slots import ALL_SLOTS, DEFAULT_SLOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceReference:
    """One slot's reference clip and its transcription."""

    #: None when the voice comes from a GGML ref_spk/ref_rvq pair instead of a clip.
    audio: Optional[str]
    text: str


class EmotionRefTable:
    """Slot -> reference clip, with a fallback chain that never fails to resolve.

    ``resolve`` walks requested slot -> ``平`` -> ``None``. A ``None`` return means the
    caller keeps whatever single reference it was configured with, so a missing or
    partial manifest degrades to the pre-existing single-voice behaviour rather than
    raising mid-utterance.
    """

    def __init__(self, references: dict[str, VoiceReference]) -> None:
        # Copied, not aliased: NO_EMOTION_REFS is shared across every handler
        # instance, so the table must not be mutable through a caller's dict.
        self._references = dict(references)

    def __bool__(self) -> bool:
        return bool(self._references)

    def references(self) -> list[VoiceReference]:
        return list(self._references.values())

    def resolve(self, slot: Optional[str]) -> Optional[VoiceReference]:
        """The reference for ``slot``, falling back to the default slot."""
        if slot is not None:
            reference = self._references.get(slot)
            if reference is not None:
                return reference
            logger.debug("No reference audio for voice slot %r; falling back", slot)
        return self._references.get(DEFAULT_SLOT)

    @classmethod
    def load(cls, manifest_path: str | Path) -> "EmotionRefTable":
        """Read and validate a manifest.

        Raises rather than warning: a typo in the manifest would otherwise surface as a
        voice that silently never changes, which is far harder to notice than a failure
        at startup.
        """
        path = Path(manifest_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"qwen3_tts_emotion_refs manifest not found: {path}")

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"qwen3_tts_emotion_refs manifest is not valid JSON ({path}): {e}") from e

        if not isinstance(raw, dict):
            raise ValueError(f"qwen3_tts_emotion_refs manifest must be a JSON object, got {type(raw).__name__}: {path}")

        references: dict[str, VoiceReference] = {}
        for slot, entry in raw.items():
            if slot not in ALL_SLOTS:
                raise ValueError(f"Unknown voice slot {slot!r} in {path}. Supported slots: {', '.join(ALL_SLOTS)}")
            if not isinstance(entry, dict):
                raise ValueError(f"Voice slot {slot!r} in {path} must map to an object with 'audio' and 'text'.")

            audio = entry.get("audio")
            text = entry.get("text")
            if not isinstance(audio, str) or not audio.strip():
                raise ValueError(f"Voice slot {slot!r} in {path} is missing a non-empty 'audio' path.")
            # An unset or mismatched ref_text leaks the reference's own words into the
            # start of the synthesised audio, so it is required, not optional.
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Voice slot {slot!r} in {path} is missing a non-empty 'text' transcription.")

            audio_path = Path(audio).expanduser()
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(f"Reference audio for voice slot {slot!r} not found: {audio_path}")

            references[slot] = VoiceReference(audio=str(audio_path.resolve()), text=text)

        if not references:
            raise ValueError(f"qwen3_tts_emotion_refs manifest defines no voice slots: {path}")
        if DEFAULT_SLOT not in references:
            raise ValueError(
                f"qwen3_tts_emotion_refs manifest must define the {DEFAULT_SLOT!r} slot, "
                f"which every other slot falls back to: {path}"
            )

        logger.info("Loaded %d Qwen3-TTS voice slots from %s: %s", len(references), path, ", ".join(references))
        return cls(references)


#: Shared "no manifest configured" table. A table is immutable once built, so one
#: instance can serve as the class-level default for every handler.
NO_EMOTION_REFS = EmotionRefTable({})
