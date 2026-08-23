"""End-to-end routing of an emotion tag from the model's tokens to the TTS voice."""

from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.LLM.emotion_tags import AIZUCHI_SLOT, DEFAULT_SLOT
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    GenerateResponseRequest,
    LLMResponseChunk,
    TTSInput,
)
from speech_to_speech.TTS.emotion_refs import EmotionRefTable, VoiceReference
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
from tests.test_responses_api_language_model import (
    _make_handler,
    _make_runtime_config,
    _make_stream,
    _make_text_delta_event,
)


def _japanese_chunks(handler, deltas):
    """Drive one Japanese audio turn and return the spoken chunks it produced."""
    # Stated here rather than inherited from _make_handler: the backchannel split is
    # defined to be inactive above 1, so a change to that helper's default would turn
    # these assertions into silent false negatives.
    handler.stream_batch_sentences = 1
    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: _make_stream([_make_text_delta_event(d) for d in deltas]),
        )
    )
    cfg = _make_runtime_config()
    cfg.chat.add_item(make_user_message("ただいま"))
    request = GenerateResponseRequest(runtime_config=cfg, language_code="ja")
    return [o for o in handler.process(request) if isinstance(o, LLMResponseChunk) and o.text]


# ── model tokens -> voice slot ────────────────────────────────────────────────


def test_opening_backchannel_and_reply_get_different_voices():
    chunks = _japanese_chunks(_make_handler(), ["[喜]うん、", "それめっちゃいいじゃん。"])

    assert [c.text for c in chunks] == ["うん、", "それめっちゃいいじゃん。"]
    # The short opener is the backchannel; the rest carries the tagged emotion.
    assert [c.voice_slot for c in chunks] == [AIZUCHI_SLOT, "喜"]


def test_tag_never_reaches_the_spoken_text():
    # remove_unspeechable keeps square brackets, so nothing downstream would strip it.
    chunks = _japanese_chunks(_make_handler(), ["[", "怒", "]ふ", "ざけないで。"])

    assert "[" not in "".join(c.text for c in chunks)
    assert chunks[-1].voice_slot == "怒"


def test_untagged_reply_falls_back_to_the_default_slot():
    chunks = _japanese_chunks(_make_handler(), ["そっか、", "それは大変だったね。"])

    assert [c.voice_slot for c in chunks] == [AIZUCHI_SLOT, DEFAULT_SLOT]


def test_a_short_complete_reply_keeps_its_emotion_voice():
    # Seven characters, so short enough -- but it ends at 。, which makes it a finished
    # sentence carrying real emotion rather than an opener. Speaking it in the
    # backchannel voice would be plainly wrong.
    chunks = _japanese_chunks(_make_handler(), ["[怒]ふざけないで。"])

    assert [c.text for c in chunks] == ["ふざけないで。"]
    assert [c.voice_slot for c in chunks] == ["怒"]


def test_a_long_opening_clause_is_not_treated_as_a_backchannel():
    chunks = _japanese_chunks(_make_handler(), ["[哀]今日はほんとうに長い一日だったね、", "ゆっくり休んで。"])

    assert all(slot == "哀" for slot in [c.voice_slot for c in chunks])


# ── voice slot -> reference audio ─────────────────────────────────────────────


def test_output_processor_forwards_the_voice_slot():
    processor = object.__new__(LMOutputProcessor)
    processor.setup()

    outputs = list(
        processor.process(
            LLMResponseChunk(
                parts=[AssistantTextPart(text="うん、")],
                voice_slot=AIZUCHI_SLOT,
            )
        )
    )

    tts_inputs = [o for o in outputs if isinstance(o, TTSInput)]
    assert [t.voice_slot for t in tts_inputs] == [AIZUCHI_SLOT]


def _clone_handler(table):
    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "mlx"
    handler.cancel_scope = None
    handler.language = "japanese"
    handler.ref_audio = "TTS/fallback.wav"
    handler.ref_text = "フォールバックの音声です。"
    handler.emotion_refs = table
    handler.xvec_only = False
    handler.parity_mode = False
    handler.max_new_tokens = 1536
    handler.gen_kwargs = {}
    handler.streaming_chunk_size = 4
    handler._prepare_mlx_ref_audio = lambda ref_audio: ref_audio
    return handler


def _table():
    return EmotionRefTable(
        {
            "平": VoiceReference(audio="/refs/neutral.wav", text="お疲れー。"),
            "怒": VoiceReference(audio="/refs/anger.wav", text="勝手なことを言わないで。"),
            AIZUCHI_SLOT: VoiceReference(audio="/refs/aizuchi.wav", text="そりゃあるわよ。"),
        }
    )


@pytest.mark.parametrize(
    ("slot", "audio", "text"),
    [
        ("怒", "/refs/anger.wav", "勝手なことを言わないで。"),
        (AIZUCHI_SLOT, "/refs/aizuchi.wav", "そりゃあるわよ。"),
        # 驚 is a real slot this manifest omits: it falls back to 平, not to silence.
        ("驚", "/refs/neutral.wav", "お疲れー。"),
        (None, "/refs/neutral.wav", "お疲れー。"),
    ],
)
def test_voice_clone_uses_the_reference_for_the_slot(slot, audio, text):
    handler = _clone_handler(_table())
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return iter(())

    handler.model = SimpleNamespace(generate=fake_generate)

    list(handler._process_voice_clone("こんばんは。", slot))

    assert captured["ref_audio"] == audio
    assert captured["ref_text"] == text


def test_without_a_manifest_every_slot_uses_the_configured_reference():
    handler = _clone_handler(EmotionRefTable({}))
    captured = {}
    handler.model = SimpleNamespace(generate=lambda **kwargs: captured.update(kwargs) or iter(()))

    list(handler._process_voice_clone("こんばんは。", "怒"))

    assert captured["ref_audio"] == "TTS/fallback.wav"
    assert captured["ref_text"] == "フォールバックの音声です。"


def test_a_manifest_alone_is_enough_to_take_the_voice_clone_path():
    handler = object.__new__(Qwen3TTSHandler)
    handler.ref_audio = None
    handler.ref_spk = None
    handler.emotion_refs = _table()

    assert handler._has_voice_clone_reference() is True


def test_a_session_voice_override_does_not_drag_a_custom_voice_into_cloning():
    # The CustomVoice override clears ref_audio precisely so process() takes the
    # custom-voice branch. If the manifest keeps _has_voice_clone_reference() true, the
    # backend is handed ref_audio=None instead.
    handler = object.__new__(Qwen3TTSHandler)
    handler.ref_audio = None
    handler.ref_spk = None
    handler.emotion_refs = _table()
    handler._session_voice_override = True

    assert handler._has_voice_clone_reference() is False


def test_a_realtime_session_voice_outranks_the_emotion_table():
    # A client naming a voice is a direct instruction; the table must not override it.
    handler = _clone_handler(_table())
    handler._session_voice_override = True

    reference = handler._resolve_voice_reference("怒")

    assert reference.audio == "TTS/fallback.wav"


# ── coalescing ────────────────────────────────────────────────────────────────


def _coalesce(handler, current, queued):
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    for item in queued:
        handler.queue_in.put(item)
    return handler._coalesce_pending_tts_input(current)


def test_coalescing_stops_at_a_voice_change():
    # Without this the backchannel is merged back into the reply and is never
    # synthesised with its own voice.
    handler = object.__new__(Qwen3TTSHandler)
    current = TTSInput(text="うん、", voice_slot=AIZUCHI_SLOT, response_key="r1")
    queued = [TTSInput(text="それめっちゃいいじゃん。", voice_slot="喜", response_key="r1")]

    text, _ = _coalesce(handler, current, queued)

    assert text == "うん、"
    assert handler.queue_in.qsize() == 1


def test_coalescing_still_merges_chunks_sharing_a_voice():
    handler = object.__new__(Qwen3TTSHandler)
    current = TTSInput(text="それいいね。", voice_slot="喜", response_key="r1")
    queued = [TTSInput(text="よかったじゃん。", voice_slot="喜", response_key="r1")]

    text, _ = _coalesce(handler, current, queued)

    assert text == "それいいね。 よかったじゃん。"
    assert handler.queue_in.qsize() == 0


# ── warmup ────────────────────────────────────────────────────────────────────


def test_warmup_pre_normalizes_every_reference():
    # Otherwise the first utterance in each slot pays for a decode and resample right
    # where the voice changes.
    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "mlx"
    handler.emotion_refs = _table()
    handler._warmup_process = lambda text: iter(())
    normalized = []
    handler._prepare_mlx_ref_audio = lambda ref_audio: normalized.append(ref_audio)

    handler.warmup()

    assert sorted(normalized) == ["/refs/aizuchi.wav", "/refs/anger.wav", "/refs/neutral.wav"]


def test_process_routes_the_slot_into_synthesis():
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.ref_audio = "TTS/fallback.wav"
    handler.speaker = None
    handler.instruct = None
    handler.language = "japanese"
    handler.backend = "mlx"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(config=SimpleNamespace(tts_model_type="base"))
    handler._apply_session_voice_override = lambda model_type, runtime_config=None, response=None: None
    seen = {}

    def fake_clone(text, voice_slot=None):
        seen["slot"] = voice_slot
        yield np.zeros(512, dtype=np.int16)

    handler._process_voice_clone = fake_clone

    list(handler.process(TTSInput(text="うん、", voice_slot=AIZUCHI_SLOT, runtime_config=RuntimeConfig())))

    assert seen["slot"] == AIZUCHI_SLOT


# ── local transformers backend ────────────────────────────────────────────────


def test_every_spoken_chunk_from_the_local_backend_carries_the_same_slot():
    """All three chunk shapes this backend emits must agree on the voice.

    They are built in three different places; when only one carried the slot, a reply
    flipped from the tagged voice back to 平 on its final chunk.
    """
    from speech_to_speech.LLM import language_model as lm

    ctx = lm.StreamContext(turn_id="t1", turn_revision=0, cancel_generation=None)
    assert ctx.emotion_tags.feed("[怒]ふざけないで。") == "ふざけないで。"

    built = [
        lm._text_chunk(ctx, "ふざけ", None, None, None),
        lm._text_chunk(ctx, "ないで。", None, None, None, response_key="r1"),
    ]

    assert {chunk.voice_slot for chunk in built} == {"怒"}
