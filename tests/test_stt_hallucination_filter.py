"""The filter that keeps a Japanese Whisper hallucination from becoming an LLM turn.

On near-silence Japanese Whisper emits YouTube boilerplate from its training data.
Without the filter a single cough runs a whole LLM turn (2 of the 12 real-voice turns
measured in phase 0).

English is not filtered: "Thanks for watching!" is something an English speaker says.

The filter runs in ``TranscriptionNotifier``, not in ``BaseSTTHandler.should_emit_output``.
A hallucination is demoted to an empty transcript there, so it takes the existing
"empty final STT result" path: the client-visible transcription item is still closed
and the LLM is still not triggered. Dropping the message at the handler would suppress
the completion event, leaving the item open on the client forever and leaking one
``InputItemState`` per dropped utterance.
"""

from queue import Queue
from threading import Event

from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.hallucinations import is_meaningful_transcription
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


def test_japanese_youtube_boilerplate_is_dropped() -> None:
    assert not is_meaningful_transcription("ご視聴ありがとうございました", "ja")
    assert not is_meaningful_transcription("ご視聴ありがとうございました。", "ja")
    assert not is_meaningful_transcription("チャンネル登録お願いします", "ja")


def test_surrounding_whitespace_does_not_smuggle_a_hallucination_through() -> None:
    assert not is_meaningful_transcription("  ご視聴ありがとうございました  ", "ja")


def test_real_japanese_speech_survives() -> None:
    assert is_meaningful_transcription("今日は疲れた", "ja")
    assert is_meaningful_transcription("コンビニ寄ってくるけど何かいる？", "ja")


def test_bare_thanks_still_passes_and_that_is_a_known_hole() -> None:
    # 「ありがとうございました」 on its own is a phrase people genuinely say, so simply
    # adding it to the blocklist would throw away real speech. The hole stays open until
    # utterance length and VAD confidence reach this stage and can separate the two.
    assert is_meaningful_transcription("ありがとうございました", "ja")


def test_punctuation_only_output_is_dropped() -> None:
    assert not is_meaningful_transcription("。。。", "ja")
    assert not is_meaningful_transcription("、", "ja")
    assert not is_meaningful_transcription("   ", "ja")


def test_single_character_output_is_dropped() -> None:
    assert not is_meaningful_transcription("あ", "ja")


def test_two_characters_survive() -> None:
    # 「うん」 is a real backchannel. It must not be dropped.
    assert is_meaningful_transcription("うん", "ja")


def test_language_code_with_surrounding_whitespace_is_still_japanese() -> None:
    assert not is_meaningful_transcription("ご視聴ありがとうございました", " ja")


def test_english_is_not_filtered() -> None:
    assert is_meaningful_transcription("Thanks for watching!", "en")
    assert is_meaningful_transcription("ご視聴ありがとうございました", "en")


def test_missing_language_is_not_filtered() -> None:
    assert is_meaningful_transcription("ご視聴ありがとうございました", None)


def _notifier(text_output_queue: Queue, should_listen: Event | None = None) -> TranscriptionNotifier:
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=text_output_queue, should_listen=should_listen)
    return notifier


def _completed(text_output_queue: Queue) -> TranscriptionCompletedEvent:
    event = text_output_queue.get_nowait()
    assert isinstance(event, TranscriptionCompletedEvent)
    return event


def test_a_hallucinated_final_does_not_reach_the_language_model() -> None:
    # An empty transcript is what stops the LLM: `RealtimeService._on_transcription_completed`
    # only appends a user message and triggers a response when `event.transcript` is truthy.
    text_output_queue: Queue = Queue()
    notifier = _notifier(text_output_queue)

    assert list(notifier.process(Transcription(text="ご視聴ありがとうございました", language_code="ja"))) == []

    assert _completed(text_output_queue).transcript == ""


def test_a_dropped_hallucination_still_terminalizes_the_item() -> None:
    # The whole reason the filter lives here. The client may already have received
    # `conversation.item.input_audio_transcription.delta` for the hallucination, so the
    # completed event still has to be emitted; without it the item stays open on the
    # client and its InputItemState is never popped.
    text_output_queue: Queue = Queue()
    should_listen = Event()
    notifier = _notifier(text_output_queue, should_listen=should_listen)

    assert list(notifier.process(PartialTranscription(text="ご視聴ありがと"))) == []
    assert (
        list(
            notifier.process(
                Transcription(
                    text="ご視聴ありがとうございました",
                    language_code="ja",
                    turn_id="turn_1",
                    turn_revision=0,
                    speech_stopped_at_s=12.5,
                )
            )
        )
        == []
    )

    partial = text_output_queue.get_nowait()
    assert isinstance(partial, PartialTranscriptionEvent)
    completed = _completed(text_output_queue)
    assert completed.transcript == ""
    assert completed.turn_id == "turn_1"
    assert completed.turn_revision == 0
    assert completed.speech_stopped_at_s == 12.5
    # A dropped utterance must also hand the microphone back.
    assert should_listen.is_set()


def test_real_japanese_speech_reaches_the_language_model() -> None:
    text_output_queue: Queue = Queue()
    notifier = _notifier(text_output_queue)

    assert list(notifier.process(Transcription(text="今日は疲れた", language_code="ja"))) == []

    assert _completed(text_output_queue).transcript == "今日は疲れた"


def test_english_finals_are_untouched() -> None:
    text_output_queue: Queue = Queue()
    notifier = _notifier(text_output_queue)

    assert list(notifier.process(Transcription(text="Thanks for watching!", language_code="en"))) == []

    assert _completed(text_output_queue).transcript == "Thanks for watching!"


def test_partial_transcriptions_bypass_the_filter() -> None:
    # Partials are display-only and never reach the LLM, so a hallucinated partial cannot
    # start a turn. Filtering them would only make the on-screen transcript flicker.
    text_output_queue: Queue = Queue()
    notifier = _notifier(text_output_queue)

    assert list(notifier.process(PartialTranscription(text="ご視聴ありがとうございました"))) == []

    partial = text_output_queue.get_nowait()
    assert isinstance(partial, PartialTranscriptionEvent)
    assert partial.delta == "ご視聴ありがとうございました"


def test_the_drop_log_line_carries_the_turn_it_dropped(caplog) -> None:
    text_output_queue: Queue = Queue()
    notifier = _notifier(text_output_queue)

    with caplog.at_level("INFO", logger="speech_to_speech.STT.transcription_notifier"):
        list(
            notifier.process(
                Transcription(
                    text="ご視聴ありがとうございました",
                    language_code="ja",
                    turn_id="turn_7",
                    turn_revision=2,
                )
            )
        )

    assert "turn=turn_7 rev=2" in caplog.text
