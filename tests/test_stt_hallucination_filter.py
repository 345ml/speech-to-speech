"""日本語 whisper の幻聴を LLM ターンにしないためのフィルタ。

無音に近い入力に対して whisper は YouTube の定型句を吐く。フィルタが無いと
咳払い1つで LLM ターンが丸ごと1回走る(phase0 の実声12ターン中2件)。

英語には適用しない。"Thanks for watching!" は英語話者が普通に言う。
"""

from queue import Queue
from threading import Event

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.STT.hallucinations import is_meaningful_transcription


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
    # STATUS.md C-1. 「ありがとうございました」は普通に言う発話なので、単純に
    # リストへ足すと正常な発話を捨てる。発話長と VAD 信頼度を判定に使えるように
    # なるまで穴のまま残す(L0-5 で配管する)。
    assert is_meaningful_transcription("ありがとうございました", "ja")


def test_punctuation_only_output_is_dropped() -> None:
    assert not is_meaningful_transcription("。。。", "ja")
    assert not is_meaningful_transcription("、", "ja")
    assert not is_meaningful_transcription("   ", "ja")


def test_single_character_output_is_dropped() -> None:
    assert not is_meaningful_transcription("あ", "ja")


def test_two_characters_survive() -> None:
    # 「うん」は相槌として実在する発話。落としてはいけない。
    assert is_meaningful_transcription("うん", "ja")


def test_english_is_not_filtered() -> None:
    assert is_meaningful_transcription("Thanks for watching!", "en")
    assert is_meaningful_transcription("ご視聴ありがとうございました", "en")


def test_missing_language_is_not_filtered() -> None:
    assert is_meaningful_transcription("ご視聴ありがとうございました", None)


class _StubSTTHandler(BaseSTTHandler):
    """Bypasses the speculative-turn gate so the test isolates the new filter."""

    def _is_latest_turn_item(self, item, *, wait_for_pending_reopen=False, wait_for_stability=False):
        return True

    def _is_completed_final_revision(self, item) -> bool:
        return False


def _stub_handler() -> _StubSTTHandler:
    return _StubSTTHandler(Event(), queue_in=Queue(), queue_out=Queue())


def _transcription(text: str, language_code: str | None) -> Transcription:
    return Transcription(text=text, language_code=language_code)


def test_handler_drops_a_hallucinated_transcription() -> None:
    handler = _stub_handler()
    assert not handler.should_emit_output(_transcription("ご視聴ありがとうございました", "ja"))


def test_handler_emits_real_speech() -> None:
    handler = _stub_handler()
    assert handler.should_emit_output(_transcription("今日は疲れた", "ja"))


def test_partial_transcriptions_bypass_the_filter() -> None:
    # PartialTranscription には language_code が無いのでフィルタは効かない。意図通り:
    # 部分文字起こしは表示用で LLM には渡らないため、幻聴がターンを起こすことはない。
    handler = _stub_handler()
    partial = PartialTranscription(text="ご視聴ありがとうございました")
    assert handler.should_emit_output(partial)
