"""言語コードによる sentence tokenizer の選択。

speech-to-speech は全言語共通で nltk の sent_tokenize を通す。punkt は「。」で
切らないので、日本語だけ差し替える。英語の挙動は1バイトも変えない。
"""

from nltk import sent_tokenize

from speech_to_speech.LLM.japanese_segmenter import JapaneseClauseTokenizer
from speech_to_speech.LLM.utils import make_sentence_tokenizer


def test_english_still_gets_the_nltk_tokenizer() -> None:
    assert make_sentence_tokenizer("en") is sent_tokenize


def test_unknown_language_falls_back_to_nltk() -> None:
    assert make_sentence_tokenizer(None) is sent_tokenize
    assert make_sentence_tokenizer("fr") is sent_tokenize


def test_japanese_gets_the_clause_tokenizer() -> None:
    assert isinstance(make_sentence_tokenizer("ja"), JapaneseClauseTokenizer)


def test_japanese_variants_are_recognised() -> None:
    for code in ("ja", "JA", "ja-JP", "jpn"):
        assert isinstance(make_sentence_tokenizer(code), JapaneseClauseTokenizer), code


def test_each_call_returns_a_fresh_japanese_tokenizer() -> None:
    # ターンをまたいで状態が残ると、2ターン目の冒頭句が切り出されなくなる。
    first = make_sentence_tokenizer("ja")
    first("うん、そっか")
    second = make_sentence_tokenizer("ja")
    assert second("うん、そっか") == ["うん、", "そっか"]


def test_english_tokenizer_behaviour_is_unchanged() -> None:
    tokenizer = make_sentence_tokenizer("en")
    assert tokenizer("Hello there. How are you?") == sent_tokenize("Hello there. How are you?")
