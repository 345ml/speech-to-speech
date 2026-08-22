"""Choosing the sentence tokenizer by language code.

speech-to-speech ran every language through NLTK's sent_tokenize. punkt does not
split on 。, so Japanese -- and only Japanese -- is swapped for a clause segmenter.
Every other language's behaviour stays byte-for-byte identical, which is why these
tests assert object identity rather than equivalent output.
"""

from nltk import sent_tokenize

from speech_to_speech.LLM.japanese_segmenter import JapaneseClauseTokenizer
from speech_to_speech.LLM.utils import (
    is_japanese_language,
    make_sentence_tokenizer,
    sentence_join_separator,
)


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
    # State surviving across turns would stop the second turn's opening interjection
    # from being cut, because the segmenter would think it had already been released.
    first = make_sentence_tokenizer("ja")
    first("うん、そっか")
    second = make_sentence_tokenizer("ja")
    assert second("うん、そっか") == ["うん、", "そっか"]


def test_english_tokenizer_behaviour_is_unchanged() -> None:
    tokenizer = make_sentence_tokenizer("en")
    assert tokenizer("Hello there. How are you?") == sent_tokenize("Hello there. How are you?")


def test_japanese_clauses_get_an_empty_separator() -> None:
    # Japanese does not put a space between clauses, and the TTS reads one out.
    assert sentence_join_separator("ja") == ""
    assert sentence_join_separator("ja-JP") == ""


def test_every_other_language_keeps_the_ascii_space() -> None:
    assert sentence_join_separator("en") == " "
    assert sentence_join_separator(None) == " "


def test_a_padded_language_code_is_still_japanese() -> None:
    # A code arriving as " ja" fell through to the non-Japanese path, silently taking
    # a Japanese turn back to punkt and to space-joined clauses.
    assert is_japanese_language(" ja")
    assert isinstance(make_sentence_tokenizer(" ja"), JapaneseClauseTokenizer)
    assert sentence_join_separator(" ja\n") == ""
