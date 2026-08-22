"""Tests for the Japanese clause segmenter.

NLTK's punkt does not split on 。, so a Japanese reply reaches the TTS as a single
sentence and nothing is spoken until generation has finished. This segmenter fills
that hole.

The core asymmetry: only the first clause of a turn is cut aggressively. It is the
only one on the time-to-first-audio critical path; cutting later clauses at 、 buys
no latency and costs prosody.
"""

from speech_to_speech.LLM.japanese_segmenter import (
    JapaneseClauseTokenizer,
    JapaneseSegmenterConfig,
)
from speech_to_speech.LLM.utils import MARKDOWN_SENTINEL, sent_tokenize_preserving_markdown_code


def test_first_clause_is_cut_at_the_opening_comma() -> None:
    # 「うん、」 is two characters plus a comma. Releasing it at once IS the TTFA win.
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("うん、そっか") == ["うん、", "そっか"]


def test_later_clauses_are_not_cut_at_a_short_comma() -> None:
    # After the first clause has been released, be conservative: a short 、 is not a cut.
    tokenizer = JapaneseClauseTokenizer()
    tokenizer("うん、")
    assert tokenizer("そっか、大変だったね") == ["そっか、大変だったね"]


def test_full_stop_always_splits() -> None:
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("今日は疲れた。ゆっくりしなよ") == ["今日は疲れた。", "ゆっくりしなよ"]


def test_full_stop_splits_after_the_first_clause_too() -> None:
    tokenizer = JapaneseClauseTokenizer()
    tokenizer("うん、")
    assert tokenizer("今日は疲れた。ゆっくりしなよ") == ["今日は疲れた。", "ゆっくりしなよ"]


def test_decimal_point_is_not_a_sentence_end() -> None:
    # The phase 0 _protected did not guard this: it inspected the punctuation character
    # itself, which decides nothing, instead of its neighbours.
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("3.14です。あとで") == ["3.14です。", "あとで"]


def test_closing_bracket_is_absorbed_into_the_clause() -> None:
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("そうだね。」うん") == ["そうだね。」", "うん"]


def test_text_without_a_boundary_stays_whole() -> None:
    # No boundary means a single element. The caller only flushes when len(parts) > 1.
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("今日はいい天気") == ["今日はいい天気"]


def test_forced_flush_at_max_chars_when_there_is_no_punctuation() -> None:
    # Keeps audio from stalling behind generation in a reply with no punctuation at all.
    tokenizer = JapaneseClauseTokenizer()
    tokenizer("うん、")
    assert tokenizer("あ" * 70) == ["あ" * 60, "あ" * 10]


def test_forced_flush_prefers_the_last_comma_so_it_never_cuts_mid_phrase() -> None:
    tokenizer = JapaneseClauseTokenizer()
    tokenizer("うん、")
    text = "あ" * 10 + "、" + "い" * 60
    assert tokenizer(text) == ["あ" * 10 + "、", "い" * 60]


def test_punctuation_only_fragment_is_not_emitted_as_a_clause() -> None:
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("。。。ねえ") == ["。。。ねえ"]


def test_config_is_overridable() -> None:
    tokenizer = JapaneseClauseTokenizer(JapaneseSegmenterConfig(first_min_chars=6))
    # 「うん、」 is three characters, so first_min_chars=6 does not cut it.
    assert tokenizer("うん、そっか") == ["うん、そっか"]


def test_forced_flush_never_cuts_inside_a_markdown_sentinel() -> None:
    # `sent_tokenize_preserving_markdown_code` hides each complete code span behind a
    # `\x00markdown-code-N\x00` sentinel and restores it by exact string match. punkt
    # never split one; the forced flush cuts at an arbitrary index and would, leaving a
    # raw NUL byte and the literal text `markdown-code-` to be read out by the TTS while
    # the code span itself disappeared.
    tokenizer = JapaneseClauseTokenizer()
    tokenizer("うん、")
    text = "あ" * 45 + "`inline_code_here`" + "い" * 30

    parts = sent_tokenize_preserving_markdown_code(text, tokenizer)

    assert all(MARKDOWN_SENTINEL not in part for part in parts)
    assert "".join(parts) == text


def test_forced_flush_backs_off_in_front_of_the_sentinel() -> None:
    # The unit-level statement of the same rule: the cut lands before the sentinel that
    # is still open, never between its two halves.
    tokenizer = JapaneseClauseTokenizer()
    tokenizer("うん、")
    protected = f"{MARKDOWN_SENTINEL}markdown-code-0{MARKDOWN_SENTINEL}"
    text = "あ" * 55 + protected + "い" * 20

    parts = tokenizer(text)

    assert parts == ["あ" * 55, protected + "い" * 20]


def test_an_unclosed_sentinel_at_index_zero_is_not_cut_to_nothing() -> None:
    # Backing off in front of a sentinel that opens the buffer gives index 0, which is
    # not a legal cut: it would emit an empty clause and loop forever on the same buffer.
    # A complete sentinel is far shorter than max_chars, so this needs a half of one --
    # what a stray control character in the model output would look like.
    tokenizer = JapaneseClauseTokenizer()
    tokenizer("うん、")
    text = MARKDOWN_SENTINEL + "あ" * 70

    assert tokenizer(text) == [text]


def test_a_newline_after_a_full_stop_is_not_emitted_twice() -> None:
    # \n is a HARD terminator, so it stays with the clause it ends. It was not stripped
    # from the remainder, so the next clause started with the same newline again.
    tokenizer = JapaneseClauseTokenizer()
    buffer = ""
    clauses: list[str] = []
    for delta in ("はい。\n", "そう", "だね。\n", "うん"):
        buffer += delta
        parts = tokenizer(buffer)
        clauses.extend(parts[:-1])
        buffer = parts[-1]
    clauses.append(buffer)

    assert clauses == ["はい。", "そうだね。", "うん"]
