"""日本語クローズ分割器のテスト。

nltk の punkt は「。」で切らないので、日本語の返答は TTS に1文として渡り、
生成が終わるまで発話が始まらない。この分割器はその穴を埋める。

核になる非対称性: ターン最初のクローズだけ積極的に切る。TTFA のクリティカルパスに
乗っているのはそれだけで、以降を「、」で切ってもレイテンシは縮まず韻律を損なう。
"""

from speech_to_speech.LLM.japanese_segmenter import (
    JapaneseClauseTokenizer,
    JapaneseSegmenterConfig,
)
from speech_to_speech.LLM.utils import MARKDOWN_SENTINEL, sent_tokenize_preserving_markdown_code


def test_first_clause_is_cut_at_the_opening_comma() -> None:
    # 「うん、」は2文字+読点。これを即座に出すことが TTFA の短縮そのもの。
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("うん、そっか") == ["うん、", "そっか"]


def test_later_clauses_are_not_cut_at_a_short_comma() -> None:
    # 最初のクローズを出した後は保守的に。短い「、」では切らない。
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
    # phase0 の _protected はここを守れていなかった（句読点自身の文字を見ていた）。
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("3.14です。あとで") == ["3.14です。", "あとで"]


def test_closing_bracket_is_absorbed_into_the_clause() -> None:
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("そうだね。」うん") == ["そうだね。」", "うん"]


def test_text_without_a_boundary_stays_whole() -> None:
    # 境界が無ければ1要素。呼び出し側は len(parts) > 1 のときだけフラッシュする。
    tokenizer = JapaneseClauseTokenizer()
    assert tokenizer("今日はいい天気") == ["今日はいい天気"]


def test_forced_flush_at_max_chars_when_there_is_no_punctuation() -> None:
    # 句読点を一つも打たない返答で音声が生成に張り付かないようにする。
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
    # 「うん、」は3文字なので first_min_chars=6 では切れない。
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
