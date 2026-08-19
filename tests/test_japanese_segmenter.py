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
