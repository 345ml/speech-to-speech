"""Tests for the console output ports used by the packaged client."""

from __future__ import annotations

from speech_to_speech.api.openai_realtime.console import StdoutConsole, display_width, fit


def test_display_width_counts_east_asian_characters_as_two_columns() -> None:
    assert display_width("") == 0
    assert display_width("abc") == 3
    assert display_width("あいう") == 6
    assert display_width("USER: こんばんは") == 16


def test_fit_keeps_the_tail_within_the_column_budget() -> None:
    assert fit("あいうえお", 10) == "あいうえお"
    assert fit("あいうえお", 6) == "…えお"
    assert fit("abcdef", 6) == "abcdef"
    assert fit("abcdef", 4) == "…def"


def test_fit_never_exceeds_the_budget_for_any_width() -> None:
    for columns in range(1, 24):
        assert display_width(fit("あ" * 20, columns)) <= columns
        assert display_width(fit("mixed かな text", columns)) <= columns


def test_fit_returns_nothing_when_the_terminal_reports_no_width() -> None:
    """An unbounded string here would wrap and push the prompt off screen."""

    assert fit("あいうえお", 0) == ""
    assert fit("あいうえお", -1) == ""


def test_stdout_console_overwrites_transient_text_in_place(capsys) -> None:
    console = StdoutConsole()
    console.transient("USER: hello")
    console.transient("USER: hi", final=True)

    assert capsys.readouterr().out == "\rUSER: hello\rUSER: hi   \n"


def test_stdout_console_erases_japanese_by_column_not_character(capsys) -> None:
    """len() under-erases CJK by half, leaving the tail of the line on screen."""

    text = "USER: こんばんは"
    console = StdoutConsole()
    console.transient(text)
    console.clear_transient()

    erased = capsys.readouterr().out.count(" ")
    assert erased >= display_width(text)


def test_stdout_console_streams_without_newlines(capsys) -> None:
    console = StdoutConsole()
    console.stream("ASSISTANT: ")
    console.stream("hi")
    assert capsys.readouterr().out == "ASSISTANT: hi"

    console.end_stream()
    assert capsys.readouterr().out == "\n"


def test_stdout_console_clear_is_a_no_op_without_transient_text(capsys) -> None:
    console = StdoutConsole()
    console.clear_transient()
    assert capsys.readouterr().out == ""
