import pytest

from speech_to_speech.LLM.emotion_tags import (
    AIZUCHI_SLOT,
    DEFAULT_SLOT,
    EmotionTagStripper,
    is_opening_backchannel,
    strip_emotion_tags,
)


def _feed_all(deltas):
    stripper = EmotionTagStripper()
    text = "".join(stripper.feed(delta) for delta in deltas) + stripper.flush()
    return text, stripper


def test_tag_split_across_deltas_is_still_recognised():
    # Providers split tokens wherever they like; "[喜]" routinely arrives in pieces.
    text, stripper = _feed_all(["[", "喜", "]", "うん、", "それいいね。"])

    assert text == "うん、それいいね。"
    assert stripper.emotion == "喜"


def test_tag_arriving_whole_is_recognised():
    text, stripper = _feed_all(["[怒]ふざけないで。"])

    assert text == "ふざけないで。"
    assert stripper.emotion == "怒"


def test_untagged_output_falls_back_to_the_default_slot():
    text, stripper = _feed_all(["うん、そっか。"])

    assert text == "うん、そっか。"
    assert stripper.emotion is None
    assert stripper.slot == DEFAULT_SLOT


def test_strip_emotion_tags_resolves_the_fallback_itself():
    # Returns a slot, not an Optional, so callers never re-spell "or DEFAULT_SLOT".
    assert strip_emotion_tags("うん、そっか。") == ("うん、そっか。", DEFAULT_SLOT)


def test_last_tag_wins_when_the_model_tags_every_sentence():
    text, stripper = _feed_all(["[平]うん。", "[怒]ふざけないで。"])

    assert text == "うん。ふざけないで。"
    assert stripper.emotion == "怒"


@pytest.mark.parametrize(
    "source",
    [
        "配列[0]を見て",  # a real bracketed aside
        "[謎]ほげ",  # bracketed, but not an emotion we know
        "[とても長い]ほげ",  # too long to be a tag
    ],
)
def test_bracketed_text_that_is_not_an_emotion_tag_survives(source):
    text, stripper = _feed_all([source])

    assert text == source
    assert stripper.emotion is None


@pytest.mark.parametrize("source", ["【喜】やった。", "［喜］やった。", "[喜]やった。"])
def test_full_width_brackets_are_recognised_too(source):
    # remove_unspeechable keeps [ ] but DELETES ［］【】, so an unhandled full-width tag
    # loses its brackets and leaves a bare 喜 for the TTS to read aloud.
    text, stripper = _feed_all([source])

    assert text == "やった。"
    assert stripper.emotion == "喜"


def test_unterminated_bracket_is_released_rather_than_swallowed():
    # Without flush() the trailing "[そう" would vanish from the spoken reply.
    text, stripper = _feed_all(["なんで[そう"])

    assert text == "なんで[そう"
    assert stripper.emotion is None


def test_strip_emotion_tags_handles_complete_text():
    assert strip_emotion_tags("[驚]えっ、まじで。") == ("えっ、まじで。", "驚")


def test_aizuchi_is_never_a_tag_the_model_can_pick():
    # The backchannel slot is assigned by position, not by the model naming it.
    text, stripper = _feed_all([f"[{AIZUCHI_SLOT}]うん。"])

    assert text == f"[{AIZUCHI_SLOT}]うん。"
    assert stripper.emotion is None


@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        ("うん、", True),
        ("そっか、", True),
        ("なるほどね、", True),
        # Ends at 。 -- a finished short sentence, not an opener the reply continues past.
        ("ふざけないで。", False),
        ("うん。", False),
        # Long enough to be a real first sentence even though it ends at 、.
        ("今日はほんとうに長い一日だったね、", False),
        ("", False),
    ],
)
def test_opening_backchannel_needs_both_brevity_and_a_soft_terminator(clause, expected):
    assert is_opening_backchannel(clause) is expected
