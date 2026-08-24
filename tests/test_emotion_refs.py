import json

import pytest

from speech_to_speech.TTS.emotion_refs import NO_EMOTION_REFS, EmotionRefTable


def _manifest(tmp_path, entries, *, clips=("neutral.wav", "joy.wav")):
    for clip in clips:
        (tmp_path / clip).write_bytes(b"RIFF")
    path = tmp_path / "refs.json"
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


def _valid(tmp_path):
    return _manifest(
        tmp_path,
        {
            "平": {"audio": "neutral.wav", "text": "お疲れー。"},
            "喜": {"audio": "joy.wav", "text": "うわー、ふわふわじゃない。"},
        },
    )


def test_load_resolves_audio_relative_to_the_manifest(tmp_path):
    table = EmotionRefTable.load(_valid(tmp_path))

    assert table.resolve("喜").audio == str((tmp_path / "joy.wav").resolve())
    assert table.resolve("喜").text == "うわー、ふわふわじゃない。"


def test_unlisted_slot_falls_back_to_the_default_slot(tmp_path):
    table = EmotionRefTable.load(_valid(tmp_path))

    # 怒 is a valid slot the manifest simply does not define.
    assert table.resolve("怒").audio.endswith("neutral.wav")
    assert table.resolve(None).audio.endswith("neutral.wav")


def test_empty_table_resolves_to_nothing_so_the_caller_keeps_its_own_reference():
    assert NO_EMOTION_REFS.resolve("喜") is None
    assert not NO_EMOTION_REFS
    assert NO_EMOTION_REFS.references() == []


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        EmotionRefTable.load(tmp_path / "absent.json")


def test_manifest_without_the_default_slot_raises(tmp_path):
    path = _manifest(tmp_path, {"喜": {"audio": "joy.wav", "text": "やった。"}})

    with pytest.raises(ValueError, match="must define the '平' slot"):
        EmotionRefTable.load(path)


def test_unknown_slot_name_raises(tmp_path):
    path = _manifest(
        tmp_path,
        {
            "平": {"audio": "neutral.wav", "text": "お疲れー。"},
            "嬉": {"audio": "joy.wav", "text": "やった。"},
        },
    )

    with pytest.raises(ValueError, match="Unknown voice slot"):
        EmotionRefTable.load(path)


def test_missing_transcription_raises(tmp_path):
    # A blank ref_text leaks the reference's own words into the start of the output,
    # so it must fail loudly at startup rather than degrade the voice silently.
    path = _manifest(tmp_path, {"平": {"audio": "neutral.wav", "text": "   "}})

    with pytest.raises(ValueError, match="non-empty 'text'"):
        EmotionRefTable.load(path)


def test_missing_audio_file_raises(tmp_path):
    path = _manifest(tmp_path, {"平": {"audio": "absent.wav", "text": "お疲れー。"}})

    with pytest.raises(FileNotFoundError, match="Reference audio for voice slot"):
        EmotionRefTable.load(path)


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "refs.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        EmotionRefTable.load(path)
