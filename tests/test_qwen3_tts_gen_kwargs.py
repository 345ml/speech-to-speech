"""Qwen3-TTS のサンプリングを CLI から止められること。

mlx-audio は temperature=0.9 / top_k=50 でサンプリングするので、同じ文でも
声が毎回変わる。同居人として同一人物性が成立しないので、greedy に落とせる
必要がある(qwen3_tts.py: `if temperature <= 0: return mx.argmax(...)`)。

配線コードは無い。backend_registry の `gen_` 接頭辞規約に乗せているだけなので、
このテストはその規約が Qwen3-TTS でも成立していることを固定する。
"""

from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.backend_registry import normalize_dataclass_config


def test_defaults_match_mlx_audio_so_behaviour_is_unchanged() -> None:
    config = normalize_dataclass_config(Qwen3TTSHandlerArguments(), "qwen3_tts")
    assert config["gen_kwargs"]["temperature"] == 0.9
    assert config["gen_kwargs"]["top_k"] == 50


def test_temperature_reaches_gen_kwargs_not_the_constructor() -> None:
    args = Qwen3TTSHandlerArguments(qwen3_tts_gen_temperature=0.0)
    config = normalize_dataclass_config(args, "qwen3_tts")
    assert config["gen_kwargs"]["temperature"] == 0.0
    # The handler takes gen_kwargs, not a temperature kwarg. A stray top-level key
    # would be a TypeError at construction time.
    assert "gen_temperature" not in config
    assert "temperature" not in config


def test_top_k_reaches_gen_kwargs() -> None:
    args = Qwen3TTSHandlerArguments(qwen3_tts_gen_top_k=1)
    config = normalize_dataclass_config(args, "qwen3_tts")
    assert config["gen_kwargs"]["top_k"] == 1
    assert "top_k" not in config


def test_existing_fields_are_untouched() -> None:
    args = Qwen3TTSHandlerArguments(qwen3_tts_model_name="mlx-community/Qwen3-TTS-12Hz-1.7B-Base")
    config = normalize_dataclass_config(args, "qwen3_tts")
    assert config["model_name"] == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base"
