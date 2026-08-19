"""Qwen3-TTS のサンプリングを CLI から止められること。

mlx-audio は temperature=0.9 / top_k=50 でサンプリングするので、同じ文でも
声が毎回変わる。同居人として同一人物性が成立しないので、greedy に落とせる
必要がある(qwen3_tts.py: `if temperature <= 0: return mx.argmax(...)`)。

配線コードは無い。backend_registry の `gen_` 接頭辞規約に乗せているだけなので、
このテストはその規約が Qwen3-TTS でも成立していることを固定する。
"""

import dataclasses
import logging
from threading import Event

import speech_to_speech.TTS.qwen3_tts_handler as qwen3_tts_module
from speech_to_speech.arguments_classes.qwen3_tts_arguments import (
    DEFAULT_GEN_TEMPERATURE,
    DEFAULT_GEN_TOP_K,
    Qwen3TTSHandlerArguments,
)
from speech_to_speech.backend_registry import normalize_dataclass_config
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler


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


def _setup_handler_off_darwin(monkeypatch, gen_kwargs):
    """Construct a handler on the faster-qwen3-tts (CUDA/CPU) backend."""
    monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", lambda self, **kwargs: None)
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", lambda self, model_name: None)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(Event(), gen_kwargs=gen_kwargs)
    return handler


def test_the_torch_backend_says_so_when_it_cannot_use_the_sampling_options(monkeypatch, caplog):
    # gen_kwargs is spread into the mlx-audio call only. Setting temperature 0 on CUDA
    # to stabilise the voice used to be accepted in silence and do nothing.
    with caplog.at_level(logging.WARNING, logger="speech_to_speech.TTS.qwen3_tts_handler"):
        handler = _setup_handler_off_darwin(monkeypatch, {"temperature": 0.0, "top_k": 50})

    assert handler.backend == "faster_qwen3_tts"
    assert "does not support temperature" in caplog.text


def test_the_defaults_do_not_warn_on_the_torch_backend(monkeypatch, caplog):
    # Every run populates gen_kwargs from the dataclass defaults, so only a value the
    # operator actually changed may produce a warning.
    with caplog.at_level(logging.WARNING, logger="speech_to_speech.TTS.qwen3_tts_handler"):
        _setup_handler_off_darwin(
            monkeypatch,
            {"temperature": DEFAULT_GEN_TEMPERATURE, "top_k": DEFAULT_GEN_TOP_K},
        )

    assert caplog.text == ""


def test_the_mlx_backend_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", lambda self, model_name: None)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    with caplog.at_level(logging.WARNING, logger="speech_to_speech.TTS.qwen3_tts_handler"):
        handler.setup(Event(), gen_kwargs={"temperature": 0.0, "top_k": 1})

    assert handler.backend == "mlx"
    assert caplog.text == ""
    assert handler.gen_kwargs == {"temperature": 0.0, "top_k": 1}


def test_the_flags_are_documented_as_mlx_only() -> None:
    fields = {field.name: field for field in dataclasses.fields(Qwen3TTSHandlerArguments)}
    for name in ("qwen3_tts_gen_temperature", "qwen3_tts_gen_top_k"):
        assert "mlx backend only" in fields[name].metadata["help"], name
