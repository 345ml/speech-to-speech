from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)


@dataclass
class ChatCompletionsLanguageModelHandlerArguments(ResponsesApiLanguageModelHandlerArguments):
    """Arguments for the ``chat-completions`` LLM backend.

    Inherits the OpenAI-compatible connection fields from the Responses-API
    arguments (``responses_api_base_url`` / ``responses_api_api_key`` /
    ``responses_api_stream`` / ``responses_api_disable_thinking``) so the same
    CLI flags and launcher env vars drive both backends. Chat Completions keeps
    the reasoning-effort default unset so provider-specific disable-thinking
    behavior remains unchanged.
    """

    responses_api_reasoning_effort: Optional[str] = field(
        default=None,
        metadata={
            "help": "Provider-specific reasoning level sent as extra_body={'reasoning_effort': <value>} on the "
            "Chat Completions request. Use to disable reasoning on providers where "
            "chat_template_kwargs.enable_thinking is ignored (e.g. 'none' / 'low'). When unset, falls back to "
            "the disable_thinking behaviour (chat_template_kwargs.enable_thinking=false). Default is None."
        },
    )

    # Sampling. The `gen_` prefix is the backend_registry convention: each of these is
    # collected into the `gen_kwargs` dict and handed to setup() with no wiring code of
    # its own (see normalize_dataclass_config). Every one defaults to None, which drops
    # the key from the request entirely and leaves the server's own value in force --
    # so an unset flag sends the byte-identical request it sent before these existed.
    #
    # They live on the Chat Completions arguments rather than on the Responses parent
    # because the two protocols disagree: Responses spells max_tokens as
    # max_output_tokens and has no penalty parameters at all. Adding them to the parent
    # means teaching the Responses backend that rename and that omission first.
    responses_api_gen_temperature: Optional[float] = field(
        default=None,
        metadata={
            "help": "Sampling temperature (0.0-2.0) sent with every Chat Completions request. Raising it varies "
            "the wording, most visibly the opening interjection, which repeats at low temperature. Unset by "
            "default, leaving the server's own temperature in place."
        },
    )
    responses_api_gen_top_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Nucleus-sampling cutoff (0.0-1.0) sent with every Chat Completions request. Tightens the "
            "candidate set without flattening the distribution the way lowering temperature does. Change this "
            "or temperature, not both. Unset by default."
        },
    )
    responses_api_gen_frequency_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Per-occurrence token penalty (-2.0-2.0). Two caveats for Japanese: llama.cpp applies it over "
            "the last repeat-last-n tokens (64 by default), so it cannot see how the PREVIOUS turn opened; and "
            "it is token-blind, so the tokens it targets first are the particles (は/が/を/に/の/て). Above "
            "roughly 0.2 it degrades grammar before it changes wording. Unset by default."
        },
    )
    responses_api_gen_presence_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Flat one-shot penalty (-2.0-2.0) for tokens already present. Same particle caveat as "
            "frequency_penalty but milder, because it does not compound with repetition. Use it for stutter "
            "WITHIN one reply; cross-turn opener variety comes from the prompt, not from here. Unset by default."
        },
    )
    responses_api_gen_max_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "Hard ceiling on generated tokens per reply. A runaway guard, not a length control: hitting it "
            "cuts the reply mid-clause and the TTS speaks the fragment. For length, write the rule into the "
            "session prompt. Unset by default."
        },
    )
    responses_api_gen_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Sampling seed, for reproducing one turn while comparing prompts. Unset by default."},
    )
