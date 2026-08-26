# LLM Summary

## Available LLM backends (`--llm_backend`)

Runtime-supported values in `s2s_pipeline.py`:

- `transformers` → `language_model.py` (Transformers backend)
- `mlx-lm` → `language_model.py` (MLX backend)
- `responses-api` → `responses_api_language_model.py`
- `chat-completions` → `chat_completions_language_model.py`

## Usage

### 1) Transformers (`--llm_backend transformers`)

- Handler: `LanguageModelHandler`
- Typical use: local GPU/CPU inference using Hugging Face Transformers
- Backend-specific args prefix: `--llm_*`
- Shared args (from base): `--model_name`, `--chat_size`, `--init_chat_prompt`, `--enable_lang_prompt`

```bash
speech-to-speech serve \
  --llm_backend transformers \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --llm_device cuda \
  --llm_torch_dtype float16 \
  --llm_gen_max_new_tokens 128
```

Common options:
- `--llm_gen_min_new_tokens`
- `--llm_gen_temperature`
- `--llm_gen_do_sample`
- `--chat_size`
- `--init_chat_prompt`

### 2) MLX-LM (`--llm_backend mlx-lm`)

- Handler: `LanguageModelHandler`
- Typical use: Apple Silicon local inference
- Backend-specific args prefix: same as Transformers (`--llm_*`)

```bash
speech-to-speech serve \
  --llm_backend mlx-lm \
  --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16 \
  --llm_device mps \
  --llm_gen_max_new_tokens 128
```

Common options:
- `--llm_gen_temperature`
- `--llm_gen_do_sample`
- `--chat_size`
- `--init_chat_prompt`

### 3) OpenAI-compatible API (`--llm_backend responses-api`)

- Handler: `ResponsesApiModelHandler`
- Typical use: remote model serving via OpenAI-compatible endpoints
- Backend-specific args prefix: `--responses_api_*`
- Shared args (from base): `--model_name`, `--chat_size`, `--init_chat_prompt`, `--enable_lang_prompt`

```bash
speech-to-speech serve \
  --llm_backend responses-api \
  --model_name gpt-5.6-terra \
  --responses_api_api_key YOUR_API_KEY \
  --responses_api_base_url https://api.example.com/v1 \
  --responses_api_stream true
```

Common options:
- `--chat_size`
- `--init_chat_prompt`
- `--user_role`

### 4) OpenAI-compatible Chat Completions (`--llm_backend chat-completions`)

- Handler: `ChatCompletionsApiModelHandler`
- Typical use: a local llama.cpp / vLLM server, or any provider that speaks Chat
  Completions rather than the Responses API
- Backend-specific args prefix: `--responses_api_*` — the connection flags are shared
  with the Responses backend so one set of env vars drives both

```bash
speech-to-speech serve \
  --llm_backend chat-completions \
  --model_name "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M" \
  --responses_api_base_url http://127.0.0.1:8080/v1 \
  --responses_api_gen_temperature 0.9
```

Common options:
- `--chat_size`
- `--init_chat_prompt`
- `--responses_api_default_language` — the language to assume for a turn that arrives
  without one. Only a transcription labels a turn's language; the session carries it
  forward to typed turns and tool follow-ups, but a session that is only ever typed to
  has nothing to inherit for its first turn. Which language a turn is in selects its
  sentence tokenizer, and the wrong one does not split Japanese at 。 — so the whole
  reply reaches the TTS as one sentence and nothing is spoken until generation ends.

Sampling (Chat Completions only). Every one is unset by default, which omits the key
from the request and leaves the server's own value in force:
- `--responses_api_gen_temperature`
- `--responses_api_gen_top_p`
- `--responses_api_gen_frequency_penalty`
- `--responses_api_gen_presence_penalty`
- `--responses_api_gen_max_tokens`
- `--responses_api_gen_seed`

These are sent with every generation request, but not with warmup or with history
compaction, which run outside the per-turn kwargs. They are declared on this backend
rather than shared with `responses-api` because that protocol spells `max_tokens` as
`max_output_tokens` and has no penalty parameters at all.

On the audio-input path (`--stt none`), an explicitly configured
`--responses_api_gen_temperature` takes precedence over
`--responses_api_audio_temperature`'s default, and a client's per-response
`max_output_tokens` takes precedence over both.

## LLM Behavior

When STT is set to language auto-detection (`--language auto`), LLM handlers can receive `(text, language_code)` and prepend a language control instruction like:

- `Please reply to my message in <language>.`

This helps the assistant respond in the detected language. The behavior is opt-in via `--enable_lang_prompt` (shared across all backends); it defaults to `False`.

## Setup

### CUDA setup

```bash
speech-to-speech serve \
  --llm_backend transformers \
  --model_name microsoft/Phi-3-mini-4k-instruct
```

### Local Mac setup

```bash
speech-to-speech local \
  --mac-optimal-settings \
  --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16
```

`--mac-optimal-settings` sets `--llm_backend mlx-lm` and defaults the model to `mlx-community/Qwen3-4B-Instruct-2507-bf16` if not overridden. The command independently selects whether to run only the server or compose it with the audio client.

### Realtime (OpenAI-compatible) setup

Run the server, then connect with the packaged audio client:

```bash
# 1. Start the pipeline server
speech-to-speech serve \
  --llm_backend mlx-lm \
  --model_name mlx-community/Qwen3-4B-Instruct-2507-bf16 \
  --host 0.0.0.0 \
  --port 8765

# 2. Connect with the audio client
speech-to-speech talk --url ws://127.0.0.1:8765/v1/realtime
```

Or with `--mac-optimal-settings` on Apple Silicon:

```bash
speech-to-speech serve \
  --mac-optimal-settings \
  --host 0.0.0.0 \
  --port 8765
```

### Remote API setup

```bash
speech-to-speech serve \
  --llm_backend responses-api \
  --model_name gpt-5.6-terra \
  --responses_api_api_key YOUR_API_KEY
```
