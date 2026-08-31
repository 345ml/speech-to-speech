#!/usr/bin/env bash
# Persistent llama.cpp server for the `chat-completions` LLM backend.
#
# The `chat-completions` backend is an HTTP client with no model of its own; it needs
# an OpenAI-compatible server on the other end. This script starts one:
#
#   ./scripts/serve_llm.sh &
#   speech-to-speech local \
#     --llm_backend chat-completions \
#     --responses_api_base_url http://127.0.0.1:8080/v1 \
#     --model_name mmnga-o/llm-jp-4-8b-instruct-gguf:Q4_K_M
#
# Wait for the server to report it is listening before starting the pipeline. The
# ChatCompletions handler warms up during setup, so a pipeline started too early dies
# with a connection error instead of retrying.
#
# Model load and Metal shader compilation (~10s on first use) happen once here, so
# neither cost lands inside a turn, and the server outlives pipeline restarts.
#
# WHY THIS MODEL. llm-jp-4-8b-instruct is the instruct (SFT-only) member of the llm-jp-4
# family, trained from scratch on 11.7T tokens by NII's LLM research centre, Apache 2.0.
# The other members -- llm-jp-4-8b-thinking, -32b-a3b-thinking, -33b-thinking -- all
# reason before answering, which puts hundreds of tokens ahead of the first speakable
# character. This one does not: it is the only llm-jp release that is both non-thinking
# and small enough to sit beside the MLX STT and TTS models in 16GB of unified memory.
#
# It replaced Qwen3-4B-Instruct-2507 on the strength of how it talks, not a benchmark.
# No MT-Bench-ja figure is published for the instruct variant; the 0.706 on the Swallow
# leaderboard belongs to llm-jp-4-8b-thinking and was earned with the thinking tokens
# this pipeline cannot afford. The comparison that decided it was a greedy A/B against
# the 4B on prompts/companion_ja.txt.
#
# What it costs. Q4_K_M is 5.30GB and ~21 tok/s here, against 2.3GB and ~34 for the 4B
# it replaced. Prefill is ~210 tok/s. Q5_K_M (6.15GB) is the last quant with any room
# left; Q6_K (7.05GB) and Q8_0 (9.13GB) do not fit beside the MLX models on a 16GB Mac,
# where the Metal working set defaults to about 10.6GB. Q8_0 will load on its own, which
# is enough to A/B the text against Q4_K_M without the pipeline.
#
# Its tokenizer is the reason 4096 of context still goes as far as it did at 4B. The
# few-shot block in prompts/companion_ja.txt measures 263 tokens here against the
# 350-450 it cost under Qwen's vocabulary, and the whole persona file is 558.
#
# Two quirks, both measured, neither fatal:
#   - Every reply arrives with a leading half-width space. The Japanese segmenter keeps
#     it on the first clause, which is the one that reaches the TTS first. It is audible
#     as nothing so far, and Reply-shape logging strips it, so it is recorded rather
#     than fixed. If a leading pause ever shows up in the audio, this is where it is.
#   - Longer replies come back as Markdown hard breaks ("...。  \n"). remove_markdown
#     leaves them and remove_unspeechable keeps whitespace, but JapaneseClauseTokenizer
#     drops them: the clauses it emits carry no newline and no trailing space.
#
# Its chat template is documented as OpenAI Harmony-compatible, which raised the worry
# that channel markers (<|channel|>final) would be spoken -- SPEECHABLE_PATTERN does not
# strip angle brackets. Probed: content comes back clean. The template also ignores
# chat_template_kwargs.enable_thinking, which the pipeline sends unconditionally to a
# non-OpenAI base URL; a request with and without it returns byte-identical text, so
# --responses_api_disable_thinking needs no change.
#
# Weights come from the shared HF cache (~/.cache/huggingface/hub) via -hf, which is
# also where the MLX models live. Override without editing this file:
#   LLM_HF=<repo>:<quant>          ./scripts/serve_llm.sh   # another HF GGUF
#   LLM_MODEL=/path/to/model.gguf  ./scripts/serve_llm.sh   # a loose local file
#   LLM_PORT=8081                  ./scripts/serve_llm.sh
#
# Nothing on the pipeline side changes when you swap models. --model_name is only a
# label: llama-server serves the model it loaded and echoes back its own name, and the
# pipeline never logs the value, so the two need not match.
#
# ALTERNATIVES, all already in that cache and all env-only:
#   LLM_HF=unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
#       The previous default. 2.3GB and ~34 tok/s -- the one to go back to when the
#       machine is under memory pressure or TTFA matters more than the wording.
#   LLAMA_ARG_MMPROJ_AUTO=0 LLM_HF=unsloth/Qwen3.5-9B-GGUF:Q4_K_M
#       5.3GB and ~17 tok/s. Two catches. It is a hybrid that spends the reply inside
#       <think> unless chat_template_kwargs.enable_thinking=false reaches it -- a
#       120-token probe came back with content empty -- which is what the pipeline's
#       responses_api_disable_thinking and --jinja below are for together. And its
#       linear-attention layers make --cache-reuse inert ("cache_reuse is not supported
#       by this context" at startup), so every turn that drops an exchange re-prefills
#       from the system prompt. MMPROJ_AUTO=0 because the unsloth repo ships a vision
#       projector that -hf would otherwise download and load for nothing.
#   LLM_HF=mmnga/ELYZA-Shortcut-1.0-Qwen-7B-gguf:Q4_K_M
#   LLM_HF=elyza/Llama-3-ELYZA-JP-8B-GGUF:Q4_K_M
#       Japanese continued-pretraining of Qwen2.5-7B and Llama-3-8B. Both non-thinking.
#
# llm-jp-4 logs no cache_reuse warning, so the tuning below is live for the default.
#
# FLAGS
# --parallel 1   one slot holds the whole KV prefix, so --cache-reuse actually hits.
# --cache-reuse  32 rather than the 256 this defaults to elsewhere. The value is the
#                SMALLEST chunk llama-server will try to recover by shifting KV after
#                the prefix diverges; when the history drops an exchange the tail to
#                recover is only a couple hundred tokens, so a larger value recovers
#                nothing and every such turn re-prefills from the system prompt. The
#                figure comes from the phase0 latency harness (its Finding 8).
# --ctx-size     4096, not the model's native max: a larger context only inflates the
#                KV cache, and unified memory is the binding constraint here.
# no --mlock     wiring the weights would squeeze the MLX STT/TTS models sharing the
#                same unified memory. Add it back only on a machine with room to spare.
# no --metrics   add it if you want the Prometheus endpoint.
set -euo pipefail

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server not found in PATH. Install llama.cpp (e.g. brew install llama.cpp)." >&2
  exit 1
fi

# QWEN3.5-9B: uncomment both lines to switch (see ALTERNATIVES above). An LLM_HF or
# LLAMA_ARG_MMPROJ_AUTO already set in the environment still wins.
# : "${LLM_HF:=unsloth/Qwen3.5-9B-GGUF:Q4_K_M}"
# export LLAMA_ARG_MMPROJ_AUTO=0

if [ -n "${LLM_MODEL:-}" ]; then
  model_args=(-m "$LLM_MODEL")
else
  model_args=(-hf "${LLM_HF:-mmnga-o/llm-jp-4-8b-instruct-gguf:Q4_K_M}")
fi

exec llama-server "${model_args[@]}" \
  --host "${LLM_HOST:-127.0.0.1}" --port "${LLM_PORT:-8080}" \
  -ngl 99 --flash-attn on \
  --ctx-size "${LLM_CTX_SIZE:-4096}" --parallel 1 --cache-reuse 32 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja
