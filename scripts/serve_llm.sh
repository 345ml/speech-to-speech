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
#     --model_name unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
#
# Wait for the server to report it is listening before starting the pipeline. The
# ChatCompletions handler warms up during setup, so a pipeline started too early dies
# with a connection error instead of retrying.
#
# Model load and Metal shader compilation (~10s on first use) happen once here, so
# neither cost lands inside a turn, and the server outlives pipeline restarts.
#
# WHY THIS MODEL. Qwen3-4B-Instruct-2507 is non-thinking by construction: unlike the
# hybrid Qwen3 models it cannot emit a <think> block, which would put hundreds of
# tokens ahead of the first speakable character. Q4_K_M keeps the weights near 2.3GB,
# which is what matters on a 16GB Mac where the MLX STT and TTS models are resident in
# the same unified memory.
#
# Weights come from the shared HF cache (~/.cache/huggingface/hub) via -hf, which is
# also where the MLX models live. Override without editing this file:
#   LLM_HF=<repo>:<quant>          ./scripts/serve_llm.sh   # another HF GGUF
#   LLM_MODEL=/path/to/model.gguf  ./scripts/serve_llm.sh   # a loose local file
#   LLM_PORT=8081                  ./scripts/serve_llm.sh
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

if [ -n "${LLM_MODEL:-}" ]; then
  model_args=(-m "$LLM_MODEL")
else
  model_args=(-hf "${LLM_HF:-unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M}")
fi

exec llama-server "${model_args[@]}" \
  --host "${LLM_HOST:-127.0.0.1}" --port "${LLM_PORT:-8080}" \
  -ngl 99 --flash-attn on \
  --ctx-size "${LLM_CTX_SIZE:-4096}" --parallel 1 --cache-reuse 32 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja
