#!/usr/bin/env bash
# The companion pipeline as a LAN-reachable Realtime server, for the UE client.
# Same models and flags as run_companion.sh, minus the bundled audio client and
# typed input. Start ./scripts/serve_llm.sh first and wait for "server is listening".
#
# ENV (in addition to run_companion.sh's):
#   HOST   interface to bind        (default 0.0.0.0 -- reachable from the LAN)
#   PORT   Realtime server port     (default 8765)
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -n "${PROMPT_FILE:-}" ] && [ "${PROMPT_FILE#/}" = "$PROMPT_FILE" ]; then
  PROMPT_FILE="$OLDPWD/$PROMPT_FILE"
fi
PROMPT_FILE="${PROMPT_FILE:-prompts/companion_ja.txt}"
if [ ! -f "$PROMPT_FILE" ]; then
  echo "Persona prompt not found: $PROMPT_FILE" >&2
  exit 1
fi

: "${TTS_REF_AUDIO:?set TTS_REF_AUDIO to the reference voice wav (see RUN.md)}"
if [ ! -f "$TTS_REF_AUDIO" ]; then
  echo "Reference audio not found: $TTS_REF_AUDIO" >&2
  exit 1
fi
: "${TTS_REF_TEXT:?set TTS_REF_TEXT to the transcript of \$TTS_REF_AUDIO (see RUN.md)}"

if command -v speech-to-speech >/dev/null 2>&1; then
  s2s=(speech-to-speech)
elif [ -x ./.venv/bin/speech-to-speech ]; then
  s2s=(./.venv/bin/speech-to-speech)
else
  echo "speech-to-speech not found. Run 'uv sync', or activate the environment it is installed in." >&2
  exit 1
fi

args=(
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8765}"
  --stt mlx-audio-whisper
  --mlx_audio_whisper_model_name mlx-community/whisper-large-v3-turbo
  --language ja
  --llm_backend chat-completions
  --responses_api_base_url "http://${LLM_HOST:-127.0.0.1}:${LLM_PORT:-8080}/v1"
  --model_name "${LLM_MODEL_NAME:-mmnga-o/llm-jp-4-8b-instruct-gguf:Q4_K_M}"
  --init_chat_prompt "$(cat "$PROMPT_FILE")"
  --responses_api_default_language ja
  --chat_size "${CHAT_SIZE:-12}"
  --stream_batch_sentences 1
  --tts qwen3
  --qwen3_tts_device mps
  --qwen3_tts_model_name mlx-community/Qwen3-TTS-12Hz-1.7B-Base
  --qwen3_tts_mlx_quantization 8bit
  --qwen3_tts_gen_temperature "${TTS_TEMP:-0.9}"
  --qwen3_tts_ref_audio "$TTS_REF_AUDIO"
  --qwen3_tts_ref_text "$TTS_REF_TEXT"
  --qwen3_tts_language ja
)

if [ -n "${LLM_TEMP:-}" ]; then args+=(--responses_api_gen_temperature "$LLM_TEMP"); fi
if [ -n "${LLM_TOP_P:-}" ]; then args+=(--responses_api_gen_top_p "$LLM_TOP_P"); fi
if [ -n "${LLM_PRESENCE_PENALTY:-}" ]; then args+=(--responses_api_gen_presence_penalty "$LLM_PRESENCE_PENALTY"); fi
if [ -n "${LLM_FREQUENCY_PENALTY:-}" ]; then args+=(--responses_api_gen_frequency_penalty "$LLM_FREQUENCY_PENALTY"); fi
if [ -n "${LLM_MAX_TOKENS:-}" ]; then args+=(--responses_api_gen_max_tokens "$LLM_MAX_TOKENS"); fi
if [ -n "${LLM_SEED:-}" ]; then args+=(--responses_api_gen_seed "$LLM_SEED"); fi

exec "${s2s[@]}" serve "${args[@]}"
