#!/usr/bin/env bash
# The companion pipeline. Start ./scripts/serve_llm.sh first and wait for
# "server is listening": the ChatCompletions handler warms up during setup, so a
# pipeline started too early dies with a connection error instead of retrying.
#
# See RUN.md for the tuning procedure -- which knob to turn for which symptom, and
# what to listen for afterwards. The per-flag reasoning lives next to each flag below.
#
# ENV
#   PROMPT_FILE            persona prompt         (default prompts/companion_ja.txt)
#   TTS_REF_AUDIO          voice to clone         (required; no portable default)
#   TTS_REF_TEXT           its transcript         (must match the audio -- see below)
#   TTS_TEMP               TTS sampling           (default 0.9, mlx-audio's own)
#   TEXT_INPUT=0           microphone only, no typed input
#   ECHO_CANCELLATION=os   keep the speakers out of the microphone (macOS; see RUN.md)
#   CHAT_SIZE              retained user turns    (default 12, see below)
#   LLM_HOST / LLM_PORT    where serve_llm.sh is  (default 127.0.0.1:8080)
#   LLM_MODEL_NAME         label sent to the server, not a model selector
#   LLM_TEMP, LLM_TOP_P, LLM_PRESENCE_PENALTY, LLM_FREQUENCY_PENALTY,
#   LLM_MAX_TOKENS, LLM_SEED
#                          sampling; each unset one is not sent at all, leaving
#                          llama-server's own value in force
set -euo pipefail

cd "$(dirname "$0")/.."

# Resolved before the cd so a relative path still means what the caller meant. The
# default is repo-relative and resolves after it, which is why the cd comes first.
if [ -n "${PROMPT_FILE:-}" ] && [ "${PROMPT_FILE#/}" = "$PROMPT_FILE" ]; then
  PROMPT_FILE="$OLDPWD/$PROMPT_FILE"
fi
PROMPT_FILE="${PROMPT_FILE:-prompts/companion_ja.txt}"
if [ ! -f "$PROMPT_FILE" ]; then
  echo "Persona prompt not found: $PROMPT_FILE" >&2
  exit 1
fi

# The reference audio is the voice. It cannot live in the repo and there is no sane
# default, so fail here rather than several minutes into model loading. TTS_REF_TEXT
# must be its transcript: the handler mixes a mismatched one into the start of the
# output, and the built-in default is an unrelated English paragraph.
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
  --stt mlx-audio-whisper
  --mlx_audio_whisper_model_name mlx-community/whisper-large-v3-turbo
  --language ja
  --llm_backend chat-completions
  --responses_api_base_url "http://${LLM_HOST:-127.0.0.1}:${LLM_PORT:-8080}/v1"
  # Inert against llama-server: it serves whatever it loaded and the pipeline never
  # logs this. Kept matching serve_llm.sh so the two files do not disagree.
  --model_name "${LLM_MODEL_NAME:-mmnga-o/llm-jp-4-8b-instruct-gguf:Q4_K_M}"
  --init_chat_prompt "$(cat "$PROMPT_FILE")"
  # Only a transcription labels a turn's language. The session carries it forward to
  # typed turns and tool follow-ups, but the first turn of a typed-only session has
  # nothing to inherit -- and no language means no Japanese clause splitting.
  --responses_api_default_language ja
  # 12, not the default 30. The few-shot block put roughly 400 tokens into a system
  # message that is prefixed to every request, and serve_llm.sh runs at --ctx-size
  # 4096. Overflowing it is not a graceful degradation: the request is rejected and
  # the pipeline speaks its English failure line into a Japanese conversation.
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

if [ "${TEXT_INPUT:-1}" = "1" ]; then
  args+=(--text-input)
fi

if [ -n "${ECHO_CANCELLATION:-}" ]; then
  args+=(--echo-cancellation "$ECHO_CANCELLATION")
fi

# Only pass what was actually set: an unset sampling flag must stay absent from the
# request rather than become an explicit null.
if [ -n "${LLM_TEMP:-}" ]; then args+=(--responses_api_gen_temperature "$LLM_TEMP"); fi
if [ -n "${LLM_TOP_P:-}" ]; then args+=(--responses_api_gen_top_p "$LLM_TOP_P"); fi
if [ -n "${LLM_PRESENCE_PENALTY:-}" ]; then args+=(--responses_api_gen_presence_penalty "$LLM_PRESENCE_PENALTY"); fi
if [ -n "${LLM_FREQUENCY_PENALTY:-}" ]; then args+=(--responses_api_gen_frequency_penalty "$LLM_FREQUENCY_PENALTY"); fi
if [ -n "${LLM_MAX_TOKENS:-}" ]; then args+=(--responses_api_gen_max_tokens "$LLM_MAX_TOKENS"); fi
if [ -n "${LLM_SEED:-}" ]; then args+=(--responses_api_gen_seed "$LLM_SEED"); fi

exec "${s2s[@]}" local "${args[@]}"
