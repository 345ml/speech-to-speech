from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class LocalAudioArguments:
    local_audio_tool_module: Optional[str] = field(
        default=None,
        metadata={
            "help": "Importable module defining TOOLS and async execute_tool(name, arguments).",
            "aliases": ["--tool-module"],
        },
    )
    local_audio_input_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice input device index used by the local command."},
    )
    local_audio_output_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice output device index used by the local command."},
    )
    local_audio_chunk_size: int = field(
        default=1024,
        metadata={"help": "Microphone and speaker callback block size in samples. Default is 1024."},
    )
    local_audio_block_mic_during_playback: bool = field(
        default=False,
        metadata={
            "help": "Pause local microphone capture while audio is playing. Disabled by default so barge-in works."
        },
    )
    local_audio_echo_cancellation: Literal["off", "os"] = field(
        default="off",
        metadata={
            "help": (
                "Keep the speaker out of the microphone. 'os' routes both directions through the "
                "macOS voice-processing unit, which cancels the assistant's own voice while leaving "
                "barge-in working; it needs the macos-aec extra and falls back to 'off' elsewhere."
            ),
            "aliases": ["--echo-cancellation"],
        },
    )
    local_audio_thinking_sound: bool = field(
        default=True,
        metadata={
            "help": (
                "Play a quiet looping cue between the end of your turn and the first response audio, "
                "so the generation gap does not sound like a hang. Disable with --thinking-sound false."
            ),
            "aliases": ["--thinking-sound"],
        },
    )
    local_audio_thinking_sound_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Audio file looped as the processing cue instead of the built-in one.",
            "aliases": ["--thinking-sound-file"],
        },
    )
    local_audio_thinking_sound_gain: float = field(
        default=0.12,
        metadata={
            "help": "Processing cue level. Keep it low: the speaker feeds back into the microphone.",
            "aliases": ["--thinking-sound-gain"],
        },
    )
    local_audio_thinking_sound_delay: float = field(
        default=0.3,
        metadata={
            "help": "Seconds to wait before the processing cue starts, so fast turns stay silent.",
            "aliases": ["--thinking-sound-delay"],
        },
    )
    local_audio_print_json: bool = field(
        default=False,
        metadata={"help": "Print raw Realtime events received by the packaged local audio client."},
    )
    local_audio_text_input: bool = field(
        default=False,
        metadata={
            "help": (
                "Type turns alongside speaking. Keeps an input line pinned to the bottom of the "
                "terminal; requires a terminal and prompt_toolkit."
            ),
            "aliases": ["--text-input"],
        },
    )
