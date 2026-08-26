"""Voice-channel system prompt: lead + session prompt + tail (strongest constraints last).

The tail is assembled rather than fixed, because two unrelated things sat in one block.

How long a reply should be is a product decision and every persona makes it
differently, so the tail may not assert one. Placed last, a length rule does not
suggest -- it wins, over whatever the session prompt printed above it asked for. What
belongs last is which text decides, not the decision.

Tool choreography is the bulk of the tail by word count -- 146 of 197 -- and with no
tools configured every line of it teaches a capability that does not exist, spending
that budget on diluting the session prompt. It is sent only to a session that has tools.
"""

VOICE_SYSTEM_PROMPT_LEAD = """\
You are in a spoken conversation. The user speaks and hears you.
The session prompt defines persona, facts, goals, and tool descriptions. These channel rules only control spoken output and tool-use behavior.
"""

# True of a spoken channel whoever is speaking and whatever is configured.
VOICE_SYSTEM_PROMPT_SHAPE = """\
## Voice Rules
- The session prompt sets reply length. Absent a rule there, use two to four spoken sentences.
- Speak naturally. No markdown, bullets, headings, visual formatting, or action/emote text like *laughs*.
- Treat transcripts as noisy. Correct likely mishearings only if asked or meaning depends on it.\
"""

# Sent only to a session that has tools to call.
VOICE_SYSTEM_PROMPT_TOOLS = """\
- Speech is the default. Use tools when they help fulfill the request or fit the moment.
- Never mention tools or their function names in spoken output.
- For information tools, act immediately rather than merely offering. You may give one brief acknowledgement before the first call. After tool results, make further calls without speaking. Once you have enough results, give one final answer; do not narrate individual calls.
- For expression/background tools, speak first. If asked to show an expression, use a short pattern like "Sure, here's my best <emotion>." Otherwise use a fitting empathetic sentence.
- After completed expression/background/physical-action tools, do not add a second spoken comment unless the result has user-facing information.
- Use motion, dance, emotion, and similar tools sparingly when they add empathy, celebration, playfulness, or a requested physical action.
- If unsure whether a tool is needed, just speak.\
"""

# Skeleton for the assembled system message (placeholders filled in build_voice_system_prompt).
_VOICE_SYSTEM_PROMPT_FULL = """\
{lead}

Session Prompt:
{session_prompt}{optional_tools}

{tail}
"""


def build_voice_rules(*, has_tools: bool = True) -> str:
    """Assemble the voice rules from the parts that apply to this session.

    ``has_tools`` defaults to True so a caller that says nothing still gets the whole
    tail, which is what every caller got before the split.
    """
    if not has_tools:
        return VOICE_SYSTEM_PROMPT_SHAPE
    return f"{VOICE_SYSTEM_PROMPT_SHAPE}\n{VOICE_SYSTEM_PROMPT_TOOLS}"


def build_voice_system_prompt(session_prompt: str, *, tool_section: str = "", has_tools: bool = True) -> str:
    """Context → session prompt → optional tool block → strongest voice rules last.

    ``tool_section`` is the local backend's tool-protocol block; supplying one implies
    the session has tools. Backends with native tool calling send no block and say so
    with ``has_tools`` instead.
    """
    tools = tool_section.strip()
    optional_tools = f"\n\n{tools}" if tools else ""
    return _VOICE_SYSTEM_PROMPT_FULL.format(
        lead=VOICE_SYSTEM_PROMPT_LEAD.rstrip(),
        session_prompt=session_prompt.strip(),
        optional_tools=optional_tools,
        tail=build_voice_rules(has_tools=has_tools or bool(tools)),
    )


# Full voice instructions without a separate session block (legacy / rare direct use).
# The tail here is the one a tool session gets: this constant predates the split and
# has always meant "everything".
VOICE_SYSTEM_PROMPT = "{lead}\n\n{tail}".format(
    lead=VOICE_SYSTEM_PROMPT_LEAD.rstrip(),
    tail=build_voice_rules(),
)
