"""
Shared voice system prompt: language + verbosity template for the LLM only.

These settings affect only the system message sent to the chat model. They do NOT
change TTS, playback, barge-in, or any other voice pipeline behavior. Response
length is controlled by the prompt (verbosity), not by truncation or voice logic.

Configure via env: VOICE_VERBOSITY (brief|normal|detailed), VOICE_PROMPT_TEMPLATE (optional override).
"""
import os

# Language instruction so the model responds in the user's language
LANGUAGE_SYSTEM_MESSAGE = {
    "ja": "The user's interface language is Japanese. You must respond only in Japanese. Use Japanese for all replies, including greetings and goodbyes. Speech recognition can mishear: only end the conversation when the user clearly and unambiguously says goodbye or that they are done (e.g. さようなら、以上です). If in doubt, respond normally and do not end.",
    "en": "The user's interface language is English. You must respond only in English. Use English for all replies, including greetings and goodbyes. Speech recognition can mishear: only end the conversation when the user clearly and unambiguously says goodbye or that they are done (e.g. goodbye, that's all for now, I'm done). If in doubt, respond normally and do not end.",
}

# Verbosity levels: control response length for voice (short = better for TTS / phone)
VERBOSITY_INSTRUCTIONS = {
    "brief": "Keep all responses very brief: 1–2 short sentences maximum. Avoid lists or long explanations.",
    "normal": "Respond concisely. Prefer a few clear sentences; avoid unnecessary detail.",
    "detailed": "You may give longer, detailed responses when helpful. Still prefer clarity over length.",
}

DEFAULT_VOICE_PROMPT_TEMPLATE = (
    "You are a helpful voice assistant. {language_instruction} {verbosity_instruction}"
)


def build_voice_system_message(lang: str, verbosity: str | None = None) -> str:
    """
    Build the system message for voice chat.
    - verbosity: if provided (brief|normal|detailed), use it; else use env VOICE_VERBOSITY (default: normal).
    - VOICE_PROMPT_TEMPLATE: optional full template with {language_instruction} and {verbosity_instruction}
    """
    language_instruction = LANGUAGE_SYSTEM_MESSAGE.get(lang) or LANGUAGE_SYSTEM_MESSAGE["en"]
    v = (verbosity or os.getenv("VOICE_VERBOSITY") or "normal").strip().lower()
    verbosity_instruction = VERBOSITY_INSTRUCTIONS.get(v) or VERBOSITY_INSTRUCTIONS["normal"]
    template = os.getenv("VOICE_PROMPT_TEMPLATE") or DEFAULT_VOICE_PROMPT_TEMPLATE
    return template.format(
        language_instruction=language_instruction,
        verbosity_instruction=verbosity_instruction,
    ).strip()
