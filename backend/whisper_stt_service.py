"""
OpenAI Whisper API for speech-to-text. Good Japanese support; same API key as chat.
Use STT_PROVIDER=whisper to select (default: google).
"""
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class WhisperSTTService:
    """OpenAI Whisper API (transcriptions). Same interface as GoogleSTTService for drop-in use."""

    def __init__(self):
        self.available = bool(os.getenv("OPENAI_API_KEY", "").strip())
        if not self.available:
            logger.warning("OPENAI_API_KEY not set. Whisper STT disabled.")

    def is_available(self) -> bool:
        return self.available

    def transcribe(
        self,
        audio_path: str,
        language_code: str = "ja-JP",
        model: str = "whisper-1",
        use_v2: bool = True,
        sample_rate_hertz: int = 16000,
    ) -> Dict:
        """
        Transcribe audio using OpenAI Whisper API.
        language_code: ja-JP -> language="ja", en-US -> language="en".
        Other args kept for interface compatibility; Whisper API uses file + language.
        """
        if not self.available:
            raise RuntimeError("Whisper STT not available. Set OPENAI_API_KEY.")
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file does not exist: {audio_path}")
        if os.path.getsize(audio_path) == 0:
            raise ValueError("Audio file is empty")

        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # ISO-639-1: ja-JP -> ja, en-US -> en (improves accuracy and latency per docs)
        lang = "ja" if language_code.startswith("ja") else "en"
        api_model = model if (model and "whisper" in model.lower()) else "whisper-1"

        with open(audio_path, "rb") as f:
            # Supported: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
            transcript = client.audio.transcriptions.create(
                model=api_model,
                file=f,
                language=lang,
                response_format="text",
            )
        # API returns string when response_format="text"
        text = transcript if isinstance(transcript, str) else getattr(transcript, "text", "") or ""
        return {
            "text": text.strip(),
            "language": language_code,
            "alternatives": [],
            "model": api_model,
        }
