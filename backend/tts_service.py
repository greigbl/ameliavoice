"""
Google Cloud Text-to-Speech service.
"""
import os
import logging
from typing import Optional
import io

logger = logging.getLogger(__name__)


class TTSService:
    """Google Cloud TTS service wrapper."""
    
    def __init__(self):
        self.client = None
        self.available = False
        self._init_client()
    
    def _init_client(self):
        """Initialize Google Cloud TTS client."""
        try:
            from google.cloud import texttospeech
            
            # Check for credentials
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not creds_path or not os.path.exists(creds_path):
                logger.warning(
                    "Google Cloud credentials not found. "
                    "Set GOOGLE_APPLICATION_CREDENTIALS environment variable. "
                    "TTS will be disabled."
                )
                return
            
            self.client = texttospeech.TextToSpeechClient()
            self.available = True
            logger.info("Google Cloud TTS initialized successfully")
        except ImportError:
            logger.warning("google-cloud-texttospeech not installed. TTS disabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize Google TTS: {e}")
    
    def is_available(self) -> bool:
        """Check if TTS service is available."""
        return self.available
    
    def synthesize(
        self,
        text: str,
        language_code: str = "ja-JP",
        voice_name: Optional[str] = None,
        audio_encoding: str = "MP3"
    ) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language_code: Language code (e.g., "ja-JP")
            voice_name: Specific voice name (optional)
            audio_encoding: Audio encoding format (MP3, LINEAR16, etc.)
        
        Returns:
            Audio data as bytes
        """
        if not self.available:
            raise RuntimeError("TTS service not available. Check Google Cloud credentials.")
        
        from google.cloud import texttospeech
        
        # Set up input text
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build voice selection
        if voice_name:
            voice = texttospeech.VoiceSelectionParams(
                name=voice_name,
                language_code=language_code
            )
        else:
            # Use default Japanese voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
        
        # Set audio config
        audio_config = texttospeech.AudioConfig(
            audio_encoding=getattr(texttospeech.AudioEncoding, audio_encoding)
        )
        
        # Perform synthesis
        try:
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            return response.audio_content
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise
    
    def list_voices(self, language_code: str = "ja-JP"):
        """List available voices for a language."""
        if not self.available:
            return []
        
        try:
            voices = self.client.list_voices(language_code=language_code)
            return [
                {
                    "name": voice.name,
                    "ssml_gender": voice.ssml_gender.name,
                    "natural_sample_rate_hertz": voice.natural_sample_rate_hertz
                }
                for voice in voices.voices
            ]
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []
