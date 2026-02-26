"""
Google Cloud Speech-to-Text service with Chirp 3 support (batch and streaming).
"""
import os
import logging
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class GoogleSTTService:
    """Google Cloud Speech-to-Text service wrapper with Chirp 3 support."""
    
    def __init__(self):
        self.client = None
        self.available = False
        self._init_client()
    
    def _init_client(self):
        """Initialize Google Cloud Speech-to-Text client."""
        try:
            from google.cloud import speech
            
            # Check for credentials
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not creds_path or not os.path.exists(creds_path):
                logger.warning(
                    "Google Cloud credentials not found. "
                    "Set GOOGLE_APPLICATION_CREDENTIALS environment variable. "
                    "STT will be disabled."
                )
                return
            
            # V2 client will be created per-request with correct endpoint
            # Also keep V1 client for compatibility
            self.client = speech.SpeechClient()
            self.available = True
            logger.info("Google Cloud Speech-to-Text initialized successfully")
        except ImportError:
            logger.warning("google-cloud-speech not installed. STT disabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize Google STT: {e}")
    
    def is_available(self) -> bool:
        """Check if STT service is available."""
        return self.available
    
    def transcribe(
        self,
        audio_path: str,
        language_code: str = "ja-JP",
        model: str = "chirp_3",
        use_v2: bool = True,
        sample_rate_hertz: int = 16000,
    ) -> Dict:
        """
        Transcribe audio file using Google Cloud Speech-to-Text.
        
        Args:
            audio_path: Path to audio file
            language_code: Language code (e.g., "ja-JP")
            model: Model to use ("chirp_3", "latest_long", etc.)
            use_v2: Use V2 API (required for Chirp 3)
            sample_rate_hertz: Audio sample rate (16000 for web; 8000 for telephony/phone)
        
        Returns:
            Dict with transcription results
        """
        if not self.available:
            raise RuntimeError("STT service not available. Check Google Cloud credentials.")
        
        try:
            # Verify file exists and has content
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file does not exist: {audio_path}")
            
            file_size = os.path.getsize(audio_path)
            logger.info(f"Reading audio file: {audio_path} ({file_size} bytes, {sample_rate_hertz} Hz)")
            
            if file_size == 0:
                raise ValueError("Audio file is empty")
            
            # Read audio file
            with open(audio_path, "rb") as audio_file:
                audio_content = audio_file.read()
            
            if len(audio_content) != file_size:
                raise ValueError(f"File size mismatch: expected {file_size}, read {len(audio_content)}")
            
            # Minimum audio size check (WAV header is ~44 bytes, need some actual audio data)
            if len(audio_content) < 1000:
                logger.warning(f"Audio file is very small ({len(audio_content)} bytes) - may be too short for transcription")
            
            if use_v2 and model == "chirp_3":
                return self._transcribe_v2_chirp3(audio_content, language_code, sample_rate_hertz)
            else:
                return self._transcribe_v1(audio_content, language_code, model, sample_rate_hertz)
                
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            raise
    
    def _transcribe_v2_chirp3(self, audio_content: bytes, language_code: str, sample_rate_hertz: int = 16000) -> Dict:
        """Transcribe using V2 API with Chirp 3 model."""
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
        from google.api_core.client_options import ClientOptions
        
        # Get project ID from credentials
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            # Try to extract from credentials file
            import json
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and os.path.exists(creds_path):
                with open(creds_path) as f:
                    creds = json.load(f)
                    project_id = creds.get("project_id")
        
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT not set. Set it as environment variable.")
        
        # Use asia-northeast1 region for Japanese (Chirp 3 available there)
        region = "asia-northeast1"
        recognizer_id = f"projects/{project_id}/locations/{region}/recognizers/_"
        
        # Create client with correct endpoint
        client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{region}-speech.googleapis.com"
            )
        )
        
        # Configure recognition: native sample rate (8kHz for telephony, 16kHz for web)
        config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate_hertz,
                audio_channel_count=1,
            ),
            language_codes=[language_code],
            model="chirp_3",  # Chirp 3 model identifier (with underscore)
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
            ),
        )
        
        # Create recognition request
        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer_id,
            config=config,
            content=audio_content,
        )
        
        # Perform recognition
        response = client.recognize(request=request)
        
        # Log response for debugging
        logger.info(f"Google STT response: {len(response.results)} results")
        
        # Extract transcription
        full_text = ""
        alternatives = []
        
        if not response.results:
            logger.warning("Google STT returned no results - audio may be too short, silent, or unrecognized")
            return {
                "text": "",
                "language": language_code,
                "alternatives": [],
                "model": "chirp_3"
            }
        
        for result in response.results:
            if result.alternatives:
                best_alternative = result.alternatives[0]
                transcript = best_alternative.transcript.strip()
                if transcript:
                    full_text += transcript + " "
                    alternatives.append({
                        "transcript": transcript,
                        "confidence": best_alternative.confidence
                    })
                    logger.info(f"Google STT transcript: {transcript[:50]}... (confidence: {best_alternative.confidence})")
            else:
                logger.warning(f"Google STT result has no alternatives")
        
        final_text = full_text.strip()
        if not final_text:
            logger.warning("Google STT returned empty transcript - check audio quality and language")
        
        return {
            "text": final_text,
            "language": language_code,
            "alternatives": alternatives,
            "model": "chirp_3"
        }
    
    def _transcribe_v1(self, audio_content: bytes, language_code: str, model: str, sample_rate_hertz: int = 16000) -> Dict:
        """Transcribe using V1 API (fallback)."""
        from google.cloud import speech
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hertz,
            language_code=language_code,
            model=model,
            enable_automatic_punctuation=True,
        )
        
        audio = speech.RecognitionAudio(content=audio_content)
        
        # Perform recognition
        response = self.client.recognize(config=config, audio=audio)
        
        # Extract transcription
        full_text = ""
        alternatives = []
        
        for result in response.results:
            if result.alternatives:
                best_alternative = result.alternatives[0]
                full_text += best_alternative.transcript + " "
                alternatives.append({
                    "transcript": best_alternative.transcript,
                    "confidence": best_alternative.confidence
                })
        
        return {
            "text": full_text.strip(),
            "language": language_code,
            "alternatives": alternatives,
            "model": model
        }

    def _get_v2_client_and_recognizer(self):
        """Create Speech V2 client and recognizer ID for asia-northeast1 Chirp 3."""
        from google.cloud.speech_v2 import SpeechClient
        from google.api_core.client_options import ClientOptions
        import json
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            with open(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")) as f:
                project_id = json.load(f).get("project_id")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT not set")
        region = "asia-northeast1"
        recognizer_id = f"projects/{project_id}/locations/{region}/recognizers/_"
        client = SpeechClient(
            client_options=ClientOptions(api_endpoint=f"{region}-speech.googleapis.com")
        )
        return client, recognizer_id

    def streaming_recognize(
        self,
        audio_chunks: Iterator[bytes],
        language_code: str = "ja-JP",
        sample_rate_hertz: int = 16000,
    ) -> Iterator[Dict]:
        """
        Stream audio chunks to Google STT and yield interim/final results.
        Yields dicts: {"is_final": bool, "text": str, "confidence": float|None}.
        """
        if not self.available:
            raise RuntimeError("STT service not available.")
        from google.cloud.speech_v2.types import cloud_speech

        client, recognizer_id = self._get_v2_client_and_recognizer()
        config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate_hertz,
                audio_channel_count=1,
            ),
            language_codes=[language_code],
            model="chirp_3",
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
            ),
        )
        streaming_config = cloud_speech.StreamingRecognitionConfig(config=config)

        def request_generator():
            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=recognizer_id,
                streaming_config=streaming_config,
            )
            for chunk in audio_chunks:
                if chunk:
                    yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

        stream = client.streaming_recognize(requests=request_generator())
        for response in stream:
            for result in response.results or []:
                if result.alternatives:
                    alt = result.alternatives[0]
                    yield {
                        "is_final": getattr(result, "is_final", False),
                        "text": (alt.transcript or "").strip(),
                        "confidence": getattr(alt, "confidence", None),
                    }
