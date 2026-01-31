"""
ASR Model Manager for Whisper, Parakeet, and Google STT models.
"""
import os
import logging
from typing import Dict, List, Optional
import threading
import torch
import whisper
import numpy as np

logger = logging.getLogger(__name__)


class ASRModelManager:
    """Manages different ASR models (Whisper, Parakeet, Google STT)."""
    
    def __init__(self):
        self.current_model = None
        self.current_model_type = None
        # Cache Whisper models by size to support parallel comparisons safely.
        self.whisper_models: Dict[str, object] = {}
        self._whisper_model_locks: Dict[str, threading.Lock] = {}
        self.parakeet_model = None
        self.parakeet_available = False  # Initialize before checking
        self.google_stt_service = None
        self._init_google_stt()
        
        # Check device
        # Note: Whisper has issues with MPS backend (sparse tensor operations not supported)
        # So we'll use CPU for Whisper models on M1 Mac to avoid errors
        if torch.cuda.is_available():
            self.device = "cuda"
            self.whisper_device = "cuda"
            logger.info("Using CUDA backend")
        elif torch.backends.mps.is_available():
            # MPS available but Whisper doesn't fully support sparse tensor ops
            # Use CPU for Whisper models to avoid "SparseMPS" backend errors
            self.device = "cpu"
            self.whisper_device = "cpu"
            logger.info("MPS available but using CPU for Whisper (MPS has sparse tensor limitations)")
        else:
            self.device = "cpu"
            self.whisper_device = "cpu"
            logger.info("Using CPU backend")
        
        # Available models
        self.available_models = {
            "whisper-tiny": {"type": "whisper", "model": "tiny"},
            "whisper-base": {"type": "whisper", "model": "base"},
            "whisper-small": {"type": "whisper", "model": "small"},
            "whisper-medium": {"type": "whisper", "model": "medium"},
            "whisper-large": {"type": "whisper", "model": "large"},
            "whisper-large-v2": {"type": "whisper", "model": "large-v2"},
            "whisper-large-v3": {"type": "whisper", "model": "large-v3"},
        }
        
        # Always add Parakeet models to the list (they'll show as available or unavailable)
        # This way users can see them and get helpful error messages if not installed
        parakeet_models = {
            "parakeet-tiny": "nvidia/parakeet-tiny-ctc-600M",
            "parakeet-small": "nvidia/parakeet-small-ctc-600M",
            "parakeet-medium": "nvidia/parakeet-medium-ctc-600M",
        }
        
        for name, model_id in parakeet_models.items():
            self.available_models[name] = {
                "type": "parakeet",
                "model": model_id
            }
        
        # Always include Google STT in the list (will show as available/unavailable)
        # This mirrors Parakeet behavior so users can see the option in the dropdown.
        self.available_models["google-chirp-3"] = {
            "type": "google_stt",
            "model": "chirp_3",
        }
        if self.google_stt_service and self.google_stt_service.is_available():
            logger.info("Google STT (Chirp 3) available")
        else:
            logger.info("Google STT (Chirp 3) not configured; set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT")
        
        # Check if Parakeet/Nemo is actually available
        self._check_parakeet_availability()
    
    def _init_google_stt(self):
        """Initialize Google STT service."""
        # Skip if already initialized and available
        if self.google_stt_service and self.google_stt_service.is_available():
            return
            
        try:
            # Import at runtime to avoid issues if module not available
            import google_stt_service
            self.google_stt_service = google_stt_service.GoogleSTTService()
            if self.google_stt_service.is_available():
                logger.info("Google STT service initialized and available")
            else:
                logger.warning("Google STT service initialized but not available (check credentials)")
        except (ImportError, ModuleNotFoundError) as e:
            logger.info(f"Google STT service not available: {e}")
            self.google_stt_service = None
        except Exception as e:
            logger.warning(f"Failed to initialize Google STT: {e}")
            self.google_stt_service = None
    
    def _check_parakeet_availability(self):
        """Check if Parakeet/Nemo toolkit is available."""
        import platform
        
        # Check if we're on M1 Mac (ARM64)
        is_m1_mac = platform.machine() == "arm64" and platform.system() == "Darwin"
        
        if is_m1_mac:
            self.parakeet_available = False
            logger.warning("Parakeet/Nemo is not compatible with M1 Mac (ARM64)")
            logger.warning("Parakeet requires triton which doesn't support macOS/ARM")
            logger.warning("Please use Whisper models instead on M1 Mac")
            return
        
        try:
            import nemo.collections.asr as nemo_asr
            self.parakeet_available = True
            logger.info("Parakeet/Nemo toolkit is available")
        except ImportError:
            self.parakeet_available = False
            logger.info("Parakeet/Nemo not installed. Models will show but won't work until installed.")
            logger.info("Install with: uv sync --extra parakeet")
            logger.info("Note: Parakeet is not compatible with M1 Mac (ARM64)")
        except Exception as e:
            self.parakeet_available = False
            logger.warning(f"Parakeet availability check failed: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.available_models.keys())

    def transcribe_with_model(self, model_name: str, audio_path: str, language: Optional[str] = "ja") -> Dict:
        """
        Transcribe audio using the specified model WITHOUT mutating global selection.
        This is important for running multiple models in parallel.
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Available: {self.get_available_models()}")

        model_info = self.available_models[model_name]
        model_type = model_info["type"]

        if model_type == "whisper":
            return self._transcribe_whisper_by_size(audio_path, language=language, model_size=model_info["model"])
        if model_type == "google_stt":
            return self._transcribe_google_stt(audio_path, language=language)
        if model_type == "parakeet":
            # Parakeet is not supported on M1; keep behavior consistent with select_model.
            self.select_model(model_name)  # will raise a helpful error on unsupported platforms
            return self._transcribe_parakeet(audio_path)

        raise ValueError(f"Unknown model type: {model_type}")
    
    def select_model(self, model_name: str):
        """Select and load an ASR model."""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Available: {self.get_available_models()}")
        
        model_info = self.available_models[model_name]
        model_type = model_info["type"]
        
        if model_type == "whisper":
            self._load_whisper(model_info["model"])
        elif model_type == "parakeet":
            if not self.parakeet_available:
                import platform
                is_m1_mac = platform.machine() == "arm64" and platform.system() == "Darwin"
                
                if is_m1_mac:
                    raise ValueError(
                        "Parakeet models are not compatible with M1 Mac (ARM64).\n"
                        "Parakeet requires triton which doesn't support macOS/ARM.\n"
                        "Please use Whisper models instead (they work great on M1 Mac!)."
                    )
                else:
                    raise ValueError(
                        "Parakeet models require Nemo toolkit.\n"
                        "Install with: uv sync --extra parakeet"
                    )
            self._load_parakeet(model_info["model"])
        elif model_type == "google_stt":
            if not self.google_stt_service or not self.google_stt_service.is_available():
                raise ValueError(
                    "Google STT not available. Check:\n"
                    "1. GOOGLE_APPLICATION_CREDENTIALS is set\n"
                    "2. GOOGLE_CLOUD_PROJECT is set\n"
                    "3. Speech-to-Text API is enabled in Google Cloud Console"
                )
            # Google STT doesn't need pre-loading, just mark as selected
            logger.info(f"Selected Google STT model: {model_info['model']}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.current_model = model_name
        self.current_model_type = model_type
        logger.info(f"Selected model: {model_name} ({model_type})")
    
    def _load_whisper(self, model_size: str):
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {model_size} on device: {self.whisper_device}")
            # Use whisper_device which is always CPU on M1 Mac to avoid MPS sparse tensor errors
            model = whisper.load_model(model_size, device=self.whisper_device)
            self.whisper_models[model_size] = model
            logger.info(f"Whisper model loaded successfully on {self.whisper_device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _load_parakeet(self, model_id: str):
        """Load Parakeet model."""
        try:
            import nemo.collections.asr as nemo_asr
            logger.info(f"Loading Parakeet model: {model_id}")
            # Store model ID for on-demand loading
            self.parakeet_model_id = model_id
            # Reset model to force reload on first use
            self.parakeet_model = None
            logger.info("Parakeet model ready (will load on first transcription)")
        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}")
            raise
    
    def transcribe(self, audio_path: str, language: Optional[str] = "ja") -> Dict:
        """Transcribe audio file."""
        if self.current_model is None:
            # Default to whisper-base if no model selected
            self.select_model("whisper-base")
        
        if self.current_model_type == "whisper":
            # Use the selected Whisper size based on current_model mapping
            model_size = self.available_models[self.current_model]["model"]
            return self._transcribe_whisper_by_size(audio_path, language=language, model_size=model_size)
        elif self.current_model_type == "parakeet":
            return self._transcribe_parakeet(audio_path)
        elif self.current_model_type == "google_stt":
            return self._transcribe_google_stt(audio_path, language)
        else:
            raise ValueError(f"Unknown model type: {self.current_model_type}")
    
    def _get_whisper_lock(self, model_size: str) -> threading.Lock:
        if model_size not in self._whisper_model_locks:
            self._whisper_model_locks[model_size] = threading.Lock()
        return self._whisper_model_locks[model_size]

    def _ensure_whisper_loaded(self, model_size: str) -> None:
        if model_size in self.whisper_models:
            return
        lock = self._get_whisper_lock(model_size)
        with lock:
            # Double-check in case another thread loaded it while waiting.
            if model_size in self.whisper_models:
                return
            self._load_whisper(model_size)

    def _transcribe_whisper_by_size(self, audio_path: str, language: Optional[str], model_size: str) -> Dict:
        """Transcribe using Whisper for a specific model size."""
        self._ensure_whisper_loaded(model_size)

        model = self.whisper_models.get(model_size)
        if model is None:
            raise ValueError(f"Whisper model not loaded: {model_size}")

        try:
            result = model.transcribe(
                audio_path,
                language=language if language else None,
                task="transcribe",
            )

            return {
                "text": result["text"].strip(),
                "language": result.get("language", language or "ja"),
                "segments": result.get("segments", []),
            }
        except Exception as e:
            logger.error(f"Whisper transcription error ({model_size}): {e}")
            raise
    
    def _transcribe_parakeet(self, audio_path: str) -> Dict:
        """Transcribe using Parakeet."""
        try:
            import nemo.collections.asr as nemo_asr
            import soundfile as sf
            
            # Load model on first use
            if self.parakeet_model is None:
                self.parakeet_model = nemo_asr.models.ASRModel.from_pretrained(
                    self.parakeet_model_id
                )
            
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Transcribe
            transcriptions = self.parakeet_model.transcribe([audio_path])
            text = transcriptions[0] if transcriptions else ""
            
            return {
                "text": text.strip(),
                "language": "ja"
            }
        except Exception as e:
            logger.error(f"Parakeet transcription error: {e}")
            raise
    
    def _transcribe_google_stt(self, audio_path: str, language: Optional[str] = "ja") -> Dict:
        """Transcribe using Google STT (Chirp 3)."""
        # Ensure Google STT service is initialized
        if self.google_stt_service is None:
            self._init_google_stt()
        
        if not self.google_stt_service:
            raise ValueError("Google STT service not available (failed to initialize)")
        
        # Re-check availability - credentials might have been set after initialization
        if not self.google_stt_service.is_available():
            # Try re-initializing in case credentials were set after startup
            self._init_google_stt()
            if not self.google_stt_service or not self.google_stt_service.is_available():
                raise ValueError(
                    "Google STT service not available. Check:\n"
                    "1. GOOGLE_APPLICATION_CREDENTIALS is set and points to valid JSON file\n"
                    "2. GOOGLE_CLOUD_PROJECT is set\n"
                    "3. Speech-to-Text API is enabled in Google Cloud Console"
                )
        
        try:
            # Convert language code format (ja -> ja-JP)
            language_code = "ja-JP" if language == "ja" else language
            
            result = self.google_stt_service.transcribe(
                audio_path=audio_path,
                language_code=language_code,
                model="chirp_3",
                use_v2=True
            )
            
            return {
                "text": result["text"],
                "language": result.get("language", language_code),
                "model": "google-chirp-3",
                "alternatives": result.get("alternatives", [])
            }
        except Exception as e:
            logger.error(f"Google STT transcription error: {e}")
            raise
