"""
Audio utility functions for format conversion and processing.
"""
import os
import logging
from pathlib import Path
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def convert_audio_to_wav(input_path: str, output_path: str = None, sample_rate: int = 16000) -> str:
    """
    Convert audio file to WAV format with specified sample rate.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output WAV file (optional, creates temp file if not provided)
        sample_rate: Target sample rate in Hz (default: 16000 for ASR)
    
    Returns:
        Path to converted WAV file
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample if needed
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        
        # Set output path if not provided
        if output_path is None:
            output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
        
        # Export as WAV with explicit parameters for Google STT compatibility
        # Use PCM format (sample_width=2 for 16-bit) which Google STT expects
        audio.export(
            output_path, 
            format="wav",
            parameters=["-ac", "1", "-ar", str(sample_rate), "-sample_fmt", "s16"]  # mono, sample rate, 16-bit PCM
        )
        
        # Verify the output file exists and has content
        if not os.path.exists(output_path):
            raise ValueError(f"Conversion failed: output file not created: {output_path}")
        
        output_size = os.path.getsize(output_path)
        if output_size < 1000:  # WAV header is ~44 bytes, need some audio data
            logger.warning(f"Converted WAV file is very small ({output_size} bytes) - may indicate conversion issue")
        
        logger.info(f"Converted {input_path} to {output_path} ({sample_rate}Hz, mono, {output_size} bytes)")
        return output_path
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        # If conversion fails, return original path (might already be in correct format)
        return input_path


def ensure_wav_format(audio_path: str) -> str:
    """
    Ensure audio file is in WAV format, converting if necessary.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Path to WAV file (may be original or converted)
    """
    ext = Path(audio_path).suffix.lower()
    
    # If already WAV, check if conversion is needed
    if ext == '.wav':
        # Could check sample rate here, but for now just return
        return audio_path
    
    # Convert to WAV
    return convert_audio_to_wav(audio_path)
