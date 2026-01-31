"""
Twilio Media Streams audio conversion: μ-law 8kHz ↔ linear16 16kHz, TTS → μ-law 8kHz.
"""
import audioop
import base64
import io
import logging
from typing import Optional

from pydub import AudioSegment

logger = logging.getLogger(__name__)

TWILIO_SAMPLE_RATE = 8000
ASR_SAMPLE_RATE = 16000
# 20ms chunks for Twilio (160 samples at 8kHz = 160 bytes μ-law)
CHUNK_MS = 20
CHUNK_SAMPLES_8K = int(TWILIO_SAMPLE_RATE * CHUNK_MS / 1000)


def mulaw_8k_to_linear16_16k_wav(mulaw_bytes: bytes) -> bytes:
    """
    Convert μ-law 8kHz mono to 16-bit linear PCM 16kHz WAV bytes (no header).
    Caller can prepend WAV header or write to file.
    """
    if not mulaw_bytes:
        raise ValueError("Empty mulaw bytes")
    # audioop.ulaw2lin: (fragment, width) -> linear 16-bit, same sample count
    linear_8k = audioop.ulaw2lin(mulaw_bytes, 1)  # 1 = 16-bit output
    # Build segment: 16-bit, 8kHz, mono
    segment = AudioSegment.from_raw(
        io.BytesIO(linear_8k),
        sample_width=2,
        frame_rate=TWILIO_SAMPLE_RATE,
        channels=1,
    )
    segment = segment.set_frame_rate(ASR_SAMPLE_RATE)
    out = io.BytesIO()
    segment.export(out, format="wav", parameters=["-ac", "1", "-ar", str(ASR_SAMPLE_RATE)])
    return out.getvalue()


def mulaw_8k_to_wav_file(mulaw_bytes: bytes, wav_path: str) -> str:
    """Write μ-law 8kHz to a 16kHz WAV file for Google ASR (web / high-quality path)."""
    wav_bytes = mulaw_8k_to_linear16_16k_wav(mulaw_bytes)
    with open(wav_path, "wb") as f:
        f.write(wav_bytes)
    return wav_path


def mulaw_8k_to_wav_file_8k(mulaw_bytes: bytes, wav_path: str) -> str:
    """
    Write μ-law 8kHz to an 8kHz linear16 WAV file (no resampling).
    Use for telephony/phone path so Google STT gets native 8kHz (recommended for phone).
    """
    if not mulaw_bytes:
        raise ValueError("Empty mulaw bytes")
    linear_8k = audioop.ulaw2lin(mulaw_bytes, 1)
    segment = AudioSegment.from_raw(
        io.BytesIO(linear_8k),
        sample_width=2,
        frame_rate=TWILIO_SAMPLE_RATE,
        channels=1,
    )
    out = io.BytesIO()
    segment.export(out, format="wav", parameters=["-ac", "1", "-ar", str(TWILIO_SAMPLE_RATE)])
    with open(wav_path, "wb") as f:
        f.write(out.getvalue())
    return wav_path


def pcm_to_mulaw_8k(pcm_16bit_bytes: bytes, sample_rate: int) -> bytes:
    """
    Convert 16-bit PCM at given sample_rate to μ-law 8kHz mono.
    Returns raw μ-law bytes (for Twilio media payload).
    """
    if not pcm_16bit_bytes:
        return b""
    segment = AudioSegment.from_raw(
        io.BytesIO(pcm_16bit_bytes),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    segment = segment.set_frame_rate(TWILIO_SAMPLE_RATE).set_channels(1)
    pcm_8k = segment.raw_data
    # audioop.lin2ulaw: 16-bit linear fragment -> μ-law (1 byte per sample)
    return audioop.lin2ulaw(pcm_8k, 2)


def mp3_to_mulaw_8k_chunks(mp3_bytes: bytes, chunk_size: int = 160) -> list[bytes]:
    """
    Decode MP3 to PCM, resample to 8kHz, convert to μ-law, yield chunks of chunk_size bytes.
    chunk_size=160 = 20ms at 8kHz (Twilio-friendly).
    """
    segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    segment = segment.set_frame_rate(TWILIO_SAMPLE_RATE).set_channels(1)
    pcm = segment.raw_data
    mulaw = audioop.lin2ulaw(pcm, 2)
    chunks = []
    for i in range(0, len(mulaw), chunk_size):
        chunks.append(mulaw[i : i + chunk_size])
    return chunks


def mulaw_chunks_to_base64(chunks: list[bytes]) -> list[str]:
    """Encode μ-law chunks to base64 for Twilio media messages."""
    return [base64.b64encode(c).decode("ascii") for c in chunks]
