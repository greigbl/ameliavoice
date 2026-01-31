"""
Twilio Media Stream WebSocket handler: per-call state, buffer, VAD, and pipeline (ASR → chat → TTS).
"""
import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.google_stt_service import GoogleSTTService
from backend import voice_calls
from backend import voice_calls_live
from backend.tts_service import TTSService
from backend.twilio_audio import (
    mulaw_8k_to_wav_file_8k,
    mp3_to_mulaw_8k_chunks,
    mulaw_chunks_to_base64,
    TWILIO_SAMPLE_RATE,
)
from backend.audio_utils import ensure_wav_format
from backend.voice_prompt import build_voice_system_message

logger = logging.getLogger(__name__)

# Silence in μ-law: 0xff is common for silence. Chunks with mean near 0xff are "silent".
# Use 250 so phone line noise still counts as silence (254 is too strict on PSTN).
SILENCE_MULAW_THRESHOLD_BYTE = 250
MIN_UTTERANCE_MS = 600
SILENCE_MS = 1000
MIN_BUFFER_BYTES = int(TWILIO_SAMPLE_RATE * MIN_UTTERANCE_MS / 1000)  # 4800 bytes at 8kHz
SILENCE_CHUNKS = int(SILENCE_MS / 20)  # 20ms chunks -> 50 chunks for 1s silence
# If we've buffered this much without silence, run pipeline anyway (avoid infinite hang on noisy lines).
MAX_UTTERANCE_BYTES = int(TWILIO_SAMPLE_RATE * 8)  # 8 seconds at 8kHz = 64000 bytes


def _strip_markdown_for_tts(text: str) -> str:
    """Strip markdown so TTS does not read asterisks etc."""
    if not text or not text.strip():
        return text or ""
    out = text
    out = re.sub(r"```[\s\S]*?```", " ", out)
    out = re.sub(r"`[^`]+`", lambda m: m.group(0)[1:-1], out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"__([^_]+)__", r"\1", out)
    out = re.sub(r"\*([^*]+)\*", r"\1", out)
    out = re.sub(r"_([^_]+)_", r"\1", out)
    out = re.sub(r"^#{1,6}\s+", "", out, flags=re.MULTILINE)
    out = re.sub(r"~~([^~]+)~~", r"\1", out)
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)
    out = re.sub(r"\n{2,}", "\n", out)
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()


# Log media progress every N inbound chunks to avoid flooding logs
MEDIA_LOG_EVERY_N_CHUNKS = 50

@dataclass
class CallState:
    stream_sid: str = ""
    call_sid: str = ""
    messages: list[dict] = field(default_factory=list)
    buffer: bytearray = field(default_factory=bytearray)
    silent_chunk_count: int = 0
    processing: bool = False
    language_code: str = "ja-JP"
    integration: str = "openai"
    media_chunk_count: int = 0  # inbound media chunks received


def _is_silent_chunk(payload: bytes) -> bool:
    if len(payload) < 10:
        return True
    mean_byte = sum(payload) / len(payload)
    return mean_byte >= SILENCE_MULAW_THRESHOLD_BYTE


def _run_pipeline_sync(
    state: CallState,
    stt: GoogleSTTService,
    tts: TTSService,
    utterance: bytes,
) -> tuple[Optional[str], Optional[str], Optional[bytes], float, float, float]:
    """
    Sync pipeline: transcribe → chat → TTS.
    Returns (user_text, assistant_content, tts_mp3_bytes, stt_ms, llm_ms, tts_ms).
    Runs in thread; raises on error.
    """
    stt_ms = llm_ms = tts_ms = 0.0
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    wav_path = tmp_path
    try:
        # Telephony: send native 8kHz to Google (no upsampling); better for phone.
        mulaw_8k_to_wav_file_8k(utterance, tmp_path)
        wav_path = ensure_wav_format(tmp_path)
        t0 = time.perf_counter()
        result = stt.transcribe(
            wav_path,
            language_code=state.language_code,
            model="chirp_3",
            use_v2=True,
            sample_rate_hertz=TWILIO_SAMPLE_RATE,
        )
        stt_ms = (time.perf_counter() - t0) * 1000
        user_text = (result.get("text") or "").strip()
        voice_calls_live.emit(state.call_sid, "stt_done", {"user_text": user_text, "stt_ms": round(stt_ms, 1)})
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if wav_path != tmp_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    if not user_text:
        return None, None, None, stt_ms, 0.0, 0.0

    state.messages.append({"role": "user", "content": user_text})
    logger.info("Twilio user said: %s", user_text[:80])

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    lang = "ja" if state.language_code.startswith("ja") else "en"
    system_content = build_voice_system_message(lang)
    chat_messages = [{"role": "system", "content": system_content}] + [{"role": m["role"], "content": m["content"]} for m in state.messages]
    # max_tokens is generous so the model is not truncated; length is controlled by the prompt only
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_messages,
        max_tokens=2048,
    )
    llm_ms = (time.perf_counter() - t0) * 1000
    assistant_content = (resp.choices[0].message.content or "").strip()
    state.messages.append({"role": "assistant", "content": assistant_content})
    voice_calls_live.emit(state.call_sid, "llm_done", {"assistant_text": assistant_content, "llm_ms": round(llm_ms, 1)})

    plain = _strip_markdown_for_tts(assistant_content) or assistant_content
    voice_calls_live.emit(state.call_sid, "tts_start", {})
    t0 = time.perf_counter()
    audio_mp3 = tts.synthesize(
        text=plain,
        language_code=state.language_code,
        audio_encoding="MP3",
    )
    tts_ms = (time.perf_counter() - t0) * 1000
    voice_calls_live.emit(state.call_sid, "tts_done", {"tts_ms": round(tts_ms, 1)})
    return user_text, assistant_content, audio_mp3, stt_ms, llm_ms, tts_ms


async def _run_pipeline(
    state: CallState,
    stt: GoogleSTTService,
    tts: TTSService,
    ws_send,
) -> None:
    """Take buffered μ-law, run sync pipeline in thread, send TTS media back."""
    if state.processing or len(state.buffer) < MIN_BUFFER_BYTES:
        return
    state.processing = True
    utterance = bytes(state.buffer)
    buf_len = len(utterance)
    state.buffer.clear()
    state.silent_chunk_count = 0

    logger.info("[voice call_sid=%s] pipeline started buffer=%d bytes (%.1fs)", state.call_sid, buf_len, buf_len / (TWILIO_SAMPLE_RATE * 1.0))

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _run_pipeline_sync(state, stt, tts, utterance),
        )
        user_text, assistant_content, audio_mp3, stt_ms, llm_ms, tts_ms = result
        if not audio_mp3:
            logger.info("[voice call_sid=%s] pipeline done no speech (empty or silent)", state.call_sid)
            state.processing = False
            return
        voice_calls.add_turn(
            state.call_sid,
            user_text or "",
            assistant_content or "",
            stt_ms,
            llm_ms,
            tts_ms,
        )

        chunks = mp3_to_mulaw_8k_chunks(audio_mp3)
        b64_chunks = mulaw_chunks_to_base64(chunks)
        for payload_b64 in b64_chunks:
            msg = {
                "event": "media",
                "streamSid": state.stream_sid,
                "media": {"payload": payload_b64},
            }
            await ws_send(json.dumps(msg))
        mark_name = f"tts-{state.call_sid}-{len(state.messages)}"
        await ws_send(json.dumps({
            "event": "mark",
            "streamSid": state.stream_sid,
            "mark": {"name": mark_name},
        }))
        logger.info("[voice call_sid=%s] pipeline done user=%r sent TTS %d chunks", state.call_sid, (user_text or "")[:60], len(b64_chunks))
    except Exception as e:
        logger.exception("[voice call_sid=%s] pipeline error: %s", state.call_sid, e)
    finally:
        state.processing = False


async def handle_twilio_stream(websocket, stt: GoogleSTTService, tts: TTSService):
    """
    Handle one Twilio Media Stream WebSocket. Parse connected/start/media/stop;
    buffer inbound audio, run VAD, then ASR → chat → TTS and send media back.
    """
    state = CallState()
    language = (os.getenv("TWILIO_LANGUAGE") or "ja").strip().lower()
    integration = (os.getenv("TWILIO_AI") or "openai").strip().lower()
    state.language_code = "ja-JP" if language == "ja" else "en-US"
    state.integration = integration if integration in ("openai", "amelia") else "openai"

    async def send(msg: str):
        try:
            await websocket.send_text(msg)
        except Exception as e:
            logger.warning("Twilio stream send error: %s", e)

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            event = data.get("event")

            if event == "connected":
                logger.info("[voice] stream connected")
                continue
            if event == "start":
                start = data.get("start") or {}
                state.stream_sid = data.get("streamSid") or start.get("streamSid") or ""
                state.call_sid = start.get("callSid") or ""
                tracks = start.get("tracks", [])
                voice_calls.register_call(state.call_sid, state.stream_sid)
                logger.info(
                    "[voice call_sid=%s] stream start stream_sid=%s tracks=%s",
                    state.call_sid, state.stream_sid, tracks,
                )
                continue
            if event == "stop":
                voice_calls.end_call(state.call_sid)
                logger.info(
                    "[voice call_sid=%s] stream stop media_chunks_received=%d buffer_at_end=%d",
                    state.call_sid, state.media_chunk_count, len(state.buffer),
                )
                break
            if event == "media":
                media = data.get("media") or {}
                track = media.get("track", "")
                payload_b64 = media.get("payload")
                if not payload_b64:
                    continue
                if track and track != "inbound":
                    continue
                try:
                    chunk = base64.b64decode(payload_b64)
                except Exception as ex:
                    logger.warning("[voice call_sid=%s] media decode error: %s", state.call_sid, ex)
                    continue
                state.media_chunk_count += 1
                if state.media_chunk_count == 1:
                    logger.info("[voice call_sid=%s] first inbound media chunk received", state.call_sid)
                elif state.media_chunk_count % MEDIA_LOG_EVERY_N_CHUNKS == 0:
                    logger.info(
                        "[voice call_sid=%s] media chunks=%d buffer=%d bytes silent_run=%d",
                        state.call_sid, state.media_chunk_count, len(state.buffer), state.silent_chunk_count,
                    )
                buf_len_before = len(state.buffer)
                state.buffer.extend(chunk)
                if _is_silent_chunk(chunk):
                    state.silent_chunk_count += 1
                    if (
                        len(state.buffer) >= MIN_BUFFER_BYTES
                        and state.silent_chunk_count >= SILENCE_CHUNKS
                    ):
                        asyncio.create_task(_run_pipeline(state, stt, tts, send))
                else:
                    state.silent_chunk_count = 0
                # Fallback: after ~8s of continuous speech with no silence, run pipeline anyway
                if (
                    buf_len_before < MAX_UTTERANCE_BYTES
                    and len(state.buffer) >= MAX_UTTERANCE_BYTES
                    and not state.processing
                ):
                    logger.info(
                        "[voice call_sid=%s] max utterance reached buffer=%d bytes, running pipeline",
                        state.call_sid, len(state.buffer),
                    )
                    asyncio.create_task(_run_pipeline(state, stt, tts, send))
                continue
            logger.debug("[voice call_sid=%s] unhandled event=%s", state.call_sid, event)
    except Exception as e:
        logger.exception("[voice call_sid=%s] stream error: %s", state.call_sid, e)
