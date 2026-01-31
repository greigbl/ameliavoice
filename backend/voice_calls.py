"""
In-memory store for Twilio voice calls: transcript and per-turn latency (STT/LLM/TTS).
Used by twilio_stream and exposed via GET /api/voice/calls.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional
import threading

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """One user utterance + assistant response with latency breakdown."""
    user_text: str
    assistant_text: str
    stt_ms: float
    llm_ms: float
    tts_ms: float


@dataclass
class CallRecord:
    call_sid: str
    stream_sid: str = ""
    start_time: Optional[float] = None  # from time.time()
    end_time: Optional[float] = None
    turns: list[Turn] = field(default_factory=list)


_lock = threading.Lock()
_calls: dict[str, CallRecord] = {}


def register_call(call_sid: str, stream_sid: str = "") -> None:
    import time
    with _lock:
        _calls[call_sid] = CallRecord(
            call_sid=call_sid,
            stream_sid=stream_sid,
            start_time=time.time(),
        )
    logger.info("[voice_calls] registered call_sid=%s", call_sid)


def add_turn(
    call_sid: str,
    user_text: str,
    assistant_text: str,
    stt_ms: float,
    llm_ms: float,
    tts_ms: float,
) -> None:
    with _lock:
        rec = _calls.get(call_sid)
        if not rec:
            return
        rec.turns.append(Turn(
            user_text=user_text,
            assistant_text=assistant_text,
            stt_ms=stt_ms,
            llm_ms=llm_ms,
            tts_ms=tts_ms,
        ))


def end_call(call_sid: str) -> None:
    import time
    with _lock:
        rec = _calls.get(call_sid)
        if rec:
            rec.end_time = time.time()
    logger.info("[voice_calls] ended call_sid=%s", call_sid)


def get_calls() -> list[dict]:
    """Return list of call summaries (newest first)."""
    with _lock:
        records = list(_calls.values())
    return [
        {
            "call_sid": r.call_sid,
            "stream_sid": r.stream_sid,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "turn_count": len(r.turns),
        }
        for r in sorted(records, key=lambda x: (x.start_time or 0), reverse=True)
    ]


def get_call(call_sid: str) -> Optional[dict]:
    """Return full call record with transcript and per-turn latencies."""
    with _lock:
        rec = _calls.get(call_sid)
        if not rec:
            return None
        return {
            "call_sid": rec.call_sid,
            "stream_sid": rec.stream_sid,
            "start_time": rec.start_time,
            "end_time": rec.end_time,
            "turns": [
                {
                    "user_text": t.user_text,
                    "assistant_text": t.assistant_text,
                    "stt_ms": round(t.stt_ms, 1),
                    "llm_ms": round(t.llm_ms, 1),
                    "tts_ms": round(t.tts_ms, 1),
                }
                for t in rec.turns
            ],
        }
