"""
Voice conversation app - Phase 1: Google ASR and TTS, OpenAI chat.
"""
import asyncio
import logging
import os
import tempfile
import time
from urllib.parse import urlparse, urlunparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
if os.getenv("GOOGLE_PROJECT") and not os.getenv("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ["GOOGLE_PROJECT"]
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.google_stt_service import GoogleSTTService
from backend.whisper_stt_service import WhisperSTTService
from backend.tts_service import TTSService
from backend.audio_utils import ensure_wav_format
from backend.twilio_stream import handle_twilio_stream
from backend import voice_calls
from backend import voice_calls_live
from backend import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Amelia Voice", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to built frontend (so we can serve it from backend when behind a proxy)
_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

# Lazy-init services (STT provider: google | whisper via STT_PROVIDER)
_stt_service: Optional[object] = None
_stt_provider: Optional[str] = None
_tts_service: Optional[TTSService] = None


def get_stt():
    global _stt_service, _stt_provider
    provider = (os.getenv("STT_PROVIDER") or "google").strip().lower()
    if _stt_service is None or _stt_provider != provider:
        _stt_provider = provider
        if provider == "whisper":
            _stt_service = WhisperSTTService()
        else:
            _stt_service = GoogleSTTService()
    return _stt_service


def get_tts():
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service


# --- Request/Response models ---


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
class ChatRequestX(BaseModel):
    query: str
    history: list[ChatMessage]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    integration: str = "openai"  # "openai" | "amelia" (amelia stub for Phase 2)
    language: str = "en"  # "ja" | "en" for localized tool fallbacks (e.g. end_conversation default message)
    verbosity: str | None = None  # "brief" | "normal" | "detailed"; overrides VOICE_VERBOSITY when set (e.g. from web app)


class ChatResponse(BaseModel):
    message: ChatMessage
    done: bool = True
    end_conversation: bool = False  # agent invoked end_conversation tool; client should stop listening


class ChatQueryRequest(BaseModel):
    """RAG-style chat: single query + history. Used by POST /api/chat/query."""
    query: str
    history: list[ChatMessage] = []


class ChatQueryResponse(BaseModel):
    answer: str
    sources: list[str] = []
    intent: str = "SEARCH"


class TranscribeResponse(BaseModel):
    text: str
    language: str
    model: str = "google"


class TTSRequest(BaseModel):
    text: str
    language_code: str = "ja-JP"


# --- Routes ---


@app.api_route("/", methods=["GET", "POST"])
async def root(request: Request):
    """Root: POST = Twilio hint. GET = API info or serve built frontend if frontend/dist exists."""
    if request.method == "POST":
        logger.warning("Received POST / (likely Twilio). Set voice webhook to https://your-host/voice/incoming")
        return Response(
            content="Voice webhook is at /voice/incoming. Set Twilio A CALL COMES IN to: https://your-host/voice/incoming",
            status_code=404,
            media_type="text/plain",
        )
    if os.path.isdir(_FRONTEND_DIST):
        index_path = os.path.join(_FRONTEND_DIST, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
    return {"message": "Amelia Voice API", "docs": "/docs", "voice_webhook": "/voice/incoming"}


@app.get("/api/health")
def health():
    stt = get_stt()
    tts = get_tts()
    return {
        "ok": True,
        "stt_available": stt.is_available(),
        "tts_available": tts.is_available(),
    }


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("ja"),
    model: str = Form("google"),  # "google" | "whisper"
):
    """Transcribe audio. model=google (Google Chirp) or model=whisper (OpenAI Whisper; good for Japanese)."""
    if model not in ("google", "whisper"):
        raise HTTPException(
            status_code=400,
            detail="model must be 'google' or 'whisper'",
        )
    stt = WhisperSTTService() if model == "whisper" else GoogleSTTService()
    if not stt.is_available():
        raise HTTPException(status_code=503, detail=f"{model} STT not available")
    lang_code = "ja-JP" if language in ("ja", "JA") else "en-US"
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio")
    suffix = "." + (audio.filename or "audio").split(".")[-1] if "." in (audio.filename or "") else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    wav_path = tmp_path
    try:
        wav_path = ensure_wav_format(tmp_path)
        result = stt.transcribe(
            wav_path,
            language_code=lang_code,
            model="chirp_3" if model == "google" else "whisper-1",
            use_v2=True,
        )
        return TranscribeResponse(
            text=result.get("text", ""),
            language=result.get("language", lang_code),
            model=model,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if wav_path != tmp_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """Synthesize speech from text using Google TTS. Returns MP3."""
    from fastapi.responses import Response
    tts = get_tts()
    if not tts.is_available():
        raise HTTPException(status_code=503, detail="Google TTS not available")
    audio_bytes = tts.synthesize(
        text=req.text,
        language_code=req.language_code,
        audio_encoding="MP3",
    )
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.get("/api/tts")
async def text_to_speech_get(
    text: str,
    language_code: str = "ja-JP",
):
    """GET variant for TTS (e.g. for audio src URL). Returns MP3."""
    tts = get_tts()
    if not tts.is_available():
        raise HTTPException(status_code=503, detail="Google TTS not available")
    audio_bytes = tts.synthesize(
        text=text,
        language_code=language_code,
        audio_encoding="MP3",
    )
    from fastapi.responses import Response
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
    )


# --- Twilio Voice (Phase 3) ---


def _twilio_stream_url() -> str:
    """Build WSS URL for Twilio Media Stream using only the origin of TWILIO_VOICE_WEBHOOK_URL (no path)."""
    base = (os.getenv("TWILIO_VOICE_WEBHOOK_URL") or "").strip().rstrip("/")
    if not base:
        base = "https://localhost:8000"
    parsed = urlparse(base if "://" in base else "https://" + base)
    netloc = parsed.netloc or parsed.path
    scheme = "wss" if (parsed.scheme or "https") == "https" else "ws"
    origin = urlunparse((scheme, netloc, "", "", "", ""))
    stream_url = f"{origin}/voice/stream"
    return stream_url


@app.post("/voice/incoming")
async def voice_incoming(request: Request):
    """
    Twilio voice webhook: validate signature, return TwiML that connects the call
    to our Media Stream WebSocket (bidirectional).
    """
    auth_token = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    skip_validation = (os.getenv("TWILIO_SKIP_VALIDATION") or "").strip().lower() in ("1", "true", "yes")

    if skip_validation:
        logger.warning("TWILIO_SKIP_VALIDATION is set; skipping Twilio signature validation")
    elif not auth_token:
        logger.warning("TWILIO_AUTH_TOKEN not set; skipping signature validation")
    else:
        try:
            from twilio.request_validator import RequestValidator
            form = await request.form()
            params = {k: v for k, v in form.items()}
            base = (os.getenv("TWILIO_VOICE_WEBHOOK_URL") or "").strip().rstrip("/")
            url = (base + "/voice/incoming") if base else str(request.url)
            signature = (request.headers.get("X-Twilio-Signature") or "").strip()
            validator = RequestValidator(auth_token)
            if not validator.validate(url, params, signature):
                logger.warning(
                    "Twilio signature validation failed. Check: TWILIO_VOICE_WEBHOOK_URL=%s/voice/incoming "
                    "matches the URL configured in Twilio exactly (https, no trailing slash on base); "
                    "TWILIO_AUTH_TOKEN is your Auth Token (not SID).",
                    base or "(not set)",
                )
                return Response(content="Forbidden", status_code=403)
        except Exception as e:
            logger.exception("Twilio validation error: %s", e)
            return Response(content="Forbidden", status_code=403)

    stream_url = _twilio_stream_url()
    logger.info("Twilio voice incoming: returning TwiML with Stream url=%s", stream_url)
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="{stream_url}"/></Connect></Response>'
    )
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """Twilio Media Stream WebSocket: receive inbound μ-law, run ASR → chat → TTS, send media back."""
    logger.info("Twilio Media Stream WebSocket: handshake starting")
    await websocket.accept()
    logger.info("Twilio Media Stream WebSocket: handshake complete")
    stt = get_stt()
    tts = get_tts()
    if not stt.is_available() or not tts.is_available():
        await websocket.close(code=1011, reason="STT or TTS not available")
        return
    try:
        await handle_twilio_stream(websocket, stt, tts)
    except WebSocketDisconnect:
        logger.info("Twilio stream disconnected")
    except Exception as e:
        logger.exception("Twilio stream error: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# --- Chat: backend selected by CHAT_BACKEND (greig | fred) ---

def _chat_backend() -> str:
    """Return CHAT_BACKEND from env: 'greig' (OpenAI with tools), 'fred' (RAG), or 'passthru'. Default greig."""
    backend = (os.getenv("CHAT_BACKEND") or "greig").strip().lower()
    if backend not in ("greig", "fred", "passthru"):
        backend = "greig"
    return backend


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Chat endpoint. Backend is selected by env CHAT_BACKEND:
    - greig: OpenAI chat with end_conversation and web_search tools. Returns { message, done, end_conversation }.
    - fred: RAG-style (classify_intent → hybrid_search/get_knowledge_summary → generate_answer). Returns { answer, sources, intent } + X-Process-Time header.
    - passthru: POST to CHAT_PASSTHRU_URL with query + history; use only the returned answer. Returns { message, done, end_conversation }.
    """
    backend = _chat_backend()
    if backend == "fred":
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages required")
        last = req.messages[-1]
        if last.role != "user":
            raise HTTPException(status_code=400, detail="last message must be from user (query)")
        query = last.content
        history_list = [{"role": m.role, "content": m.content} for m in req.messages[:-1]]
        start = time.perf_counter()
        result = utils.process_query(query, history_list)
        elapsed = time.perf_counter() - start
        return JSONResponse(
            content=result,
            headers={"X-Process-Time": f"{elapsed:.3f}"},
        )
    if backend == "passthru":
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages required")
        last = req.messages[-1]
        if last.role != "user":
            raise HTTPException(status_code=400, detail="last message must be from user (query)")
        query = (last.content or "").strip()
        history_list = [{"role": (m.role or "user"), "content": (m.content or "")} for m in req.messages[:-1]]
        passthru_url = (os.getenv("CHAT_PASSTHRU_URL") or "http://localhost:8000/chat").strip()
        body = {"query": query, "history": history_list, "source": "voice"}

        def _do_passthru() -> dict:
            import urllib.request
            import json as _json
            data = _json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                passthru_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return _json.loads(resp.read().decode())
            except urllib.error.HTTPError as err:
                body_err = ""
                if err.fp:
                    try:
                        body_err = err.fp.read().decode()
                    except Exception:
                        pass
                err.remote_detail = body_err  # type: ignore[attr-defined]
                raise

        try:
            result = await asyncio.to_thread(_do_passthru)
            answer = result.get("answer") or ""
            return ChatResponse(
                message=ChatMessage(role="assistant", content=answer),
                done=True,
                end_conversation=False,
            )
        except Exception as e:
            remote_detail = getattr(e, "remote_detail", None)
            if remote_detail:
                logger.exception("Passthru chat error (remote 422 detail): %s", remote_detail)
                detail_msg = f"Passthru failed: remote returned validation error. Ensure remote expects body {{'query', 'history'}} (e.g. use CHAT_PASSTHRU_URL=.../api/chat/query if applicable). {str(remote_detail)[:400]}"
                raise HTTPException(status_code=502, detail=detail_msg)
            logger.exception("Passthru chat error: %s", e)
            raise HTTPException(status_code=502, detail=f"Passthru failed: {e!s}")

    # greig
    if req.integration == "amelia":
        raise HTTPException(
            status_code=501,
            detail="Amelia integration is Phase 2; use integration=openai for now.",
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        lang = (req.language or "en").strip().lower()
        if lang not in ("ja", "en"):
            lang = "en"
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        content, end_conversation = utils.run_openai_chat(client, messages, lang, verbosity=req.verbosity)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            done=True,
            end_conversation=end_conversation,
        )
    except Exception as e:
        logger.exception("OpenAI chat error")
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/chat/query", response_model=ChatQueryResponse)
async def chat_query(req: ChatQueryRequest):
    """
    RAG-style chat: query + history -> answer, sources, intent.
    Uses classify_intent, hybrid_search (or get_knowledge_summary for META), generate_answer.
    """
    start = time.perf_counter()
    history_list = [{"role": m.role, "content": m.content} for m in req.history]
    result = utils.process_query(req.query, history_list)
    elapsed = time.perf_counter() - start
    return JSONResponse(
        content=result,
        headers={"X-Process-Time": f"{elapsed:.3f}"},
    )


@app.post("/api/voice/end")
def voice_end():
    """Signal that the voice session should end (close listening). Called by the client when the agent returns end_conversation."""
    return {"ok": True, "message": "Voice session ended"}


@app.get("/api/voice/calls")
def list_voice_calls():
    """List voice calls (Twilio) with summary: call_sid, start_time, end_time, turn_count."""
    return voice_calls.get_calls()


@app.get("/api/voice/calls/{call_sid}")
def get_voice_call(call_sid: str):
    """Get one voice call with full transcript and per-turn STT/LLM/TTS latency."""
    rec = voice_calls.get_call(call_sid)
    if not rec:
        raise HTTPException(status_code=404, detail="Call not found")
    return rec


@app.websocket("/api/voice/calls/live")
async def voice_calls_live_ws(websocket: WebSocket):
    """WebSocket for real-time transcript updates. Send {"subscribe": "CA123"} first."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        call_sid = data.get("subscribe") or data.get("call_sid")
        if not call_sid:
            await websocket.close(code=4000, reason="Send {'subscribe': '<call_sid>'}")
            return
        voice_calls_live.subscribe(call_sid, websocket)
        voice_calls_live.ensure_consumer_started(asyncio.get_event_loop())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        voice_calls_live.unsubscribe(websocket)


# --- Serve built frontend (when frontend/dist exists) so one URL works behind a proxy ---
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")), name="frontend-assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        """Serve index.html for SPA client routes (e.g. /calls). Skip API/docs/assets."""
        if full_path.startswith(("api/", "voice/", "docs", "openapi.json", "redoc", "assets/")):
            raise HTTPException(status_code=404, detail="Not found")
        index_path = os.path.join(_FRONTEND_DIST, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
