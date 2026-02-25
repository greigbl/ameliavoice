"""
Voice conversation app - Phase 1: Google ASR and TTS, OpenAI chat.
"""
import asyncio
import json
import logging
import os
import tempfile
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
from fastapi.responses import Response
from pydantic import BaseModel

from backend.google_stt_service import GoogleSTTService
from backend.whisper_stt_service import WhisperSTTService
from backend.tts_service import TTSService
from backend.audio_utils import ensure_wav_format
from backend.twilio_stream import handle_twilio_stream
from backend import voice_calls
from backend import voice_calls_live
from backend.voice_prompt import build_voice_system_message

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


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    integration: str = "openai"  # "openai" | "amelia" (amelia stub for Phase 2)
    language: str = "en"  # "ja" | "en" for localized tool fallbacks (e.g. end_conversation default message)
    verbosity: str | None = None  # "brief" | "normal" | "detailed"; overrides VOICE_VERBOSITY when set (e.g. from web app)


class ChatResponse(BaseModel):
    message: ChatMessage
    done: bool = True
    end_conversation: bool = False  # agent invoked end_conversation tool; client should stop listening


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
    """Root has no handler. If Twilio is hitting POST /, set the voice webhook to /voice/incoming."""
    if request.method == "POST":
        logger.warning("Received POST / (likely Twilio). Set voice webhook to https://your-host/voice/incoming")
        return Response(
            content="Voice webhook is at /voice/incoming. Set Twilio A CALL COMES IN to: https://your-host/voice/incoming",
            status_code=404,
            media_type="text/plain",
        )
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


# --- Tavily web search (optional; requires TAVILY_API_KEY) ---

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use when the user asks about recent events, facts, or anything you need to look up. Provide a clear search query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to run (e.g. 'weather Tokyo today', 'latest news about X')"},
            },
            "required": ["query"],
        },
    },
}


def _tavily_search(query: str) -> str:
    """Run a Tavily web search and return a string summary for the model. Uses TAVILY_API_KEY."""
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return "Web search is not configured (TAVILY_API_KEY not set)."
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        # include_answer=True gives an LLM-generated answer; we also get results for context
        response = client.search(query=query, search_depth="basic", max_results=5, include_answer=True)
        parts = []
        if response.get("answer"):
            parts.append(response["answer"])
        results = response.get("results") or []
        if results:
            parts.append("Sources:")
            for r in results[:5]:
                title = r.get("title") or "No title"
                url = r.get("url") or ""
                content = (r.get("content") or "")[:500]
                parts.append(f"- {title} ({url}): {content}")
        return "\n\n".join(parts) if parts else "No results found."
    except Exception as e:
        logger.warning("Tavily search error: %s", e)
        return f"Web search failed: {e!s}"


# Localized tool descriptions. Be conservative: speech recognition can mishear; only call when intent is clearly to end.
END_CONVERSATION_TOOL_DESCRIPTIONS = {
    "ja": "Call this ONLY when the user unambiguously says they want to end (e.g. さようなら、ありがとうございました、以上です、もう結構です). Speech recognition can mishear: if the phrase is short or could be a misheard question, do NOT call this—respond normally. When calling: reply with a brief thank you in Japanese, then call this tool.",
    "en": "Call this ONLY when the user unambiguously says they want to end (e.g. goodbye, that's all for now, I'm done thanks, no more questions). Speech recognition can mishear: if the phrase is short or could be a misheard question, do NOT call this—respond normally. When calling: reply with a brief thank you in English, then call this tool.",
}


def _end_conversation_tool(lang: str):
    """Return the end_conversation tool with description in the given language."""
    desc = END_CONVERSATION_TOOL_DESCRIPTIONS.get(lang) or END_CONVERSATION_TOOL_DESCRIPTIONS["en"]
    return {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": desc,
            "parameters": {"type": "object", "properties": {}},
        },
    }


# Localized default when the model calls end_conversation but returns no content
END_CONVERSATION_DEFAULT = {
    "ja": "お話しできてありがとうございました。またね。",
    "en": "Thank you for talking with me. Goodbye!",
}

def _run_chat_with_tools(client, messages: list, lang: str) -> tuple[str, bool]:
    """Run chat completion with tools; handle tool_calls in a loop (max 5 rounds). Returns (final_content, end_conversation)."""
    tools = [_end_conversation_tool(lang), WEB_SEARCH_TOOL]
    end_conversation = False
    max_rounds = 5
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            if end_conversation and not content:
                content = END_CONVERSATION_DEFAULT.get(lang) or END_CONVERSATION_DEFAULT["en"]
            return content, end_conversation
        # Append assistant message with tool_calls
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        assistant_msg["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ]
        messages.append(assistant_msg)
        # Execute each tool and append results
        for tc in tool_calls:
            name = getattr(tc.function, "name", None) or ""
            args_str = getattr(tc.function, "arguments", None) or "{}"
            if name == "end_conversation":
                end_conversation = True
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
            elif name == "web_search":
                try:
                    args = json.loads(args_str)
                    query = args.get("query") or ""
                except Exception:
                    query = args_str
                result = _tavily_search(query)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Unknown tool."})
    # Max rounds reached; use last content
    if end_conversation and not content:
        content = END_CONVERSATION_DEFAULT.get(lang) or END_CONVERSATION_DEFAULT["en"]
    return content or "I'm sorry, I hit a limit. Please try again.", end_conversation


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat with OpenAI. Supports end_conversation and web_search (Tavily) tools."""
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
        #from openai import AzureOpenAI
        client = OpenAI(api_key=api_key)
        lang = (req.language or "en").strip().lower()
        if lang not in ("ja", "en"):
            lang = "en"
        system_content = build_voice_system_message(lang, verbosity=req.verbosity)
        messages = [{"role": "system", "content": system_content}] + [{"role": m.role, "content": m.content} for m in req.messages]
        content, end_conversation = _run_chat_with_tools(client, messages, lang)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            done=True,
            end_conversation=end_conversation,
        )
    except Exception as e:
        logger.exception("OpenAI chat error")
        raise HTTPException(status_code=502, detail=str(e))


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
