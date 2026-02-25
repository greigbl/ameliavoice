# Twilio Integration Design (Phase 3)

## 1. Goal

Expose the existing Amelia Voice bot (Google ASR → OpenAI/Amelia → Google TTS) over **Twilio Programmable Voice** so that users can have the same conversational experience via a **phone call**. A caller dials a Twilio number, and the same pipeline (speech-to-text, LLM, text-to-speech) runs in the cloud with audio flowing to and from the phone.

## 2. High-level options

| Approach | Pros | Cons |
|----------|------|-----|
| **Twilio Media Streams** (WebSocket) | Real-time, low latency, natural turn-taking; one WebSocket per call with bidirectional audio. | Requires WebSocket server, audio format conversion (mulaw ↔ our pipeline). |
| **TwiML + Record/Play** | Simple: answer → `<Record>` → POST to our API → we return TwiML with `<Play>` URL. | High latency (full utterance before processing), less natural (push-to-talk style). |

**Recommendation:** Use **Twilio Media Streams** so the experience is close to the current web app (continuous or pause-based turn-taking, quick replies).

---

## 3. Architecture (Media Streams)

```
Caller (phone)  ←→  Twilio  ←→  Our backend (FastAPI)
                            WebSocket (Media Stream)
                            - In:  base64 mulaw 8kHz
                            - Out: base64 mulaw 8kHz

Our backend per call:
  1. Receive mulaw chunks from Twilio over WebSocket.
  2. Buffer / VAD → form “utterance” (e.g. silence-based or time-based).
  3. Convert utterance to WAV 16kHz linear16 → existing /api/transcribe (Google ASR).
  4. Build messages (existing chat state) + new user text → existing /api/chat (OpenAI/Amelia).
  5. Existing /api/tts (Google TTS) → MP3 (or request linear16).
  6. Convert TTS output to 8kHz mulaw, stream base64 back to Twilio over same WebSocket.
  7. Repeat from step 1 for next turn (conversation history kept per call).
```

- **Incoming call flow**
  1. Call comes in to a Twilio number.
  2. Twilio sends HTTP request to our **voice webhook** (e.g. `POST /voice/incoming`).
  3. We respond with TwiML that connects the call to a **Media Stream** (WebSocket URL to our backend, e.g. `wss://our-host/voice/stream`).
  4. Twilio opens the WebSocket; we run the pipeline above for the lifetime of the call.

---

## 4. Audio format handling

| Stage | Format | Notes |
|-------|--------|-------|
| Twilio → us | 8 kHz, 8-bit μ-law (PCMU), base64 in Media Stream JSON | 20ms chunks typical (160 samples → 160 bytes). |
| Our ASR input (phone) | 8 kHz, 16-bit linear PCM (WAV) | Native 8kHz to Google STT (telephony; no upsampling). |
| Our TTS output | MP3 (or we could add linear16) | Existing Google TTS. |
| Us → Twilio | 8 kHz, 8-bit μ-law, base64 | Same as Twilio’s send format. |

**Conversions:**

- **Twilio → ASR (phone):** Decode base64 → μ-law bytes → convert to 16-bit linear PCM at 8 kHz (no resampling). Write 8 kHz WAV and call transcribe with `sample_rate_hertz=8000` so Google gets native telephony rate.
- **TTS → Twilio:** Get audio from TTS (MP3 or linear16). Decode to PCM, resample to 8 kHz if needed, convert to μ-law, encode base64, send in Media Stream “media” payloads.

Libraries: `pydub` (already in use), `audioop` (μ-law; stdlib in Python), or `numpy`/`scipy` for resampling.

---

## 5. Backend components to add

### 5.1 Configuration (implemented)

- **Env (e.g. `.env`):**
  - `TWILIO_ACCOUNT_SID` – Twilio account SID
  - `TWILIO_AUTH_TOKEN` – used for webhook signature validation
  - `TWILIO_VOICE_WEBHOOK_URL` – public base URL (e.g. `https://xxx.ngrok.io`); used to build the WebSocket URL in TwiML and for signature validation
  - `TWILIO_AI` – `openai` or `amelia` (default: openai; amelia not yet supported in stream)
  - `TWILIO_LANGUAGE` – `ja` or `en` (default: ja); sets ASR/TTS language (ja-JP / en-US)
  - `STT_PROVIDER` – `google` (default) or `whisper`; Whisper (OpenAI API) can improve accuracy for Japanese and uses the same `OPENAI_API_KEY` as chat.

### 5.2 New HTTP endpoints

- **`POST /voice/incoming`** (Twilio “voice” webhook)
  - Input: Twilio form params (e.g. `CallSid`, `From`, `To`).
  - Validate `X-Twilio-Signature` using `TWILIO_AUTH_TOKEN` and request URL/body.
  - Response: TwiML that starts a Media Stream to our WebSocket URL, e.g.:
    - `<Response><Connect><Stream url="wss://our-host/voice/stream"/></Connect></Response>`
  - Optional: append query params to `url` (e.g. `?lang=ja`) and parse them in the WebSocket.

### 5.3 WebSocket endpoint

- **`/voice/stream`** (or `GET /voice/stream` upgraded to WebSocket)
  - Twilio connects here when the call is answered (TwiML `<Stream>`).
  - Twilio sends JSON messages; we care about:
    - `start`: call metadata (CallSid, etc.).
    - `media`: `payload` = base64 μ-law audio chunk.
    - `stop`: call ended.
  - We send JSON messages with `event: "media"`, `payload: <base64 mulaw>`, `sequenceNumber` for playback.
  - One long-lived WebSocket per call; one “conversation” (message history) per `CallSid`.

### 5.4 Per-call state and pipeline

- **State:** In-memory (or Redis later) keyed by `CallSid`:
  - `messages: list[ChatMessage]` (same as current chat).
  - Optional: language, integration (openai/amelia), ASR/TTS settings.
- **Pipeline (pseudo):**
  1. Append incoming `media` payloads to an in-memory buffer (μ-law).
  2. Run VAD or simple silence detection on the buffer (or fixed-length chunks); when an “end of utterance” is detected, take the segment.
  3. Convert segment to 16 kHz WAV → call existing STT service (or internal transcribe); get text.
  4. If text is empty, go back to step 1.
  5. Append `{ role: "user", content: text }` to `messages`; call existing chat API; append assistant message.
  6. Call existing TTS with assistant text (and optional markdown stripping); get audio.
  7. Convert TTS audio to 8 kHz μ-law; send base64 chunks in `media` events to Twilio.
  8. When playback is done, go back to step 1 (listen for next utterance).

Reuse existing backend services: `GoogleSTTService`, `TTSService`, and the same chat/OpenAI (and later Amelia) logic as the web app.

---

## 6. Security and production

- **Webhook validation:** Always validate `X-Twilio-Signature` for `/voice/incoming` (and any other Twilio webhooks) using `TWILIO_AUTH_TOKEN` and the full request URL and body.
- **HTTPS/WSS:** Twilio requires public HTTPS for the voice URL and public WSS for the Media Stream URL. Use a reverse proxy (e.g. nginx, Caddy) or a tunnel (e.g. ngrok) for local testing.
- **Secrets:** Keep `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` in env only; do not commit.

---

## 7. Startup procedure (local testing)

1. Buy or use a Twilio phone number.
2. Configure the number’s “Voice” webhook to: `https://our-host/voice/incoming` (HTTP POST).
3. Ensure your backend is reachable at that URL and at `wss://your-host/voice/stream`.

The bot is in **continuous mode**: multiple turns until the caller hangs up (speak → pause → bot replies → speak again → …).

**Step-by-step:**

1. **Terminal 1:** Run `task webhook` to start ngrok; copy the **https** URL (e.g. `https://abc123.ngrok.io`).
2. In `.env`, set `TWILIO_VOICE_WEBHOOK_URL=https://YOUR-NGROK-URL` (no trailing slash).
3. **Terminal 2:** Run `task dev` (backend on 8000, frontend on 5173).
4. In **Twilio Console → Phone Numbers → Manage → Active Numbers** → click your number → **Voice** section → **A CALL COMES IN**: set webhook to `https://YOUR-NGROK-URL/voice/incoming` (must include `/voice/incoming` — if you use only the base URL you’ll get 404). Method: **HTTP POST**. Save.
5. Call your Twilio number; speak, pause ~1s; bot replies. Multiple turns until you hang up (continuous mode).

**Twilio Console page:** [Phone Numbers → Manage → Active Numbers](https://console.twilio.com/us1/develop/phone-numbers/manage/incoming) — click your number, then under **Voice** set **A CALL COMES IN** to your webhook URL.

**If you get 403 Forbidden:** Twilio’s signature check failed. Ensure:
- `TWILIO_VOICE_WEBHOOK_URL` is exactly the base Twilio uses (e.g. `https://abc123.ngrok.io` with **https**, no trailing slash).
- `TWILIO_AUTH_TOKEN` is your [Auth Token](https://console.twilio.com) (not Account SID), with no extra spaces/quotes in `.env`.
- The webhook URL in Twilio is exactly `https://YOUR-NGROK-URL/voice/incoming` (same as `TWILIO_VOICE_WEBHOOK_URL` + `/voice/incoming`).
- For local debugging only you can set `TWILIO_SKIP_VALIDATION=1` in `.env` to skip the check (do not use in production).

**If you get Twilio error 31920 (Stream - WebSocket - Handshake Error):** Twilio could not complete the WebSocket handshake (server must return HTTP 101). Common causes:
- **Wrong stream URL:** The backend builds the Media Stream URL from `TWILIO_VOICE_WEBHOOK_URL` using only the **origin** (scheme + host, no path). So set `TWILIO_VOICE_WEBHOOK_URL=https://YOUR-NGROK-URL` (no path). The stream URL will be `wss://YOUR-NGROK-URL/voice/stream`. If you had a path (e.g. `/voice/incoming`) in the env, the stream URL would have been wrong and Twilio would get 404 → handshake error.
- **Tunnel/proxy:** Ensure your tunnel (ngrok, etc.) supports WebSocket and forwards `Upgrade` and `Connection` headers. Check backend logs: you should see "Twilio Media Stream WebSocket: handshake starting" when Twilio connects; if not, the WebSocket request is not reaching the server.

---

## 8. Twilio console (reference)

- **Page:** Twilio Console → **Phone Numbers** → **Manage** → **Active Numbers** → [your number].
- Under **Voice**, set **A CALL COMES IN** to: `https://your-host/voice/incoming` (HTTP POST).
- Backend must be reachable at that URL and at `wss://your-host/voice/stream`.

---

## 9. Out of scope / later

- **Outbound calls:** Design above is for incoming only; outbound can re-use the same Media Stream and pipeline with a different TwiML entry point.
- **Amelia vs OpenAI:** Can be selected per number (e.g. different webhook URLs or query params) or via a single default; no change to the Media Stream protocol.
- **Persistence:** First version can keep `messages` in memory per CallSid; later, add Redis or DB for history and analytics.
- **Language selection:** Can be fixed (e.g. EN/JA from env) or passed in TwiML/query params and stored in per-call state for ASR/TTS and optional prompts.

---

## 10. Implementation checklist (Phase 3)

- [x] Add `twilio` (and `audioop` via stdlib, `pydub`) deps to `pyproject.toml`.
- [x] Implement `POST /voice/incoming` with signature validation and TwiML `<Stream>` response.
- [x] Implement WebSocket `/voice/stream`: parse `connected`/`start`/`media`/`stop`, maintain buffer and per-call state.
- [x] Add μ-law ↔ linear PCM and 8 kHz ↔ 16 kHz conversion (and TTS MP3 → mulaw 8 kHz path) in `backend/twilio_audio.py`.
- [x] Integrate buffer + simple silence-based VAD → call existing transcribe.
- [x] Wire transcribe output → existing chat (OpenAI) → existing TTS; append to per-call `messages`.
- [x] Convert TTS output to mulaw and send `media` (and optional `mark`) events back to Twilio.
- [x] Document env vars and Twilio console steps in README and this doc.

This design reuses the existing bot (ASR, chat, TTS) and adds a thin Twilio-facing layer (webhook + WebSocket + audio conversion) so the same bot is available over the phone.
