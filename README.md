# Amelia Voice

Voice conversation app (Phase 1): Google ASR and TTS, OpenAI chat.

## Prerequisites

- **Backend:** Python 3.11+, [uv](https://docs.astral.sh/uv/) (if `uv` not found: `task install-uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh` then add `~/.local/bin` to PATH)
- **Frontend:** Node 18+
- **Task:** [Task](https://taskfile.dev/) (optional; all operations can be run via Task)
- **Env:** Create `.env` in project root with:
  - `OPENAI_API_KEY` – for chat
  - `GOOGLE_APPLICATION_CREDENTIALS` – path to Google service account JSON
  - `GOOGLE_CLOUD_PROJECT` – Google Cloud project ID (or `GOOGLE_PROJECT`)
  - **Twilio (Phase 3):** `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_VOICE_WEBHOOK_URL` (public base URL, e.g. `https://xxx.ngrok.io`), `TWILIO_AI=openai`, `TWILIO_LANGUAGE=ja`
  - **STT (optional):** `STT_PROVIDER=google` (default) or `STT_PROVIDER=whisper` – Whisper (OpenAI) often improves accuracy for Japanese; uses `OPENAI_API_KEY`
  - **Voice prompt (optional):** `VOICE_VERBOSITY=brief|normal|detailed` (default: `normal`) – controls how verbose the LLM is for voice (web and Twilio). Affects only the chat model’s system prompt; does not change TTS, playback, or other voice behavior. `VOICE_PROMPT_TEMPLATE` – optional full system prompt; use placeholders `{language_instruction}` and `{verbosity_instruction}`.
  - **Web search (optional):** `TAVILY_API_KEY` – when set, the agent can use the web_search tool (Tavily) to look up current information; used for both text and voice chat.

## Operations (Task)

All operations are available via [Taskfile.yaml](Taskfile.yaml). From project root:

```bash
task --list              # List tasks
task install             # Install backend + frontend deps
task install-backend     # uv sync
task install-frontend    # npm install in frontend
task backend            # Run FastAPI on :8000
task frontend           # Run Vite dev server on :5173
task dev                # Run backend and frontend in parallel
task build              # Build frontend for production
task lint-frontend      # Lint frontend
task check              # install-backend + build (sanity check)
task webhook            # expose backend via ngrok (for Twilio; run in second terminal)
```

## Manual run (without Task)

**Backend** (from project root):

```bash
uv sync
uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: http://localhost:8000/api/health
- API docs: http://localhost:8000/docs

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

- App: http://localhost:5173 (proxies `/api` to backend)

## Phase 1 features

- **Left sidebar:** Domain, Integration (OpenAI / Amelia stub), Language (EN / JA), Voice model (Whisper V2/V3 stubbed, Google)
- **Right:** Chat history (agent left, user right), text input, “Start voice conversation” button
- **Voice:** Auto-detect start/stop (silence-based). Records → Google ASR → OpenAI chat → Google TTS playback
- **i18n:** English and Japanese

Phase 2 will add Amelia integration.

## Phase 3: Twilio (phone calls)

Incoming calls to a Twilio number can talk to the same bot (Google ASR → OpenAI → Google TTS). Bot runs in **continuous mode** (multiple turns until hang up). See [twilio_integration.md](twilio_integration.md) for the full startup procedure (task webhook first to get URL, then task dev, Twilio Console, test call).

- **Webhook:** Configure the Twilio number’s Voice webhook to `https://your-host/voice/incoming` (POST).
- **Local testing:** Run `task webhook` first (ngrok), set `TWILIO_VOICE_WEBHOOK_URL` in `.env` to the ngrok https URL, then run `task dev`; configure the voice webhook in Twilio Console (Phone Numbers → your number → Voice → A CALL COMES IN).
- **Env:** `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_VOICE_WEBHOOK_URL`, `TWILIO_AI`, `TWILIO_LANGUAGE`.
