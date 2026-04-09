#!/usr/bin/env python3
"""
Mock passthru server for local testing.

Listens on http://127.0.0.1:8000/api/chat (POST), logs the JSON body, returns a minimal
answer shape expected by CHAT_BACKEND=passthru (answer / sources / intent).

Run (from repo root, with uv):
  uv run python test_server.py

Point the main app at this server:
  CHAT_PASSTHRU_URL=http://127.0.0.1:8000/api/chat
  CHAT_BACKEND=passthru

Keep Amelia Voice on another port (e.g. 8080) so this can bind to 8000.
"""
from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Request
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("passthru_test")

app = FastAPI(title="Passthru test server", version="0.1.0")


@app.post("/api/chat")
async def passthru_chat(request: Request):
    body = await request.json()
    logger.info("Passthru payload:\n%s", json.dumps(body, ensure_ascii=False, indent=2))
    q = (body.get("query") or "")[:80]
    return {
        "answer": f"[test_server] Received query (truncated): {q!r}",
        "sources": [],
        "intent": "TEST",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
