"""
Real-time broadcast for voice call transcript: STT/LLM/TTS events via WebSocket.
Sync pipeline calls emit(); async consumer broadcasts to subscribed clients.
"""
import asyncio
import logging
import queue
import threading
from typing import Any

logger = logging.getLogger(__name__)

_event_queue: queue.Queue = queue.Queue()
_subscribers: dict[str, set[Any]] = {}
_subscribers_lock = threading.Lock()
_consumer_task: asyncio.Task | None = None


def emit(call_sid: str, event: str, payload: dict[str, Any]) -> None:
    """Called from sync pipeline (thread-safe). Queues event for broadcast."""
    _event_queue.put((call_sid, event, payload))


def subscribe(call_sid: str, websocket: Any) -> None:
    """Add WebSocket to subscribers for this call_sid."""
    with _subscribers_lock:
        _subscribers.setdefault(call_sid, set()).add(websocket)
    logger.debug("[voice_calls_live] subscribe call_sid=%s", call_sid)


def unsubscribe(websocket: Any) -> None:
    """Remove WebSocket from all call_sids."""
    to_remove = []
    with _subscribers_lock:
        for call_sid, s in list(_subscribers.items()):
            s.discard(websocket)
            if not s:
                to_remove.append(call_sid)
        for k in to_remove:
            del _subscribers[k]
    logger.debug("[voice_calls_live] unsubscribe")


async def _consume_events(loop: asyncio.AbstractEventLoop) -> None:
    """Run in background: get events from queue and broadcast to subscribers."""
    while True:
        try:
            item = await loop.run_in_executor(None, _event_queue.get)
        except asyncio.CancelledError:
            break
        call_sid, event, payload = item
        msg = {"call_sid": call_sid, "event": event, "payload": payload}
        with _subscribers_lock:
            wss = list(_subscribers.get(call_sid, []))
        dead = []
        for ws in wss:
            try:
                await ws.send_json(msg)
            except Exception as e:
                logger.debug("voice_calls_live send error: %s", e)
                dead.append(ws)
        for ws in dead:
            unsubscribe(ws)


def ensure_consumer_started(loop: asyncio.AbstractEventLoop) -> None:
    """Start the consumer task if not already running."""
    global _consumer_task
    if _consumer_task is None or _consumer_task.done():
        _consumer_task = loop.create_task(_consume_events(loop))
        logger.debug("[voice_calls_live] consumer started")