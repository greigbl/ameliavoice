"""
JWT cookie auth. Credentials load from AUTH_USERS_FILE (JSON, not in repo).
Set AUTH_JWT_SECRET to enable protection on /api/* (except login/logout).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import jwt

logger = logging.getLogger(__name__)

COOKIE_NAME = "ameliavoice_token"
JWT_ALG = "HS256"
# Default session length: 48h. Override with AUTH_JWT_EXPIRE_HOURS or AUTH_JWT_EXPIRE_DAYS.
DEFAULT_EXPIRE_HOURS = 48


def jwt_secret() -> str:
    return (os.getenv("AUTH_JWT_SECRET") or "").strip()


def auth_enabled() -> bool:
    return bool(jwt_secret())


def jwt_ttl_seconds() -> int:
    """JWT lifetime and cookie max-age. AUTH_JWT_EXPIRE_DAYS wins if set; else AUTH_JWT_EXPIRE_HOURS (default 48)."""
    days_raw = (os.getenv("AUTH_JWT_EXPIRE_DAYS") or "").strip()
    if days_raw:
        return max(3600, int(days_raw) * 86400)
    hours = int((os.getenv("AUTH_JWT_EXPIRE_HOURS") or str(DEFAULT_EXPIRE_HOURS)).strip())
    return max(3600, hours * 3600)


def repo_root() -> Path:
    """Project root (directory that contains `backend/`)."""
    return Path(__file__).resolve().parent.parent


def _users_path() -> Optional[Path]:
    """
    Path to the JSON user file. Relative paths are resolved from repo_root() so
    AUTH_USERS_FILE=./auth_users.json works regardless of process cwd.

    If AUTH_USERS_FILE is unset, uses repo_root()/auth_users.json when that file exists.
    """
    raw = (os.getenv("AUTH_USERS_FILE") or "").strip()
    root = repo_root()
    if not raw:
        default = root / "auth_users.json"
        return default if default.is_file() else None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = root / p
    return p


def load_user_table() -> dict[str, dict[str, str]]:
    """
    Returns username -> {"password": ..., "client_id": ...}.
    JSON shape: {"users": [{"username","password","client_id"}, ...]}
    or {"<username>": {"password","client_id"}, ...}
    """
    path = _users_path()
    if path is None:
        logger.warning(
            "No auth user file: set AUTH_USERS_FILE or add auth_users.json at the project root "
            "(see auth_users.example.json)"
        )
        return {}
    if not path.is_file():
        logger.warning("AUTH_USERS_FILE does not exist: %s", path)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to read AUTH_USERS_FILE: %s", e)
        return {}

    out: dict[str, dict[str, str]] = {}
    if isinstance(data, dict) and "users" in data:
        for row in data.get("users") or []:
            if not isinstance(row, dict):
                continue
            u = (row.get("username") or "").strip()
            if not u:
                continue
            out[u] = {
                "password": str(row.get("password") or "").strip(),
                "client_id": str(row.get("client_id") or "").strip(),
            }
    elif isinstance(data, dict):
        for u, row in data.items():
            if not isinstance(row, dict) or not str(u).strip():
                continue
            out[str(u).strip()] = {
                "password": str(row.get("password") or "").strip(),
                "client_id": str(row.get("client_id") or "").strip(),
            }
    return out


def match_credentials(username: str, password: str, table: dict[str, dict[str, str]]) -> Optional[str]:
    """Return client_id if username/password match a row in table, else None."""
    u = (username or "").strip()
    pw = (password or "").strip()
    if not u:
        return None
    row = table.get(u)
    if not row:
        return None
    if (row.get("password") or "").strip() != pw:
        return None
    cid = (row.get("client_id") or "").strip()
    return cid if cid else None


def verify_login(username: str, password: str) -> Optional[str]:
    """Return client_id if credentials match, else None."""
    return match_credentials(username, password, load_user_table())


def create_token(username: str, client_id: str) -> str:
    now = int(time.time())
    exp = now + jwt_ttl_seconds()
    payload = {
        "sub": username,
        "client_id": client_id,
        "iat": now,
        "exp": exp,
    }
    return jwt.encode(payload, jwt_secret(), algorithm=JWT_ALG)


def decode_token(token: Optional[str]) -> Optional[dict[str, Any]]:
    if not token or not jwt_secret():
        return None
    try:
        return jwt.decode(token, jwt_secret(), algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        return None


def payload_from_request(request) -> Optional[dict[str, Any]]:
    token = request.cookies.get(COOKIE_NAME)
    return decode_token(token)


def payload_from_websocket(websocket) -> Optional[dict[str, Any]]:
    token = websocket.cookies.get(COOKIE_NAME)
    return decode_token(token)


def cookie_secure() -> bool:
    return (os.getenv("AUTH_COOKIE_SECURE") or "").strip().lower() in ("1", "true", "yes")


def attach_auth_cookie(response, token: str) -> None:
    max_age = jwt_ttl_seconds()
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=cookie_secure(),
        path="/",
    )


def clear_auth_cookie(response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/")
