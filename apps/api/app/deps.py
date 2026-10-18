"""Shared FastAPI dependencies: pagination + cursor + ownership guard."""

from __future__ import annotations

import base64

from fastapi import Depends, HTTPException, Query

from app.services.auth_service import get_current_user


def pagination(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
) -> tuple[int, int]:
    """Simple offset pagination. Keep small and explicit."""
    return page, limit


def require_owner_or_admin(
    user_id: int,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """BOLA guard: user can only access their own resource (or admin)."""
    is_admin = current_user.get("role") == "admin"
    if not is_admin and current_user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return current_user


def encode_cursor(popularity: float, movie_id: int) -> str:
    """Opaque cursor for keyset pagination: base64('pop:id')."""
    raw = f"{popularity}:{movie_id}".encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str | None) -> tuple[float, int] | None:
    """Decode cursor. Returns None if missing, raises 400 if malformed."""
    if not cursor:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        pop_str, mid_str = raw.split(":")
        return float(pop_str), int(mid_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid cursor.")
