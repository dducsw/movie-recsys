"""
session.py
----------
Extract or create session_id from cookie. Validates format to avoid fixation.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import Cookie, Response

logger = logging.getLogger(__name__)

SESSION_COOKIE = "movienex_sid"
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30 days


def _is_valid_sid(value: str) -> bool:
    try:
        uuid.UUID(str(value), version=4)
        return len(str(value)) == 36
    except (ValueError, TypeError):
        return False


def get_session_id(
    response: Response,
    movienex_sid: str | None = Cookie(default=None),
) -> str:
    """Return session_id from cookie, or create a new UUID."""
    if movienex_sid and _is_valid_sid(movienex_sid):
        return movienex_sid
    if movienex_sid:
        logger.debug("Invalid session id rejected, issuing new one.")
    new_sid = str(uuid.uuid4())
    response.set_cookie(
        key=SESSION_COOKIE,
        value=new_sid,
        max_age=COOKIE_MAX_AGE,
        httponly=False,  # JS needs to read it for X-Session-Id header
        samesite="lax",
    )
    return new_sid
