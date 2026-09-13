"""
session.py
----------
FastAPI dependency: extract hoặc tạo mới session_id từ cookie.

Flow:
  Request có cookie "movienex_sid" → dùng giá trị đó.
  Request không có cookie          → tạo UUID mới, set-cookie vào response.

Dùng trong route handler:
  @router.get("/foo")
  def foo(session_id: str = Depends(get_session_id)):
      emit(EventType.SOMETHING, user_id=session_id, ...)
"""

import uuid
from fastapi import Cookie, Response


SESSION_COOKIE = "movienex_sid"
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30 ngày


def get_session_id(
    response: Response,
    movienex_sid: str | None = Cookie(default=None),
) -> str:
    """
    Trả về session_id từ cookie.
    Nếu chưa có, tạo UUID mới và set cookie vào response.
    """
    if movienex_sid:
        return movienex_sid

    new_sid = str(uuid.uuid4())
    response.set_cookie(
        key=SESSION_COOKIE,
        value=new_sid,
        max_age=COOKIE_MAX_AGE,
        httponly=False,   # False để JS đọc được nếu cần
        samesite="lax",
    )
    return new_sid
