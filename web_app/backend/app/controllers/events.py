"""
events.py — controller cho click tracking từ frontend.

Frontend gọi POST /api/events/click ngay khi user click vào movie card
trước khi navigate vào detail page. Backend emit event "click" lên Kafka
với đúng source listing và position — align với simulator.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from app.services.event_producer import emit, EventType
from app.services.session import get_session_id

router = APIRouter(prefix="/api/events", tags=["Events"])


class ClickEvent(BaseModel):
    movie_id: int
    source: str           # "trending" | "latest" | "search" | "recommendations" | "similar" | "chatbot"
    position: Optional[int] = None   # vị trí trong listing (0-indexed), None nếu không biết


@router.post("/click", status_code=204)
def track_click(
    body: ClickEvent,
    session_id: str = Depends(get_session_id),
):
    """
    Nhận click event từ frontend khi user click vào movie card.
    Emit event "click" lên Kafka — align với simulator's click event.

    extra fields:
      source:   listing nào phát sinh click (trending, search, ...)
      position: vị trí card trong listing (để tính position bias)
    """
    emit(
        EventType.CLICK,
        user_id=session_id,
        movie_id=body.movie_id,
        extra={"source": body.source, "position": body.position},
    )
