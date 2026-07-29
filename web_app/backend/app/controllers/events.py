"""
events.py — Controller cho interaction tracking từ frontend.
Bao gồm: Click, Impression, Star Rating, Watchlist.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from app.services.event_producer import emit, EventType
from app.services.session import get_session_id
from app.services.auth_service import get_current_user_optional
from app.models.user import UserModel

router = APIRouter(prefix="/api/events", tags=["Events"])


class ClickEvent(BaseModel):
    movie_id: int
    source: str           # "trending" | "latest" | "search" | "recommendations" | "similar" | "chatbot"
    position: Optional[int] = None   # vị trí trong listing (0-indexed)


class ImpressionItem(BaseModel):
    movie_id: int
    source: str
    position: Optional[int] = None


class ImpressionBatchRequest(BaseModel):
    impressions: List[ImpressionItem]


class RatingEvent(BaseModel):
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0)


class WatchlistEvent(BaseModel):
    movie_id: int


@router.post("/click", status_code=204)
def track_click(
    body: ClickEvent,
    session_id: str = Depends(get_session_id),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Ghi nhận click event từ frontend."""
    user_identifier = str(current_user["id"]) if current_user else session_id
    emit(
        EventType.CLICK,
        user_id=user_identifier,
        movie_id=body.movie_id,
        extra={"source": body.source, "position": body.position},
    )


@router.post("/impression", status_code=204)
def track_impression(
    body: ImpressionBatchRequest,
    session_id: str = Depends(get_session_id),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Ghi nhận danh sách các card phim xuất hiện trong viewport (Impression Tracking)."""
    user_identifier = str(current_user["id"]) if current_user else session_id
    for item in body.impressions[:50]:
        emit(
            EventType.IMPRESSION,
            user_id=user_identifier,
            movie_id=item.movie_id,
            extra={"source": item.source, "position": item.position},
        )


@router.post("/rating")
def track_rating(
    body: RatingEvent,
    session_id: str = Depends(get_session_id),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Ghi nhận đánh giá 1-5 sao của user. Lưu DB PostgreSQL nếu đã đăng nhập & emit Kafka."""
    user_identifier = str(current_user["id"]) if current_user else session_id
    
    saved_rating = None
    if current_user:
        try:
            saved_rating = UserModel.save_rating(current_user["id"], body.movie_id, body.rating)
        except Exception as e:
            print(f"[Warning] Could not save rating to PostgreSQL DB: {e}")

    emit(
        EventType.RATING,
        user_id=user_identifier,
        movie_id=body.movie_id,
        extra={"rating": body.rating},
    )
    return {"message": "Rating recorded successfully", "rating": body.rating, "data": saved_rating}


@router.post("/watchlist")
def track_watchlist(
    body: WatchlistEvent,
    session_id: str = Depends(get_session_id),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Toggle phim trong danh sách Watchlist của user."""
    user_identifier = str(current_user["id"]) if current_user else session_id
    is_in_watchlist = False
    if current_user:
        try:
            is_in_watchlist = UserModel.toggle_watchlist(current_user["id"], body.movie_id)
        except Exception as e:
            print(f"[Warning] Could not update watchlist in DB: {e}")

    emit(
        EventType.WATCHLIST,
        user_id=user_identifier,
        movie_id=body.movie_id,
        extra={"in_watchlist": is_in_watchlist},
    )
    return {"message": "Watchlist updated", "movie_id": body.movie_id, "in_watchlist": is_in_watchlist}
