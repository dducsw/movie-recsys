import re
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, ConfigDict

from app.services.chatbot import ChatbotService
from app.services.event_producer import emit, EventType
from app.services.session import get_session_id
from app.services.auth_service import get_current_user_optional, require_admin
from app.config.security import limiter
from app.config.settings import get_settings
from app.models.user import UserModel

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/chatbot", tags=["Chatbot"])


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=2000, description="User prompt or movie question")
    session_id: Optional[str] = Field(default="default", max_length=100, description="Client session identifier")


class ChatResponse(BaseModel):
    text: str
    movies: List[Dict[str, Any]]
    session_id: str
    message_count: int


class HistoryResponse(BaseModel):
    messages: List[Dict[str, Any]]


class ClearResponse(BaseModel):
    status: str
    session_id: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movie_id: int
    liked: bool
    session_id: Optional[str] = Field(default="default", max_length=100)


class FeedbackResponse(BaseModel):
    movie_id: int
    liked: bool
    message: str


def validate_session_identifier(session_id: str, current_user: Optional[dict] = None) -> str:
    """
    Validates session identifier to prevent BOLA, IDOR, and injection.
    Restricts characters to safe alphanumeric, hyphen, and underscore.
    If session is prefixed with user_, enforces ownership match.
    """
    clean_id = session_id.strip()
    if not clean_id or not re.match(r"^[a-zA-Z0-9_\-]{1,100}$", clean_id):
        raise HTTPException(status_code=400, detail="Invalid session identifier format.")

    # If session is user-bound (e.g. user_42_...), ensure authenticated user owns it
    if clean_id.startswith("user_"):
        parts = clean_id.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            expected_user_id = int(parts[1])
            if not current_user or current_user["id"] != expected_user_id:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied: You do not have permission to view or manage this session."
                )

    return clean_id


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_CHATBOT)
def chat(
    request: Request,
    body: ChatRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Send a message and get personalized AI response with movie recommendations.
    """
    validated_sid = validate_session_identifier(body.session_id or "default", current_user)
    user_id = current_user["id"] if current_user else None
    user_liked_ids = []
    if user_id:
        try:
            watchlist = UserModel.get_user_watchlist(user_id) or []
            # get_user_watchlist returns List[int]; tolerate List[dict] too.
            user_liked_ids = [w.get("movie_id", w) if isinstance(w, dict) else w
                              for w in watchlist]
            user_liked_ids = [int(x) for x in user_liked_ids]
        except Exception:
            user_liked_ids = []
    if user_id:
        try:
            from app.services.chat_profile import get_profile

            profile = get_profile(user_id)
            for mid in profile.get("liked_ids", []):
                if mid not in user_liked_ids:
                    user_liked_ids.append(mid)
        except Exception:
            pass

    try:
        result = ChatbotService.get_reply(
            message=body.message,
            session_id=validated_sid,
            user_id=user_id,
            user_liked_ids=user_liked_ids
        )

        # Chat-to-RecSys Feedback Loop:
        # Sync top recommended / discussed movie IDs into nearline session state
        # so home feed immediately adapts to recent conversational discoveries.
        result_movies = result.get("movies", [])
        if result_movies:
            from app.services.event_producer import record_nearline_interaction
            top_rec_id = result_movies[0].get("movieId") or result_movies[0].get("id")
            if top_rec_id:
                try:
                    record_nearline_interaction(validated_sid, int(top_rec_id))
                    if user_id:
                        record_nearline_interaction(str(user_id), int(top_rec_id))
                except Exception:
                    pass

        emit(
            EventType.CHATBOT_MESSAGE,
            user_id=str(user_id) if user_id else (validated_sid or "anonymous"),
            extra={
                "message_len": len(body.message),
                "result_movie_count": len(result_movies),
                "message_count": result.get("message_count", 0),
            },
        )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chatbot processing error for session {validated_sid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat message. Please try again.")


@router.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(
    session_id: str = "default",
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Get conversation history for a session with ownership validation."""
    validated_sid = validate_session_identifier(session_id, current_user)
    messages = ChatbotService.get_history(validated_sid)
    return HistoryResponse(messages=messages)


@router.delete("/history/{session_id}", response_model=ClearResponse)
def clear_history(
    session_id: str = "default",
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Clear conversation history for a session with ownership validation."""
    validated_sid = validate_session_identifier(session_id, current_user)
    result = ChatbotService.clear_history(validated_sid)
    return ClearResponse(**result)


@router.post("/feedback", response_model=FeedbackResponse)
def chat_feedback(
    body: FeedbackRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Card like/dislike inside chat flows back into the user profile."""
    validated_sid = validate_session_identifier(body.session_id or "default",
                                                current_user)
    user_id = current_user["id"] if current_user else None
    movie = None
    try:
        from app.models.movie import MovieModel

        movie = MovieModel.get_by_id(body.movie_id)
    except Exception:
        movie = {"movieId": body.movie_id}
    try:
        from app.services.chat_profile import record_feedback

        record_feedback(user_id, movie or {"movieId": body.movie_id},
                        body.liked)
    except Exception:
        pass
    if user_id and body.liked:
        try:
            from app.models.user import UserModel as _UM

            _UM.toggle_watchlist(user_id, body.movie_id)
        except Exception:
            pass
    try:
        emit(EventType.CHATBOT_MESSAGE, user_id=str(user_id) if user_id else validated_sid,
             movie_id=body.movie_id, extra={"feedback": body.liked})
    except Exception:
        pass
    return FeedbackResponse(movie_id=body.movie_id, liked=body.liked,
                            message="Feedback recorded.")


@router.get("/session/{session_id}")
def get_session_info(
    session_id: str = "default",
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Get debugging info about a session (restricted in production to admins)."""
    if settings.is_production and (not current_user or current_user.get("role") != "admin"):
        raise HTTPException(status_code=403, detail="Session debugging info is disabled in production.")

    validated_sid = validate_session_identifier(session_id, current_user)
    return ChatbotService.get_session_info(validated_sid)
