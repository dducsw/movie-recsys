from fastapi import APIRouter, Query, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from app.services.recsys import RecsysService
from app.views.schemas import RecommendationResponse
from app.services.event_producer import emit, EventType
from app.services.session import get_session_id
from app.services.auth_service import get_current_user_optional
from app.models.user import UserModel

router = APIRouter(prefix="/api", tags=["Recommendation System"])

@router.get("/recommendations", response_model=RecommendationResponse)
def get_personalized_recommendations(
    background_tasks: BackgroundTasks,
    movie_ids: Optional[str] = Query(None, description="Comma-separated list of liked movie IDs"),
    session_id: str = Depends(get_session_id),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Generate personalized recommendations based on liked movie IDs and User Profile Store.
    """
    liked_ids = []
    if movie_ids:
        try:
            liked_ids = [int(x.strip()) for x in movie_ids.split(",") if x.strip()]
        except ValueError:
            pass

    # Mix in user preference profile if logged in
    user_identifier = session_id
    if current_user:
        user_identifier = str(current_user["id"])
        try:
            user_prefs = UserModel.get_user_preferences(current_user["id"])
            pref_ids = user_prefs.get("favorite_movie_ids", [])
            for pid in pref_ids:
                if pid not in liked_ids:
                    liked_ids.append(pid)
        except Exception as e:
            print(f"[Warning] Could not fetch user preferences: {e}")

    recommendations = RecsysService.get_personalized_recommendations(liked_ids)
    background_tasks.add_task(
        emit,
        EventType.RECOMMENDATION_REQUEST,
        user_id=user_identifier,
        extra={"liked_movie_ids": liked_ids, "result_count": len(recommendations)},
    )
    return {"results": recommendations}

@router.get("/movies/{movie_id}/recommendations", response_model=RecommendationResponse)
def get_movie_recommendations(
    movie_id: int,
    background_tasks: BackgroundTasks,
    limit: int = Query(default=12, ge=1, le=50),
    session_id: str = Depends(get_session_id),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Get similar movies for a specific movie details page.
    """
    user_identifier = str(current_user["id"]) if current_user else session_id
    similar_movies = RecsysService.get_similar_movies(movie_id, limit)
    background_tasks.add_task(
        emit,
        EventType.SIMILAR_MOVIE_REQUEST,
        user_id=user_identifier,
        movie_id=movie_id,
        extra={"limit": limit},
    )
    return {"results": similar_movies}

class ChatMessageRequest(BaseModel):
    message: str

@router.post("/chatbot/chat")
def chat_with_bot(request: ChatMessageRequest):
    from app.services.chatbot import ChatbotService
    return ChatbotService.get_reply(request.message)
