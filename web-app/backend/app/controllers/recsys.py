from fastapi import APIRouter, Query
from pydantic import BaseModel
import random
from app.services.recsys import RecsysService
from app.views.schemas import RecommendationResponse
from typing import List, Optional

router = APIRouter(prefix="/api", tags=["Recommendation System"])

@router.get("/recommendations", response_model=RecommendationResponse)
def get_personalized_recommendations(
    movie_ids: Optional[str] = Query(None, description="Comma-separated list of liked movie IDs")
):
    """
    Generate personalized recommendations based on liked movie IDs.
    """
    liked_ids = []
    if movie_ids:
        try:
            liked_ids = [int(x.strip()) for x in movie_ids.split(",") if x.strip()]
        except ValueError:
            pass
            
    recommendations = RecsysService.get_personalized_recommendations(liked_ids)
    return {"results": recommendations}

@router.get("/movies/{movie_id}/recommendations", response_model=RecommendationResponse)
def get_movie_recommendations(movie_id: int, limit: int = Query(default=12, ge=1, le=50)):
    """
    Get similar movies for a specific movie details page.
    """
    similar_movies = RecsysService.get_similar_movies(movie_id, limit)
    return {"results": similar_movies}

class ChatMessageRequest(BaseModel):
    message: str

@router.post("/chatbot/chat")
def chat_with_bot(request: ChatMessageRequest):
    from app.services.chatbot import ChatbotService
    return ChatbotService.get_reply(request.message)
