import logging
from typing import List, Optional

from fastapi import APIRouter, Query, Depends, BackgroundTasks, HTTPException
from app.services.recsys import RecsysService
from app.views.schemas import RecommendationResponse
from app.services.event_producer import emit, EventType
from app.services.session import get_session_id
from app.services.auth_service import get_current_user_optional
from app.deps import require_owner_or_admin
from app.models.user import UserModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Recommendation System"])

MAX_LIKED_IDS = 50


def _parse_liked_ids(raw: Optional[str]) -> List[int]:
    """Parse comma-separated ids. Raises 400 on bad input (no silent swallow)."""
    if not raw:
        return []
    try:
        ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="movie_ids must be comma-separated integers.")
    seen: List[int] = []
    for mid in ids:
        if mid not in seen:
            seen.append(mid)
        if len(seen) >= MAX_LIKED_IDS:
            break
    return seen


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    background_tasks: BackgroundTasks,
    movie_ids: Optional[str] = Query(default=None, description="Comma-separated liked movie IDs"),
    session_id: str = Depends(get_session_id),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Generate personalized recommendations from liked movie IDs (non-blocking)."""
    liked_ids = _parse_liked_ids(movie_ids)

    # Mix in user preference profile if logged in
    user_identifier = session_id
    if current_user:
        user_identifier = str(current_user["id"])
        try:
            user_prefs = UserModel.get_user_preferences(current_user["id"])
            for pid in user_prefs.get("favorite_movie_ids", []):
                if pid not in liked_ids and len(liked_ids) < MAX_LIKED_IDS:
                    liked_ids.append(pid)
        except Exception as e:
            logger.warning("Could not fetch user preferences: %s", e)

    user_id_int = current_user["id"] if current_user else None
    recommendations = await RecsysService.get_personalized_recommendations_async(
        liked_ids,
        user_id=user_id_int,
        session_id=session_id
    )
    background_tasks.add_task(
        emit,
        EventType.RECOMMENDATION_REQUEST,
        user_id=user_identifier,
        extra={"liked_movie_ids": liked_ids, "result_count": len(recommendations)},
    )
    return {"results": recommendations}

@router.get("/movies/{movie_id}/recommendations", response_model=RecommendationResponse)
async def get_movie_recommendations(
    movie_id: int,
    background_tasks: BackgroundTasks,
    limit: int = Query(default=12, ge=1, le=50),
    session_id: str = Depends(get_session_id),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Get similar movies for a specific movie details page (non-blocking).
    """
    user_identifier = str(current_user["id"]) if current_user else session_id
    similar_movies = await RecsysService.get_similar_movies_async(movie_id, limit)
    background_tasks.add_task(
        emit,
        EventType.SIMILAR_MOVIE_REQUEST,
        user_id=user_identifier,
        movie_id=movie_id,
        extra={"limit": limit},
    )
    return {"results": similar_movies}


@router.get("/recommendations/user/{user_id}", response_model=RecommendationResponse)
def get_user_ml_recommendations(
    user_id: int,
    limit: int = Query(default=20, ge=1, le=50),
    _owner: dict = Depends(require_owner_or_admin),
):
    """Personalized recs. BOLA-guarded: only owner or admin."""
    recs = RecsysService.get_user_recommendations(user_id=user_id, limit=limit)
    return {"results": recs}
