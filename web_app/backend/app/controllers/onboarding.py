import json
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from app.models.user import UserModel
from app.services.auth_service import get_current_user_optional
from app.services.recsys import get_redis_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/users", tags=["Onboarding"])


class OnboardingRequest(BaseModel):
    favorite_genres: List[str]
    favorite_movie_ids: List[int]


@router.post("/onboarding")
def save_user_onboarding(
    body: OnboardingRequest,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Lưu sở thích ban đầu khi người dùng mới hoàn thành Cold-start Onboarding.
    Lưu vào PostgreSQL và cập nhật Redis User Profile.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required for onboarding.")

    user_id = current_user["id"]
    try:
        # 1. Save to PostgreSQL
        saved_pref = UserModel.save_preferences(
            user_id,
            body.favorite_genres,
            body.favorite_movie_ids
        )

        # 2. Sync to Redis User Profile Store
        r = get_redis_client()
        if r:
            try:
                profile_key = f"user:{user_id}:profile"
                profile_data = {
                    "favorite_genres": body.favorite_genres,
                    "favorite_movie_ids": body.favorite_movie_ids,
                    "updated_at": saved_pref.get("updated_at")
                }
                r.set(profile_key, json.dumps(profile_data, default=str))
                logger.info(f"Updated Redis user profile for user_id={user_id}")
            except Exception as re:
                logger.warning(f"Failed to update Redis user profile: {re}")

        return {
            "message": "Onboarding preferences saved successfully",
            "preferences": {
                "favorite_genres": body.favorite_genres,
                "favorite_movie_ids": body.favorite_movie_ids
            }
        }
    except Exception as e:
        logger.error(f"Failed to save onboarding preferences for user_id={user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save onboarding preferences.")
