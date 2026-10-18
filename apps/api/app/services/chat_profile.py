"""Long-term chat profile: liked genres + liked movies, Redis-backed."""

from __future__ import annotations

import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)

_KEY = "chat:profile:{uid}"
_TTL = 86400 * 90


def _redis():
    try:
        from app.services.redis_factory import get_redis

        return get_redis(db=0, timeout=0.5)
    except Exception:
        return None


def get_profile(user_id: int | None) -> dict:
    """Return {liked_genres: [...], liked_ids: [...]}. Empty when anonymous."""
    if not user_id:
        return {"liked_genres": [], "liked_ids": []}
    r = _redis()
    if not r:
        return {"liked_genres": [], "liked_ids": []}
    try:
        raw = r.get(_KEY.format(uid=user_id))
        if raw:
            data = json.loads(raw)
            return {"liked_genres": data.get("liked_genres", []),
                    "liked_ids": data.get("liked_ids", [])}
    except Exception as e:
        logger.debug("Profile read failed: %s", e)
    return {"liked_genres": [], "liked_ids": []}


def record_feedback(user_id: int | None, movie: dict | None,
                    liked: bool) -> None:
    """Update profile from a card like/dislike. Fail-open."""
    if not user_id or not movie:
        return
    r = _redis()
    if not r:
        return
    try:
        profile = get_profile(user_id)
        mid = movie.get("movieId") or movie.get("movieid")
        if mid and liked and mid not in profile["liked_ids"]:
            profile["liked_ids"] = (profile["liked_ids"] + [mid])[-200:]
        genres = str(movie.get("genres", "")).split("|") if liked else []
        counts = Counter(profile["liked_genres"])
        for g in genres:
            if g:
                counts[g] += 1
        profile["liked_genres"] = [g for g, _ in counts.most_common(10)]
        r.set(_KEY.format(uid=user_id), json.dumps(profile), ex=_TTL)
    except Exception as e:
        logger.debug("Profile write failed: %s", e)
