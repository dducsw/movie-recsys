"""CRAG-style reflect + rerank: drop context-mismatch, boost profile fit."""

from __future__ import annotations


def reflect_and_rerank(candidates: list[dict], intent=None,
                       profile: dict | None = None,
                       liked_ids: list[int] | None = None,
                       limit: int = 5) -> list[dict]:
    """Deterministic post-filter. No extra LLM call (fast, cheap)."""
    if not candidates:
        return []
    profile = profile or {}
    liked_ids = set(liked_ids or [])
    want_genres = {g.lower() for g in (intent.genres if intent else []) or []}
    want_year = str(intent.year if intent else "" or "").strip()
    min_rating = intent.min_rating if intent else None
    prof_genres = {g.lower() for g in profile.get("liked_genres", [])}

    scored: list[tuple[float, dict]] = []
    for m in candidates:
        mid = m.get("movieId") or m.get("movieid")
        if mid in liked_ids and len(candidates) > limit:
            continue  # don't recommend already-liked when we have depth
        genres = {g.strip().lower() for g in str(m.get("genres", "")).split("|")}
        try:
            rating = float(m.get("vote_average") or 0)
        except (TypeError, ValueError):
            rating = 0.0
        year = str(m.get("release_date", ""))[:4]
        if want_genres and not (genres & want_genres):
            continue
        if want_year and want_year not in str(m.get("release_date", "")):
            if not str(year).startswith(want_year[:4]):
                continue
        if min_rating is not None and rating < float(min_rating):
            continue
        bonus = len(genres & prof_genres) * 0.5 + rating * 0.05
        try:
            bonus += float(m.get("popularity") or 0) * 0.001
        except (TypeError, ValueError):
            pass
        scored.append((bonus, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]
