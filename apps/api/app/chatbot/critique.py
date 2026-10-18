"""Critiquing: 'funnier, newer, higher-rated' and negative exclusions on the last slate."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_GENRE_HINTS = {
    "funny": "Comedy", "funnier": "Comedy", "hài": "Comedy", "hài hước": "Comedy",
    "scary": "Horror", "scarier": "Horror", "kinh dị": "Horror", "ma": "Horror",
    "romantic": "Romance", "tình cảm": "Romance", "lãng mạn": "Romance",
    "action-packed": "Action", "hành động": "Action",
    "animation": "Animation", "hoạt hình": "Animation",
    "sci-fi": "Sci-Fi", "khoa học viễn tưởng": "Sci-Fi",
    "drama": "Drama", "tâm lý": "Drama",
    "thriller": "Thriller", "giật gân": "Thriller",
}

_NEGATIVE_TRIGGERS = (
    "không thích", "đừng", "bỏ qua", "không muốn", "ko thích", "đừng gợi ý",
    "without", "no ", "don't like", "hate", "exclude", "not "
)

_RATING_RE = re.compile(r"(?:above|over|>=?)\s*([0-9](?:\.[0-9])?)|([0-9](?:\.[0-9])?)\s*\+")
_YEAR_RE = re.compile(r"\b(19\d\d|20\d\d)\b")


def parse_critique(text: str) -> dict | None:
    """Parse a follow-up refinement or negative exclusion. Returns None if not a critique."""
    if not text:
        return None
    low = text.lower().strip()
    triggers = (
        "more", "funnier", "scarier", "newer", "older", "higher",
        "better rated", "hài hơn", "mới hơn", "cũ hơn", "hay hơn",
        "another", "khác", "tiếp", "đừng", "không thích", "không muốn",
        "bỏ qua", "without", "no "
    )
    if not any(t in low for t in triggers):
        return None

    out: dict = {"is_critique": True}

    # 1. Negative Exclusions (Negative Critiquing)
    is_negative = any(neg in low for neg in _NEGATIVE_TRIGGERS)
    if is_negative:
        excluded_genres = []
        for hint, genre in _GENRE_HINTS.items():
            if hint in low:
                excluded_genres.append(genre)
        if excluded_genres:
            out["exclude_genres"] = list(set(excluded_genres))

    # 2. Positive Critiquing (only when not explicitly negated)
    if "exclude_genres" not in out:
        for hint, genre in _GENRE_HINTS.items():
            if hint in low:
                out["genre"] = genre
                break

    # 3. Rating threshold
    m = _RATING_RE.search(low)
    if m:
        try:
            out["min_rating"] = float(m.group(1) or m.group(2))
        except ValueError:
            pass

    # 4. Temporal threshold
    y = _YEAR_RE.search(low)
    if y:
        out["newer_than"] = int(y.group(1))
    elif "newer" in low or "mới hơn" in low:
        out["newer_than"] = 2015
    elif "older" in low or "cũ hơn" in low or "classic" in low:
        out["older_than"] = 2005

    if not any(k in out for k in ("genre", "exclude_genres", "min_rating", "newer_than", "older_than")):
        out["diverse"] = True  # plain "more / khác" = next batch
    return out


def apply_critique(pool: list[dict], critique: dict, limit: int = 5) -> list[dict]:
    """Filter a saved pool by critique constraints and negative exclusions."""
    if not pool:
        return []
    out = list(pool)

    # Filter out excluded genres
    if critique.get("exclude_genres"):
        excluded = [g.lower() for g in critique["exclude_genres"]]
        out = [
            m for m in out
            if not any(ex in str(m.get("genres", "")).lower() for ex in excluded)
        ]

    # Positive genre focus
    if critique.get("genre"):
        want = critique["genre"].lower()
        filtered = [m for m in out if want in str(m.get("genres", "")).lower()]
        out = filtered or out

    # Rating threshold
    if critique.get("min_rating") is not None:
        try:
            out = [m for m in out
                   if float(m.get("vote_average") or 0) >= float(critique["min_rating"])]
        except (TypeError, ValueError):
            pass

    # Temporal bounds
    def _year(m: dict) -> int:
        try:
            return int(str(m.get("release_date", ""))[:4])
        except ValueError:
            return 0

    if critique.get("newer_than") is not None:
        out = [m for m in out if _year(m) >= int(critique["newer_than"])]

    if critique.get("older_than") is not None:
        out = [m for m in out if 0 < _year(m) <= int(critique["older_than"])]

    return out[:limit]
