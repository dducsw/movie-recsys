import logging
from typing import List, Dict, Any, Optional
from app.config.db import get_db_cursor

logger = logging.getLogger(__name__)


class CommentModel:
    """Threaded movie comments. Simple, explicit SQL via get_db_cursor."""

    @staticmethod
    def create_comment(
        movie_id: int,
        user_id: int,
        content: str,
        parent_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create comment or reply. Parent must belong to the same movie."""
        text = content.strip()
        if not text:
            raise ValueError("Content must not be empty.")
        with get_db_cursor(commit_on_exit=True) as cur:
            if parent_id is not None:
                cur.execute(
                    "SELECT movie_id FROM movie_comments WHERE id = %s;",
                    (parent_id,))
                parent = cur.fetchone()
                if not parent:
                    raise ValueError(f"Parent comment #{parent_id} not found.")
                if int(parent["movie_id"]) != int(movie_id):
                    raise ValueError("Parent comment belongs to another movie.")

            cur.execute(
                """
                INSERT INTO movie_comments (movie_id, user_id, parent_id, content)
                VALUES (%s, %s, %s, %s)
                RETURNING id, movie_id, user_id, parent_id, content, created_at;
                """,
                (movie_id, user_id, parent_id, text))
            created = dict(cur.fetchone())

            cur.execute(
                "SELECT username, email FROM users WHERE id = %s;", (user_id,))
            user = cur.fetchone() or {}

        result = dict(created)
        raw_dt = result.get("created_at")
        if hasattr(raw_dt, "isoformat"):
            result["created_at"] = raw_dt.isoformat() + ("Z" if raw_dt.tzinfo is None else "")
        else:
            result["created_at"] = str(raw_dt) if raw_dt else ""
        result["username"] = user.get("username", "User")
        result["email"] = user.get("email", "")
        result["likes_count"] = 0
        result["is_liked"] = False
        result["replies"] = []
        return result

    @staticmethod
    def get_movie_comments(
        movie_id: int,
        current_user_id: Optional[int] = None,
        page: int = 1,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Paginated top-level comments (replies attached, no O(n^2))."""
        offset = (page - 1) * limit
        with get_db_cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id, c.movie_id, c.user_id, c.parent_id, c.content, c.created_at,
                    u.username, u.email,
                    COALESCE(like_stats.likes_count, 0) AS likes_count,
                    CASE
                        WHEN %s IS NOT NULL AND user_liked.id IS NOT NULL THEN TRUE
                        ELSE FALSE
                    END AS is_liked
                FROM movie_comments c
                JOIN users u ON c.user_id = u.id
                LEFT JOIN (
                    SELECT comment_id, COUNT(*) AS likes_count
                    FROM comment_likes GROUP BY comment_id
                ) like_stats ON c.id = like_stats.comment_id
                LEFT JOIN comment_likes user_liked
                    ON c.id = user_liked.comment_id AND user_liked.user_id = %s
                WHERE c.movie_id = %s AND c.parent_id IS NULL
                ORDER BY c.created_at DESC
                LIMIT %s OFFSET %s;
                """,
                (current_user_id, current_user_id, movie_id, limit, offset))
            top_rows = [dict(r) for r in cur.fetchall()]
            if not top_rows:
                return []
            top_ids = tuple(r["id"] for r in top_rows)

            cur.execute(
                """
                SELECT
                    c.id, c.movie_id, c.user_id, c.parent_id, c.content, c.created_at,
                    u.username, u.email,
                    COALESCE(like_stats.likes_count, 0) AS likes_count,
                    CASE
                        WHEN %s IS NOT NULL AND user_liked.id IS NOT NULL THEN TRUE
                        ELSE FALSE
                    END AS is_liked
                FROM movie_comments c
                JOIN users u ON c.user_id = u.id
                LEFT JOIN (
                    SELECT comment_id, COUNT(*) AS likes_count
                    FROM comment_likes GROUP BY comment_id
                ) like_stats ON c.id = like_stats.comment_id
                LEFT JOIN comment_likes user_liked
                    ON c.id = user_liked.comment_id AND user_liked.user_id = %s
                WHERE c.parent_id IN %s
                ORDER BY c.created_at ASC;
                """,
                (current_user_id, current_user_id, top_ids))
            reply_rows = [dict(r) for r in cur.fetchall()]

        def _fmt(row: Dict[str, Any]) -> Dict[str, Any]:
            raw_dt = row["created_at"]
            if hasattr(raw_dt, "isoformat"):
                created_iso = raw_dt.isoformat() + ("Z" if raw_dt.tzinfo is None else "")
            else:
                created_iso = str(raw_dt) if raw_dt else ""
            return {
                "id": row["id"], "movie_id": row["movie_id"],
                "user_id": row["user_id"], "parent_id": row["parent_id"],
                "content": row["content"], "created_at": created_iso,
                "username": row["username"], "email": row["email"],
                "likes_count": int(row["likes_count"]),
                "is_liked": bool(row["is_liked"]), "replies": [],
            }

        by_id = {r["id"]: _fmt(r) for r in top_rows}
        for r in reply_rows:
            item = _fmt(r)
            parent = by_id.get(item["parent_id"])
            if parent is not None:
                parent["replies"].append(item)
        return [by_id[r["id"]] for r in top_rows]

    @staticmethod
    def toggle_like(comment_id: int, user_id: int) -> Dict[str, Any]:
        """Atomic toggle via ON CONFLICT. Returns new state + count."""
        with get_db_cursor(commit_on_exit=True) as cur:
            cur.execute(
                """
                INSERT INTO comment_likes (comment_id, user_id)
                VALUES (%s, %s) ON CONFLICT DO NOTHING
                RETURNING comment_id;
                """,
                (comment_id, user_id))
            inserted = cur.fetchone()
            if inserted is None:
                cur.execute(
                    "DELETE FROM comment_likes WHERE comment_id = %s AND user_id = %s;",
                    (comment_id, user_id))
                is_liked = False
            else:
                is_liked = True
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM comment_likes WHERE comment_id = %s;",
                (comment_id,))
            count = int((cur.fetchone() or {}).get("cnt", 0))
        return {"comment_id": comment_id, "is_liked": is_liked, "likes_count": count}

    @staticmethod
    def delete_comment(comment_id: int, user_id: int, is_admin: bool = False) -> bool:
        """Delete if author or admin."""
        with get_db_cursor(commit_on_exit=True) as cur:
            cur.execute(
                "SELECT user_id FROM movie_comments WHERE id = %s;", (comment_id,))
            row = cur.fetchone()
            if not row:
                return False
            if not is_admin and row["user_id"] != user_id:
                raise PermissionError("You can only delete your own comments.")
            cur.execute("DELETE FROM movie_comments WHERE id = %s;", (comment_id,))
            return True
