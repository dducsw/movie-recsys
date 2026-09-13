import logging
from typing import List, Dict, Any, Optional
from app.config.db import get_db_connection

logger = logging.getLogger(__name__)


class CommentModel:
    """Model managing movie comments and comment likes (Facebook-style threaded comments)."""

    @staticmethod
    def create_comment(
        movie_id: int,
        user_id: int,
        content: str,
        parent_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new top-level comment or reply to an existing comment."""
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # If parent_id is provided, verify parent exists and belongs to same movie
            if parent_id is not None:
                cur.execute("SELECT movie_id FROM movie_comments WHERE id = %s;", (parent_id,))
                parent = cur.fetchone()
                if not parent:
                    raise ValueError(f"Parent comment #{parent_id} not found.")

            query = """
                INSERT INTO movie_comments (movie_id, user_id, parent_id, content)
                VALUES (%s, %s, %s, %s)
                RETURNING id, movie_id, user_id, parent_id, content, created_at;
            """
            cur.execute(query, (movie_id, user_id, parent_id, content.strip()))
            created = cur.fetchone()
            conn.commit()

            # Retrieve author info
            cur.execute("SELECT username, email FROM users WHERE id = %s;", (user_id,))
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
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_movie_comments(movie_id: int, current_user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch all comments for a movie structured in a hierarchical tree:
        Top-level comments with ordered nested replies, author metadata, like counts, and user reaction state.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = """
                SELECT 
                    c.id,
                    c.movie_id,
                    c.user_id,
                    c.parent_id,
                    c.content,
                    c.created_at,
                    u.username,
                    u.email,
                    COALESCE(like_stats.likes_count, 0) AS likes_count,
                    CASE 
                        WHEN %s IS NOT NULL AND user_liked.id IS NOT NULL THEN TRUE 
                        ELSE FALSE 
                    END AS is_liked
                FROM movie_comments c
                JOIN users u ON c.user_id = u.id
                LEFT JOIN (
                    SELECT comment_id, COUNT(*) AS likes_count
                    FROM comment_likes
                    GROUP BY comment_id
                ) like_stats ON c.id = like_stats.comment_id
                LEFT JOIN comment_likes user_liked 
                    ON c.id = user_liked.comment_id AND user_liked.user_id = %s
                WHERE c.movie_id = %s
                ORDER BY c.created_at ASC;
            """
            cur.execute(query, (current_user_id, current_user_id, movie_id))
            rows = [dict(r) for r in cur.fetchall()]

            # Build hierarchical structure: separate top-level comments and replies
            top_level_map = {}
            replies_list = []

            for row in rows:
                raw_dt = row["created_at"]
                if hasattr(raw_dt, "isoformat"):
                    created_iso = raw_dt.isoformat() + ("Z" if raw_dt.tzinfo is None else "")
                else:
                    created_iso = str(raw_dt) if raw_dt else ""

                item = {
                    "id": row["id"],
                    "movie_id": row["movie_id"],
                    "user_id": row["user_id"],
                    "parent_id": row["parent_id"],
                    "content": row["content"],
                    "created_at": created_iso,
                    "username": row["username"],
                    "email": row["email"],
                    "likes_count": int(row["likes_count"]),
                    "is_liked": bool(row["is_liked"]),
                    "replies": []
                }
                if row["parent_id"] is None:
                    top_level_map[row["id"]] = item
                else:
                    replies_list.append(item)

            # Attach replies to their respective parents
            for rep in replies_list:
                p_id = rep["parent_id"]
                if p_id in top_level_map:
                    top_level_map[p_id]["replies"].append(rep)
                else:
                    # In case parent is another reply (support 2+ level collapse into top-level)
                    for top_c in top_level_map.values():
                        if any(r["id"] == p_id for r in top_c["replies"]):
                            top_c["replies"].append(rep)
                            break

            # Return top-level comments ordered descending (newest comments first)
            result = list(top_level_map.values())
            result.reverse()
            return result
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def toggle_like(comment_id: int, user_id: int) -> Dict[str, Any]:
        """Toggle like status on a comment for the current user."""
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            # Check if already liked
            cur.execute("SELECT id FROM comment_likes WHERE comment_id = %s AND user_id = %s;", (comment_id, user_id))
            existing = cur.fetchone()

            if existing:
                # Remove like
                cur.execute("DELETE FROM comment_likes WHERE comment_id = %s AND user_id = %s;", (comment_id, user_id))
                is_liked = False
            else:
                # Add like
                cur.execute("INSERT INTO comment_likes (comment_id, user_id) VALUES (%s, %s);", (comment_id, user_id))
                is_liked = True

            conn.commit()

            # Get updated count
            cur.execute("SELECT COUNT(*) AS cnt FROM comment_likes WHERE comment_id = %s;", (comment_id,))
            cnt_row = cur.fetchone()
            count = int(cnt_row["cnt"]) if cnt_row else 0

            return {
                "comment_id": comment_id,
                "is_liked": is_liked,
                "likes_count": count
            }
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def delete_comment(comment_id: int, user_id: int) -> bool:
        """Delete a comment if user is author."""
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT user_id FROM movie_comments WHERE id = %s;", (comment_id,))
            row = cur.fetchone()
            if not row:
                return False
            if row["user_id"] != user_id:
                raise PermissionError("You can only delete your own comments.")

            cur.execute("DELETE FROM movie_comments WHERE id = %s;", (comment_id,))
            conn.commit()
            return True
        finally:
            cur.close()
            conn.close()
