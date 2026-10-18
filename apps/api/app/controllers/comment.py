import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, ConfigDict

from app.models.comment import CommentModel
from app.services.auth_service import get_current_user, get_current_user_optional, require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Comments"])


class CreateCommentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=2000, description="Content of the comment")
    parent_id: Optional[int] = Field(None, description="Optional ID of parent comment to reply to")


class CommentLikeResponse(BaseModel):
    comment_id: int
    is_liked: bool
    likes_count: int


@router.get("/api/movies/{movie_id}/comments")
def get_movie_comments(
    movie_id: int,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Paginated top-level comments with nested replies."""
    user_id = current_user["id"] if current_user else None
    try:
        comments = CommentModel.get_movie_comments(
            movie_id=movie_id, current_user_id=user_id, page=page, limit=limit)
        return {
            "movie_id": movie_id,
            "page": page,
            "limit": limit,
            "total_count": len(comments),
            "comments": comments
        }
    except Exception as e:
        logger.error("Error fetching comments for movie %s: %s", movie_id, e)
        raise HTTPException(status_code=500, detail="Không thể tải bình luận cho bộ phim này.")


@router.post("/api/movies/{movie_id}/comments")
def create_movie_comment(
    movie_id: int,
    body: CreateCommentRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Đăng bình luận mới cho phim hoặc gửi phản hồi cho bình luận khác (yêu cầu đăng nhập).
    """
    clean_content = body.content.strip()
    if not clean_content:
        raise HTTPException(status_code=400, detail="Nội dung bình luận không được để trống.")

    try:
        new_comment = CommentModel.create_comment(
            movie_id=movie_id,
            user_id=current_user["id"],
            content=clean_content,
            parent_id=body.parent_id
        )
        return {
            "message": "Bình luận đã được đăng thành công.",
            "comment": new_comment
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Error creating comment on movie %s: %s", movie_id, e)
        raise HTTPException(status_code=500, detail="Lỗi khi đăng bình luận.")


@router.post("/api/comments/{comment_id}/like", response_model=CommentLikeResponse)
def toggle_comment_like(
    comment_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Thả tim / bỏ thích một bình luận (yêu cầu đăng nhập).
    """
    try:
        result = CommentModel.toggle_like(comment_id=comment_id, user_id=current_user["id"])
        return result
    except Exception as e:
        logger.error("Error toggling like on comment %s: %s", comment_id, e)
        raise HTTPException(status_code=500, detail="Không thể cập nhật lượt thích.")


@router.delete("/api/comments/{comment_id}")
def delete_comment(
    comment_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Xóa bình luận của chính mình hoặc admin."""
    try:
        is_admin = current_user.get("role") == "admin"
        success = CommentModel.delete_comment(
            comment_id=comment_id, user_id=current_user["id"], is_admin=is_admin)
        if not success:
            raise HTTPException(status_code=404, detail="Bình luận không tồn tại.")
        return {"message": "Đã xóa bình luận thành công."}
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting comment %s: %s", comment_id, e)
        raise HTTPException(status_code=500, detail="Lỗi khi xóa bình luận.")


@router.delete("/api/admin/comments/{comment_id}")
def admin_delete_comment(
    comment_id: int,
    _admin: Dict[str, Any] = Depends(require_admin),
):
    """Admin xóa bất kỳ bình luận nào."""
    ok = CommentModel.delete_comment(
        comment_id=comment_id, user_id=_admin["id"], is_admin=True)
    if not ok:
        raise HTTPException(status_code=404, detail="Bình luận không tồn tại.")
    return {"message": "Đã xóa bình luận thành công."}
