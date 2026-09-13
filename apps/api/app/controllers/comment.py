import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.models.comment import CommentModel
from app.services.auth_service import get_current_user, get_current_user_optional

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Comments"])


class CreateCommentRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000, description="Content of the comment")
    parent_id: Optional[int] = Field(None, description="Optional ID of parent comment to reply to")


class CommentLikeResponse(BaseModel):
    comment_id: int
    is_liked: bool
    likes_count: int


@router.get("/api/movies/{movie_id}/comments")
def get_movie_comments(
    movie_id: int,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """
    Lấy danh sách tất cả bình luận của phim, kèm danh sách câu trả lời lồng nhau,
    số lượng like và trạng thái thích của người dùng hiện tại (nếu đã đăng nhập).
    """
    user_id = current_user["id"] if current_user else None
    try:
        comments = CommentModel.get_movie_comments(movie_id=movie_id, current_user_id=user_id)
        return {
            "movie_id": movie_id,
            "total_count": len(comments),
            "comments": comments
        }
    except Exception as e:
        logger.error(f"Error fetching comments for movie {movie_id}: {e}")
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
        logger.error(f"Error creating comment on movie {movie_id}: {e}")
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
        logger.error(f"Error toggling like on comment {comment_id}: {e}")
        raise HTTPException(status_code=500, detail="Không thể cập nhật lượt thích.")


@router.delete("/api/comments/{comment_id}")
def delete_comment(
    comment_id: int,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Xóa bình luận của chính mình (yêu cầu đăng nhập và đúng chủ sở hữu).
    """
    try:
        success = CommentModel.delete_comment(comment_id=comment_id, user_id=current_user["id"])
        if not success:
            raise HTTPException(status_code=404, detail="Bình luận không tồn tại.")
        return {"message": "Đã xóa bình luận thành công."}
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comment {comment_id}: {e}")
        raise HTTPException(status_code=500, detail="Lỗi khi xóa bình luận.")
