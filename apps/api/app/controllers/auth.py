import time
import random
import re
import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from pydantic import BaseModel, EmailStr, Field, ConfigDict

from app.models.user import UserModel
from app.config.settings import get_settings
from app.config.security import limiter
from app.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    set_refresh_cookie,
    clear_refresh_cookie,
    issue_token_pair,
    get_current_user_optional,
    get_current_user
)

def _generate_username(email_clean: str) -> str:
    """Generate a unique username from email prefix. Simple loop with cap."""
    prefix = email_clean.split("@")[0]
    base = re.sub(r"[^a-zA-Z0-9_]", "", prefix) or "movienex_user"
    candidate = base
    for _ in range(20):
        if UserModel.get_by_username(candidate) is None:
            return candidate
        candidate = f"{base}_{random.randint(100, 9999)}"
    return f"user_{int(time.time())}"

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# --- Strict Input Schemas (extra="forbid" to prevent mass assignment) ---

class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(..., max_length=128)
    username: Optional[str] = Field(default=None, min_length=3, max_length=50)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: Optional[str] = Field(default=None, max_length=255)
    username: Optional[str] = Field(default=None, max_length=50)
    identifier: Optional[str] = Field(default=None, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


# --- Safe Output Schemas ---

class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str = "user"
    created_at: Optional[Any] = None


class AuthSuccessResponse(BaseModel):
    message: str
    access_token: str
    token_type: str = "bearer"
    user: UserProfileResponse


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfileResponse


@router.post("/register", response_model=AuthSuccessResponse)
@limiter.limit(settings.RATE_LIMIT_REGISTER)
def register_user(request: Request, response: Response, body: RegisterRequest):
    """Đăng ký tài khoản mới bằng Email và Mật khẩu (Username tùy chọn)."""
    email_clean = body.email.strip().lower()

    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Mật khẩu phải có ít nhất 6 ký tự.")

    # Check if email is already registered
    existing_email = UserModel.get_by_email(email_clean)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email này đã được đăng ký tài khoản.")

    # Determine or generate username
    if body.username and body.username.strip():
        chosen_username = body.username.strip()
        if len(chosen_username) < 3:
            raise HTTPException(status_code=400, detail="Tên người dùng phải có ít nhất 3 ký tự.")
        existing_username = UserModel.get_by_username(chosen_username)
        if existing_username:
            raise HTTPException(status_code=400, detail="Tên người dùng đã tồn tại. Vui lòng chọn tên khác.")
    else:
        chosen_username = _generate_username(email_clean)

    hashed = hash_password(body.password)
    try:
        user = UserModel.create_user(chosen_username, email_clean, hashed)
        role = user.get("role", "user")

        access_token = issue_token_pair(response, user["id"], user["username"], role=role)

        return {
            "message": "User registered successfully",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": role,
                "created_at": user.get("created_at")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")


@router.post("/login", response_model=Dict[str, Any])
@limiter.limit(settings.RATE_LIMIT_LOGIN)
def login_user(request: Request, response: Response, body: LoginRequest):
    """Đăng nhập bằng Email hoặc Username và Mật khẩu."""
    ident = body.identifier or body.email or body.username
    if not ident or not ident.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập Email hoặc Tên người dùng.")

    user = UserModel.get_by_identifier(ident.strip())
    if not user or not verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Email/Tên người dùng hoặc mật khẩu không chính xác.")

    role = user.get("role", "user")
    access_token = issue_token_pair(response, user["id"], user["username"], role=role)

    prefs = UserModel.get_user_preferences(user["id"])
    watchlist = UserModel.get_user_watchlist(user["id"])

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": role,
            "preferences": prefs,
            "watchlist": watchlist
        }
    }


@router.post("/refresh", response_model=RefreshResponse)
@limiter.limit(settings.RATE_LIMIT_REFRESH)
def refresh_access_token(request: Request, response: Response):
    """
    Refreshes short-lived access token using secure HttpOnly refresh token cookie.
    Rotates refresh token for enhanced session security.
    """
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token missing from cookies.")

    payload = decode_refresh_token(refresh_token)
    if not payload:
        clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.")

    user_id = int(payload["sub"])
    user = UserModel.get_by_id(user_id)
    if not user:
        clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="User account no longer exists.")

    role = user.get("role", "user")
    new_access_token = issue_token_pair(response, user["id"], user["username"], role=role)

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": role,
            "created_at": user.get("created_at")
        }
    }


@router.post("/logout")
def logout_user(response: Response):
    """Xóa cookie Refresh Token khi người dùng đăng xuất."""
    clear_refresh_cookie(response)
    return {"message": "Logged out successfully"}


@router.get("/me")
def get_current_user_profile(current_user: Optional[dict] = Depends(get_current_user_optional)):
    """Lấy thông tin chi tiết của người dùng đang đăng nhập."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    user_id = current_user["id"]
    prefs = UserModel.get_user_preferences(user_id)
    watchlist = UserModel.get_user_watchlist(user_id)
    ratings = UserModel.get_user_ratings(user_id)

    return {
        "user": {
            "id": current_user["id"],
            "username": current_user["username"],
            "email": current_user["email"],
            "role": current_user.get("role", "user"),
            "created_at": current_user.get("created_at")
        },
        "preferences": prefs,
        "watchlist": watchlist,
        "ratings": ratings
    }
