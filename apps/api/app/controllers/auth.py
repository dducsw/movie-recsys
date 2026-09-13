import time
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from typing import Optional
from app.models.user import UserModel
from app.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user_optional,
    get_current_user
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    username: Optional[str] = None


class LoginRequest(BaseModel):
    email: Optional[str] = None
    username: Optional[str] = None
    identifier: Optional[str] = None
    password: str


@router.post("/register")
def register_user(body: RegisterRequest):
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
        # Auto-generate friendly unique username from email prefix
        prefix = email_clean.split("@")[0]
        # Keep alphanumeric
        import re
        import random
        base_prefix = re.sub(r"[^a-zA-Z0-9_]", "", prefix) or "movienex_user"
        chosen_username = base_prefix
        counter = 1
        while UserModel.get_by_username(chosen_username) is not None:
            chosen_username = f"{base_prefix}_{random.randint(100, 9999)}"
            counter += 1
            if counter > 20:
                chosen_username = f"user_{int(time.time())}"
                break

    hashed = hash_password(body.password)
    try:
        user = UserModel.create_user(chosen_username, email_clean, hashed)
        token = create_access_token(user["id"], user["username"])
        return {
            "message": "User registered successfully",
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/login")
def login_user(body: LoginRequest):
    """Đăng nhập bằng Email hoặc Username và Mật khẩu."""
    ident = body.identifier or body.email or body.username
    if not ident or not ident.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập Email hoặc Tên người dùng.")

    user = UserModel.get_by_identifier(ident.strip())
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email/Tên người dùng hoặc mật khẩu không chính xác.")

    token = create_access_token(user["id"], user["username"])
    prefs = UserModel.get_user_preferences(user["id"])
    watchlist = UserModel.get_user_watchlist(user["id"])

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "preferences": prefs,
            "watchlist": watchlist
        }
    }


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
            "created_at": current_user.get("created_at")
        },
        "preferences": prefs,
        "watchlist": watchlist,
        "ratings": ratings
    }
