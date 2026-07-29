from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from typing import Optional
from app.models.user import UserModel
from app.services.auth_service import hash_password, verify_password, create_access_token, get_current_user_optional

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/register")
def register_user(body: RegisterRequest):
    """Đăng ký tài khoản mới."""
    if len(body.username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    existing = UserModel.get_by_email(body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email is already registered.")

    hashed = hash_password(body.password)
    try:
        user = UserModel.create_user(body.username.strip(), body.email.strip().lower(), hashed)
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
    """Đăng nhập bằng email và mật khẩu."""
    user = UserModel.get_by_email(body.email.strip().lower())
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

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
