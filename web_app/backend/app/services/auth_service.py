import os
import time
import jwt
import hashlib
from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Depends
from app.models.user import UserModel

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "recsys_super_secret_jwt_key_2026")
ALGORITHM = "HS256"
TOKEN_EXPIRE_SECONDS = 30 * 24 * 3600  # 30 days


def hash_password(password: str) -> str:
    """Mã hóa mật khẩu sử dụng passlib/bcrypt hoặc fallback sha256 + salt."""
    try:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)
    except Exception:
        salt = "recsys_salt_v1"
        return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Xác thực mật khẩu."""
    try:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        if pwd_context.verify(plain_password, hashed_password):
            return True
    except Exception:
        pass
    
    salt = "recsys_salt_v1"
    fallback_hash = hashlib.sha256((plain_password + salt).encode("utf-8")).hexdigest()
    return fallback_hash == hashed_password


def create_access_token(user_id: int, username: str) -> str:
    """Tạo JWT Token cho user."""
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRE_SECONDS
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Giải mã và kiểm tra JWT Token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        return None


def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """
    Dependency lấy thông tin user từ Authorization header (Bearer token) nếu có.
    Trả về Dict thông tin user hoặc None nếu không đăng nhập / token không hợp lệ.
    """
    if not authorization:
        return None

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    token = parts[1]
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None

    try:
        user_id = int(payload["sub"])
        return UserModel.get_by_id(user_id)
    except Exception:
        return None
