import time
import logging
from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Depends, Request, Response
from passlib.context import CryptContext
import jwt

from app.config.settings import get_settings
from app.models.user import UserModel

logger = logging.getLogger(__name__)
settings = get_settings()

SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM  # "HS256"

# Strictly use bcrypt via passlib (no weak SHA-256 fallback)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash password securely using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash."""
    if not hashed_password or not plain_password:
        return False
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.warning(f"Password verification error: {e}")
        return False


def create_access_token(user_id: int, username: str, role: str = "user") -> str:
    """Create a short-lived Access Token (default 15 minutes)."""
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "token_type": "access",
        "iat": now,
        "exp": now + (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: int, username: str) -> str:
    """Create a long-lived Refresh Token (default 7 days)."""
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "username": username,
        "token_type": "refresh",
        "iat": now,
        "exp": now + (settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and strictly validates an access token.
    Enforces algorithm HS256, rejects 'none', and validates exp, sub, and token_type.
    """
    if not token or not isinstance(token, str):
        return None

    try:
        # options require exp and sub claims
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={
                "verify_signature": True,
                "require": ["exp", "sub"],
                "verify_exp": True
            }
        )
        if payload.get("token_type") != "access":
            logger.warning("Rejected token: expected token_type 'access'")
            return None

        # Validate that sub is integer representation
        int(payload["sub"])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, ValueError, TypeError) as e:
        logger.debug(f"Access token validation failure: {e}")
        return None


def decode_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decodes and strictly validates a refresh token.
    Enforces algorithm HS256, rejects 'none', and validates exp, sub, and token_type.
    """
    if not token or not isinstance(token, str):
        return None

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={
                "verify_signature": True,
                "require": ["exp", "sub"],
                "verify_exp": True
            }
        )
        if payload.get("token_type") != "refresh":
            logger.warning("Rejected token: expected token_type 'refresh'")
            return None

        int(payload["sub"])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, ValueError, TypeError) as e:
        logger.debug(f"Refresh token validation failure: {e}")
        return None


def set_refresh_cookie(response: Response, refresh_token: str) -> None:
    """Sets the Refresh Token in a secure, HttpOnly cookie."""
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        path="/api/auth",
    )


def clear_refresh_cookie(response: Response) -> None:
    """Clears the Refresh Token cookie."""
    response.delete_cookie(
        key="refresh_token",
        path="/api/auth",
        httponly=True,
        secure=settings.is_production,
        samesite="lax"
    )


def get_current_user_optional(
    request: Request,
    authorization: Optional[str] = Header(None)
) -> Optional[Dict[str, Any]]:
    """
    Dependency returning current user from Bearer header if present and valid.
    Returns None if unauthenticated.
    """
    token = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]

    if not token:
        return None

    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None

    try:
        user_id = int(payload["sub"])
        return UserModel.get_by_id(user_id)
    except Exception:
        return None


def get_current_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
    """Dependency that strictly requires an authenticated user."""
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Vui lòng đăng nhập để thực hiện thao tác này.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return current_user


def require_admin(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """RBAC Dependency: requires admin privileges."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Quyền truy cập bị từ chối: Yêu cầu quyền quản trị viên."
        )
    return current_user
