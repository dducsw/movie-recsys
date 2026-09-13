import os
import time
import json
import base64
import hmac
import hashlib
from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Depends
from app.models.user import UserModel

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "movienex_default_secret_key_2026_dev_mode")
ALGORITHM = "HS256"
TOKEN_EXPIRE_SECONDS = 30 * 24 * 3600  # 30 days


def hash_password(password: str) -> str:
    """Mã hóa mật khẩu an toàn sử dụng passlib (hỗ trợ bcrypt, pbkdf2_sha256) hoặc SHA256 Salted."""
    try:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")
        return pwd_context.hash(password)
    except Exception:
        # Secure salt-based SHA256 fallback if native bcrypt binary backend is missing
        salt = os.urandom(16).hex()
        hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return f"sha256${salt}${hashed}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Xác thực mật khẩu."""
    if not hashed_password or not plain_password:
        return False

    if hashed_password.startswith("sha256$"):
        parts = hashed_password.split("$")
        if len(parts) == 3:
            salt, expected_hash = parts[1], parts[2]
            computed = hashlib.sha256((salt + plain_password).encode("utf-8")).hexdigest()
            return hmac.compare_digest(computed, expected_hash)

    try:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def create_access_token(user_id: int, username: str) -> str:
    """Tạo JWT Token cho user (sử dụng PyJWT hoặc HMAC-SHA256 signature)."""
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRE_SECONDS
    }

    try:
        import jwt as pyjwt
        if hasattr(pyjwt, "encode"):
            token = pyjwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
            return token if isinstance(token, str) else token.decode("utf-8")
    except Exception:
        pass

    # Built-in RFC 7519 HMAC-SHA256 JWT
    def b64_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    header_bytes = json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8")
    payload_bytes = json.dumps(payload).encode("utf-8")
    signing_input = f"{b64_encode(header_bytes)}.{b64_encode(payload_bytes)}"
    sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
    return f"{signing_input}.{b64_encode(sig)}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Giải mã và kiểm tra JWT Token."""
    if not token or not isinstance(token, str):
        return None

    try:
        import jwt as pyjwt
        if hasattr(pyjwt, "decode"):
            return pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        pass

    # Built-in RFC 7519 HMAC-SHA256 JWT verification
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}"

        def b64_decode(data: str) -> bytes:
            rem = len(data) % 4
            if rem > 0:
                data += "=" * (4 - rem)
            return base64.urlsafe_b64decode(data.encode("utf-8"))

        # Verify signature
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
        provided_sig = b64_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, provided_sig):
            return None

        # Parse payload
        payload = json.loads(b64_decode(payload_b64).decode("utf-8"))
        if "exp" in payload and payload["exp"] < time.time():
            return None

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


def get_current_user(current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)) -> Dict[str, Any]:
    """Dependency that strictly requires an authenticated user (raises 401 if missing)."""
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Vui lòng đăng nhập để thực hiện thao tác này.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return current_user
