import pytest
import jwt
import time
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import app
from app.config.settings import get_settings
from app.config.security import validate_safe_url
from app.services.auth_service import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    hash_password,
    verify_password
)
from app.chatbot.sql_retriever import validate_and_sanitize_sql, heuristic_generate_sql
from app.controllers.auth import RegisterRequest, LoginRequest
from app.controllers.comment import CreateCommentRequest
from app.controllers.events import ClickEvent, RatingEvent, WatchlistEvent
from app.controllers.onboarding import OnboardingRequest
from app.controllers.movie import MovieBatchRequest
from app.controllers.chatbot import ChatRequest
from app.models.comment import CommentModel

client = TestClient(app)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pydantic Mass Assignment & Extra Fields Rejection
# ─────────────────────────────────────────────────────────────────────────────

def test_extra_forbidden_fields_in_schemas():
    """Verify extra='forbid' rejects injection of administrative or rogue attributes."""
    # RegisterRequest rejects extra fields (e.g. role, is_admin)
    with pytest.raises(ValidationError):
        RegisterRequest(
            email="test@example.com",
            password="secure_password_123",
            role="admin"  # Injected field
        )

    with pytest.raises(ValidationError):
        LoginRequest(
            identifier="testuser",
            password="password",
            is_superuser=True  # Injected field
        )

    with pytest.raises(ValidationError):
        CreateCommentRequest(
            content="Valid comment content",
            user_id=1  # Injected field
        )

    with pytest.raises(ValidationError):
        ClickEvent(
            movie_id=1,
            source="trending",
            admin_override=True
        )

    with pytest.raises(ValidationError):
        RatingEvent(
            movie_id=1,
            rating=4.5,
            user_id=999
        )

    with pytest.raises(ValidationError):
        WatchlistEvent(
            movie_id=1,
            bypass=True
        )

    with pytest.raises(ValidationError):
        OnboardingRequest(
            favorite_genres=["Action"],
            favorite_movie_ids=[1],
            is_vip=True
        )

    with pytest.raises(ValidationError):
        MovieBatchRequest(
            movie_ids=[1, 2],
            raw_sql="SELECT 1"
        )

    with pytest.raises(ValidationError):
        ChatRequest(
            message="Hello",
            system_role="admin"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. JWT Hardening, Algorithm Enforcement & Type Separation
# ─────────────────────────────────────────────────────────────────────────────

def test_jwt_algorithm_none_rejected():
    """Verify that unverified or 'none' algorithm tokens are strictly rejected."""
    # Craft forged token with alg: none
    header = {"alg": "none", "typ": "JWT"}
    payload = {
        "sub": "100",
        "username": "attacker",
        "token_type": "access",
        "exp": int(time.time()) + 3600
    }
    # Unsigned token string
    import base64, json
    b64_h = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    b64_p = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    forged_token = f"{b64_h}.{b64_p}."

    decoded = decode_access_token(forged_token)
    assert decoded is None


def test_jwt_token_type_separation():
    """Verify that refresh tokens cannot be used to authenticate access token routes."""
    user_id = 42
    username = "alice"
    refresh_token = create_refresh_token(user_id=user_id, username=username)

    # Attempt to decode as access token
    as_access = decode_access_token(refresh_token)
    assert as_access is None

    # Valid when decoded as refresh token
    as_refresh = decode_refresh_token(refresh_token)
    assert as_refresh is not None
    assert as_refresh["sub"] == str(user_id)
    assert as_refresh["token_type"] == "refresh"


def test_jwt_expired_token_rejected():
    """Verify that expired tokens return None."""
    past = int(time.time()) - 100
    payload = {
        "sub": "42",
        "username": "expired_user",
        "token_type": "access",
        "iat": past - 1000,
        "exp": past
    }
    expired_token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")
    assert decode_access_token(expired_token) is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Security Headers & CORS Policy
# ─────────────────────────────────────────────────────────────────────────────

def test_security_headers_injected():
    """Verify all responses include mandatory defense-in-depth headers."""
    res = client.get("/")
    assert res.status_code == 200

    headers = res.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert "max-age=" in headers.get("Strict-Transport-Security", "")
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "default-src 'self'" in headers.get("Content-Security-Policy", "")


def test_cors_options_preflight():
    """Verify CORS preflight returns restricted methods and allowed origins."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type,Authorization,X-Requested-With"
    }
    res = client.options("/api/movies/search", headers=headers)
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert res.headers.get("access-control-allow-credentials") == "true"


# ─────────────────────────────────────────────────────────────────────────────
# 4. SQL Injection Guardrails & Parameter Sanitization
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_injection_defense():
    """Verify SQL validation rejects dangerous injection payloads and comments."""
    # Comments injection
    valid, _, _ = validate_and_sanitize_sql("SELECT * FROM movies WHERE id = 1 -- comment")
    assert valid is False

    valid, _, _ = validate_and_sanitize_sql("SELECT * FROM movies WHERE id = 1 /* block */")
    assert valid is False

    # UNION SELECT extraction
    valid, _, _ = validate_and_sanitize_sql("SELECT * FROM movies UNION SELECT username, password_hash FROM users")
    assert valid is False

    # Catalog snooping
    valid, _, _ = validate_and_sanitize_sql("SELECT * FROM pg_shadow")
    assert valid is False

    valid, _, _ = validate_and_sanitize_sql("SELECT * FROM pg_user")
    assert valid is False


def test_heuristic_sql_input_sanitization():
    """Verify heuristic SQL generator strips harmful characters from user inputs."""
    # Attempt injection via director name
    malicious_query = "Phim của đạo diễn Christopher Nolan'; DROP TABLE users; --"
    sql = heuristic_generate_sql(malicious_query)
    if sql:
        assert "DROP" not in sql
        assert "--" not in sql
        # Only the terminal semicolon allowed
        assert sql.strip().endswith(";")
        assert sql.count(";") == 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. SSRF Validation Guard
# ─────────────────────────────────────────────────────────────────────────────

def test_ssrf_validator_blocks_internal_ips():
    """Verify SSRF validator blocks loopback, private networks, and cloud metadata."""
    import socket
    from unittest.mock import patch

    # Loopback
    assert validate_safe_url("http://127.0.0.1:8000/internal") is False
    assert validate_safe_url("http://localhost:8000/internal") is False

    # Private RFC 1918
    assert validate_safe_url("http://10.0.0.1:5432") is False
    assert validate_safe_url("http://192.168.1.1/admin") is False
    assert validate_safe_url("http://172.16.0.1/status") is False

    # Cloud Metadata Service (169.254.169.254)
    assert validate_safe_url("http://169.254.169.254/latest/meta-data/") is False

    # Non-HTTP protocols
    assert validate_safe_url("file:///etc/passwd") is False
    assert validate_safe_url("ftp://server/file") is False
    assert validate_safe_url("gopher://127.0.0.1:70") is False

    # Public valid IP (e.g. 93.184.216.34)
    with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]):
        assert validate_safe_url("https://api.themoviedb.org/3/movie/550") is True


# ─────────────────────────────────────────────────────────────────────────────
# 6. Authorization & BOLA / IDOR Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_comment_deletion_authorization_unit():
    """Verify non-authors cannot delete comments, but administrators can."""
    import unittest.mock as mock

    # Mock DB cursor to simulate a comment belonging to user 10
    with mock.patch("app.models.comment.get_db_connection") as mock_conn:
        mock_cur = mock.MagicMock()
        mock_cur.fetchone.return_value = {"user_id": 10}
        mock_conn.return_value.cursor.return_value = mock_cur

        # User 20 (not author, not admin) -> PermissionError
        with pytest.raises(PermissionError):
            CommentModel.delete_comment(comment_id=1, user_id=20, is_admin=False)

        # User 10 (author) -> Succeeds
        success_author = CommentModel.delete_comment(comment_id=1, user_id=10, is_admin=False)
        assert success_author is True

        # Admin user 999 (admin override) -> Succeeds
        success_admin = CommentModel.delete_comment(comment_id=1, user_id=999, is_admin=True)
        assert success_admin is True


def test_chatbot_session_ownership_enforcement():
    """Verify users cannot access user-scoped sessions belonging to other users."""
    # Alice (user_id=1) tries to access Bob's session (user_2_chat)
    alice_token = create_access_token(user_id=1, username="alice")
    headers = {"Authorization": f"Bearer {alice_token}"}

    res = client.get("/api/chatbot/history/user_2_session_alpha", headers=headers)
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cookie & Refresh Token Endpoint Flow
# ─────────────────────────────────────────────────────────────────────────────

def test_refresh_token_missing_cookie():
    """Calling /api/auth/refresh without cookie yields 401."""
    res = client.post("/api/auth/refresh")
    assert res.status_code == 401
    assert "missing" in res.json()["detail"].lower()


def test_logout_endpoint_clears_cookie():
    """Calling /api/auth/logout clears refresh cookie."""
    res = client.post("/api/auth/logout")
    assert res.status_code == 200
    assert "Logged out successfully" in res.json()["message"]
    # Check that Set-Cookie header has empty or expired refresh_token
    cookie_header = res.headers.get("set-cookie", "")
    assert "refresh_token=" in cookie_header
