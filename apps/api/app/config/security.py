import ipaddress
import logging
import socket
from urllib.parse import urlparse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)

# Singleton slowapi Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects comprehensive security headers into all responses:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - Strict-Transport-Security (HSTS)
    - Referrer-Policy
    - Content-Security-Policy (CSP)
    - Permissions-Policy
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        # Robust Content-Security-Policy supporting API usage and embedded trailer players
        csp_policy = (
            "default-src 'self'; "
            "frame-src 'self' https://www.youtube.com https://www.youtube-nocookie.com; "
            "img-src 'self' data: https:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "connect-src 'self' http://localhost:* http://127.0.0.1:* ws://localhost:* ws://127.0.0.1:* https:;"
        )
        response.headers["Content-Security-Policy"] = csp_policy

        return response


# Blocked IP ranges for SSRF mitigation
BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback IPv4
    ipaddress.ip_network("10.0.0.0/8"),       # RFC 1918 Private
    ipaddress.ip_network("172.16.0.0/12"),    # RFC 1918 Private
    ipaddress.ip_network("192.168.0.0/16"),   # RFC 1918 Private
    ipaddress.ip_network("169.254.0.0/16"),   # IPv4 Link-local / AWS & Cloud Metadata
    ipaddress.ip_network("::1/128"),          # Loopback IPv6
    ipaddress.ip_network("fc00::/7"),         # Unique Local Address IPv6
    ipaddress.ip_network("fe80::/10"),        # Link-local IPv6
]


def validate_safe_url(target_url: str) -> bool:
    """
    Validates external URLs to prevent Server-Side Request Forgery (SSRF).
    Blocks private IP addresses, loopback addresses, metadata IP (169.254.169.254),
    and non-HTTP/HTTPS protocols.
    """
    if not target_url or not isinstance(target_url, str):
        return False

    try:
        parsed = urlparse(target_url.strip())
        if parsed.scheme.lower() not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Attempt to resolve hostname to IP
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip_obj = ipaddress.ip_address(ip_str)
                for net in BLOCKED_NETWORKS:
                    if ip_obj in net:
                        logger.warning(f"SSRF block: {target_url} resolves to private/internal IP {ip_str}")
                        return False
        except socket.gaierror:
            # Domain could not be resolved
            return False

        return True
    except Exception as e:
        logger.warning(f"SSRF validation exception for {target_url}: {e}")
        return False
