"""
CSRF protection middleware.
"""
import secrets
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from .logging_config import get_logger

logger = get_logger("csrf")

# CSRF token storage (in production, use Redis or database)
_csrf_tokens = {}


def generate_csrf_token() -> str:
    """Generate a CSRF token."""
    return secrets.token_urlsafe(32)


def validate_csrf_token(token: str, session_id: Optional[str] = None) -> bool:
    """Validate a CSRF token."""
    if not token:
        return False
    
    # In production, validate against stored tokens
    # For now, we'll use a simple check
    if session_id and session_id in _csrf_tokens:
        return _csrf_tokens[session_id] == token
    
    return True  # Allow if no session (for API-only endpoints)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware."""
    
    def __init__(self, app, exempt_paths: list = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/ws",  # WebSocket connections
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request with CSRF protection."""
        # Skip CSRF check for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Skip CSRF check for GET, HEAD, OPTIONS
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            return await call_next(request)
        
        # Get CSRF token from header
        csrf_token = request.headers.get("X-CSRF-Token")
        if not csrf_token:
            logger.warning(
                "csrf_token_missing",
                path=request.url.path,
                method=request.method,
                client_ip=request.client.host if request.client else "unknown"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token missing"
            )
        
        # Get session ID from cookie or header
        session_id = request.cookies.get("session_id") or request.headers.get("X-Session-ID")
        
        # Validate token
        if not validate_csrf_token(csrf_token, session_id):
            logger.warning(
                "csrf_token_invalid",
                path=request.url.path,
                method=request.method,
                client_ip=request.client.host if request.client else "unknown"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid CSRF token"
            )
        
        response = await call_next(request)
        return response


def get_csrf_token(request: Request) -> str:
    """Get or generate a CSRF token for the current session."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = secrets.token_urlsafe(16)
    
    if session_id not in _csrf_tokens:
        _csrf_tokens[session_id] = generate_csrf_token()
    
    return _csrf_tokens[session_id]

