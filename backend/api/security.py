"""
Security utilities and middleware.
"""
import re
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer
from .database import get_db_connection
from .logging_config import get_logger, activity_logger

logger = get_logger("security")

# Account lockout configuration
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15
LOGIN_ATTEMPTS_TABLE = "login_attempts"


def init_security_tables():
    """Initialize security-related database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Login attempts tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            ip_address TEXT,
            success INTEGER NOT NULL DEFAULT 0,
            attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            locked_until TIMESTAMP
        )
    """)
    
    # Create index for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_login_attempts_username_time 
        ON login_attempts(username, attempted_at DESC)
    """)
    
    conn.commit()


def check_password_strength(password: str) -> tuple:
    """
    Check password strength.
    Returns (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if len(password) > 128:
        return False, "Password must be at most 128 characters long"
    
    # Check for at least one uppercase letter
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    # Check for at least one lowercase letter
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    # Check for at least one digit
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    # Check for at least one special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"
    
    # Check for common weak passwords
    common_passwords = ['password', '12345678', 'qwerty', 'abc123', 'password123']
    if password.lower() in common_passwords:
        return False, "Password is too common. Please choose a stronger password"
    
    return True, ""


def record_login_attempt(username: str, success: bool, ip_address: Optional[str] = None):
    """Record a login attempt."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    locked_until = None
    if not success:
        # Check if account should be locked
        recent_failures = cursor.execute("""
            SELECT COUNT(*) FROM login_attempts
            WHERE username = ? AND success = 0 
            AND attempted_at > datetime('now', '-' || ? || ' minutes')
        """, (username, LOCKOUT_DURATION_MINUTES)).fetchone()[0]
        
        if recent_failures >= MAX_LOGIN_ATTEMPTS - 1:
            locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
    
    cursor.execute("""
        INSERT INTO login_attempts (username, ip_address, success, locked_until)
        VALUES (?, ?, ?, ?)
    """, (username, ip_address, 1 if success else 0, locked_until))
    
    conn.commit()
    
    # Log the attempt
    activity_logger.log_user_action(
        user_id="",  # Not available before login
        username=username,
        action="login_attempt",
        details={"success": success, "ip_address": ip_address},
        ip_address=ip_address
    )


def is_account_locked(username: str) -> tuple:
    """
    Check if an account is locked.
    Returns (is_locked, locked_until)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get the most recent lock
    cursor.execute("""
        SELECT locked_until FROM login_attempts
        WHERE username = ? AND locked_until IS NOT NULL
        ORDER BY attempted_at DESC
        LIMIT 1
    """, (username,))
    
    result = cursor.fetchone()
    if not result:
        return False, None
    
    locked_until_str = result[0]
    if locked_until_str:
        try:
            locked_until = datetime.fromisoformat(locked_until_str.replace('Z', '+00:00'))
            if locked_until > datetime.utcnow():
                return True, locked_until
        except Exception as e:
            logger.error("error_parsing_lock_time", error=str(e), locked_until=locked_until_str)
    
    return False, None


def get_recent_failed_attempts(username: str, minutes: int = LOCKOUT_DURATION_MINUTES) -> int:
    """Get the number of recent failed login attempts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM login_attempts
        WHERE username = ? AND success = 0 
        AND attempted_at > datetime('now', '-' || ? || ' minutes')
    """, (username, minutes))
    
    return cursor.fetchone()[0]


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    # Check for forwarded IP (from proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct client IP
    if request.client:
        return request.client.host
    
    return "unknown"


def require_https(request: Request):
    """Check if request is over HTTPS (in production)."""
    # In development, allow HTTP
    import os
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        # Check if request is secure
        is_secure = (
            request.url.scheme == "https" or
            request.headers.get("X-Forwarded-Proto") == "https" or
            request.headers.get("X-Forwarded-Ssl") == "on"
        )
        
        if not is_secure:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="HTTPS is required in production"
            )

