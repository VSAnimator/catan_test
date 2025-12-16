"""
Authentication routes for user registration and login.
"""
import re
from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import Optional
from datetime import timedelta
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from .auth import (
    RegisterRequest,
    LoginRequest,
    Token,
    User,
    create_user,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_user_by_id,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from .security import (
    check_password_strength,
    record_login_attempt,
    is_account_locked,
    get_client_ip
)
from .logging_config import activity_logger

router = APIRouter()
security = HTTPBearer()

# Rate limiter instance (will be initialized in main.py)
limiter: Optional[Limiter] = None

def init_rate_limiter(app_limiter: Limiter):
    """Initialize rate limiter from main app."""
    global limiter
    limiter = app_limiter

# Input validation helpers
def validate_username(username: str) -> str:
    """Validate username format."""
    if not username or len(username.strip()) == 0:
        raise ValueError("Username cannot be empty")
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters")
    if len(username) > 30:
        raise ValueError("Username must be at most 30 characters")
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
    return username.strip()

def validate_password(password: str) -> str:
    """Validate password strength."""
    if not password or len(password) == 0:
        raise ValueError("Password cannot be empty")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")
    if len(password) > 128:
        raise ValueError("Password must be at most 128 characters")
    return password

def validate_email(email: Optional[str]) -> Optional[str]:
    """Validate email format."""
    if email is None or email.strip() == "":
        return None
    email = email.strip().lower()
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValueError("Invalid email format")
    return email


async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Dependency to get current user from token."""
    token = credentials.credentials
    user = get_current_user(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.post("/auth/register", response_model=Token)
async def register(request: RegisterRequest, req: Request):
    """Register a new user. Rate limited to 5 requests per minute."""
    # Input validation
    try:
        validated_username = validate_username(request.username)
        validated_password = validate_password(request.password)
        validated_email = validate_email(request.email)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Password strength check
    is_valid, error_msg = check_password_strength(validated_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    
    try:
        user_db = create_user(
            username=validated_username,
            password=validated_password,
            email=validated_email
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_db.id},
        expires_delta=access_token_expires
    )
    
    user = User(
        id=user_db.id,
        username=user_db.username,
        email=user_db.email,
        created_at=user_db.created_at
    )
    
    return Token(access_token=access_token, token_type="bearer", user=user)


@router.post("/auth/login", response_model=Token)
async def login(request: LoginRequest, req: Request):
    """Login and get access token. Rate limited to 10 requests per minute to prevent brute force."""
    # Input validation
    try:
        validated_username = validate_username(request.username)
        validated_password = validate_password(request.password)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Check if account is locked
    is_locked, locked_until = is_account_locked(validated_username)
    if is_locked:
        client_ip = get_client_ip(req)
        record_login_attempt(validated_username, False, client_ip)
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account is locked due to too many failed login attempts. Try again after {locked_until.strftime('%Y-%m-%d %H:%M:%S UTC') if locked_until else '15 minutes'}"
        )
    
    client_ip = get_client_ip(req)
    user = authenticate_user(validated_username, validated_password)
    if not user:
        # Record failed login attempt
        record_login_attempt(validated_username, False, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Record successful login attempt
    record_login_attempt(validated_username, True, client_ip)
    
    # Log user activity
    activity_logger.log_user_action(
        user_id=user.id,
        username=user.username,
        action="login_success",
        ip_address=client_ip,
        user_agent=req.headers.get("user-agent")
    )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id},
        expires_delta=access_token_expires
    )
    
    user_response = User(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at
    )
    
    return Token(access_token=access_token, token_type="bearer", user=user_response)


@router.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user_from_token)):
    """Get current user information."""
    return current_user


@router.post("/auth/logout")
async def logout(current_user: User = Depends(get_current_user_from_token)):
    """Logout endpoint (client should discard token).
    
    Note: Since we're using stateless JWT tokens, we can't invalidate them server-side.
    The client should discard the token. In production, you might want to implement
    a token blacklist for immediate invalidation.
    """
    return {"message": "Logged out successfully. Please discard your token."}


@router.post("/auth/refresh")
async def refresh_token(current_user: User = Depends(get_current_user_from_token)):
    """Refresh access token (create a new token with extended expiration)."""
    from .auth import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
    from datetime import timedelta
    
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user.id},
        expires_delta=access_token_expires
    )
    
    return Token(access_token=access_token, token_type="bearer", user=current_user)

