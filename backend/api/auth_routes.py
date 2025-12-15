"""
Authentication routes for user registration and login.
"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from datetime import timedelta

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

router = APIRouter()
security = HTTPBearer()


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
async def register(request: RegisterRequest):
    """Register a new user."""
    try:
        user_db = create_user(
            username=request.username,
            password=request.password,
            email=request.email
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
async def login(request: LoginRequest):
    """Login and get access token."""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
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

