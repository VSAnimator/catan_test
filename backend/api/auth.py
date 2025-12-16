"""
Authentication module for user management and JWT tokens.
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel
from fastapi import HTTPException, status
import secrets
import hashlib
import os
from dotenv import load_dotenv
from .database import get_db_connection

# Load environment variables
load_dotenv()

# Secret key for JWT (from environment variable, fallback to generated)
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))  # Default 7 days


class User(BaseModel):
    """User model."""
    id: str
    username: str
    email: Optional[str] = None
    created_at: str


class UserInDB(User):
    """User in database with hashed password."""
    hashed_password: str


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    user: User


class RegisterRequest(BaseModel):
    """User registration request."""
    username: str
    password: str
    email: Optional[str] = None


class LoginRequest(BaseModel):
    """User login request."""
    username: str
    password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    try:
        # Bcrypt has a 72-byte limit, but normal passwords should be fine
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Hash a password."""
    # Bcrypt has a 72-byte limit, truncate if password is too long
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_user_by_username(username: str) -> Optional[UserInDB]:
    """Get user by username from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Case-insensitive search
    cursor.execute("""
        SELECT id, username, email, hashed_password, created_at
        FROM users
        WHERE LOWER(username) = LOWER(?)
    """, (username,))
    
    row = cursor.fetchone()
    if not row:
        return None
    
    return UserInDB(
        id=row['id'],
        username=row['username'],
        email=row['email'],
        hashed_password=row['hashed_password'],
        created_at=row['created_at']
    )


def get_user_by_id(user_id: str) -> Optional[UserInDB]:
    """Get user by ID from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, email, hashed_password, created_at
        FROM users
        WHERE id = ?
    """, (user_id,))
    
    row = cursor.fetchone()
    if not row:
        return None
    
    return UserInDB(
        id=row['id'],
        username=row['username'],
        email=row['email'],
        hashed_password=row['hashed_password'],
        created_at=row['created_at']
    )


def create_user(username: str, password: str, email: Optional[str] = None) -> UserInDB:
    """Create a new user."""
    # Check if username already exists
    if get_user_by_username(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Generate user ID
    user_id = hashlib.sha256(f"{username}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
    
    # Create user in database
    hashed_password = get_password_hash(password)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO users (id, username, email, hashed_password, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, username, email, hashed_password, datetime.utcnow().isoformat()))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create user: {str(e)}"
        )
    
    return UserInDB(
        id=user_id,
        username=username,
        email=email,
        hashed_password=hashed_password,
        created_at=datetime.utcnow().isoformat()
    )


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user."""
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def get_current_user(token: str) -> Optional[User]:
    """Get current user from token."""
    payload = verify_token(token)
    if payload is None:
        return None
    
    user_id: str = payload.get("sub")
    if user_id is None:
        return None
    
    user_db = get_user_by_id(user_id)
    if user_db is None:
        return None
    
    return User(
        id=user_db.id,
        username=user_db.username,
        email=user_db.email,
        created_at=user_db.created_at
    )

