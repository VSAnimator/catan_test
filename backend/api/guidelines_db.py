"""
Database operations for storing and retrieving guidelines and feedback for LLM agents.
"""
import json
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from api.database import get_db_connection
# Note: Don't import init_db here to avoid circular import


_guidelines_db_initialized = False
_guidelines_db_lock = __import__('threading').Lock()

def init_guidelines_db():
    """Initialize the guidelines database tables."""
    global _guidelines_db_initialized
    
    # Use a lock to ensure thread-safe initialization
    with _guidelines_db_lock:
        if _guidelines_db_initialized:
            return
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create guidelines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS guidelines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                guideline_text TEXT NOT NULL,
                context TEXT,
                priority INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active BOOLEAN DEFAULT 1
            )
        """)
        
        # Create feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                step_idx INTEGER,
                player_id TEXT,
                action_taken TEXT,
                feedback_text TEXT NOT NULL,
                feedback_type TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_guidelines_player_active 
            ON guidelines(player_id, active)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_game_step 
            ON feedback(game_id, step_idx)
        """)
        
        conn.commit()
        _guidelines_db_initialized = True


def add_guideline(
    guideline_text: str,
    player_id: Optional[str] = None,
    context: Optional[str] = None,
    priority: int = 0
) -> int:
    """
    Add a new guideline.
    
    Args:
        guideline_text: The guideline text
        player_id: Optional player ID (None for global guidelines)
        context: Optional context (e.g., "early_game", "trading", etc.)
        priority: Priority level (higher = more important)
        
    Returns:
        ID of the created guideline
    """
    init_guidelines_db()  # Ensure tables exist
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO guidelines (player_id, guideline_text, context, priority)
        VALUES (?, ?, ?, ?)
    """, (player_id, guideline_text, context, priority))
    
    conn.commit()
    return cursor.lastrowid


def get_guidelines(
    player_id: Optional[str] = None,
    context: Optional[str] = None,
    active_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Get guidelines.
    
    Args:
        player_id: Optional player ID (None for all players + global)
        context: Optional context filter
        active_only: Only return active guidelines
        
    Returns:
        List of guideline dictionaries
    """
    init_guidelines_db()  # Ensure tables exist
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM guidelines WHERE 1=1"
    params = []
    
    if active_only:
        query += " AND active = 1"
    
    if player_id is not None:
        query += " AND (player_id = ? OR player_id IS NULL)"
        params.append(player_id)
    
    if context:
        query += " AND (context = ? OR context IS NULL)"
        params.append(context)
    
    query += " ORDER BY priority DESC, created_at DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    return [
        {
            "id": row["id"],
            "player_id": row["player_id"],
            "guideline_text": row["guideline_text"],
            "context": row["context"],
            "priority": row["priority"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "active": bool(row["active"])
        }
        for row in rows
    ]


def update_guideline(guideline_id: int, **kwargs) -> bool:
    """
    Update a guideline.
    
    Args:
        guideline_id: ID of the guideline to update
        **kwargs: Fields to update (guideline_text, context, priority, active)
        
    Returns:
        True if updated, False if not found
    """
    init_guidelines_db()  # Ensure tables exist
    conn = get_db_connection()
    cursor = conn.cursor()
    
    allowed_fields = ["guideline_text", "context", "priority", "active"]
    updates = []
    params = []
    
    for key, value in kwargs.items():
        if key in allowed_fields:
            updates.append(f"{key} = ?")
            params.append(value)
    
    if not updates:
        return False
    
    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(guideline_id)
    
    cursor.execute(
        f"UPDATE guidelines SET {', '.join(updates)} WHERE id = ?",
        params
    )
    
    conn.commit()
    return cursor.rowcount > 0


def delete_guideline(guideline_id: int) -> bool:
    """
    Delete (deactivate) a guideline.
    
    Args:
        guideline_id: ID of the guideline to delete
        
    Returns:
        True if deleted, False if not found
    """
    return update_guideline(guideline_id, active=False)


def add_feedback(
    game_id: str,
    feedback_text: str,
    step_idx: Optional[int] = None,
    player_id: Optional[str] = None,
    action_taken: Optional[str] = None,
    feedback_type: str = "general"
) -> int:
    """
    Add feedback for a specific move.
    
    Args:
        game_id: Game ID
        feedback_text: Feedback text
        step_idx: Optional step index
        player_id: Optional player ID
        action_taken: Optional action that was taken
        feedback_type: Type of feedback (e.g., "positive", "negative", "suggestion")
        
    Returns:
        ID of the created feedback
    """
    init_guidelines_db()  # Ensure tables exist
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO feedback (game_id, step_idx, player_id, action_taken, feedback_text, feedback_type)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (game_id, step_idx, player_id, action_taken, feedback_text, feedback_type))
    
    conn.commit()
    return cursor.lastrowid


def get_feedback(
    game_id: Optional[str] = None,
    player_id: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get feedback.
    
    Args:
        game_id: Optional game ID filter
        player_id: Optional player ID filter
        limit: Maximum number of results
        
    Returns:
        List of feedback dictionaries
    """
    init_guidelines_db()  # Ensure tables exist
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM feedback WHERE 1=1"
    params = []
    
    if game_id:
        query += " AND game_id = ?"
        params.append(game_id)
    
    if player_id:
        query += " AND player_id = ?"
        params.append(player_id)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    return [
        {
            "id": row["id"],
            "game_id": row["game_id"],
            "step_idx": row["step_idx"],
            "player_id": row["player_id"],
            "action_taken": row["action_taken"],
            "feedback_text": row["feedback_text"],
            "feedback_type": row["feedback_type"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]

