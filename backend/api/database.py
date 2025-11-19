"""
Database module for SQLite storage of games and steps.
"""
import sqlite3
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

# Database file path
DB_PATH = Path(__file__).parent.parent / "catan.db"


def get_db_connection():
    """Get a database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_db():
    """Initialize the database with tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create games table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rng_seed INTEGER,
            metadata TEXT,
            current_state_json TEXT
        )
    """)
    
    # Create steps table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            step_idx INTEGER NOT NULL,
            player_id TEXT,
            state_before_json TEXT NOT NULL,
            state_after_json TEXT NOT NULL,
            action_json TEXT NOT NULL,
            dice_roll INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            state_text TEXT,
            legal_actions_text TEXT,
            chosen_action_text TEXT,
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    """)
    
    # Create index on game_id for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_steps_game_id ON steps(game_id)
    """)
    
    # Create index on step_idx for ordering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_steps_game_step ON steps(game_id, step_idx)
    """)
    
    conn.commit()
    conn.close()


def create_game(
    game_id: str,
    rng_seed: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    initial_state_json: Optional[Dict[str, Any]] = None,
) -> None:
    """Create a new game record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    metadata_json = json.dumps(metadata) if metadata else None
    state_json = json.dumps(initial_state_json) if initial_state_json else None
    
    cursor.execute("""
        INSERT INTO games (id, rng_seed, metadata, current_state_json)
        VALUES (?, ?, ?, ?)
    """, (game_id, rng_seed, metadata_json, state_json))
    
    conn.commit()
    conn.close()


def get_game(game_id: str) -> Optional[sqlite3.Row]:
    """Get a game record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM games WHERE id = ?", (game_id,))
    row = cursor.fetchone()
    
    conn.close()
    return row


def save_game_state(game_id: str, state_json: Dict[str, Any]) -> None:
    """Save the current game state as the latest snapshot."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE games
        SET current_state_json = ?
        WHERE id = ?
    """, (json.dumps(state_json), game_id))
    
    conn.commit()
    conn.close()


def add_step(
    game_id: str,
    step_idx: int,
    player_id: str,
    state_before_json: Dict[str, Any],
    state_after_json: Dict[str, Any],
    action_json: Dict[str, Any],
    dice_roll: Optional[int] = None,
    state_text: Optional[str] = None,
    legal_actions_text: Optional[str] = None,
    chosen_action_text: Optional[str] = None,
) -> None:
    """Add a step to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO steps (
            game_id, step_idx, player_id,
            state_before_json, state_after_json, action_json,
            dice_roll, state_text, legal_actions_text, chosen_action_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_id,
        step_idx,
        player_id,
        json.dumps(state_before_json),
        json.dumps(state_after_json),
        json.dumps(action_json),
        dice_roll,
        state_text,
        legal_actions_text,
        chosen_action_text,
    ))
    
    conn.commit()
    conn.close()


def get_steps(game_id: str) -> List[sqlite3.Row]:
    """Get all steps for a game, ordered by step_idx."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM steps
        WHERE game_id = ?
        ORDER BY step_idx ASC
    """, (game_id,))
    
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_latest_state(game_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest game state.
    
    First tries to get from the most recent step, then falls back to
    current_state_json in games table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Try to get from most recent step
    cursor.execute("""
        SELECT state_after_json
        FROM steps
        WHERE game_id = ?
        ORDER BY step_idx DESC
        LIMIT 1
    """, (game_id,))
    
    row = cursor.fetchone()
    if row:
        conn.close()
        return json.loads(row[0])
    
    # Fall back to current_state_json in games table
    cursor.execute("""
        SELECT current_state_json
        FROM games
        WHERE id = ?
    """, (game_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0]:
        return json.loads(row[0])
    return None


def get_step_count(game_id: str) -> int:
    """Get the number of steps for a game."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM steps
        WHERE game_id = ?
    """, (game_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    return row[0] if row else 0

