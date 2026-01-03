"""
Database module for SQLite storage of games and steps.
Optimized for parallel execution with WAL mode and batched writes.
"""
import sqlite3
import json
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from collections import deque
import time

# Database file path
# Use test database if TEST_DB environment variable is set, otherwise use production
import os
if os.getenv("TEST_DB") == "1" or os.getenv("USE_TEST_DB") == "1":
    DB_PATH = Path(__file__).parent.parent / "catan_test.db"
else:
    DB_PATH = Path(__file__).parent.parent / "catan.db"

# Process-local storage for database connections (works in both threading and multiprocessing)
# Use a simple dict keyed by process/thread ID
_connection_cache = {}
_connection_lock = threading.Lock()

# Process-local write queue for batched writes (each process has its own)
# In multiprocessing, each process gets its own copy of this dict
_write_queue = {}
_write_lock = threading.Lock()  # This is per-process, so threading.Lock is fine
_write_batch_size = 50  # Write in batches of 50 steps
_write_timeout = 0.5  # Write after 0.5 seconds even if batch not full


def get_db_connection():
    """Get a database connection (process-local for better concurrency)."""
    import os
    import threading
    
    # Use process ID + thread ID as key (works in both threading and multiprocessing)
    process_id = os.getpid()
    thread_id = threading.get_ident()
    cache_key = (process_id, thread_id)
    
    with _connection_lock:
        if cache_key not in _connection_cache:
            conn = sqlite3.connect(str(DB_PATH), timeout=30.0)  # 30 second timeout for concurrent access
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent reads/writes
            conn.execute("PRAGMA journal_mode=WAL")
            # Optimize for performance
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL, still safe
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            # Disable foreign key checks for faster inserts
            conn.execute("PRAGMA foreign_keys=OFF")
            _connection_cache[cache_key] = conn
        return _connection_cache[cache_key]


def close_db_connection():
    """Close the process-local database connection."""
    import os
    import threading
    
    process_id = os.getpid()
    thread_id = threading.get_ident()
    cache_key = (process_id, thread_id)
    
    with _connection_lock:
        if cache_key in _connection_cache:
            _connection_cache[cache_key].close()
            del _connection_cache[cache_key]


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
            reasoning TEXT,
            raw_llm_response TEXT,
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    """)
    
    # Add reasoning column if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE steps ADD COLUMN reasoning TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Add raw_llm_response column if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE steps ADD COLUMN raw_llm_response TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Create index on game_id for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_steps_game_id ON steps(game_id)
    """)
    
    # Create index on step_idx for ordering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_steps_game_step ON steps(game_id, step_idx)
    """)

    # ---------------------------------------------------------------------
    # Drills tables (curated "best action" datasets for agent evaluation)
    # ---------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            name TEXT,
            guideline_text TEXT,
            source_game_id TEXT,
            source_step_idx INTEGER,
            player_id TEXT,
            metadata TEXT
        )
    """)

    # Add guideline_text column if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE drills ADD COLUMN guideline_text TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drill_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drill_id INTEGER NOT NULL,
            idx INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            state_json TEXT NOT NULL,
            expected_action_json TEXT NOT NULL,
            state_text TEXT,
            legal_actions_text TEXT,
            correct_actions_json TEXT,
            incorrect_actions_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (drill_id) REFERENCES drills(id)
        )
    """)
    
    # Add new columns for enhanced drill format (backward compatible)
    try:
        cursor.execute("ALTER TABLE drill_steps ADD COLUMN correct_actions_json TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE drill_steps ADD COLUMN incorrect_actions_json TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_drills_created_at ON drills(created_at)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_drill_steps_drill_idx ON drill_steps(drill_id, idx)
    """)
    
    # ---------------------------------------------------------------------
    # Optimized prompts table (for DSPy-optimized system prompts)
    # ---------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS optimized_prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            system_prompt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            is_default BOOLEAN DEFAULT FALSE
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_optimized_prompts_name ON optimized_prompts(name)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_optimized_prompts_default ON optimized_prompts(is_default)
    """)
    
    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL")
    
    conn.commit()
    
    # Initialize guidelines database (separate call to avoid circular import)
    # Do this lazily to avoid blocking on startup
    # Guidelines DB will be initialized when first used
    
    # Don't close - keep connection for thread


# =============================================================================
# Drills persistence helpers
# =============================================================================

def create_drill(
    *,
    name: Optional[str],
    guideline_text: Optional[str],
    source_game_id: Optional[str],
    source_step_idx: Optional[int],
    player_id: str,
    metadata: Optional[Dict[str, Any]],
    steps: List[Dict[str, Any]],
) -> int:
    """
    Create a drill and its ordered drill_steps.

    steps item format:
      {
        "idx": int,
        "player_id": str,
        "state_json": dict,
        "expected_action_json": dict,
        "state_text": Optional[str],
        "legal_actions_text": Optional[str],
        "correct_actions_json": Optional[List[dict]],
        "incorrect_actions_json": Optional[List[dict]],
      }
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO drills (name, guideline_text, source_game_id, source_step_idx, player_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            guideline_text,
            source_game_id,
            source_step_idx,
            player_id,
            json.dumps(metadata) if metadata else None,
        ),
    )
    drill_id = int(cursor.lastrowid)

    cursor.executemany(
        """
        INSERT INTO drill_steps (
            drill_id, idx, player_id,
            state_json, expected_action_json,
            state_text, legal_actions_text,
            correct_actions_json, incorrect_actions_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                drill_id,
                int(step["idx"]),
                step["player_id"],
                json.dumps(step["state_json"]),
                json.dumps(step["expected_action_json"]),
                step.get("state_text"),
                step.get("legal_actions_text"),
                json.dumps(step.get("correct_actions_json")) if step.get("correct_actions_json") else None,
                json.dumps(step.get("incorrect_actions_json")) if step.get("incorrect_actions_json") else None,
            )
            for step in steps
        ],
    )

    conn.commit()
    return drill_id


def list_drills(limit: int = 200) -> List[sqlite3.Row]:
    """List drills newest-first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Use LEFT JOIN with GROUP BY instead of correlated subquery for better performance
    cursor.execute(
        """
        SELECT
            d.id,
            d.created_at,
            d.name,
            d.guideline_text,
            d.source_game_id,
            d.source_step_idx,
            d.player_id,
            d.metadata,
            COUNT(ds.id) AS num_steps
        FROM drills d
        LEFT JOIN drill_steps ds ON ds.drill_id = d.id
        GROUP BY d.id
        ORDER BY d.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return cursor.fetchall()


def get_drill(drill_id: int) -> Optional[sqlite3.Row]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM drills WHERE id = ?", (drill_id,))
    return cursor.fetchone()


def get_drill_steps(drill_id: int) -> List[sqlite3.Row]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT *
        FROM drill_steps
        WHERE drill_id = ?
        ORDER BY idx ASC
        """,
        (drill_id,),
    )
    return cursor.fetchall()


def update_drill(
    drill_id: int,
    *,
    name: Optional[str] = None,
    guideline_text: Optional[str] = None,
) -> bool:
    """Update drill fields. Returns True if updated, False if drill not found."""
    fields = []
    params = []
    if name is not None:
        fields.append("name = ?")
        params.append(name)
    if guideline_text is not None:
        fields.append("guideline_text = ?")
        params.append(guideline_text)
    if not fields:
        return False

    params.append(drill_id)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE drills SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    return cursor.rowcount > 0


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
    # Don't close - keep connection for thread


def get_game(game_id: str) -> Optional[sqlite3.Row]:
    """Get a game record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM games WHERE id = ?", (game_id,))
    row = cursor.fetchone()
    
    # Don't close - keep connection for thread
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
    # Don't close - keep connection for thread


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
    reasoning: Optional[str] = None,
    raw_llm_response: Optional[str] = None,
    batch_write: bool = False,
) -> None:
    """
    Add a step to the database.
    
    Args:
        batch_write: If True, queue for batched write instead of immediate write
    """
    if batch_write:
        # Queue for batched write
        should_flush = False
        with _write_lock:
            if game_id not in _write_queue:
                _write_queue[game_id] = deque()
            _write_queue[game_id].append({
                'game_id': game_id,
                'step_idx': step_idx,
                'player_id': player_id,
                'state_before_json': state_before_json,
                'state_after_json': state_after_json,
                'action_json': action_json,
                'dice_roll': dice_roll,
                'state_text': state_text,
                'legal_actions_text': legal_actions_text,
                'chosen_action_text': chosen_action_text,
                'reasoning': reasoning,
                'raw_llm_response': raw_llm_response,
            })
            
            queue_size = len(_write_queue[game_id])
            # Check if we should flush, but do it outside the lock to avoid deadlock
            if queue_size >= _write_batch_size:
                should_flush = True
        
        # Flush outside the lock to avoid deadlock
        if should_flush:
            _flush_write_queue(game_id)
    else:
        # Immediate write
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO steps (
                game_id, step_idx, player_id,
                state_before_json, state_after_json, action_json,
                dice_roll, state_text, legal_actions_text, chosen_action_text, reasoning, raw_llm_response
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            reasoning,
            raw_llm_response,
        ))
        
        conn.commit()
        # Don't close - keep connection for thread


def _flush_write_queue(game_id: str) -> None:
    """Flush the write queue for a specific game."""
    with _write_lock:
        if game_id not in _write_queue or len(_write_queue[game_id]) == 0:
            return
        
        steps = list(_write_queue[game_id])
        _write_queue[game_id].clear()
    
    # Write all steps in a single transaction
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.executemany("""
            INSERT INTO steps (
                game_id, step_idx, player_id,
                state_before_json, state_after_json, action_json,
                dice_roll, state_text, legal_actions_text, chosen_action_text, reasoning, raw_llm_response
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                step['game_id'],
                step['step_idx'],
                step['player_id'],
                json.dumps(step['state_before_json']),
                json.dumps(step['state_after_json']),
                json.dumps(step['action_json']),
                step['dice_roll'],
                step['state_text'],
                step['legal_actions_text'],
                step['chosen_action_text'],
                step.get('reasoning'),  # Use .get() for backward compatibility
                step.get('raw_llm_response'),  # Use .get() for backward compatibility
            )
            for step in steps
        ])
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e


def flush_all_write_queues() -> None:
    """Flush all pending write queues (call at end of game)."""
    with _write_lock:
        game_ids = list(_write_queue.keys())
    
    for game_id in game_ids:
        _flush_write_queue(game_id)


def get_steps(game_id: str) -> List[sqlite3.Row]:
    """Get all steps for a game, ordered by step_idx."""
    # Flush any pending writes for this game first
    _flush_write_queue(game_id)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM steps
        WHERE game_id = ?
        ORDER BY step_idx ASC
    """, (game_id,))
    
    rows = cursor.fetchall()
    # Don't close - keep connection for thread
    return rows


def get_latest_state(game_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest game state.
    
    First tries to get from the most recent step, then falls back to
    current_state_json in games table.
    """
    # Flush any pending writes for this game first
    _flush_write_queue(game_id)
    
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
        # Don't close - keep connection for thread
        return json.loads(row[0])
    
    # Fall back to current_state_json in games table
    cursor.execute("""
        SELECT current_state_json
        FROM games
        WHERE id = ?
    """, (game_id,))
    
    row = cursor.fetchone()
    # Don't close - keep connection for thread
    
    if row and row[0]:
        return json.loads(row[0])
    return None


def get_step_count(game_id: str) -> int:
    """Get the number of steps for a game."""
    # Flush any pending writes for this game first
    _flush_write_queue(game_id)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM steps
        WHERE game_id = ?
    """, (game_id,))
    
    row = cursor.fetchone()
    # Don't close - keep connection for thread
    
    return row[0] if row else 0


def get_state_at_step(game_id: str, step_idx: int, use_state_before: bool = True) -> Optional[Dict[str, Any]]:
    """Get game state at a specific step index.
    
    Args:
        game_id: The game ID
        step_idx: The step index (0-based)
        use_state_before: If True, returns state_before_json. If False, returns state_after_json.
        
    Returns:
        Game state dictionary, or None if step not found.
    """
    # Flush any pending writes for this game first
    _flush_write_queue(game_id)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    state_column = "state_before_json" if use_state_before else "state_after_json"
    cursor.execute(f"""
        SELECT {state_column}
        FROM steps
        WHERE game_id = ? AND step_idx = ?
    """, (game_id, step_idx))
    
    row = cursor.fetchone()
    # Don't close - keep connection for thread
    
    if row and row[0]:
        return json.loads(row[0])
    return None


# =============================================================================
# Optimized prompts persistence helpers
# =============================================================================

def save_optimized_prompt(
    name: str,
    system_prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
    is_default: bool = False
) -> int:
    """
    Save an optimized prompt to the database.
    
    Args:
        name: Unique name for the prompt
        system_prompt: The optimized system prompt text
        metadata: Optional metadata (optimization params, performance metrics, etc.)
        is_default: Whether this should be the default prompt
        
    Returns:
        The ID of the saved prompt
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # If setting as default, unset other defaults
    if is_default:
        cursor.execute("UPDATE optimized_prompts SET is_default = FALSE WHERE is_default = TRUE")
    
    cursor.execute(
        """
        INSERT OR REPLACE INTO optimized_prompts (name, system_prompt, metadata, is_default)
        VALUES (?, ?, ?, ?)
        """,
        (
            name,
            system_prompt,
            json.dumps(metadata) if metadata else None,
            is_default,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def get_optimized_prompt(name: str) -> Optional[sqlite3.Row]:
    """Get an optimized prompt by name."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM optimized_prompts WHERE name = ?", (name,))
    return cursor.fetchone()


def get_default_optimized_prompt() -> Optional[sqlite3.Row]:
    """Get the default optimized prompt."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM optimized_prompts WHERE is_default = TRUE LIMIT 1")
    return cursor.fetchone()


def list_optimized_prompts() -> List[sqlite3.Row]:
    """List all optimized prompts, newest first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM optimized_prompts ORDER BY created_at DESC"
    )
    return cursor.fetchall()


def set_default_prompt(name: str) -> bool:
    """
    Set a prompt as the default.
    
    Returns:
        True if successful, False if prompt not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # First check if prompt exists
    cursor.execute("SELECT id FROM optimized_prompts WHERE name = ?", (name,))
    if not cursor.fetchone():
        return False
    
    # Unset other defaults
    cursor.execute("UPDATE optimized_prompts SET is_default = FALSE WHERE is_default = TRUE")
    
    # Set this one as default
    cursor.execute("UPDATE optimized_prompts SET is_default = TRUE WHERE name = ?", (name,))
    conn.commit()
    return cursor.rowcount > 0


def delete_optimized_prompt(name: str) -> bool:
    """
    Delete an optimized prompt.
    
    Returns:
        True if deleted, False if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM optimized_prompts WHERE name = ?", (name,))
    conn.commit()
    return cursor.rowcount > 0

