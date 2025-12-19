#!/usr/bin/env python3
"""
Clean up the database by copying relevant games to a new database:
1. The last 20 game IDs (by created_at)
2. Any game IDs that are necessary for unit tests

Then replaces the old database with the new one.
"""
import sqlite3
import json
from pathlib import Path
import sys
import argparse
import shutil

# Add parent directory to path to import database module
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import DB_PATH

# Game IDs used in unit tests (from test files)
TEST_GAME_IDS = {
    "test_game",  # Used in test_engine_basic.py and test_serialization.py
    "21fe6375-9825-4b3b-accd-9e7335d87b6f",  # From test_registry.json
    "0ab49856-0674-4fde-a94d-4ca17aa44f4c",  # From test_registry.json and test_llm_discard.py
    "f226af0d-26c5-4711-a12c-0fa22c8ada32",  # From test_registry.json and test_agent_discard_auto_advance.py
}


def get_last_20_game_ids(conn):
    """Get the last 20 game IDs ordered by created_at DESC."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id 
        FROM games 
        ORDER BY created_at DESC 
        LIMIT 20
    """)
    rows = cursor.fetchall()
    return [row[0] for row in rows]


def get_all_game_ids(conn):
    """Get all game IDs in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM games")
    rows = cursor.fetchall()
    return [row[0] for row in rows]


def get_game_ids_to_keep(conn):
    """Get the set of game IDs to keep."""
    last_20 = get_last_20_game_ids(conn)
    all_test_ids = TEST_GAME_IDS.copy()
    
    # Combine both sets
    game_ids_to_keep = set(last_20) | all_test_ids
    
    return game_ids_to_keep


def copy_games_to_new_db(old_conn, new_conn, game_ids_to_keep):
    """Copy games and their steps to the new database."""
    if not game_ids_to_keep:
        return 0, 0
    
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()
    
    # Filter to only game IDs that actually exist
    placeholders = ','.join(['?'] * len(game_ids_to_keep))
    
    # Copy games
    old_cursor.execute(f"""
        SELECT id, created_at, rng_seed, metadata, current_state_json
        FROM games
        WHERE id IN ({placeholders})
    """, list(game_ids_to_keep))
    
    games = old_cursor.fetchall()
    games_count = len(games)
    
    if games_count > 0:
        new_cursor.executemany("""
            INSERT INTO games (id, created_at, rng_seed, metadata, current_state_json)
            VALUES (?, ?, ?, ?, ?)
        """, games)
    
    # Copy steps for these games
    old_cursor.execute(f"""
        SELECT 
            game_id, step_idx, player_id,
            state_before_json, state_after_json, action_json,
            dice_roll, timestamp, state_text, legal_actions_text,
            chosen_action_text, reasoning, raw_llm_response
        FROM steps
        WHERE game_id IN ({placeholders})
        ORDER BY game_id, step_idx
    """, list(game_ids_to_keep))
    
    steps = old_cursor.fetchall()
    steps_count = len(steps)
    
    if steps_count > 0:
        new_cursor.executemany("""
            INSERT INTO steps (
                game_id, step_idx, player_id,
                state_before_json, state_after_json, action_json,
                dice_roll, timestamp, state_text, legal_actions_text,
                chosen_action_text, reasoning, raw_llm_response
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, steps)
    
    # Copy users table (keep all users)
    old_cursor.execute("""
        SELECT id, username, email, hashed_password, created_at
        FROM users
    """)
    users = old_cursor.fetchall()
    users_count = len(users)
    
    if users_count > 0:
        new_cursor.executemany("""
            INSERT INTO users (id, username, email, hashed_password, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, users)
    
    new_conn.commit()
    
    return games_count, steps_count


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description='Clean up database by keeping last 20 games and test games')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Database Cleanup Script")
    print("=" * 60)
    print()
    
    # Check if database exists
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return 1
    
    print(f"Database path: {DB_PATH}")
    
    # Get database file info
    db_size = DB_PATH.stat().st_size / (1024 * 1024)  # Size in MB
    print(f"Current database size: {db_size:.2f} MB")
    print()
    
    # Connect to old database
    old_conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    old_conn.row_factory = sqlite3.Row
    
    try:
        # Get current stats
        all_game_ids = get_all_game_ids(old_conn)
        total_games = len(all_game_ids)
        print(f"Total games in database: {total_games}")
        
        # Get game IDs to keep
        game_ids_to_keep = get_game_ids_to_keep(old_conn)
        # Filter to only IDs that actually exist
        game_ids_to_keep = {gid for gid in game_ids_to_keep if gid in all_game_ids}
        
        print(f"Game IDs to keep: {len(game_ids_to_keep)}")
        last_20 = get_last_20_game_ids(old_conn)
        print(f"  - Last 20 games: {len(last_20)}")
        print(f"  - Test game IDs found: {len([gid for gid in TEST_GAME_IDS if gid in all_game_ids])}")
        print()
        
        # Show which test game IDs are actually in the database
        test_ids_in_db = [gid for gid in TEST_GAME_IDS if gid in all_game_ids]
        if test_ids_in_db:
            print(f"Test game IDs found in database:")
            for gid in test_ids_in_db:
                print(f"  - {gid}")
        print()
        
        if not game_ids_to_keep:
            print("No games to keep. This would result in an empty database!")
            return 1
        
        games_to_delete = total_games - len(game_ids_to_keep)
        
        if games_to_delete == 0:
            print("No games to delete. Database is already clean!")
            return 0
        
        print(f"Games to remove: {games_to_delete}")
        print()
        
        # Get step counts
        old_cursor = old_conn.cursor()
        placeholders = ','.join(['?'] * len(game_ids_to_keep)) if game_ids_to_keep else 'NULL'
        if game_ids_to_keep:
            old_cursor.execute(f"""
                SELECT COUNT(*) 
                FROM steps 
                WHERE game_id NOT IN ({placeholders})
            """, list(game_ids_to_keep))
        else:
            old_cursor.execute("SELECT COUNT(*) FROM steps")
        steps_to_delete = old_cursor.fetchone()[0]
        print(f"Steps to remove: {steps_to_delete}")
        print()
        
        # Confirm
        if not args.force:
            print("WARNING: This will create a new database with only the selected games!")
            print(f"  - Keeping {len(game_ids_to_keep)} games")
            print(f"  - Removing {games_to_delete} games")
            print(f"  - Removing {steps_to_delete} steps")
            print()
            response = input("Do you want to proceed? (yes/no): ")
            
            if response.lower() != 'yes':
                print("Aborted.")
                return 0
        else:
            print("WARNING: This will create a new database with only the selected games!")
            print(f"  - Keeping {len(game_ids_to_keep)} games")
            print(f"  - Removing {games_to_delete} games")
            print(f"  - Removing {steps_to_delete} steps")
            print("Proceeding with --force flag...")
            print()
        
        # Create new database file
        new_db_path = DB_PATH.parent / f"{DB_PATH.stem}_new{DB_PATH.suffix}"
        print(f"Creating new database: {new_db_path}")
        
        # Initialize new database
        new_conn = sqlite3.connect(str(new_db_path), timeout=30.0)
        new_conn.row_factory = sqlite3.Row
        
        # Initialize schema in new database
        new_cursor = new_conn.cursor()
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                rng_seed INTEGER,
                metadata TEXT,
                current_state_json TEXT
            )
        """)
        new_cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                hashed_password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        new_cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
        """)
        new_cursor.execute("""
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
        new_cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_game_id ON steps(game_id)
        """)
        new_cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_game_step ON steps(game_id, step_idx)
        """)
        new_conn.commit()
        
        print("Copying games and steps to new database...")
        games_copied, steps_copied = copy_games_to_new_db(old_conn, new_conn, game_ids_to_keep)
        
        print(f"✓ Copied {games_copied} games")
        print(f"✓ Copied {steps_copied} steps")
        print()
        
        # Close connections
        old_conn.close()
        new_conn.close()
        
        # Get new database size
        new_db_size = new_db_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"New database size: {new_db_size:.2f} MB")
        print(f"Space saved: {db_size - new_db_size:.2f} MB")
        print()
        
        # Remove old database files
        print("Removing old database files...")
        wal_path = DB_PATH.parent / f"{DB_PATH.name}-wal"
        shm_path = DB_PATH.parent / f"{DB_PATH.name}-shm"
        
        if DB_PATH.exists():
            DB_PATH.unlink()
            print(f"✓ Removed {DB_PATH.name}")
        
        if wal_path.exists():
            wal_path.unlink()
            print(f"✓ Removed {wal_path.name}")
        
        if shm_path.exists():
            shm_path.unlink()
            print(f"✓ Removed {shm_path.name}")
        
        print()
        
        # Rename new database to old name
        print(f"Renaming new database to {DB_PATH.name}...")
        new_db_path.rename(DB_PATH)
        print("✓ Database replacement complete!")
        print()
        
        # Show final stats
        final_conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
        final_cursor = final_conn.cursor()
        final_cursor.execute("SELECT COUNT(*) FROM games")
        final_games = final_cursor.fetchone()[0]
        final_cursor.execute("SELECT COUNT(*) FROM steps")
        final_steps = final_cursor.fetchone()[0]
        final_conn.close()
        
        print(f"Final database stats:")
        print(f"  - Games: {final_games}")
        print(f"  - Steps: {final_steps}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up new database if it exists
        new_db_path = DB_PATH.parent / f"{DB_PATH.stem}_new{DB_PATH.suffix}"
        if new_db_path.exists():
            print(f"\nCleaning up new database file: {new_db_path}")
            new_db_path.unlink()
        
        return 1
    finally:
        if 'old_conn' in locals():
            old_conn.close()


if __name__ == "__main__":
    sys.exit(main())
