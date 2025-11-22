#!/usr/bin/env python3
"""
Run a single test game with behavior tree agents and return the game ID.

Usage:
    python -m scripts.run_single_test
"""
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.create_game import create_game_script
from scripts.run_agents import run_agents_script
from api.database import init_db, close_db_connection

def main():
    # Initialize database
    init_db()
    
    # Create a game
    print("Creating game...")
    game_id = create_game_script(num_players=4, rng_seed=None)
    print(f"Game created: {game_id}")
    
    # Run behavior tree agents
    print(f"\nRunning behavior tree agents on game {game_id}...")
    exit_code = run_agents_script(game_id, max_turns=1000, fast_mode=True, agent_type="behavior_tree")
    
    if exit_code == 0:
        print(f"\n✓ Game completed successfully!")
    else:
        print(f"\n⚠ Game ended with exit code {exit_code}")
    
    print(f"\nGame ID for replay: {game_id}")
    
    close_db_connection()
    return game_id

if __name__ == "__main__":
    game_id = main()
    sys.exit(0)

