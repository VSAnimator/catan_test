#!/usr/bin/env python3
"""
Create a new game and return the game ID.

Usage:
    python -m scripts.create_game [--num-players N] [--rng-seed SEED]
"""
import sys
import json
import random
import uuid
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import init_db, create_game as create_game_in_db
from engine import GameState, Player
from engine.serialization import serialize_game_state

# Random name generators (same as in routes.py)
FIRST_NAMES = [
    "Alex", "Blake", "Casey", "Drew", "Emery", "Finley", "Gray", "Harper",
    "Jordan", "Kai", "Logan", "Morgan", "Parker", "Quinn", "Riley", "Sage",
    "Taylor", "Avery", "Cameron", "Dakota", "Ellis", "Hayden", "Jamie", "Kendall",
    "Lane", "Marley", "Noah", "Ocean", "Peyton", "Reese", "River", "Skylar"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris", "Clark",
    "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright", "Scott"
]

PLAYER_COLORS = ["#FF0000", "#00AA00", "#2196F3", "#F5F5F5"]  # Red, Green, Blue, White

def generate_random_name() -> str:
    """Generate a random player name."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"

def get_player_color(index: int) -> str:
    """Get a color for a player based on their index."""
    return PLAYER_COLORS[index % len(PLAYER_COLORS)]


def create_game_script(num_players: int = 4, rng_seed: int = None):
    """Create a new game and return the game ID."""
    # Initialize database
    init_db()
    
    # Set RNG seed if provided
    if rng_seed is not None:
        random.seed(rng_seed)
    
    # Generate game ID
    game_id = str(uuid.uuid4())
    
    # Generate unique random names for players
    final_names = []
    used_names = set()
    for _ in range(num_players):
        name = generate_random_name()
        while name in used_names:
            name = generate_random_name()
        used_names.add(name)
        final_names.append(name)
    
    # Create players with colors
    players = [
        Player(id=f"player_{i}", name=final_names[i], color=get_player_color(i))
        for i in range(len(final_names))
    ]
    
    # Create initial game state
    initial_state = GameState(
        game_id=game_id,
        players=players,
        current_player_index=0,
        phase="setup"
    )
    
    # Initialize the board (tiles, intersections, road edges)
    initial_state = initial_state._create_initial_board(initial_state)
    
    # Initialize robber on desert tile
    desert_tile = next((t for t in initial_state.tiles if t.resource_type is None), None)
    if desert_tile:
        initial_state.robber_tile_id = desert_tile.id
    
    # Serialize initial state
    serialized_state = serialize_game_state(initial_state)
    
    # Save game to database with initial state
    metadata = {
        "player_names": final_names,
        "num_players": len(final_names),
    }
    create_game_in_db(
        game_id,
        rng_seed=rng_seed,
        metadata=metadata,
        initial_state_json=serialized_state,
    )
    
    print(f"Created game: {game_id}")
    print(f"Players: {', '.join(final_names)}")
    if rng_seed is not None:
        print(f"RNG Seed: {rng_seed}")
    
    return game_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a new game and return the game ID."
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        choices=[2, 3, 4],
        help="Number of players (default: 4)"
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility"
    )
    
    args = parser.parse_args()
    game_id = create_game_script(args.num_players, args.rng_seed)
    print(f"\nGame ID: {game_id}")
    sys.exit(0)

