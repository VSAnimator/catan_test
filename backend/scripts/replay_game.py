#!/usr/bin/env python3
"""
Replay a game from the database.

Usage:
    python -m scripts.replay_game <game_id>
"""
import sys
import json
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import get_game, get_steps
from engine import deserialize_game_state


def replay_game(game_id: str):
    """Replay a game from the database and print a text log."""
    # Get game info
    game_row = get_game(game_id)
    if not game_row:
        print(f"Error: Game {game_id} not found in database.")
        return 1
    
    print(f"=== Replaying Game: {game_id} ===")
    print(f"Created at: {game_row['created_at']}")
    if game_row['rng_seed'] is not None:
        print(f"RNG Seed: {game_row['rng_seed']}")
    if game_row['metadata']:
        metadata = json.loads(game_row['metadata'])
        print(f"Players: {', '.join(metadata.get('player_names', []))}")
    print()
    
    # Get all steps
    steps = get_steps(game_id)
    
    if not steps:
        print("No steps found for this game.")
        return 0
    
    print(f"Total steps: {len(steps)}")
    print()
    print("=" * 80)
    print()
    
    # Replay each step
    for i, step in enumerate(steps, 1):
        print(f"--- Step {step['step_idx']} ---")
        print(f"Player: {step['player_id']}")
        print(f"Timestamp: {step['timestamp']}")
        
        if step['dice_roll'] is not None:
            print(f"Dice Roll: {step['dice_roll']}")
        
        # Print state text if available
        if step['state_text']:
            print("\nState before action:")
            print(step['state_text'])
        
        # Print legal actions if available
        if step['legal_actions_text']:
            print("\nLegal actions:")
            print(step['legal_actions_text'])
        
        # Print chosen action
        if step['chosen_action_text']:
            print(f"\nChosen action: {step['chosen_action_text']}")
        else:
            # Fallback to action JSON
            action = json.loads(step['action_json'])
            action_type = action.get('type', 'unknown')
            print(f"\nChosen action: {action_type}")
        
        # Optionally show state after
        if i < len(steps) or True:  # Always show final state
            state_after = json.loads(step['state_after_json'])
            state_obj = deserialize_game_state(state_after)
            
            # Print summary of state after
            print("\nState after action:")
            print(f"  Phase: {state_obj.phase}")
            print(f"  Turn: {state_obj.turn_number}")
            if state_obj.phase == "playing":
                current_player = state_obj.players[state_obj.current_player_index]
                print(f"  Current player: {current_player.name} ({current_player.id})")
            
            print("  Player victory points:")
            for player in state_obj.players:
                print(f"    {player.name}: {player.victory_points} VP")
        
        print()
        print("=" * 80)
        print()
    
    print("Replay complete.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.replay_game <game_id>")
        sys.exit(1)
    
    game_id = sys.argv[1]
    sys.exit(replay_game(game_id))

