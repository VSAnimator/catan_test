#!/usr/bin/env python3
"""
Run agents automatically on a game until completion, error, or max turns.

Usage:
    python -m scripts.run_agents <game_id> [--max-turns N]
"""
import sys
import json
import random
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import get_game, get_latest_state, save_game_state, add_step, get_step_count
from engine import deserialize_game_state, serialize_game_state
from engine.serialization import (
    legal_actions,
    state_to_text,
    legal_actions_to_text,
    serialize_action,
    serialize_action_payload,
)
from agents import RandomAgent
from agents.agent_runner import AgentRunner


def run_agents_script(game_id: str, max_turns: int = 1000):
    """Run agents automatically on a game."""
    # Get game info
    game_row = get_game(game_id)
    if not game_row:
        print(f"Error: Game {game_id} not found in database.")
        return 1
    
    print(f"=== Running Agents on Game: {game_id} ===")
    print(f"Created at: {game_row['created_at']}")
    if game_row['rng_seed'] is not None:
        print(f"RNG Seed: {game_row['rng_seed']}")
        random.seed(game_row['rng_seed'])
    if game_row['metadata']:
        metadata = json.loads(game_row['metadata'])
        print(f"Players: {', '.join(metadata.get('player_names', []))}")
    print(f"Max turns: {max_turns}")
    print()
    
    # Get current state from database
    state_json = get_latest_state(game_id)
    if state_json is None:
        print(f"Error: Game state not found for game {game_id}.")
        return 1
    
    # Deserialize current state
    current_state = deserialize_game_state(state_json)
    
    print(f"Initial state:")
    print(f"  Phase: {current_state.phase}")
    print(f"  Turn: {current_state.turn_number}")
    if current_state.phase == "playing":
        current_player = current_state.players[current_state.current_player_index]
        print(f"  Current player: {current_player.name} ({current_player.id})")
    print("  Player victory points:")
    for player in current_state.players:
        print(f"    {player.name}: {player.victory_points} VP")
    print()
    
    # Create agents for all players
    agents = {}
    for player in current_state.players:
        agents[player.id] = RandomAgent(player.id)
        print(f"  Created RandomAgent for {player.name} ({player.id})")
    print()
    
    # Callback to save state after each action
    def save_state_callback(game_id: str, state_before, state_after, action: dict, player_id: str):
        # Serialize states
        state_before_json = serialize_game_state(state_before)
        state_after_json = serialize_game_state(state_after)
        
        # Update current state in games table
        save_game_state(game_id, state_after_json)
        
        # Get step index
        step_idx = get_step_count(game_id)
        
        # Get legal actions and text representations (use state_before for context)
        legal_actions_list = legal_actions(state_before, player_id)
        legal_actions_text = legal_actions_to_text(legal_actions_list)
        state_text = state_to_text(state_before, player_id)
        
        # Format chosen action text
        action_type_str = action.get("type", "")
        chosen_action_text = action_type_str.replace("_", " ").title()
        if "payload" in action and action["payload"]:
            payload_dict = action["payload"]
            if isinstance(payload_dict, dict):
                if "intersection_id" in payload_dict:
                    chosen_action_text += f" at intersection {payload_dict['intersection_id']}"
                elif "road_edge_id" in payload_dict:
                    chosen_action_text += f" on road edge {payload_dict['road_edge_id']}"
                elif "card_type" in payload_dict:
                    chosen_action_text += f" ({payload_dict['card_type']})"
        
        # Save step to database
        add_step(
            game_id=game_id,
            step_idx=step_idx,
            player_id=player_id,
            state_before_json=state_before_json,
            state_after_json=state_after_json,
            action_json=action,
            dice_roll=state_after.dice_roll,
            state_text=state_text,
            legal_actions_text=legal_actions_text,
            chosen_action_text=chosen_action_text,
        )
    
    # Create agent runner
    runner = AgentRunner(current_state, agents, max_turns=max_turns)
    
    print("Running agents...")
    print("=" * 80)
    print()
    
    # Run the game automatically
    final_state, completed, error = runner.run_automatic(save_state_callback=save_state_callback)
    
    # Print results
    print("=" * 80)
    print()
    print("=== Results ===")
    print(f"Game ID: {game_id}")
    print(f"Turns played: {runner.turn_count}")
    print(f"Completed: {completed}")
    if error:
        print(f"Error: {error}")
    print()
    
    print("Final state:")
    print(f"  Phase: {final_state.phase}")
    print(f"  Turn: {final_state.turn_number}")
    if final_state.phase == "playing":
        current_player = final_state.players[final_state.current_player_index]
        print(f"  Current player: {current_player.name} ({current_player.id})")
    print("  Player victory points:")
    for player in final_state.players:
        print(f"    {player.name}: {player.victory_points} VP")
    print()
    
    # Check for winner
    winner = None
    for player in final_state.players:
        if player.victory_points >= 10:
            winner = player
            break
    
    if winner:
        print(f"🏆 Winner: {winner.name} ({winner.id}) with {winner.victory_points} victory points!")
    elif completed:
        print("Game completed normally (no winner reached 10 VP yet).")
    else:
        print("Game stopped early (error or max turns reached).")
    print()
    
    print(f"To replay this game, use:")
    print(f"  python -m scripts.replay_game {game_id}")
    print()
    
    return 0 if completed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run agents automatically on a game until completion, error, or max turns."
    )
    parser.add_argument("game_id", help="Game ID to run agents on")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1000,
        help="Maximum number of turns before stopping (default: 1000)"
    )
    
    args = parser.parse_args()
    sys.exit(run_agents_script(args.game_id, args.max_turns))

