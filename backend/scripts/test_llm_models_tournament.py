#!/usr/bin/env python3
"""
Run tournament: 1 gpt-5.2 agent vs 3 gpt-4o agents
Runs multiple games in parallel with real-time progress updates.
"""
import sys
import os
import multiprocessing
import time
import threading
from pathlib import Path
from collections import defaultdict

# Load environment variables
if 'OPENAI_API_KEY' not in os.environ:
    import subprocess
    try:
        result = subprocess.run(['bash', '-c', 'source ~/.zshrc 2>/dev/null && echo $OPENAI_API_KEY'], 
                              capture_output=True, text=True, shell=False)
        if result.stdout.strip():
            os.environ['OPENAI_API_KEY'] = result.stdout.strip()
    except:
        pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
import random
from typing import Optional
from engine import GameState, Player
from agents import LLMAgent
from agents.agent_runner import AgentRunner
from api.database import init_db, create_game, get_latest_state, add_step, get_step_count
from engine.serialization import deserialize_game_state, legal_actions, serialize_game_state

# Global lock for printing
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs, flush=True)

def get_player_color(index: int) -> str:
    """Get color for player by index."""
    colors = ["#FF0000", "#00AA00", "#2196F3", "#FF8C00"]
    return colors[index % len(colors)]

def generate_random_name() -> str:
    """Generate a random player name."""
    adjectives = ["Brave", "Swift", "Clever", "Bold", "Wise", "Fierce", "Noble", "Cunning"]
    nouns = ["Wolf", "Eagle", "Lion", "Bear", "Fox", "Hawk", "Tiger", "Panther"]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"

def run_single_game(args):
    """Run a single game: 1 gpt-5.2 vs 3 gpt-4o agents."""
    game_num, total_games, max_turns = args
    
    try:
        # Initialize database for this process
        init_db()
        
        game_id = str(uuid.uuid4())
        
        # Create 4 players
        player_names = [generate_random_name() for _ in range(4)]
        players = [
            Player(id=f"player_{i}", name=player_names[i], color=get_player_color(i))
            for i in range(4)
        ]
        
        # Create initial game state
        initial_state = GameState(
            game_id=game_id,
            players=players,
            current_player_index=0,
            phase="setup"
        )
        
        # Initialize the board
        initial_state = initial_state._create_initial_board(initial_state)
        
        # Initialize robber on desert tile
        desert_tile = next((t for t in initial_state.tiles if t.resource_type is None), None)
        if desert_tile:
            initial_state.robber_tile_id = desert_tile.id
        
        # Serialize and save to database
        serialized_state = serialize_game_state(initial_state)
        metadata = {
            "player_names": player_names,
            "num_players": 4,
            "experiment": "gpt-5.2-vs-gpt-4o",
            "game_num": game_num
        }
        create_game(
            game_id,
            rng_seed=None,
            metadata=metadata,
            initial_state_json=serialized_state,
        )
        
        # Get initial state
        state_data = get_latest_state(game_id)
        state = deserialize_game_state(state_data)
        
        # Create agents: player_0 uses gpt-5.2, others use gpt-4o
        player_ids = [p.id for p in state.players]
        agents = {
            player_ids[0]: LLMAgent(player_ids[0], model="gpt-5.2", enable_retrieval=False),
            player_ids[1]: LLMAgent(player_ids[1], model="gpt-4o", enable_retrieval=False),
            player_ids[2]: LLMAgent(player_ids[2], model="gpt-4o", enable_retrieval=False),
            player_ids[3]: LLMAgent(player_ids[3], model="gpt-4o", enable_retrieval=False),
        }
        
        safe_print(f"[Game {game_num}/{total_games}] Started: {game_id[:8]}... | gpt-5.2: {state.players[0].name} vs 3x gpt-4o")
        
        # Track turns
        turn_count = 0
        
        # Callback to save state and print progress
        def save_state_callback(game_id: str, state_before, state_after, action: dict, player_id: str, raw_llm_response: Optional[str] = None):
            nonlocal turn_count
            turn_count += 1
            
            # Find player name and VP
            player = next((p for p in state_after.players if p.id == player_id), None)
            player_name = player.name if player else player_id
            player_vp = player.victory_points if player else 0
            
            # Determine which model this player uses
            model_used = "gpt-5.2" if player_id == player_ids[0] else "gpt-4o"
            
            # Get current phase and turn info
            phase = state_after.phase
            
            safe_print(f"[Game {game_num}] Turn {turn_count}: {player_name} ({model_used}) | Phase: {phase} | Action: {action.get('type', 'unknown')} | {player_name} VP: {player_vp}")
            
            # Save state
            step_idx = get_step_count(game_id)
            state_before_json = serialize_game_state(state_before)
            state_after_json = serialize_game_state(state_after)
            add_step(
                game_id=game_id,
                step_idx=step_idx,
                player_id=player_id,
                state_before_json=state_before_json,
                state_after_json=state_after_json,
                action_json=action,
                dice_roll=state_after.dice_roll,
                raw_llm_response=raw_llm_response,
                batch_write=True  # Use batched writes for performance
            )
        
        # Create agent runner
        runner = AgentRunner(state, agents, max_turns=max_turns)
        
        # Run the game
        start_time = time.time()
        final_state, completed, error = runner.run_automatic(save_state_callback=save_state_callback)
        elapsed = time.time() - start_time
        
        if error:
            safe_print(f"[Game {game_num}] ERROR during gameplay: {error}")
            return {
                "game_id": game_id,
                "game_num": game_num,
                "winner": None,
                "error": error
            }
        
        # Determine winner
        winner = None
        max_vp = 0
        for player in final_state.players:
            if player.victory_points >= 10 and player.victory_points > max_vp:
                max_vp = player.victory_points
                winner = player
        
        # Determine which model won
        winner_model = "gpt-5.2" if winner and winner.id == player_ids[0] else "gpt-4o"
        
        safe_print(f"[Game {game_num}] COMPLETED in {elapsed:.1f}s | Winner: {winner.name if winner else 'None'} ({winner_model}) | Turns: {turn_count}")
        
        return {
            "game_id": game_id,
            "game_num": game_num,
            "winner": winner.name if winner else None,
            "winner_id": winner.id if winner else None,
            "winner_model": winner_model,
            "turns": turn_count,
            "elapsed": elapsed,
            "final_vps": {p.name: p.victory_points for p in final_state.players}
        }
        
    except Exception as e:
        import traceback
        safe_print(f"[Game {game_num}] ERROR: {str(e)}")
        safe_print(traceback.format_exc())
        return {
            "game_id": None,
            "game_num": game_num,
            "winner": None,
            "error": str(e)
        }

def main():
    """Run tournament: 10 games in parallel."""
    num_games = 10
    max_turns = 1000
    num_workers = min(10, multiprocessing.cpu_count() or 4)
    
    safe_print("=" * 80)
    safe_print(f"LLM MODEL TOURNAMENT: gpt-5.2 vs gpt-4o")
    safe_print("=" * 80)
    safe_print(f"Running {num_games} games in parallel ({num_workers} workers)")
    safe_print(f"Each game: 1 gpt-5.2 agent vs 3 gpt-4o agents")
    safe_print(f"Max turns per game: {max_turns}")
    safe_print("=" * 80)
    safe_print()
    
    # Initialize database in main process
    init_db()
    
    # Prepare arguments
    game_args = [(i+1, num_games, max_turns) for i in range(num_games)]
    
    start_time = time.time()
    
    # Run games in parallel
    with multiprocessing.Pool(processes=num_workers, initializer=init_db) as pool:
        results = pool.map(run_single_game, game_args)
    
    elapsed = time.time() - start_time
    
    # Analyze results
    safe_print()
    safe_print("=" * 80)
    safe_print("TOURNAMENT RESULTS")
    safe_print("=" * 80)
    
    gpt52_wins = 0
    gpt4o_wins = 0
    errors = 0
    total_turns = 0
    
    for result in results:
        if result.get("error"):
            errors += 1
            continue
        
        total_turns += result.get("turns", 0)
        
        if result.get("winner_model") == "gpt-5.2":
            gpt52_wins += 1
        elif result.get("winner_model") == "gpt-4o":
            gpt4o_wins += 1
    
    safe_print(f"Total games: {num_games}")
    safe_print(f"gpt-5.2 wins: {gpt52_wins} ({gpt52_wins/num_games*100:.1f}%)")
    safe_print(f"gpt-4o wins: {gpt4o_wins} ({gpt4o_wins/num_games*100:.1f}%)")
    safe_print(f"Errors: {errors}")
    safe_print(f"Total turns: {total_turns}")
    safe_print(f"Average turns per game: {total_turns/num_games:.1f}")
    safe_print(f"Total time: {elapsed:.1f}s")
    safe_print(f"Average time per game: {elapsed/num_games:.1f}s")
    safe_print()
    safe_print("=" * 80)
    
    # Detailed results
    safe_print("DETAILED RESULTS:")
    safe_print("-" * 80)
    for result in results:
        if result.get("error"):
            safe_print(f"Game {result['game_num']}: ERROR - {result.get('error')}")
        else:
            safe_print(f"Game {result['game_num']}: {result.get('winner', 'No winner')} ({result.get('winner_model')}) | {result.get('turns')} turns")
    safe_print("=" * 80)

if __name__ == "__main__":
    main()

