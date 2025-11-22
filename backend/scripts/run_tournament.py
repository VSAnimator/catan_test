#!/usr/bin/env python3
"""
Run a tournament between behavior tree agent variants.

Usage:
    python -m scripts.run_tournament [--num-games N] [--workers W]
"""
import sys
import argparse
import multiprocessing
import time
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.create_game import create_game_script
from scripts.run_agents import run_agents_script
from agents.variants import (
    BalancedAgent,
    AggressiveBuilderAgent,
    DevCardFocusedAgent,
    ExpansionAgent,
    DefensiveAgent,
)


# Agent registry
AGENT_REGISTRY = {
    'balanced': BalancedAgent,
    'aggressive_builder': AggressiveBuilderAgent,
    'dev_card_focused': DevCardFocusedAgent,
    'expansion': ExpansionAgent,
    'defensive': DefensiveAgent,
}

AGENT_NAMES = list(AGENT_REGISTRY.keys())


def run_single_tournament_game(args):
    """Run a single tournament game (for multiprocessing)."""
    game_num, total_games, max_turns, fast_mode, agent_types = args
    import sys
    import traceback
    import time
    
    start_time = time.time()
    
    try:
        # Initialize database connection in this process
        from api.database import init_db
        init_db()
        
        # Create a game
        game_id = create_game_script(num_players=4, rng_seed=None)
        
        # Create agent mapping: assign agent types to players
        from api.database import get_latest_state
        from engine import deserialize_game_state
        state_json = get_latest_state(game_id)
        if state_json:
            current_state = deserialize_game_state(state_json)
            agent_mapping = {}
            for i, player in enumerate(current_state.players):
                agent_mapping[player.id] = agent_types[i]
        else:
            # Fallback
            agent_mapping = {
                f"player_{i}": agent_types[i] for i in range(4)
            }
        
        # Run agents
        exit_code = run_agents_script(game_id, max_turns=max_turns, fast_mode=fast_mode, agent_mapping=agent_mapping)
        
        # Check final state
        final_state_json = get_latest_state(game_id)
        if final_state_json:
            final_state = deserialize_game_state(final_state_json)
            game_completed = (final_state.phase == "finished" or 
                            any(p.victory_points >= 10 for p in final_state.players))
            hit_max_turns = (final_state.turn_number >= max_turns and not game_completed)
            
            # Determine winner and agent type
            winner = None
            winner_agent_type = None
            if game_completed:
                for player in final_state.players:
                    if player.victory_points >= 10:
                        winner = player
                        player_index = int(player.id.split("_")[1])
                        winner_agent_type = agent_types[player_index]
                        break
        else:
            game_completed = False
            hit_max_turns = False
            winner = None
            winner_agent_type = None
        
        # Clean up
        from api.database import close_db_connection
        close_db_connection()
        
        if game_completed:
            return ("PASSED", game_id, winner_agent_type, game_num, agent_types)
        elif hit_max_turns:
            return ("MAX_TURNS", game_id, None, game_num, agent_types)
        else:
            return ("FAILED", game_id, None, game_num, agent_types)
            
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        try:
            from api.database import close_db_connection
            close_db_connection()
        except:
            pass
        return ("ERROR", None, None, game_num, agent_types)


def run_tournament(
    num_games_per_matchup: int = 1000,
    max_turns: int = 1000,
    fast_mode: bool = True,
    workers: int = None
):
    """Run a tournament between all agent variants."""
    import os
    
    if workers is None:
        workers = min(os.cpu_count() or 4, 8)
    
    print(f"=== Tournament: {len(AGENT_NAMES)} agent types ===")
    print(f"Agent types: {', '.join(AGENT_NAMES)}")
    print(f"Games per matchup: {num_games_per_matchup}")
    print(f"Fast mode: {fast_mode}, Workers: {workers}\n")
    
    # Generate all matchups (all 4-player combinations with agent types)
    # For a full tournament, we want all combinations of 4 agents (with replacement)
    # This means each game has 4 agents, and we track which agent type wins
    
    # Statistics: matchup -> {agent_type -> wins}
    stats = defaultdict(lambda: defaultdict(int))
    total_games = defaultdict(int)
    max_turns_games = defaultdict(int)
    
    # Create tournament directory
    tournament_dir = Path(__file__).parent.parent.parent / "tournament_results"
    tournament_dir.mkdir(exist_ok=True)
    
    print("Starting tournament...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run games with random agent assignments
    # Each game randomly assigns agent types to players
    import random
    game_args = []
    for game_num in range(1, num_games_per_matchup + 1):
        # Randomly assign 4 agent types to 4 players
        agent_types = random.choices(AGENT_NAMES, k=4)
        game_args.append((game_num, num_games_per_matchup, max_turns, fast_mode, agent_types))
    
    print(f"Prepared {len(game_args)} games for execution\n")
    
    # Run games in parallel
    with multiprocessing.Pool(processes=workers) as pool:
        print(f"✓ Pool created with {workers} workers")
        print(f"✓ Submitting {num_games_per_matchup} games...\n")
        
        async_results = []
        for game_arg in game_args:
            async_result = pool.apply_async(run_single_tournament_game, (game_arg,))
            async_results.append(async_result)
        
        # Collect results
        completed = 0
        for i, async_result in enumerate(async_results):
            try:
                result = async_result.get(timeout=300)
                completed += 1
                status, game_id, winner_agent_type, game_num, agent_types = result
                
                # Track statistics
                matchup_key = tuple(sorted(agent_types))
                total_games[matchup_key] += 1
                
                if status == "PASSED" and winner_agent_type:
                    stats[matchup_key][winner_agent_type] += 1
                elif status == "MAX_TURNS":
                    max_turns_games[matchup_key] += 1
                
                if completed % 100 == 0 or completed == num_games_per_matchup:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = num_games_per_matchup - completed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"Progress: {completed}/{num_games_per_matchup} ({completed*100//num_games_per_matchup}%) | "
                          f"Rate: {rate:.2f} games/sec | ETA: {eta:.1f}s", flush=True)
                    
            except Exception as e:
                completed += 1
                print(f"✗ Game {i+1} ERROR: {str(e)}", flush=True)
    
    # Calculate overall statistics
    overall_stats = defaultdict(int)
    overall_total = 0
    overall_max_turns = 0
    
    for matchup_key, agent_wins in stats.items():
        for agent_type, wins in agent_wins.items():
            overall_stats[agent_type] += wins
        overall_total += total_games[matchup_key]
        overall_max_turns += max_turns_games[matchup_key]
    
    # Print results
    print("\n" + "=" * 80)
    print("=== Tournament Results ===")
    print(f"Total games: {overall_total}")
    print(f"Max turns games: {overall_max_turns}")
    print(f"Completed games: {overall_total - overall_max_turns}")
    print()
    
    print("=== Overall Win Statistics ===")
    sorted_agents = sorted(overall_stats.items(), key=lambda x: -x[1])
    for agent_type, wins in sorted_agents:
        win_rate = (wins / (overall_total - overall_max_turns) * 100) if (overall_total - overall_max_turns) > 0 else 0
        print(f"{agent_type:20s}: {wins:4d} wins ({win_rate:5.1f}%)")
    
    # Save statistics to file
    results_file = tournament_dir / f"tournament_results_{int(time.time())}.json"
    results_data = {
        'timestamp': time.time(),
        'num_games_per_matchup': num_games_per_matchup,
        'max_turns': max_turns,
        'agent_types': AGENT_NAMES,
        'overall_stats': dict(overall_stats),
        'overall_total': overall_total,
        'overall_max_turns': overall_max_turns,
        'matchup_stats': {
            str(k): {
                'wins': dict(v),
                'total': total_games[k],
                'max_turns': max_turns_games[k]
            }
            for k, v in stats.items()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a tournament between behavior tree agent variants."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to run (default: 1000)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1000,
        help="Maximum number of turns per game (default: 1000)"
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Disable fast mode"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(CPU cores, 8))"
    )
    
    args = parser.parse_args()
    sys.exit(run_tournament(
        num_games_per_matchup=args.num_games,
        max_turns=args.max_turns,
        fast_mode=not args.no_fast,
        workers=args.workers
    ))

