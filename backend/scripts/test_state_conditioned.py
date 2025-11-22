#!/usr/bin/env python3
"""
Test state-conditioned agent against balanced agent.

Usage:
    python -m scripts.test_state_conditioned [--num-games N] [--workers W]
"""
import sys
import argparse
import multiprocessing
import time
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.create_game import create_game_script
from scripts.run_agents import run_agents_script


def run_single_comparison_game(args):
    """Run a single comparison game (for multiprocessing)."""
    game_num, total_games, max_turns, fast_mode = args
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
        
        # Create agent mapping: 2 balanced, 2 state_conditioned
        from api.database import get_latest_state
        from engine import deserialize_game_state
        state_json = get_latest_state(game_id)
        if state_json:
            current_state = deserialize_game_state(state_json)
            agent_mapping = {}
            for i, player in enumerate(current_state.players):
                if i < 2:
                    agent_mapping[player.id] = "balanced"
                else:
                    agent_mapping[player.id] = "state_conditioned"
        else:
            agent_mapping = {
                "player_0": "balanced",
                "player_1": "balanced",
                "player_2": "state_conditioned",
                "player_3": "state_conditioned",
            }
        
        # Run agents (ignore exit code, check final state instead)
        try:
            exit_code = run_agents_script(game_id, max_turns=max_turns, fast_mode=fast_mode, agent_mapping=agent_mapping)
            if game_num == 1:
                print(f"[DEBUG Game 1] run_agents_script returned exit_code={exit_code}", flush=True)
        except Exception as e:
            if game_num == 1:
                print(f"[DEBUG Game 1] Error in run_agents_script: {e}", flush=True)
            raise
        
        # Check final state (same logic as test_agents_batch.py)
        final_state_json = get_latest_state(game_id)
        if final_state_json:
            final_state = deserialize_game_state(final_state_json)
            # Check if game completed normally (phase == "finished" or someone has 10+ VPs)
            game_completed = (final_state.phase == "finished" or 
                            any(p.victory_points >= 10 for p in final_state.players))
            # Check if we hit max turns (turn_number >= max_turns and not completed)
            # Note: turn_number is 0-indexed, so we check >= max_turns
            hit_max_turns = (final_state.turn_number >= max_turns and not game_completed)
            
            # Determine winner and agent type
            winner = None
            winner_agent_type = None
            if game_completed:
                for player in final_state.players:
                    if player.victory_points >= 10:
                        winner = player
                        player_index = int(player.id.split("_")[1])
                        if player_index < 2:
                            winner_agent_type = "balanced"
                        else:
                            winner_agent_type = "state_conditioned"
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
            return ("PASSED", game_id, winner_agent_type, game_num)
        elif hit_max_turns:
            return ("MAX_TURNS", game_id, None, game_num)
        else:
            return ("FAILED", game_id, None, game_num)
            
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        try:
            from api.database import close_db_connection
            close_db_connection()
        except:
            pass
        return ("ERROR", None, None, game_num)


def test_state_conditioned_vs_balanced(
    num_games: int = 1000,
    max_turns: int = 1000,
    fast_mode: bool = True,
    workers: int = None
):
    """Test state-conditioned agent against balanced agent."""
    import os
    
    if workers is None:
        workers = min(os.cpu_count() or 4, 8)
    
    print(f"=== Testing State-Conditioned vs Balanced Agent ===")
    print(f"Games: {num_games}")
    print(f"Fast mode: {fast_mode}, Workers: {workers}\n")
    
    results = []
    
    # Prepare arguments for each game
    game_args = [(i+1, num_games, max_turns, fast_mode) for i in range(num_games)]
    
    print("Starting games...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run games in parallel
    with multiprocessing.Pool(processes=workers) as pool:
        print(f"✓ Pool created with {workers} workers")
        print(f"✓ Submitting {num_games} games...\n")
        
        async_results = []
        for game_arg in game_args:
            async_result = pool.apply_async(run_single_comparison_game, (game_arg,))
            async_results.append(async_result)
        
        # Collect results
        completed = 0
        status_counts = defaultdict(int)
        for i, async_result in enumerate(async_results):
            try:
                result = async_result.get(timeout=300)
                completed += 1
                status, game_id, winner_agent_type, game_num = result
                status_counts[status] += 1
                
                results.append((status, game_id, winner_agent_type, game_num))
                
                if completed % 100 == 0 or completed == num_games:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = num_games - completed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"Progress: {completed}/{num_games} ({completed*100//num_games}%) | "
                          f"Rate: {rate:.2f} games/sec | ETA: {eta:.1f}s | "
                          f"Status: {dict(status_counts)}", flush=True)
                    
            except Exception as e:
                completed += 1
                status_counts["ERROR"] += 1
                print(f"✗ Game {i+1} ERROR: {str(e)}", flush=True)
                results.append(("ERROR", None, None, i+1))
    
    # Calculate statistics
    from collections import Counter
    status_breakdown = Counter(r[0] for r in results)
    
    passed = [r for r in results if r[0] == "PASSED"]
    max_turns = [r for r in results if r[0] == "MAX_TURNS"]
    failed = [r for r in results if r[0] == "FAILED"]
    errors = [r for r in results if r[0] == "ERROR"]
    
    balanced_wins = sum(1 for r in passed if r[2] == "balanced")
    state_conditioned_wins = sum(1 for r in passed if r[2] == "state_conditioned")
    
    # Print results
    print("\n" + "=" * 80)
    print("=== Test Results ===")
    print(f"Total games: {num_games}")
    print(f"Status breakdown: {dict(status_breakdown)}")
    print(f"Completed: {len(passed)}")
    print(f"Max turns: {len(max_turns)}")
    print(f"Failed: {len(failed)}")
    print(f"Errors: {len(errors)}")
    print()
    
    print("=== Win Statistics ===")
    print(f"Balanced Agent Wins: {balanced_wins} ({balanced_wins*100//len(passed) if passed else 0}%)")
    print(f"State-Conditioned Agent Wins: {state_conditioned_wins} ({state_conditioned_wins*100//len(passed) if passed else 0}%)")
    
    if passed:
        print(f"\nTotal completed games: {len(passed)}")
        print(f"Balanced win rate: {balanced_wins/len(passed)*100:.1f}%")
        print(f"State-Conditioned win rate: {state_conditioned_wins/len(passed)*100:.1f}%")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test state-conditioned agent against balanced agent."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to test (default: 1000)"
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
    sys.exit(test_state_conditioned_vs_balanced(
        num_games=args.num_games,
        max_turns=args.max_turns,
        fast_mode=not args.no_fast,
        workers=args.workers
    ))

