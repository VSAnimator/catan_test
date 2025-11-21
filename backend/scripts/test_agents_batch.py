#!/usr/bin/env python3
"""
Run multiple agent games and report results.

Usage:
    python -m scripts.test_agents_batch [--num-games N] [--max-turns N] [--workers W]
"""
import sys
import argparse
import multiprocessing
import time
from pathlib import Path
from functools import partial

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.create_game import create_game_script
from scripts.run_agents import run_agents_script


def run_single_game(args):
    """Run a single game (for multiprocessing)."""
    game_num, total_games, max_turns, fast_mode, agent_type = args
    import sys
    import traceback
    import time
    
    start_time = time.time()
    
    try:
        print(f"[Game {game_num}] Starting at {time.time():.2f}...", flush=True)
        
        # Initialize database connection in this process
        print(f"[Game {game_num}] Initializing database...", flush=True)
        from api.database import init_db
        init_db()
        print(f"[Game {game_num}] Database initialized ({time.time() - start_time:.2f}s)", flush=True)
        
        # Create a game
        print(f"[Game {game_num}] Creating game...", flush=True)
        game_id = create_game_script(num_players=4, rng_seed=None)
        print(f"[Game {game_num}] Game created: {game_id[:8]}... ({time.time() - start_time:.2f}s)", flush=True)
        
        # Run agents in fast mode for headless testing
        print(f"[Game {game_num}] Running agents (fast_mode={fast_mode}, agent_type={agent_type})...", flush=True)
        agent_start = time.time()
        exit_code = run_agents_script(game_id, max_turns=max_turns, fast_mode=fast_mode, agent_type=agent_type)
        agent_time = time.time() - agent_start
        
        # Check the actual completion status by reading the final state
        from api.database import get_latest_state
        from engine import deserialize_game_state
        final_state_json = get_latest_state(game_id)
        if final_state_json:
            final_state = deserialize_game_state(final_state_json)
            # Check if game completed normally (phase == "finished" or someone has 10+ VPs)
            game_completed = (final_state.phase == "finished" or 
                            any(p.victory_points >= 10 for p in final_state.players))
            # Check if we hit max turns (turn_count reached max_turns)
            hit_max_turns = (final_state.turn_number >= max_turns and not game_completed)
        else:
            game_completed = False
            hit_max_turns = False
        
        print(f"[Game {game_num}] Agents finished (exit_code={exit_code}, completed={game_completed}, max_turns={hit_max_turns}, took {agent_time:.2f}s)", flush=True)
        
        # Clean up database connection
        print(f"[Game {game_num}] Cleaning up...", flush=True)
        from api.database import close_db_connection
        close_db_connection()
        print(f"[Game {game_num}] Cleanup done", flush=True)
        
        total_time = time.time() - start_time
        
        # Determine result based on completion status
        if game_completed:
            print(f"[Game {game_num}] ✓ PASSED - Game completed normally (total: {total_time:.2f}s)", flush=True)
            return ("PASSED", game_id, None, game_num)
        elif hit_max_turns:
            print(f"[Game {game_num}] ⚠ MAX_TURNS - Reached {max_turns} turn limit (total: {total_time:.2f}s)", flush=True)
            return ("MAX_TURNS", game_id, f"Reached maximum turn limit ({max_turns})", game_num)
        else:
            print(f"[Game {game_num}] ✗ FAILED - Non-zero exit code (total: {total_time:.2f}s)", flush=True)
            return ("FAILED", game_id, f"Non-zero exit code: {exit_code}", game_num)
            
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Game {game_num}] ✗ ERROR after {total_time:.2f}s: {error_msg}", flush=True)
        # Clean up on error
        try:
            from api.database import close_db_connection
            close_db_connection()
        except:
            pass
        return ("ERROR", None, str(e), game_num)


def test_agents_batch(num_games: int = 10, max_turns: int = 1000, fast_mode: bool = True, workers: int = None, agent_type: str = "random"):
    # Store max_turns for use in summary message
    max_turns_arg = max_turns
    """Run multiple agent games and report results.
    
    Args:
        num_games: Number of games to test
        max_turns: Maximum turns per game
        fast_mode: Use fast mode (batched writes, no text serialization) for better performance
        workers: Number of parallel workers (None = use CPU count)
    """
    import os
    
    # Determine number of workers
    if workers is None:
        workers = min(os.cpu_count() or 4, 8)  # Cap at 8 to avoid overwhelming system
    
    results = []
    
    print(f"=== Testing {num_games} games with agents (fast_mode={fast_mode}, workers={workers}, agent_type={agent_type}) ===\n")
    print(f"Starting multiprocessing pool with {workers} workers...", flush=True)
    
    # Prepare arguments for each game
    game_args = [(i+1, num_games, max_turns, fast_mode, agent_type) for i in range(num_games)]
    print(f"Prepared {len(game_args)} games for execution\n", flush=True)
    
    # Run games in parallel using multiprocessing
    print("=" * 80, flush=True)
    print("Starting parallel execution...", flush=True)
    print("=" * 80, flush=True)
    
    start_time = time.time()
    last_heartbeat = start_time
    
    with multiprocessing.Pool(processes=workers) as pool:
        print(f"✓ Pool created with {workers} workers", flush=True)
        print(f"✓ Submitting {num_games} games to pool...", flush=True)
        
        # Submit all tasks
        async_results = []
        for game_arg in game_args:
            async_result = pool.apply_async(run_single_game, (game_arg,))
            async_results.append(async_result)
        
        print(f"✓ All {num_games} games submitted, waiting for results...\n", flush=True)
        
        # Use a heartbeat thread to show we're alive
        import threading
        heartbeat_active = threading.Event()
        heartbeat_active.set()
        
        def heartbeat():
            while heartbeat_active.is_set():
                time.sleep(5)  # Heartbeat every 5 seconds
                if heartbeat_active.is_set():
                    elapsed = time.time() - start_time
                    print(f"💓 Heartbeat: Still running... ({elapsed:.1f}s elapsed)", flush=True)
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        
        # Collect results as they complete
        completed = 0
        for i, async_result in enumerate(async_results):
            try:
                # Wait for result with timeout to show we're alive
                result = async_result.get(timeout=300)  # 5 minute timeout per game
                completed += 1
                status, game_id, error, game_num = result
                
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                
                if status == "PASSED":
                    print(f"✓ Game {game_num}/{num_games} PASSED (completed: {completed}/{num_games}, rate: {rate:.2f} games/sec)", flush=True)
                elif status == "MAX_TURNS":
                    print(f"⚠ Game {game_num}/{num_games} MAX_TURNS (completed: {completed}/{num_games}, rate: {rate:.2f} games/sec)", flush=True)
                else:
                    print(f"✗ Game {game_num}/{num_games} {status} (completed: {completed}/{num_games}, rate: {rate:.2f} games/sec)", flush=True)
                
                results.append((status, game_id, error, game_num))
                
                # Show progress every 5 games or at completion
                if completed % 5 == 0 or completed == num_games:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = num_games - completed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"\n📊 Progress: {completed}/{num_games} games completed ({completed*100//num_games}%)", flush=True)
                    print(f"   Rate: {rate:.2f} games/sec | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s\n", flush=True)
                    
            except Exception as e:
                completed += 1
                print(f"✗ Game {i+1}/{num_games} ERROR getting result: {str(e)}", flush=True)
                results.append(("ERROR", None, str(e), i+1))
        
        # Stop heartbeat
        heartbeat_active.clear()
        heartbeat_thread.join(timeout=1)
        
        print(f"\n✓ All games completed. Finalizing...", flush=True)
    
    # Sort results by game number for consistent output
    results.sort(key=lambda x: x[3] if len(x) > 3 else 0)
    # Remove game_num from results for compatibility
    results = [(r[0], r[1], r[2]) for r in results]
    
    # Print summary
    print("\n" + "=" * 80)
    print("=== Test Summary ===")
    print(f"Total games: {num_games}")
    
    passed = [r for r in results if r[0] == "PASSED"]
    max_turns = [r for r in results if r[0] == "MAX_TURNS"]
    failed = [r for r in results if r[0] == "FAILED"]
    errors = [r for r in results if r[0] == "ERROR"]
    
    print(f"Passed: {len(passed)}")
    print(f"Max Turns: {len(max_turns)}")
    print(f"Failed: {len(failed)}")
    print(f"Errors: {len(errors)}")
    
    if max_turns:
        print(f"\n⚠ Note: {len(max_turns)} games hit the {max_turns_arg} turn limit (expected for random agents)")
    
    if failed or errors:
        print("\n=== Failed/Error Details ===")
        for status, game_id, error in results:
            if status in ["FAILED", "ERROR"]:
                print(f"\n{status}:")
                if game_id:
                    print(f"  Game ID: {game_id}")
                    print(f"  Replay: python -m scripts.replay_game {game_id}")
                if error:
                    print(f"  Error: {error}")
    
    # Return exit code based on results (only fail if there are actual failures, not max turns)
    return 0 if len(failed) == 0 and len(errors) == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multiple agent games and report results."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10,
        help="Number of games to test (default: 10)"
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
        help="Disable fast mode (use full text serialization)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(CPU cores, 8))"
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="random",
        choices=["random", "behavior_tree"],
        help="Type of agent to use (default: random)"
    )
    
    args = parser.parse_args()
    sys.exit(test_agents_batch(args.num_games, args.max_turns, fast_mode=not args.no_fast, workers=args.workers, agent_type=args.agent_type))

