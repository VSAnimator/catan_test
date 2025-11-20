#!/usr/bin/env python3
"""
Run multiple agent games and report results.

Usage:
    python -m scripts.test_agents_batch [--num-games N] [--max-turns N]
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.create_game import create_game_script
from scripts.run_agents import run_agents_script


def test_agents_batch(num_games: int = 10, max_turns: int = 1000):
    """Run multiple agent games and report results."""
    results = []
    
    print(f"=== Testing {num_games} games with agents ===\n")
    
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}...", end=" ", flush=True)
        
        try:
            # Create a game
            game_id = create_game_script(num_players=4, rng_seed=None)
            
            # Run agents
            exit_code = run_agents_script(game_id, max_turns=max_turns)
            
            if exit_code == 0:
                print("✓ PASSED")
                results.append(("PASSED", game_id, None))
            else:
                print("✗ FAILED")
                results.append(("FAILED", game_id, "Non-zero exit code"))
                
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            results.append(("ERROR", None, str(e)))
    
    # Print summary
    print("\n" + "=" * 80)
    print("=== Test Summary ===")
    print(f"Total games: {num_games}")
    
    passed = [r for r in results if r[0] == "PASSED"]
    failed = [r for r in results if r[0] == "FAILED"]
    errors = [r for r in results if r[0] == "ERROR"]
    
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Errors: {len(errors)}")
    
    if failed or errors:
        print("\n=== Failed/Error Details ===")
        for status, game_id, error in results:
            if status != "PASSED":
                print(f"\n{status}:")
                if game_id:
                    print(f"  Game ID: {game_id}")
                    print(f"  Replay: python -m scripts.replay_game {game_id}")
                if error:
                    print(f"  Error: {error}")
    
    # Return exit code based on results
    return 0 if len(passed) == num_games else 1


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
    
    args = parser.parse_args()
    sys.exit(test_agents_batch(args.num_games, args.max_turns))

