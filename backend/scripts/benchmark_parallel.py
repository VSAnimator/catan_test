#!/usr/bin/env python3
"""
Benchmark parallel agent testing with different worker counts.
"""
import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_agents_batch import run_single_game


def benchmark_workers(num_games: int, worker_counts: list, fast_mode: bool = True):
    """Run benchmarks with different worker counts."""
    print(f"=== Benchmarking {num_games} games with different worker counts ===\n")
    print(f"Fast mode: {fast_mode}\n")
    
    results = []
    
    for workers in worker_counts:
        print(f"\n{'='*80}")
        print(f"Testing with {workers} worker(s)...")
        print(f"{'='*80}\n")
        
        # Prepare arguments for each game
        game_args = [
            (i+1, num_games, 1000, fast_mode)
            for i in range(num_games)
        ]
        
        start_time = time.time()
        
        # Run games sequentially if workers=1, otherwise use multiprocessing
        if workers == 1:
            # Sequential execution
            completed = 0
            for args in game_args:
                result = run_single_game(args)
                completed += 1
                status, game_id, error, game_num = result
                if status == "PASSED":
                    print(f"✓ Game {game_num}/{num_games} PASSED", flush=True)
                else:
                    print(f"✗ Game {game_num}/{num_games} {status}", flush=True)
        else:
            # Parallel execution
            import multiprocessing
            with multiprocessing.Pool(processes=workers) as pool:
                completed = 0
                for result in pool.imap_unordered(run_single_game, game_args):
                    completed += 1
                    status, game_id, error, game_num = result
                    if status == "PASSED":
                        print(f"✓ Game {game_num}/{num_games} PASSED", flush=True)
                    else:
                        print(f"✗ Game {game_num}/{num_games} {status}", flush=True)
        
        elapsed = time.time() - start_time
        rate = num_games / elapsed if elapsed > 0 else 0
        
        results.append({
            'workers': workers,
            'time': elapsed,
            'rate': rate
        })
        
        print(f"\n{workers} worker(s): {elapsed:.2f}s ({rate:.2f} games/sec)")
    
    # Calculate speedup
    if len(results) > 0:
        baseline_time = results[0]['time']  # Time with 1 worker
        
        print(f"\n{'='*80}")
        print("=== Benchmark Results ===")
        print(f"{'='*80}\n")
        
        print(f"{'Workers':<10} {'Time (s)':<12} {'Rate (games/s)':<18} {'Speedup':<12} {'Efficiency':<12}")
        print("-" * 80)
        
        for r in results:
            speedup = baseline_time / r['time'] if r['time'] > 0 else 0
            efficiency = speedup / r['workers'] * 100 if r['workers'] > 0 else 0
            print(f"{r['workers']:<10} {r['time']:<12.2f} {r['rate']:<18.2f} {speedup:<12.2f} {efficiency:<12.1f}%")
        
        print(f"\n{'='*80}")
        print(f"Baseline (1 worker): {baseline_time:.2f}s")
        print(f"{'='*80}\n")
        
        # Summary
        best_result = max(results, key=lambda x: x['rate'])
        best_speedup = baseline_time / best_result['time'] if best_result['time'] > 0 else 0
        print(f"Best performance: {best_result['workers']} workers")
        print(f"  Time: {best_result['time']:.2f}s")
        print(f"  Rate: {best_result['rate']:.2f} games/sec")
        print(f"  Speedup: {best_speedup:.2f}x")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark parallel agent testing with different worker counts."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10,
        help="Number of games to run for each worker count (default: 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Worker counts to test (default: 1 2 4 8)"
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Disable fast mode (use full serialization)"
    )
    
    args = parser.parse_args()
    
    benchmark_workers(
        num_games=args.num_games,
        worker_counts=args.workers,
        fast_mode=not args.no_fast
    )

