#!/usr/bin/env python3
"""
Validate multiple games and report rule violations.
Supports parallel validation for faster processing.
"""
import sys
import argparse
import multiprocessing
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_catan_rules import validate_game
from api.database import init_db, close_db_connection


def validate_single_game(args):
    """Validate a single game (for multiprocessing)."""
    game_num, total_games, game_id = args
    try:
        # Initialize database connection for this process
        init_db()
        
        violations, summary = validate_game(game_id)
        
        status = "PASS" if summary['total_violations'] == 0 else "FAIL"
        return (status, game_id, violations, summary, game_num)
    except Exception as e:
        return ("ERROR", game_id, None, {"total_violations": 0, "errors": 1, "warnings": 0, "violation_counts": {}}, game_num)
    finally:
        close_db_connection()


def main():
    parser = argparse.ArgumentParser(description="Validate multiple Catan games for rule violations")
    parser.add_argument("--num-games", type=int, default=10, help="Number of recent games to validate")
    parser.add_argument("--game-ids", nargs="+", help="Specific game IDs to validate")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for validation")
    
    args = parser.parse_args()
    
    if args.game_ids:
        game_ids = args.game_ids
    else:
        # Get recent games from database
        from api.database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM games ORDER BY rowid DESC LIMIT ?", (args.num_games,))
        game_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"Validating {len(game_ids)} games with {args.workers} workers...\n")
    
    if args.workers == 1:
        # Sequential validation
        total_violations = 0
        total_errors = 0
        total_warnings = 0
        violation_summary = defaultdict(int)
        games_with_violations = []
        
        for i, game_id in enumerate(game_ids, 1):
            print(f"[{i}/{len(game_ids)}] Validating {game_id}...", end=" ", flush=True)
            violations, summary = validate_game(game_id)
            
            if summary['total_violations'] > 0:
                print(f"❌ {summary['total_violations']} violations ({summary['errors']} errors, {summary['warnings']} warnings)")
                games_with_violations.append((game_id, violations, summary))
            else:
                print("✓ No violations")
            
            total_violations += summary['total_violations']
            total_errors += summary['errors']
            total_warnings += summary['warnings']
            
            for rule, count in summary['violation_counts'].items():
                violation_summary[rule] += count
    else:
        # Parallel validation
        print(f"Starting parallel validation with {args.workers} workers...\n")
        start_time = time.time()
        
        # Prepare arguments for workers
        game_args = [(i+1, len(game_ids), game_id) for i, game_id in enumerate(game_ids)]
        
        total_violations = 0
        total_errors = 0
        total_warnings = 0
        violation_summary = defaultdict(int)
        games_with_violations = []
        
        completed = 0
        with multiprocessing.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(validate_single_game, game_args):
                completed += 1
                status, game_id, violations, summary, game_num = result
                
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                
                if status == "PASS":
                    print(f"✓ Game {game_num}/{len(game_ids)} PASSED (completed: {completed}/{len(game_ids)}, rate: {rate:.2f} games/sec)", flush=True)
                elif status == "FAIL":
                    print(f"❌ Game {game_num}/{len(game_ids)} FAILED: {summary['total_violations']} violations (completed: {completed}/{len(game_ids)}, rate: {rate:.2f} games/sec)", flush=True)
                    games_with_violations.append((game_id, violations, summary))
                else:
                    print(f"✗ Game {game_num}/{len(game_ids)} ERROR (completed: {completed}/{len(game_ids)}, rate: {rate:.2f} games/sec)", flush=True)
                
                total_violations += summary['total_violations']
                total_errors += summary['errors']
                total_warnings += summary['warnings']
                
                for rule, count in summary['violation_counts'].items():
                    violation_summary[rule] += count
                
                # Show progress every 100 games or at completion
                if completed % 100 == 0 or completed == len(game_ids):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = len(game_ids) - completed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"\n📊 Progress: {completed}/{len(game_ids)} games validated ({completed*100//len(game_ids)}%)", flush=True)
                    print(f"   Rate: {rate:.2f} games/sec | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s\n", flush=True)
    
        elapsed_total = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("=== Validation Summary ===")
    print(f"Total games validated: {len(game_ids)}")
    print(f"Games with violations: {len(games_with_violations)}")
    print(f"Total violations: {total_violations}")
    print(f"  Errors: {total_errors}")
    print(f"  Warnings: {total_warnings}")
    if args.workers > 1:
        print(f"Total time: {elapsed_total:.1f}s")
        print(f"Average rate: {len(game_ids)/elapsed_total:.2f} games/sec")
    
    if violation_summary:
        print(f"\nViolation breakdown:")
        for rule, count in sorted(violation_summary.items(), key=lambda x: -x[1]):
            print(f"  {rule}: {count}")
    
    if games_with_violations:
        print(f"\n=== Games with Violations (showing first 10) ===")
        for game_id, violations, summary in games_with_violations[:10]:
            print(f"\nGame {game_id}:")
            print(f"  {summary['total_violations']} violations ({summary['errors']} errors, {summary['warnings']} warnings)")
            # Show first few violations
            for v in violations[:3]:
                print(f"    {v}")
            if len(violations) > 3:
                print(f"    ... and {len(violations) - 3} more")
        if len(games_with_violations) > 10:
            print(f"\n... and {len(games_with_violations) - 10} more games with violations")
    else:
        print("\n✓ All games passed validation!")
    
    print()


if __name__ == "__main__":
    main()

