#!/usr/bin/env python3
"""
Analyze drill difficulty and classify as easy vs hard.
"""
import sys
import argparse
import json
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.difficulty_analysis import DifficultyAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze drill difficulty")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--baseline-results", help="Optional baseline results JSON (drill_id -> accuracy)")
    
    args = parser.parse_args()
    
    # Load baseline results if provided
    baseline_results = None
    if args.baseline_results:
        print(f"Loading baseline results from {args.baseline_results}...", flush=True)
        with open(args.baseline_results, 'r') as f:
            baseline_data = json.load(f)
            # Convert to {drill_id: accuracy}
            baseline_results = {
                entry['drill_id']: entry['accuracy']
                for entry in baseline_data.get('drills', [])
            }
        print(f"Loaded baseline results for {len(baseline_results)} drills", flush=True)
    
    # Initialize analyzer
    analyzer = DifficultyAnalyzer()
    
    # Load drills
    print("Loading drills from database...", flush=True)
    drills = analyzer.load_all_drills()
    print(f"Loaded {len(drills)} drills", flush=True)
    
    # Classify difficulty
    print("Classifying drill difficulty...", flush=True)
    difficulty_infos = analyzer.classify_difficulty(drills, baseline_results)
    
    # Get statistics
    stats = analyzer.get_statistics(difficulty_infos)
    print("\nDifficulty Analysis Statistics:", flush=True)
    print(f"  Total drills: {stats['total_drills']}", flush=True)
    print(f"  Hard drills: {stats['hard_drills']}", flush=True)
    print(f"  Easy drills: {stats['easy_drills']}", flush=True)
    print(f"  Drills with human guidelines: {stats['drills_with_guidelines']}", flush=True)
    print(f"  Hard drills WITHOUT guidelines: {stats['hard_without_guidelines']}", flush=True)
    print(f"\nConfidence distribution:", flush=True)
    print(f"  High (≥0.8): {stats['confidence_distribution']['high']}", flush=True)
    print(f"  Medium (0.5-0.8): {stats['confidence_distribution']['medium']}", flush=True)
    print(f"  Low (<0.5): {stats['confidence_distribution']['low']}", flush=True)
    
    # Save analysis
    analyzer.save_analysis(difficulty_infos, args.output)
    
    print(f"\n✓ Difficulty analysis complete!", flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

