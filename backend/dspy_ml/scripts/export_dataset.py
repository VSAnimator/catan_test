#!/usr/bin/env python3
"""
Export drills from database to dataset format.

Only exports the first step (idx=0) of each drill.
"""
import sys
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.dataset import DrillDataset


def main():
    parser = argparse.ArgumentParser(description="Export drills to dataset format")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--drill-ids", type=int, nargs="+", help="Specific drill IDs to export")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of drills to export")
    
    args = parser.parse_args()
    
    print("Loading drills from database...", flush=True)
    dataset = DrillDataset()
    
    examples = dataset.load_from_database(
        drill_ids=args.drill_ids,
        limit=args.limit
    )
    
    print(f"Loaded {len(examples)} examples (first step of each drill)", flush=True)
    
    if not examples:
        print("No examples found!", flush=True)
        return
    
    print(f"Exporting to {args.output}...", flush=True)
    dataset.export_to_json(args.output)
    
    print(f"Exported {len(examples)} examples to {args.output}", flush=True)


if __name__ == "__main__":
    main()

