#!/usr/bin/env python3
"""
Run a single drill with DSPy agent (for testing).
"""
import sys
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.dataset import DrillDataset
from dspy_ml.optimizer import DrillOptimizer
from dspy_ml.agent import DSPyDrillAgent


def main():
    parser = argparse.ArgumentParser(description="Run a single drill with DSPy agent")
    parser.add_argument("--module", required=True, help="DSPy module file path (.pkl)")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file path")
    parser.add_argument("--drill-id", type=int, required=True, help="Drill ID to run")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    
    # Find drill
    example = None
    for ex in dataset.examples:
        if ex.drill_id == args.drill_id:
            example = ex
            break
    
    if not example:
        print(f"Drill {args.drill_id} not found in dataset!", flush=True)
        return
    
    # Load module
    optimizer = DrillOptimizer()
    module, metadata = optimizer.load(args.module)
    
    # Create agent
    agent = DSPyDrillAgent(module)
    
    # Run drill
    print(f"Running drill {args.drill_id}...", flush=True)
    accuracy, reasoning, chosen_action = agent.evaluate_step(example)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Drill {args.drill_id} Results", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Correct: {'Yes' if accuracy > 0 else 'No'}", flush=True)
    print(f"\nReasoning:", flush=True)
    print(reasoning, flush=True)
    print(f"\nChosen Action:", flush=True)
    print(chosen_action, flush=True)
    print(f"\nExpected Action:", flush=True)
    print(example.expected_action, flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

