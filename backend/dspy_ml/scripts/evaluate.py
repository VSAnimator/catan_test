#!/usr/bin/env python3
"""
Evaluate optimized DSPy module on test dataset.
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.dataset import DrillDataset
from dspy_ml.optimizer import DrillOptimizer
from dspy_ml.agent import DSPyDrillAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimized DSPy module")
    parser.add_argument("--module", required=True, help="Optimized module file path (.pkl)")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file path")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test", help="Which split to evaluate")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...", flush=True)
    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    
    # Split dataset
    train, val, test = dataset.split()
    
    # Select split
    if args.split == "train":
        examples = train
    elif args.split == "val":
        examples = val
    elif args.split == "test":
        examples = test
    else:  # all
        examples = train + val + test
    
    print(f"Evaluating on {len(examples)} examples ({args.split} split)", flush=True)
    
    # Load module
    print(f"Loading module from {args.module}...", flush=True)
    optimizer = DrillOptimizer()
    module, metadata = optimizer.load(args.module)
    
    print(f"Module metadata: {metadata}", flush=True)
    
    # Create agent
    agent = DSPyDrillAgent(module)
    
    # Evaluate
    print("Running evaluation...", flush=True)
    correct = 0
    total = 0
    action_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for example in examples:
        total += 1
        accuracy, reasoning, chosen_action = agent.evaluate_step(example)
        
        if accuracy > 0:
            correct += 1
        
        # Track by action type
        expected_action_type = example.expected_action.get("type", "unknown")
        action_type_stats[expected_action_type]["total"] += 1
        if accuracy > 0:
            action_type_stats[expected_action_type]["correct"] += 1
    
    # Report results
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluation Results ({args.split} split)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total examples: {total}", flush=True)
    print(f"Correct: {correct}", flush=True)
    print(f"Accuracy: {accuracy:.2%}", flush=True)
    print(f"\nPer-action-type accuracy:", flush=True)
    for action_type, stats in sorted(action_type_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  {action_type}: {stats['correct']}/{stats['total']} ({acc:.2%})", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

