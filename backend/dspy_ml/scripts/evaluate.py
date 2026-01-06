#!/usr/bin/env python3
"""
Evaluate optimized DSPy module on test dataset.
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.dataset import DrillDataset
from dspy_ml.optimizer import DrillOptimizer
from dspy_ml.agent import DSPyDrillAgent


def evaluate_example(agent, example, index):
    """Evaluate a single example (for parallel execution)."""
    accuracy, _, _ = agent.evaluate_step(example)
    return index, accuracy, example.expected_action.get("type", "unknown")


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimized DSPy module")
    parser.add_argument("--module", required=True, help="Optimized module file path (.pkl)")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file path")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all", help="Which split to evaluate")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model to use")
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Initialize DSPy with the model (temperature=0 for deterministic evaluation)
    print(f"Initializing DSPy with model: {args.model}", flush=True)
    lm = dspy.LM(model=args.model, temperature=0.0)
    dspy.configure(lm=lm)
    
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
    
    # Evaluate (in parallel)
    print(f"Running evaluation (parallel={args.parallel})...", flush=True)
    total = len(examples)
    results = {}
    action_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        futures = {
            executor.submit(evaluate_example, agent, example, i): i
            for i, example in enumerate(examples)
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            index, accuracy, action_type = future.result()
            results[index] = accuracy
            completed += 1
            
            # Track by action type
            action_type_stats[action_type]["total"] += 1
            if accuracy > 0:
                action_type_stats[action_type]["correct"] += 1
            
            if completed % 10 == 0:
                print(f"  Processed {completed}/{total} examples...", flush=True)
    
    # Count correct
    correct = sum(1 for acc in results.values() if acc > 0)
    
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

