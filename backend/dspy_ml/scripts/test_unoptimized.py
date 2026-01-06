#!/usr/bin/env python3
"""
Test unoptimized DSPy agent on drill dataset.
"""
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.dataset import DrillDataset
from dspy_ml.signature import CatanDrillSignature
from dspy_ml.agent import DSPyDrillAgent


def evaluate_example(agent, example, index):
    """Evaluate a single example (for parallel execution)."""
    accuracy, _, _ = agent.evaluate_step(example)
    return index, accuracy


def main():
    parser = argparse.ArgumentParser(description="Test unoptimized DSPy agent")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file path")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model to use")
    parser.add_argument("--api-key", help="API key (optional, uses env vars if not provided)")
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel workers (default: 10)")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        import os
        if "gpt" in args.model.lower() or "openai" in args.model.lower():
            os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Initialize DSPy
    lm = dspy.LM(model=args.model)
    dspy.configure(lm=lm)
    
    # Create unoptimized module
    print(f"Creating unoptimized DSPy module (model={args.model})...", flush=True)
    module = dspy.ChainOfThought(CatanDrillSignature)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...", flush=True)
    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    
    print(f"Loaded {len(dataset.examples)} examples", flush=True)
    
    if not dataset.examples:
        print("No examples in dataset!", flush=True)
        return
    
    # Create agent
    agent = DSPyDrillAgent(module)
    
    # Test on all examples (in parallel)
    print(f"Testing unoptimized agent on all examples (parallel={args.parallel})...", flush=True)
    total = len(dataset.examples)
    results = {}
    
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        futures = {
            executor.submit(evaluate_example, agent, example, i): i
            for i, example in enumerate(dataset.examples)
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            index, accuracy = future.result()
            results[index] = accuracy
            completed += 1
            
            if completed % 10 == 0:
                print(f"  Processed {completed}/{total} examples...", flush=True)
    
    # Count correct
    correct = sum(1 for acc in results.values() if acc > 0)
    
    # Report results
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*60}", flush=True)
    print(f"Unoptimized Agent Results", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total examples: {total}", flush=True)
    print(f"Correct: {correct}", flush=True)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

