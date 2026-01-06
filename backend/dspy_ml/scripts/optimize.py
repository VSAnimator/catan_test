#!/usr/bin/env python3
"""
Optimize DSPy module using GEPA on drill dataset.
"""
import sys
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.dataset import DrillDataset
from dspy_ml.optimizer import DrillOptimizer


def main():
    parser = argparse.ArgumentParser(description="Optimize DSPy module using GEPA")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file path")
    parser.add_argument("--output", required=True, help="Output module file path (.pkl)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--gepa-auto", choices=["light", "full"], default="light", help="GEPA auto mode")
    parser.add_argument("--train-split", type=float, default=1.0, help="Training split ratio (1.0 = use all data)")
    parser.add_argument("--val-split", type=float, default=0.0, help="Validation split ratio")
    parser.add_argument("--reflection-model", help="Model for GEPA reflection (defaults to --model)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="Reasoning effort for reflection model")
    parser.add_argument("--api-key", help="API key (optional, uses env vars if not provided)")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...", flush=True)
    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    
    print(f"Loaded {len(dataset.examples)} examples", flush=True)
    
    if not dataset.examples:
        print("No examples in dataset!", flush=True)
        return
    
    # Split dataset
    train, val, test = dataset.split(
        train_ratio=args.train_split,
        val_ratio=args.val_split
    )
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}", flush=True)
    
    # Initialize optimizer
    reflection_model = args.reflection_model or args.model
    print(f"Initializing GEPA optimizer:", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Reflection Model: {reflection_model}", flush=True)
    if args.reasoning_effort:
        print(f"  Reasoning Effort: {args.reasoning_effort}", flush=True)
    print(f"  GEPA Auto: {args.gepa_auto}", flush=True)
    
    optimizer = DrillOptimizer(
        model_name=args.model,
        api_key=args.api_key,
        gepa_auto=args.gepa_auto,
        reflection_model=reflection_model,
        reflection_reasoning_effort=args.reasoning_effort
    )
    
    # Optimize
    print("Starting GEPA optimization...", flush=True)
    optimized_module = optimizer.optimize(
        train_examples=train,
        val_examples=val if val else None
    )
    
    # Save module
    print(f"Saving optimized module to {args.output}...", flush=True)
    metadata = {
        "train_examples": len(train),
        "val_examples": len(val),
        "test_examples": len(test),
    }
    optimizer.save(optimized_module, args.output, metadata)
    
    print("Optimization complete!", flush=True)
    print(f"Optimized module saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

