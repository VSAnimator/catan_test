#!/usr/bin/env python3
"""
Run GEPA optimization on distilled agent (no guideline field).
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.dataset import DrillDataset
from dspy_ml.signature_distilled import CatanDrillSignatureDistilled
from dspy_ml.optimizer import DrillOptimizer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Optimize distilled DSPy agent with GEPA")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model")
    parser.add_argument("--reflection-model", default="gpt-5.2", help="Reflection model for GEPA")
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"], 
                       help="Reasoning effort for reflection model")
    parser.add_argument("--output", default="dspy_ml/data/optimized_distilled_module.pkl", 
                       help="Output path for optimized module")
    parser.add_argument("--num-candidates", type=int, default=10, help="Number of GEPA candidates per generation")
    parser.add_argument("--max-iters", type=int, default=10, help="Maximum GEPA iterations")
    
    args = parser.parse_args()
    
    # Initialize DSPy
    lm = dspy.LM(model=args.model)
    dspy.configure(lm=lm)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    print(f"Loaded {len(dataset.examples)} examples")

    # Append strategic principles to game_rules (match test_distilled behavior)
    principles_path = Path(__file__).parent.parent / "data" / "strategic_principles_v2.txt"
    principles_text = ""
    if principles_path.exists():
        with open(principles_path) as f:
            principles_text = f.read().strip()
    if principles_text:
        for ex in dataset.examples:
            ex.game_rules = f"{ex.game_rules}\n\n=== Strategic Principles ===\n{principles_text}"
    
    # Load seeded module as starting point
    seeded_path = Path(__file__).parent.parent / 'data' / 'seeded_module_initial.pkl'
    print(f"Loading seeded module from {seeded_path}...")
    module = dspy.ChainOfThought(CatanDrillSignatureDistilled)
    if seeded_path.exists():
        module.load(str(seeded_path))
        print("✓ Loaded seeded module with strategic principles")
    else:
        print("Warning: Seeded module not found, using default")
    
    # Initialize optimizer
    print("Initializing GEPA optimizer...")
    optimizer = DrillOptimizer(
        model_name=args.model,
        reflection_model=args.reflection_model,
        reflection_reasoning_effort=args.reasoning_effort
    )
    
    # Run optimization
    print()
    print("=" * 80)
    print(f"Starting GEPA optimization (distilled signature, no guidelines)")
    print(f"  Model: {args.model}")
    print(f"  Reflection Model: {args.reflection_model}")
    print(f"  Reasoning Effort: {args.reasoning_effort}")
    print(f"  Training examples: {len(dataset.examples)}")
    print(f"  Max iterations: {args.max_iters}")
    print("=" * 80)
    print()
    
    # Use the seeded module as the student for GEPA
    optimizer.module = module
    optimized_module = optimizer.optimize(
        train_examples=dataset.examples,
        val_examples=None
    )
    
    # Save optimized module
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print()
    print(f"Saving optimized module to {output_path}...")
    optimizer.save(optimized_module, str(output_path))
    
    print()
    print("=" * 80)
    print("✓ Optimization complete!")
    print(f"✓ Saved to: {output_path}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

