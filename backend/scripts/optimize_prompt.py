#!/usr/bin/env python3
"""
Script to optimize LLM agent prompts using DSPy.

This script:
1. Loads drills from the database
2. Uses DSPy to optimize the system prompt
3. Evaluates the optimized prompt
4. Saves it to the database
"""
import sys
import os
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import (
    list_drills,
    get_drill_steps,
    save_optimized_prompt,
    get_drill,
)
from agents.dspy_optimizer import PromptOptimizer, DrillExample
from agents.llm_agent import LLMAgent


def load_drills(drill_ids=None, limit=200):
    """Load drills from database."""
    if drill_ids:
        drills = []
        for drill_id in drill_ids:
            drill_row = get_drill(drill_id)
            if drill_row:
                steps = get_drill_steps(drill_id)
                drills.append({
                    "id": drill_id,
                    "steps": [
                        {
                            "idx": r["idx"],
                            "player_id": r["player_id"],
                            "state": json.loads(r["state_json"]),
                            "expected_action": json.loads(r["expected_action_json"]),
                            "correct_actions": json.loads(r["correct_actions_json"]) if r.get("correct_actions_json") else None,
                            "incorrect_actions": json.loads(r["incorrect_actions_json"]) if r.get("incorrect_actions_json") else None,
                        }
                        for r in steps
                    ]
                })
        return drills
    else:
        drill_rows = list_drills(limit=limit)
        drills = []
        for drill_row in drill_rows:
            drill_id = drill_row["id"]
            steps = get_drill_steps(drill_id)
            drills.append({
                "id": drill_id,
                "steps": [
                    {
                        "idx": r["idx"],
                        "player_id": r["player_id"],
                        "state": json.loads(r["state_json"]),
                        "expected_action": json.loads(r["expected_action_json"]),
                        "correct_actions": json.loads(r["correct_actions_json"]) if r.get("correct_actions_json") else None,
                        "incorrect_actions": json.loads(r["incorrect_actions_json"]) if r.get("incorrect_actions_json") else None,
                    }
                    for r in steps
                ]
            })
        return drills


def main():
    parser = argparse.ArgumentParser(description="Optimize LLM agent prompts using DSPy")
    parser.add_argument("--drill-ids", type=int, nargs="+", help="Specific drill IDs to use")
    parser.add_argument("--train-split", type=float, default=None, help="Fraction for training (default: None = use all for training, no test split)")
    parser.add_argument("--optimization-method", choices=["bootstrap", "miprov2", "gepa", "copro"], default="bootstrap", help="Optimization method (bootstrap=BootstrapFewShot, miprov2=MIPROv2, gepa=GEPA, copro=COPRO)")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation after optimization (useful when using all data for training)")
    parser.add_argument("--include-higher-level-features", action="store_true", help="Include higher-level features in optimization")
    parser.add_argument("--output-name", required=True, help="Name for the optimized prompt")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use for optimization")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of drills to load")
    
    args = parser.parse_args()
    
    print(f"Loading drills...", flush=True)
    drills = load_drills(args.drill_ids, args.limit)
    print(f"Loaded {len(drills)} drills", flush=True)
    
    if not drills:
        print("No drills found!", flush=True)
        return
    
    # Get base system prompt from default LLM agent
    base_agent = LLMAgent("player_0")
    base_system_prompt = base_agent._get_default_system_prompt()
    
    # Initialize optimizer
    print(f"Initializing optimizer with method={args.optimization_method}...", flush=True)
    optimizer = PromptOptimizer(
        base_system_prompt=base_system_prompt,
        model_name=args.model,
        include_higher_level_features=args.include_higher_level_features
    )
    
    # Prepare examples
    print("Preparing examples...", flush=True)
    examples = optimizer.prepare_examples(drills)
    print(f"Prepared {len(examples)} examples", flush=True)
    
    if not examples:
        print("No valid examples found!", flush=True)
        return
    
    # Split train/test (or use all for training)
    if args.train_split is not None:
        split_idx = int(len(examples) * args.train_split)
        train_examples = examples[:split_idx]
        test_examples = examples[split_idx:]
        print(f"Train examples: {len(train_examples)}, Test examples: {len(test_examples)}", flush=True)
    else:
        # Use all examples for training
        train_examples = examples
        test_examples = []
        print(f"Using all {len(train_examples)} examples for training (no test split)", flush=True)
    
    # Optimize
    print(f"Optimizing prompt...", flush=True)
    # Pass val_examples only if we have a separate test set
    val_examples_for_optimization = test_examples if test_examples else None
    optimized_prompt = optimizer.optimize(
        train_examples,
        method=args.optimization_method,
        num_iterations=10,
        val_examples=val_examples_for_optimization
    )
    
    # Evaluate (optional)
    eval_results = None
    if not args.skip_evaluation and test_examples:
        print("Evaluating optimized prompt on test set...", flush=True)
        eval_results = optimizer.evaluate(test_examples, optimized_prompt)
        print(f"Test set evaluation results: {eval_results}", flush=True)
    elif not args.skip_evaluation and train_examples:
        print("Evaluating optimized prompt on training set...", flush=True)
        eval_results = optimizer.evaluate(train_examples, optimized_prompt)
        print(f"Training set evaluation results: {eval_results}", flush=True)
    else:
        print("Skipping evaluation (--skip-evaluation flag set or no test examples)", flush=True)
    
    # Save to database
    metadata = {
        "optimization_method": args.optimization_method,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples) if test_examples else 0,
        "train_split": args.train_split,
        "evaluation": eval_results,
        "include_higher_level_features": args.include_higher_level_features,
        "model": args.model,
    }
    
    print(f"Saving optimized prompt as '{args.output_name}'...", flush=True)
    save_optimized_prompt(
        name=args.output_name,
        system_prompt=optimized_prompt,
        metadata=metadata,
        is_default=False
    )
    
    print("Done!", flush=True)


if __name__ == "__main__":
    main()

