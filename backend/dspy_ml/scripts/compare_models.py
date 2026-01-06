"""
Compare unoptimized vs optimized DSPy agent predictions on the full dataset.
Saves detailed results to a JSON file for analysis.
"""
import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.signature import CatanDrillSignature
from dspy_ml.dataset import DrillDataset
from dspy_ml.agent import DSPyDrillAgent
from dspy_ml.optimizer import DrillOptimizer


def main():
    parser = argparse.ArgumentParser(description="Compare unoptimized vs optimized DSPy agents")
    parser.add_argument("--dataset", default="dspy_ml/data/drills_dataset.json", help="Path to dataset JSON")
    parser.add_argument("--optimized-module", default="dspy_ml/data/optimized_modules/gepa_gpt52_recovered.pkl", help="Path to optimized module")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model to use")
    parser.add_argument("--output", default="dspy_ml/data/comparison_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Initialize DSPy with deterministic settings
    print(f"Initializing DSPy with model: {args.model}", flush=True)
    lm = dspy.LM(model=args.model, temperature=0.0)
    dspy.configure(lm=lm)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...", flush=True)
    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    print(f"Loaded {len(dataset.examples)} examples", flush=True)
    
    # Create unoptimized agent
    print("Creating unoptimized agent...", flush=True)
    unopt_module = dspy.ChainOfThought(CatanDrillSignature)
    unopt_agent = DSPyDrillAgent(unopt_module)
    
    # Load optimized agent
    print(f"Loading optimized agent from {args.optimized_module}...", flush=True)
    optimizer = DrillOptimizer()
    opt_module, opt_metadata = optimizer.load(args.optimized_module)
    opt_agent = DSPyDrillAgent(opt_module)
    
    # Run comparison
    print(f"Running comparison on {len(dataset.examples)} examples...", flush=True)
    
    results = []
    unopt_correct = 0
    opt_correct = 0
    both_correct = 0
    neither_correct = 0
    unopt_only = 0
    opt_only = 0
    
    for i, example in enumerate(dataset.examples):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(dataset.examples)} examples...", flush=True)
        
        # Run both agents
        unopt_acc, unopt_reasoning, unopt_pred = unopt_agent.evaluate_step(example)
        opt_acc, opt_reasoning, opt_pred = opt_agent.evaluate_step(example)
        
        # Track stats
        if unopt_acc == 1.0:
            unopt_correct += 1
        if opt_acc == 1.0:
            opt_correct += 1
        if unopt_acc == 1.0 and opt_acc == 1.0:
            both_correct += 1
        elif unopt_acc == 0.0 and opt_acc == 0.0:
            neither_correct += 1
        elif unopt_acc == 1.0 and opt_acc == 0.0:
            unopt_only += 1
        elif unopt_acc == 0.0 and opt_acc == 1.0:
            opt_only += 1
        
        # Store result
        result = {
            "index": i,
            "expected_action": example.expected_action,
            "unopt_accuracy": unopt_acc,
            "unopt_prediction": unopt_pred,
            "opt_accuracy": opt_acc,
            "opt_prediction": opt_pred,
        }
        results.append(result)
    
    # Create summary
    summary = {
        "total_examples": len(dataset.examples),
        "unopt_correct": unopt_correct,
        "unopt_accuracy": unopt_correct / len(dataset.examples),
        "opt_correct": opt_correct,
        "opt_accuracy": opt_correct / len(dataset.examples),
        "both_correct": both_correct,
        "neither_correct": neither_correct,
        "unopt_only_correct": unopt_only,
        "opt_only_correct": opt_only,
        "optimized_module_metadata": opt_metadata,
    }
    
    # Save results
    output_data = {
        "summary": summary,
        "results": results,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}", flush=True)
    print("Comparison Results", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total examples: {len(dataset.examples)}", flush=True)
    print(f"\nUnoptimized: {unopt_correct}/{len(dataset.examples)} ({100*unopt_correct/len(dataset.examples):.2f}%)", flush=True)
    print(f"Optimized:   {opt_correct}/{len(dataset.examples)} ({100*opt_correct/len(dataset.examples):.2f}%)", flush=True)
    print(f"\nBoth correct:     {both_correct}", flush=True)
    print(f"Neither correct:  {neither_correct}", flush=True)
    print(f"Unopt only:       {unopt_only}", flush=True)
    print(f"Opt only:         {opt_only}", flush=True)
    print(f"\nResults saved to: {args.output}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

