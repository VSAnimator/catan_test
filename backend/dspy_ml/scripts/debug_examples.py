#!/usr/bin/env python3
"""
Debug DSPy agent predictions on specific examples.
"""
import sys
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.dataset import DrillDataset
from dspy_ml.signature import CatanDrillSignature
from dspy_ml.agent import DSPyDrillAgent


def main():
    parser = argparse.ArgumentParser(description="Debug DSPy agent predictions")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file path")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model to use")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of examples to debug")
    
    args = parser.parse_args()
    
    # Initialize DSPy
    lm = dspy.LM(model=args.model)
    dspy.configure(lm=lm)
    
    # Create unoptimized module
    module = dspy.ChainOfThought(CatanDrillSignature)
    
    # Load dataset
    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    
    print(f"Loaded {len(dataset.examples)} examples", flush=True)
    
    # Create agent
    agent = DSPyDrillAgent(module)
    
    # Debug first few examples
    print(f"\nDebugging first {args.num_examples} examples...\n", flush=True)
    
    for i, example in enumerate(dataset.examples[:args.num_examples]):
        print(f"{'='*80}", flush=True)
        print(f"Example {i+1} (Drill ID: {example.drill_id})", flush=True)
        print(f"{'='*80}", flush=True)
        
        # Show viable actions
        print("\nViable Actions:", flush=True)
        viable_lines = example.viable_actions.split('\n')
        for line in viable_lines[:20]:  # Show first 20 lines
            print(f"  {line}", flush=True)
        if len(viable_lines) > 20:
            print(f"  ... ({len(viable_lines) - 20} more lines)", flush=True)
        
        # Show expected action
        print("\nExpected Action:", flush=True)
        print(f"  {example.expected_action}", flush=True)
        
        # Show correct actions
        print("\nCorrect Actions:", flush=True)
        for ca in example.correct_actions:
            print(f"  {ca}", flush=True)
        
        # Show incorrect actions if any
        if example.incorrect_actions:
            print(f"\nIncorrect Actions ({len(example.incorrect_actions)}):", flush=True)
            for ia in example.incorrect_actions[:5]:
                print(f"  {ia}", flush=True)
            if len(example.incorrect_actions) > 5:
                print(f"  ... ({len(example.incorrect_actions) - 5} more)", flush=True)
        
        # Run prediction
        print("\nRunning prediction...", flush=True)
        accuracy, reasoning, chosen_action = agent.evaluate_step(example)
        
        # Show prediction
        print("\nPredicted Action:", flush=True)
        print(f"  {chosen_action}", flush=True)
        
        print("\nReasoning:", flush=True)
        reasoning_lines = reasoning.split('\n')
        for line in reasoning_lines[:10]:
            print(f"  {line}", flush=True)
        if len(reasoning_lines) > 10:
            print(f"  ... ({len(reasoning_lines) - 10} more lines)", flush=True)
        
        print(f"\nResult: {'✓ CORRECT' if accuracy > 0 else '✗ INCORRECT'}", flush=True)
        print()


if __name__ == "__main__":
    main()

