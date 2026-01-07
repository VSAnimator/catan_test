#!/usr/bin/env python3
"""
Test distilled DSPy agent (no guideline field) on drill dataset.
"""
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.dataset import DrillDataset
from dspy_ml.signature_distilled import CatanDrillSignatureDistilled
from api.routes import _canonical_action_dict
import json

# Load strategic principles (v2) once for all predictions
PRINCIPLES_PATH = Path(__file__).parent.parent / "data" / "strategic_principles_v2.txt"
if PRINCIPLES_PATH.exists():
    with open(PRINCIPLES_PATH) as f:
        STRATEGIC_PRINCIPLES_TEXT = f.read().strip()
else:
    STRATEGIC_PRINCIPLES_TEXT = ""


class DistilledDrillAgent:
    """Agent using distilled signature (no guideline field)."""
    
    def __init__(self, module):
        self.module = module
    
    def predict(self, example):
        """Make prediction without using guideline field."""
        # Inject strategic principles into game_rules so the model can reference them explicitly
        if STRATEGIC_PRINCIPLES_TEXT:
            combined_rules = f"{example.game_rules}\n\n=== Strategic Principles ===\n{STRATEGIC_PRINCIPLES_TEXT}"
        else:
            combined_rules = example.game_rules

        # Call DSPy module WITHOUT guideline
        result = self.module(
            game_rules=combined_rules,
            observation=example.observation,
            viable_actions=example.viable_actions
            # NO guideline field!
        )
        
        reasoning = result.reasoning if hasattr(result, 'reasoning') else ""
        
        # Parse chosen_action JSON string with robust error handling
        chosen_action_str = result.chosen_action if hasattr(result, 'chosen_action') else "null"
        chosen_action_dict = None
        
        if chosen_action_str and chosen_action_str.lower() != "null":
            try:
                chosen_action_dict = json.loads(chosen_action_str)
            except (json.JSONDecodeError, TypeError) as e:
                # Try to fix common JSON errors
                try:
                    cleaned = chosen_action_str.rstrip('}').rstrip() + '}'
                    chosen_action_dict = json.loads(cleaned)
                except:
                    import re
                    match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', chosen_action_str)
                    if match:
                        try:
                            chosen_action_dict = json.loads(match.group(0))
                        except:
                            chosen_action_dict = None
                    else:
                        chosen_action_dict = None
        
        return reasoning, chosen_action_dict
    
    def evaluate_step(self, example):
        """Evaluate on a single drill step."""
        reasoning, predicted_action = self.predict(example)
        
        if not predicted_action:
            return 0.0, reasoning, predicted_action
        
        # Check if prediction matches any correct action
        canonical_predicted = _canonical_action_dict(predicted_action, state=example.state)
        for correct_action in example.correct_actions:
            if _canonical_action_dict(correct_action, state=example.state) == canonical_predicted:
                return 1.0, reasoning, predicted_action
        
        return 0.0, reasoning, predicted_action


def evaluate_example(agent, example, index):
    """Evaluate a single example (for parallel execution)."""
    accuracy, _, _ = agent.evaluate_step(example)
    return index, accuracy


def main():
    parser = argparse.ArgumentParser(description="Test distilled DSPy agent (no guidelines)")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file path")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model to use")
    parser.add_argument("--module", help="Path to saved module (optional, uses seeded if not provided)")
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Initialize DSPy
    lm = dspy.LM(model=args.model)
    dspy.configure(lm=lm)
    
    # Load module
    if args.module:
        print(f"Loading module from {args.module}...")
        module = dspy.ChainOfThought(CatanDrillSignatureDistilled)
        module.load(args.module)
    else:
        print("Creating fresh seeded module...")
        module = dspy.ChainOfThought(CatanDrillSignatureDistilled)
        # Load seeded instructions
        seeded_path = Path(__file__).parent.parent / 'data' / 'seeded_module_initial.pkl'
        if seeded_path.exists():
            module.load(str(seeded_path))
            print(f"Loaded seeded module from {seeded_path}")
        else:
            print("Warning: Using default instructions (seeded module not found)")
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    
    print(f"Loaded {len(dataset.examples)} examples")
    
    # Create agent
    agent = DistilledDrillAgent(module)
    
    # Evaluate (in parallel)
    print(f"Testing distilled agent (NO guideline field) with parallel={args.parallel}...")
    correct = 0
    total = len(dataset.examples)
    
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(evaluate_example, agent, example, i): i
            for i, example in enumerate(dataset.examples)
        }
        
        for future in as_completed(futures):
            idx, accuracy = future.result()
            if accuracy == 1.0:
                correct += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{total}...")
    
    print()
    print("=" * 60)
    print("Distilled Agent Results (NO guidelines)")
    print("=" * 60)
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {100*correct/total:.2f}% ({correct}/{total})")
    print("=" * 60)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

