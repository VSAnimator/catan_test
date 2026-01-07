#!/usr/bin/env python3
"""
Compare distilled DSPy agents with and without strategic principles.

- Agent A: seeded module with situational strategic principles (from strategic_principles_v2.txt)
- Agent B: plain instructions without the strategic principles section

Outputs accuracy for each and shows drills where their predictions differ.
"""
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Dict, Any
import argparse

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.dataset import DrillDataset
from dspy_ml.signature_distilled import CatanDrillSignatureDistilled
from api.routes import _canonical_action_dict

# Load strategic principles (v2) once
PRINCIPLES_PATH = Path(__file__).parent.parent / "data" / "strategic_principles_v2.txt"
if PRINCIPLES_PATH.exists():
    with open(PRINCIPLES_PATH) as f:
        STRATEGIC_PRINCIPLES_TEXT = f.read().strip()
else:
    STRATEGIC_PRINCIPLES_TEXT = ""


def build_plain_module() -> dspy.Module:
    """Create a distilled module without the strategic principles section."""
    module = dspy.ChainOfThought(CatanDrillSignatureDistilled)

    base_instructions = """You are selecting the NEXT single action in a step-by-step Catan "drill" loop.

INPUTS (you will be given these 3 blocks)
- game_rules: rules reference text (no strategic principles)
- observation: the full current game state (resources, VP, board, robber, whose turn, pending trade offers, actions taken this turn, etc.)
- viable_actions: the ONLY actions you are allowed to choose from RIGHT NOW (the list may be filtered to correct/incorrect options for training)

TASK
Pick the best immediate next action. You MUST pick an action that appears in viable_actions (exactly one).

REQUIRED REASONING CHECKLIST (be concise):
- Prefer immediate, concrete value over speculative trades/dev plays.
- Only propose trades/dev plays if they create a concrete, immediate payoff this turn.
- Do NOT invent actions outside viable_actions.

"""

    output_instructions = """CRITICAL OUTPUT FORMAT
1) Your final output must be JSON with exactly these keys:
   - "reasoning" (string) - explain your strategic thinking
   - "chosen_action" (string) - JSON string of the action

2) The value of "chosen_action" MUST itself be a JSON string encoding an object with:
   - "type" (the action type as a string)
   - "payload" (dict with action-specific fields, or null if no payload)

Example chosen_action values:
- {"type": "build_road", "payload": {"road_edge_id": 14}}
- {"type": "end_turn", "payload": null}
- {"type": "propose_trade", "payload": {"give_resources": {"wood": 1}, "receive_resources": {"brick": 1}, "target_player_ids": ["player_1"]}}

When reasoning, consider which factors (board, resources, tempo) matter most to choose the action from viable_actions.
"""

    instructions_plain = base_instructions + output_instructions

    if hasattr(module, "predict") and hasattr(module.predict, "signature"):
        module.predict.signature.instructions = instructions_plain
    return module


def robust_parse_action(chosen_action_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse chosen_action JSON string with resilience to minor formatting issues."""
    if not chosen_action_str or str(chosen_action_str).lower() == "null":
        return None
    try:
        return json.loads(chosen_action_str)
    except (json.JSONDecodeError, TypeError):
        # Attempt simple cleanup (extra trailing brace)
        try:
            cleaned = chosen_action_str.rstrip("}").rstrip() + "}"
            return json.loads(cleaned)
        except Exception:
            # Try to extract first JSON object substring
            import re
            match = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", chosen_action_str)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return None
            return None


class DistilledEvalAgent:
    """Evaluation helper mirroring test_distilled logic, with optional principles injection."""

    def __init__(self, module: dspy.Module, include_principles: bool):
        self.module = module
        self.include_principles = include_principles

    def predict(self, example):
        if self.include_principles and STRATEGIC_PRINCIPLES_TEXT:
            combined_rules = f"{example.game_rules}\n\n=== Strategic Principles ===\n{STRATEGIC_PRINCIPLES_TEXT}"
        else:
            combined_rules = example.game_rules

        result = self.module(
            game_rules=combined_rules,
            observation=example.observation,
            viable_actions=example.viable_actions,
        )

        reasoning = getattr(result, "reasoning", "") or ""
        chosen_action_str = getattr(result, "chosen_action", "null") or "null"
        chosen_action_dict = robust_parse_action(chosen_action_str)
        return reasoning, chosen_action_dict

    def evaluate_step(self, example):
        reasoning, predicted_action = self.predict(example)
        if not predicted_action:
            return 0.0, reasoning, predicted_action

        canonical_pred = _canonical_action_dict(predicted_action, state=example.state)
        for correct_action in example.correct_actions:
            if _canonical_action_dict(correct_action, state=example.state) == canonical_pred:
                return 1.0, reasoning, predicted_action
        return 0.0, reasoning, predicted_action


def evaluate_example(agent: DistilledEvalAgent, example, idx: int) -> Tuple[int, float, Optional[Dict[str, Any]]]:
    acc, _, chosen = agent.evaluate_step(example)
    return idx, acc, chosen


def run_eval(agent: DistilledEvalAgent, dataset: DrillDataset, parallel: int = 10):
    """Run evaluation in parallel, return (correct_count, total, per_index_results)."""
    correct = 0
    total = len(dataset.examples)
    results = {}

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(evaluate_example, agent, ex, i): i
            for i, ex in enumerate(dataset.examples)
        }
        for fut in as_completed(futures):
            idx, acc, chosen = fut.result()
            results[idx] = {"acc": acc, "chosen": chosen}
            if acc == 1.0:
                correct += 1
    return correct, total, results


def main():
    parser = argparse.ArgumentParser(description="Compare distilled modules with vs without strategic principles")
    parser.add_argument("--dataset", required=True, help="Dataset JSON file")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model")
    parser.add_argument("--parallel", type=int, default=10, help="Parallel workers")
    args = parser.parse_args()

    # Init LM
    lm = dspy.LM(model=args.model)
    dspy.configure(lm=lm)

    # Load dataset
    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    print(f"Loaded {len(dataset.examples)} examples from {args.dataset}")

    # Modules
    principled_path = Path(__file__).parent.parent / "data" / "seeded_module_initial.pkl"
    module_principled = dspy.ChainOfThought(CatanDrillSignatureDistilled)
    module_principled.load(str(principled_path))
    print(f"Loaded principled module from {principled_path}")

    module_plain = build_plain_module()
    print("Built plain module (no strategic principles section)")

    agent_principled = DistilledEvalAgent(module_principled, include_principles=True)
    agent_plain = DistilledEvalAgent(module_plain, include_principles=False)

    # Evaluate
    print("Evaluating principled module...")
    correct_p, total, results_p = run_eval(agent_principled, dataset, args.parallel)
    print("Evaluating plain module...")
    correct_b, _, results_b = run_eval(agent_plain, dataset, args.parallel)

    acc_p = 100 * correct_p / total
    acc_b = 100 * correct_b / total

    print("\n============================================================")
    print("Comparison")
    print("============================================================")
    print(f"Principled accuracy: {acc_p:.2f}% ({correct_p}/{total})")
    print(f"Plain accuracy     : {acc_b:.2f}% ({correct_b}/{total})")

    # Mismatches where outputs differ
    diffs = []
    for idx, ex in enumerate(dataset.examples):
        cp = results_p[idx]["chosen"]
        cb = results_b[idx]["chosen"]
        ap = results_p[idx]["acc"]
        ab = results_b[idx]["acc"]
        if cp != cb or ap != ab:
            diffs.append((idx, ex, cp, cb, ap, ab))

    print(f"\nCases where outputs differ: {len(diffs)}")
    print("Showing up to 12 examples:")
    print("------------------------------------------------------------")
    for i, (idx, ex, cp, cb, ap, ab) in enumerate(diffs[:12], 1):
        print(f"{i}. Drill {ex.drill_id} ({ex.expected_action['type']})")
        print(f"   Principled: {'CORRECT' if ap==1.0 else 'WRONG'} | {cp}")
        print(f"   Plain     : {'CORRECT' if ab==1.0 else 'WRONG'} | {cb}")
        print("   Viable actions excerpt:")
        print(f"     {ex.viable_actions[:200].replace(chr(10), ' ')}...")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
