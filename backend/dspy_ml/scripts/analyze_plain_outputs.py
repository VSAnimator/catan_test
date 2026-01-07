#!/usr/bin/env python3
"""
Analyze plain distilled agent outputs and dump mismatches.
"""
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from dspy_ml.dataset import DrillDataset
from dspy_ml.signature_distilled import CatanDrillSignatureDistilled
from api.routes import _canonical_action_dict


def build_plain_module():
    module = dspy.ChainOfThought(CatanDrillSignatureDistilled)
    base = """You are selecting the NEXT single action in a step-by-step Catan "drill" loop.
INPUTS
- game_rules: rules reference text (no strategic principles)
- observation: full current game state
- viable_actions: ONLY actions you can choose right now
TASK: pick exactly one action from viable_actions.
"""
    out = """CRITICAL OUTPUT FORMAT
- reasoning: string
- chosen_action: JSON string {"type": "...", "payload": {...} or null}
"""
    module.predict.signature.instructions = base + out
    return module

def robust_parse(chosen_action_str: Optional[str]) -> Optional[Dict[str, Any]]:
    if not chosen_action_str or str(chosen_action_str).lower() == "null":
        return None
    try:
        return json.loads(chosen_action_str)
    except Exception:
        try:
            cleaned = chosen_action_str.rstrip('}').rstrip() + '}'
            return json.loads(cleaned)
        except Exception:
            import re
            m = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", chosen_action_str)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

def eval_example(module, ex):
    res = module(
        game_rules=ex.game_rules,
        observation=ex.observation,
        viable_actions=ex.viable_actions,
    )
    chosen = robust_parse(getattr(res, 'chosen_action', None))
    if not chosen:
        return False, None, getattr(res, 'reasoning', '')
    pred = _canonical_action_dict(chosen, state=ex.state)
    for ca in ex.correct_actions:
        if _canonical_action_dict(ca, state=ex.state) == pred:
            return True, chosen, getattr(res, 'reasoning', '')
    return False, chosen, getattr(res, 'reasoning', '')


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', default='gpt-5.2')
    p.add_argument('--parallel', type=int, default=10)
    p.add_argument('--dump', default='dspy_ml/data/plain_mismatches.json')
    args = p.parse_args()

    lm = dspy.LM(model=args.model)
    dspy.configure(lm=lm)

    dataset = DrillDataset()
    dataset.load_from_json(args.dataset)
    module = build_plain_module()

    mismatches = []
    correct = 0
    total = len(dataset.examples)

    def worker(idx_ex):
        idx, ex = idx_ex
        ok, chosen, reasoning = eval_example(module, ex)
        return idx, ok, chosen, reasoning

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = [pool.submit(worker, (i, ex)) for i, ex in enumerate(dataset.examples)]
        for f in as_completed(futs):
            idx, ok, chosen, reasoning = f.result()
            ex = dataset.examples[idx]
            if ok:
                correct += 1
            else:
                mismatches.append({
                    'idx': idx,
                    'drill_id': ex.drill_id,
                    'expected': ex.correct_actions,
                    'predicted': chosen,
                    'expected_action': ex.expected_action,
                    'viable_actions_excerpt': ex.viable_actions[:400],
                    'reasoning': reasoning,
                })

    print(f"Correct: {correct}/{total} ({100*correct/total:.2f}%)")
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dump, 'w') as f:
        json.dump(mismatches, f, indent=2)
    print(f"Saved mismatches to {args.dump} (n={len(mismatches)})")

if __name__ == '__main__':
    sys.exit(main())
