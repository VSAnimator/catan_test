#!/usr/bin/env python3
"""
Test hybrid guideline agent on all drills to confirm minimal performance drop.
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dspy_ml.dataset import DrillDataset
from engine.serialization import state_to_text, legal_actions_to_text, legal_actions
from api.routes import _canonical_action_dict
from agents.hybrid_guideline_agent import HybridGuidelineAgent


def evaluate_drill_hybrid(
    drill_id: int,
    ex: Any,
    game_rules: str
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate a single drill using hybrid guidelines."""
    if not ex.state:
        return False, {"drill_id": drill_id, "error": "No state"}
    
    # Get player_id from state
    player_id = ex.state.players[ex.state.current_player_index].id if ex.state.current_player_index is not None else ex.state.players[ex.state.setup_phase_player_index].id if ex.state.setup_phase_player_index is not None else None
    if not player_id:
        return False, {"drill_id": drill_id, "error": "No player_id"}
    
    # Generate legal actions and apply filtering (same as DrillDataset)
    try:
        legal_actions_list = legal_actions(ex.state, player_id)
    except Exception as e:
        return False, {"drill_id": drill_id, "error": f"Failed to get legal actions: {e}"}
    
    if not legal_actions_list:
        return False, {"drill_id": drill_id, "error": "No legal actions"}
    
    # Apply same filtering as DrillDataset (restrict to correct + incorrect actions)
    from api.routes import _filter_legal_actions
    if ex.correct_actions:
        # Build action_dicts_to_include (correct + incorrect)
        action_dicts_to_include = ex.correct_actions.copy()
        if ex.incorrect_actions:
            action_dicts_to_include.extend(ex.incorrect_actions)
        
        # Filter legal actions to match the restricted space
        legal_actions_list = _filter_legal_actions(legal_actions_list, action_dicts_to_include)
        
        if not legal_actions_list:
            return False, {"drill_id": drill_id, "error": "Filter rejected all legal actions"}
    
    # Create agent
    try:
        agent = HybridGuidelineAgent(
            player_id=player_id,
            exclude_strategic_advice=True,
            exclude_higher_level_features=False,
        )
    except Exception as e:
        return False, {"drill_id": drill_id, "error": f"Failed to create agent: {e}"}
    
    # Choose action (pass drill_id for direct guideline lookup)
    try:
        action, payload, reasoning = agent.choose_action(ex.state, legal_actions_list, drill_id=drill_id)
    except Exception as e:
        return False, {"drill_id": drill_id, "error": f"Failed to choose action: {e}"}
    
    # Convert to action dict
    from engine.serialization import serialize_action, serialize_action_payload
    predicted = {"type": serialize_action(action)}
    if payload is not None:
        predicted["payload"] = serialize_action_payload(payload)
    
    # Compare with expected
    expected_actions = ex.correct_actions if ex.correct_actions else [ex.expected_action]
    
    matched = False
    pred_canon = _canonical_action_dict(predicted, state=ex.state)
    
    def normalize_setup_build(canon):
        if not isinstance(canon, dict):
            return canon
        t = canon.get("type")
        if t == "build_road":
            canon = canon.copy()
            canon["type"] = "setup_place_road"
        elif t == "build_settlement":
            canon = canon.copy()
            canon["type"] = "setup_place_settlement"
        return canon
    
    for ca in expected_actions:
        ca_canon = _canonical_action_dict(ca, state=ex.state)
        if normalize_setup_build(ca_canon) == normalize_setup_build(pred_canon):
            matched = True
            break
    
    return matched, {
        "drill_id": drill_id,
        "expected": expected_actions[0] if expected_actions else {},
        "predicted": predicted,
        "matched": matched,
        "reasoning": reasoning,
    }


def main():
    # Load drills from database
    print("Loading drills from database...", flush=True)
    dataset = DrillDataset()
    dataset.load_from_database()
    
    if not dataset.examples:
        print("No drills loaded from database!", flush=True)
        return
    
    print(f"Loaded {len(dataset.examples)} drills from database", flush=True)
    
    # Get game rules
    game_rules = dataset.examples[0].game_rules if dataset.examples else ""
    
    # Evaluate in parallel
    print(f"Evaluating {len(dataset.examples)} drills with hybrid guidelines...", flush=True)
    
    results = []
    with ProcessPoolExecutor(max_workers=min(8, len(dataset.examples))) as executor:
        futures = {
            executor.submit(evaluate_drill_hybrid, ex.drill_id, ex, game_rules): ex
            for ex in dataset.examples
        }
        
        for future in as_completed(futures):
            try:
                matched, details = future.result()
                results.append((matched, details))
            except Exception as e:
                ex = futures[future]
                print(f"Error evaluating drill {ex.drill_id}: {e}", flush=True)
                results.append((False, {"drill_id": ex.drill_id, "error": str(e)}))
    
    # Calculate results
    correct = sum(1 for matched, _ in results if matched)
    total = len(results)
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"  Hybrid guidelines: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*80}")
    
    # Save detailed results
    output_path = Path(__file__).parent.parent / "data" / "hybrid_guidelines_evaluation.json"
    with open(output_path, "w") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "results": [{"matched": m, **d} for m, d in results],
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()

