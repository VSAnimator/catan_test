#!/usr/bin/env python3
"""
Debug why certain drills are missing from the export.
"""
import sys
import json
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine import deserialize_game_state
from engine.serialization import legal_actions
from api.database import get_drill, get_drill_steps
from api.routes import _filter_legal_actions

def debug_drill(drill_id):
    """Debug a specific drill to see why it's being skipped."""
    print(f"\n{'='*60}", flush=True)
    print(f"Debugging Drill {drill_id}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Get drill
    drill_row = get_drill(drill_id)
    if not drill_row:
        print(f"Drill {drill_id} not found!", flush=True)
        return
    
    # Get steps
    steps = get_drill_steps(drill_id)
    if not steps:
        print(f"No steps found!", flush=True)
        return
    
    # Get first step
    first_step = None
    for step in steps:
        if step["idx"] == 0:
            first_step = step
            break
    
    if not first_step:
        print(f"No step with idx=0 found!", flush=True)
        return
    
    print(f"Found first step", flush=True)
    
    # Parse correct/incorrect actions
    correct_actions_json = first_step["correct_actions_json"] if "correct_actions_json" in first_step.keys() else None
    incorrect_actions_json = first_step["incorrect_actions_json"] if "incorrect_actions_json" in first_step.keys() else None
    expected_action_json = first_step["expected_action_json"] if "expected_action_json" in first_step.keys() else None
    
    print(f"\ncorrect_actions_json: {correct_actions_json}", flush=True)
    print(f"incorrect_actions_json: {incorrect_actions_json}", flush=True)
    print(f"expected_action_json: {expected_action_json}", flush=True)
    
    correct_actions = json.loads(correct_actions_json) if correct_actions_json else None
    incorrect_actions = json.loads(incorrect_actions_json) if incorrect_actions_json else None
    expected_action = json.loads(expected_action_json) if expected_action_json else None
    
    if not correct_actions:
        if expected_action:
            correct_actions = [expected_action]
    
    if not correct_actions:
        print(f"\n❌ SKIPPED: No correct actions!", flush=True)
        return
    
    print(f"\ncorrect_actions: {correct_actions}", flush=True)
    print(f"incorrect_actions: {incorrect_actions}", flush=True)
    
    # Load state
    state_json = json.loads(first_step["state_json"])
    try:
        state = deserialize_game_state(state_json)
    except Exception as e:
        print(f"\n❌ SKIPPED: Failed to deserialize state: {e}", flush=True)
        return
    
    player_id = first_step["player_id"]
    
    # Get legal actions
    try:
        la_list = legal_actions(state, player_id)
    except Exception as e:
        print(f"\n❌ SKIPPED: Failed to get legal actions: {e}", flush=True)
        return
    
    if not la_list:
        print(f"\n❌ SKIPPED: No legal actions!", flush=True)
        return
    
    print(f"\nLegal actions before filtering: {len(la_list)}", flush=True)
    for i, (action, payload) in enumerate(la_list[:5]):
        print(f"  {i}: {action.value} - {payload}", flush=True)
    if len(la_list) > 5:
        print(f"  ... ({len(la_list) - 5} more)", flush=True)
    
    # Filter actions
    if correct_actions or incorrect_actions:
        action_dicts_to_include = correct_actions.copy()
        if incorrect_actions:
            action_dicts_to_include.extend(incorrect_actions)
        print(f"\nFiltering to include {len(action_dicts_to_include)} action dicts", flush=True)
        la_list_filtered = _filter_legal_actions(la_list, action_dicts_to_include)
    else:
        la_list_filtered = la_list
    
    print(f"\nLegal actions after filtering: {len(la_list_filtered)}", flush=True)
    
    if not la_list_filtered:
        print(f"\n❌ SKIPPED: No legal actions after filtering!", flush=True)
        print(f"\nThis is the issue! The filter rejected all legal actions.", flush=True)
    else:
        print(f"\n✓ Would be included in dataset", flush=True)


def main():
    missing_drill_ids = [79, 123, 131, 132]
    
    for drill_id in missing_drill_ids:
        debug_drill(drill_id)


if __name__ == "__main__":
    main()

