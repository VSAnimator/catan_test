"""
Dataset conversion and loading for DSPy drill optimization.

Converts drills from database to standalone dataset format.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine import (
    GameState,
    Action,
    ActionPayload,
    deserialize_game_state,
    serialize_action,
    serialize_action_payload,
)
from engine.serialization import (
    state_to_text,
    legal_actions_to_text,
    legal_actions,
)
from agents.llm_agent import LLMAgent
from api.database import (
    list_drills,
    get_drill_steps,
    get_drill,
)
from api.routes import _filter_legal_actions, _canonical_action_dict


@dataclass
class DrillExample:
    """A single drill step example for optimization."""
    drill_id: int
    game_rules: str
    observation: str
    viable_actions: str
    expected_action: Dict[str, Any]
    correct_actions: List[Dict[str, Any]]
    incorrect_actions: Optional[List[Dict[str, Any]]] = None
    state: Optional[GameState] = None  # Game state for phase-aware comparison


class DrillDataset:
    """Dataset for DSPy drill optimization."""
    
    def __init__(self):
        self.examples: List[DrillExample] = []
        self._game_rules_cache: Optional[str] = None
    
    def _get_game_rules(self) -> str:
        """Get game rules without strategic advice."""
        if self._game_rules_cache is None:
            # Create temporary agent to get default system prompt
            temp_agent = LLMAgent("player_0", exclude_strategic_advice=True)
            self._game_rules_cache = temp_agent._get_default_system_prompt()
        return self._game_rules_cache
    
    def load_from_database(
        self,
        drill_ids: Optional[List[int]] = None,
        limit: int = 200
    ) -> List[DrillExample]:
        """
        Load drills from database and convert to dataset format.
        
        Only exports the first step (idx=0) of each drill.
        
        Args:
            drill_ids: Specific drill IDs to load (if None, loads all up to limit)
            limit: Maximum number of drills to load
            
        Returns:
            List of DrillExample objects
        """
        examples = []
        
        # Get game rules once
        game_rules = self._get_game_rules()
        
        # Load drills
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
                                "correct_actions": json.loads(r["correct_actions_json"]) if ("correct_actions_json" in r.keys() and r["correct_actions_json"]) else None,
                                "incorrect_actions": json.loads(r["incorrect_actions_json"]) if ("incorrect_actions_json" in r.keys() and r["incorrect_actions_json"]) else None,
                            }
                            for r in steps
                        ]
                    })
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
                                "correct_actions": json.loads(r["correct_actions_json"]) if ("correct_actions_json" in r.keys() and r["correct_actions_json"]) else None,
                                "incorrect_actions": json.loads(r["incorrect_actions_json"]) if ("incorrect_actions_json" in r.keys() and r["incorrect_actions_json"]) else None,
                        }
                        for r in steps
                    ]
                })
        
        # Convert each drill's first step to example
        for drill in drills:
            drill_id = drill["id"]
            steps = drill.get("steps", [])
            
            if not steps:
                continue
            
            # Only use first step (idx=0)
            first_step = None
            for step in steps:
                if step["idx"] == 0:
                    first_step = step
                    break
            
            if not first_step:
                continue
            
            # Load state
            state_json = first_step["state"]
            try:
                state = deserialize_game_state(state_json)
            except Exception as e:
                print(f"Warning: Failed to deserialize state for drill {drill_id}: {e}", flush=True)
                continue
            
            player_id = first_step["player_id"]
            
            # Get legal actions
            try:
                la_list = legal_actions(state, player_id)
            except Exception as e:
                print(f"Warning: Failed to get legal actions for drill {drill_id}: {e}", flush=True)
                continue
            
            if not la_list:
                continue
            
            # Filter actions if correct/incorrect specified
            correct_actions = first_step.get("correct_actions")
            if not correct_actions:
                # Fall back to expected_action
                expected_action = first_step.get("expected_action")
                if expected_action:
                    correct_actions = [expected_action]
            
            if not correct_actions:
                continue  # Skip steps without correct actions
            
            incorrect_actions = first_step.get("incorrect_actions")
            
            # Filter legal actions if correct actions are specified
            # This matches the API endpoint behavior in routes.py
            if correct_actions:
                # Auto-populate incorrect_actions if empty (matching API behavior)
                if not incorrect_actions:
                    # Convert all legal actions to action dicts
                    all_legal_action_dicts = []
                    for legal_action, legal_payload in la_list:
                        action_dict = {"type": serialize_action(legal_action)}
                        if legal_payload is not None:
                            action_dict["payload"] = serialize_action_payload(legal_payload)
                        all_legal_action_dicts.append(action_dict)
                    
                    # Filter out correct actions to get incorrect actions
                    def dict_to_hashable(obj):
                        """Recursively convert dict to hashable tuple."""
                        if isinstance(obj, dict):
                            return tuple(sorted((k, dict_to_hashable(v)) for k, v in obj.items()))
                        elif isinstance(obj, list):
                            return tuple(dict_to_hashable(item) for item in obj)
                        else:
                            return obj
                    
                    def canonical_to_tuple(ca):
                        canonical = _canonical_action_dict(ca)
                        payload = canonical.get("payload")
                        if payload:
                            payload_tuple = dict_to_hashable(payload)
                        else:
                            payload_tuple = None
                        return (canonical.get("type"), payload_tuple)
                    
                    correct_action_set = {canonical_to_tuple(ca) for ca in correct_actions}
                    incorrect_actions = [
                        action_dict for action_dict in all_legal_action_dicts
                        if canonical_to_tuple(action_dict) not in correct_action_set
                    ]
                
                # Filter to only include correct + incorrect actions
                action_dicts_to_include = correct_actions.copy()
                if incorrect_actions:
                    action_dicts_to_include.extend(incorrect_actions)
                la_list = _filter_legal_actions(la_list, action_dicts_to_include)
                
                if not la_list:
                    print(f"Warning: Filter rejected all legal actions for drill {drill_id}, skipping", flush=True)
                    continue
            
            if not la_list:
                continue
            
            # Generate observation (with higher-level features)
            observation = state_to_text(
                state,
                player_id,
                exclude_higher_level_features=False
            )
            
            # Format viable actions
            viable_actions = legal_actions_to_text(la_list, state=state, player_id=player_id)
            
            # Get expected action
            expected_action = correct_actions[0] if correct_actions else first_step.get("expected_action")
            
            examples.append(DrillExample(
                drill_id=drill_id,
                game_rules=game_rules,
                observation=observation,
                viable_actions=viable_actions,
                expected_action=expected_action,
                correct_actions=correct_actions,
                incorrect_actions=incorrect_actions,
                state=state  # Include state for phase-aware comparison
            ))
        
        self.examples = examples
        return examples
    
    def export_to_json(self, filepath: str) -> None:
        """Export dataset to JSON file."""
        data = []
        for ex in self.examples:
            data.append({
                "drill_id": ex.drill_id,
                "game_rules": ex.game_rules,
                "observation": ex.observation,
                "viable_actions": ex.viable_actions,
                "expected_action": ex.expected_action,
                "correct_actions": ex.correct_actions,
                "incorrect_actions": ex.incorrect_actions,
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_json(self, filepath: str) -> List[DrillExample]:
        """Load dataset from JSON file. Also loads game states from database for phase-aware comparison."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            drill_id = item["drill_id"]
            
            # Load state from database for phase-aware comparison
            state = None
            try:
                drill = get_drill(drill_id)
                if drill:
                    steps = get_drill_steps(drill_id)
                    if steps:
                        first_step = steps[0]
                        state_json = json.loads(first_step["state_json"])
                        state = deserialize_game_state(state_json)
            except Exception as e:
                print(f"Warning: Could not load state for drill {drill_id}: {e}", flush=True)
            
            examples.append(DrillExample(
                drill_id=drill_id,
                game_rules=item["game_rules"],
                observation=item["observation"],
                viable_actions=item["viable_actions"],
                expected_action=item["expected_action"],
                correct_actions=item["correct_actions"],
                incorrect_actions=item.get("incorrect_actions"),
                state=state,
            ))
        
        self.examples = examples
        return examples
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[DrillExample], List[DrillExample], List[DrillExample]]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            
        Returns:
            (train_examples, val_examples, test_examples)
        """
        import random
        random.seed(42)  # For reproducibility
        
        shuffled = self.examples.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]
        
        return train, val, test

