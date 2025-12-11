"""
Test runner for bug regression tests.

Restores game states and executes tests with reproducible RNG.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import random
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import get_db_connection, get_steps, get_state_at_step
from engine.serialization import (
    deserialize_game_state, 
    serialize_game_state,
    deserialize_action,
    deserialize_action_payload
)
from engine import GameState, Action
from bug_tests.test_registry import BugTestCase, TestRegistry


def compute_step_seed(game_id: str, step_id: int) -> int:
    """Compute a deterministic seed from game_id and step_id.
    
    This ensures that when we restore a game at a specific step,
    the RNG will produce the same sequence as it did in the original game
    from that point forward.
    
    Args:
        game_id: The game ID
        step_id: The step index to restore at
        
    Returns:
        An integer seed value
    """
    # Create a hash from game_id and step_id
    seed_string = f"{game_id}:{step_id}"
    seed_hash = hashlib.md5(seed_string.encode()).hexdigest()
    # Convert to integer (use first 8 hex digits to avoid overflow)
    return int(seed_hash[:8], 16)


def restore_state_at_step(game_id: str, step_id: int) -> Tuple[GameState, int]:
    """Restore game state at a specific step.
    
    Args:
        game_id: The game ID
        step_id: The step index to restore at (0-based)
        
    Returns:
        Tuple of (GameState, seed) where seed is the step-based seed
    """
    # Get state at the specified step
    state_json = get_state_at_step(game_id, step_id, use_state_before=True)
    
    if state_json is None:
        # Fallback: try to get from steps list
        steps = get_steps(game_id)
        if step_id < 0 or step_id >= len(steps):
            raise ValueError(f"Step {step_id} not found. Game has {len(steps)} steps.")
        step_row = steps[step_id]
        state_json = json.loads(step_row["state_before_json"])
    
    # Deserialize the state
    state = deserialize_game_state(state_json)
    
    # Compute step-based seed for reproducible RNG
    seed = compute_step_seed(game_id, step_id)
    
    # Set the seed
    random.seed(seed)
    
    return state, seed


def run_test(test: BugTestCase) -> Dict[str, Any]:
    """Run a single bug test case.
    
    Args:
        test: The test case to run
        
    Returns:
        Dictionary with test results:
        - success: bool
        - error: Optional[str]
        - state_after: Optional[Dict] - final state after test
        - messages: List[str] - informational messages
    """
    result = {
        'success': False,
        'error': None,
        'state_after': None,
        'messages': []
    }
    
    try:
        # Restore state at the specified step
        result['messages'].append(f"Restoring state from game {test.game_id} at step {test.step_id}")
        state, seed = restore_state_at_step(test.game_id, test.step_id)
        result['messages'].append(f"State restored with seed {seed}")
        
        # If test has an action, execute it
        if test.test_action:
            action_type = test.test_action.get('type')
            payload_data = test.test_action.get('payload')
            
            # Convert action string to Action enum
            try:
                action = deserialize_action(action_type)
            except ValueError:
                result['error'] = f"Invalid action type: {action_type}"
                return result
            
            # Deserialize payload if present
            payload = None
            if payload_data is not None:
                try:
                    # Payload dict should have a "type" field for deserialization
                    # If it doesn't, we need to infer it from the action type
                    if isinstance(payload_data, dict):
                        # Check if payload already has type field
                        if "type" not in payload_data:
                            # Infer payload type from action
                            payload_type_map = {
                                "build_road": "BuildRoadPayload",
                                "build_settlement": "BuildSettlementPayload",
                                "build_city": "BuildCityPayload",
                                "play_dev_card": "PlayDevCardPayload",
                                "trade_bank": "TradeBankPayload",
                                "trade_player": "TradePlayerPayload",
                                "propose_trade": "ProposeTradePayload",
                                "select_trade_partner": "SelectTradePartnerPayload",
                                "move_robber": "MoveRobberPayload",
                                "steal_resource": "StealResourcePayload",
                                "discard_resources": "DiscardResourcesPayload",
                            }
                            payload_type = payload_type_map.get(action_type)
                            if payload_type:
                                payload_data = payload_data.copy()
                                payload_data["type"] = payload_type
                        
                        payload = deserialize_action_payload(payload_data)
                except Exception as e:
                    result['error'] = f"Error deserializing payload: {str(e)}"
                    return result
            
            # Execute the action
            result['messages'].append(f"Executing action: {action_type}")
            try:
                new_state = state.step(action, payload)
                result['state_after'] = serialize_game_state(new_state)
                result['messages'].append("Action executed successfully")
            except Exception as e:
                result['error'] = f"Error executing action: {str(e)}"
                return result
        else:
            # No action to execute, just serialize current state
            result['state_after'] = serialize_game_state(state)
            result['messages'].append("No action specified, returning current state")
        
        # Test completed successfully
        result['success'] = True
        
    except Exception as e:
        result['error'] = f"Test execution failed: {str(e)}"
        import traceback
        result['messages'].append(f"Traceback: {traceback.format_exc()}")
    
    return result


def run_all_tests(registry: Optional[TestRegistry] = None) -> Dict[str, Any]:
    """Run all tests in the registry.
    
    Args:
        registry: TestRegistry instance. If None, creates a new one.
        
    Returns:
        Dictionary with test results summary
    """
    if registry is None:
        registry = TestRegistry()
    
    tests = registry.list_tests()
    results = {
        'total': len(tests),
        'passed': 0,
        'failed': 0,
        'test_results': {}
    }
    
    for test in tests:
        result = run_test(test)
        results['test_results'][test.test_id] = result
        
        if result['success']:
            results['passed'] += 1
        else:
            results['failed'] += 1
    
    return results


if __name__ == "__main__":
    # Example usage
    registry = TestRegistry()
    results = run_all_tests(registry)
    
    print(f"Test Results:")
    print(f"  Total: {results['total']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    
    for test_id, result in results['test_results'].items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"\n{test_id}: {status}")
        if result['error']:
            print(f"  Error: {result['error']}")
        for msg in result['messages']:
            print(f"  {msg}")

