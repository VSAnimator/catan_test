#!/usr/bin/env python3
"""
Test LLM agent discard functionality.

This test verifies that LLM agents can properly discard resources after rolling 7.
It tests the full flow: agent choosing action, generating payload, and executing discard.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import init_db, get_latest_state
from engine.serialization import deserialize_game_state, legal_actions
from agents.llm_agent import LLMAgent


def test_llm_agent_discard(game_id: str, player_id: str = "player_3") -> dict:
    """
    Test that LLM agent can properly discard resources.
    
    Args:
        game_id: Game ID to test (should be at state where player needs to discard)
        player_id: Player ID to test
        
    Returns:
        Dictionary with test results:
        - success: bool
        - error: Optional[str]
        - messages: List[str]
    """
    result = {
        'success': False,
        'error': None,
        'messages': []
    }
    
    try:
        init_db()
        
        # Get the game state
        state_json = get_latest_state(game_id)
        if not state_json:
            result['error'] = f"Game {game_id} not found"
            return result
        
        state = deserialize_game_state(state_json)
        result['messages'].append(f"Loaded game state: {game_id}")
        
        # Find the player
        player = next((p for p in state.players if p.id == player_id), None)
        if not player:
            result['error'] = f"Player {player_id} not found"
            return result
        
        total_resources = sum(player.resources.values())
        result['messages'].append(f"Player {player.name} ({player_id}) has {total_resources} resources")
        result['messages'].append(f"Dice roll: {state.dice_roll}")
        
        # Get legal actions
        legal_actions_list = legal_actions(state, player_id)
        result['messages'].append(f"Legal actions: {[a.value for a, _ in legal_actions_list]}")
        
        # Check if discard is available
        discard_actions = [(a, p) for a, p in legal_actions_list if a.value == "discard_resources"]
        if not discard_actions:
            result['error'] = "No discard actions available - test setup incorrect"
            return result
        
        result['messages'].append(f"Found {len(discard_actions)} discard action(s)")
        
        # Create LLM agent
        result['messages'].append("Creating LLM agent...")
        agent = LLMAgent(
            player_id=player_id,
            model="gpt-4o-mini",
            enable_retrieval=False  # Disable retrieval for faster testing
        )
        
        # Have agent choose action
        result['messages'].append("Agent choosing action...")
        result_obj = agent.choose_action(state, legal_actions_list)
        
        if len(result_obj) == 4:
            action, payload, reasoning, raw_response = result_obj
        elif len(result_obj) == 3:
            action, payload, reasoning = result_obj
            raw_response = None
        else:
            action, payload = result_obj
            reasoning = None
            raw_response = None
        
        result['messages'].append(f"Action chosen: {action.value}")
        
        if not payload:
            result['error'] = "Agent did not generate payload for discard action"
            return result
        
        result['messages'].append(f"Payload type: {type(payload).__name__}")
        
        if not hasattr(payload, 'resources'):
            result['error'] = "Payload does not have resources attribute"
            return result
        
        resources_to_discard = dict(payload.resources)
        total_discard = sum(resources_to_discard.values())
        expected_discard = total_resources // 2
        
        result['messages'].append(f"Resources to discard: {resources_to_discard}")
        result['messages'].append(f"Total to discard: {total_discard} (expected: {expected_discard})")
        
        if total_discard != expected_discard:
            result['error'] = f"Wrong discard amount: got {total_discard}, expected {expected_discard}"
            return result
        
        # Try to execute the action
        result['messages'].append("Executing action...")
        new_state = state.step(action, payload, player_id=player_id)
        result['messages'].append("Action executed successfully")
        
        # Check results
        new_player = next((p for p in new_state.players if p.id == player_id), None)
        if not new_player:
            result['error'] = "Player not found in new state"
            return result
        
        new_total = sum(new_player.resources.values())
        old_total = total_resources
        discarded = old_total - new_total
        
        result['messages'].append(f"Resources before: {old_total}")
        result['messages'].append(f"Resources after: {new_total}")
        result['messages'].append(f"Discarded: {discarded}")
        
        if discarded != expected_discard:
            result['error'] = f"Wrong amount discarded: got {discarded}, expected {expected_discard}"
            return result
        
        if player_id not in new_state.players_discarded:
            result['error'] = "Player not marked as having discarded"
            return result
        
        result['messages'].append("✓ Player marked as having discarded")
        result['messages'].append("✓ Correct amount discarded")
        result['success'] = True
        
    except Exception as e:
        result['error'] = f"Test execution failed: {str(e)}"
        import traceback
        result['messages'].append(f"Traceback: {traceback.format_exc()}")
    
    return result


if __name__ == "__main__":
    # Test with the verification game
    test_game_id = "0ab49856-0674-4fde-a94d-4ca17aa44f4c"
    
    print("Testing LLM agent discard functionality")
    print("=" * 60)
    
    result = test_llm_agent_discard(test_game_id, "player_3")
    
    print(f"\nTest Result: {'PASS' if result['success'] else 'FAIL'}")
    if result['error']:
        print(f"Error: {result['error']}")
    print("\nMessages:")
    for msg in result['messages']:
        print(f"  {msg}")
    
    sys.exit(0 if result['success'] else 1)

