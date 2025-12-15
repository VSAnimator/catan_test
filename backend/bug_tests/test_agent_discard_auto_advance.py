#!/usr/bin/env python3
"""
Test that agents automatically discard when a 7 is rolled, even if it's not their turn.

This test verifies the fix for the bug where games would get stuck when an agent needed
to discard after a 7 was rolled, but the current player was a human player.

Bug scenario:
- Game has mixed players (humans + LLM agents)
- A human player rolls a 7
- An LLM agent has 8+ resources and needs to discard
- The game gets stuck because watch_agents_step only checks if current player is an agent
- Fix: watch_agents_step now checks for agents needing discard BEFORE checking current player
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from api.database import init_db, get_latest_state
from engine.serialization import deserialize_game_state


API_BASE = "http://localhost:8000/api"


def test_agent_discard_auto_advance(game_id: str, agent_mapping: dict) -> dict:
    """
    Test that watch_agents_step properly handles agent discards when current player is human.
    
    Args:
        game_id: Game ID to test (should be at state where agent needs to discard after 7)
        agent_mapping: Dictionary mapping player_id to agent_type
        
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
        
        # Get initial game state
        response = requests.get(f"{API_BASE}/games/{game_id}")
        if response.status_code != 200:
            result['error'] = f"Failed to get game state: {response.status_code}"
            return result
        
        initial_state = response.json()
        result['messages'].append(f"Loaded game state: {game_id}")
        
        # Verify we're in the discard phase (or check if already processed)
        dice_roll = initial_state.get('dice_roll')
        if dice_roll != 7:
            # Game might have already been processed - check if agents already discarded
            all_agents_discarded = True
            for p in initial_state['players']:
                total_resources = sum(p['resources'].values())
                has_discarded = p['id'] in (initial_state.get('players_discarded') or [])
                if total_resources >= 8 and not has_discarded and p['id'] in agent_mapping:
                    all_agents_discarded = False
                    break
            
            if all_agents_discarded:
                result['messages'].append(f"Game already processed (dice_roll={dice_roll}), but all agents have discarded - this is OK")
                result['success'] = True
                return result
            else:
                result['error'] = f"Game is not in discard phase (dice_roll={dice_roll}) and agents still need to discard"
                return result
        
        if initial_state.get('waiting_for_robber_move') or initial_state.get('waiting_for_robber_steal'):
            result['error'] = "Game is already in robber phase, not discard phase"
            return result
        
        # Find agents who need to discard
        agents_needing_discard = []
        for p in initial_state['players']:
            total_resources = sum(p['resources'].values())
            has_discarded = p['id'] in (initial_state.get('players_discarded') or [])
            if total_resources >= 8 and not has_discarded and p['id'] in agent_mapping:
                agents_needing_discard.append({
                    'id': p['id'],
                    'name': p['name'],
                    'resources': total_resources
                })
        
        if not agents_needing_discard:
            result['error'] = "No agents need to discard - test setup incorrect"
            return result
        
        result['messages'].append(f"Found {len(agents_needing_discard)} agent(s) needing discard:")
        for agent in agents_needing_discard:
            result['messages'].append(f"  - {agent['name']} ({agent['id']}): {agent['resources']} resources")
        
        # Check current player
        current_player_idx = initial_state.get('current_player_index')
        current_player = initial_state['players'][current_player_idx] if current_player_idx is not None else None
        if current_player:
            result['messages'].append(f"Current player: {current_player['name']} ({current_player['id']})")
            if current_player['id'] in agent_mapping:
                result['messages'].append("  → Current player is an agent (normal case)")
            else:
                result['messages'].append("  → Current player is human (this is the bug scenario!)")
        
        # Call watch_agents_step - this should process the discard even if current player is human
        result['messages'].append("Calling watch_agents_step...")
        response = requests.post(
            f"{API_BASE}/games/{game_id}/watch_agents_step",
            json={'agent_mapping': agent_mapping}
        )
        
        if response.status_code != 200:
            result['error'] = f"watch_agents_step failed: {response.status_code} - {response.text}"
            return result
        
        step_result = response.json()
        result['messages'].append(f"watch_agents_step returned: player_id={step_result.get('player_id')}, error={step_result.get('error')}")
        
        # Get updated game state
        response = requests.get(f"{API_BASE}/games/{game_id}")
        if response.status_code != 200:
            result['error'] = f"Failed to get updated game state: {response.status_code}"
            return result
        
        updated_state = response.json()
        
        # Verify that at least one agent discarded
        agents_still_needing_discard = []
        agents_who_discarded = []
        
        for agent_info in agents_needing_discard:
            agent_id = agent_info['id']
            # Find the player in updated state
            updated_player = next((p for p in updated_state['players'] if p['id'] == agent_id), None)
            if not updated_player:
                result['error'] = f"Agent {agent_id} not found in updated state"
                return result
            
            total_resources = sum(updated_player['resources'].values())
            has_discarded = agent_id in (updated_state.get('players_discarded') or [])
            
            if has_discarded:
                agents_who_discarded.append({
                    'id': agent_id,
                    'name': agent_info['name'],
                    'resources_before': agent_info['resources'],
                    'resources_after': total_resources
                })
            elif total_resources >= 8:
                agents_still_needing_discard.append(agent_id)
        
        if agents_still_needing_discard:
            result['error'] = f"Agents still need to discard: {agents_still_needing_discard}"
            return result
        
        if not agents_who_discarded:
            result['error'] = "No agents discarded - watch_agents_step did not process discard"
            return result
        
        # Verify discard amounts are correct
        for agent_info in agents_who_discarded:
            expected_discard = agent_info['resources_before'] // 2
            actual_discard = agent_info['resources_before'] - agent_info['resources_after']
            
            result['messages'].append(f"{agent_info['name']}: {agent_info['resources_before']} → {agent_info['resources_after']} (discarded {actual_discard})")
            
            if actual_discard != expected_discard:
                result['error'] = f"{agent_info['name']} discarded {actual_discard}, expected {expected_discard}"
                return result
        
        result['messages'].append("✓ All agents discarded correctly")
        result['messages'].append("✓ Discard amounts are correct")
        result['success'] = True
        
    except Exception as e:
        result['error'] = f"Test execution failed: {str(e)}"
        import traceback
        result['messages'].append(f"Traceback: {traceback.format_exc()}")
    
    return result


if __name__ == "__main__":
    # Test with the stuck game
    test_game_id = "f226af0d-26c5-4711-a12c-0fa22c8ada32"
    test_agent_mapping = {'player_2': 'llm', 'player_3': 'llm'}
    
    print("Testing agent discard auto-advance (bug fix)")
    print("=" * 60)
    
    result = test_agent_discard_auto_advance(test_game_id, test_agent_mapping)
    
    print(f"\nTest Result: {'PASS' if result['success'] else 'FAIL'}")
    if result['error']:
        print(f"Error: {result['error']}")
    print("\nMessages:")
    for msg in result['messages']:
        print(f"  {msg}")
    
    sys.exit(0 if result['success'] else 1)

