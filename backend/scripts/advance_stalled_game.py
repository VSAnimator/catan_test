#!/usr/bin/env python3
"""
Manually advance a stalled game by having an agent take the next action.
Useful for debugging stalled games.
"""
import sys
import os
from pathlib import Path

# Load environment variables
if 'OPENAI_API_KEY' not in os.environ:
    import subprocess
    try:
        result = subprocess.run(['bash', '-c', 'source ~/.zshrc 2>/dev/null && echo $OPENAI_API_KEY'], 
                              capture_output=True, text=True, shell=False)
        if result.stdout.strip():
            os.environ['OPENAI_API_KEY'] = result.stdout.strip()
    except:
        pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import get_latest_state, save_game_state, add_step, get_step_count
from engine.serialization import deserialize_game_state, legal_actions, serialize_game_state, serialize_action_payload
from agents.llm_agent import LLMAgent
from engine import Action
import json


def advance_game(game_id: str, player_id: str = None):
    """Advance a stalled game by having an agent take an action."""
    # Get current state
    state_data = get_latest_state(game_id)
    if not state_data:
        print(f"Game {game_id} not found")
        return
    
    state = deserialize_game_state(state_data)
    
    # Determine current player
    if state.phase == "setup":
        current_player = state.players[state.setup_phase_player_index]
    elif state.phase == "playing":
        current_player = state.players[state.current_player_index]
    else:
        print(f"Game is in phase: {state.phase}")
        return
    
    # If player_id specified, use that player
    if player_id:
        current_player = next((p for p in state.players if p.id == player_id), None)
        if not current_player:
            print(f"Player {player_id} not found")
            return
    
    print(f"Current player: {current_player.name} ({current_player.id})")
    
    # Get legal actions
    legal_actions_list = legal_actions(state, current_player.id)
    print(f"Legal actions: {[a.value for a, _ in legal_actions_list]}")
    
    if not legal_actions_list:
        print("No legal actions available")
        return
    
    # Create agent for this player
    agent = LLMAgent(current_player.id, model="gpt-5.1", enable_retrieval=False)
    
    # Agent chooses action
    print("\nAgent choosing action...")
    result = agent.choose_action(state, legal_actions_list)
    if len(result) == 4:
        action, payload, reasoning, raw_llm_response = result
    elif len(result) == 3:
        action, payload, reasoning = result
        raw_llm_response = None
    else:
        action, payload = result
        reasoning = None
        raw_llm_response = None
    
    print(f"Action chosen: {action.value}")
    if reasoning:
        print(f"Reasoning: {reasoning[:200]}...")
    
    # Execute action
    new_state = state.step(action, payload, current_player.id)
    
    # Save state
    state_after_json = serialize_game_state(new_state)
    save_game_state(game_id, state_after_json)
    
    # Save step
    action_json = {"type": action.value}
    if payload:
        action_json["payload"] = serialize_action_payload(payload)
    if reasoning:
        action_json["reasoning"] = reasoning
    
    step_idx = get_step_count(game_id)
    add_step(
        game_id=game_id,
        step_idx=step_idx,
        player_id=current_player.id,
        state_before_json=state_data,
        state_after_json=state_after_json,
        action_json=action_json,
        reasoning=reasoning,
        batch_write=False
    )
    
    print(f"\nGame advanced! New phase: {new_state.phase}")
    if new_state.phase == "playing":
        next_player = new_state.players[new_state.current_player_index]
        print(f"Next player: {next_player.name} ({next_player.id})")


if __name__ == "__main__":
    import sys
    game_id = sys.argv[1] if len(sys.argv) > 1 else "6952596d-4fa0-4ead-93cd-63257d010f55"
    player_id = sys.argv[2] if len(sys.argv) > 2 else None
    advance_game(game_id, player_id)

