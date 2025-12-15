#!/usr/bin/env python3
"""
Quick test script for mixed multiplayer - completes setup only.
Creates a 4-player game with 2 real players and 2 LLM agents,
plays through settlement setup, then provides game ID for manual testing.
"""
import requests
import json
import time
import sys
from typing import Optional, Dict, Any

API_BASE = "http://localhost:8000/api"

def wait_for_backend(max_retries=10, delay=1):
    """Wait for backend to be ready."""
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        if i < max_retries - 1:
            print(f"Waiting for backend... ({i+1}/{max_retries})")
            time.sleep(delay)
    return False

def register_user(username: str, password: str, email: str = None):
    """Register a new user."""
    response = requests.post(
        f"{API_BASE}/auth/register",
        json={
            "username": username,
            "password": password,
            "email": email
        }
    )
    if response.status_code != 200:
        # Try to login if already exists
        return login_user(username, password)
    data = response.json()
    print(f"✓ Registered user: {username} (ID: {data['user']['id']})")
    return data['access_token'], data['user']

def login_user(username: str, password: str):
    """Login a user."""
    response = requests.post(
        f"{API_BASE}/auth/login",
        json={
            "username": username,
            "password": password
        }
    )
    if response.status_code != 200:
        print(f"Failed to login {username}: {response.status_code} - {response.text}")
        return None, None
    data = response.json()
    print(f"✓ Logged in user: {username}")
    return data['access_token'], data['user']

def create_game(token: str, player_names: list, agent_mapping: Dict[str, str] = None):
    """Create a new game."""
    payload = {
        "player_names": player_names,
        "rng_seed": None
    }
    if agent_mapping:
        payload["agent_mapping"] = agent_mapping
    
    response = requests.post(
        f"{API_BASE}/games",
        headers={"Authorization": f"Bearer {token}"},
        json=payload
    )
    if response.status_code != 200:
        print(f"Failed to create game: {response.status_code} - {response.text}")
        return None, None
    
    game_data = response.json()
    game_id = game_data['game_id']
    initial_state = game_data['initial_state']
    
    print(f"✓ Created game: {game_id}")
    
    return game_id, initial_state

def get_game_state(game_id: str):
    """Get current game state."""
    response = requests.get(f"{API_BASE}/games/{game_id}")
    if response.status_code != 200:
        print(f"Failed to get game state: {response.status_code} - {response.text}")
        return None
    return response.json()

def get_legal_actions(game_id: str, player_id: str):
    """Get legal actions for a player."""
    response = requests.get(
        f"{API_BASE}/games/{game_id}/legal_actions",
        params={"player_id": player_id}
    )
    if response.status_code != 200:
        return []
    data = response.json()
    return data.get('legal_actions', [])

def execute_action(game_id: str, player_id: str, action: Dict[str, Any], token: str = None):
    """Execute a game action."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    response = requests.post(
        f"{API_BASE}/games/{game_id}/act",
        headers=headers,
        json={
            "player_id": player_id,
            "action": action
        }
    )
    if response.status_code != 200:
        return None
    return response.json()['new_state']

def watch_agents_step(game_id: str, agent_mapping: Dict[str, str]):
    """Execute one step with agents."""
    response = requests.post(
        f"{API_BASE}/games/{game_id}/watch_agents_step",
        json={
            "agent_mapping": agent_mapping
        }
    )
    if response.status_code != 200:
        print(f"Failed to watch agents step: {response.status_code} - {response.text}")
        return None
    return response.json()

def play_setup_phase(game_id: str, real_player_tokens: Dict[str, str], agent_mapping: Dict[str, str]):
    """Play through the setup phase."""
    print("\n" + "=" * 60)
    print("Playing Setup Phase")
    print("=" * 60)
    
    max_setup_steps = 100
    step_count = 0
    
    while step_count < max_setup_steps:
        state = get_game_state(game_id)
        if not state:
            return False
        
        if state['phase'] != 'setup':
            print(f"\n✓ Setup phase complete! Game is now in '{state['phase']}' phase")
            return True
        
        current_player_idx = state.get('setup_phase_player_index', state.get('current_player_index', 0))
        current_player = state['players'][current_player_idx]
        player_id = current_player['id']
        player_name = current_player['name']
        
        print(f"Step {step_count + 1}: {player_name} ({player_id})")
        
        # Check if this is an LLM agent
        if player_id in agent_mapping:
            # Use watch_agents_step for LLM agents
            print(f"  → LLM agent turn, using watch_agents_step")
            result = watch_agents_step(game_id, agent_mapping)
            if not result or result.get('error'):
                print(f"  ✗ Agent step failed: {result.get('error') if result else 'No result'}")
                return False
            print(f"  ✓ Agent action executed")
        else:
            # Human player - get legal actions and execute manually
            legal_actions = get_legal_actions(game_id, player_id)
            if not legal_actions:
                break
            
            # Choose action
            action = None
            for legal_action in legal_actions:
                if legal_action['type'] in ['setup_place_settlement', 'build_settlement']:
                    action = legal_action
                    print(f"  → Placing settlement at intersection {legal_action['payload']['intersection_id']}")
                    break
                elif legal_action['type'] in ['setup_place_road', 'build_road']:
                    action = legal_action
                    print(f"  → Placing road on edge {legal_action['payload']['road_edge_id']}")
                    break
            
            if not action:
                action = legal_actions[0]
                print(f"  → Executing: {action['type']}")
            
            token = real_player_tokens.get(player_id)
            new_state = execute_action(game_id, player_id, action, token)
            
            if not new_state:
                return False
            
            print(f"  ✓ Action executed")
        
        step_count += 1
        time.sleep(0.1)
    
    return False

def main():
    print("=" * 60)
    print("Mixed Multiplayer Quick Test (Setup Only)")
    print("=" * 60)
    print()
    
    if not wait_for_backend():
        print("✗ Backend is not responding")
        return 1
    print("✓ Backend is ready")
    print()
    
    # Step 1: Register/Login two real users
    print("Step 1: Setting up real players...")
    token1, user1 = register_user("realplayer1", "test123", "real1@test.com")
    if not token1:
        return 1
    
    token2, user2 = register_user("realplayer2", "test123", "real2@test.com")
    if not token2:
        return 1
    print()
    
    # Step 2: Create a 4-player game
    print("Step 2: Creating 4-player game...")
    player_names = ["realplayer1", "realplayer2", "LLMAgent1", "LLMAgent2"]
    
    # Create agent mapping before creating game
    agent_mapping = {
        'player_2': 'llm',  # LLMAgent1
        'player_3': 'llm'   # LLMAgent2
    }
    
    game_id, initial_state = create_game(token1, player_names, agent_mapping)
    if not game_id:
        return 1
    
    real_player_tokens = {}
    for i, player in enumerate(initial_state['players']):
        if i >= 2:
            print(f"  {player['name']} ({player['id']}) → LLM Agent")
        else:
            print(f"  {player['name']} ({player['id']}) → Real Player")
            if i == 0:
                real_player_tokens[player['id']] = token1
            else:
                real_player_tokens[player['id']] = token2
    
    print(f"\n✓ Game created: {game_id}")
    print(f"  Phase: {initial_state['phase']}")
    print()
    
    # Step 3: Play through setup phase
    print("Step 3: Playing through setup phase...")
    if not play_setup_phase(game_id, real_player_tokens, agent_mapping):
        print("Failed to complete setup")
        return 1
    
    # Verify final state
    final_state = get_game_state(game_id)
    if final_state:
        print("\n" + "=" * 60)
        print("Game Ready for Manual Testing")
        print("=" * 60)
        print(f"Game ID: {game_id}")
        print(f"Phase: {final_state['phase']}")
        print(f"Turn: {final_state['turn_number']}")
        print("\nPlayers:")
        for player in final_state['players']:
            is_agent = player['id'] in agent_mapping
            print(f"  {player['name']} ({player['id']}) - {'LLM Agent' if is_agent else 'Real Player'}")
        print("\n" + "=" * 60)
        print("✓ Setup complete! Game is ready for manual testing")
        print("=" * 60)
        print("\nTo test:")
        print(f"  1. Open http://localhost:5175")
        print(f"  2. Login as 'realplayer1' / 'test123'")
        print(f"  3. Load game: {game_id}")
        print(f"  4. Open another browser/incognito")
        print(f"  5. Login as 'realplayer2' / 'test123'")
        print(f"  6. Load same game: {game_id}")
        print(f"  7. Both players can now play - LLM agents will auto-play their turns")
        print()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

