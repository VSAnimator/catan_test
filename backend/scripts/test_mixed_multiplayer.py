#!/usr/bin/env python3
"""
Test script for mixed multiplayer (real players + LLM agents).
Creates a 4-player game with 2 real players and 2 LLM agents,
plays through settlement setup, then continues with LLM agents.
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
    response = requests.post(
        f"{API_BASE}/games",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "player_names": player_names,
            "rng_seed": None
        }
    )
    if response.status_code != 200:
        print(f"Failed to create game: {response.status_code} - {response.text}")
        return None
    
    game_data = response.json()
    game_id = game_data['game_id']
    print(f"✓ Created game: {game_id}")
    
    # Set agent mapping if provided
    if agent_mapping:
        # Update game metadata with agent mapping
        # We need to update the game's metadata
        # For now, we'll store it in the game state metadata
        # The agent_mapping will be used when calling watch_agents_step
        print(f"  Agent mapping: {agent_mapping}")
    
    return game_id, game_data['initial_state']

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
        print(f"Failed to get legal actions: {response.status_code} - {response.text}")
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
        print(f"Failed to execute action: {response.status_code} - {response.text}")
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
    
    max_setup_steps = 100  # Safety limit
    step_count = 0
    
    while step_count < max_setup_steps:
        state = get_game_state(game_id)
        if not state:
            print("Failed to get game state")
            return False
        
        if state['phase'] != 'setup':
            print(f"\n✓ Setup phase complete! Game is now in '{state['phase']}' phase")
            return True
        
        # Determine current player
        if 'setup_phase_player_index' in state:
            current_player_idx = state['setup_phase_player_index']
        else:
            current_player_idx = state.get('current_player_index', 0)
        
        current_player = state['players'][current_player_idx]
        player_id = current_player['id']
        player_name = current_player['name']
        
        print(f"\nStep {step_count + 1}: {player_name} ({player_id})")
        
        # Get legal actions
        legal_actions = get_legal_actions(game_id, player_id)
        if not legal_actions:
            print(f"  No legal actions available for {player_name}")
            break
        
        # Choose action
        action = None
        
        # Priority: place settlement, then place road
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
            # Use first available action
            action = legal_actions[0]
            print(f"  → Executing: {action['type']}")
        
        # Execute action
        token = real_player_tokens.get(player_id)
        new_state = execute_action(game_id, player_id, action, token)
        
        if not new_state:
            print(f"  ✗ Failed to execute action")
            return False
        
        print(f"  ✓ Action executed successfully")
        step_count += 1
        time.sleep(0.1)  # Small delay
    
    print(f"\n⚠ Setup phase incomplete after {max_setup_steps} steps")
    return False

def play_with_agents(game_id: str, agent_mapping: Dict[str, str], max_turns: int = 20):
    """Play game with LLM agents."""
    print("\n" + "=" * 60)
    print("Playing with LLM Agents")
    print("=" * 60)
    
    for turn in range(max_turns):
        state = get_game_state(game_id)
        if not state:
            print("Failed to get game state")
            return False
        
        if state['phase'] == 'finished':
            winner = None
            for player in state['players']:
                if player['victory_points'] >= 10:
                    winner = player
                    break
            if winner:
                print(f"\n✓ Game finished! Winner: {winner['name']} with {winner['victory_points']} VP")
            else:
                print("\n✓ Game finished!")
            return True
        
        if state['phase'] != 'playing':
            print(f"Game is in '{state['phase']}' phase, waiting...")
            time.sleep(1)
            continue
        
        print(f"\nTurn {turn + 1}:")
        print(f"  Current player: {state['players'][state['current_player_index']]['name']}")
        print(f"  Turn number: {state['turn_number']}")
        
        # Execute agent step
        result = watch_agents_step(game_id, agent_mapping)
        if not result:
            print("  ✗ Failed to execute agent step")
            return False
        
        if result.get('error'):
            print(f"  ✗ Agent error: {result['error']}")
            return False
        
        if not result.get('game_continues', True):
            print("  Game ended")
            return True
        
        if result.get('reasoning'):
            print(f"  Agent reasoning: {result['reasoning'][:100]}...")
        
        time.sleep(0.5)  # Small delay between turns
    
    print(f"\n⚠ Reached max turns ({max_turns})")
    return True

def main():
    print("=" * 60)
    print("Mixed Multiplayer Test (2 Real Players + 2 LLM Agents)")
    print("=" * 60)
    print()
    
    # Wait for backend
    print("Checking backend connection...")
    if not wait_for_backend():
        print("✗ Backend is not responding. Please ensure the server is running.")
        return 1
    print("✓ Backend is ready")
    print()
    
    # Step 1: Register/Login two real users
    print("Step 1: Setting up real players...")
    token1, user1 = register_user("realplayer1", "test123", "real1@test.com")
    if not token1:
        print("Failed to setup first real player")
        return 1
    
    token2, user2 = register_user("realplayer2", "test123", "real2@test.com")
    if not token2:
        print("Failed to setup second real player")
        return 1
    
    print()
    
    # Step 2: Create a 4-player game
    print("Step 2: Creating 4-player game...")
    player_names = ["realplayer1", "realplayer2", "LLMAgent1", "LLMAgent2"]
    game_id, initial_state = create_game(token1, player_names)
    if not game_id:
        print("Failed to create game")
        return 1
    
    # Map player IDs to agent types
    # Players 2 and 3 (indices 2, 3) will be LLM agents
    agent_mapping = {}
    for i, player in enumerate(initial_state['players']):
        if i >= 2:  # Third and fourth players are LLM agents
            agent_mapping[player['id']] = 'llm'
            print(f"  {player['name']} ({player['id']}) → LLM Agent")
        else:
            print(f"  {player['name']} ({player['id']}) → Real Player")
    
    # Map player IDs to tokens for real players
    real_player_tokens = {}
    for i, player in enumerate(initial_state['players']):
        if i < 2:  # First two players are real
            if i == 0:
                real_player_tokens[player['id']] = token1
            else:
                real_player_tokens[player['id']] = token2
    
    print(f"\n✓ Game created: {game_id}")
    print(f"  Phase: {initial_state['phase']}")
    print(f"  Players: {len(initial_state['players'])}")
    print()
    
    # Step 3: Play through setup phase
    print("Step 3: Playing through setup phase...")
    setup_success = play_setup_phase(game_id, real_player_tokens, agent_mapping)
    if not setup_success:
        print("Failed to complete setup phase")
        return 1
    
    # Step 4: Play with LLM agents
    print("\nStep 4: Playing with LLM agents...")
    play_success = play_with_agents(game_id, agent_mapping, max_turns=20)
    
    # Final state
    final_state = get_game_state(game_id)
    if final_state:
        print("\n" + "=" * 60)
        print("Final Game State")
        print("=" * 60)
        print(f"Phase: {final_state['phase']}")
        print(f"Turn: {final_state['turn_number']}")
        print("\nPlayers:")
        for player in final_state['players']:
            print(f"  {player['name']}: {player['victory_points']} VP, "
                  f"{sum(player['resources'].values())} resources")
    
    print("\n" + "=" * 60)
    print("✓ Test completed!")
    print("=" * 60)
    print(f"\nGame ID: {game_id}")
    print("\nYou can now:")
    print(f"  1. Open http://localhost:5175 in your browser")
    print(f"  2. Login as 'realplayer1' with password 'test123'")
    print(f"  3. Load game ID: {game_id}")
    print(f"  4. Open another browser/incognito")
    print(f"  5. Login as 'realplayer2' with password 'test123'")
    print(f"  6. Load the same game ID")
    print(f"  7. Both players can now play against LLM agents")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

