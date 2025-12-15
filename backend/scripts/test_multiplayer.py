#!/usr/bin/env python3
"""
Test script for multiplayer functionality.
Registers two users, creates a room, joins, and starts a game.
"""
import requests
import json
import time
import sys

API_BASE = "http://localhost:8000/api"

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
        print(f"Failed to register {username}: {response.status_code} - {response.text}")
        return None
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
        return None
    data = response.json()
    print(f"✓ Logged in user: {username}")
    return data['access_token'], data['user']

def create_room(token: str, max_players: int = 2, min_players: int = 2):
    """Create a game room."""
    response = requests.post(
        f"{API_BASE}/rooms",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "max_players": max_players,
            "min_players": min_players,
            "is_private": False
        }
    )
    if response.status_code != 200:
        print(f"Failed to create room: {response.status_code} - {response.text}")
        return None
    room = response.json()
    print(f"✓ Created room: {room['room_id']}")
    return room

def join_room(token: str, room_id: str):
    """Join a game room."""
    response = requests.post(
        f"{API_BASE}/rooms/{room_id}/join",
        headers={"Authorization": f"Bearer {token}"},
        json={}
    )
    if response.status_code != 200:
        print(f"Failed to join room: {response.status_code} - {response.text}")
        return None
    room = response.json()
    print(f"✓ Joined room: {room_id}")
    return room

def start_game(token: str, room_id: str):
    """Start a game from a room."""
    response = requests.post(
        f"{API_BASE}/rooms/{room_id}/start",
        headers={"Authorization": f"Bearer {token}"}
    )
    if response.status_code != 200:
        print(f"Failed to start game: {response.status_code} - {response.text}")
        return None
    data = response.json()
    print(f"✓ Started game: {data['game_id']}")
    return data

def get_game_state(game_id: str):
    """Get game state."""
    response = requests.get(f"{API_BASE}/games/{game_id}")
    if response.status_code != 200:
        print(f"Failed to get game state: {response.status_code} - {response.text}")
        return None
    return response.json()

def main():
    print("=" * 60)
    print("Multiplayer Test Script")
    print("=" * 60)
    print()
    
    # Step 1: Register two users
    print("Step 1: Registering users...")
    token1, user1 = register_user("player1", "test123", "player1@test.com")
    if not token1:
        print("Failed to register first user")
        return 1
    
    token2, user2 = register_user("player2", "test123", "player2@test.com")
    if not token2:
        print("Failed to register second user")
        return 1
    
    print()
    
    # Step 2: Create a room (as player1)
    print("Step 2: Creating room...")
    room = create_room(token1, max_players=2, min_players=2)
    if not room:
        print("Failed to create room")
        return 1
    
    room_id = room['room_id']
    print()
    
    # Step 3: Player2 joins the room
    print("Step 3: Player2 joining room...")
    room = join_room(token2, room_id)
    if not room:
        print("Failed to join room")
        return 1
    
    print(f"  Room now has {room['player_count']}/{room['max_players']} players")
    print()
    
    # Step 4: Start the game (as player1, the host)
    print("Step 4: Starting game...")
    game_data = start_game(token1, room_id)
    if not game_data:
        print("Failed to start game")
        return 1
    
    game_id = game_data['game_id']
    print()
    
    # Step 5: Verify game state
    print("Step 5: Verifying game state...")
    game_state = get_game_state(game_id)
    if not game_state:
        print("Failed to get game state")
        return 1
    
    print(f"  Game ID: {game_id}")
    print(f"  Phase: {game_state['phase']}")
    print(f"  Players: {len(game_state['players'])}")
    for i, player in enumerate(game_state['players']):
        print(f"    {i+1}. {player['name']} (ID: {player['id']})")
    print()
    
    print("=" * 60)
    print("✓ Multiplayer test completed successfully!")
    print("=" * 60)
    print()
    print(f"Game ID: {game_id}")
    print(f"Room ID: {room_id}")
    print()
    print("You can now:")
    print(f"  1. Open http://localhost:5175 in your browser")
    print(f"  2. Login as 'player1' with password 'test123'")
    print(f"  3. Open another browser/incognito")
    print(f"  4. Login as 'player2' with password 'test123'")
    print(f"  5. Both can load game ID: {game_id}")
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

