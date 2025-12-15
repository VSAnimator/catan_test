"""
Tests for the FastAPI endpoints using TestClient.
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_create_game():
    """Test creating a new game."""
    response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob"]}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "game_id" in data
    assert "initial_state" in data
    assert data["initial_state"]["game_id"] == data["game_id"]
    assert len(data["initial_state"]["players"]) == 2
    assert data["initial_state"]["players"][0]["name"] == "Alice"
    assert data["initial_state"]["players"][1]["name"] == "Bob"
    assert data["initial_state"]["phase"] == "setup"


def test_create_game_invalid_player_count():
    """Test creating a game with invalid player count."""
    # Too few players
    response = client.post(
        "/api/games",
        json={"player_names": ["Alice"]}
    )
    assert response.status_code == 400
    
    # Too many players
    response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob", "Charlie", "David", "Eve"]}
    )
    assert response.status_code == 400


def test_get_game():
    """Test getting a game state."""
    # First create a game
    create_response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob"]}
    )
    game_id = create_response.json()["game_id"]
    
    # Then get it
    response = client.get(f"/api/games/{game_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["game_id"] == game_id
    assert len(data["players"]) == 2
    assert data["phase"] == "setup"


def test_get_game_not_found():
    """Test getting a non-existent game."""
    response = client.get("/api/games/nonexistent")
    assert response.status_code == 404


def test_act_roll_dice():
    """Test performing a roll_dice action."""
    # Create a game
    create_response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob"]}
    )
    game_id = create_response.json()["game_id"]
    
    # Start the game first (need to create board)
    # We need to call START_GAME action first
    act_response = client.post(
        f"/api/games/{game_id}/act",
        json={
            "player_id": "player_0",
            "action": {
                "type": "start_game",
                "payload": None
            }
        }
    )
    
    # Now roll dice
    act_response = client.post(
        f"/api/games/{game_id}/act",
        json={
            "player_id": "player_0",
            "action": {
                "type": "roll_dice",
                "payload": None
            }
        }
    )
    
    assert act_response.status_code == 200
    data = act_response.json()
    assert "new_state" in data
    # Dice roll should be set (2-12)
    assert data["new_state"]["dice_roll"] is not None
    assert 2 <= data["new_state"]["dice_roll"] <= 12


def test_act_invalid_player():
    """Test performing an action with invalid player."""
    # Create a game
    create_response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob"]}
    )
    game_id = create_response.json()["game_id"]
    
    # Try to act with non-existent player
    response = client.post(
        f"/api/games/{game_id}/act",
        json={
            "player_id": "player_999",
            "action": {
                "type": "roll_dice",
                "payload": None
            }
        }
    )
    
    assert response.status_code == 400


def test_act_wrong_turn():
    """Test performing an action when it's not the player's turn."""
    # Create a game
    create_response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob"]}
    )
    game_id = create_response.json()["game_id"]
    
    # Try to act with player_1 when it's player_0's turn
    response = client.post(
        f"/api/games/{game_id}/act",
        json={
            "player_id": "player_1",
            "action": {
                "type": "roll_dice",
                "payload": None
            }
        }
    )
    
    assert response.status_code == 400


def test_replay():
    """Test getting game replay logs."""
    # Create a game
    create_response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob"]}
    )
    game_id = create_response.json()["game_id"]
    
    # Initially, replay should be empty
    replay_response = client.get(f"/api/games/{game_id}/replay")
    assert replay_response.status_code == 200
    data = replay_response.json()
    assert data["game_id"] == game_id
    assert len(data["steps"]) == 0
    
    # Perform an action
    act_response = client.post(
        f"/api/games/{game_id}/act",
        json={
            "player_id": "player_0",
            "action": {
                "type": "start_game",
                "payload": None
            }
        }
    )
    assert act_response.status_code == 200
    
    # Now replay should have one step
    replay_response = client.get(f"/api/games/{game_id}/replay")
    assert replay_response.status_code == 200
    data = replay_response.json()
    assert len(data["steps"]) == 1
    step = data["steps"][0]
    assert "state_before" in step
    assert "action" in step
    assert "state_after" in step
    assert "timestamp" in step
    assert step["action"]["type"] == "start_game"


def test_replay_not_found():
    """Test getting replay for non-existent game."""
    response = client.get("/api/games/nonexistent/replay")
    assert response.status_code == 404


def test_happy_path_full_flow():
    """Test a complete happy path: create game, start, roll dice, end turn."""
    # Create game
    create_response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob"]}
    )
    assert create_response.status_code == 200
    game_id = create_response.json()["game_id"]
    
    # Get initial state
    get_response = client.get(f"/api/games/{game_id}")
    assert get_response.status_code == 200
    initial_state = get_response.json()
    assert initial_state["phase"] == "setup"
    
    # Start game
    act_response = client.post(
        f"/api/games/{game_id}/act",
        json={
            "player_id": "player_0",
            "action": {
                "type": "start_game",
                "payload": None
            }
        }
    )
    assert act_response.status_code == 200
    state_after_start = act_response.json()["new_state"]
    assert state_after_start["phase"] == "playing"
    assert len(state_after_start["tiles"]) > 0  # Board should be created
    
    # Roll dice
    act_response = client.post(
        f"/api/games/{game_id}/act",
        json={
            "player_id": "player_0",
            "action": {
                "type": "roll_dice",
                "payload": None
            }
        }
    )
    assert act_response.status_code == 200
    state_after_roll = act_response.json()["new_state"]
    assert state_after_roll["dice_roll"] is not None
    
    # End turn
    act_response = client.post(
        f"/api/games/{game_id}/act",
        json={
            "player_id": "player_0",
            "action": {
                "type": "end_turn",
                "payload": None
            }
        }
    )
    assert act_response.status_code == 200
    state_after_end = act_response.json()["new_state"]
    assert state_after_end["current_player_index"] == 1  # Should be Bob's turn now
    
    # Check replay has all steps
    replay_response = client.get(f"/api/games/{game_id}/replay")
    assert replay_response.status_code == 200
    steps = replay_response.json()["steps"]
    assert len(steps) == 3  # start_game, roll_dice, end_turn
    assert steps[0]["action"]["type"] == "start_game"
    assert steps[1]["action"]["type"] == "roll_dice"
    assert steps[2]["action"]["type"] == "end_turn"

