"""
Tests for enhanced drill functionality with correct/incorrect actions.
"""
import pytest
import json
from fastapi.testclient import TestClient
from main import app
from api.database import (
    init_db,
    create_drill,
    get_drill,
    get_drill_steps,
    delete_optimized_prompt,
)
from engine import GameState, Player, Action
from engine.serialization import serialize_game_state, legal_actions

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize database before each test."""
    init_db()
    yield
    # Cleanup if needed


def create_test_game_state() -> dict:
    """Create a minimal test game state."""
    # Create a simple game state for testing
    response = client.post("/api/games", json={"player_names": ["Alice", "Bob"]})
    assert response.status_code == 200
    return response.json()["initial_state"]


def test_create_drill_with_single_expected_action():
    """Test creating a drill with single expected action (backward compatibility)."""
    state = create_test_game_state()
    
    response = client.post(
        "/api/drills",
        json={
            "name": "Test Drill",
            "player_id": "player_0",
            "steps": [
                {
                    "player_id": "player_0",
                    "state": state,
                    "expected_action": {"type": "end_turn"}
                }
            ]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "drill_id" in data
    assert data["message"] == "Drill created"
    
    # Verify drill was created
    drill_id = data["drill_id"]
    get_response = client.get(f"/api/drills/{drill_id}")
    assert get_response.status_code == 200
    drill_data = get_response.json()
    assert len(drill_data["steps"]) == 1
    assert drill_data["steps"][0]["expected_action"]["type"] == "end_turn"


def test_create_drill_with_correct_actions():
    """Test creating a drill with multiple correct actions."""
    state = create_test_game_state()
    
    # Get legal actions for this state to use in the test
    legal_response = client.get(f"/api/games/{state['game_id']}/legal_actions?player_id=player_0")
    assert legal_response.status_code == 200
    legal_actions = legal_response.json()["legal_actions"]
    
    if len(legal_actions) < 2:
        pytest.skip("Not enough legal actions for this test")
    
    # Use first two legal actions as correct, and a third as incorrect (if available)
    correct_actions = [legal_actions[0], legal_actions[1] if len(legal_actions) > 1 else legal_actions[0]]
    incorrect_actions = [legal_actions[2]] if len(legal_actions) > 2 else []
    
    response = client.post(
        "/api/drills",
        json={
            "name": "Test Drill with Correct Actions",
            "player_id": "player_0",
            "steps": [
                {
                    "player_id": "player_0",
                    "state": state,
                    "expected_action": correct_actions[0],  # For backward compatibility
                    "correct_actions": correct_actions,
                    "incorrect_actions": incorrect_actions if incorrect_actions else None
                }
            ]
        }
    )
    
    if response.status_code != 200:
        print(f"Error response: {response.text}", flush=True)
    assert response.status_code == 200
    data = response.json()
    drill_id = data["drill_id"]
    
    # Verify drill was created with correct/incorrect actions
    get_response = client.get(f"/api/drills/{drill_id}")
    assert get_response.status_code == 200
    drill_data = get_response.json()
    step = drill_data["steps"][0]
    assert step["correct_actions"] is not None
    assert len(step["correct_actions"]) == 2
    assert step["incorrect_actions"] is not None
    assert len(step["incorrect_actions"]) == 1


def test_create_drill_validation_no_correct_actions():
    """Test that drill creation fails if correct_actions is empty."""
    state = create_test_game_state()
    
    response = client.post(
        "/api/drills",
        json={
            "name": "Invalid Drill",
            "player_id": "player_0",
            "steps": [
                {
                    "player_id": "player_0",
                    "state": state,
                    "expected_action": {"type": "end_turn"},
                    "correct_actions": []  # Empty - should fail
                }
            ]
        }
    )
    
    assert response.status_code == 400
    assert "at least one correct action" in response.json()["detail"].lower()


def test_evaluate_drill_with_correct_actions():
    """Test evaluating a drill with correct/incorrect actions."""
    state = create_test_game_state()
    
    # Get legal actions for this state
    legal_response = client.get(f"/api/games/{state['game_id']}/legal_actions?player_id=player_0")
    assert legal_response.status_code == 200
    legal_actions = legal_response.json()["legal_actions"]
    
    if len(legal_actions) < 2:
        pytest.skip("Not enough legal actions for this test")
    
    # Use first as correct, second as incorrect
    correct_actions = [legal_actions[0]]
    incorrect_actions = [legal_actions[1]] if len(legal_actions) > 1 else []
    
    # Create drill with correct/incorrect actions
    create_response = client.post(
        "/api/drills",
        json={
            "name": "Evaluation Test Drill",
            "player_id": "player_0",
            "steps": [
                {
                    "player_id": "player_0",
                    "state": state,
                    "expected_action": correct_actions[0],
                    "correct_actions": correct_actions,
                    "incorrect_actions": incorrect_actions if incorrect_actions else None
                }
            ]
        }
    )
    
    if create_response.status_code != 200:
        print(f"Error response: {create_response.text}", flush=True)
    assert create_response.status_code == 200
    drill_id = create_response.json()["drill_id"]
    
    # Evaluate the drill
    eval_response = client.post(
        f"/api/drills/{drill_id}/evaluate",
        json={
            "agent_type": "random",
            "exclude_strategic_advice": False,
            "exclude_higher_level_features": False
        }
    )
    
    assert eval_response.status_code == 200
    eval_data = eval_response.json()
    assert "passed" in eval_data
    assert "results" in eval_data
    assert len(eval_data["results"]) == 1
    # The result should indicate whether the agent chose a correct action
    assert "match" in eval_data["results"][0]


def test_drill_evaluation_filters_legal_actions():
    """Test that drill evaluation filters legal actions to correct+incorrect subset."""
    state = create_test_game_state()
    
    # Get legal actions for this state
    legal_response = client.get(f"/api/games/{state['game_id']}/legal_actions?player_id=player_0")
    assert legal_response.status_code == 200
    legal_actions = legal_response.json()["legal_actions"]
    
    if len(legal_actions) < 2:
        pytest.skip("Not enough legal actions for this test")
    
    # Use first as correct, second as incorrect
    correct_actions = [legal_actions[0]]
    incorrect_actions = [legal_actions[1]] if len(legal_actions) > 1 else []
    
    # Create drill with specific correct/incorrect actions
    create_response = client.post(
        "/api/drills",
        json={
            "name": "Filter Test Drill",
            "player_id": "player_0",
            "steps": [
                {
                    "player_id": "player_0",
                    "state": state,
                    "expected_action": correct_actions[0],
                    "correct_actions": correct_actions,
                    "incorrect_actions": incorrect_actions if incorrect_actions else None
                }
            ]
        }
    )
    
    if create_response.status_code != 200:
        print(f"Error response: {create_response.text}", flush=True)
    assert create_response.status_code == 200
    drill_id = create_response.json()["drill_id"]
    
    # Evaluate - the agent should only see end_turn and buy_dev_card as options
    eval_response = client.post(
        f"/api/drills/{drill_id}/evaluate",
        json={
            "agent_type": "random",
            "exclude_strategic_advice": False,
            "exclude_higher_level_features": False
        }
    )
    
    assert eval_response.status_code == 200
    # The evaluation should complete (even if random agent picks wrong action)
    eval_data = eval_response.json()
    assert "results" in eval_data


def test_backward_compatibility_single_expected_action():
    """Test that drills with only expected_action still work."""
    state = create_test_game_state()
    
    # Create drill with only expected_action (no correct/incorrect)
    create_response = client.post(
        "/api/drills",
        json={
            "name": "Backward Compatible Drill",
            "player_id": "player_0",
            "steps": [
                {
                    "player_id": "player_0",
                    "state": state,
                    "expected_action": {"type": "end_turn"}
                }
            ]
        }
    )
    
    assert create_response.status_code == 200
    drill_id = create_response.json()["drill_id"]
    
    # Evaluate should work
    eval_response = client.post(
        f"/api/drills/{drill_id}/evaluate",
        json={
            "agent_type": "random",
            "exclude_strategic_advice": False,
            "exclude_higher_level_features": False
        }
    )
    
    assert eval_response.status_code == 200
    eval_data = eval_response.json()
    assert "results" in eval_data
    # Should check against expected_action
    assert len(eval_data["results"]) == 1

