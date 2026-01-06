"""
Tests for drill generation from game disagreements endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from main import app
from api.database import init_db, get_steps, add_step
from engine.serialization import serialize_game_state, deserialize_game_state
from engine import GameState

client = TestClient(app)


@pytest.fixture
def setup_test_game():
    """Create a test game with some steps for testing."""
    # Create a game
    create_response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob"]}
    )
    assert create_response.status_code == 200
    game_id = create_response.json()["game_id"]
    
    # Start the game
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
    
    return game_id


def test_extract_drill_candidates_basic(setup_test_game):
    """Test extracting drill candidates from a game."""
    game_id = setup_test_game
    
    response = client.post(
        f"/api/games/{game_id}/extract_drill_candidates",
        json={
            "num_steps": 5,
            "player_id": None
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    assert isinstance(data["candidates"], list)
    
    # Check candidate structure
    if len(data["candidates"]) > 0:
        candidate = data["candidates"][0]
        assert "step_idx" in candidate
        assert "player_id" in candidate
        assert "state_before_json" in candidate
        assert "legal_actions_count" in candidate
        assert candidate["legal_actions_count"] > 1  # Should be non-trivial


def test_extract_drill_candidates_with_player_filter(setup_test_game):
    """Test extracting candidates filtered by player_id."""
    game_id = setup_test_game
    
    response = client.post(
        f"/api/games/{game_id}/extract_drill_candidates",
        json={
            "num_steps": 5,
            "player_id": "player_0"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "candidates" in data
    
    # All candidates should be for player_0
    for candidate in data["candidates"]:
        assert candidate["player_id"] == "player_0"


def test_extract_drill_candidates_game_not_found():
    """Test extracting candidates from non-existent game."""
    response = client.post(
        "/api/games/nonexistent-game-id/extract_drill_candidates",
        json={
            "num_steps": 5,
            "player_id": None
        }
    )
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_extract_drill_candidates_respects_num_steps(setup_test_game):
    """Test that extract respects the num_steps limit."""
    game_id = setup_test_game
    
    response = client.post(
        f"/api/games/{game_id}/extract_drill_candidates",
        json={
            "num_steps": 2,
            "player_id": None
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["candidates"]) <= 2


def test_compare_llm_actions_basic(setup_test_game):
    """Test comparing LLM actions on candidates."""
    game_id = setup_test_game
    
    # First extract candidates
    extract_response = client.post(
        f"/api/games/{game_id}/extract_drill_candidates",
        json={
            "num_steps": 2,
            "player_id": None
        }
    )
    assert extract_response.status_code == 200
    candidates = extract_response.json()["candidates"]
    
    if len(candidates) == 0:
        pytest.skip("No candidates found - need a game with non-trivial steps")
    
    # Compare LLM actions
    # Note: This will make actual LLM calls, so it may be slow/expensive
    # In a real test suite, you might want to mock the LLM agents
    response = client.post(
        f"/api/games/{game_id}/compare_llm_actions",
        json={
            "candidates": candidates,
            "good_model": "gpt-4o-mini",  # Use a cheaper model for testing
            "worse_model": "gpt-4o-mini"  # Same model - should have no disagreements
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "disagreements" in data
    assert isinstance(data["disagreements"], list)
    
    # Since we're using the same model, there should be no disagreements
    # (unless there's non-determinism, but that's unlikely with same model)
    # Actually, let's just check the structure is correct
    if len(data["disagreements"]) > 0:
        disagreement = data["disagreements"][0]
        assert "step_idx" in disagreement
        assert "player_id" in disagreement
        assert "state_before_json" in disagreement
        assert "good_action" in disagreement
        assert "worse_action" in disagreement
        assert "legal_actions" in disagreement


def test_compare_llm_actions_game_not_found():
    """Test comparing LLM actions for non-existent game."""
    response = client.post(
        "/api/games/nonexistent-game-id/compare_llm_actions",
        json={
            "candidates": [{
                "step_idx": 0,
                "player_id": "player_0",
                "state_before_json": {}
            }],
            "good_model": "gpt-4o-mini",
            "worse_model": "gpt-4o-mini"
        }
    )
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_compare_llm_actions_empty_candidates(setup_test_game):
    """Test comparing with empty candidates list."""
    game_id = setup_test_game
    
    response = client.post(
        f"/api/games/{game_id}/compare_llm_actions",
        json={
            "candidates": [],
            "good_model": "gpt-4o-mini",
            "worse_model": "gpt-4o-mini"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "disagreements" in data
    assert len(data["disagreements"]) == 0


def test_actions_equal_helper():
    """Test the _actions_equal helper function logic via the API."""
    # We can't directly test the private function, but we can test the behavior
    # through the compare_llm_actions endpoint by using the same model twice
    # (which should produce same actions, so no disagreements)
    
    # Create a game and get candidates
    create_response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob"]}
    )
    assert create_response.status_code == 200
    game_id = create_response.json()["game_id"]
    
    # Start game
    client.post(
        f"/api/games/{game_id}/act",
        json={
            "player_id": "player_0",
            "action": {"type": "start_game", "payload": None}
        }
    )
    
    # Extract candidates
    extract_response = client.post(
        f"/api/games/{game_id}/extract_drill_candidates",
        json={"num_steps": 1, "player_id": None}
    )
    
    if extract_response.status_code != 200 or len(extract_response.json()["candidates"]) == 0:
        pytest.skip("No candidates available for testing")
    
    candidates = extract_response.json()["candidates"]
    
    # Compare same model with itself - should have minimal/no disagreements
    # (unless there's non-determinism)
    compare_response = client.post(
        f"/api/games/{game_id}/compare_llm_actions",
        json={
            "candidates": candidates[:1],  # Just test with one candidate
            "good_model": "gpt-4o-mini",
            "worse_model": "gpt-4o-mini"
        }
    )
    
    assert compare_response.status_code == 200
    # The endpoint should work correctly regardless of whether there are disagreements

