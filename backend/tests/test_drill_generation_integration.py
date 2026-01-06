"""
Integration test for drill generation that simulates the full frontend flow.
This test makes real LLM API calls and may take several minutes.

Run with: pytest tests/test_drill_generation_integration.py -v -s
"""
import pytest
from fastapi.testclient import TestClient
from main import app
import time

client = TestClient(app)


def test_full_drill_generation_flow_simulates_frontend():
    """
    Test the full flow that happens when user clicks "Extract Candidates" then "Run Comparison".
    This makes real LLM API calls and may take several minutes.
    
    This test simulates exactly what the frontend does:
    1. User enters game ID and clicks "Extract Candidates"
    2. User clicks "Run Comparison" with good/worse models
    3. System processes and returns disagreements
    """
    # Create a game and play a few moves to generate steps
    create_response = client.post(
        "/api/games",
        json={"player_names": ["Alice", "Bob", "Charlie"]}
    )
    assert create_response.status_code == 200
    game_id = create_response.json()["game_id"]
    
    # Start the game
    client.post(
        f"/api/games/{game_id}/act",
        json={"player_id": "player_0", "action": {"type": "start_game", "payload": None}}
    )
    
    # Play a few turns to generate steps with multiple legal actions
    for i in range(5):
        # Roll dice
        client.post(
            f"/api/games/{game_id}/act",
            json={"player_id": f"player_{i % 3}", "action": {"type": "roll_dice", "payload": None}}
        )
        # End turn
        client.post(
            f"/api/games/{game_id}/act",
            json={"player_id": f"player_{i % 3}", "action": {"type": "end_turn", "payload": None}}
        )
    
    # Step 1: Extract candidates (simulates clicking "Extract Candidates")
    print("\n[TEST] Step 1: Extracting candidates (simulating frontend 'Extract Candidates' click)...")
    extract_response = client.post(
        f"/api/games/{game_id}/extract_drill_candidates",
        json={"num_steps": 5, "player_id": None},
        timeout=120.0
    )
    
    assert extract_response.status_code == 200, f"Extract failed: {extract_response.json()}"
    candidates_data = extract_response.json()
    candidates = candidates_data.get("candidates", [])
    
    print(f"[TEST] Found {len(candidates)} candidates")
    
    if len(candidates) == 0:
        pytest.skip("No candidates found - need a game with more non-trivial steps")
    
    # Step 2: Compare LLM actions (simulates clicking "Run Comparison")
    print(f"\n[TEST] Step 2: Comparing LLM actions (simulating frontend 'Run Comparison' click)...")
    print(f"[TEST] Good model: claude-sonnet-4-5-20250929")
    print(f"[TEST] Worse model: gpt-4o")
    print("[TEST] This will make real LLM API calls...")
    
    start_time = time.time()
    compare_response = client.post(
        f"/api/games/{game_id}/compare_llm_actions",
        json={
            "candidates": candidates,
            "good_model": "claude-sonnet-4-5-20250929",
            "worse_model": "gpt-4o"
        },
        timeout=600.0  # 10 minute timeout
    )
    elapsed = time.time() - start_time
    
    print(f"[TEST] Comparison completed in {elapsed:.2f}s")
    print(f"[TEST] Status: {compare_response.status_code}")
    
    if compare_response.status_code != 200:
        error_data = compare_response.json()
        print(f"[TEST] Error response: {error_data}")
        pytest.fail(f"Comparison failed with status {compare_response.status_code}: {error_data}")
    
    result = compare_response.json()
    disagreements = result.get("disagreements", [])
    
    print(f"[TEST] Found {len(disagreements)} disagreements")
    
    # Verify response structure matches what frontend expects
    assert "disagreements" in result
    assert isinstance(disagreements, list)
    
    # If we have disagreements, verify their structure matches frontend interface
    if len(disagreements) > 0:
        first = disagreements[0]
        required_fields = ["step_idx", "player_id", "state_before_json", "good_action", "worse_action", "legal_actions"]
        for field in required_fields:
            assert field in first, f"Missing required field: {field}"
        
        # Verify action structure
        assert "type" in first["good_action"]
        assert "type" in first["worse_action"]
        assert isinstance(first["legal_actions"], list)
        
        print(f"[TEST] ✓ First disagreement structure is valid")
        print(f"[TEST]   Step: {first['step_idx']}, Player: {first['player_id']}")
        print(f"[TEST]   Good: {first['good_action']['type']}, Worse: {first['worse_action']['type']}")
    
    print(f"[TEST] ✓ Test passed - frontend flow simulation successful")
    
    # Step 1: Extract candidates (simulates clicking "Extract Candidates" button)
    print("\n[TEST] Step 1: Extracting candidates...")
    extract_response = client.post(
        f"/api/games/{game_id}/extract_drill_candidates",
        json={"num_steps": 10, "player_id": None},
        timeout=120.0
    )
    
    assert extract_response.status_code == 200, f"Extract failed: {extract_response.json()}"
    candidates_data = extract_response.json()
    candidates = candidates_data.get("candidates", [])
    assert len(candidates) > 0, "No candidates found"
    
    print(f"[TEST] Found {len(candidates)} candidates")
    step_indices = [c["step_idx"] for c in candidates]
    print(f"[TEST] Step indices: {step_indices}")
    
    # Step 2: Compare LLM actions (simulates clicking "Run Comparison" button)
    print(f"\n[TEST] Step 2: Comparing LLM actions on {len(candidates)} candidates...")
    print("[TEST] This will make real LLM API calls and may take several minutes...")
    
    start_time = time.time()
    compare_response = client.post(
        f"/api/games/{game_id}/compare_llm_actions",
        json={
            "candidates": candidates,
            "good_model": "claude-sonnet-4-5-20250929",
            "worse_model": "gpt-4o"
        },
        timeout=600.0  # 10 minute timeout
    )
    elapsed = time.time() - start_time
    
    print(f"[TEST] Comparison completed in {elapsed:.2f}s")
    print(f"[TEST] Status: {compare_response.status_code}")
    
    if compare_response.status_code != 200:
        error_data = compare_response.json()
        print(f"[TEST] Error response: {error_data}")
        pytest.fail(f"Comparison failed with status {compare_response.status_code}: {error_data}")
    
    result = compare_response.json()
    disagreements = result.get("disagreements", [])
    
    print(f"[TEST] Found {len(disagreements)} disagreements")
    
    # Verify response structure
    assert "disagreements" in result
    assert isinstance(disagreements, list)
    
    # If we have disagreements, verify their structure
    if len(disagreements) > 0:
        first = disagreements[0]
        assert "step_idx" in first
        assert "player_id" in first
        assert "state_before_json" in first
        assert "good_action" in first
        assert "worse_action" in first
        assert "legal_actions" in first
        
        print(f"[TEST] First disagreement at step_idx={first['step_idx']}")
        print(f"[TEST] Good action: {first['good_action'].get('type')}")
        print(f"[TEST] Worse action: {first['worse_action'].get('type')}")
    
    # Test passes if we get a valid response (even if no disagreements)
    assert True


def test_compare_llm_actions_with_small_sample():
    """
    Test comparison with just 2 candidates to verify it works quickly.
    Uses a real game ID - set REAL_GAME_ID environment variable or it will be skipped.
    """
    import os
    game_id = os.getenv("REAL_GAME_ID", "e74509c8-a115-48e2-acf0-a070176affbc")
    
    # Check if game exists
    check_response = client.get(f"/api/games/{game_id}")
    if check_response.status_code == 404:
        pytest.skip(f"Game {game_id} not found in database. Set REAL_GAME_ID env var to test with a different game.")
    
    # Extract just 2 candidates
    extract_response = client.post(
        f"/api/games/{game_id}/extract_drill_candidates",
        json={"num_steps": 2, "player_id": None},
        timeout=120.0
    )
    
    assert extract_response.status_code == 200, f"Extract failed: {extract_response.json()}"
    candidates = extract_response.json()["candidates"]
    
    if len(candidates) == 0:
        pytest.skip("No candidates found")
    
    # Compare with same model (should have no disagreements, but tests the endpoint)
    compare_response = client.post(
        f"/api/games/{game_id}/compare_llm_actions",
        json={
            "candidates": candidates[:1],  # Just test with 1 candidate
            "good_model": "gpt-4o-mini",
            "worse_model": "gpt-4o-mini"  # Same model = should agree
        },
        timeout=120.0
    )
    
    assert compare_response.status_code == 200, f"Compare failed: {compare_response.json()}"
    result = compare_response.json()
    assert "disagreements" in result
    # With same model, there should be no disagreements (unless non-deterministic)

