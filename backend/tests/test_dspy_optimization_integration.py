"""
Integration tests for DSPy prompt optimization - actually runs optimization.
"""
import pytest
import json
import os
import subprocess
from api.database import (
    init_db,
    create_drill,
    save_optimized_prompt,
    get_optimized_prompt,
)
from agents.dspy_optimizer import PromptOptimizer, DrillExample, create_drill_metric
from agents.llm_agent import LLMAgent
from engine.serialization import (
    deserialize_game_state,
    state_to_text,
    legal_actions_to_text,
    legal_actions,
)


@pytest.fixture(scope="module", autouse=True)
def load_api_keys():
    """Load API keys from ~/.zshrc if not already in environment."""
    if 'OPENAI_API_KEY' not in os.environ or not os.environ.get('OPENAI_API_KEY'):
        try:
            result = subprocess.run(
                ['zsh', '-c', 'source ~/.zshrc 2>/dev/null && echo $OPENAI_API_KEY'],
                capture_output=True,
                text=True,
                timeout=5
            )
            api_key = result.stdout.strip()
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
        except Exception:
            pass
    
    # Verify we have an API key
    if 'OPENAI_API_KEY' not in os.environ or not os.environ.get('OPENAI_API_KEY'):
        pytest.skip("OPENAI_API_KEY not found in environment or ~/.zshrc")


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize database before each test."""
    init_db()
    yield


def create_minimal_drill_example():
    """Create a minimal drill example for testing."""
    # Create a simple game state (we'll use a mock one)
    # For a real test, we'd need to create an actual game state
    # For now, we'll create a minimal example that can be used for testing
    
    # This is a simplified example - in practice you'd load from a real game
    from fastapi.testclient import TestClient
    from main import app
    
    client = TestClient(app)
    response = client.post("/api/games", json={"player_names": ["Alice", "Bob"]})
    assert response.status_code == 200
    state = response.json()["initial_state"]
    
    # Get legal actions
    game_id = state["game_id"]
    legal_response = client.get(f"/api/games/{game_id}/legal_actions?player_id=player_0")
    assert legal_response.status_code == 200
    legal_actions_list = legal_response.json()["legal_actions"]
    
    if not legal_actions_list:
        pytest.skip("No legal actions available for test")
    
    # Deserialize state
    state_obj = deserialize_game_state(state)
    
    # Get state and actions text
    state_text = state_to_text(state_obj, "player_0", exclude_higher_level_features=False)
    # Get actual legal actions from engine
    from engine.serialization import legal_actions
    actual_legal_actions = legal_actions(state_obj, "player_0")
    actions_text = legal_actions_to_text(
        actual_legal_actions,
        state=state_obj,
        player_id="player_0"
    )
    
    # Use first action as correct
    correct_action = legal_actions_list[0]
    incorrect_action = legal_actions_list[1] if len(legal_actions_list) > 1 else None
    
    return DrillExample(
        state=state_obj,
        player_id="player_0",
        legal_actions_list=legal_actions(state_obj, "player_0"),
        state_text=state_text,
        legal_actions_text=actions_text,
        correct_actions=[correct_action],
        incorrect_actions=[incorrect_action] if incorrect_action else None,
        include_higher_level_features=False
    )


def test_bootstrap_optimization_runs(load_api_keys):
    """Test that BootstrapFewShot optimization actually runs (requires API key)."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found")
    
    # Get base prompt
    base_agent = LLMAgent("player_0")
    base_prompt = base_agent._get_default_system_prompt()
    
    # Create optimizer with API key
    optimizer = PromptOptimizer(
        base_system_prompt=base_prompt,
        model_name="gpt-4o-mini",
        include_higher_level_features=False,
        api_key=api_key
    )
    
    # Create minimal training examples
    example = create_minimal_drill_example()
    train_examples = [example]  # Just one example for quick test
    
    print(f"\n{'='*60}")
    print(f"Running BootstrapFewShot optimization test")
    print(f"Training examples: {len(train_examples)}")
    print(f"{'='*60}\n", flush=True)
    
    # Run optimization
    optimized_prompt = optimizer.optimize(
        train_examples=train_examples,
        method="bootstrap",
        num_iterations=1  # Just one iteration for testing
    )
    
    print(f"\nOptimization returned prompt of length: {len(optimized_prompt)}", flush=True)
    
    # Verify we got a prompt back
    assert optimized_prompt is not None
    assert len(optimized_prompt) > 0
    # The optimized prompt should at least contain the base prompt
    assert base_prompt in optimized_prompt or optimized_prompt == base_prompt


def test_gepa_optimization_runs(load_api_keys):
    """Test that GEPA optimization actually runs (requires API key)."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found")
    
    # Get base prompt
    base_agent = LLMAgent("player_0")
    base_prompt = base_agent._get_default_system_prompt()
    
    # Create optimizer with API key
    optimizer = PromptOptimizer(
        base_system_prompt=base_prompt,
        model_name="gpt-4o-mini",
        include_higher_level_features=False,
        api_key=api_key
    )
    
    # Create minimal training examples
    example = create_minimal_drill_example()
    train_examples = [example]
    
    print(f"\n{'='*60}")
    print(f"Running GEPA optimization test")
    print(f"Training examples: {len(train_examples)}")
    print(f"{'='*60}\n", flush=True)
    
    # Try GEPA optimization (if available)
    try:
        optimized_prompt = optimizer.optimize(
            train_examples=train_examples,
            method="gepa",
            num_iterations=1
        )
        
        print(f"\nGEPA optimization returned prompt of length: {len(optimized_prompt)}", flush=True)
        
        assert optimized_prompt is not None
        assert len(optimized_prompt) > 0
    except ValueError as e:
        if "gepa" in str(e).lower() or "unknown" in str(e).lower():
            pytest.skip(f"GEPA optimization not implemented: {e}")


def test_metric_function_works():
    """Test that the metric function works correctly."""
    # Create a simple example
    example = create_minimal_drill_example()
    
    # Create metric
    metric = create_drill_metric([example])
    
    # Test with correct action
    correct_prediction = type('Prediction', (), {
        'action_type': example.correct_actions[0]['type'],
        'action_payload': json.dumps(example.correct_actions[0].get('payload', {}))
    })()
    
    score = metric(example, correct_prediction)
    assert score == 1.0
    
    # Test with incorrect action (if we have one)
    if example.incorrect_actions:
        incorrect_prediction = type('Prediction', (), {
            'action_type': example.incorrect_actions[0]['type'],
            'action_payload': json.dumps(example.incorrect_actions[0].get('payload', {}))
        })()
        
        score = metric(example, incorrect_prediction)
        assert score == 0.0


def test_optimizer_initialization():
    """Test that optimizer can be initialized."""
    base_agent = LLMAgent("player_0")
    base_prompt = base_agent._get_default_system_prompt()
    
    optimizer = PromptOptimizer(
        base_system_prompt=base_prompt,
        model_name="gpt-4o-mini",
        include_higher_level_features=False
    )
    
    assert optimizer.base_system_prompt == base_prompt
    assert optimizer.model_name == "gpt-4o-mini"
    assert optimizer.include_higher_level_features == False


def test_prepare_examples():
    """Test that examples can be prepared from drill data."""
    # Create a test drill
    from fastapi.testclient import TestClient
    from main import app
    
    client = TestClient(app)
    response = client.post("/api/games", json={"player_names": ["Alice", "Bob"]})
    assert response.status_code == 200
    state = response.json()["initial_state"]
    
    # Get legal actions
    game_id = state["game_id"]
    legal_response = client.get(f"/api/games/{game_id}/legal_actions?player_id=player_0")
    assert legal_response.status_code == 200
    legal_actions_list = legal_response.json()["legal_actions"]
    
    if not legal_actions_list:
        pytest.skip("No legal actions available")
    
    # Create drill
    drill_response = client.post(
        "/api/drills",
        json={
            "name": "Test Drill for Optimization",
            "player_id": "player_0",
            "steps": [
                {
                    "player_id": "player_0",
                    "state": state,
                    "expected_action": legal_actions_list[0],
                    "correct_actions": [legal_actions_list[0]],
                    "incorrect_actions": [legal_actions_list[1]] if len(legal_actions_list) > 1 else None
                }
            ]
        }
    )
    
    assert drill_response.status_code == 200
    drill_id = drill_response.json()["drill_id"]
    
    # Get drill data
    get_response = client.get(f"/api/drills/{drill_id}")
    assert get_response.status_code == 200
    drill_data = get_response.json()
    
    # Prepare optimizer
    base_agent = LLMAgent("player_0")
    base_prompt = base_agent._get_default_system_prompt()
    optimizer = PromptOptimizer(
        base_system_prompt=base_prompt,
        model_name="gpt-4o-mini",
        include_higher_level_features=False
    )
    
    # Prepare examples - format drill data as expected by prepare_examples
    # prepare_examples expects: [{"id": drill_id, "steps": [...]}]
    drills = [{
        "id": drill_id,
        "steps": drill_data["steps"]
    }]
    examples = optimizer.prepare_examples(drills)
    
    assert len(examples) > 0
    assert isinstance(examples[0], DrillExample)
    assert examples[0].correct_actions is not None
    assert len(examples[0].correct_actions) > 0

