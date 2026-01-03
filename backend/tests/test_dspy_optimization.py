"""
Tests for DSPy prompt optimization functionality.
"""
import pytest
import json
from fastapi.testclient import TestClient
from main import app
from api.database import (
    init_db,
    save_optimized_prompt,
    get_optimized_prompt,
    list_optimized_prompts,
    set_default_prompt,
    delete_optimized_prompt,
    get_default_optimized_prompt,
)
from agents.llm_agent import LLMAgent

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize database before each test."""
    init_db()
    yield
    # Cleanup: remove any test prompts
    try:
        delete_optimized_prompt("test_prompt")
        delete_optimized_prompt("test_prompt_2")
    except:
        pass


def test_save_and_get_optimized_prompt():
    """Test saving and retrieving an optimized prompt."""
    test_prompt = "You are a test agent. Follow these rules."
    metadata = {"test": True, "accuracy": 0.95}
    
    prompt_id = save_optimized_prompt(
        name="test_prompt",
        system_prompt=test_prompt,
        metadata=metadata,
        is_default=False
    )
    
    assert prompt_id > 0
    
    # Retrieve the prompt
    prompt_row = get_optimized_prompt("test_prompt")
    assert prompt_row is not None
    assert prompt_row["name"] == "test_prompt"
    assert prompt_row["system_prompt"] == test_prompt
    assert prompt_row["is_default"] == False
    
    metadata_loaded = json.loads(prompt_row["metadata"])
    assert metadata_loaded["test"] == True
    assert metadata_loaded["accuracy"] == 0.95


def test_list_optimized_prompts():
    """Test listing all optimized prompts."""
    # Create a couple of prompts
    save_optimized_prompt("test_prompt", "Prompt 1", {"test": 1})
    save_optimized_prompt("test_prompt_2", "Prompt 2", {"test": 2})
    
    prompts = list_optimized_prompts()
    assert len(prompts) >= 2
    
    # Check that our prompts are in the list
    names = [p["name"] for p in prompts]
    assert "test_prompt" in names
    assert "test_prompt_2" in names


def test_set_default_prompt():
    """Test setting a prompt as default."""
    save_optimized_prompt("test_prompt", "Prompt 1", {})
    save_optimized_prompt("test_prompt_2", "Prompt 2", {})
    
    # Set first as default
    success = set_default_prompt("test_prompt")
    assert success == True
    
    # Check it's default
    prompt = get_optimized_prompt("test_prompt")
    assert prompt["is_default"] == True
    
    # Set second as default - first should no longer be default
    set_default_prompt("test_prompt_2")
    prompt1 = get_optimized_prompt("test_prompt")
    prompt2 = get_optimized_prompt("test_prompt_2")
    assert prompt1["is_default"] == False
    assert prompt2["is_default"] == True


def test_get_default_optimized_prompt():
    """Test getting the default prompt."""
    save_optimized_prompt("test_prompt", "Prompt 1", {})
    save_optimized_prompt("test_prompt_2", "Prompt 2", {})
    
    # Set one as default
    set_default_prompt("test_prompt")
    
    default = get_default_optimized_prompt()
    assert default is not None
    assert default["name"] == "test_prompt"


def test_delete_optimized_prompt():
    """Test deleting an optimized prompt."""
    save_optimized_prompt("test_prompt", "Prompt 1", {})
    
    # Verify it exists
    prompt = get_optimized_prompt("test_prompt")
    assert prompt is not None
    
    # Delete it
    success = delete_optimized_prompt("test_prompt")
    assert success == True
    
    # Verify it's gone
    prompt = get_optimized_prompt("test_prompt")
    assert prompt is None


def test_llm_agent_loads_custom_prompt():
    """Test that LLMAgent can load a custom prompt from database."""
    test_prompt = "Custom test prompt for LLM agent."
    save_optimized_prompt("test_prompt", test_prompt, {})
    
    # Create agent with custom prompt
    agent = LLMAgent("player_0", prompt_name="test_prompt")
    
    # Get system prompt - should be the custom one
    system_prompt = agent._get_system_prompt()
    assert system_prompt == test_prompt


def test_llm_agent_falls_back_to_default():
    """Test that LLMAgent falls back to default prompt if name not found."""
    test_prompt = "Default test prompt."
    save_optimized_prompt("test_prompt", test_prompt, {}, is_default=True)
    
    # Create agent with non-existent prompt name
    agent = LLMAgent("player_0", prompt_name="nonexistent_prompt")
    
    # Should fall back to default
    system_prompt = agent._get_system_prompt()
    assert system_prompt == test_prompt


def test_llm_agent_uses_hardcoded_default_if_no_prompts():
    """Test that LLMAgent uses hardcoded default if no prompts in database."""
    # Create agent without any prompts in database
    agent = LLMAgent("player_0")
    
    # Should use hardcoded default
    system_prompt = agent._get_system_prompt()
    assert system_prompt is not None
    assert len(system_prompt) > 0
    assert "Catan" in system_prompt or "catan" in system_prompt.lower()


def test_api_list_prompts_endpoint():
    """Test the API endpoint for listing prompts."""
    save_optimized_prompt("test_prompt", "Prompt 1", {"test": 1})
    
    response = client.get("/api/prompts")
    assert response.status_code == 200
    data = response.json()
    assert "prompts" in data
    assert len(data["prompts"]) >= 1
    
    # Check our prompt is in the list
    names = [p["name"] for p in data["prompts"]]
    assert "test_prompt" in names


def test_api_get_prompt_endpoint():
    """Test the API endpoint for getting a specific prompt."""
    test_prompt = "Test prompt content"
    save_optimized_prompt("test_prompt", test_prompt, {"test": True})
    
    response = client.get("/api/prompts/test_prompt")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_prompt"
    assert data["system_prompt"] == test_prompt


def test_api_create_prompt_endpoint():
    """Test the API endpoint for creating a prompt."""
    response = client.post(
        "/api/prompts",
        json={
            "name": "test_prompt",
            "system_prompt": "Test prompt content",
            "metadata": {"test": True},
            "is_default": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["name"] == "test_prompt"
    
    # Verify it was saved
    prompt = get_optimized_prompt("test_prompt")
    assert prompt is not None
    assert prompt["system_prompt"] == "Test prompt content"


def test_api_set_default_prompt_endpoint():
    """Test the API endpoint for setting default prompt."""
    save_optimized_prompt("test_prompt", "Prompt 1", {})
    
    response = client.put("/api/prompts/test_prompt/set_default")
    assert response.status_code == 200
    
    # Verify it's default
    prompt = get_optimized_prompt("test_prompt")
    assert prompt["is_default"] == True


def test_api_delete_prompt_endpoint():
    """Test the API endpoint for deleting a prompt."""
    save_optimized_prompt("test_prompt", "Prompt 1", {})
    
    response = client.delete("/api/prompts/test_prompt")
    assert response.status_code == 200
    
    # Verify it's deleted
    prompt = get_optimized_prompt("test_prompt")
    assert prompt is None

