#!/usr/bin/env python3
"""
Test script to verify card counts work with LLM agent taking actions.
"""
import sys
import os
from pathlib import Path

# Load environment variables from ~/.zshrc if OPENAI_API_KEY not set
if 'OPENAI_API_KEY' not in os.environ or not os.environ.get('OPENAI_API_KEY'):
    import subprocess
    try:
        # Use zsh to source .zshrc and get the API key
        result = subprocess.run(
            ['zsh', '-c', 'source ~/.zshrc 2>/dev/null && echo $OPENAI_API_KEY'],
            capture_output=True,
            text=True,
            shell=False
        )
        api_key = result.stdout.strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            print(f"Loaded API key from .zshrc: {api_key[:15]}...")
    except Exception as e:
        print(f"Warning: Could not load API key from .zshrc: {e}")

# Verify API key is set
if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY']:
    print(f"Using API key: {os.environ['OPENAI_API_KEY'][:15]}...")
else:
    print("Warning: OPENAI_API_KEY not set")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
import random
from engine import GameState, Player, ResourceType
from agents import LLMAgent
from api.database import init_db, create_game, get_latest_state, save_game_state
from engine.serialization import deserialize_game_state, legal_actions, serialize_game_state


def get_player_color(index: int) -> str:
    """Get color for player by index."""
    colors = ["#FF0000", "#00AA00", "#2196F3", "#FF8C00"]
    return colors[index % len(colors)]


def generate_random_name() -> str:
    """Generate a random player name."""
    adjectives = ["Brave", "Swift", "Clever"]
    nouns = ["Wolf", "Eagle", "Lion"]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"


def test_card_counts_with_llm():
    """Test card counts with LLM agent taking actions."""
    print("=" * 60)
    print("Testing Card Counts with LLM Agent")
    print("=" * 60)
    
    # Initialize database
    init_db()
    
    # Generate game ID
    game_id = str(uuid.uuid4())
    
    # Create players
    player_names = [generate_random_name() for _ in range(2)]
    players = [
        Player(id=f"player_{i}", name=player_names[i], color=get_player_color(i))
        for i in range(2)
    ]
    
    # Create initial game state
    initial_state = GameState(
        game_id=game_id,
        players=players,
        current_player_index=0,
        phase="setup"
    )
    
    # Initialize the board
    initial_state = initial_state._create_initial_board(initial_state)
    
    # Initialize robber on desert tile
    desert_tile = next((t for t in initial_state.tiles if t.resource_type is None), None)
    if desert_tile:
        initial_state.robber_tile_id = desert_tile.id
    
    # Verify initial card counts
    print("\n1. Verifying initial card counts...")
    for resource in ResourceType:
        count = initial_state.resource_card_counts.get(resource, 0)
        assert count == 19, f"Expected 19 {resource.value} cards, got {count}"
    print("✓ All resource cards initialized to 19")
    
    expected_dev_counts = {
        "year_of_plenty": 2,
        "monopoly": 2,
        "road_building": 2,
        "victory_point": 5,
        "knight": 14,
    }
    for card_type, expected_count in expected_dev_counts.items():
        count = initial_state.dev_card_counts.get(card_type, 0)
        assert count == expected_count, f"Expected {expected_count} {card_type} cards, got {count}"
    print("✓ All development cards initialized correctly")
    
    # Serialize and save to database
    serialized_state = serialize_game_state(initial_state)
    metadata = {
        "player_names": player_names,
        "num_players": 2,
    }
    create_game(
        game_id,
        rng_seed=None,
        metadata=metadata,
        initial_state_json=serialized_state,
    )
    
    print(f"\n2. Created game: {game_id[:8]}...")
    
    # Get initial state
    state_data = get_latest_state(game_id)
    state = deserialize_game_state(state_data)
    
    # Create LLM agent for first player
    print("\n3. Creating LLM agent...")
    player_ids = [p.id for p in state.players]
    
    # Check if API key is available
    if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY']:
        print("❌ OPENAI_API_KEY not found. Cannot test with LLM agent.")
        return False
    
    try:
        agent = LLMAgent(player_ids[0], model="gpt-4o-mini", enable_retrieval=False)
        print(f"✓ Agent created for {state.players[0].name}")
    except Exception as e:
        print(f"❌ Failed to create LLM agent: {e}")
        return False
    
    # Take a few actions to test card counts
    print("\n4. Taking actions to test card counts...")
    max_actions = 5
    action_count = 0
    
    while action_count < max_actions and state.phase != "finished":
        current_player = state.players[state.current_player_index]
        
        # Only let the LLM agent act if it's their turn
        if current_player.id != player_ids[0]:
            print(f"   Turn {action_count + 1}: {current_player.name}'s turn (skipping)")
            # For simplicity, just end turn for other player
            state = state.step(state.Action.END_TURN)
            action_count += 1
            continue
        
        print(f"\n   Turn {action_count + 1}: {current_player.name}'s turn")
        
        # Get legal actions
        legal_actions_list = legal_actions(state, current_player.id)
        if not legal_actions_list:
            print("   No legal actions available")
            break
        
        # Track card counts before action
        resource_counts_before = {r: state.resource_card_counts.get(r, 0) for r in ResourceType}
        dev_counts_before = sum(state.dev_card_counts.values())
        
        # Agent chooses action
        try:
            print(f"   Getting action from LLM agent...")
            result = agent.choose_action(state, legal_actions_list)
            if len(result) >= 2:
                action, payload = result[0], result[1]
            else:
                action, payload = result[0], None
            
            print(f"   ✓ Action chosen: {action.value}")
            
            # Execute action
            state = state.step(action, payload, current_player.id)
            
            # Track card counts after action
            resource_counts_after = {r: state.resource_card_counts.get(r, 0) for r in ResourceType}
            dev_counts_after = sum(state.dev_card_counts.values())
            
            # Check if card counts changed
            resource_changed = any(
                resource_counts_before[r] != resource_counts_after[r]
                for r in ResourceType
            )
            dev_changed = dev_counts_before != dev_counts_after
            
            if resource_changed:
                print("   Resource card counts changed:")
                for r in ResourceType:
                    before = resource_counts_before[r]
                    after = resource_counts_after[r]
                    if before != after:
                        print(f"     {r.value}: {before} -> {after}")
            
            if dev_changed:
                print(f"   Dev card counts: {dev_counts_before} -> {dev_counts_after}")
            
            # Verify counts are still valid
            for r in ResourceType:
                count = state.resource_card_counts.get(r, 0)
                assert 0 <= count <= 19, f"Invalid resource count for {r.value}: {count}"
            
            total_dev = sum(state.dev_card_counts.values())
            assert 0 <= total_dev <= 25, f"Invalid dev card count: {total_dev}"
            
            # Save state
            state_json = serialize_game_state(state)
            save_game_state(game_id, state_json)
            
        except Exception as e:
            print(f"   Error: {e}")
            break
        
        action_count += 1
    
    print("\n5. Final card counts:")
    print("   Resource cards:")
    for r in ResourceType:
        count = state.resource_card_counts.get(r, 0)
        print(f"     {r.value}: {count}")
    print("   Development cards:")
    for card_type, count in state.dev_card_counts.items():
        print(f"     {card_type}: {count}")
    
    print("\n" + "=" * 60)
    print("✓ Card count test with LLM agent completed!")
    print("=" * 60)
    print(f"\nGame ID: {game_id}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_card_counts_with_llm()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

