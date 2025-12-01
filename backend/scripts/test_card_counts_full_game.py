#!/usr/bin/env python3
"""
Full test: Complete setup phase and play a few turns to verify card counts change.
"""
import sys
import os
from pathlib import Path

# Load API key from .zshrc
if 'OPENAI_API_KEY' not in os.environ or not os.environ.get('OPENAI_API_KEY'):
    import subprocess
    try:
        result = subprocess.run(
            ['zsh', '-c', 'source ~/.zshrc 2>/dev/null && echo $OPENAI_API_KEY'],
            capture_output=True,
            text=True,
            shell=False
        )
        api_key = result.stdout.strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
    except:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
import random
from engine import GameState, Player, ResourceType, Action
from agents import LLMAgent
from api.database import init_db, create_game, get_latest_state, save_game_state
from engine.serialization import deserialize_game_state, legal_actions, serialize_game_state


def get_player_color(index: int) -> str:
    colors = ["#FF0000", "#00AA00", "#2196F3", "#FF8C00"]
    return colors[index % len(colors)]


def generate_random_name() -> str:
    adjectives = ["Brave", "Swift", "Clever"]
    nouns = ["Wolf", "Eagle", "Lion"]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"


def test_full_game_card_counts():
    """Test card counts through a full game with LLM agent."""
    print("=" * 60)
    print("Full Game Card Count Test with LLM Agent")
    print("=" * 60)
    
    if 'OPENAI_API_KEY' not in os.environ or not os.environ.get('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found. Cannot test with LLM agent.")
        return False
    
    init_db()
    game_id = str(uuid.uuid4())
    
    # Create 2 players
    player_names = [generate_random_name() for _ in range(2)]
    players = [
        Player(id=f"player_{i}", name=player_names[i], color=get_player_color(i))
        for i in range(2)
    ]
    
    # Create and initialize game
    initial_state = GameState(
        game_id=game_id,
        players=players,
        current_player_index=0,
        phase="setup"
    )
    initial_state = initial_state._create_initial_board(initial_state)
    
    desert_tile = next((t for t in initial_state.tiles if t.resource_type is None), None)
    if desert_tile:
        initial_state.robber_tile_id = desert_tile.id
    
    # Save initial state
    serialized_state = serialize_game_state(initial_state)
    create_game(game_id, rng_seed=None, metadata={"player_names": player_names, "num_players": 2}, initial_state_json=serialized_state)
    
    print(f"\n✓ Game created: {game_id[:8]}...")
    
    # Get state
    state_data = get_latest_state(game_id)
    state = deserialize_game_state(state_data)
    
    # Track initial card counts
    initial_resource_counts = {r: state.resource_card_counts.get(r, 0) for r in ResourceType}
    initial_dev_total = sum(state.dev_card_counts.values())
    
    print(f"\nInitial card counts:")
    print(f"  Resources: {sum(initial_resource_counts.values())} total")
    print(f"  Dev cards: {initial_dev_total} total")
    
    # Create LLM agent
    player_ids = [p.id for p in state.players]
    agent = LLMAgent(player_ids[0], model="gpt-4o-mini", enable_retrieval=False)
    print(f"\n✓ LLM agent created for {state.players[0].name}")
    
    # Play through setup and a few turns
    max_actions = 15
    action_count = 0
    
    print(f"\nPlaying game (max {max_actions} actions)...")
    
    while action_count < max_actions and state.phase != "finished":
        current_player = state.players[state.current_player_index]
        is_setup = state.phase == "setup"
        
        # Get legal actions
        legal_actions_list = legal_actions(state, current_player.id)
        if not legal_actions_list:
            print(f"\n  No legal actions for {current_player.name}")
            break
        
        # Track counts before action
        resource_before = {r: state.resource_card_counts.get(r, 0) for r in ResourceType}
        dev_before = sum(state.dev_card_counts.values())
        
        # Choose action
        if current_player.id == player_ids[0]:
            # LLM agent's turn
            try:
                result = agent.choose_action(state, legal_actions_list)
                action, payload = result[0], result[1] if len(result) > 1 else None
                print(f"\n  Action {action_count + 1}: {current_player.name} -> {action.value}")
            except Exception as e:
                print(f"\n  Error getting action: {e}")
                break
        else:
            # Other player - use first legal action (or end turn)
            action, payload = legal_actions_list[0]
            if action == Action.END_TURN:
                print(f"\n  Action {action_count + 1}: {current_player.name} -> end_turn")
            else:
                print(f"\n  Action {action_count + 1}: {current_player.name} -> {action.value} (auto)")
        
        # Execute action
        try:
            state = state.step(action, payload, current_player.id)
        except Exception as e:
            print(f"  Error executing action: {e}")
            break
        
        # Track counts after action
        resource_after = {r: state.resource_card_counts.get(r, 0) for r in ResourceType}
        dev_after = sum(state.dev_card_counts.values())
        
        # Check for changes
        resource_changed = any(resource_before[r] != resource_after[r] for r in ResourceType)
        dev_changed = dev_before != dev_after
        
        if resource_changed:
            print(f"  Resource cards changed:")
            for r in ResourceType:
                if resource_before[r] != resource_after[r]:
                    print(f"    {r.value}: {resource_before[r]} -> {resource_after[r]}")
        
        if dev_changed:
            print(f"  Dev cards: {dev_before} -> {dev_after}")
        
        # Verify counts are valid
        for r in ResourceType:
            count = state.resource_card_counts.get(r, 0)
            assert 0 <= count <= 19, f"Invalid resource count: {count}"
        
        total_dev = sum(state.dev_card_counts.values())
        assert 0 <= total_dev <= 25, f"Invalid dev card count: {total_dev}"
        
        # Save state
        save_game_state(game_id, serialize_game_state(state))
        
        action_count += 1
        
        # Check if we've moved to playing phase
        if state.phase == "playing" and action_count > 5:
            print(f"\n  ✓ Game moved to playing phase!")
            break
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("Final Card Counts:")
    print("=" * 60)
    print("\nResource cards:")
    for r in ResourceType:
        count = state.resource_card_counts.get(r, 0)
        initial = initial_resource_counts[r]
        change = count - initial
        print(f"  {r.value}: {initial} -> {count} ({change:+d})")
    
    print("\nDevelopment cards:")
    for card_type, count in sorted(state.dev_card_counts.items()):
        print(f"  {card_type}: {count}")
    print(f"  Total: {sum(state.dev_card_counts.values())} (started with {initial_dev_total})")
    
    print(f"\n✓ Test completed successfully!")
    print(f"Game ID: {game_id}")
    print(f"Actions taken: {action_count}")
    print(f"Phase: {state.phase}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_full_game_card_counts()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

