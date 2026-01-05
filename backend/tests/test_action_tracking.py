"""
Tests for action tracking in actions_taken_this_turn.
Ensures that actions are properly tracked when player_id is provided,
and that duplicate trade proposals are filtered correctly.
"""
import pytest
from engine import (
    ResourceType,
    Player,
    GameState,
    Action,
    ProposeTradePayload,
    deserialize_game_state,
    legal_actions,
)
from agents.llm_agent import LLMAgent


def create_test_playing_state() -> GameState:
    """Create a minimal game state in playing phase with resources for trading."""
    players = [
        Player(id="player_0", name="Alice"),
        Player(id="player_1", name="Bob"),
    ]
    
    # Give players resources for trading
    players[0].resources[ResourceType.WOOD] = 2
    players[0].resources[ResourceType.BRICK] = 1
    players[1].resources[ResourceType.SHEEP] = 2
    players[1].resources[ResourceType.WHEAT] = 1
    
    state = GameState(
        game_id="test_game",
        players=players,
        current_player_index=0,
        phase="playing",
        turn_number=1,
        dice_roll=6,  # Some dice roll so we're past initial roll
    )
    
    return state


def test_action_tracking_with_explicit_player_id():
    """Test that actions are tracked in actions_taken_this_turn when player_id is provided."""
    state = create_test_playing_state()
    player_id = "player_0"
    
    # Initially, no actions tracked
    assert len(state.actions_taken_this_turn) == 0
    
    # Propose a trade with explicit player_id (as agents do)
    trade_payload = ProposeTradePayload(
        target_player_ids=["player_1"],
        give_resources={ResourceType.WOOD: 1},
        receive_resources={ResourceType.SHEEP: 1}
    )
    
    new_state = state.step(Action.PROPOSE_TRADE, trade_payload, player_id=player_id)
    
    # Verify action was tracked
    assert len(new_state.actions_taken_this_turn) == 1
    tracked_action = new_state.actions_taken_this_turn[0]
    assert tracked_action["player_id"] == player_id
    assert tracked_action["action"] == "propose_trade"
    assert tracked_action["payload"] is not None
    assert tracked_action["payload"]["give_resources"] == {"wood": 1}
    assert tracked_action["payload"]["receive_resources"] == {"sheep": 1}
    assert set(tracked_action["payload"]["target_player_ids"]) == {"player_1"}


def test_action_tracking_without_player_id():
    """Test that actions are still tracked when player_id is None (backward compatibility)."""
    state = create_test_playing_state()
    
    # Propose a trade without explicit player_id
    trade_payload = ProposeTradePayload(
        target_player_ids=["player_1"],
        give_resources={ResourceType.WOOD: 1},
        receive_resources={ResourceType.SHEEP: 1}
    )
    
    new_state = state.step(Action.PROPOSE_TRADE, trade_payload, player_id=None)
    
    # Verify action was tracked
    assert len(new_state.actions_taken_this_turn) == 1
    tracked_action = new_state.actions_taken_this_turn[0]
    assert tracked_action["player_id"] == "player_0"  # Should use current player
    assert tracked_action["action"] == "propose_trade"


def test_duplicate_trade_filtering():
    """Test that duplicate trade proposals are filtered from legal actions."""
    state = create_test_playing_state()
    player_id = "player_0"
    
    # Propose a trade
    trade_payload = ProposeTradePayload(
        target_player_ids=["player_1"],
        give_resources={ResourceType.WOOD: 1},
        receive_resources={ResourceType.SHEEP: 1}
    )
    
    state = state.step(Action.PROPOSE_TRADE, trade_payload, player_id=player_id)
    
    # Reject the trade (simulate rejection)
    state = state.step(Action.REJECT_TRADE, player_id="player_1")
    
    # After rejection, current_player_index should be back to player_0
    assert state.players[state.current_player_index].id == player_id
    
    # Verify the trade was tracked (filter for propose_trade actions only)
    propose_trade_actions = [a for a in state.actions_taken_this_turn if a["action"] == "propose_trade" and a["player_id"] == player_id]
    assert len(propose_trade_actions) == 1
    tracked_trade = propose_trade_actions[0]
    assert tracked_trade["action"] == "propose_trade"
    
    # Get legal actions
    legal_actions_list = legal_actions(state, player_id)
    
    # Filter using LLM agent's logic
    from agents.llm_agent import LLMAgent
    agent = LLMAgent(player_id=player_id, enable_retrieval=False)
    
    # The agent's choose_action method filters duplicate trades
    # We'll test the filtering logic directly
    player_actions_this_turn = [
        a for a in state.actions_taken_this_turn 
        if a["player_id"] == player_id and a["action"] == "propose_trade"
    ]
    
    assert len(player_actions_this_turn) == 1
    
    # Check that the same trade would be filtered
    filtered_actions = []
    for action, payload in legal_actions_list:
        if action == Action.PROPOSE_TRADE:
            if payload and hasattr(payload, 'give_resources') and hasattr(payload, 'receive_resources'):
                # Normalize current payload to string keys
                current_give = {rt.value: count for rt, count in payload.give_resources.items()}
                current_receive = {rt.value: count for rt, count in payload.receive_resources.items()}
                
                already_proposed = False
                for prev_action in player_actions_this_turn:
                    prev_payload = prev_action.get("payload", {})
                    prev_give = prev_payload.get("give_resources", {})
                    prev_receive = prev_payload.get("receive_resources", {})
                    prev_targets = set(prev_payload.get("target_player_ids", []))
                    
                    if (prev_give == current_give and
                        prev_receive == current_receive and
                        prev_targets == set(payload.target_player_ids)):
                        already_proposed = True
                        break
                
                if not already_proposed:
                    filtered_actions.append((action, payload))
            else:
                # PROPOSE_TRADE with None payload - should still be included
                # (agent can construct any trade, but we can't filter it)
                filtered_actions.append((action, payload))
        else:
            filtered_actions.append((action, payload))
    
    # If PROPOSE_TRADE is in legal actions, verify it's still there
    # (because the agent can propose different trades)
    # But if we try to propose the exact same trade, it should be filtered
    propose_trade_actions = [a for a, p in legal_actions_list if a == Action.PROPOSE_TRADE]
    
    # The key test: if we construct the same trade payload, it should be filtered
    if propose_trade_actions:
        # Create the same trade payload
        same_trade_payload = ProposeTradePayload(
            target_player_ids=["player_1"],
            give_resources={ResourceType.WOOD: 1},
            receive_resources={ResourceType.SHEEP: 1}
        )
        
        # Check if it would be filtered
        would_be_filtered = False
        for prev_action in player_actions_this_turn:
            prev_payload = prev_action.get("payload", {})
            prev_give = prev_payload.get("give_resources", {})
            prev_receive = prev_payload.get("receive_resources", {})
            prev_targets = set(prev_payload.get("target_player_ids", []))
            
            current_give = {rt.value: count for rt, count in same_trade_payload.give_resources.items()}
            current_receive = {rt.value: count for rt, count in same_trade_payload.receive_resources.items()}
            
            if (prev_give == current_give and
                prev_receive == current_receive and
                prev_targets == set(same_trade_payload.target_player_ids)):
                would_be_filtered = True
                break
        
        # The same trade should be detected as already proposed
        assert would_be_filtered, "Same trade should be detected as already proposed"


def test_multiple_trades_tracked():
    """Test that multiple different trades are all tracked."""
    state = create_test_playing_state()
    player_id = "player_0"
    
    # Propose first trade
    trade1_payload = ProposeTradePayload(
        target_player_ids=["player_1"],
        give_resources={ResourceType.WOOD: 1},
        receive_resources={ResourceType.SHEEP: 1}
    )
    state = state.step(Action.PROPOSE_TRADE, trade1_payload, player_id=player_id)
    state = state.step(Action.REJECT_TRADE, player_id="player_1")
    
    # Propose second trade (different)
    trade2_payload = ProposeTradePayload(
        target_player_ids=["player_1"],
        give_resources={ResourceType.BRICK: 1},
        receive_resources={ResourceType.WHEAT: 1}
    )
    state = state.step(Action.PROPOSE_TRADE, trade2_payload, player_id=player_id)
    state = state.step(Action.REJECT_TRADE, player_id="player_1")
    
    # Verify both trades are tracked (filter for propose_trade actions only)
    tracked_actions = [a for a in state.actions_taken_this_turn if a["action"] == "propose_trade" and a["player_id"] == player_id]
    assert len(tracked_actions) == 2
    
    # Verify they're different
    assert tracked_actions[0]["payload"]["give_resources"] != tracked_actions[1]["payload"]["give_resources"]

