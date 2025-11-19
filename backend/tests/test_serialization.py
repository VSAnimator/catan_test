"""
Tests for serialization and LLM-friendly text conversion.
"""
import pytest
import json
from engine import (
    ResourceType,
    Tile,
    NumberToken,
    Intersection,
    RoadEdge,
    Player,
    GameState,
    Action,
    BuildSettlementPayload,
    BuildRoadPayload,
    BuildCityPayload,
    PlayDevCardPayload,
    TradeBankPayload,
    TradePlayerPayload,
    serialize_game_state,
    deserialize_game_state,
    serialize_action,
    deserialize_action,
    serialize_action_payload,
    deserialize_action_payload,
    legal_actions,
    state_to_text,
    legal_actions_to_text,
    parse_action_from_text,
)


def test_serialize_deserialize_game_state_roundtrip():
    """Test that GameState can be serialized and deserialized correctly."""
    # Create a game state
    players = [
        Player(id="player_0", name="Alice"),
        Player(id="player_1", name="Bob"),
    ]
    
    tiles = [
        Tile(id=0, resource_type=ResourceType.WOOD, number_token=NumberToken(5), position=(0, 0)),
        Tile(id=1, resource_type=ResourceType.BRICK, number_token=NumberToken(6), position=(1, 0)),
    ]
    
    intersections = [
        Intersection(
            id=0,
            position=(0.0, 0.0),
            adjacent_tiles={0},
            adjacent_intersections={1},
            owner="player_0",
            building_type="settlement"
        ),
        Intersection(
            id=1,
            position=(1.0, 0.0),
            adjacent_tiles={1},
            adjacent_intersections={0},
            owner=None,
            building_type=None
        ),
    ]
    
    road_edges = [
        RoadEdge(id=0, intersection1_id=0, intersection2_id=1, owner="player_0"),
        RoadEdge(id=1, intersection1_id=1, intersection2_id=2, owner=None),
    ]
    
    original_state = GameState(
        game_id="test_game",
        players=players,
        current_player_index=0,
        phase="playing",
        tiles=tiles,
        intersections=intersections,
        road_edges=road_edges,
        dice_roll=7,
        turn_number=5,
        setup_round=0,
        setup_phase_player_index=0,
    )
    
    # Serialize
    serialized = serialize_game_state(original_state)
    
    # Verify it's JSON-serializable
    json_str = json.dumps(serialized)
    assert json_str is not None
    
    # Deserialize
    deserialized_state = deserialize_game_state(serialized)
    
    # Verify all fields match
    assert deserialized_state.game_id == original_state.game_id
    assert deserialized_state.current_player_index == original_state.current_player_index
    assert deserialized_state.phase == original_state.phase
    assert deserialized_state.dice_roll == original_state.dice_roll
    assert deserialized_state.turn_number == original_state.turn_number
    assert deserialized_state.setup_round == original_state.setup_round
    assert deserialized_state.setup_phase_player_index == original_state.setup_phase_player_index
    
    # Verify players
    assert len(deserialized_state.players) == len(original_state.players)
    for i, (orig, deser) in enumerate(zip(original_state.players, deserialized_state.players)):
        assert orig.id == deser.id
        assert orig.name == deser.name
        assert orig.victory_points == deser.victory_points
        for rt in ResourceType:
            assert orig.resources[rt] == deser.resources[rt]
    
    # Verify tiles
    assert len(deserialized_state.tiles) == len(original_state.tiles)
    for orig, deser in zip(original_state.tiles, deserialized_state.tiles):
        assert orig.id == deser.id
        assert orig.resource_type == deser.resource_type
        assert orig.number_token == deser.number_token
        assert orig.position == deser.position
    
    # Verify intersections
    assert len(deserialized_state.intersections) == len(original_state.intersections)
    for orig, deser in zip(original_state.intersections, deserialized_state.intersections):
        assert orig.id == deser.id
        assert orig.owner == deser.owner
        assert orig.building_type == deser.building_type
        assert orig.adjacent_tiles == deser.adjacent_tiles
    
    # Verify road edges
    assert len(deserialized_state.road_edges) == len(original_state.road_edges)
    for orig, deser in zip(original_state.road_edges, deserialized_state.road_edges):
        assert orig.id == deser.id
        assert orig.owner == deser.owner
        assert orig.intersection1_id == deser.intersection1_id
        assert orig.intersection2_id == deser.intersection2_id


def test_serialize_deserialize_action_roundtrip():
    """Test that Action can be serialized and deserialized."""
    for action in Action:
        serialized = serialize_action(action)
        deserialized = deserialize_action(serialized)
        assert deserialized == action


def test_serialize_deserialize_action_payload_roundtrip():
    """Test that ActionPayload can be serialized and deserialized."""
    payloads = [
        BuildRoadPayload(road_edge_id=5),
        BuildSettlementPayload(intersection_id=10),
        BuildCityPayload(intersection_id=15),
        PlayDevCardPayload(card_type="knight"),
        TradeBankPayload(
            give_resource=ResourceType.WOOD,
            give_amount=4,
            receive_resource=ResourceType.BRICK,
            receive_amount=1,
        ),
        TradePlayerPayload(
            other_player_id="player_1",
            give_resource=ResourceType.WHEAT,
            give_amount=2,
            receive_resource=ResourceType.ORE,
            receive_amount=2,
        ),
    ]
    
    for payload in payloads:
        serialized = serialize_action_payload(payload)
        deserialized = deserialize_action_payload(serialized)
        
        if isinstance(payload, BuildRoadPayload):
            assert isinstance(deserialized, BuildRoadPayload)
            assert deserialized.road_edge_id == payload.road_edge_id
        elif isinstance(payload, BuildSettlementPayload):
            assert isinstance(deserialized, BuildSettlementPayload)
            assert deserialized.intersection_id == payload.intersection_id
        elif isinstance(payload, BuildCityPayload):
            assert isinstance(deserialized, BuildCityPayload)
            assert deserialized.intersection_id == payload.intersection_id
        elif isinstance(payload, PlayDevCardPayload):
            assert isinstance(deserialized, PlayDevCardPayload)
            assert deserialized.card_type == payload.card_type
        elif isinstance(payload, TradeBankPayload):
            assert isinstance(deserialized, TradeBankPayload)
            assert deserialized.give_resource == payload.give_resource
            assert deserialized.give_amount == payload.give_amount
            assert deserialized.receive_resource == payload.receive_resource
            assert deserialized.receive_amount == payload.receive_amount
        elif isinstance(payload, TradePlayerPayload):
            assert isinstance(deserialized, TradePlayerPayload)
            assert deserialized.other_player_id == payload.other_player_id
            assert deserialized.give_resource == payload.give_resource
            assert deserialized.give_amount == payload.give_amount
            assert deserialized.receive_resource == payload.receive_resource
            assert deserialized.receive_amount == payload.receive_amount


def test_state_to_text_deterministic():
    """Test that state_to_text produces deterministic output."""
    players = [
        Player(id="player_0", name="Alice"),
        Player(id="player_1", name="Bob"),
    ]
    
    state = GameState(
        game_id="test_game",
        players=players,
        current_player_index=0,
        phase="playing",
        dice_roll=7,
        turn_number=3,
    )
    
    # Generate text twice
    text1 = state_to_text(state, "player_0")
    text2 = state_to_text(state, "player_0")
    
    # Should be identical
    assert text1 == text2


def test_state_to_text_concise():
    """Test that state_to_text produces reasonably concise output."""
    players = [
        Player(id="player_0", name="Alice"),
        Player(id="player_1", name="Bob"),
    ]
    
    state = GameState(
        game_id="test_game",
        players=players,
        current_player_index=0,
        phase="playing",
        dice_roll=7,
        turn_number=3,
    )
    
    text = state_to_text(state, "player_0")
    
    # Should not be excessively long (arbitrary threshold: 5000 chars)
    assert len(text) < 5000
    
    # Should contain key information
    assert "player_0" in text or "Alice" in text
    assert "test_game" in text
    assert "playing" in text


def test_state_to_text_includes_history():
    """Test that state_to_text includes action history."""
    players = [
        Player(id="player_0", name="Alice"),
    ]
    
    state = GameState(
        game_id="test_game",
        players=players,
        current_player_index=0,
        phase="playing",
    )
    
    history = [
        (Action.ROLL_DICE, None),
        (Action.BUILD_SETTLEMENT, BuildSettlementPayload(intersection_id=5)),
    ]
    
    text = state_to_text(state, "player_0", history)
    
    # Should mention recent actions
    assert "Recent Actions" in text or "recent" in text.lower()


def test_legal_actions_basic():
    """Test that legal_actions returns actions for current player."""
    players = [
        Player(
            id="player_0",
            name="Alice",
            resources={
                ResourceType.WOOD: 1,
                ResourceType.BRICK: 1,
                ResourceType.WHEAT: 1,
                ResourceType.SHEEP: 1,
                ResourceType.ORE: 0,
            }
        ),
        Player(id="player_1", name="Bob"),
    ]
    
    # Create a simple board
    tile = Tile(id=0, resource_type=ResourceType.WOOD, number_token=NumberToken(5), position=(0, 0))
    intersection = Intersection(
        id=0,
        position=(0.0, 0.0),
        adjacent_tiles={0},
        adjacent_intersections=set(),
        owner=None,
        building_type=None
    )
    road_edge = RoadEdge(id=0, intersection1_id=0, intersection2_id=1, owner=None)
    
    state = GameState(
        game_id="test_game",
        players=players,
        current_player_index=0,
        phase="playing",
        tiles=[tile],
        intersections=[intersection],
        road_edges=[road_edge],
        dice_roll=7,  # Already rolled
    )
    
    # Get legal actions for current player
    actions = legal_actions(state, "player_0")
    
    # Should have some actions
    assert len(actions) > 0
    
    # Should include END_TURN
    action_types = [a[0] for a in actions]
    assert Action.END_TURN in action_types


def test_legal_actions_to_text_format():
    """Test that legal_actions_to_text produces readable output."""
    actions = [
        (Action.ROLL_DICE, None),
        (Action.BUILD_SETTLEMENT, BuildSettlementPayload(intersection_id=5)),
        (Action.END_TURN, None),
    ]
    
    text = legal_actions_to_text(actions)
    
    # Should be non-empty
    assert len(text) > 0
    
    # Should contain action names
    assert "Roll Dice" in text or "roll" in text.lower()
    assert "End Turn" in text or "end" in text.lower()


def test_parse_action_from_text_build_settlement():
    """Test parsing 'BUILD_SETTLEMENT at intersection 17' from text."""
    actions = [
        (Action.BUILD_SETTLEMENT, BuildSettlementPayload(intersection_id=17)),
        (Action.BUILD_SETTLEMENT, BuildSettlementPayload(intersection_id=18)),
        (Action.END_TURN, None),
    ]
    
    # Test exact match
    result = parse_action_from_text("BUILD_SETTLEMENT at intersection 17", actions)
    assert result[0] == Action.BUILD_SETTLEMENT
    assert result[1] is not None
    assert isinstance(result[1], BuildSettlementPayload)
    assert result[1].intersection_id == 17
    
    # Test case-insensitive
    result = parse_action_from_text("build settlement at intersection 17", actions)
    assert result[0] == Action.BUILD_SETTLEMENT
    assert result[1].intersection_id == 17
    
    # Test with variations
    result = parse_action_from_text("I want to build a settlement at intersection 17", actions)
    assert result[0] == Action.BUILD_SETTLEMENT
    assert result[1].intersection_id == 17


def test_parse_action_from_text_simple_actions():
    """Test parsing simple actions without payloads."""
    actions = [
        (Action.ROLL_DICE, None),
        (Action.END_TURN, None),
        (Action.BUY_DEV_CARD, None),
    ]
    
    # Test roll dice
    result = parse_action_from_text("roll dice", actions)
    assert result[0] == Action.ROLL_DICE
    assert result[1] is None
    
    # Test end turn
    result = parse_action_from_text("end turn", actions)
    assert result[0] == Action.END_TURN
    assert result[1] is None
    
    # Test with variations
    result = parse_action_from_text("I'll roll the dice now", actions)
    assert result[0] == Action.ROLL_DICE


def test_parse_action_from_text_fuzzy_matching():
    """Test that fuzzy matching works for variations."""
    actions = [
        (Action.BUILD_ROAD, BuildRoadPayload(road_edge_id=5)),
        (Action.BUILD_SETTLEMENT, BuildSettlementPayload(intersection_id=10)),
        (Action.END_TURN, None),
    ]
    
    # Test with typos and variations
    result = parse_action_from_text("build road on edge 5", actions)
    assert result[0] == Action.BUILD_ROAD
    assert result[1].road_edge_id == 5
    
    result = parse_action_from_text("settlement at 10", actions)
    assert result[0] == Action.BUILD_SETTLEMENT
    assert result[1].intersection_id == 10


def test_parse_action_from_text_no_match_raises():
    """Test that parse_action_from_text raises error when no good match."""
    actions = [
        (Action.ROLL_DICE, None),
    ]
    
    # Should raise error for completely unrelated text
    with pytest.raises(ValueError, match="Could not parse"):
        parse_action_from_text("completely unrelated text", actions)

