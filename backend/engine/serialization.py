"""
Serialization and LLM-friendly text conversion for the Catan game engine.
"""
import json
import re
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict, field
from enum import Enum
from difflib import SequenceMatcher

from .engine import (
    ResourceType,
    Tile,
    NumberToken,
    Intersection,
    RoadEdge,
    Player,
    GameState,
    Action,
    ActionPayload,
    BuildRoadPayload,
    BuildSettlementPayload,
    BuildCityPayload,
    PlayDevCardPayload,
    TradeBankPayload,
    TradePlayerPayload,
    ProposeTradePayload,
    SelectTradePartnerPayload,
    MoveRobberPayload,
    StealResourcePayload,
    DiscardResourcesPayload,
)


def serialize_game_state(state: GameState) -> Dict[str, Any]:
    """
    Serialize GameState to a JSON-serializable dictionary.
    """
    result = {
        "game_id": state.game_id,
        "players": [serialize_player(p) for p in state.players],
        "current_player_index": state.current_player_index,
        "phase": state.phase,
        "tiles": [serialize_tile(t) for t in state.tiles],
        "intersections": [serialize_intersection(i) for i in state.intersections],
        "road_edges": [serialize_road_edge(r) for r in state.road_edges],
        "dice_roll": state.dice_roll,
        "turn_number": state.turn_number,
        "setup_round": state.setup_round,
        "setup_phase_player_index": state.setup_phase_player_index,
        "setup_last_settlement_id": state.setup_last_settlement_id,
        "setup_first_settlement_player_index": state.setup_first_settlement_player_index,
        "robber_tile_id": state.robber_tile_id,
        "waiting_for_robber_move": state.waiting_for_robber_move,
        "waiting_for_robber_steal": state.waiting_for_robber_steal,
        "players_discarded": list(state.players_discarded),
        "robber_initial_tile_id": state.robber_initial_tile_id,
        "roads_from_road_building": state.roads_from_road_building,
        "dev_cards_bought_this_turn": list(state.dev_cards_bought_this_turn),
        "dev_cards_played_this_turn": list(state.dev_cards_played_this_turn),
    }
    # Add trade state fields
    if state.pending_trade_offer:
        # Convert ResourceType enums to strings for JSON serialization
        result["pending_trade_offer"] = {
            "proposer_id": state.pending_trade_offer["proposer_id"],
            "target_player_ids": state.pending_trade_offer["target_player_ids"],
            "give_resources": {rt.value: count for rt, count in state.pending_trade_offer["give_resources"].items()},
            "receive_resources": {rt.value: count for rt, count in state.pending_trade_offer["receive_resources"].items()},
        }
    else:
        result["pending_trade_offer"] = None
    result["pending_trade_responses"] = state.pending_trade_responses
    result["pending_trade_current_responder_index"] = state.pending_trade_current_responder_index
    return result


def deserialize_game_state(data: Dict[str, Any]) -> GameState:
    """
    Deserialize a dictionary to GameState.
    """
    # Handle trade state fields - need to convert resource dicts from strings to ResourceType
    pending_trade_offer = data.get("pending_trade_offer")
    if pending_trade_offer:
        # Convert resource type strings to ResourceType enums
        give_resources = {
            ResourceType(rt): count
            for rt, count in pending_trade_offer.get("give_resources", {}).items()
        }
        receive_resources = {
            ResourceType(rt): count
            for rt, count in pending_trade_offer.get("receive_resources", {}).items()
        }
        pending_trade_offer = {
            "proposer_id": pending_trade_offer["proposer_id"],
            "target_player_ids": pending_trade_offer["target_player_ids"],
            "give_resources": give_resources,
            "receive_resources": receive_resources,
        }
    
    return GameState(
        game_id=data["game_id"],
        players=[deserialize_player(p) for p in data["players"]],
        current_player_index=data["current_player_index"],
        phase=data["phase"],
        tiles=[deserialize_tile(t) for t in data.get("tiles", [])],
        intersections=[deserialize_intersection(i) for i in data.get("intersections", [])],
        road_edges=[deserialize_road_edge(r) for r in data.get("road_edges", [])],
        dice_roll=data.get("dice_roll"),
        turn_number=data.get("turn_number", 0),
        setup_round=data.get("setup_round", 0),
        setup_phase_player_index=data.get("setup_phase_player_index", 0),
        setup_last_settlement_id=data.get("setup_last_settlement_id"),
        setup_first_settlement_player_index=data.get("setup_first_settlement_player_index"),
        robber_tile_id=data.get("robber_tile_id"),
        waiting_for_robber_move=data.get("waiting_for_robber_move", False),
        waiting_for_robber_steal=data.get("waiting_for_robber_steal", False),
        players_discarded=set(data.get("players_discarded", [])),
        robber_initial_tile_id=data.get("robber_initial_tile_id"),
        roads_from_road_building=dict(data.get("roads_from_road_building", {})),
        dev_cards_bought_this_turn=set(data.get("dev_cards_bought_this_turn", [])),
        dev_cards_played_this_turn=set(data.get("dev_cards_played_this_turn", [])),
        pending_trade_offer=pending_trade_offer,
        pending_trade_responses=dict(data.get("pending_trade_responses", {})),
        pending_trade_current_responder_index=data.get("pending_trade_current_responder_index", 0),
    )


def serialize_player(player: Player) -> Dict[str, Any]:
    """Serialize a Player to a dictionary."""
    return {
        "id": player.id,
        "name": player.name,
        "color": player.color,
        "resources": {rt.value: count for rt, count in player.resources.items()},
        "victory_points": player.victory_points,
        "roads_built": player.roads_built,
        "settlements_built": player.settlements_built,
        "cities_built": player.cities_built,
        "dev_cards": player.dev_cards,
        "knights_played": player.knights_played,
        "longest_road": player.longest_road,
        "largest_army": player.largest_army,
    }


def deserialize_player(data: Dict[str, Any]) -> Player:
    """Deserialize a dictionary to a Player."""
    resources = {
        ResourceType(rt): count
        for rt, count in data.get("resources", {}).items()
    }
    # Ensure all resource types are present
    for rt in ResourceType:
        if rt not in resources:
            resources[rt] = 0
    
    return Player(
        id=data["id"],
        name=data["name"],
        color=data.get("color", "blue"),  # Default to blue for backwards compatibility
        resources=resources,
        victory_points=data.get("victory_points", 0),
        roads_built=data.get("roads_built", 0),
        settlements_built=data.get("settlements_built", 0),
        cities_built=data.get("cities_built", 0),
        dev_cards=data.get("dev_cards", []),
        knights_played=data.get("knights_played", 0),
        longest_road=data.get("longest_road", False),
        largest_army=data.get("largest_army", False),
    )


def serialize_tile(tile: Tile) -> Dict[str, Any]:
    """Serialize a Tile to a dictionary."""
    return {
        "id": tile.id,
        "resource_type": tile.resource_type.value if tile.resource_type else None,
        "number_token": tile.number_token.value if tile.number_token else None,
        "position": list(tile.position),
    }


def deserialize_tile(data: Dict[str, Any]) -> Tile:
    """Deserialize a dictionary to a Tile."""
    return Tile(
        id=data["id"],
        resource_type=ResourceType(data["resource_type"]) if data.get("resource_type") else None,
        number_token=NumberToken(data["number_token"]) if data.get("number_token") else None,
        position=tuple(data["position"]),
    )


def serialize_intersection(intersection: Intersection) -> Dict[str, Any]:
    """Serialize an Intersection to a dictionary."""
    return {
        "id": intersection.id,
        "position": list(intersection.position),
        "adjacent_tiles": list(intersection.adjacent_tiles),
        "adjacent_intersections": list(intersection.adjacent_intersections),
        "owner": intersection.owner,
        "building_type": intersection.building_type,
        "port_type": intersection.port_type,
    }


def deserialize_intersection(data: Dict[str, Any]) -> Intersection:
    """Deserialize a dictionary to an Intersection."""
    return Intersection(
        id=data["id"],
        position=tuple(data["position"]),
        adjacent_tiles=set(data.get("adjacent_tiles", [])),
        adjacent_intersections=set(data.get("adjacent_intersections", [])),
        owner=data.get("owner"),
        building_type=data.get("building_type"),
        port_type=data.get("port_type"),
    )


def serialize_road_edge(road_edge: RoadEdge) -> Dict[str, Any]:
    """Serialize a RoadEdge to a dictionary."""
    return {
        "id": road_edge.id,
        "intersection1_id": road_edge.intersection1_id,
        "intersection2_id": road_edge.intersection2_id,
        "owner": road_edge.owner,
    }


def deserialize_road_edge(data: Dict[str, Any]) -> RoadEdge:
    """Deserialize a dictionary to a RoadEdge."""
    return RoadEdge(
        id=data["id"],
        intersection1_id=data["intersection1_id"],
        intersection2_id=data["intersection2_id"],
        owner=data.get("owner"),
    )


def serialize_action(action: Action) -> str:
    """Serialize an Action to a string."""
    return action.value


def deserialize_action(value: str) -> Action:
    """Deserialize a string to an Action."""
    return Action(value)


def serialize_action_payload(payload: ActionPayload) -> Dict[str, Any]:
    """Serialize an ActionPayload to a dictionary."""
    if isinstance(payload, BuildRoadPayload):
        return {
            "type": "BuildRoadPayload",
            "road_edge_id": payload.road_edge_id,
        }
    elif isinstance(payload, BuildSettlementPayload):
        return {
            "type": "BuildSettlementPayload",
            "intersection_id": payload.intersection_id,
        }
    elif isinstance(payload, BuildCityPayload):
        return {
            "type": "BuildCityPayload",
            "intersection_id": payload.intersection_id,
        }
    elif isinstance(payload, PlayDevCardPayload):
        result = {
            "type": "PlayDevCardPayload",
            "card_type": payload.card_type,
        }
        if payload.year_of_plenty_resources:
            result["year_of_plenty_resources"] = {rt.value: count for rt, count in payload.year_of_plenty_resources.items()}
        if payload.monopoly_resource_type:
            result["monopoly_resource_type"] = payload.monopoly_resource_type.value
        return result
    elif isinstance(payload, TradeBankPayload):
        return {
            "type": "TradeBankPayload",
            "give_resources": {rt.value: count for rt, count in payload.give_resources.items()},
            "receive_resources": {rt.value: count for rt, count in payload.receive_resources.items()},
            "port_intersection_id": payload.port_intersection_id,
        }
    elif isinstance(payload, TradePlayerPayload):
        return {
            "type": "TradePlayerPayload",
            "other_player_id": payload.other_player_id,
            "give_resources": {rt.value: count for rt, count in payload.give_resources.items()},
            "receive_resources": {rt.value: count for rt, count in payload.receive_resources.items()},
        }
    elif isinstance(payload, ProposeTradePayload):
        return {
            "type": "ProposeTradePayload",
            "target_player_ids": payload.target_player_ids,
            "give_resources": {rt.value: count for rt, count in payload.give_resources.items()},
            "receive_resources": {rt.value: count for rt, count in payload.receive_resources.items()},
        }
    elif isinstance(payload, SelectTradePartnerPayload):
        return {
            "type": "SelectTradePartnerPayload",
            "selected_player_id": payload.selected_player_id,
        }
    elif isinstance(payload, MoveRobberPayload):
        return {
            "type": "MoveRobberPayload",
            "tile_id": payload.tile_id,
        }
    elif isinstance(payload, StealResourcePayload):
        return {
            "type": "StealResourcePayload",
            "other_player_id": payload.other_player_id,
        }
    elif isinstance(payload, DiscardResourcesPayload):
        return {
            "type": "DiscardResourcesPayload",
            "resources": {rt.value: count for rt, count in payload.resources.items()},
        }
    else:
        raise ValueError(f"Unknown payload type: {type(payload)}")


def deserialize_action_payload(data: Dict[str, Any]) -> ActionPayload:
    """Deserialize a dictionary to an ActionPayload."""
    payload_type = data["type"]
    
    if payload_type == "BuildRoadPayload":
        return BuildRoadPayload(road_edge_id=data["road_edge_id"])
    elif payload_type == "BuildSettlementPayload":
        return BuildSettlementPayload(intersection_id=data["intersection_id"])
    elif payload_type == "BuildCityPayload":
        return BuildCityPayload(intersection_id=data["intersection_id"])
    elif payload_type == "PlayDevCardPayload":
        year_of_plenty_resources = None
        if "year_of_plenty_resources" in data and data["year_of_plenty_resources"]:
            year_of_plenty_resources = {
                ResourceType(rt): count
                for rt, count in data["year_of_plenty_resources"].items()
            }
        
        monopoly_resource_type = None
        if "monopoly_resource_type" in data and data["monopoly_resource_type"]:
            monopoly_resource_type = ResourceType(data["monopoly_resource_type"])
        
        return PlayDevCardPayload(
            card_type=data["card_type"],
            year_of_plenty_resources=year_of_plenty_resources,
            monopoly_resource_type=monopoly_resource_type
        )
    elif payload_type == "TradeBankPayload":
        give_resources = {
            ResourceType(rt): count
            for rt, count in data.get("give_resources", {}).items()
        }
        receive_resources = {
            ResourceType(rt): count
            for rt, count in data.get("receive_resources", {}).items()
        }
        return TradeBankPayload(
            give_resources=give_resources,
            receive_resources=receive_resources,
            port_intersection_id=data.get("port_intersection_id"),
        )
    elif payload_type == "TradePlayerPayload":
        give_resources = {
            ResourceType(rt): count
            for rt, count in data.get("give_resources", {}).items()
        }
        receive_resources = {
            ResourceType(rt): count
            for rt, count in data.get("receive_resources", {}).items()
        }
        return TradePlayerPayload(
            other_player_id=data["other_player_id"],
            give_resources=give_resources,
            receive_resources=receive_resources,
        )
    elif payload_type == "ProposeTradePayload":
        give_resources = {
            ResourceType(rt): count
            for rt, count in data.get("give_resources", {}).items()
        }
        receive_resources = {
            ResourceType(rt): count
            for rt, count in data.get("receive_resources", {}).items()
        }
        return ProposeTradePayload(
            target_player_ids=data["target_player_ids"],
            give_resources=give_resources,
            receive_resources=receive_resources,
        )
    elif payload_type == "SelectTradePartnerPayload":
        return SelectTradePartnerPayload(
            selected_player_id=data["selected_player_id"]
        )
    elif payload_type == "MoveRobberPayload":
        return MoveRobberPayload(tile_id=data["tile_id"])
    elif payload_type == "StealResourcePayload":
        return StealResourcePayload(other_player_id=data["other_player_id"])
    elif payload_type == "DiscardResourcesPayload":
        resources = {
            ResourceType(rt): count
            for rt, count in data.get("resources", {}).items()
        }
        return DiscardResourcesPayload(resources=resources)
    else:
        raise ValueError(f"Unknown payload type: {payload_type}")


# Text conversion functions

def legal_actions(state: GameState, player_id: str) -> List[Tuple[Action, Optional[ActionPayload]]]:
    """
    Get all legal actions for a given player in the current state.
    Returns a list of (Action, Optional[ActionPayload]) tuples.
    """
    legal = []
    
    # Find the player
    player = next((p for p in state.players if p.id == player_id), None)
    if not player:
        return legal
    
    # Check if it's the player's turn
    is_current_player = (
        state.players[state.current_player_index].id == player_id
        if state.phase == "playing"
        else state.players[state.setup_phase_player_index].id == player_id
    )
    
    # Special case: when a 7 is rolled, any player with 8+ resources can discard
    # (even if it's not their turn)
    can_discard = False
    if state.phase == "playing" and state.dice_roll == 7:
        player = next((p for p in state.players if p.id == player_id), None)
        if player and sum(player.resources.values()) >= 8:
            can_discard = True
    
    if not is_current_player and not can_discard:
        return legal  # Not this player's turn (unless they can discard)
    
    if state.phase == "setup":
        # Setup phase actions
        # Only the current setup player can act
        current_setup_player = state.players[state.setup_phase_player_index]
        if player_id != current_setup_player.id:
            return legal  # Not this player's turn in setup
        
        # Count how many settlements and roads this player has placed
        player_settlements = sum(1 for i in state.intersections if i.owner == player_id and i.building_type == "settlement")
        player_roads = sum(1 for r in state.road_edges if r.owner == player_id)
        
        # In setup, each player places exactly 2 settlements (one per round)
        # Round 0: players should have 0 settlements, 0 roads
        # Round 1: players should have 1 settlement, 1 road
        expected_settlements = state.setup_round
        expected_roads = state.setup_round
        
        # Can place settlement if not placed yet in this round
        if player_settlements == expected_settlements:
            for intersection in state.intersections:
                if not intersection.owner:
                    # Check distance rule
                    can_build = True
                    for adj_id in intersection.adjacent_intersections:
                        adj_inter = next((i for i in state.intersections if i.id == adj_id), None)
                        if adj_inter and adj_inter.owner:
                            can_build = False
                            break
                    if can_build:
                        legal.append((Action.SETUP_PLACE_SETTLEMENT, BuildSettlementPayload(intersection.id)))
        
        # Can place road adjacent to settlement (only after placing settlement this round, and not yet placed road)
        # Road must be adjacent to the settlement just placed (setup_last_settlement_id)
        if player_settlements == expected_settlements + 1 and player_roads == expected_roads:
            # Player has placed settlement for this round, can now place road
            # Road must attach to the settlement just placed
            if state.setup_last_settlement_id is not None:
                for road_edge in state.road_edges:
                    if not road_edge.owner:
                        # Check if road is adjacent to the last settlement placed
                        if (road_edge.intersection1_id == state.setup_last_settlement_id or 
                            road_edge.intersection2_id == state.setup_last_settlement_id):
                            legal.append((Action.SETUP_PLACE_ROAD, BuildRoadPayload(road_edge.id)))
        
        # Can start game if setup is complete (simplified - in real game, this is automatic)
        if state.setup_round == 1 and state.setup_phase_player_index == len(state.players) - 1:
            legal.append((Action.START_GAME, None))
    
    elif state.phase == "playing":
        # Playing phase actions
        
        # Check for pending trade first (takes priority over other actions)
        if state.pending_trade_offer is not None:
            offer = state.pending_trade_offer
            current_player = state.players[state.current_player_index]
            
            # Check if current player is a target of the trade
            if current_player.id in offer['target_player_ids']:
                # Check if this player has already responded
                if current_player.id not in state.pending_trade_responses:
                    # Player needs to respond - can accept or reject
                    # Verify player can afford the trade (they give receive_resources, get give_resources)
                    can_afford = True
                    for resource, amount in offer['receive_resources'].items():
                        if current_player.resources[resource] < amount:
                            can_afford = False
                            break
                    
                    if can_afford:
                        legal.append((Action.ACCEPT_TRADE, None))
                    legal.append((Action.REJECT_TRADE, None))
                    # Don't show other actions while trade is pending
                    return legal
                else:
                    # Player has already responded - wait for other players
                    return legal
            # Check if current player is the proposer and multiple players accepted
            elif current_player.id == offer['proposer_id']:
                accepting_players = [pid for pid, accepted in state.pending_trade_responses.items() if accepted]
                if len(accepting_players) > 1:
                    # Multiple accepted - proposer must choose
                    for accepting_player_id in accepting_players:
                        legal.append((Action.SELECT_TRADE_PARTNER, SelectTradePartnerPayload(selected_player_id=accepting_player_id)))
                    # Don't show other actions while trade is pending
                    return legal
                else:
                    # Trade is being processed or no one accepted - wait
                    return legal
            else:
                # Not involved in trade - wait
                return legal
        
        if state.dice_roll is None:
            # Must roll dice first
            legal.append((Action.ROLL_DICE, None))
        elif state.waiting_for_robber_move:
            # Robber must be moved (either from 7 roll or knight card)
            if is_current_player:
                # Can move robber to any tile except current
                # (Moving to own tiles is allowed if it also blocks opponents - strategic decision)
                for tile in state.tiles:
                    if tile.id != state.robber_tile_id:
                        legal.append((Action.MOVE_ROBBER, MoveRobberPayload(tile.id)))
            # Don't show other actions while waiting to move robber
            return legal
        elif state.waiting_for_robber_steal:
            # After moving robber, can steal from players on that tile
            if is_current_player:
                robber_tile = next((t for t in state.tiles if t.id == state.robber_tile_id), None)
                if robber_tile:
                    # Find players with buildings on this tile
                    players_on_tile = set()
                    for intersection in state.intersections:
                        if (robber_tile.id in intersection.adjacent_tiles and 
                            intersection.owner and 
                            intersection.building_type):
                            if intersection.owner != player_id:
                                players_on_tile.add(intersection.owner)
                    
                    valid_steal_targets = []
                    for other_player_id in players_on_tile:
                        other_player = next((p for p in state.players if p.id == other_player_id), None)
                        if other_player and sum(other_player.resources.values()) > 0:
                            valid_steal_targets.append(other_player_id)
                            legal.append((Action.STEAL_RESOURCE, StealResourcePayload(other_player_id)))
                    
                    # If there are no valid steal targets, allow normal turn actions to continue
                    # (The waiting_for_robber_steal flag will be cleared when we try to end turn)
                    if not valid_steal_targets:
                        # No one to steal from, fall through to normal turn actions
                        pass
                    else:
                        # Don't show other actions while waiting to steal
                        return legal
            else:
                # Not current player, no actions available
                return legal
        elif state.dice_roll == 7:
            # Handle rolling 7
            # First phase: ALL players with 8+ resources must discard
            # Players can discard even if it's not their turn, and even if the robber phase has started
            # (though ideally the robber shouldn't move until all discards are done)
            
            # Check if discard phase is still active
            # Discard phase is over if:
            # 1. Robber has been moved (robber_tile_id != robber_initial_tile_id), OR
            # 2. Both waiting_for_robber_move and waiting_for_robber_steal are False (robber phase complete)
            robber_has_been_moved = (state.robber_initial_tile_id is not None and 
                                     state.robber_tile_id != state.robber_initial_tile_id)
            discard_phase_complete = robber_has_been_moved or (
                not state.waiting_for_robber_move and not state.waiting_for_robber_steal
            )
            
            # Check if this player needs to discard (has 8+ resources and hasn't discarded yet)
            # BUT only if the discard phase hasn't completed yet
            total_resources = sum(player.resources.values())
            if total_resources >= 8 and player_id not in state.players_discarded and not discard_phase_complete:
                # Player must discard - but only if discard phase is still active
                legal.append((Action.DISCARD_RESOURCES, None))  # Payload will be provided by frontend
                # Don't show other actions while this player needs to discard
                return legal
            
            # Check if we're still in discard phase (other players need to discard)
            # Only check if discard phase hasn't completed yet
            if not discard_phase_complete:
                # Check if any other players still need to discard
                any_player_needs_discard = False
                for p in state.players:
                    if p.id not in state.players_discarded and sum(p.resources.values()) >= 8:
                        any_player_needs_discard = True
                        break
                
                # If other players need to discard and we're not in robber phase yet, don't show other actions
                if any_player_needs_discard and not state.waiting_for_robber_move and not state.waiting_for_robber_steal:
                    # Other players still need to discard, and we're not in robber phase
                    # This player can't do anything else until all discards are done
                    return legal
            
            # All discards done, handle robber phase (for 7 roll only)
            # Note: waiting_for_robber_move from knight cards is handled above
            if state.waiting_for_robber_move:
                # Verify that all players with 8+ resources have actually discarded
                all_discarded = True
                for p in state.players:
                    if p.id not in state.players_discarded and sum(p.resources.values()) >= 8:
                        all_discarded = False
                        break
                
                # Only allow moving robber if all discards are complete
                if all_discarded:
                    # All discards done, now ONLY the player who rolled the 7 can move robber
                    if is_current_player:
                        # Can move robber to any tile except current
                        # (Moving to own tiles is allowed if it also blocks opponents - strategic decision)
                        for tile in state.tiles:
                            if tile.id != state.robber_tile_id:
                                legal.append((Action.MOVE_ROBBER, MoveRobberPayload(tile.id)))
                    # Don't show other actions while waiting to move robber
                    return legal
                else:
                    # Not all players have discarded yet - don't allow moving robber
                    # (This shouldn't happen if the engine logic is correct, but handle it gracefully)
                    return legal
            
            if state.waiting_for_robber_steal:
                # After moving robber, can steal from players on that tile
                if is_current_player:
                    robber_tile = next((t for t in state.tiles if t.id == state.robber_tile_id), None)
                    if robber_tile:
                        # Find players with buildings on this tile
                        players_on_tile = set()
                        for intersection in state.intersections:
                            if (robber_tile.id in intersection.adjacent_tiles and 
                                intersection.owner and 
                                intersection.building_type):
                                if intersection.owner != player_id:
                                    players_on_tile.add(intersection.owner)
                        
                        valid_steal_targets = []
                        for other_player_id in players_on_tile:
                            other_player = next((p for p in state.players if p.id == other_player_id), None)
                            if other_player and sum(other_player.resources.values()) > 0:
                                valid_steal_targets.append(other_player_id)
                                legal.append((Action.STEAL_RESOURCE, StealResourcePayload(other_player_id)))
                        
                        # If there are no valid steal targets, allow normal turn actions to continue
                        # (The waiting_for_robber_steal flag will be cleared when we try to end turn)
                        if not valid_steal_targets:
                            # No one to steal from, fall through to normal turn actions
                            pass
                        else:
                            # Don't show other actions while waiting to steal
                            return legal
                else:
                    # Not current player, no actions available
                    return legal
            
            # If we get here, robber phase is complete - fall through to normal turn actions
        
        # Normal turn actions (after dice roll, or after completing robber phase on 7)
        # Trading is only allowed after dice is rolled AND all 7-roll phases are complete
        can_trade = False
        if state.dice_roll is not None and is_current_player:
            # Check if we're in a 7-roll phase that hasn't been completed
            if state.dice_roll == 7:
                # Check if we're still in discard phase
                robber_has_been_moved = (state.robber_initial_tile_id is not None and 
                                         state.robber_tile_id != state.robber_initial_tile_id)
                in_discard_phase = (not state.waiting_for_robber_move and 
                                   not state.waiting_for_robber_steal and 
                                   not robber_has_been_moved)
                
                if in_discard_phase:
                    # Check if any player still needs to discard
                    any_player_needs_discard = False
                    for p in state.players:
                        if p.id not in state.players_discarded and sum(p.resources.values()) >= 8:
                            any_player_needs_discard = True
                            break
                    if any_player_needs_discard:
                        can_trade = False  # Still in discard phase
                    else:
                        can_trade = False  # Discard phase just finished, but robber phase hasn't started yet
                elif state.waiting_for_robber_move or state.waiting_for_robber_steal:
                    can_trade = False  # In robber move/steal phase
                else:
                    can_trade = True  # All 7-roll phases complete
            else:
                can_trade = True  # Not a 7 roll, trading allowed after dice roll
        
        if state.dice_roll is not None and is_current_player:
            # Can build road (either with resources or using road building card)
            free_roads_remaining = state.roads_from_road_building.get(player_id, 0)
            has_resources = (player.resources[ResourceType.WOOD] >= 1 and  
                            player.resources[ResourceType.BRICK] >= 1)
            can_build_road = has_resources or free_roads_remaining > 0
            
            if can_build_road:
                for road_edge in state.road_edges:
                    if not road_edge.owner:
                        # Check if player has a road/settlement adjacent
                        has_connection = False
                        for intersection in state.intersections:
                            if intersection.owner == player_id:
                                if (road_edge.intersection1_id == intersection.id or 
                                    road_edge.intersection2_id == intersection.id):
                                    has_connection = True
                                    break
                            # Or check if player has adjacent road
                            for other_road in state.road_edges:
                                if other_road.owner == player_id:
                                    if (road_edge.intersection1_id == other_road.intersection1_id or
                                        road_edge.intersection1_id == other_road.intersection2_id or
                                        road_edge.intersection2_id == other_road.intersection1_id or
                                        road_edge.intersection2_id == other_road.intersection2_id):
                                        has_connection = True
                                        break
                        if has_connection:
                            legal.append((Action.BUILD_ROAD, BuildRoadPayload(road_edge.id)))
            
            # Can build settlement
            if (player.resources[ResourceType.WOOD] >= 1 and 
                player.resources[ResourceType.BRICK] >= 1 and
                player.resources[ResourceType.WHEAT] >= 1 and
                player.resources[ResourceType.SHEEP] >= 1):
                for intersection in state.intersections:
                    if not intersection.owner:
                        # Check distance rule
                        can_build = True
                        for adj_id in intersection.adjacent_intersections:
                            adj_inter = next((i for i in state.intersections if i.id == adj_id), None)
                            if adj_inter and adj_inter.owner:
                                can_build = False
                                break
                        if can_build:
                            # Check if player has road/settlement adjacent
                            has_connection = False
                            for road_edge in state.road_edges:
                                if road_edge.owner == player_id:
                                    if (road_edge.intersection1_id == intersection.id or 
                                        road_edge.intersection2_id == intersection.id):
                                        has_connection = True
                                        break
                            if has_connection:
                                legal.append((Action.BUILD_SETTLEMENT, BuildSettlementPayload(intersection.id)))
            
            # Can build city (upgrade settlement)
            if (player.resources[ResourceType.WHEAT] >= 2 and 
                player.resources[ResourceType.ORE] >= 3):
                for intersection in state.intersections:
                    if (intersection.owner == player_id and 
                        intersection.building_type == "settlement"):
                        legal.append((Action.BUILD_CITY, BuildCityPayload(intersection.id)))
            
            # Can buy dev card
            if (player.resources[ResourceType.WHEAT] >= 1 and 
                player.resources[ResourceType.SHEEP] >= 1 and
                player.resources[ResourceType.ORE] >= 1):
                legal.append((Action.BUY_DEV_CARD, None))
            
            # Can play dev cards (with restrictions)
            # Cannot play if bought this turn (unless VP)
            # Cannot play if already played one this turn (unless VP)
            for card_type in player.dev_cards:
                # Skip if restrictions apply (unless VP)
                if card_type != "victory_point":
                    if player_id in state.dev_cards_bought_this_turn:
                        continue  # Cannot play same turn as buying
                    if player_id in state.dev_cards_played_this_turn:
                        continue  # Cannot play two in one turn
                
                if card_type == "year_of_plenty":
                    # Year of plenty: player can choose 2 resources
                    # Generate actions for all valid combinations:
                    # - 2 of the same resource (5 options: 2 wood, 2 brick, 2 wheat, 2 sheep, 2 ore)
                    # - 2 different resources (C(5,2) = 10 combinations)
                    resource_types = list(ResourceType)
                    # Same resource twice
                    for res in resource_types:
                        legal.append((Action.PLAY_DEV_CARD, PlayDevCardPayload(
                            card_type=card_type,
                            year_of_plenty_resources={res: 2}
                        )))
                    # Two different resources
                    for i, res1 in enumerate(resource_types):
                        for j, res2 in enumerate(resource_types):
                            if i < j:  # Different resources, avoid duplicates
                                legal.append((Action.PLAY_DEV_CARD, PlayDevCardPayload(
                                    card_type=card_type,
                                    year_of_plenty_resources={res1: 1, res2: 1}
                                )))
                elif card_type == "monopoly":
                    # Monopoly: player can choose a resource type to steal
                    for resource_type in ResourceType:
                        legal.append((Action.PLAY_DEV_CARD, PlayDevCardPayload(
                            card_type=card_type,
                            monopoly_resource_type=resource_type
                        )))
                else:
                    # Other cards (victory_point, knight, road_building) don't need payloads
                    legal.append((Action.PLAY_DEV_CARD, PlayDevCardPayload(card_type)))
            
            # Check for ports owned by player
            # Ports span 2 adjacent intersections - player must own at least one
            owned_ports = []
            port_types_seen = set()
            for intersection in state.intersections:
                if intersection.port_type is not None and intersection.port_type not in port_types_seen:
                    # Check if player owns this intersection or an adjacent one with same port
                    owns_port = False
                    if intersection.owner == player_id:
                        owns_port = True
                    else:
                        # Check adjacent intersections
                        for adj_id in intersection.adjacent_intersections:
                            adj_inter = next((i for i in state.intersections if i.id == adj_id), None)
                            if (adj_inter and 
                                adj_inter.port_type == intersection.port_type and
                                adj_inter.owner == player_id):
                                owns_port = True
                                break
                    
                    if owns_port:
                        owned_ports.append(intersection)
                        port_types_seen.add(intersection.port_type)
            
            # Can trade with bank using ports (only if trading is allowed)
            if can_trade:
                for port_inter in owned_ports:
                    if port_inter.port_type == "3:1":
                        # 3:1 generic port - can trade 3 of any resource for 1 of any resource
                        for give_rt in ResourceType:
                            if player.resources[give_rt] >= 3:
                                for receive_rt in ResourceType:
                                    if give_rt != receive_rt:
                                        legal.append((Action.TRADE_BANK, TradeBankPayload(
                                            give_resources={give_rt: 3},
                                            receive_resources={receive_rt: 1},
                                            port_intersection_id=port_inter.id,
                                        )))
                    else:
                        # 2:1 specific resource port
                        port_resource = ResourceType(port_inter.port_type)
                        if player.resources[port_resource] >= 2:
                            for receive_rt in ResourceType:
                                if port_resource != receive_rt:
                                    legal.append((Action.TRADE_BANK, TradeBankPayload(
                                        give_resources={port_resource: 2},
                                        receive_resources={receive_rt: 1},
                                        port_intersection_id=port_inter.id,
                                    )))
            
            # Can trade with bank (standard 4:1, no port) (only if trading is allowed)
            if can_trade:
                for give_rt in ResourceType:
                    if player.resources[give_rt] >= 4:
                        for receive_rt in ResourceType:
                            if give_rt != receive_rt:
                                legal.append((Action.TRADE_BANK, TradeBankPayload(
                                    give_resources={give_rt: 4},
                                    receive_resources={receive_rt: 1},
                                    port_intersection_id=None,
                                )))
            
            # Can propose trades to other players (only if no trade is pending and trading is allowed)
            if can_trade and state.pending_trade_offer is None:
                # Generate trade proposals: simple 1:1 trades for each resource type
                # Agents can propose giving 1 of any resource they have for 1 of any resource they need
                other_players = [p.id for p in state.players if p.id != player_id]
                
                if other_players:
                    # Generate trades: give 1 of X, receive 1 of Y (for each resource type the player has)
                    for give_resource in ResourceType:
                        if player.resources.get(give_resource, 0) >= 1:
                            # Can propose to give 1 of this resource
                            for receive_resource in ResourceType:
                                if give_resource != receive_resource:
                                    # Propose trade to all other players
                                    legal.append((Action.PROPOSE_TRADE, ProposeTradePayload(
                                        target_player_ids=other_players,
                                        give_resources={give_resource: 1},
                                        receive_resources={receive_resource: 1}
                                    )))
                    
                    # Also generate 2:1 trades (give 2, receive 1) for resources player has in excess
                    for give_resource in ResourceType:
                        if player.resources.get(give_resource, 0) >= 2:
                            for receive_resource in ResourceType:
                                if give_resource != receive_resource:
                                    legal.append((Action.PROPOSE_TRADE, ProposeTradePayload(
                                        target_player_ids=other_players,
                                        give_resources={give_resource: 2},
                                        receive_resources={receive_resource: 1}
                                    )))
            
            # Can end turn (but not if waiting for robber actions, or if 7 was rolled and anyone needs to discard)
            # Never allow END_TURN if waiting_for_robber_move or waiting_for_robber_steal is True
            # (The engine will clear these flags when appropriate, e.g., when there are no valid steal targets)
            if state.waiting_for_robber_move or state.waiting_for_robber_steal:
                # Cannot end turn while waiting for robber actions
                pass
            elif state.dice_roll == 7:
                # Check if discard phase is still active
                # Discard phase is over if robber has been moved or robber phase is complete
                robber_has_been_moved = (state.robber_initial_tile_id is not None and 
                                         state.robber_tile_id != state.robber_initial_tile_id)
                discard_phase_complete = robber_has_been_moved or (
                    not state.waiting_for_robber_move and not state.waiting_for_robber_steal
                )
                
                # Check if anyone still needs to discard (and hasn't discarded yet)
                # BUT only if the discard phase hasn't completed yet
                any_player_needs_discard = False
                if not discard_phase_complete:
                    for p in state.players:
                        if p.id not in state.players_discarded and sum(p.resources.values()) >= 8:
                            any_player_needs_discard = True
                            break
                
                # Can only end turn if no one needs to discard (or discard phase is complete)
                if not any_player_needs_discard:
                    legal.append((Action.END_TURN, None))
            else:
                # Normal turn - can end
                legal.append((Action.END_TURN, None))
    
    return legal


def state_to_text(state: GameState, player_id: str, history: List[Tuple[Action, Optional[ActionPayload]]] = None) -> str:
    """
    Convert game state to LLM-friendly text description from a player's perspective.
    Enhanced with spatial relationships, production analysis, and strategic information.
    """
    if history is None:
        history = []
    
    lines = []
    
    # Find the player
    player = next((p for p in state.players if p.id == player_id), None)
    if not player:
        return f"Player {player_id} not found in game"
    
    # Dice probability table (for reference)
    dice_probs = {
        2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
        7: 6/36,  # 7 doesn't produce resources
        8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
    }
    
    # Game overview
    lines.append(f"=== Catan Game State ===")
    lines.append(f"Game ID: {state.game_id}")
    lines.append(f"Phase: {state.phase}")
    lines.append(f"Turn: {state.turn_number}")
    
    if state.dice_roll:
        lines.append(f"Last dice roll: {state.dice_roll}")
    
    # Current player
    if state.phase == "playing":
        current_player = state.players[state.current_player_index]
        lines.append(f"Current player: {current_player.name} ({current_player.id})")
    elif state.phase == "setup":
        setup_player = state.players[state.setup_phase_player_index]
        lines.append(f"Setup player: {setup_player.name} ({setup_player.id})")
        lines.append(f"Setup round: {state.setup_round + 1}")
        
        # Show player order and who has already placed in this round
        lines.append(f"Player order in setup: {', '.join([f'{i+1}. {p.name} ({p.id})' for i, p in enumerate(state.players)])}")
        lines.append(f"Your position: {state.setup_phase_player_index + 1} of {len(state.players)}")
        
        # Show which players have already placed settlements in this round
        settlements_placed_this_round = []
        for i, p in enumerate(state.players):
            if i < state.setup_phase_player_index:
                # This player has already placed in this round
                settlements_placed_this_round.append(f"{p.name} ({p.id})")
        if settlements_placed_this_round:
            lines.append(f"Players who have already placed this round: {', '.join(settlements_placed_this_round)}")
        else:
            lines.append("You are the first player to place in this round")
    
    lines.append("")
    
    # Your status
    lines.append(f"=== Your Status ({player.name}) ===")
    lines.append(f"Victory Points: {player.victory_points}")
    lines.append(f"Resources:")
    for rt in ResourceType:
        lines.append(f"  {rt.value.capitalize()}: {player.resources[rt]}")
    lines.append(f"Buildings: {player.settlements_built} settlements, {player.cities_built} cities")
    lines.append(f"Roads: {player.roads_built}")
    # Calculate and show longest road length
    longest_road_length = _calculate_longest_road_length(state, player_id)
    if player.longest_road:
        lines.append(f"  * Longest Road: {longest_road_length} segments")
    elif longest_road_length > 0:
        lines.append(f"  Longest Road: {longest_road_length} segments (needs 5+ for card)")
    lines.append(f"Dev Cards: {len(player.dev_cards)}")
    if player.dev_cards:
        lines.append(f"  Types: {', '.join(player.dev_cards)}")
    if player.largest_army:
        lines.append("  * Largest Army")
    
    lines.append("")
    
    # Other players (with more detail)
    lines.append("=== Other Players ===")
    for p in state.players:
        if p.id != player_id:
            lines.append(f"{p.name} ({p.id}):")
            lines.append(f"  Victory Points: {p.victory_points}")
            lines.append(f"  Total Resources: {sum(p.resources.values())}")
            # Show resource breakdown (exact counts per type)
            resource_breakdown = ", ".join([f"{rt.value}: {p.resources[rt]}" for rt in ResourceType if p.resources[rt] > 0])
            if resource_breakdown:
                lines.append(f"  Resources: {resource_breakdown}")
            lines.append(f"  Buildings: {p.settlements_built} settlements, {p.cities_built} cities")
            lines.append(f"  Roads: {p.roads_built}")
            # Count development cards (visible count)
            dev_card_count = len(p.dev_cards)
            if dev_card_count > 0:
                lines.append(f"  Development Cards: {dev_card_count} (hidden)")
            # Longest road details - calculate actual longest road length
            longest_road_length = _calculate_longest_road_length(state, p.id)
            if p.longest_road:
                lines.append(f"  * Has Longest Road ({longest_road_length} road segments)")
            elif longest_road_length > 0:
                lines.append(f"  Longest Road: {longest_road_length} segments (needs 5+ for card)")
            # Largest army details
            if p.largest_army:
                knight_count = sum(1 for card in p.dev_cards if card == "knight")
                lines.append(f"  * Has Largest Army ({knight_count} knights played)")
            # Show opponent building locations
            opponent_buildings = [i for i in state.intersections if i.owner == p.id]
            if opponent_buildings:
                lines.append(f"  Building Locations:")
                for inter in opponent_buildings:
                    port_info = f" (port: {inter.port_type})" if inter.port_type else ""
                    lines.append(f"    Intersection {inter.id}: {inter.building_type}{port_info}")
    
    lines.append("")
    
    # Board state with spatial information
    lines.append("=== Board Layout ===")
    
    # Create tile lookup
    tile_map = {t.id: t for t in state.tiles}
    
    # Compact map representation
    lines.append("=== Compact Map View ===")
    map_lines = _generate_compact_map(state, player_id, tile_map)
    lines.extend(map_lines)
    lines.append("")
    
    # Resource scarcity analysis
    resource_counts = {}
    for tile in state.tiles:
        if tile.resource_type:
            resource_counts[tile.resource_type] = resource_counts.get(tile.resource_type, 0) + 1
    if resource_counts:
        lines.append("Resource Scarcity (tiles per resource type):")
        scarcity_str = ", ".join([f"{rt.value}: {count} tiles" for rt, count in sorted(resource_counts.items(), key=lambda x: -x[1])])
        lines.append(f"  {scarcity_str}")
        lines.append("")
    
    # Dice probability reference
    lines.append("Dice Roll Probabilities (for resource production):")
    lines.append("  2: 2.8% | 3: 5.6% | 4: 8.3% | 5: 11.1% | 6: 13.9%")
    lines.append("  8: 13.9% | 9: 11.1% | 10: 8.3% | 11: 5.6% | 12: 2.8%")
    lines.append("  (7: 16.7% - triggers robber, no production)")
    lines.append("  Best numbers: 6 and 8 (13.9% each)")
    lines.append("")
    
    # Tiles with production value
    lines.append("Tiles (with production value):")
    for tile in state.tiles:
        if tile.resource_type:
            token_value = tile.number_token.value if tile.number_token else None
            if token_value:
                prob = dice_probs[token_value] * 100
                lines.append(f"  Tile {tile.id}: {tile.resource_type.value} (token: {token_value}, {prob:.1f}% roll probability)")
            else:
                lines.append(f"  Tile {tile.id}: {tile.resource_type.value} (no token)")
        else:
            lines.append(f"  Tile {tile.id}: Desert (robber location)" if tile.id == state.robber_tile_id else f"  Tile {tile.id}: Desert")
    
    lines.append("")
    
    # Intersections with full spatial information
    lines.append("=== Intersections (Spatial Information) ===")
    
    # Group intersections by ownership
    your_intersections = []
    opponent_intersections = []
    empty_intersections = []
    
    for inter in state.intersections:
        if inter.owner == player_id:
            your_intersections.append(inter)
        elif inter.owner:
            opponent_intersections.append(inter)
        else:
            empty_intersections.append(inter)
    
    # Your intersections
    if your_intersections:
        lines.append("Your Intersections:")
        for inter in your_intersections:
            lines.append(f"  Intersection {inter.id}: {inter.building_type or 'empty'}")
            if inter.port_type:
                lines.append(f"    Port: {inter.port_type}")
            
            # Adjacent tiles with production analysis
            if inter.adjacent_tiles:
                lines.append(f"    Adjacent Tiles:")
                total_production_value = 0.0
                resource_production = {}
                for tile_id in inter.adjacent_tiles:
                    tile = tile_map.get(tile_id)
                    if tile and tile.resource_type and tile.number_token:
                        token_value = tile.number_token.value
                        prob = dice_probs[token_value]
                        total_production_value += prob
                        resource_type = tile.resource_type.value
                        if resource_type not in resource_production:
                            resource_production[resource_type] = 0.0
                        resource_production[resource_type] += prob
                        lines.append(f"      Tile {tile_id}: {tile.resource_type.value} (token {token_value}, {prob*100:.1f}% chance)")
                
                if total_production_value > 0:
                    lines.append(f"    Expected Production: {total_production_value*100:.1f}% per roll")
                    lines.append(f"    Resource Breakdown:")
                    for res_type, prob in sorted(resource_production.items(), key=lambda x: -x[1]):
                        lines.append(f"      {res_type}: {prob*100:.1f}% per roll")
            
            # Adjacent intersections
            if inter.adjacent_intersections:
                adj_intersections = []
                for adj_id in inter.adjacent_intersections:
                    adj_inter = next((i for i in state.intersections if i.id == adj_id), None)
                    if adj_inter:
                        owner_info = "you" if adj_inter.owner == player_id else (adj_inter.owner or "empty")
                        building_info = f" ({adj_inter.building_type})" if adj_inter.building_type else ""
                        adj_intersections.append(f"{adj_id} ({owner_info}{building_info})")
                if adj_intersections:
                    lines.append(f"    Adjacent Intersections: {', '.join(adj_intersections)}")
    
    # Opponent intersections (summary)
    if opponent_intersections:
        lines.append("")
        lines.append("Opponent Intersections:")
        for inter in opponent_intersections:
            owner_name = next((p.name for p in state.players if p.id == inter.owner), inter.owner)
            lines.append(f"  Intersection {inter.id}: {inter.building_type} (owned by {owner_name})")
            if inter.port_type:
                lines.append(f"    Port: {inter.port_type}")
            if inter.adjacent_tiles:
                tile_info = []
                for tile_id in inter.adjacent_tiles:
                    tile = tile_map.get(tile_id)
                    if tile and tile.resource_type and tile.number_token:
                        tile_info.append(f"T{tile_id}({tile.resource_type.value[0]}{tile.number_token.value})")
                if tile_info:
                    lines.append(f"    Tiles: {', '.join(tile_info)}")
    
    # Available intersections for building (with production analysis)
    if empty_intersections and (state.phase == "setup" or state.phase == "playing"):
        lines.append("")
        lines.append("Available Intersections (for building):")
        # Show top candidates by production value
        candidates = []
        for inter in empty_intersections:
            # Check distance rule (can't build adjacent to existing buildings)
            can_build = True
            for adj_id in inter.adjacent_intersections:
                adj_inter = next((i for i in state.intersections if i.id == adj_id), None)
                if adj_inter and adj_inter.owner:
                    can_build = False
                    break
            
            if can_build:
                total_production = 0.0
                resource_prod = {}
                tile_details = []  # Store actual tile info: (resource_type, token_value)
                port_bonus = ""
                if inter.port_type:
                    port_bonus = f" [PORT: {inter.port_type}]"
                
                for tile_id in inter.adjacent_tiles:
                    tile = tile_map.get(tile_id)
                    if tile and tile.resource_type and tile.number_token:
                        token_value = tile.number_token.value
                        prob = dice_probs[token_value]
                        total_production += prob
                        res_type = tile.resource_type.value
                        if res_type not in resource_prod:
                            resource_prod[res_type] = 0.0
                        resource_prod[res_type] += prob
                        # Store actual tile details
                        tile_details.append((res_type, token_value))
                
                candidates.append((inter.id, total_production, resource_prod, tile_details, port_bonus, inter.adjacent_intersections))
        
        # Sort by production value (descending)
        candidates.sort(key=lambda x: -x[1])
        
        # Show ALL available candidates (not just top 15)
        lines.append(f"  Total available: {len(candidates)} intersections")
        for inter_id, prod, res_prod, tile_details, port, adj_inters in candidates:
            lines.append(f"  Intersection {inter_id}{port}:")
            lines.append(f"    Production Value: {prod*100:.1f}% per roll")
            # Show actual tiles with resource types and number tokens
            if tile_details:
                tile_str = ", ".join([f"{res_type} {token}" for res_type, token in sorted(tile_details, key=lambda x: (x[0], x[1]))])
                lines.append(f"    Tiles: {tile_str}")
            # Also show aggregated percentages for quick reference
            if res_prod:
                res_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in sorted(res_prod.items(), key=lambda x: -x[1])])
                lines.append(f"    Resource Production: {res_str}")
            if adj_inters:
                adj_str = ", ".join([str(aid) for aid in sorted(adj_inters)])
                lines.append(f"    Adjacent Intersections: {adj_str}")
    
    lines.append("")
    
    # Road network information - show ALL roads systematically
    lines.append("=== Road Network ===")
    your_roads = [r for r in state.road_edges if r.owner == player_id]
    if your_roads:
        lines.append(f"Your Roads ({len(your_roads)}):")
        for road in sorted(your_roads, key=lambda r: r.id):
            inter1 = next((i for i in state.intersections if i.id == road.intersection1_id), None)
            inter2 = next((i for i in state.intersections if i.id == road.intersection2_id), None)
            tiles1 = sorted(inter1.adjacent_tiles)[:2] if inter1 and inter1.adjacent_tiles else []
            tiles2 = sorted(inter2.adjacent_tiles)[:2] if inter2 and inter2.adjacent_tiles else []
            tile_str1 = ",".join([f"T{tid}" for tid in tiles1])
            tile_str2 = ",".join([f"T{tid}" for tid in tiles2])
            lines.append(f"  Road {road.id}: I{road.intersection1_id}({tile_str1}) <-> I{road.intersection2_id}({tile_str2})")
    
    opponent_roads = [r for r in state.road_edges if r.owner and r.owner != player_id]
    if opponent_roads:
        # Group by owner
        roads_by_owner = {}
        for road in opponent_roads:
            if road.owner not in roads_by_owner:
                roads_by_owner[road.owner] = []
            roads_by_owner[road.owner].append(road)
        
        for owner_id, owner_roads in roads_by_owner.items():
            owner_name = next((p.name for p in state.players if p.id == owner_id), owner_id)
            lines.append(f"{owner_name}'s Roads ({len(owner_roads)}):")
            for road in sorted(owner_roads, key=lambda r: r.id)[:10]:  # Limit to 10 per opponent
                lines.append(f"  Road {road.id}: I{road.intersection1_id} <-> I{road.intersection2_id}")
            if len(owner_roads) > 10:
                lines.append(f"  ... and {len(owner_roads) - 10} more")
    
    # Show all unowned road edges (available for building)
    unowned_roads = [r for r in state.road_edges if not r.owner]
    if unowned_roads and state.phase == "playing":
        lines.append(f"Available Roads ({len(unowned_roads)} total, showing first 20):")
        for road in sorted(unowned_roads, key=lambda r: r.id)[:20]:
            inter1 = next((i for i in state.intersections if i.id == road.intersection1_id), None)
            inter2 = next((i for i in state.intersections if i.id == road.intersection2_id), None)
            # Check if road can be built (must connect to your existing road or settlement)
            can_build = False
            if inter1 and inter1.owner == player_id:
                can_build = True
            if inter2 and inter2.owner == player_id:
                can_build = True
            # Also check if adjacent to your roads
            if not can_build:
                for your_road in your_roads:
                    if (road.intersection1_id in [your_road.intersection1_id, your_road.intersection2_id] or
                        road.intersection2_id in [your_road.intersection1_id, your_road.intersection2_id]):
                        can_build = True
                        break
            
            marker = " [CAN BUILD]" if can_build else ""
            lines.append(f"  Road {road.id}: I{road.intersection1_id} <-> I{road.intersection2_id}{marker}")
        if len(unowned_roads) > 20:
            lines.append(f"  ... and {len(unowned_roads) - 20} more")
    
    lines.append("")
    
    # Robber information
    if state.robber_tile_id is not None:
        robber_tile = tile_map.get(state.robber_tile_id)
        if robber_tile:
            lines.append(f"=== Robber ===")
            lines.append(f"Robber is on Tile {state.robber_tile_id} ({robber_tile.resource_type.value if robber_tile.resource_type else 'Desert'})")
            # Show players who would be affected if this tile produces
            affected_players = set()
            for inter in state.intersections:
                if state.robber_tile_id in inter.adjacent_tiles and inter.owner:
                    affected_players.add(inter.owner)
            if affected_players:
                player_names = [next((p.name for p in state.players if p.id == pid), pid) for pid in affected_players]
                lines.append(f"  Blocks production for: {', '.join(player_names)}")
    
    lines.append("")
    
    # Pending trade information
    if state.pending_trade_offer:
        lines.append("=== Pending Trade Offer ===")
        offer = state.pending_trade_offer
        proposer = next((p for p in state.players if p.id == offer['proposer_id']), None)
        proposer_name = proposer.name if proposer else offer['proposer_id']
        
        lines.append(f"Proposer: {proposer_name}")
        lines.append(f"Target Players: {', '.join([next((p.name for p in state.players if p.id == pid), pid) for pid in offer['target_player_ids']])}")
        
        give_str = ", ".join([f"{count} {rt.value}" for rt, count in offer['give_resources'].items()])
        receive_str = ", ".join([f"{count} {rt.value}" for rt, count in offer['receive_resources'].items()])
        lines.append(f"Proposer gives: {give_str}")
        lines.append(f"Proposer receives: {receive_str}")
        
        # Show responses so far
        if state.pending_trade_responses:
            lines.append("Responses so far:")
            for pid, accepted in state.pending_trade_responses.items():
                responder = next((p for p in state.players if p.id == pid), None)
                responder_name = responder.name if responder else pid
                status = "ACCEPTED" if accepted else "REJECTED"
                lines.append(f"  {responder_name}: {status}")
        
        # Highlight if this player needs to respond
        if player_id in offer['target_player_ids']:
            if player_id not in state.pending_trade_responses:
                lines.append(f"⚠️ YOU NEED TO RESPOND TO THIS TRADE (accept or reject)")
            else:
                lines.append(f"  You have already responded: {'ACCEPTED' if state.pending_trade_responses.get(player_id) else 'REJECTED'}")
        elif player_id == offer['proposer_id']:
            accepting_players = [pid for pid, accepted in state.pending_trade_responses.items() if accepted]
            if len(accepting_players) > 1:
                lines.append(f"⚠️ MULTIPLE PLAYERS ACCEPTED - YOU MUST SELECT A TRADE PARTNER")
            elif len(accepting_players) == 1:
                lines.append(f"  One player accepted - trade will execute automatically")
            else:
                lines.append(f"  Waiting for responses...")
    
    lines.append("")
    
    # Recent history (last 3 actions)
    if history:
        lines.append("=== Recent Actions ===")
        for action, payload in history[-3:]:
            action_str = action.value.replace("_", " ").title()
            if payload:
                if isinstance(payload, BuildSettlementPayload):
                    action_str += f" at intersection {payload.intersection_id}"
                elif isinstance(payload, BuildRoadPayload):
                    action_str += f" on road edge {payload.road_edge_id}"
                elif isinstance(payload, BuildCityPayload):
                    action_str += f" at intersection {payload.intersection_id}"
            lines.append(f"  {action_str}")
        lines.append("")
    
    return "\n".join(lines)


def _calculate_longest_road_length(state: GameState, player_id: str) -> int:
    """Calculate the longest continuous road for a player (same algorithm as engine)."""
    player_roads = [r for r in state.road_edges if r.owner == player_id]
    if not player_roads:
        return 0
    
    # Build graph of connected roads
    road_graph = {}
    for road in player_roads:
        inter1 = road.intersection1_id
        inter2 = road.intersection2_id
        if inter1 not in road_graph:
            road_graph[inter1] = []
        if inter2 not in road_graph:
            road_graph[inter2] = []
        road_graph[inter1].append(inter2)
        road_graph[inter2].append(inter1)
    
    # Find longest path using DFS
    max_length = 0
    
    def dfs_path_length(node: int, visited_nodes: set, visited_edges: set) -> int:
        max_path = 0
        for neighbor in road_graph.get(node, []):
            edge_key = (min(node, neighbor), max(node, neighbor))
            if edge_key not in visited_edges:
                visited_edges.add(edge_key)
                if neighbor not in visited_nodes:
                    path_len = 1 + dfs_path_length(neighbor, visited_nodes | {neighbor}, visited_edges)
                    max_path = max(max_path, path_len)
                visited_edges.remove(edge_key)
        return max_path
    
    for start_node in road_graph.keys():
        path_len = dfs_path_length(start_node, {start_node}, set())
        max_length = max(max_length, path_len)
    
    return max_length


def _calculate_distance_between_intersections(state: GameState, inter1_id: int, inter2_id: int) -> Optional[int]:
    """Calculate shortest path distance between two intersections using BFS."""
    if inter1_id == inter2_id:
        return 0
    
    # Build intersection graph
    inter_graph = {}
    for inter in state.intersections:
        inter_graph[inter.id] = list(inter.adjacent_intersections)
    
    # BFS to find shortest path
    from collections import deque
    queue = deque([(inter1_id, 0)])
    visited = {inter1_id}
    
    while queue:
        current, distance = queue.popleft()
        if current == inter2_id:
            return distance
        
        for neighbor in inter_graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return None  # Not reachable


def _generate_compact_map(state: GameState, player_id: str, tile_map: Dict[int, 'Tile']) -> List[str]:
    """
    Generate a compact ASCII map representation of the Catan board.
    Shows tiles with resources/tokens, buildings, roads, and robber location.
    """
    lines = []
    
    # Create intersection ownership and building maps
    inter_owner_map = {i.id: i.owner for i in state.intersections}
    inter_building_map = {i.id: i.building_type for i in state.intersections}
    inter_port_map = {i.id: i.port_type for i in state.intersections}
    
    # Create tile-to-intersections mapping
    tile_to_intersections = {}
    for inter in state.intersections:
        for tile_id in inter.adjacent_tiles:
            if tile_id not in tile_to_intersections:
                tile_to_intersections[tile_id] = []
            tile_to_intersections[tile_id].append(inter.id)
    
    # Get player names
    player_names = {p.id: p.name for p in state.players}
    
    # Helper to format tile compactly
    def format_tile_compact(tile_id: int) -> str:
        if tile_id not in tile_map:
            return "???"
        tile = tile_map[tile_id]
        if tile.resource_type is None:
            robber = "🔴" if state.robber_tile_id == tile_id else " "
            return f"{robber}DES"
        
        # Resource abbreviations: W=Wood, B=Brick, H=Wheat, S=Sheep, O=Ore
        res_map = {"wood": "W", "brick": "B", "wheat": "H", "sheep": "S", "ore": "O"}
        res_abbr = res_map.get(tile.resource_type.value, tile.resource_type.value[0].upper())
        token = tile.number_token.value if tile.number_token else "?"
        robber = "🔴" if state.robber_tile_id == tile_id else " "
        return f"{robber}{res_abbr}{token:2d}"
    
    # Standard Catan hex layout (3-4-5-4-3 rows)
    # Group tiles by row (r coordinate)
    rows = {}
    for tile in state.tiles:
        q, r = tile.position
        if r not in rows:
            rows[r] = []
        rows[r].append((q, tile.id))
    
    # Sort rows by r (top to bottom: -2, -1, 0, 1, 2)
    sorted_rows = sorted(rows.items(), key=lambda x: x[0])
    
    lines.append("=== Compact Map (Hexagonal Layout) ===")
    lines.append("Format: [🔴=Robber][Resource][Token], e.g., ' W 6' = Wood token 6")
    lines.append("Intersections: I{id}[owner][building][port]")
    lines.append("  Owner: Y=You, 0/1/2/3=Player ID, _=Empty")
    lines.append("  Building: S=Settlement, C=City, _=Empty")
    lines.append("  Port: P=3:1, W/B/H/S/O=Resource port")
    lines.append("")
    
    # Create player ID to short name mapping
    player_short_names = {}
    for i, p in enumerate(state.players):
        if p.id == player_id:
            player_short_names[p.id] = "Y"  # You
        else:
            # Use player index as short name
            player_short_names[p.id] = str(i)
    
    # Display tiles in rows with intersections labeled below
    for r, tiles_in_row in sorted_rows:
        tiles_in_row.sort(key=lambda x: x[0])  # Sort by q coordinate
        
        # Indent based on row (hexagonal pattern)
        indent = abs(r) * 3  # More indent for outer rows
        
        # Format tiles
        tile_strs = []
        for q, tile_id in tiles_in_row:
            tile_str = format_tile_compact(tile_id)
            tile_strs.append(tile_str)
        
        lines.append(" " * indent + "  ".join(tile_strs))
        
        # Show intersections around tiles in this row
        # Prioritize showing intersections with buildings or ports
        intersection_parts = []
        for q, tile_id in tiles_in_row:
            intersections = sorted(tile_to_intersections.get(tile_id, []))
            inter_labels = []
            
            # First pass: collect all intersections with their info
            inter_info = []
            for inter_id in intersections:
                owner = inter_owner_map.get(inter_id)
                building = inter_building_map.get(inter_id)
                port = inter_port_map.get(inter_id)
                
                # Build intersection label: I{id}_[owner][building][port]
                # Format: I19_0S_ = Intersection 19, owned by player 0, Settlement, no port
                label = f"I{inter_id}"
                
                # Add owner indicator (with underscore separator)
                if owner:
                    owner_label = player_short_names.get(owner, "?")
                    label += f"_{owner_label}"
                else:
                    label += "__"  # Empty (two underscores for alignment)
                
                # Add building indicator
                if building:
                    if building == "settlement":
                        label += "S"
                    elif building == "city":
                        label += "C"
                else:
                    label += "_"  # Empty
                
                # Add port indicator
                if port:
                    if port == "3:1":
                        label += "P"
                    else:
                        # Resource port: use first letter capitalized
                        port_abbr = port[0].upper()
                        label += port_abbr
                else:
                    label += "_"  # No port
                
                # Prioritize: buildings > ports > empty
                priority = 0
                if building:
                    priority = 1  # Buildings first
                elif port:
                    priority = 2  # Ports second
                else:
                    priority = 3  # Empty last
                
                inter_info.append((priority, label))
            
            # Sort by priority (buildings first, then ports, then empty)
            inter_info.sort(key=lambda x: x[0])
            
            # Show up to 6 intersections per tile, prioritizing buildings and ports
            # But always show ALL buildings and ports, even if more than 6
            shown_labels = []
            buildings_and_ports = [label for pri, label in inter_info if pri <= 2]
            empty_intersections = [label for pri, label in inter_info if pri == 3]
            
            # Show all buildings/ports first
            shown_labels.extend(buildings_and_ports)
            # Then fill remaining slots with empty intersections (up to 6 total)
            remaining_slots = max(0, 6 - len(shown_labels))
            shown_labels.extend(empty_intersections[:remaining_slots])
            
            if shown_labels:
                intersection_parts.append(" ".join(shown_labels))
            else:
                intersection_parts.append("")  # Empty space for alignment
        
        # Show intersections below tiles with proper spacing
        if any(intersection_parts):
            # Align intersection labels with tiles above
            inter_line_parts = []
            for i, part in enumerate(intersection_parts):
                if part:
                    inter_line_parts.append(part)
                else:
                    inter_line_parts.append("")
            inter_line = "  ".join(inter_line_parts)
            if inter_line.strip():
                lines.append(" " * indent + inter_line)
    
    lines.append("")
    
    # Summary: Your buildings with tile locations
    your_intersections = [i for i in state.intersections if i.owner == player_id]
    if your_intersections:
        lines.append("Your Buildings:")
        for inter in your_intersections[:8]:  # Top 8
            tiles = [f"T{tid}" for tid in sorted(inter.adjacent_tiles)[:3]]
            port_str = f" [PORT:{inter.port_type}]" if inter.port_type else ""
            lines.append(f"  I{inter.id}: {inter.building_type}{port_str} on {','.join(tiles)}")
    
    # Summary: All opponent buildings (with full details)
    opp_intersections = [i for i in state.intersections if i.owner and i.owner != player_id]
    if opp_intersections:
        # Group by player
        buildings_by_player = {}
        for inter in opp_intersections:
            if inter.owner not in buildings_by_player:
                buildings_by_player[inter.owner] = []
            buildings_by_player[inter.owner].append(inter)
        
        for owner_id, buildings in buildings_by_player.items():
            owner_name = player_names.get(owner_id, owner_id)
            owner_short = player_short_names.get(owner_id, "?")
            lines.append(f"{owner_name} (Player {owner_short}) Buildings:")
            for inter in buildings:
                tiles = [f"T{tid}" for tid in sorted(inter.adjacent_tiles)]
                port_str = f" [PORT:{inter.port_type}]" if inter.port_type else ""
                lines.append(f"  I{inter.id}: {inter.building_type}{port_str} on {','.join(tiles)}")
    
    # Road network summary - show both your roads and opponent roads
    your_roads = [r for r in state.road_edges if r.owner == player_id]
    if your_roads:
        road_connections = [f"I{r.intersection1_id}-I{r.intersection2_id}" for r in your_roads[:12]]
        lines.append(f"Your Roads ({len(your_roads)}): {', '.join(road_connections)}")
        if len(your_roads) > 12:
            lines.append(f"  ... and {len(your_roads) - 12} more")
    
    # Show opponent roads too
    opp_roads = [r for r in state.road_edges if r.owner and r.owner != player_id]
    if opp_roads:
        # Group by owner
        roads_by_owner = {}
        for road in opp_roads:
            if road.owner not in roads_by_owner:
                roads_by_owner[road.owner] = []
            roads_by_owner[road.owner].append(road)
        
        for owner_id, owner_roads in roads_by_owner.items():
            owner_name = player_names.get(owner_id, owner_id)
            road_connections = [f"I{r.intersection1_id}-I{r.intersection2_id}" for r in owner_roads[:8]]
            lines.append(f"{owner_name}'s Roads ({len(owner_roads)}): {', '.join(road_connections)}")
            if len(owner_roads) > 8:
                lines.append(f"  ... and {len(owner_roads) - 8} more")
    
    return lines


def legal_actions_to_text(actions: List[Tuple[Action, Optional[ActionPayload]]], state: Optional[GameState] = None, player_id: Optional[str] = None) -> str:
    """
    Convert a list of legal actions to LLM-friendly text with spatial context.
    """
    if not actions:
        return "No legal actions available."
    
    lines = []
    lines.append("=== Legal Actions ===")
    
    # Create lookup maps if state is provided
    inter_map = {}
    road_map = {}
    tile_map = {}
    player_tiles = set()  # Tiles where player has buildings (for informational purposes)
    if state:
        inter_map = {i.id: i for i in state.intersections}
        road_map = {r.id: r for r in state.road_edges}
        tile_map = {t.id: t for t in state.tiles}
        
        # Get tiles where player has buildings (for informational purposes)
        if player_id:
            player_buildings = [i for i in state.intersections if i.owner == player_id]
            for inter in player_buildings:
                player_tiles.update(inter.adjacent_tiles)
    
    # Helper to get intersection context
    def get_intersection_context(inter_id: int) -> str:
        if not state or inter_id not in inter_map:
            return ""
        inter = inter_map[inter_id]
        context_parts = []
        
        # Show adjacent tiles with resources
        if inter.adjacent_tiles:
            tile_info = []
            for tile_id in sorted(inter.adjacent_tiles):
                tile = tile_map.get(tile_id)
                if tile:
                    if tile.resource_type:
                        res_abbr = tile.resource_type.value[0].upper()
                        if tile.resource_type.value == "wheat":
                            res_abbr = "H"
                        token = tile.number_token.value if tile.number_token else "?"
                        tile_info.append(f"T{tile_id}({res_abbr}{token})")
                    else:
                        tile_info.append(f"T{tile_id}(DES)")
            if tile_info:
                context_parts.append(f"tiles: {','.join(tile_info[:3])}")
        
        # Show port if exists
        if inter.port_type:
            context_parts.append(f"PORT:{inter.port_type}")
        
        # Show building if exists
        if inter.owner:
            owner_name = next((p.name for p in state.players if p.id == inter.owner), inter.owner) if state else inter.owner
            building = inter.building_type or "empty"
            if inter.owner == player_id:
                context_parts.append(f"YOUR {building.upper()}")
            else:
                context_parts.append(f"{owner_name}'s {building}")
        
        if context_parts:
            return f" ({'; '.join(context_parts)})"
        return ""
    
    # Helper to get road context
    def get_road_context(road_id: int) -> str:
        if not state or road_id not in road_map:
            return ""
        road = road_map[road_id]
        context_parts = []
        
        # Show which intersections it connects
        inter1 = inter_map.get(road.intersection1_id)
        inter2 = inter_map.get(road.intersection2_id)
        
        if inter1 and inter2:
            # Show tile info for each intersection
            tiles1 = sorted(inter1.adjacent_tiles)[:2] if inter1.adjacent_tiles else []
            tiles2 = sorted(inter2.adjacent_tiles)[:2] if inter2.adjacent_tiles else []
            tile_str1 = ",".join([f"T{tid}" for tid in tiles1])
            tile_str2 = ",".join([f"T{tid}" for tid in tiles2])
            context_parts.append(f"I{road.intersection1_id}({tile_str1})<->I{road.intersection2_id}({tile_str2})")
            
            # Show reachable intersections (1-2 steps away) and their value
            # Determine which intersection is the "destination" (the one you're building toward)
            # In setup, the destination is the one that's NOT the last settlement
            # In playing phase, the destination is the one further from your existing roads/buildings
            destination_inter = None
            if state.phase == "setup" and state.setup_last_settlement_id:
                # Destination is the one that's NOT the last settlement
                if road.intersection1_id == state.setup_last_settlement_id:
                    destination_inter = inter2
                elif road.intersection2_id == state.setup_last_settlement_id:
                    destination_inter = inter1
                else:
                    # Neither is the last settlement - use the one with more adjacent intersections (more expansion options)
                    destination_inter = inter1 if len(inter1.adjacent_intersections) >= len(inter2.adjacent_intersections) else inter2
            else:
                # In playing phase, determine destination based on player's existing roads/buildings
                if player_id:
                    player_buildings = [i.id for i in state.intersections if i.owner == player_id]
                    player_roads = [r for r in state.road_edges if r.owner == player_id]
                    connected_intersections = set(player_buildings)
                    for r in player_roads:
                        connected_intersections.add(r.intersection1_id)
                        connected_intersections.add(r.intersection2_id)
                    
                    # Destination is the intersection further from player's existing network
                    if road.intersection1_id in connected_intersections:
                        destination_inter = inter2
                    elif road.intersection2_id in connected_intersections:
                        destination_inter = inter1
                    else:
                        # Neither is connected - use the one with more expansion options
                        destination_inter = inter1 if len(inter1.adjacent_intersections) >= len(inter2.adjacent_intersections) else inter2
            
            if destination_inter:
                # Find intersections reachable in 1-2 steps from destination
                reachable_1_step = []
                reachable_2_steps = []
                high_value_reachable = []
                
                # 1-step reachable (adjacent to destination)
                for neighbor_id in destination_inter.adjacent_intersections:
                    neighbor = inter_map.get(neighbor_id)
                    if neighbor and neighbor_id != road.intersection1_id and neighbor_id != road.intersection2_id:
                        reachable_1_step.append(neighbor_id)
                        
                        # Check if this neighbor is on a high-value tile (6 or 8)
                        high_value = False
                        for tile_id in neighbor.adjacent_tiles:
                            tile = tile_map.get(tile_id)
                            if tile and tile.number_token and tile.number_token.value in [6, 8]:
                                high_value = True
                                break
                        if high_value:
                            high_value_reachable.append(neighbor_id)
                
                # 2-step reachable (through 1-step neighbors)
                for neighbor_id in reachable_1_step:
                    neighbor = inter_map.get(neighbor_id)
                    if neighbor:
                        for neighbor2_id in neighbor.adjacent_intersections:
                            if (neighbor2_id not in reachable_1_step and 
                                neighbor2_id != destination_inter.id and
                                neighbor2_id != road.intersection1_id and 
                                neighbor2_id != road.intersection2_id):
                                if neighbor2_id not in reachable_2_steps:
                                    reachable_2_steps.append(neighbor2_id)
                                    
                                    # Check if this 2-step neighbor is on a high-value tile
                                    neighbor2 = inter_map.get(neighbor2_id)
                                    if neighbor2:
                                        for tile_id in neighbor2.adjacent_tiles:
                                            tile = tile_map.get(tile_id)
                                            if tile and tile.number_token and tile.number_token.value in [6, 8]:
                                                if neighbor2_id not in high_value_reachable:
                                                    high_value_reachable.append(neighbor2_id)
                                                break
                
                # Build reachability info
                if reachable_1_step or reachable_2_steps:
                    reachable_info = []
                    if reachable_1_step:
                        reachable_info.append(f"1-step: I{','.join(map(str, sorted(reachable_1_step[:5])))}")
                    if reachable_2_steps:
                        reachable_info.append(f"2-step: I{','.join(map(str, sorted(reachable_2_steps[:5])))}")
                    if reachable_info:
                        context_parts.append(f"→{destination_inter.id} leads to: {'; '.join(reachable_info)}")
                    
                    # Highlight high-value intersections
                    if high_value_reachable:
                        context_parts.append(f"high-value(6/8): I{','.join(map(str, sorted(high_value_reachable[:3])))}")
                    
                    # Show ALL intersections on resource tiles (not just high-value ones)
                    # But mark which are available vs occupied/illegal
                    resource_intersections_available = {}
                    resource_intersections_occupied = {}
                    resource_intersections_illegal = {}
                    
                    for neighbor_id in reachable_1_step + reachable_2_steps:
                        neighbor = inter_map.get(neighbor_id)
                        if neighbor:
                            # Check if available
                            is_occupied = neighbor.owner is not None
                            is_too_close = False
                            for adj_id in neighbor.adjacent_intersections:
                                adj_inter = inter_map.get(adj_id)
                                if adj_inter and adj_inter.owner:
                                    is_too_close = True
                                    break
                            
                            for tile_id in neighbor.adjacent_tiles:
                                tile = tile_map.get(tile_id)
                                if tile and tile.resource_type and tile.number_token:
                                    resource_key = f"{tile.resource_type.value}-{tile.number_token.value}"
                                    if is_occupied:
                                        if resource_key not in resource_intersections_occupied:
                                            resource_intersections_occupied[resource_key] = []
                                        if neighbor_id not in resource_intersections_occupied[resource_key]:
                                            resource_intersections_occupied[resource_key].append(neighbor_id)
                                    elif is_too_close:
                                        if resource_key not in resource_intersections_illegal:
                                            resource_intersections_illegal[resource_key] = []
                                        if neighbor_id not in resource_intersections_illegal[resource_key]:
                                            resource_intersections_illegal[resource_key].append(neighbor_id)
                                    else:
                                        if resource_key not in resource_intersections_available:
                                            resource_intersections_available[resource_key] = []
                                        if neighbor_id not in resource_intersections_available[resource_key]:
                                            resource_intersections_available[resource_key].append(neighbor_id)
                    
                    # Show ALL resource types found, clearly marking availability
                    # Sort by resource type, then by number (descending, so 8 comes before 6)
                    all_resources = set(resource_intersections_available.keys()) | \
                                   set(resource_intersections_occupied.keys()) | \
                                   set(resource_intersections_illegal.keys())
                    
                    shown_resources = []
                    for res_key in sorted(all_resources, key=lambda x: (x.split('-')[0], -int(x.split('-')[1]))):
                        parts = []
                        if res_key in resource_intersections_available:
                            inter_list = sorted(resource_intersections_available[res_key])[:5]  # Show up to 5 available
                            if inter_list:
                                parts.append(f"AVAIL:I{','.join(map(str, inter_list))}")
                        if res_key in resource_intersections_occupied:
                            inter_list = sorted(resource_intersections_occupied[res_key])[:3]  # Show up to 3 occupied
                            if inter_list:
                                parts.append(f"OCCUPIED:I{','.join(map(str, inter_list))}")
                        if res_key in resource_intersections_illegal:
                            inter_list = sorted(resource_intersections_illegal[res_key])[:3]  # Show up to 3 too-close
                            if inter_list:
                                parts.append(f"TOO_CLOSE:I{','.join(map(str, inter_list))}")
                        if parts:
                            shown_resources.append(f"{res_key}({';'.join(parts)})")
                    
                    if shown_resources:
                        context_parts.append(f"resources: {'; '.join(shown_resources)}")
                    
                    # Also show availability summary - count actually available intersections
                    available_intersections = []
                    for neighbor_id in reachable_1_step + reachable_2_steps:
                        neighbor = inter_map.get(neighbor_id)
                        if neighbor:
                            is_occupied = neighbor.owner is not None
                            is_too_close = False
                            for adj_id in neighbor.adjacent_intersections:
                                adj_inter = inter_map.get(adj_id)
                                if adj_inter and adj_inter.owner:
                                    is_too_close = True
                                    break
                            if not is_occupied and not is_too_close:
                                available_intersections.append(neighbor_id)
                    
                    available_count = len(available_intersections)
                    if available_count == 0 and (reachable_1_step or reachable_2_steps):
                        context_parts.append("⚠️ NO_AVAILABLE_SPOTS")
                    elif available_count < 3:
                        context_parts.append(f"⚠️ ONLY_{available_count}_AVAILABLE")
        
        if context_parts:
            return f" ({'; '.join(context_parts)})"
        return ""
    
    # Group actions by type
    action_groups = {}
    for action, payload in actions:
        action_key = action.value
        if action_key not in action_groups:
            action_groups[action_key] = []
        action_groups[action_key].append((action, payload))
    
    # Format each group
    for action_key, group in sorted(action_groups.items()):
        action = group[0][0]
        action_name = action.value.replace("_", " ").title()
        
        if len(group) == 1 and group[0][1] is None:
            # Simple action without payload
            lines.append(f"- {action_name}")
        elif len(group) == 1:
            # Single action with payload
            payload = group[0][1]
            if isinstance(payload, BuildSettlementPayload):
                context = get_intersection_context(payload.intersection_id)
                lines.append(f"- {action_name} at intersection {payload.intersection_id}{context}")
            elif isinstance(payload, BuildRoadPayload):
                context = get_road_context(payload.road_edge_id)
                lines.append(f"- {action_name} on road edge {payload.road_edge_id}{context}")
            elif isinstance(payload, BuildCityPayload):
                context = get_intersection_context(payload.intersection_id)
                lines.append(f"- {action_name} at intersection {payload.intersection_id}{context}")
            elif isinstance(payload, PlayDevCardPayload):
                lines.append(f"- {action_name} ({payload.card_type})")
            elif isinstance(payload, TradeBankPayload):
                give_str = ", ".join([f"{count} {rt.value}" for rt, count in payload.give_resources.items()])
                receive_str = ", ".join([f"{count} {rt.value}" for rt, count in payload.receive_resources.items()])
                port_info = f" (via port at intersection {payload.port_intersection_id})" if payload.port_intersection_id else ""
                lines.append(f"- {action_name}: Give {give_str}, receive {receive_str}{port_info}")
            elif isinstance(payload, TradePlayerPayload):
                give_str = ", ".join([f"{count} {rt.value}" for rt, count in payload.give_resources.items()])
                receive_str = ", ".join([f"{count} {rt.value}" for rt, count in payload.receive_resources.items()])
                lines.append(f"- {action_name} with {payload.other_player_id}: Give {give_str}, receive {receive_str}")
        else:
            # Multiple actions of same type
            lines.append(f"- {action_name}:")
            for action, payload in group:
                if isinstance(payload, BuildSettlementPayload):
                    context = get_intersection_context(payload.intersection_id)
                    lines.append(f"  * At intersection {payload.intersection_id}{context}")
                elif isinstance(payload, BuildRoadPayload):
                    context = get_road_context(payload.road_edge_id)
                    lines.append(f"  * On road edge {payload.road_edge_id}{context}")
                elif isinstance(payload, BuildCityPayload):
                    context = get_intersection_context(payload.intersection_id)
                    lines.append(f"  * At intersection {payload.intersection_id}{context}")
                elif isinstance(payload, PlayDevCardPayload):
                    lines.append(f"  * Card: {payload.card_type}")
                elif isinstance(payload, TradeBankPayload):
                    give_str = ", ".join([f"{count} {rt.value}" for rt, count in payload.give_resources.items()])
                    receive_str = ", ".join([f"{count} {rt.value}" for rt, count in payload.receive_resources.items()])
                    port_info = f" (port at {payload.port_intersection_id})" if payload.port_intersection_id else ""
                    lines.append(f"  * Give {give_str}, receive {receive_str}{port_info}")
                elif isinstance(payload, TradePlayerPayload):
                    give_str = ", ".join([f"{count} {rt.value}" for rt, count in payload.give_resources.items()])
                    receive_str = ", ".join([f"{count} {rt.value}" for rt, count in payload.receive_resources.items()])
                    lines.append(f"  * With {payload.other_player_id}: Give {give_str}, receive {receive_str}")
                elif isinstance(payload, MoveRobberPayload):
                    # Show tile info and which players are on it
                    tile = tile_map.get(payload.tile_id) if state and tile_map else None
                    if tile:
                        res_info = f"{tile.resource_type.value} {tile.number_token.value}" if tile.resource_type and tile.number_token else "Desert"
                        # Check which players have buildings on this tile
                        players_on_tile = set()
                        if state:
                            for inter in state.intersections:
                                if payload.tile_id in inter.adjacent_tiles and inter.owner and inter.building_type:
                                    players_on_tile.add(inter.owner)
                        
                        # Note if this is the player's own tile (for informational purposes only)
                        own_tile_note = ""
                        if player_id and payload.tile_id in player_tiles:
                            own_tile_note = " (your buildings also on this tile)"
                        
                        if players_on_tile:
                            player_names = [next((p.name for p in state.players if p.id == pid), pid) for pid in players_on_tile] if state else []
                            lines.append(f"  * To tile {payload.tile_id} ({res_info}) - players on tile: {', '.join(player_names)}{own_tile_note}")
                        else:
                            lines.append(f"  * To tile {payload.tile_id} ({res_info}) - no players on tile{own_tile_note}")
                    else:
                        lines.append(f"  * To tile {payload.tile_id}")
    
    return "\n".join(lines)


def _similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def parse_action_from_text(model_output: str, legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]) -> Tuple[Action, Optional[ActionPayload]]:
    """
    Parse an action from LLM text output using fuzzy matching.
    Robust to variations in formatting and wording.
    
    Returns the best matching (Action, Optional[ActionPayload]) tuple.
    Raises ValueError if no good match is found.
    """
    if not legal_actions_list:
        raise ValueError("No legal actions available")
    
    # Normalize input
    text = model_output.strip().lower()
    
    # Try exact matches first
    for action, payload in legal_actions_list:
        action_name = action.value.replace("_", " ").lower()
        
        # Check if action name appears in text
        if action_name in text or action.value.lower() in text:
            # For actions with payloads, try to extract parameters
            if payload is None:
                return (action, None)
            
            # Try to match payload parameters
            if isinstance(payload, BuildSettlementPayload):
                # Look for intersection ID
                match = re.search(r'intersection\s+(\d+)', text)
                if match:
                    inter_id = int(match.group(1))
                    # Check if this intersection ID matches a legal action
                    for a, p in legal_actions_list:
                        if (a == action and p and 
                            isinstance(p, BuildSettlementPayload) and 
                            p.intersection_id == inter_id):
                            return (a, p)
                # If no match, return the first matching action type
                return (action, payload)
            
            elif isinstance(payload, BuildRoadPayload):
                match = re.search(r'road\s+edge\s+(\d+)', text)
                if match:
                    road_id = int(match.group(1))
                    for a, p in legal_actions_list:
                        if (a == action and p and 
                            isinstance(p, BuildRoadPayload) and 
                            p.road_edge_id == road_id):
                            return (a, p)
                return (action, payload)
            
            elif isinstance(payload, BuildCityPayload):
                match = re.search(r'intersection\s+(\d+)', text)
                if match:
                    inter_id = int(match.group(1))
                    for a, p in legal_actions_list:
                        if (a == action and p and 
                            isinstance(p, BuildCityPayload) and 
                            p.intersection_id == inter_id):
                            return (a, p)
                return (action, payload)
            
            elif isinstance(payload, PlayDevCardPayload):
                # Try to match card type
                for card_type in ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"]:
                    if card_type in text:
                        for a, p in legal_actions_list:
                            if (a == action and p and 
                                isinstance(p, PlayDevCardPayload) and 
                                p.card_type == card_type):
                                return (a, p)
                return (action, payload)
            
            else:
                return (action, payload)
    
    # Fuzzy matching: find best similarity
    best_match = None
    best_score = 0.0
    threshold = 0.3  # Minimum similarity threshold
    
    for action, payload in legal_actions_list:
        action_name = action.value.replace("_", " ").lower()
        score = _similarity(text, action_name)
        
        # Boost score if key words match
        if "build" in text and "build" in action_name:
            score += 0.2
        if "settlement" in text and "settlement" in action_name:
            score += 0.2
        if "road" in text and "road" in action_name:
            score += 0.2
        if "city" in text and "city" in action_name:
            score += 0.2
        if "trade" in text and "trade" in action_name:
            score += 0.2
        if "end" in text and "end" in action_name:
            score += 0.2
        
        # For actions with payloads, try to extract and match IDs
        if payload:
            if isinstance(payload, (BuildSettlementPayload, BuildCityPayload)):
                match = re.search(r'(\d+)', text)
                if match:
                    inter_id = int(match.group(1))
                    if payload.intersection_id == inter_id:
                        score += 0.3
            elif isinstance(payload, BuildRoadPayload):
                match = re.search(r'(\d+)', text)
                if match:
                    road_id = int(match.group(1))
                    if payload.road_edge_id == road_id:
                        score += 0.3
        
        if score > best_score:
            best_score = score
            best_match = (action, payload)
    
    if best_match and best_score >= threshold:
        return best_match
    
    # If still no match, return the first legal action as fallback
    # (or raise error - let's raise for now to be safe)
    raise ValueError(f"Could not parse action from text: '{model_output}'. "
                    f"Best match had similarity {best_score:.2f} (threshold: {threshold})")

