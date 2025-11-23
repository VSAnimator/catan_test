"""
Serialization and LLM-friendly text conversion for the Catan game engine.
"""
import json
import re
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
            
            # Player trades are now constructed via the UI, not auto-generated
            # This allows for multi-resource trades with custom give/receive amounts
            
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
    
    lines.append("")
    
    # Your status
    lines.append(f"=== Your Status ({player.name}) ===")
    lines.append(f"Victory Points: {player.victory_points}")
    lines.append(f"Resources:")
    for rt in ResourceType:
        lines.append(f"  {rt.value.capitalize()}: {player.resources[rt]}")
    lines.append(f"Buildings: {player.settlements_built} settlements, {player.cities_built} cities")
    lines.append(f"Roads: {player.roads_built}")
    lines.append(f"Dev Cards: {len(player.dev_cards)}")
    if player.dev_cards:
        lines.append(f"  Types: {', '.join(player.dev_cards)}")
    if player.longest_road:
        lines.append("  * Longest Road")
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
            lines.append(f"  Buildings: {p.settlements_built} settlements, {p.cities_built} cities")
            lines.append(f"  Roads: {p.roads_built}")
            if p.longest_road:
                lines.append("  * Has Longest Road")
            if p.largest_army:
                lines.append("  * Has Largest Army")
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
                
                candidates.append((inter.id, total_production, resource_prod, port_bonus, inter.adjacent_intersections))
        
        # Sort by production value (descending)
        candidates.sort(key=lambda x: -x[1])
        
        # Show top 15 candidates
        for inter_id, prod, res_prod, port, adj_inters in candidates[:15]:
            lines.append(f"  Intersection {inter_id}{port}:")
            lines.append(f"    Production Value: {prod*100:.1f}% per roll")
            if res_prod:
                res_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in sorted(res_prod.items(), key=lambda x: -x[1])])
                lines.append(f"    Resources: {res_str}")
            if adj_inters:
                adj_str = ", ".join([str(aid) for aid in sorted(adj_inters)])
                lines.append(f"    Adjacent Intersections: {adj_str}")
    
    lines.append("")
    
    # Road network information
    lines.append("=== Road Network ===")
    your_roads = [r for r in state.road_edges if r.owner == player_id]
    if your_roads:
        lines.append(f"Your Roads ({len(your_roads)}):")
        for road in your_roads:
            lines.append(f"  Road {road.id}: connects intersections {road.intersection1_id} <-> {road.intersection2_id}")
    
    opponent_roads = [r for r in state.road_edges if r.owner and r.owner != player_id]
    if opponent_roads:
        lines.append(f"Opponent Roads ({len(opponent_roads)} total)")
    
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


def legal_actions_to_text(actions: List[Tuple[Action, Optional[ActionPayload]]]) -> str:
    """
    Convert a list of legal actions to LLM-friendly text.
    """
    if not actions:
        return "No legal actions available."
    
    lines = []
    lines.append("=== Legal Actions ===")
    
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
                lines.append(f"- {action_name} at intersection {payload.intersection_id}")
            elif isinstance(payload, BuildRoadPayload):
                lines.append(f"- {action_name} on road edge {payload.road_edge_id}")
            elif isinstance(payload, BuildCityPayload):
                lines.append(f"- {action_name} at intersection {payload.intersection_id}")
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
                    lines.append(f"  * At intersection {payload.intersection_id}")
                elif isinstance(payload, BuildRoadPayload):
                    lines.append(f"  * On road edge {payload.road_edge_id}")
                elif isinstance(payload, BuildCityPayload):
                    lines.append(f"  * At intersection {payload.intersection_id}")
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

