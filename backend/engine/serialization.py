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
    MoveRobberPayload,
    StealResourcePayload,
    DiscardResourcesPayload,
)


def serialize_game_state(state: GameState) -> Dict[str, Any]:
    """
    Serialize GameState to a JSON-serializable dictionary.
    """
    return {
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
        "robber_tile_id": state.robber_tile_id,
        "waiting_for_robber_move": state.waiting_for_robber_move,
        "waiting_for_robber_steal": state.waiting_for_robber_steal,
    }


def deserialize_game_state(data: Dict[str, Any]) -> GameState:
    """
    Deserialize a dictionary to GameState.
    """
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
        robber_tile_id=data.get("robber_tile_id"),
        waiting_for_robber_move=data.get("waiting_for_robber_move", False),
        waiting_for_robber_steal=data.get("waiting_for_robber_steal", False),
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
        return {
            "type": "PlayDevCardPayload",
            "card_type": payload.card_type,
        }
    elif isinstance(payload, TradeBankPayload):
        return {
            "type": "TradeBankPayload",
            "give_resource": payload.give_resource.value,
            "give_amount": payload.give_amount,
            "receive_resource": payload.receive_resource.value,
            "receive_amount": payload.receive_amount,
        }
    elif isinstance(payload, TradePlayerPayload):
        return {
            "type": "TradePlayerPayload",
            "other_player_id": payload.other_player_id,
            "give_resource": payload.give_resource.value,
            "give_amount": payload.give_amount,
            "receive_resource": payload.receive_resource.value,
            "receive_amount": payload.receive_amount,
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
        return PlayDevCardPayload(card_type=data["card_type"])
    elif payload_type == "TradeBankPayload":
        return TradeBankPayload(
            give_resource=ResourceType(data["give_resource"]),
            give_amount=data["give_amount"],
            receive_resource=ResourceType(data["receive_resource"]),
            receive_amount=data["receive_amount"],
        )
    elif payload_type == "TradePlayerPayload":
        return TradePlayerPayload(
            other_player_id=data["other_player_id"],
            give_resource=ResourceType(data["give_resource"]),
            give_amount=data["give_amount"],
            receive_resource=ResourceType(data["receive_resource"]),
            receive_amount=data["receive_amount"],
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
        # Can place settlement if not placed yet in this round
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
        
        # Can place road adjacent to settlement
        player_settlements = [i for i in state.intersections if i.owner == player_id]
        for settlement in player_settlements:
            for road_edge in state.road_edges:
                if not road_edge.owner:
                    # Check if road is adjacent to settlement
                    if (road_edge.intersection1_id == settlement.id or 
                        road_edge.intersection2_id == settlement.id):
                        legal.append((Action.SETUP_PLACE_ROAD, BuildRoadPayload(road_edge.id)))
        
        # Can start game if setup is complete (simplified - in real game, this is automatic)
        if state.setup_round == 1 and state.setup_phase_player_index == len(state.players) - 1:
            legal.append((Action.START_GAME, None))
    
    elif state.phase == "playing":
        # Playing phase actions
        if state.dice_roll is None:
            # Must roll dice first
            legal.append((Action.ROLL_DICE, None))
        elif state.dice_roll == 7:
            # Handle rolling 7
            # First phase: ALL players with 8+ resources must discard
            # Check if ANY player needs to discard
            any_player_needs_discard = False
            for p in state.players:
                if sum(p.resources.values()) >= 8:
                    any_player_needs_discard = True
                    break
            
            if any_player_needs_discard:
                # This player can discard if they have 8+ resources (even if not their turn)
                total_resources = sum(player.resources.values())
                if total_resources >= 8:
                    # Player must discard half their resources (rounded down)
                    legal.append((Action.DISCARD_RESOURCES, None))  # Payload will be provided by frontend
                # Don't show other actions while discarding
                return legal
            
            # All discards done, handle robber phase
            if state.waiting_for_robber_move:
                # All discards done, now ONLY the player who rolled the 7 can move robber
                if is_current_player:
                    # Can move robber to any tile except current
                    for tile in state.tiles:
                        if tile.id != state.robber_tile_id:
                            legal.append((Action.MOVE_ROBBER, MoveRobberPayload(tile.id)))
                # Don't show other actions while waiting to move robber
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
                        
                        for other_player_id in players_on_tile:
                            other_player = next((p for p in state.players if p.id == other_player_id), None)
                            if other_player and sum(other_player.resources.values()) > 0:
                                legal.append((Action.STEAL_RESOURCE, StealResourcePayload(other_player_id)))
                # Don't show other actions while waiting to steal
                return legal
            
            # If we get here, robber phase is complete - fall through to normal turn actions
        
        # Normal turn actions (after dice roll, or after completing robber phase on 7)
        if state.dice_roll is not None and is_current_player:
            # Can build road
            if (player.resources[ResourceType.WOOD] >= 1 and  
                player.resources[ResourceType.BRICK] >= 1):
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
            
            # Can play dev cards
            for card_type in player.dev_cards:
                legal.append((Action.PLAY_DEV_CARD, PlayDevCardPayload(card_type)))
            
            # Can trade with bank (4:1)
            for give_rt in ResourceType:
                if player.resources[give_rt] >= 4:
                    for receive_rt in ResourceType:
                        if give_rt != receive_rt:
                            legal.append((Action.TRADE_BANK, TradeBankPayload(
                                give_resource=give_rt,
                                give_amount=4,
                                receive_resource=receive_rt,
                                receive_amount=1,
                            )))
            
            # Can trade with other players
            for other_player in state.players:
                if other_player.id != player_id:
                    for give_rt in ResourceType:
                        if player.resources[give_rt] > 0:
                            for receive_rt in ResourceType:
                                if give_rt != receive_rt and other_player.resources[receive_rt] > 0:
                                    # Simplified: allow 1:1 trades
                                    legal.append((Action.TRADE_PLAYER, TradePlayerPayload(
                                        other_player_id=other_player.id,
                                        give_resource=give_rt,
                                        give_amount=1,
                                        receive_resource=receive_rt,
                                        receive_amount=1,
                                    )))
            
            # Can end turn (but not if waiting for robber actions, or if 7 was rolled and anyone needs to discard)
            if state.dice_roll == 7:
                # Check if anyone still needs to discard
                any_player_needs_discard = False
                for p in state.players:
                    if sum(p.resources.values()) >= 8:
                        any_player_needs_discard = True
                        break
                
                # Can only end turn if no one needs to discard AND robber actions are complete
                if not any_player_needs_discard and not state.waiting_for_robber_move and not state.waiting_for_robber_steal:
                    legal.append((Action.END_TURN, None))
            elif not state.waiting_for_robber_move and not state.waiting_for_robber_steal:
                legal.append((Action.END_TURN, None))
    
    return legal


def state_to_text(state: GameState, player_id: str, history: List[Tuple[Action, Optional[ActionPayload]]] = None) -> str:
    """
    Convert game state to LLM-friendly text description from a player's perspective.
    """
    if history is None:
        history = []
    
    lines = []
    
    # Find the player
    player = next((p for p in state.players if p.id == player_id), None)
    if not player:
        return f"Player {player_id} not found in game"
    
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
    
    # Other players
    lines.append("=== Other Players ===")
    for p in state.players:
        if p.id != player_id:
            lines.append(f"{p.name}: {p.victory_points} VP, "
                        f"{sum(p.resources.values())} resources, "
                        f"{p.settlements_built} settlements, {p.cities_built} cities")
    
    lines.append("")
    
    # Board state
    lines.append("=== Board ===")
    lines.append(f"Tiles: {len(state.tiles)}")
    for tile in state.tiles:
        if tile.resource_type:
            token_str = f" (token: {tile.number_token.value})" if tile.number_token else ""
            lines.append(f"  Tile {tile.id}: {tile.resource_type.value}{token_str}")
        else:
            lines.append(f"  Tile {tile.id}: Desert")
    
    lines.append(f"Intersections: {len(state.intersections)}")
    your_buildings = [i for i in state.intersections if i.owner == player_id]
    if your_buildings:
        lines.append("  Your buildings:")
        for inter in your_buildings:
            lines.append(f"    Intersection {inter.id}: {inter.building_type}")
    
    lines.append(f"Roads: {len(state.road_edges)}")
    your_roads = [r for r in state.road_edges if r.owner == player_id]
    if your_roads:
        lines.append(f"  Your roads: {len(your_roads)}")
    
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
                lines.append(f"- {action_name}: Give {payload.give_amount} {payload.give_resource.value}, "
                           f"receive {payload.receive_amount} {payload.receive_resource.value}")
            elif isinstance(payload, TradePlayerPayload):
                lines.append(f"- {action_name} with {payload.other_player_id}: "
                           f"Give {payload.give_amount} {payload.give_resource.value}, "
                           f"receive {payload.receive_amount} {payload.receive_resource.value}")
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
                    lines.append(f"  * Give {payload.give_amount} {payload.give_resource.value}, "
                               f"receive {payload.receive_amount} {payload.receive_resource.value}")
                elif isinstance(payload, TradePlayerPayload):
                    lines.append(f"  * With {payload.other_player_id}: "
                               f"Give {payload.give_amount} {payload.give_resource.value}, "
                               f"receive {payload.receive_amount} {payload.receive_resource.value}")
    
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

