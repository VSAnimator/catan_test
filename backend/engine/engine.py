"""
Pure game engine for Catan-like game.
No I/O, no globals - pure functional game logic.
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import random
import copy
import math


class ResourceType(Enum):
    """Resource types in the game."""
    WOOD = "wood"
    BRICK = "brick"
    WHEAT = "wheat"
    SHEEP = "sheep"
    ORE = "ore"


@dataclass(frozen=True)
class NumberToken:
    """Number token on a tile (2-12, excluding 7)."""
    value: int
    
    def __post_init__(self):
        if self.value < 2 or self.value > 12 or self.value == 7:
            raise ValueError(f"Number token must be 2-6 or 8-12, got {self.value}")


@dataclass(frozen=True)
class Tile:
    """A hex tile on the board."""
    id: int
    resource_type: Optional[ResourceType]  # None for desert
    number_token: Optional[NumberToken]  # None for desert
    position: Tuple[int, int]  # Hex coordinates (q, r)


@dataclass(frozen=True)
class Intersection:
    """An intersection (vertex) where settlements/cities can be built."""
    id: int
    position: Tuple[float, float]  # Pixel or hex coordinates
    adjacent_tiles: Set[int] = field(default_factory=set)  # Tile IDs
    adjacent_intersections: Set[int] = field(default_factory=set)  # Intersection IDs
    owner: Optional[str] = None  # Player ID
    building_type: Optional[str] = None  # "settlement" or "city"
    port_type: Optional[str] = None  # None = no port, "3:1" = generic port, or resource type (e.g., "sheep") for 2:1 port


@dataclass(frozen=True)
class RoadEdge:
    """A road edge connecting two intersections."""
    id: int
    intersection1_id: int
    intersection2_id: int
    owner: Optional[str] = None  # Player ID


@dataclass
class Player:
    """Represents a player in the game."""
    id: str
    name: str
    color: str = "blue"  # Player color for UI display
    resources: Dict[ResourceType, int] = field(default_factory=lambda: {
        ResourceType.WOOD: 0,
        ResourceType.BRICK: 0,
        ResourceType.WHEAT: 0,
        ResourceType.SHEEP: 0,
        ResourceType.ORE: 0,
    })
    victory_points: int = 0
    roads_built: int = 0
    settlements_built: int = 0
    cities_built: int = 0
    dev_cards: List[str] = field(default_factory=list)  # List of dev card types
    knights_played: int = 0  # Number of knight cards played
    longest_road: bool = False
    largest_army: bool = False


@dataclass
class GameState:
    """Current state of the game - immutable operations via step()."""
    game_id: str
    players: List[Player]
    current_player_index: int = 0
    phase: str = "setup"  # "setup", "playing", "finished"
    tiles: List[Tile] = field(default_factory=list)
    intersections: List[Intersection] = field(default_factory=list)
    road_edges: List[RoadEdge] = field(default_factory=list)
    dice_roll: Optional[int] = None  # Last dice roll (2-12)
    turn_number: int = 0
    setup_round: int = 0  # 0 = first round (clockwise), 1 = second round (counter-clockwise)
    setup_phase_player_index: int = 0  # Current player in setup phase
    setup_last_settlement_id: Optional[int] = None  # Intersection ID of the last settlement placed in setup (for road placement)
    setup_first_settlement_player_index: Optional[int] = None  # Index of player who placed first settlement (goes first after setup)
    robber_tile_id: Optional[int] = None  # Tile ID where robber is located (None = desert)
    waiting_for_robber_move: bool = False  # True if robber needs to be moved (after 7 or knight)
    waiting_for_robber_steal: bool = False  # True if resource needs to be stolen (after moving robber)
    players_discarded: Set[str] = field(default_factory=set)  # Players who have already discarded this turn (when 7 is rolled)
    robber_initial_tile_id: Optional[int] = None  # Robber position when 7 was rolled (to detect if it's been moved)
    roads_from_road_building: Dict[str, int] = field(default_factory=dict)  # Player ID -> number of free roads remaining from road building card
    dev_cards_bought_this_turn: Set[str] = field(default_factory=set)  # Player IDs who bought dev cards this turn
    dev_cards_played_this_turn: Set[str] = field(default_factory=set)  # Player IDs who played dev cards this turn
    
    # Trade state
    pending_trade_offer: Optional[Dict] = None  # Current trade offer: {proposer_id, target_player_ids, give_resources, receive_resources}
    pending_trade_responses: Dict[str, bool] = field(default_factory=dict)  # Player ID -> True if accepted, False if rejected
    pending_trade_current_responder_index: int = 0  # Index into target_player_ids for current responder
    
    # Turn tracking
    actions_taken_this_turn: List[Dict[str, Any]] = field(default_factory=list)  # List of actions taken this turn: [{"player_id": str, "action": str, "payload": dict}]
    consecutive_rejected_trades: Dict[str, int] = field(default_factory=dict)  # Player ID -> count of consecutive rejected trades this turn
    
    # Resource card counts (19 of each resource type available)
    resource_card_counts: Dict[ResourceType, int] = field(default_factory=lambda: {
        ResourceType.WOOD: 19,
        ResourceType.BRICK: 19,
        ResourceType.WHEAT: 19,
        ResourceType.SHEEP: 19,
        ResourceType.ORE: 19,
    })
    
    # Development card counts (finite supply)
    dev_card_counts: Dict[str, int] = field(default_factory=lambda: {
        "year_of_plenty": 2,
        "monopoly": 2,
        "road_building": 2,
        "victory_point": 5,
        "knight": 14,
    })
    
    def step(self, action: 'Action', payload: Optional['ActionPayload'] = None, player_id: Optional[str] = None) -> 'GameState':
        """
        Pure function that takes an action and returns a new GameState.
        No side effects, no I/O, no globals.
        
        Args:
            action: The action to perform
            payload: Optional payload for the action
            player_id: Optional player ID (used for actions like DISCARD_RESOURCES that can be done out of turn)
        """
        new_state = copy.deepcopy(self)
        
        # Track the current player BEFORE handlers potentially change it
        # (e.g., propose_trade changes current_player_index to target player)
        current_player_before = new_state.players[new_state.current_player_index] if new_state.phase == "playing" else None
        
        if action == Action.ROLL_DICE:
            result = self._handle_roll_dice(new_state)
        elif action == Action.BUILD_ROAD:
            result = self._handle_build_road(new_state, payload)
        elif action == Action.BUILD_SETTLEMENT:
            result = self._handle_build_settlement(new_state, payload)
        elif action == Action.BUILD_CITY:
            result = self._handle_build_city(new_state, payload)
        elif action == Action.BUY_DEV_CARD:
            result = self._handle_buy_dev_card(new_state)
        elif action == Action.PLAY_DEV_CARD:
            result = self._handle_play_dev_card(new_state, payload)
        elif action == Action.TRADE_BANK:
            result = self._handle_trade_bank(new_state, payload)
        elif action == Action.TRADE_PLAYER:
            result = self._handle_trade_player(new_state, payload)
        elif action == Action.PROPOSE_TRADE:
            result = self._handle_propose_trade(new_state, payload)
        elif action == Action.ACCEPT_TRADE:
            result = self._handle_accept_trade(new_state)
        elif action == Action.REJECT_TRADE:
            result = self._handle_reject_trade(new_state)
        elif action == Action.SELECT_TRADE_PARTNER:
            result = self._handle_select_trade_partner(new_state, payload)
        elif action == Action.END_TURN:
            result = self._handle_end_turn(new_state)
        elif action == Action.START_GAME:
            result = self._handle_start_game(new_state)
        elif action == Action.SETUP_PLACE_SETTLEMENT:
            result = self._handle_setup_place_settlement(new_state, payload)
        elif action == Action.SETUP_PLACE_ROAD:
            result = self._handle_setup_place_road(new_state, payload)
        elif action == Action.MOVE_ROBBER:
            result = self._handle_move_robber(new_state, payload)
        elif action == Action.STEAL_RESOURCE:
            result = self._handle_steal_resource(new_state, payload)
        elif action == Action.DISCARD_RESOURCES:
            result = self._handle_discard_resources(new_state, payload, player_id)
        else:
            raise ValueError(f"Unknown action: {action}")
        
        # Track action taken this turn (only for playing phase, and only for current player's actions)
        # Use current_player_before because handlers may change current_player_index (e.g., propose_trade)
        if result.phase == "playing" and player_id is None and current_player_before:
            # Convert payload to dict for storage
            payload_dict = None
            if payload:
                def serialize_value(v):
                    """Recursively serialize values, handling ResourceType enums in dict keys."""
                    if hasattr(v, 'value'):
                        return v.value
                    elif isinstance(v, dict):
                        # Handle dicts with ResourceType enum keys
                        result = {}
                        for k, val in v.items():
                            # Convert enum keys to strings
                            key = k.value if hasattr(k, 'value') else k
                            # Recursively serialize values
                            result[key] = serialize_value(val)
                        return result
                    elif isinstance(v, list):
                        return [serialize_value(item) for item in v]
                    else:
                        return v
                
                payload_dict = {
                    k: serialize_value(v)
                    for k, v in payload.__dict__.items()
                }
            result.actions_taken_this_turn.append({
                "player_id": current_player_before.id,
                "action": action.value,
                "payload": payload_dict
            })
        
        # Check for victory condition after any action (first player to reach 10 VPs wins)
        # Only check during playing phase (can't reach 10 VPs during setup - max is 2 from 2 settlements)
        if result.phase == "playing":
            for player in result.players:
                if player.victory_points >= 10:
                    result.phase = "finished"
                    break
        
        return result
    
    def _handle_roll_dice(self, new_state: 'GameState') -> 'GameState':
        """Handle dice roll - distribute resources based on number, or handle 7."""
        if new_state.phase != "playing":
            raise ValueError("Can only roll dice during playing phase")
        
        # Can't roll if already rolled this turn
        if new_state.dice_roll is not None:
            raise ValueError("Dice already rolled this turn")
        
        # Roll two 6-sided dice and sum them (result: 2-12, with proper distribution)
        # 7 is most likely (6 ways), 2 and 12 are least likely (1 way each)
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        roll = die1 + die2
        new_state.dice_roll = roll
        
        # Handle rolling 7
        if roll == 7:
            # Reset the discarded players set for this new 7 roll
            new_state.players_discarded = set()
            # Store the initial robber position to detect if it's been moved
            new_state.robber_initial_tile_id = new_state.robber_tile_id
            # Check if anyone needs to discard
            any_player_needs_discard = False
            for p in new_state.players:
                if sum(p.resources.values()) >= 8:
                    any_player_needs_discard = True
                    break
            
            # If no one needs to discard, set waiting_for_robber_move immediately
            if not any_player_needs_discard:
                new_state.waiting_for_robber_move = True
            # Otherwise, wait for all players to discard before setting the flag
        else:
        # Distribute resources if not 7
            self._distribute_resources(new_state, roll)
        
        return new_state
    
    def _distribute_resources(self, state: 'GameState', roll: int):
        """Distribute resources to players based on dice roll.
        
        Resources are additive - if a player has multiple buildings on a tile,
        they get resources for each building (1 per settlement, 2 per city).
        
        Resource scarcity: If there aren't enough cards available to distribute
        to all players, no one gets resources from that tile.
        """
        for tile in state.tiles:
            # Skip if robber is on this tile
            if state.robber_tile_id == tile.id:
                continue
                
            if tile.number_token and tile.number_token.value == roll and tile.resource_type:
                # Count all buildings on this tile per player
                player_resources = {}  # player_id -> amount to give
                
                # Find all intersections on this tile that have settlements/cities
                for intersection in state.intersections:
                    # Check if this intersection is adjacent to this tile
                    if tile.id in intersection.adjacent_tiles:
                        # Check if intersection has a building
                        if intersection.owner and intersection.building_type:
                            player_id = intersection.owner
                            if player_id not in player_resources:
                                player_resources[player_id] = 0
                            
                            # Add resources based on building type (additive)
                            if intersection.building_type == "settlement":
                                player_resources[player_id] += 1
                            elif intersection.building_type == "city":
                                player_resources[player_id] += 2
                
                # Calculate total resources needed
                total_needed = sum(player_resources.values())
                
                # Check if there are enough cards available
                available_cards = state.resource_card_counts.get(tile.resource_type, 0)
                
                # Only distribute if we have enough cards for everyone
                if total_needed > 0 and available_cards >= total_needed:
                    # Distribute the resources
                    for player_id, amount in player_resources.items():
                        player = next(p for p in state.players if p.id == player_id)
                        player.resources[tile.resource_type] += amount
                    
                    # Decrement available cards
                    state.resource_card_counts[tile.resource_type] -= total_needed
                # If not enough cards, no one gets resources (scarcity rule)
    
    def _handle_build_road(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle building a road."""
        if payload is None or not isinstance(payload, BuildRoadPayload):
            raise ValueError("BUILD_ROAD requires BuildRoadPayload")
        
        current_player = new_state.players[new_state.current_player_index]
        
        # Check if this is a free road from road building card
        free_roads_remaining = new_state.roads_from_road_building.get(current_player.id, 0)
        using_road_building = free_roads_remaining > 0
        
        # Check resources: 1 wood, 1 brick (unless using road building card)
        if not using_road_building:
            if (current_player.resources[ResourceType.WOOD] < 1 or 
                current_player.resources[ResourceType.BRICK] < 1):
                raise ValueError("Insufficient resources to build road")
        
        # Check if road edge exists and is unowned
        road_edge = next((r for r in new_state.road_edges if r.id == payload.road_edge_id), None)
        if not road_edge:
            raise ValueError(f"Road edge {payload.road_edge_id} not found")
        if road_edge.owner:
            raise ValueError(f"Road edge {payload.road_edge_id} already owned")
        
        # Check that road doesn't go through opponent buildings
        inter1 = next((i for i in new_state.intersections if i.id == road_edge.intersection1_id), None)
        inter2 = next((i for i in new_state.intersections if i.id == road_edge.intersection2_id), None)
        
        if inter1 and inter1.owner and inter1.owner != current_player.id:
            raise ValueError(f"Cannot build road through opponent building at intersection {road_edge.intersection1_id}")
        if inter2 and inter2.owner and inter2.owner != current_player.id:
            raise ValueError(f"Cannot build road through opponent building at intersection {road_edge.intersection2_id}")
        
        # Check that road connects to player's infrastructure
        has_connection = False
        # Check if either intersection is owned by player
        if inter1 and inter1.owner == current_player.id:
            has_connection = True
        if inter2 and inter2.owner == current_player.id:
            has_connection = True
        
        # Check if adjacent to player's roads
        if not has_connection:
            player_roads = [r for r in new_state.road_edges if r.owner == current_player.id]
            for player_road in player_roads:
                if (road_edge.intersection1_id in [player_road.intersection1_id, player_road.intersection2_id] or
                    road_edge.intersection2_id in [player_road.intersection1_id, player_road.intersection2_id]):
                    # Verify connection point doesn't have opponent building
                    connection_point = None
                    if road_edge.intersection1_id in [player_road.intersection1_id, player_road.intersection2_id]:
                        connection_point = road_edge.intersection1_id
                    elif road_edge.intersection2_id in [player_road.intersection1_id, player_road.intersection2_id]:
                        connection_point = road_edge.intersection2_id
                    
                    if connection_point:
                        conn_inter = next((i for i in new_state.intersections if i.id == connection_point), None)
                        if not conn_inter or not conn_inter.owner or conn_inter.owner == current_player.id:
                            has_connection = True
                            break
        
        if not has_connection:
            raise ValueError(f"Road edge {payload.road_edge_id} does not connect to player's infrastructure")
        
        # Deduct resources (unless using road building card)
        if not using_road_building:
            current_player.resources[ResourceType.WOOD] -= 1
            current_player.resources[ResourceType.BRICK] -= 1
            # Return resources to bank
            new_state.resource_card_counts[ResourceType.WOOD] += 1
            new_state.resource_card_counts[ResourceType.BRICK] += 1
        else:
            # Decrement free roads remaining
            new_state.roads_from_road_building[current_player.id] = free_roads_remaining - 1
            if new_state.roads_from_road_building[current_player.id] == 0:
                # Remove from dict when exhausted
                del new_state.roads_from_road_building[current_player.id]
        
        # Build road
        road_index = next(i for i, r in enumerate(new_state.road_edges) if r.id == payload.road_edge_id)
        new_state.road_edges[road_index] = RoadEdge(
            id=road_edge.id,
            intersection1_id=road_edge.intersection1_id,
            intersection2_id=road_edge.intersection2_id,
            owner=current_player.id
        )
        current_player.roads_built += 1
        
        # Check for longest road
        self._check_longest_road(new_state, current_player)
        
        return new_state
    
    def _handle_build_settlement(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle building a settlement."""
        if payload is None or not isinstance(payload, BuildSettlementPayload):
            raise ValueError("BUILD_SETTLEMENT requires BuildSettlementPayload")
        
        current_player = new_state.players[new_state.current_player_index]
        
        # Check resources: 1 wood, 1 brick, 1 wheat, 1 sheep
        required = {
            ResourceType.WOOD: 1,
            ResourceType.BRICK: 1,
            ResourceType.WHEAT: 1,
            ResourceType.SHEEP: 1,
        }
        for resource, amount in required.items():
            if current_player.resources[resource] < amount:
                raise ValueError(f"Insufficient {resource.value} to build settlement")
        
        # Check if intersection exists and is unowned
        intersection = next((i for i in new_state.intersections if i.id == payload.intersection_id), None)
        if not intersection:
            raise ValueError(f"Intersection {payload.intersection_id} not found")
        if intersection.owner:
            raise ValueError(f"Intersection {payload.intersection_id} already owned")
        
        # Check distance rule: no adjacent settlements
        for adj_id in intersection.adjacent_intersections:
            adj_intersection = next((i for i in new_state.intersections if i.id == adj_id), None)
            if adj_intersection and adj_intersection.owner:
                raise ValueError("Cannot build settlement adjacent to another settlement")
        
        # Deduct resources
        for resource, amount in required.items():
            current_player.resources[resource] -= amount
            # Return resources to bank
            new_state.resource_card_counts[resource] += amount
        
        # Build settlement
        intersection_index = next(i for i, inter in enumerate(new_state.intersections) if inter.id == payload.intersection_id)
        new_state.intersections[intersection_index] = Intersection(
            id=intersection.id,
            position=intersection.position,
            adjacent_tiles=intersection.adjacent_tiles,
            adjacent_intersections=intersection.adjacent_intersections,
            owner=current_player.id,
            building_type="settlement",
            port_type=intersection.port_type  # Preserve port type
        )
        current_player.settlements_built += 1
        current_player.victory_points += 1
        
        return new_state
    
    def _handle_build_city(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle upgrading a settlement to a city."""
        if payload is None or not isinstance(payload, BuildCityPayload):
            raise ValueError("BUILD_CITY requires BuildCityPayload")
        
        current_player = new_state.players[new_state.current_player_index]
        
        # Check resources: 2 wheat, 3 ore
        if (current_player.resources[ResourceType.WHEAT] < 2 or 
            current_player.resources[ResourceType.ORE] < 3):
            raise ValueError("Insufficient resources to build city")
        
        # Check if intersection exists and has player's settlement
        intersection = next((i for i in new_state.intersections if i.id == payload.intersection_id), None)
        if not intersection:
            raise ValueError(f"Intersection {payload.intersection_id} not found")
        if intersection.owner != current_player.id:
            raise ValueError("Can only upgrade your own settlement")
        if intersection.building_type != "settlement":
            raise ValueError("Can only upgrade settlements to cities")
        
        # Deduct resources
        current_player.resources[ResourceType.WHEAT] -= 2
        current_player.resources[ResourceType.ORE] -= 3
        # Return resources to bank
        new_state.resource_card_counts[ResourceType.WHEAT] += 2
        new_state.resource_card_counts[ResourceType.ORE] += 3
        
        # Upgrade to city
        intersection_index = next(i for i, inter in enumerate(new_state.intersections) if inter.id == payload.intersection_id)
        new_state.intersections[intersection_index] = Intersection(
            id=intersection.id,
            position=intersection.position,
            adjacent_tiles=intersection.adjacent_tiles,
            adjacent_intersections=intersection.adjacent_intersections,
            owner=current_player.id,
            building_type="city",
            port_type=intersection.port_type  # Preserve port type
        )
        current_player.settlements_built -= 1
        current_player.cities_built += 1
        current_player.victory_points += 1  # City gives +1 VP over settlement
        
        return new_state
    
    def _handle_buy_dev_card(self, new_state: 'GameState') -> 'GameState':
        """Handle buying a development card."""
        current_player = new_state.players[new_state.current_player_index]
        
        # Check resources: 1 wheat, 1 sheep, 1 ore
        if (current_player.resources[ResourceType.WHEAT] < 1 or 
            current_player.resources[ResourceType.SHEEP] < 1 or
            current_player.resources[ResourceType.ORE] < 1):
            raise ValueError("Insufficient resources to buy development card")
        
        # Check if any dev cards are available
        total_available = sum(new_state.dev_card_counts.values())
        if total_available == 0:
            raise ValueError("No development cards available")
        
        # Build weighted list of available cards
        available_cards = []
        for card_type, count in new_state.dev_card_counts.items():
            available_cards.extend([card_type] * count)
        
        if not available_cards:
            raise ValueError("No development cards available")
        
        # Randomly select from available cards
        card_type = random.choice(available_cards)
        
        # Decrement the card count
        new_state.dev_card_counts[card_type] -= 1
        
        # Deduct resources
        current_player.resources[ResourceType.WHEAT] -= 1
        current_player.resources[ResourceType.SHEEP] -= 1
        current_player.resources[ResourceType.ORE] -= 1
        # Return resources to bank (resources are reusable)
        new_state.resource_card_counts[ResourceType.WHEAT] += 1
        new_state.resource_card_counts[ResourceType.SHEEP] += 1
        new_state.resource_card_counts[ResourceType.ORE] += 1
        
        # Add the card to player's hand
        current_player.dev_cards.append(card_type)
        
        # Track that this player bought a dev card this turn
        new_state.dev_cards_bought_this_turn.add(current_player.id)
        
        return new_state
    
    def _handle_play_dev_card(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle playing a development card."""
        if not payload or not isinstance(payload, PlayDevCardPayload):
            raise ValueError("PLAY_DEV_CARD requires PlayDevCardPayload")
        
        current_player = new_state.players[new_state.current_player_index]
        
        # Check if player has the card
        if payload.card_type not in current_player.dev_cards:
            raise ValueError(f"Player does not have {payload.card_type} card")
        
        # Check if player bought a dev card this turn (cannot play same turn as buying, unless VP)
        if current_player.id in new_state.dev_cards_bought_this_turn and payload.card_type != "victory_point":
            raise ValueError("Cannot play a development card the same turn it was bought (except victory point cards)")
        
        # Check if player already played a dev card this turn (cannot play two, unless VPs)
        if current_player.id in new_state.dev_cards_played_this_turn and payload.card_type != "victory_point":
            raise ValueError("Cannot play two development cards in one turn (except victory point cards)")
        
        # Remove card from hand
        current_player.dev_cards.remove(payload.card_type)
        
        # Track that this player played a dev card this turn
        new_state.dev_cards_played_this_turn.add(current_player.id)
        
        # Handle card effects
        if payload.card_type == "victory_point":
            current_player.victory_points += 1
        elif payload.card_type == "knight":
            current_player.knights_played += 1
            # Check for largest army
            self._check_largest_army(new_state, current_player)
            # Knight card requires moving robber and stealing (handled by separate actions)
            new_state.waiting_for_robber_move = True
        elif payload.card_type == "road_building":
            # Road building card allows building 2 roads for free
            new_state.roads_from_road_building[current_player.id] = 2
        elif payload.card_type == "year_of_plenty":
            # Year of plenty gives player 2 resources of their choice
            if not payload.year_of_plenty_resources:
                raise ValueError("year_of_plenty requires year_of_plenty_resources payload")
            
            # Verify exactly 2 resources total
            total_resources = sum(payload.year_of_plenty_resources.values())
            if total_resources != 2:
                raise ValueError(f"year_of_plenty must give exactly 2 resources, got {total_resources}")
            
            # Check if all requested resources are available
            for resource, amount in payload.year_of_plenty_resources.items():
                if amount > 0:
                    available = new_state.resource_card_counts.get(resource, 0)
                    if available < amount:
                        raise ValueError(f"Insufficient {resource.value} cards available (need {amount}, have {available})")
            
            # Add the resources to player and decrement card counts
            for resource, amount in payload.year_of_plenty_resources.items():
                if amount > 0:
                    current_player.resources[resource] += amount
                    new_state.resource_card_counts[resource] -= amount
        elif payload.card_type == "monopoly":
            # Monopoly lets player steal all of a resource type from other players
            if not payload.monopoly_resource_type:
                raise ValueError("monopoly requires monopoly_resource_type payload")
            
            # Steal all of this resource type from all other players
            total_stolen = 0
            for other_player in new_state.players:
                if other_player.id != current_player.id:
                    stolen_amount = other_player.resources[payload.monopoly_resource_type]
                    if stolen_amount > 0:
                        other_player.resources[payload.monopoly_resource_type] = 0
                        current_player.resources[payload.monopoly_resource_type] += stolen_amount
                        total_stolen += stolen_amount
        
        return new_state
    
    def _handle_trade_bank(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle trading with the bank."""
        if not payload or not isinstance(payload, TradeBankPayload):
            raise ValueError("TRADE_BANK requires TradeBankPayload")
        
        # Trading is only allowed after dice is rolled
        if new_state.dice_roll is None:
            raise ValueError("Cannot trade before rolling dice")
        
        # If a 7 is rolled, trading is not allowed until all phases are complete
        if new_state.dice_roll == 7:
            # Check if we're still in discard phase
            robber_has_been_moved = (new_state.robber_initial_tile_id is not None and 
                                     new_state.robber_tile_id != new_state.robber_initial_tile_id)
            in_discard_phase = (not new_state.waiting_for_robber_move and 
                               not new_state.waiting_for_robber_steal and 
                               not robber_has_been_moved)
            
            if in_discard_phase:
                # Check if any player still needs to discard
                any_player_needs_discard = False
                for p in new_state.players:
                    if p.id not in new_state.players_discarded and sum(p.resources.values()) >= 8:
                        any_player_needs_discard = True
                        break
                if any_player_needs_discard:
                    raise ValueError("Cannot trade while players need to discard after rolling 7")
            
            # Check if we're in robber move/steal phase
            if new_state.waiting_for_robber_move or new_state.waiting_for_robber_steal:
                raise ValueError("Cannot trade while robber phase is active after rolling 7")
        
        current_player = new_state.players[new_state.current_player_index]
        
        # Check if player has enough of all resources they're giving
        for resource, amount in payload.give_resources.items():
            if current_player.resources[resource] < amount:
                raise ValueError(f"Insufficient {resource.value} to trade (have {current_player.resources[resource]}, need {amount})")
        
        # Validate trade ratio based on port or default 4:1
        total_give = sum(payload.give_resources.values())
        total_receive = sum(payload.receive_resources.values())
        
        if payload.port_intersection_id is not None:
            # Using a port - check ownership and validate ratio
            # Ports span 2 adjacent intersections - player must own at least one
            port_intersection = next((i for i in new_state.intersections if i.id == payload.port_intersection_id), None)
            if not port_intersection:
                raise ValueError(f"Port intersection {payload.port_intersection_id} not found")
            
            if port_intersection.port_type is None:
                raise ValueError(f"Intersection {payload.port_intersection_id} does not have a port")
            
            # Check if player owns this intersection or an adjacent intersection with the same port
            owns_port = False
            if port_intersection.owner == current_player.id:
                owns_port = True
            else:
                # Check adjacent intersections for the same port type
                for adj_id in port_intersection.adjacent_intersections:
                    adj_inter = next((i for i in new_state.intersections if i.id == adj_id), None)
                    if (adj_inter and 
                        adj_inter.port_type == port_intersection.port_type and
                        adj_inter.owner == current_player.id):
                        owns_port = True
                        break
            
            if not owns_port:
                raise ValueError(f"Player does not own port at intersection {payload.port_intersection_id} or adjacent intersection")
            
            if port_intersection.port_type == "3:1":
                # 3:1 generic port - can trade 3 of any resource for 1 of any resource
                if total_give != 3 or total_receive != 1:
                    raise ValueError("3:1 port trades must be exactly 3 resources for 1 resource")
            else:
                # 2:1 specific resource port
                port_resource = ResourceType(port_intersection.port_type)
                # Must give exactly 2 of the port's resource type
                if payload.give_resources.get(port_resource, 0) != 2 or total_give != 2:
                    raise ValueError(f"2:1 {port_resource.value} port requires exactly 2 {port_resource.value} for 1 of any resource")
                if total_receive != 1:
                    raise ValueError("2:1 port trades must receive exactly 1 resource")
        else:
            # Standard 4:1 trade (no port)
            if total_give != 4 or total_receive != 1:
                raise ValueError("Standard bank trades must be 4:1 ratio")
        
        # Check if all received resources are available
        for resource, amount in payload.receive_resources.items():
            if amount > 0:
                available = new_state.resource_card_counts.get(resource, 0)
                if available < amount:
                    raise ValueError(f"Insufficient {resource.value} cards available (need {amount}, have {available})")
        
        # Execute trade - remove given resources (these go back to the bank pool)
        for resource, amount in payload.give_resources.items():
            current_player.resources[resource] -= amount
            # Return resources to the bank pool
            new_state.resource_card_counts[resource] += amount
        
        # Add received resources (these come from the bank pool)
        for resource, amount in payload.receive_resources.items():
            if resource not in current_player.resources:
                current_player.resources[resource] = 0
            current_player.resources[resource] += amount
            # Remove resources from the bank pool
            new_state.resource_card_counts[resource] -= amount
        
        return new_state
    
    def _handle_trade_player(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle trading with another player."""
        if not payload or not isinstance(payload, TradePlayerPayload):
            raise ValueError("TRADE_PLAYER requires TradePlayerPayload")
        
        # Trading is only allowed after dice is rolled
        if new_state.dice_roll is None:
            raise ValueError("Cannot trade before rolling dice")
        
        # If a 7 is rolled, trading is not allowed until all phases are complete
        if new_state.dice_roll == 7:
            # Check if we're still in discard phase
            robber_has_been_moved = (new_state.robber_initial_tile_id is not None and 
                                     new_state.robber_tile_id != new_state.robber_initial_tile_id)
            in_discard_phase = (not new_state.waiting_for_robber_move and 
                               not new_state.waiting_for_robber_steal and 
                               not robber_has_been_moved)
            
            if in_discard_phase:
                # Check if any player still needs to discard
                any_player_needs_discard = False
                for p in new_state.players:
                    if p.id not in new_state.players_discarded and sum(p.resources.values()) >= 8:
                        any_player_needs_discard = True
                        break
                if any_player_needs_discard:
                    raise ValueError("Cannot trade while players need to discard after rolling 7")
            
            # Check if we're in robber move/steal phase
            if new_state.waiting_for_robber_move or new_state.waiting_for_robber_steal:
                raise ValueError("Cannot trade while robber phase is active after rolling 7")
        
        current_player = new_state.players[new_state.current_player_index]
        other_player = next((p for p in new_state.players if p.id == payload.other_player_id), None)
        
        if not other_player:
            raise ValueError(f"Player {payload.other_player_id} not found")
        
        # Check current player has enough of all resources they're giving
        for resource, amount in payload.give_resources.items():
            if current_player.resources[resource] < amount:
                raise ValueError(f"Insufficient {resource.value} to trade (have {current_player.resources[resource]}, need {amount})")
        
        # Check other player has enough of all resources they're giving
        for resource, amount in payload.receive_resources.items():
            if other_player.resources[resource] < amount:
                raise ValueError(f"Other player has insufficient {resource.value} (they have {other_player.resources[resource]}, need {amount})")
        
        # Execute trade - remove given resources from current player
        for resource, amount in payload.give_resources.items():
            current_player.resources[resource] -= amount
        
        # Add received resources to current player
        for resource, amount in payload.receive_resources.items():
            current_player.resources[resource] += amount
        
        # Remove given resources from other player (what current player receives)
        for resource, amount in payload.receive_resources.items():
            other_player.resources[resource] -= amount
        
        # Add received resources to other player (what current player gives)
        for resource, amount in payload.give_resources.items():
            other_player.resources[resource] += amount
        
        return new_state
    
    def _handle_discard_resources(self, new_state: 'GameState', payload: Optional['ActionPayload'], player_id: Optional[str] = None) -> 'GameState':
        """Handle discarding resources when rolling 7 with 8+ resources.
        
        Note: When a 7 is rolled, ANY player with 8+ resources can discard,
        not just the current player. This allows all players to discard before
        the robber is moved.
        
        Args:
            player_id: The ID of the player discarding. If None, uses current player.
        """
        if not payload or not isinstance(payload, DiscardResourcesPayload):
            raise ValueError("DISCARD_RESOURCES requires DiscardResourcesPayload")
        
        # Find the player discarding (could be current player or another player)
        if player_id:
            current_player = next((p for p in new_state.players if p.id == player_id), None)
            if not current_player:
                raise ValueError(f"Player {player_id} not found")
        else:
            current_player = new_state.players[new_state.current_player_index]
        
        # Calculate total resources
        total_resources = sum(current_player.resources.values())
        
        # Must have 8+ resources to discard
        if total_resources < 8:
            raise ValueError("Must have 8+ resources to discard")
        
        # Calculate how many to discard (half, rounded down)
        discard_count = total_resources // 2
        
        # Verify the discard matches the required amount
        total_discarded = sum(payload.resources.values())
        if total_discarded != discard_count:
            raise ValueError(f"Must discard exactly {discard_count} resources (half of {total_resources})")
        
        # Verify player has enough of each resource being discarded
        for resource, amount in payload.resources.items():
            if current_player.resources[resource] < amount:
                raise ValueError(f"Insufficient {resource.value} to discard")
        
        # Discard the resources
        for resource, amount in payload.resources.items():
            current_player.resources[resource] -= amount
            # Return discarded resources to bank
            new_state.resource_card_counts[resource] += amount
        
        # Mark this player as having discarded
        new_state.players_discarded.add(current_player.id)
        
        # After discarding, check if we can proceed to robber move
        # Only the player who rolled the 7 can move the robber
        if new_state.dice_roll == 7:
            # Check if anyone still needs to discard (and hasn't discarded yet)
            any_player_needs_discard = False
            for p in new_state.players:
                if p.id not in new_state.players_discarded and sum(p.resources.values()) >= 8:
                    any_player_needs_discard = True
                    break
            
            # If no one needs to discard, allow robber move (only for the player who rolled)
            if not any_player_needs_discard:
                current_turn_player = new_state.players[new_state.current_player_index]
                new_state.waiting_for_robber_move = True
        
        return new_state
    
    def _handle_move_robber(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle moving the robber to a new tile."""
        if not payload or not isinstance(payload, MoveRobberPayload):
            raise ValueError("MOVE_ROBBER requires MoveRobberPayload")
        
        # If a 7 was rolled, ensure ALL players with 8+ resources have discarded before moving robber
        if new_state.dice_roll == 7:
            for p in new_state.players:
                if p.id not in new_state.players_discarded and sum(p.resources.values()) >= 8:
                    raise ValueError(f"Cannot move robber until all players have discarded. Player {p.name} ({p.id}) still needs to discard.")
        
        # Verify tile exists
        tile = next((t for t in new_state.tiles if t.id == payload.tile_id), None)
        if not tile:
            raise ValueError(f"Tile {payload.tile_id} not found")
        
        # Can't move robber to the same tile
        if new_state.robber_tile_id == payload.tile_id:
            raise ValueError("Robber is already on this tile")
        
        # Move the robber (allowing strategic moves to own tiles if it also blocks opponents)
        new_state.robber_tile_id = payload.tile_id
        new_state.waiting_for_robber_move = False
        
        # Check if there are players on this tile to steal from
        robber_tile = next((t for t in new_state.tiles if t.id == payload.tile_id), None)
        if robber_tile:
            # Find players with buildings on this tile
            players_on_tile = set()
            for intersection in new_state.intersections:
                if (robber_tile.id in intersection.adjacent_tiles and 
                    intersection.owner and 
                    intersection.building_type):
                    if intersection.owner != new_state.players[new_state.current_player_index].id:
                        players_on_tile.add(intersection.owner)
            
            # Check if any players on the tile have resources to steal
            has_valid_steal_targets = False
            for player_id in players_on_tile:
                player = next((p for p in new_state.players if p.id == player_id), None)
                if player and sum(player.resources.values()) > 0:
                    has_valid_steal_targets = True
                    break
            
            if has_valid_steal_targets:
                # Set flag that we're waiting for steal
                new_state.waiting_for_robber_steal = True
            else:
                # No one to steal from (either no players or no resources), clear the flags
                new_state.waiting_for_robber_steal = False
        
        return new_state
    
    def _handle_steal_resource(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle stealing a resource from another player."""
        if not payload or not isinstance(payload, StealResourcePayload):
            raise ValueError("STEAL_RESOURCE requires StealResourcePayload")
        
        current_player = new_state.players[new_state.current_player_index]
        other_player = next((p for p in new_state.players if p.id == payload.other_player_id), None)
        
        if not other_player:
            raise ValueError(f"Player {payload.other_player_id} not found")
        
        if other_player.id == current_player.id:
            raise ValueError("Cannot steal from yourself")
        
        # Check if other player has any resources
        total_resources = sum(other_player.resources.values())
        if total_resources == 0:
            raise ValueError(f"{other_player.name} has no resources to steal")
        
        # Check if other player has a settlement/city on the robber's tile
        robber_tile = next((t for t in new_state.tiles if t.id == new_state.robber_tile_id), None)
        if not robber_tile:
            raise ValueError("Robber not placed on any tile")
        
        has_building_on_tile = False
        for intersection in new_state.intersections:
            if (robber_tile.id in intersection.adjacent_tiles and 
                intersection.owner == other_player.id and 
                intersection.building_type):
                has_building_on_tile = True
                break
        
        if not has_building_on_tile:
            raise ValueError(f"{other_player.name} has no building on the robber's tile")
        
        # Randomly select a resource to steal
        available_resources = []
        for resource_type, amount in other_player.resources.items():
            if amount > 0:
                available_resources.extend([resource_type] * amount)
        
        if not available_resources:
            raise ValueError(f"{other_player.name} has no resources to steal")
        
        stolen_resource = random.choice(available_resources)
        
        # Steal the resource
        other_player.resources[stolen_resource] -= 1
        current_player.resources[stolen_resource] += 1
        
        # Clear the waiting flag
        new_state.waiting_for_robber_steal = False
        
        return new_state
    
    def _check_largest_army(self, state: 'GameState', player: 'Player'):
        """Check and award largest army if applicable."""
        if player.knights_played < 3:
            return
        
        # Check if player has more knights than others
        has_largest = True
        for other_player in state.players:
            if other_player.id != player.id and other_player.knights_played >= player.knights_played:
                has_largest = False
                break
        
        if has_largest and not player.largest_army:
            # Remove from previous holder
            for other_player in state.players:
                if other_player.largest_army:
                    other_player.largest_army = False
                    other_player.victory_points -= 2
            
            # Award to current player
            player.largest_army = True
            player.victory_points += 2
    
    def _calculate_longest_road(self, state: 'GameState', player_id: str) -> int:
        """Calculate the longest continuous road for a player."""
        player_roads = [r for r in state.road_edges if r.owner == player_id]
        if not player_roads:
            return 0
        
        # Build lookup map for intersections to check ownership
        intersection_map = {inter.id: inter for inter in state.intersections}
        
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
        # The longest path in an undirected graph (without cycles) is found by
        # doing DFS from each node and tracking the longest path found
        # Road paths cannot go through intersections owned by other players
        max_length = 0
        
        def dfs_path_length(node: int, visited_edges: Set[Tuple[int, int]]) -> int:
            """DFS to find longest path from current node, returning number of edges."""
            max_path = 0
            for neighbor in road_graph.get(node, []):
                edge_key = (min(node, neighbor), max(node, neighbor))
                if edge_key not in visited_edges:
                    # Check if neighbor intersection is owned by a different player
                    # If so, the road path cannot continue through it
                    neighbor_inter = intersection_map.get(neighbor)
                    if neighbor_inter and neighbor_inter.owner is not None and neighbor_inter.owner != player_id:
                        # Road path is broken at this intersection, don't continue
                        continue
                    
                    visited_edges.add(edge_key)
                    # Continue path from neighbor
                    path_len = 1 + dfs_path_length(neighbor, visited_edges)
                    max_path = max(max_path, path_len)
                    visited_edges.remove(edge_key)
            return max_path
        
        # Try starting from each node to find the longest path
        # You can start from any intersection, but paths cannot continue through opponent-owned intersections
        for start_node in road_graph.keys():
            path_len = dfs_path_length(start_node, set())
            max_length = max(max_length, path_len)
        
        return max_length
    
    def _check_longest_road(self, state: 'GameState', player: 'Player'):
        """Check and award longest road if applicable."""
        player_road_length = self._calculate_longest_road(state, player.id)
        
        if player_road_length < 5:
            return
        
        # Check if player has longer road than others
        has_longest = True
        for other_player in state.players:
            if other_player.id != player.id:
                other_length = self._calculate_longest_road(state, other_player.id)
                if other_length >= player_road_length:
                    has_longest = False
                    break
        
        if has_longest and not player.longest_road:
            # Remove from previous holder
            for other_player in state.players:
                if other_player.longest_road:
                    other_player.longest_road = False
                    other_player.victory_points -= 2
            
            # Award to current player
            player.longest_road = True
            player.victory_points += 2
    
    def _handle_propose_trade(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle proposing a trade to one or more players."""
        if not payload or not isinstance(payload, ProposeTradePayload):
            raise ValueError("PROPOSE_TRADE requires ProposeTradePayload")
        
        # Trading is only allowed after dice is rolled
        if new_state.dice_roll is None:
            raise ValueError("Cannot trade before rolling dice")
        
        # If a 7 is rolled, trading is not allowed until all phases are complete
        if new_state.dice_roll == 7:
            # Check if we're still in discard phase
            robber_has_been_moved = (new_state.robber_initial_tile_id is not None and 
                                     new_state.robber_tile_id != new_state.robber_initial_tile_id)
            in_discard_phase = (not new_state.waiting_for_robber_move and 
                               not new_state.waiting_for_robber_steal and 
                               not robber_has_been_moved)
            
            if in_discard_phase:
                # Check if any player still needs to discard
                any_player_needs_discard = False
                for p in new_state.players:
                    if p.id not in new_state.players_discarded and sum(p.resources.values()) >= 8:
                        any_player_needs_discard = True
                        break
                if any_player_needs_discard:
                    raise ValueError("Cannot trade while players need to discard after rolling 7")
            
            # Check if we're in robber move/steal phase
            if new_state.waiting_for_robber_move or new_state.waiting_for_robber_steal:
                raise ValueError("Cannot trade while robber phase is active after rolling 7")
        
        # Can't propose trade if there's already a pending trade
        if new_state.pending_trade_offer is not None:
            raise ValueError("There is already a pending trade offer")
        
        current_player = new_state.players[new_state.current_player_index]
        
        # Check if this player has had 3 consecutive rejected trades this turn
        # If so, automatically end their turn instead of allowing another trade proposal
        consecutive_rejections = new_state.consecutive_rejected_trades.get(current_player.id, 0)
        if consecutive_rejections >= 3:
            # Auto-end turn instead of proposing trade
            return self._handle_end_turn(new_state)
        
        # Validate proposer has enough resources
        for resource, amount in payload.give_resources.items():
            if current_player.resources[resource] < amount:
                raise ValueError(f"Insufficient {resource.value} to trade (have {current_player.resources[resource]}, need {amount})")
        
        # Validate target players exist
        for target_id in payload.target_player_ids:
            if target_id == current_player.id:
                raise ValueError("Cannot propose trade to yourself")
            target_player = next((p for p in new_state.players if p.id == target_id), None)
            if not target_player:
                raise ValueError(f"Target player {target_id} not found")
        
        # Create pending trade offer
        new_state.pending_trade_offer = {
            'proposer_id': current_player.id,
            'target_player_ids': payload.target_player_ids,
            'give_resources': payload.give_resources,
            'receive_resources': payload.receive_resources,
        }
        new_state.pending_trade_responses = {}
        new_state.pending_trade_current_responder_index = 0
        
        # Switch to first target player to respond
        if payload.target_player_ids:
            first_target_id = payload.target_player_ids[0]
            target_index = next((i for i, p in enumerate(new_state.players) if p.id == first_target_id), None)
            if target_index is not None:
                new_state.current_player_index = target_index
        
        return new_state
    
    def _handle_accept_trade(self, new_state: 'GameState') -> 'GameState':
        """Handle accepting a pending trade offer."""
        if new_state.pending_trade_offer is None:
            raise ValueError("No pending trade offer to accept")
        
        current_player = new_state.players[new_state.current_player_index]
        offer = new_state.pending_trade_offer
        
        # Verify this player is a target of the trade
        if current_player.id not in offer['target_player_ids']:
            raise ValueError(f"Player {current_player.id} is not a target of this trade offer")
        
        # Verify this player hasn't already responded
        if current_player.id in new_state.pending_trade_responses:
            raise ValueError(f"Player {current_player.id} has already responded to this trade")
        
        # Verify player can afford the trade (they give receive_resources, get give_resources)
        for resource, amount in offer['receive_resources'].items():
            if current_player.resources[resource] < amount:
                raise ValueError(f"Insufficient {resource.value} to accept trade (have {current_player.resources[resource]}, need {amount})")
        
        # Mark as accepted
        new_state.pending_trade_responses[current_player.id] = True
        
        # Check if there are more players to respond
        responded_count = len(new_state.pending_trade_responses)
        total_targets = len(offer['target_player_ids'])
        
        if responded_count < total_targets:
            # Move to next target player
            next_index = new_state.pending_trade_current_responder_index + 1
            if next_index < total_targets:
                next_target_id = offer['target_player_ids'][next_index]
                target_index = next((i for i, p in enumerate(new_state.players) if p.id == next_target_id), None)
                if target_index is not None:
                    new_state.current_player_index = target_index
                    new_state.pending_trade_current_responder_index = next_index
        else:
            # All players have responded - check if any accepted
            accepting_players = [pid for pid, accepted in new_state.pending_trade_responses.items() if accepted]
            
            if len(accepting_players) == 0:
                # No one accepted - clear trade and return to proposer
                proposer_index = next((i for i, p in enumerate(new_state.players) if p.id == offer['proposer_id']), None)
                if proposer_index is not None:
                    new_state.current_player_index = proposer_index
                new_state.pending_trade_offer = None
                new_state.pending_trade_responses = {}
                new_state.pending_trade_current_responder_index = 0
            elif len(accepting_players) == 1:
                # Exactly one accepted - execute trade immediately
                # Reset consecutive rejected trades counter for proposer (trade succeeded)
                proposer_id = offer['proposer_id']
                new_state.consecutive_rejected_trades[proposer_id] = 0
                
                new_state = self._execute_trade(new_state, proposer_id, accepting_players[0], 
                                                offer['give_resources'], offer['receive_resources'])
                # Clear trade state and return to proposer
                proposer_index = next((i for i, p in enumerate(new_state.players) if p.id == proposer_id), None)
                if proposer_index is not None:
                    new_state.current_player_index = proposer_index
                new_state.pending_trade_offer = None
                new_state.pending_trade_responses = {}
                new_state.pending_trade_current_responder_index = 0
            else:
                # Multiple accepted - proposer chooses
                proposer_index = next((i for i, p in enumerate(new_state.players) if p.id == offer['proposer_id']), None)
                if proposer_index is not None:
                    new_state.current_player_index = proposer_index
        
        return new_state
    
    def _handle_reject_trade(self, new_state: 'GameState') -> 'GameState':
        """Handle rejecting a pending trade offer."""
        if new_state.pending_trade_offer is None:
            raise ValueError("No pending trade offer to reject")
        
        current_player = new_state.players[new_state.current_player_index]
        offer = new_state.pending_trade_offer
        
        # Verify this player is a target of the trade
        if current_player.id not in offer['target_player_ids']:
            raise ValueError(f"Player {current_player.id} is not a target of this trade offer")
        
        # Verify this player hasn't already responded
        if current_player.id in new_state.pending_trade_responses:
            raise ValueError(f"Player {current_player.id} has already responded to this trade")
        
        # Mark as rejected
        new_state.pending_trade_responses[current_player.id] = False
        
        # Check if there are more players to respond
        responded_count = len(new_state.pending_trade_responses)
        total_targets = len(offer['target_player_ids'])
        
        if responded_count < total_targets:
            # Move to next target player
            next_index = new_state.pending_trade_current_responder_index + 1
            if next_index < total_targets:
                next_target_id = offer['target_player_ids'][next_index]
                target_index = next((i for i, p in enumerate(new_state.players) if p.id == next_target_id), None)
                if target_index is not None:
                    new_state.current_player_index = target_index
                    new_state.pending_trade_current_responder_index = next_index
        else:
            # All players have responded - check if any accepted
            accepting_players = [pid for pid, accepted in new_state.pending_trade_responses.items() if accepted]
            proposer_id = offer['proposer_id']
            
            if len(accepting_players) == 0:
                # No one accepted - increment consecutive rejected trades counter for proposer
                new_state.consecutive_rejected_trades[proposer_id] = new_state.consecutive_rejected_trades.get(proposer_id, 0) + 1
                
                # Clear trade and return to proposer
                proposer_index = next((i for i, p in enumerate(new_state.players) if p.id == proposer_id), None)
                if proposer_index is not None:
                    new_state.current_player_index = proposer_index
                new_state.pending_trade_offer = None
                new_state.pending_trade_responses = {}
                new_state.pending_trade_current_responder_index = 0
            elif len(accepting_players) == 1:
                # Exactly one accepted - execute trade immediately
                # Reset consecutive rejected trades counter for proposer (trade succeeded)
                new_state.consecutive_rejected_trades[proposer_id] = 0
                
                new_state = self._execute_trade(new_state, proposer_id, accepting_players[0], 
                                                offer['give_resources'], offer['receive_resources'])
                # Clear trade state and return to proposer
                proposer_index = next((i for i, p in enumerate(new_state.players) if p.id == proposer_id), None)
                if proposer_index is not None:
                    new_state.current_player_index = proposer_index
                new_state.pending_trade_offer = None
                new_state.pending_trade_responses = {}
                new_state.pending_trade_current_responder_index = 0
            else:
                # Multiple accepted - proposer chooses
                proposer_index = next((i for i, p in enumerate(new_state.players) if p.id == proposer_id), None)
                if proposer_index is not None:
                    new_state.current_player_index = proposer_index
        
        return new_state
    
    def _handle_select_trade_partner(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle selecting which accepting player to trade with."""
        if not payload or not isinstance(payload, SelectTradePartnerPayload):
            raise ValueError("SELECT_TRADE_PARTNER requires SelectTradePartnerPayload")
        
        if new_state.pending_trade_offer is None:
            raise ValueError("No pending trade offer")
        
        current_player = new_state.players[new_state.current_player_index]
        offer = new_state.pending_trade_offer
        
        # Verify current player is the proposer
        if current_player.id != offer['proposer_id']:
            raise ValueError("Only the proposer can select a trade partner")
        
        # Verify selected player accepted
        if payload.selected_player_id not in new_state.pending_trade_responses:
            raise ValueError(f"Player {payload.selected_player_id} has not responded to the trade")
        
        if not new_state.pending_trade_responses[payload.selected_player_id]:
            raise ValueError(f"Player {payload.selected_player_id} rejected the trade")
        
        # Reset consecutive rejected trades counter for proposer (trade succeeded)
        proposer_id = offer['proposer_id']
        new_state.consecutive_rejected_trades[proposer_id] = 0
        
        # Execute trade with selected player
        new_state = self._execute_trade(new_state, proposer_id, payload.selected_player_id,
                                        offer['give_resources'], offer['receive_resources'])
        
        # Clear trade state
        new_state.pending_trade_offer = None
        new_state.pending_trade_responses = {}
        new_state.pending_trade_current_responder_index = 0
        
        return new_state
    
    def _execute_trade(self, new_state: 'GameState', proposer_id: str, partner_id: str,
                      give_resources: Dict[ResourceType, int], receive_resources: Dict[ResourceType, int]) -> 'GameState':
        """Execute a trade between two players."""
        proposer = next((p for p in new_state.players if p.id == proposer_id), None)
        partner = next((p for p in new_state.players if p.id == partner_id), None)
        
        if not proposer or not partner:
            raise ValueError("Proposer or partner not found")
        
        # Verify proposer has enough resources
        for resource, amount in give_resources.items():
            if proposer.resources[resource] < amount:
                raise ValueError(f"Proposer has insufficient {resource.value}")
        
        # Verify partner has enough resources
        for resource, amount in receive_resources.items():
            if partner.resources[resource] < amount:
                raise ValueError(f"Partner has insufficient {resource.value}")
        
        # Execute trade - proposer gives give_resources, receives receive_resources
        # Partner gives receive_resources, receives give_resources
        for resource, amount in give_resources.items():
            proposer.resources[resource] -= amount
            if resource not in partner.resources:
                partner.resources[resource] = 0
            partner.resources[resource] += amount
        
        for resource, amount in receive_resources.items():
            partner.resources[resource] -= amount
            if resource not in proposer.resources:
                proposer.resources[resource] = 0
            proposer.resources[resource] += amount
        
        return new_state
    
    def _handle_end_turn(self, new_state: 'GameState') -> 'GameState':
        """Handle ending the current turn."""
        if new_state.phase != "playing":
            raise ValueError("Can only end turn during playing phase")
        
        # Can't end turn if waiting for robber actions
        if new_state.waiting_for_robber_move or new_state.waiting_for_robber_steal:
            raise ValueError("Must complete robber move and steal before ending turn")
        
        # Can't end turn if there's a pending trade
        if new_state.pending_trade_offer is not None:
            raise ValueError("Cannot end turn while trade is pending")
        
        # Get the current player (before advancing) to clear their free roads
        current_player = new_state.players[new_state.current_player_index]
        
        # Clear free roads from Road Building card for the current player
        # Free roads must be used before the end of the turn
        if current_player.id in new_state.roads_from_road_building:
            del new_state.roads_from_road_building[current_player.id]
        
        # Advance to next player
        new_state.current_player_index = (new_state.current_player_index + 1) % len(new_state.players)
        new_state.turn_number += 1
        new_state.dice_roll = None  # Reset dice roll
        new_state.waiting_for_robber_move = False  # Clear flags
        new_state.waiting_for_robber_steal = False
        new_state.players_discarded = set()  # Clear discarded players set for next turn
        new_state.robber_initial_tile_id = None  # Clear initial robber position
        # Clear dev card tracking for next turn
        new_state.dev_cards_bought_this_turn = set()
        new_state.dev_cards_played_this_turn = set()
        # Clear turn tracking for next player's turn
        new_state.actions_taken_this_turn = []
        new_state.consecutive_rejected_trades = {}
        
        return new_state
    
    def _handle_start_game(self, new_state: 'GameState') -> 'GameState':
        """Initialize the game board and start playing."""
        if new_state.phase != "setup":
            raise ValueError("Game can only be started from setup phase")
        
        # Board should already be created, just initialize robber on desert
        desert_tile = next((t for t in new_state.tiles if t.resource_type is None), None)
        if desert_tile:
            new_state.robber_tile_id = desert_tile.id
        
        new_state.phase = "playing"
        
        return new_state
    
    def _create_initial_board(self, state: 'GameState') -> 'GameState':
        """Create initial board with tiles, intersections, and road edges.
        
        Uses standard Catan board layout: 19 tiles in 3-4-5-4-3 pattern.
        Based on standard Catan hex coordinate system.
        """
        # Standard Catan hex coordinates (q, r) for 19 tiles
        # Layout: 3-4-5-4-3 rows
        hex_coords = [
            (0, 0), (0, -1), (1, -1),  # Row 0: 3 tiles
            (1, 0), (0, 1), (-1, 1), (-1, 0),  # Row 1: 4 tiles
            (0, -2), (1, -2), (2, -2), (2, -1), (2, 0),  # Row 2: 5 tiles
            (1, 1), (0, 2), (-1, 2), (-2, 2),  # Row 3: 4 tiles
            (-2, 1), (-2, 0), (-1, -1)  # Row 4: 3 tiles
        ]
        
        # Resource distribution: 3 ore, 3 brick, 4 wheat, 4 wood, 4 sheep, 1 desert
        resource_counts = {
            ResourceType.ORE: 3,
            ResourceType.BRICK: 3,
            ResourceType.WHEAT: 4,
            ResourceType.WOOD: 4,
            ResourceType.SHEEP: 4,
            None: 1  # Desert
        }
        
        # Build resource list
        resource_list = []
        for resource_type, count in resource_counts.items():
            for _ in range(count):
                resource_list.append(resource_type)
        
        # Number tokens: 2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12 (18 total, desert has none)
        number_list = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        
        # Shuffle resources and numbers
        random.shuffle(resource_list)
        random.shuffle(number_list)
        
        # Assign number tokens to non-desert resources
        num_idx = 0
        tiles = []
        for tile_id, (q, r) in enumerate(hex_coords):
            resource = resource_list[tile_id]
            
            if resource is None:  # Desert
                tile = Tile(
                    id=tile_id,
                    resource_type=None,
                    number_token=None,
                    position=(q, r)
                )
            else:
                tile = Tile(
                    id=tile_id,
                    resource_type=resource,
                    number_token=NumberToken(number_list[num_idx]) if num_idx < len(number_list) else None,
                    position=(q, r)
                )
                num_idx += 1
            
            tiles.append(tile)
        
        # Check and fix 6/8 adjacency rule
        tiles = self._fix_adjacent_6_8(tiles, hex_coords)
        
        state.tiles = tiles
        
        # Create intersections at hex corners using proper hex geometry
        # For pointy-top hexagons, corners are at specific offsets from center
        intersections = []
        intersection_id = 0
        intersection_map = {}  # (q, r) rounded -> intersection_id
        
        # Hex corner directions (pointy-top hex, 6 corners)
        # Using cube coordinates: each corner is between two directions
        corner_directions = [
            (1, 0, -1), (1, -1, 0), (0, -1, 1),
            (-1, 0, 1), (-1, 1, 0), (0, 1, -1)
        ]
        
        def axial_to_cube(q, r):
            """Convert axial (q, r) to cube coordinates (x, y, z)."""
            x = q
            z = r
            y = -x - z
            return (x, y, z)
        
        def cube_to_axial(x, y, z):
            """Convert cube to axial coordinates."""
            return (x, z)
        
        def hex_corner_position(q, r, corner):
            """Get corner position in hex coordinates."""
            # Convert to cube
            x, y, z = axial_to_cube(q, r)
            # Get corner direction
            dir1 = corner_directions[corner]
            dir2 = corner_directions[(corner + 1) % 6]
            # Corner is average of two directions
            corner_x = x + (dir1[0] + dir2[0]) / 3.0
            corner_y = y + (dir1[1] + dir2[1]) / 3.0
            corner_z = z + (dir1[2] + dir2[2]) / 3.0
            # Convert back to axial
            return cube_to_axial(corner_x, corner_y, corner_z)
        
        for tile in tiles:
            q, r = tile.position
            # Create 6 corners for this hex
            for corner_idx in range(6):
                corner_q, corner_r = hex_corner_position(q, r, corner_idx)
                # Round to avoid floating point issues
                corner_key = (round(corner_q, 4), round(corner_r, 4))
                
                if corner_key not in intersection_map:
                    intersection = Intersection(
                        id=intersection_id,
                        position=(corner_q, corner_r),
                        adjacent_tiles={tile.id},
                        adjacent_intersections=set(),
                        port_type=None  # Will assign ports later
                    )
                    intersections.append(intersection)
                    intersection_map[corner_key] = intersection_id
                    intersection_id += 1
                else:
                    # Add this tile to existing intersection
                    existing_id = intersection_map[corner_key]
                    existing_inter = intersections[existing_id]
                    intersections[existing_id] = Intersection(
                        id=existing_inter.id,
                        position=existing_inter.position,
                        adjacent_tiles=existing_inter.adjacent_tiles | {tile.id},
                        adjacent_intersections=existing_inter.adjacent_intersections,
                        owner=existing_inter.owner,
                        building_type=existing_inter.building_type,
                        port_type=existing_inter.port_type
                    )
        
        # Link adjacent intersections (those on the same hex edge)
        for tile in tiles:
            q, r = tile.position
            # Get all corners of this hex
            hex_corner_ids = []
            for corner_idx in range(6):
                corner_q, corner_r = hex_corner_position(q, r, corner_idx)
                corner_key = (round(corner_q, 4), round(corner_r, 4))
                if corner_key in intersection_map:
                    hex_corner_ids.append(intersection_map[corner_key])
            
            # Link consecutive corners (they share an edge of this hex)
            for i in range(len(hex_corner_ids)):
                corner1_id = hex_corner_ids[i]
                corner2_id = hex_corner_ids[(i + 1) % len(hex_corner_ids)]
                
                inter1 = intersections[corner1_id]
                inter2 = intersections[corner2_id]
                
                # Add to each other's adjacent list (bidirectional)
                intersections[corner1_id] = Intersection(
                    id=inter1.id,
                    position=inter1.position,
                    adjacent_tiles=inter1.adjacent_tiles,
                    adjacent_intersections=inter1.adjacent_intersections | {corner2_id},
                    owner=inter1.owner,
                    building_type=inter1.building_type,
                    port_type=inter1.port_type
                )
                
                intersections[corner2_id] = Intersection(
                    id=inter2.id,
                    position=inter2.position,
                    adjacent_tiles=inter2.adjacent_tiles,
                    adjacent_intersections=inter2.adjacent_intersections | {corner1_id},
                    owner=inter2.owner,
                    building_type=inter2.building_type,
                    port_type=inter2.port_type
                )
        
        # Assign ports to coastal edges using the user's algorithm:
        # - 30 coastal edges total
        # - Traverse coastline in order
        # - Place ports at intervals: 3, 3, 4, 3, 3, 4, 3, 3, 4 (repeating pattern, sums to 30)
        # - This gives exactly 9 ports
        
        # Find all edges along the coastline (perimeter)
        # The coastline is the perimeter of the board - all intersections with < 3 adjacent tiles
        # These form a loop of 30 edges
        
        # Identify perimeter intersections (those with < 3 adjacent tiles)
        perimeter_intersections = [i for i in intersections if len(i.adjacent_tiles) < 3]
        perimeter_ids = {i.id for i in perimeter_intersections}
        
        # Build the full coastline graph (all edges between perimeter intersections)
        coastline_edges = []
        coastline_graph = {}
        
        for inter in perimeter_intersections:
            inter_id = inter.id
            coastline_graph[inter_id] = []
            
            for adj_id in inter.adjacent_intersections:
                # Only include edges to other perimeter intersections
                if adj_id in perimeter_ids:
                    edge_key = tuple(sorted([inter_id, adj_id]))
                    if edge_key not in coastline_edges:
                        coastline_edges.append(edge_key)
                    coastline_graph[inter_id].append(adj_id)
        
        # Verify coastline forms a loop (single connected component, all nodes degree 2)
        if coastline_graph:
            visited = set()
            start_node = next(iter(coastline_graph.keys()))
            stack = [start_node]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                if node in coastline_graph:
                    for neighbor in coastline_graph[node]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            is_loop = len(visited) == len(coastline_graph)
            degrees = {node: len(neighbors) for node, neighbors in coastline_graph.items()}
            all_degree_2 = all(d == 2 for d in degrees.values())
            
            if not is_loop or not all_degree_2:
                # Coastline doesn't form a proper loop - this shouldn't happen
                pass
        
        # For port placement, we need edges that can have ports
        # Ports are on edges between edge_inter (1 tile) and coastal_inter (2 tiles)
        edge_intersections = [i for i in intersections if len(i.adjacent_tiles) == 1]
        valid_port_edges = []
        for edge_inter in edge_intersections:
            # Find adjacent coastal intersection (2 tiles)
            for adj_id in edge_inter.adjacent_intersections:
                adj_inter = intersections[adj_id]
                if len(adj_inter.adjacent_tiles) == 2:
                    valid_port_edges.append((edge_inter.id, adj_id))
                    break  # Each edge intersection connects to at most one coastal intersection
        
        # Order edges around the board by traversing the coastline loop
        # We need to traverse the perimeter in order to place ports correctly
        # Build an ordered list of perimeter intersections by following the loop
        perimeter_ordered = []
        if coastline_graph:
            # Start from any perimeter intersection
            start_node = next(iter(coastline_graph.keys()))
            visited = set()
            current = start_node
            prev = None
            
            # Traverse the perimeter loop
            while len(visited) < len(coastline_graph):
                visited.add(current)
                perimeter_ordered.append(current)
                
                # Find next unvisited neighbor (or start if we've visited all)
                next_node = None
                for neighbor in coastline_graph[current]:
                    if neighbor != prev and neighbor not in visited:
                        next_node = neighbor
                        break
                
                if next_node is None:
                    # Check if we can loop back to start
                    if start_node in coastline_graph[current] and len(visited) == len(coastline_graph):
                        break
                    else:
                        # Shouldn't happen if it's a proper loop
                        break
                
                prev = current
                current = next_node
        
        # Build ordered list of 30 edges along the perimeter loop
        # Each edge connects two consecutive nodes in the perimeter_ordered list
        perimeter_edges = []
        for i in range(len(perimeter_ordered)):
            node1 = perimeter_ordered[i]
            node2 = perimeter_ordered[(i + 1) % len(perimeter_ordered)]
            perimeter_edges.append((node1, node2))
        
        # All 30 perimeter edges are valid port edges
        valid_port_edge_indices = list(range(len(perimeter_edges)))  # All 30 edges
        
        # We still need to identify edge_inter and coastal_inter for port assignment
        edge_intersection_ids = {i.id for i in intersections if len(i.adjacent_tiles) == 1}
        coastal_intersection_ids = {i.id for i in intersections if len(i.adjacent_tiles) == 2}
        
        # Place ports at intervals: 3, 3, 4, 3, 3, 4, 3, 3, 4 along the 30-edge perimeter
        # This pattern sums to 30, giving exactly 9 ports
        # Not all 30 edges are valid port edges - only edges between edge_inter and coastal_inter
        # So we place at the target position if valid, otherwise find nearest valid edge
        port_intervals = [3, 3, 4, 3, 3, 4, 3, 3, 4]
        selected_port_edges = []
        selected_edge_indices = []  # Track which edge indices we selected
        used_intersections = set()
        
        # Start from a random offset to add variety
        start_offset = random.randint(0, len(perimeter_edges) - 1)
        current_idx = start_offset
        
        for interval in port_intervals:
            # Move to the target position along the perimeter (at the interval)
            target_idx = (current_idx + interval) % len(perimeter_edges)
            
            # Calculate last placed port index for gap checking
            last_idx = selected_edge_indices[-1] if selected_edge_indices else None
            
            # Since all 30 edges are valid, we can place at the exact target position
            # Just need to check minimum gap and intersection conflicts
            found = False
            best_idx = None
            
            # Check minimum gap from last port
            if last_idx is not None:
                gap = (target_idx - last_idx) % len(perimeter_edges)
                if gap < 3:
                    # Target too close - skip this interval
                    current_idx = target_idx
                    continue
            
            # Check if intersections are available
            n1, n2 = perimeter_edges[target_idx]
            if (n1 not in used_intersections and n2 not in used_intersections):
                best_idx = target_idx
                found = True
            
            # Place port at the target position
            if found and best_idx is not None:
                n1, n2 = perimeter_edges[best_idx]
                # For port assignment, identify edge_inter and coastal_inter
                if n1 in edge_intersection_ids:
                    edge_inter_id, coastal_inter_id = n1, n2
                elif n2 in edge_intersection_ids:
                    edge_inter_id, coastal_inter_id = n2, n1
                else:
                    # Both are coastal (2 tiles) - just pick one as edge, one as coastal
                    edge_inter_id, coastal_inter_id = n1, n2
                selected_port_edges.append((edge_inter_id, coastal_inter_id))
                selected_edge_indices.append(best_idx)
                used_intersections.add(edge_inter_id)
                used_intersections.add(coastal_inter_id)
                # Continue from target to maintain exact spacing
                current_idx = target_idx
            else:
                # If intersections are in use, skip this interval
                # Continue from target to maintain spacing for next iteration
                current_idx = target_idx
        
        # If we don't have exactly 9 ports, try to fill remaining slots
        # while respecting minimum gap of 3
        while len(selected_port_edges) < 9:
            last_idx = selected_edge_indices[-1] if selected_edge_indices else None
            found_additional = False
            
            for idx in valid_port_edge_indices:
                if idx in selected_edge_indices:
                    continue
                
                # Check minimum gap
                if last_idx is not None:
                    gap = (idx - last_idx) % len(perimeter_edges)
                    if gap < 3:
                        continue
                
                n1, n2 = perimeter_edges[idx]
                if (n1 not in used_intersections and n2 not in used_intersections):
                    if n1 in edge_intersection_ids:
                        edge_inter_id, coastal_inter_id = n1, n2
                    else:
                        edge_inter_id, coastal_inter_id = n2, n1
                    selected_port_edges.append((edge_inter_id, coastal_inter_id))
                    selected_edge_indices.append(idx)
                    used_intersections.add(edge_inter_id)
                    used_intersections.add(coastal_inter_id)
                    found_additional = True
                    break
            
            if not found_additional:
                # Can't find more ports that respect gap - break to avoid infinite loop
                break
        
        # Ensure we have exactly 9 ports (trim if we have more)
        if len(selected_port_edges) > 9:
            selected_port_edges = selected_port_edges[:9]
            selected_edge_indices = selected_edge_indices[:9]
        
        # Port types: 4 generic 3:1 ports, 5 specific 2:1 ports (one per resource)
        port_assignments = ["3:1"] * 4 + [rt.value for rt in ResourceType]  # Total: 9 ports
        random.shuffle(port_assignments)  # Randomize which ports get which type
        
        # Assign ports to selected edges (exactly 9)
        for i, port_type in enumerate(port_assignments):
            if i >= len(selected_port_edges):
                break
            if i >= len(selected_port_edges):
                break
            inter1_id, inter2_id = selected_port_edges[i]
            
            # Assign port to both intersections in the edge pair
            inter1 = intersections[inter1_id]
            inter2 = intersections[inter2_id]
            
            intersections[inter1_id] = Intersection(
                id=inter1.id,
                position=inter1.position,
                adjacent_tiles=inter1.adjacent_tiles,
                adjacent_intersections=inter1.adjacent_intersections,
                owner=inter1.owner,
                building_type=inter1.building_type,
                port_type=port_type
            )
            
            intersections[inter2_id] = Intersection(
                id=inter2.id,
                position=inter2.position,
                adjacent_tiles=inter2.adjacent_tiles,
                adjacent_intersections=inter2.adjacent_intersections,
                owner=inter2.owner,
                building_type=inter2.building_type,
                port_type=port_type
                )
        
        state.intersections = intersections
        
        # Create road edges between adjacent intersections
        road_edges = []
        road_id = 0
        edge_set = set()  # Track edges to avoid duplicates
        
        for i, inter in enumerate(intersections):
            for j in inter.adjacent_intersections:
                edge_key = (min(i, j), max(i, j))
                if edge_key not in edge_set:
                    road_edge = RoadEdge(
                        id=road_id,
                        intersection1_id=i,
                        intersection2_id=j
                    )
                    road_edges.append(road_edge)
                    edge_set.add(edge_key)
                    road_id += 1
        
        state.road_edges = road_edges
        
        return state
    
    def _fix_adjacent_6_8(self, tiles: List[Tile], hex_coords: List[Tuple[int, int]]) -> List[Tile]:
        """Ensure 6 and 8 are not adjacent. Returns new list of tiles."""
        # Hex neighbor relationships (based on standard Catan layout)
        hex_neighbors = {
            0: [1, 2, 3, 4, 5, 6],
            1: [0, 2, 6, 7, 8, 18],
            2: [0, 1, 3, 8, 9, 10],
            3: [0, 2, 4, 10, 11, 12],
            4: [0, 3, 5, 12, 13, 14],
            5: [0, 4, 6, 14, 15, 16],
            6: [0, 1, 5, 16, 17, 18],
            7: [1, 8, 18],
            8: [1, 2, 7, 9],
            9: [2, 8, 10],
            10: [2, 3, 9, 11],
            11: [3, 10, 12],
            12: [3, 4, 11, 13],
            13: [4, 12, 14],
            14: [4, 5, 13, 15],
            15: [5, 14, 16],
            16: [5, 6, 15, 17],
            17: [6, 16, 18],
            18: [1, 6, 7, 17]
        }
        
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            has_violation = False
            
            # Check each tile
            for tile_idx, tile in enumerate(tiles):
                if tile.number_token and tile.number_token.value in [6, 8]:
                    # Check neighbors
                    for neighbor_idx in hex_neighbors.get(tile_idx, []):
                        if neighbor_idx < len(tiles):
                            neighbor = tiles[neighbor_idx]
                            if neighbor.number_token and neighbor.number_token.value in [6, 8]:
                                has_violation = True
                                break
                    if has_violation:
                        break
            
            if not has_violation:
                break
            
            # Shuffle number tokens and reassign
            number_list = []
            for tile in tiles:
                if tile.number_token:
                    number_list.append(tile.number_token.value)
            
            random.shuffle(number_list)
            num_idx = 0
            
            new_tiles = []
            for tile in tiles:
                if tile.resource_type is None:  # Desert
                    new_tiles.append(tile)
                else:
                    new_tiles.append(Tile(
                        id=tile.id,
                        resource_type=tile.resource_type,
                        number_token=NumberToken(number_list[num_idx]) if num_idx < len(number_list) else None,
                        position=tile.position
                    ))
                    num_idx += 1
            
            tiles = new_tiles
            iteration += 1
        
        return tiles
    
    def _handle_setup_place_settlement(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle placing initial settlement during setup."""
        if payload is None or not isinstance(payload, BuildSettlementPayload):
            raise ValueError("SETUP_PLACE_SETTLEMENT requires BuildSettlementPayload")
        
        if new_state.phase != "setup":
            raise ValueError("Can only place setup settlement during setup phase")
        
        current_player = new_state.players[new_state.setup_phase_player_index]
        
        # Check if intersection exists and is unowned
        intersection = next((i for i in new_state.intersections if i.id == payload.intersection_id), None)
        if not intersection:
            raise ValueError(f"Intersection {payload.intersection_id} not found")
        if intersection.owner:
            raise ValueError(f"Intersection {payload.intersection_id} already owned")
        
        # Check distance rule
        for adj_id in intersection.adjacent_intersections:
            adj_intersection = next((i for i in new_state.intersections if i.id == adj_id), None)
            if adj_intersection and adj_intersection.owner:
                raise ValueError("Cannot build settlement adjacent to another settlement")
        
        # Place settlement (no resource cost in setup)
        intersection_index = next(i for i, inter in enumerate(new_state.intersections) if inter.id == payload.intersection_id)
        new_state.intersections[intersection_index] = Intersection(
            id=intersection.id,
            position=intersection.position,
            adjacent_tiles=intersection.adjacent_tiles,
            adjacent_intersections=intersection.adjacent_intersections,
            owner=current_player.id,
            building_type="settlement",
            port_type=intersection.port_type  # Preserve port type
        )
        current_player.settlements_built += 1
        current_player.victory_points += 1
        
        # Track the last settlement placed for road placement requirement
        new_state.setup_last_settlement_id = payload.intersection_id
        
        # Track first player who placed settlement (in first round, first player)
        if new_state.setup_round == 0 and new_state.setup_phase_player_index == 0:
            new_state.setup_first_settlement_player_index = 0
        
        # In second round of setup, give resources from adjacent hexes
        if new_state.setup_round == 1:
            for tile_id in intersection.adjacent_tiles:
                tile = next((t for t in new_state.tiles if t.id == tile_id), None)
                if tile and tile.resource_type:  # Not desert
                    # Check if resource is available
                    available = new_state.resource_card_counts.get(tile.resource_type, 0)
                    if available > 0:
                        current_player.resources[tile.resource_type] += 1
                        new_state.resource_card_counts[tile.resource_type] -= 1
                    # If not available, player doesn't get the resource (scarcity rule)
        
        return new_state
    
    def _handle_setup_place_road(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle placing initial road during setup."""
        if payload is None or not isinstance(payload, BuildRoadPayload):
            raise ValueError("SETUP_PLACE_ROAD requires BuildRoadPayload")
        
        if new_state.phase != "setup":
            raise ValueError("Can only place setup road during setup phase")
        
        current_player = new_state.players[new_state.setup_phase_player_index]
        
        # Check if road edge exists and is unowned
        road_edge = next((r for r in new_state.road_edges if r.id == payload.road_edge_id), None)
        if not road_edge:
            raise ValueError(f"Road edge {payload.road_edge_id} not found")
        if road_edge.owner:
            raise ValueError(f"Road edge {payload.road_edge_id} already owned")
        
        # Check that road is adjacent to the settlement just placed
        if new_state.setup_last_settlement_id is None:
            raise ValueError("Cannot place road: no settlement was placed yet")
        if (road_edge.intersection1_id != new_state.setup_last_settlement_id and 
            road_edge.intersection2_id != new_state.setup_last_settlement_id):
            raise ValueError(f"Road must be adjacent to the settlement just placed (intersection {new_state.setup_last_settlement_id})")
        
        # Place road (no resource cost in setup)
        road_index = next(i for i, r in enumerate(new_state.road_edges) if r.id == payload.road_edge_id)
        new_state.road_edges[road_index] = RoadEdge(
            id=road_edge.id,
            intersection1_id=road_edge.intersection1_id,
            intersection2_id=road_edge.intersection2_id,
            owner=current_player.id
        )
        current_player.roads_built += 1
        
        # Clear the last settlement ID since road has been placed
        new_state.setup_last_settlement_id = None
        
        # Check for longest road (though unlikely in setup)
        self._check_longest_road(new_state, current_player)
        
        # Advance setup phase
        if new_state.setup_round == 0:
            # First round: clockwise
            new_state.setup_phase_player_index = (new_state.setup_phase_player_index + 1) % len(new_state.players)
            if new_state.setup_phase_player_index == 0:
                # All players placed, go to second round
                new_state.setup_round = 1
                new_state.setup_phase_player_index = len(new_state.players) - 1  # Start from last player
        else:
            # Second round: counter-clockwise
            new_state.setup_phase_player_index = (new_state.setup_phase_player_index - 1) % len(new_state.players)
            if new_state.setup_phase_player_index == len(new_state.players) - 1 and new_state.setup_round == 1:
                # Setup complete - ensure robber is initialized
                if new_state.robber_tile_id is None:
                    desert_tile = next((t for t in new_state.tiles if t.resource_type is None), None)
                    if desert_tile:
                        new_state.robber_tile_id = desert_tile.id
                new_state.phase = "playing"
                # First player (who placed first settlement) goes first
                new_state.current_player_index = new_state.setup_first_settlement_player_index if new_state.setup_first_settlement_player_index is not None else 0
        
        return new_state


class Action(Enum):
    """Actions that can be taken in the game."""
    ROLL_DICE = "roll_dice"
    BUILD_ROAD = "build_road"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUY_DEV_CARD = "buy_dev_card"
    PLAY_DEV_CARD = "play_dev_card"
    TRADE_BANK = "trade_bank"
    TRADE_PLAYER = "trade_player"  # Legacy - now use PROPOSE_TRADE
    PROPOSE_TRADE = "propose_trade"  # Propose trade to one or more players
    ACCEPT_TRADE = "accept_trade"  # Accept a pending trade offer
    REJECT_TRADE = "reject_trade"  # Reject a pending trade offer
    SELECT_TRADE_PARTNER = "select_trade_partner"  # Choose which accepting player to trade with
    END_TURN = "end_turn"
    START_GAME = "start_game"
    SETUP_PLACE_SETTLEMENT = "setup_place_settlement"
    SETUP_PLACE_ROAD = "setup_place_road"
    MOVE_ROBBER = "move_robber"
    STEAL_RESOURCE = "steal_resource"
    DISCARD_RESOURCES = "discard_resources"


# Action payload types
from typing import Union

ActionPayload = Union[
    'BuildRoadPayload',
    'BuildSettlementPayload',
    'BuildCityPayload',
    'PlayDevCardPayload',
    'TradeBankPayload',
    'TradePlayerPayload',
    'ProposeTradePayload',
    'SelectTradePartnerPayload',
    'MoveRobberPayload',
    'StealResourcePayload',
    'DiscardResourcesPayload',
]


@dataclass(frozen=True)
class BuildRoadPayload:
    """Payload for BUILD_ROAD action."""
    road_edge_id: int


@dataclass(frozen=True)
class BuildSettlementPayload:
    """Payload for BUILD_SETTLEMENT action."""
    intersection_id: int


@dataclass(frozen=True)
class BuildCityPayload:
    """Payload for BUILD_CITY action."""
    intersection_id: int


@dataclass(frozen=True)
class PlayDevCardPayload:
    """Payload for PLAY_DEV_CARD action."""
    card_type: str
    # For year_of_plenty: resources to receive (2 resources total)
    year_of_plenty_resources: Optional[Dict[ResourceType, int]] = None
    # For monopoly: resource type to steal from all players
    monopoly_resource_type: Optional[ResourceType] = None


@dataclass(frozen=True)
class TradeBankPayload:
    """Payload for TRADE_BANK action."""
    give_resources: Dict[ResourceType, int]  # Resources to give (can be multiple types)
    receive_resources: Dict[ResourceType, int]  # Resources to receive (can be multiple types)
    port_intersection_id: Optional[int] = None  # If using a port, the intersection ID with the port


@dataclass(frozen=True)
class TradePlayerPayload:
    """Payload for TRADE_PLAYER action (legacy - use ProposeTradePayload)."""
    other_player_id: str
    give_resources: Dict[ResourceType, int]  # Resources to give (can be multiple types)
    receive_resources: Dict[ResourceType, int]  # Resources to receive (can be multiple types)


@dataclass(frozen=True)
class ProposeTradePayload:
    """Payload for PROPOSE_TRADE action."""
    target_player_ids: List[str]  # List of player IDs to propose trade to
    give_resources: Dict[ResourceType, int]  # Resources proposer will give
    receive_resources: Dict[ResourceType, int]  # Resources proposer will receive


@dataclass(frozen=True)
class SelectTradePartnerPayload:
    """Payload for SELECT_TRADE_PARTNER action."""
    selected_player_id: str  # Which accepting player to trade with


@dataclass(frozen=True)
class MoveRobberPayload:
    """Payload for MOVE_ROBBER action."""
    tile_id: int


@dataclass(frozen=True)
class StealResourcePayload:
    """Payload for STEAL_RESOURCE action."""
    other_player_id: str


@dataclass(frozen=True)
class DiscardResourcesPayload:
    """Payload for DISCARD_RESOURCES action."""
    resources: Dict[ResourceType, int]  # Resources to discard

