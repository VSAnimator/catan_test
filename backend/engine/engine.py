"""
Pure game engine for Catan-like game.
No I/O, no globals - pure functional game logic.
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import random
import copy


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
    robber_tile_id: Optional[int] = None  # Tile ID where robber is located (None = desert)
    
    def step(self, action: 'Action', payload: Optional['ActionPayload'] = None) -> 'GameState':
        """
        Pure function that takes an action and returns a new GameState.
        No side effects, no I/O, no globals.
        """
        new_state = copy.deepcopy(self)
        
        if action == Action.ROLL_DICE:
            return self._handle_roll_dice(new_state)
        elif action == Action.BUILD_ROAD:
            return self._handle_build_road(new_state, payload)
        elif action == Action.BUILD_SETTLEMENT:
            return self._handle_build_settlement(new_state, payload)
        elif action == Action.BUILD_CITY:
            return self._handle_build_city(new_state, payload)
        elif action == Action.BUY_DEV_CARD:
            return self._handle_buy_dev_card(new_state)
        elif action == Action.PLAY_DEV_CARD:
            return self._handle_play_dev_card(new_state, payload)
        elif action == Action.TRADE_BANK:
            return self._handle_trade_bank(new_state, payload)
        elif action == Action.TRADE_PLAYER:
            return self._handle_trade_player(new_state, payload)
        elif action == Action.END_TURN:
            return self._handle_end_turn(new_state)
        elif action == Action.START_GAME:
            return self._handle_start_game(new_state)
        elif action == Action.SETUP_PLACE_SETTLEMENT:
            return self._handle_setup_place_settlement(new_state, payload)
        elif action == Action.SETUP_PLACE_ROAD:
            return self._handle_setup_place_road(new_state, payload)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _handle_roll_dice(self, new_state: 'GameState') -> 'GameState':
        """Handle dice roll - distribute resources based on number."""
        if new_state.phase != "playing":
            raise ValueError("Can only roll dice during playing phase")
        
        # Roll two dice (2-12)
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        roll = die1 + die2
        new_state.dice_roll = roll
        
        # Distribute resources if not 7
        if roll != 7:
            self._distribute_resources(new_state, roll)
        
        return new_state
    
    def _distribute_resources(self, state: 'GameState', roll: int):
        """Distribute resources to players based on dice roll."""
        for tile in state.tiles:
            # Skip if robber is on this tile
            if state.robber_tile_id == tile.id:
                continue
                
            if tile.number_token and tile.number_token.value == roll and tile.resource_type:
                # Find intersections on this tile that have settlements/cities
                for intersection in state.intersections:
                    # Check if this intersection is adjacent to this tile
                    if tile.id in intersection.adjacent_tiles:
                        # Check if intersection has a building
                        if intersection.owner and intersection.building_type:
                            player = next(p for p in state.players if p.id == intersection.owner)
                            # Give resources based on building type
                            if intersection.building_type == "settlement":
                                player.resources[tile.resource_type] += 1
                            elif intersection.building_type == "city":
                                player.resources[tile.resource_type] += 2
    
    def _handle_build_road(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle building a road."""
        if payload is None or not isinstance(payload, BuildRoadPayload):
            raise ValueError("BUILD_ROAD requires BuildRoadPayload")
        
        current_player = new_state.players[new_state.current_player_index]
        
        # Check resources: 1 wood, 1 brick
        if (current_player.resources[ResourceType.WOOD] < 1 or 
            current_player.resources[ResourceType.BRICK] < 1):
            raise ValueError("Insufficient resources to build road")
        
        # Check if road edge exists and is unowned
        road_edge = next((r for r in new_state.road_edges if r.id == payload.road_edge_id), None)
        if not road_edge:
            raise ValueError(f"Road edge {payload.road_edge_id} not found")
        if road_edge.owner:
            raise ValueError(f"Road edge {payload.road_edge_id} already owned")
        
        # Deduct resources
        current_player.resources[ResourceType.WOOD] -= 1
        current_player.resources[ResourceType.BRICK] -= 1
        
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
        
        # Build settlement
        intersection_index = next(i for i, inter in enumerate(new_state.intersections) if inter.id == payload.intersection_id)
        new_state.intersections[intersection_index] = Intersection(
            id=intersection.id,
            position=intersection.position,
            adjacent_tiles=intersection.adjacent_tiles,
            adjacent_intersections=intersection.adjacent_intersections,
            owner=current_player.id,
            building_type="settlement"
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
        
        # Upgrade to city
        intersection_index = next(i for i, inter in enumerate(new_state.intersections) if inter.id == payload.intersection_id)
        new_state.intersections[intersection_index] = Intersection(
            id=intersection.id,
            position=intersection.position,
            adjacent_tiles=intersection.adjacent_tiles,
            adjacent_intersections=intersection.adjacent_intersections,
            owner=current_player.id,
            building_type="city"
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
        
        # Deduct resources
        current_player.resources[ResourceType.WHEAT] -= 1
        current_player.resources[ResourceType.SHEEP] -= 1
        current_player.resources[ResourceType.ORE] -= 1
        
        # Add a random dev card (simplified)
        dev_card_types = ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"]
        card_type = random.choice(dev_card_types)
        current_player.dev_cards.append(card_type)
        
        return new_state
    
    def _handle_play_dev_card(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle playing a development card."""
        if not payload or not isinstance(payload, PlayDevCardPayload):
            raise ValueError("PLAY_DEV_CARD requires PlayDevCardPayload")
        
        current_player = new_state.players[new_state.current_player_index]
        
        # Check if player has the card
        if payload.card_type not in current_player.dev_cards:
            raise ValueError(f"Player does not have {payload.card_type} card")
        
        # Remove card from hand
        current_player.dev_cards.remove(payload.card_type)
        
        # Handle card effects
        if payload.card_type == "victory_point":
            current_player.victory_points += 1
        elif payload.card_type == "knight":
            current_player.knights_played += 1
            # Check for largest army
            self._check_largest_army(new_state, current_player)
        # Other cards (road_building, year_of_plenty, monopoly) would need additional payload
        
        return new_state
    
    def _handle_trade_bank(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle trading with the bank."""
        if not payload or not isinstance(payload, TradeBankPayload):
            raise ValueError("TRADE_BANK requires TradeBankPayload")
        
        current_player = new_state.players[new_state.current_player_index]
        
        # Check if player has enough of the resource they're giving
        if current_player.resources[payload.give_resource] < payload.give_amount:
            raise ValueError(f"Insufficient {payload.give_resource.value} to trade")
        
        # Standard trade: 4:1 ratio
        if payload.give_amount != 4 or payload.receive_amount != 1:
            raise ValueError("Bank trades must be 4:1 ratio")
        
        # Execute trade
        current_player.resources[payload.give_resource] -= payload.give_amount
        current_player.resources[payload.receive_resource] += payload.receive_amount
        
        return new_state
    
    def _handle_trade_player(self, new_state: 'GameState', payload: Optional['ActionPayload']) -> 'GameState':
        """Handle trading with another player."""
        if not payload or not isinstance(payload, TradePlayerPayload):
            raise ValueError("TRADE_PLAYER requires TradePlayerPayload")
        
        current_player = new_state.players[new_state.current_player_index]
        other_player = next((p for p in new_state.players if p.id == payload.other_player_id), None)
        
        if not other_player:
            raise ValueError(f"Player {payload.other_player_id} not found")
        
        # Check resources
        if current_player.resources[payload.give_resource] < payload.give_amount:
            raise ValueError(f"Insufficient {payload.give_resource.value} to trade")
        if other_player.resources[payload.receive_resource] < payload.receive_amount:
            raise ValueError(f"Other player has insufficient {payload.receive_resource.value}")
        
        # Execute trade
        current_player.resources[payload.give_resource] -= payload.give_amount
        current_player.resources[payload.receive_resource] += payload.receive_amount
        other_player.resources[payload.give_resource] += payload.give_amount
        other_player.resources[payload.receive_resource] -= payload.receive_amount
        
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
        
        def dfs_path_length(node: int, visited_nodes: Set[int], visited_edges: Set[Tuple[int, int]]) -> int:
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
    
    def _handle_end_turn(self, new_state: 'GameState') -> 'GameState':
        """Handle ending the current turn."""
        if new_state.phase != "playing":
            raise ValueError("Can only end turn during playing phase")
        
        # Advance to next player
        new_state.current_player_index = (new_state.current_player_index + 1) % len(new_state.players)
        new_state.turn_number += 1
        new_state.dice_roll = None  # Reset dice roll
        
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
                        adjacent_intersections=set()
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
                        building_type=existing_inter.building_type
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
                
                # Add to each other's adjacent list
                intersections[corner1_id] = Intersection(
                    id=inter1.id,
                    position=inter1.position,
                    adjacent_tiles=inter1.adjacent_tiles,
                    adjacent_intersections=inter1.adjacent_intersections | {corner2_id},
                    owner=inter1.owner,
                    building_type=inter1.building_type
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
            building_type="settlement"
        )
        current_player.settlements_built += 1
        current_player.victory_points += 1
        
        # In second round of setup, give resources from adjacent hexes
        if new_state.setup_round == 1:
            for tile_id in intersection.adjacent_tiles:
                tile = next((t for t in new_state.tiles if t.id == tile_id), None)
                if tile and tile.resource_type:  # Not desert
                    current_player.resources[tile.resource_type] += 1
        
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
        
        # Place road (no resource cost in setup)
        road_index = next(i for i, r in enumerate(new_state.road_edges) if r.id == payload.road_edge_id)
        new_state.road_edges[road_index] = RoadEdge(
            id=road_edge.id,
            intersection1_id=road_edge.intersection1_id,
            intersection2_id=road_edge.intersection2_id,
            owner=current_player.id
        )
        current_player.roads_built += 1
        
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
                new_state.current_player_index = len(new_state.players) - 1  # Last player goes first
        
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
    TRADE_PLAYER = "trade_player"
    END_TURN = "end_turn"
    START_GAME = "start_game"
    SETUP_PLACE_SETTLEMENT = "setup_place_settlement"
    SETUP_PLACE_ROAD = "setup_place_road"


# Action payload types
from typing import Union

ActionPayload = Union[
    'BuildRoadPayload',
    'BuildSettlementPayload',
    'BuildCityPayload',
    'PlayDevCardPayload',
    'TradeBankPayload',
    'TradePlayerPayload',
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


@dataclass(frozen=True)
class TradeBankPayload:
    """Payload for TRADE_BANK action."""
    give_resource: ResourceType
    give_amount: int
    receive_resource: ResourceType
    receive_amount: int


@dataclass(frozen=True)
class TradePlayerPayload:
    """Payload for TRADE_PLAYER action."""
    other_player_id: str
    give_resource: ResourceType
    give_amount: int
    receive_resource: ResourceType
    receive_amount: int

