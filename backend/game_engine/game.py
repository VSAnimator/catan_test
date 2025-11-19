"""
Pure game engine for Catan-like game.
No web framework dependencies - pure Python game logic.
"""
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field


class ResourceType(Enum):
    """Resource types in the game."""
    WOOD = "wood"
    BRICK = "brick"
    WHEAT = "wheat"
    SHEEP = "sheep"
    ORE = "ore"


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


@dataclass
class GameState:
    """Current state of the game."""
    game_id: str
    players: List[Player]
    current_turn: int = 0
    phase: str = "setup"  # setup, playing, finished
    board: Dict = field(default_factory=dict)


class Game:
    """Main game engine class."""
    
    def __init__(self, game_id: str, player_names: List[str]):
        """Initialize a new game."""
        if len(player_names) < 2 or len(player_names) > 4:
            raise ValueError("Game must have 2-4 players")
        
        self.game_id = game_id
        self.players = [
            Player(id=f"player_{i}", name=name)
            for i, name in enumerate(player_names)
        ]
        self.current_turn = 0
        self.phase = "setup"
        self.board = {}
    
    def get_state(self) -> GameState:
        """Get current game state."""
        return GameState(
            game_id=self.game_id,
            players=self.players.copy(),
            current_turn=self.current_turn,
            phase=self.phase,
            board=self.board.copy()
        )
    
    def get_current_player(self) -> Player:
        """Get the player whose turn it is."""
        return self.players[self.current_turn % len(self.players)]
    
    def next_turn(self) -> None:
        """Advance to the next player's turn."""
        if self.phase != "playing":
            raise ValueError("Game is not in playing phase")
        self.current_turn += 1
    
    def add_resource(self, player_id: str, resource: ResourceType, amount: int = 1) -> None:
        """Add resources to a player."""
        player = self._get_player(player_id)
        player.resources[resource] += amount
    
    def remove_resource(self, player_id: str, resource: ResourceType, amount: int = 1) -> bool:
        """Remove resources from a player. Returns True if successful."""
        player = self._get_player(player_id)
        if player.resources[resource] >= amount:
            player.resources[resource] -= amount
            return True
        return False
    
    def add_victory_point(self, player_id: str, points: int = 1) -> None:
        """Add victory points to a player."""
        player = self._get_player(player_id)
        player.victory_points += points
    
    def start_game(self) -> None:
        """Start the game (transition from setup to playing)."""
        if self.phase != "setup":
            raise ValueError("Game can only be started from setup phase")
        self.phase = "playing"
    
    def _get_player(self, player_id: str) -> Player:
        """Get a player by ID."""
        for player in self.players:
            if player.id == player_id:
                return player
        raise ValueError(f"Player {player_id} not found")

