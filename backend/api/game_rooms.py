"""
Game room/lobby management system.
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class RoomStatus(Enum):
    """Room status."""
    WAITING = "waiting"  # Waiting for players
    IN_PROGRESS = "in_progress"  # Game in progress
    FINISHED = "finished"  # Game finished


@dataclass
class RoomPlayer:
    """Player in a room."""
    user_id: str
    username: str
    player_id: Optional[str] = None  # Game player ID (assigned when game starts)
    joined_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class GameRoom:
    """A game room/lobby."""
    room_id: str
    host_user_id: str
    status: RoomStatus = RoomStatus.WAITING
    max_players: int = 4
    min_players: int = 2
    players: List[RoomPlayer] = field(default_factory=list)
    game_id: Optional[str] = None  # Assigned when game starts
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_private: bool = False
    password: Optional[str] = None  # For private rooms
    
    def add_player(self, user_id: str, username: str) -> bool:
        """Add a player to the room. Returns True if successful."""
        if self.status != RoomStatus.WAITING:
            return False
        if len(self.players) >= self.max_players:
            return False
        if any(p.user_id == user_id for p in self.players):
            return False  # Player already in room
        
        self.players.append(RoomPlayer(user_id=user_id, username=username))
        return True
    
    def remove_player(self, user_id: str) -> bool:
        """Remove a player from the room. Returns True if successful."""
        if self.status != RoomStatus.WAITING:
            return False
        
        original_count = len(self.players)
        self.players = [p for p in self.players if p.user_id != user_id]
        
        # If host left and room not empty, assign new host
        if self.host_user_id == user_id and len(self.players) > 0:
            self.host_user_id = self.players[0].user_id
        
        return len(self.players) < original_count
    
    def can_start(self) -> bool:
        """Check if the game can start."""
        return (
            self.status == RoomStatus.WAITING and
            len(self.players) >= self.min_players and
            len(self.players) <= self.max_players
        )
    
    def to_dict(self) -> dict:
        """Convert room to dictionary for JSON serialization."""
        return {
            "room_id": self.room_id,
            "host_user_id": self.host_user_id,
            "status": self.status.value,
            "max_players": self.max_players,
            "min_players": self.min_players,
            "players": [
                {
                    "user_id": p.user_id,
                    "username": p.username,
                    "player_id": p.player_id,
                    "joined_at": p.joined_at
                }
                for p in self.players
            ],
            "game_id": self.game_id,
            "created_at": self.created_at,
            "is_private": self.is_private,
            "player_count": len(self.players)
        }


class GameRoomManager:
    """Manages game rooms/lobbies."""
    
    def __init__(self):
        # room_id -> GameRoom
        self.rooms: Dict[str, GameRoom] = {}
        # user_id -> Set[room_id] (rooms user is in)
        self.user_rooms: Dict[str, Set[str]] = {}
    
    def create_room(
        self,
        host_user_id: str,
        max_players: int = 4,
        min_players: int = 2,
        is_private: bool = False,
        password: Optional[str] = None
    ) -> GameRoom:
        """Create a new game room."""
        room_id = str(uuid.uuid4())
        room = GameRoom(
            room_id=room_id,
            host_user_id=host_user_id,
            max_players=max_players,
            min_players=min_players,
            is_private=is_private,
            password=password
        )
        
        # Add host as first player
        room.add_player(host_user_id, "")  # Username will be set separately
        
        self.rooms[room_id] = room
        
        if host_user_id not in self.user_rooms:
            self.user_rooms[host_user_id] = set()
        self.user_rooms[host_user_id].add(room_id)
        
        return room
    
    def get_room(self, room_id: str) -> Optional[GameRoom]:
        """Get a room by ID."""
        return self.rooms.get(room_id)
    
    def join_room(self, room_id: str, user_id: str, username: str, password: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """Join a room. Returns (success, error_message)."""
        room = self.get_room(room_id)
        if not room:
            return False, "Room not found"
        
        if room.status != RoomStatus.WAITING:
            return False, "Room is not accepting new players"
        
        if room.is_private and room.password != password:
            return False, "Incorrect password"
        
        if len(room.players) >= room.max_players:
            return False, "Room is full"
        
        if any(p.user_id == user_id for p in room.players):
            return False, "Already in room"
        
        success = room.add_player(user_id, username)
        if success:
            if user_id not in self.user_rooms:
                self.user_rooms[user_id] = set()
            self.user_rooms[user_id].add(room_id)
            return True, None
        else:
            return False, "Failed to join room"
    
    def leave_room(self, room_id: str, user_id: str) -> bool:
        """Leave a room."""
        room = self.get_room(room_id)
        if not room:
            return False
        
        success = room.remove_player(user_id)
        if success:
            # Remove from user_rooms
            if user_id in self.user_rooms:
                self.user_rooms[user_id].discard(room_id)
            
            # Delete room if empty
            if len(room.players) == 0:
                del self.rooms[room_id]
        
        return success
    
    def start_game(self, room_id: str, game_id: str) -> bool:
        """Start a game in a room."""
        room = self.get_room(room_id)
        if not room:
            return False
        
        if not room.can_start():
            return False
        
        room.status = RoomStatus.IN_PROGRESS
        room.game_id = game_id
        return True
    
    def finish_game(self, room_id: str):
        """Mark a game as finished."""
        room = self.get_room(room_id)
        if room:
            room.status = RoomStatus.FINISHED
    
    def list_public_rooms(self) -> List[GameRoom]:
        """List all public waiting rooms."""
        return [
            room for room in self.rooms.values()
            if room.status == RoomStatus.WAITING and not room.is_private
        ]
    
    def get_user_rooms(self, user_id: str) -> List[GameRoom]:
        """Get all rooms a user is in."""
        room_ids = self.user_rooms.get(user_id, set())
        return [self.rooms[rid] for rid in room_ids if rid in self.rooms]


# Global room manager instance
room_manager = GameRoomManager()

