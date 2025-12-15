"""
Game room/lobby routes.
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Optional, List

from .auth_routes import get_current_user_from_token
from .auth import User
from .game_rooms import room_manager, GameRoom, RoomStatus

router = APIRouter()


class CreateRoomRequest(BaseModel):
    """Request to create a room."""
    max_players: int = 4
    min_players: int = 2
    is_private: bool = False
    password: Optional[str] = None


class JoinRoomRequest(BaseModel):
    """Request to join a room."""
    password: Optional[str] = None


class RoomResponse(BaseModel):
    """Room response model."""
    room_id: str
    host_user_id: str
    status: str
    max_players: int
    min_players: int
    players: List[dict]
    game_id: Optional[str]
    created_at: str
    is_private: bool
    player_count: int


@router.post("/rooms", response_model=RoomResponse)
async def create_room(
    request: CreateRoomRequest,
    current_user: User = Depends(get_current_user_from_token)
):
    """Create a new game room."""
    if request.max_players < 2 or request.max_players > 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_players must be between 2 and 4"
        )
    if request.min_players < 2 or request.min_players > request.max_players:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_players must be between 2 and max_players"
        )
    
    room = room_manager.create_room(
        host_user_id=current_user.id,
        max_players=request.max_players,
        min_players=request.min_players,
        is_private=request.is_private,
        password=request.password
    )
    
    # Update host's username
    if room.players:
        room.players[0].username = current_user.username
    
    return RoomResponse(**room.to_dict())


@router.get("/rooms", response_model=List[RoomResponse])
async def list_rooms():
    """List all public waiting rooms."""
    rooms = room_manager.list_public_rooms()
    return [RoomResponse(**room.to_dict()) for room in rooms]


@router.get("/rooms/{room_id}", response_model=RoomResponse)
async def get_room(room_id: str):
    """Get room information."""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Room not found"
        )
    return RoomResponse(**room.to_dict())


@router.post("/rooms/{room_id}/join", response_model=RoomResponse)
async def join_room(
    room_id: str,
    request: JoinRoomRequest,
    current_user: User = Depends(get_current_user_from_token)
):
    """Join a game room."""
    success, error = room_manager.join_room(
        room_id=room_id,
        user_id=current_user.id,
        username=current_user.username,
        password=request.password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Failed to join room"
        )
    
    room = room_manager.get_room(room_id)
    return RoomResponse(**room.to_dict())


@router.post("/rooms/{room_id}/leave")
async def leave_room(
    room_id: str,
    current_user: User = Depends(get_current_user_from_token)
):
    """Leave a game room."""
    success = room_manager.leave_room(room_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to leave room"
        )
    return {"message": "Left room successfully"}


@router.get("/rooms/user/my-rooms", response_model=List[RoomResponse])
async def get_my_rooms(current_user: User = Depends(get_current_user_from_token)):
    """Get all rooms the current user is in."""
    rooms = room_manager.get_user_rooms(current_user.id)
    return [RoomResponse(**room.to_dict()) for room in rooms]


@router.post("/rooms/{room_id}/start")
async def start_game_from_room(
    room_id: str,
    current_user: User = Depends(get_current_user_from_token)
):
    """Start a game from a room. Only the host can start the game."""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Room not found"
        )
    
    if room.host_user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the host can start the game"
        )
    
    if not room.can_start():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot start game: not enough players or room not in waiting status"
        )
    
    # Import game creation logic
    from api.routes import create_game_in_db
    from engine import GameState, Player, serialize_game_state
    import uuid
    import random
    
    # Generate game ID
    game_id = str(uuid.uuid4())
    
    # Create players from room players
    player_names = [p.username for p in room.players]
    players = [
        Player(
            id=f"player_{i}",
            name=player_names[i] if i < len(player_names) else f"Player {i+1}",
            color=["#FF0000", "#00AA00", "#2196F3", "#FF8C00"][i % 4]
        )
        for i in range(len(room.players))
    ]
    
    # Create initial game state
    initial_state = GameState(
        game_id=game_id,
        players=players,
        current_player_index=0,
        phase="setup"
    )
    
    # Initialize the board
    initial_state = initial_state._create_initial_board(initial_state)
    
    # Initialize robber on desert tile
    desert_tile = next((t for t in initial_state.tiles if t.resource_type is None), None)
    if desert_tile:
        initial_state.robber_tile_id = desert_tile.id
    
    # Serialize initial state
    serialized_state = serialize_game_state(initial_state)
    
    # Save game to database
    metadata = {
        "player_names": player_names,
        "num_players": len(players),
        "room_id": room_id,
    }
    create_game_in_db(
        game_id,
        rng_seed=None,
        metadata=metadata,
        initial_state_json=serialized_state,
    )
    
    # Update room with game ID and status
    room_manager.start_game(room_id, game_id)
    
    # Assign player IDs to room players
    for i, room_player in enumerate(room.players):
        room_player.player_id = players[i].id
    
    return {
        "game_id": game_id,
        "room": RoomResponse(**room.to_dict()),
        "initial_state": serialized_state
    }

