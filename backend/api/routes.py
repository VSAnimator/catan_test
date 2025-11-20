"""API routes for the Catan game."""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import uuid
import random
import json

# Random name generators
FIRST_NAMES = [
    "Alex", "Blake", "Casey", "Drew", "Emery", "Finley", "Gray", "Harper",
    "Jordan", "Kai", "Logan", "Morgan", "Parker", "Quinn", "Riley", "Sage",
    "Taylor", "Avery", "Cameron", "Dakota", "Ellis", "Hayden", "Jamie", "Kendall",
    "Lane", "Marley", "Noah", "Ocean", "Peyton", "Reese", "River", "Skylar"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris", "Clark",
    "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright", "Scott"
]

PLAYER_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F"]

def generate_random_name() -> str:
    """Generate a random player name."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"

def get_player_color(index: int) -> str:
    """Get a color for a player based on their index."""
    return PLAYER_COLORS[index % len(PLAYER_COLORS)]

from engine import (
    GameState,
    Player,
    Action,
    ActionPayload,
    serialize_game_state,
    deserialize_game_state,
    serialize_action,
    deserialize_action,
    serialize_action_payload,
    deserialize_action_payload,
    legal_actions,
    state_to_text,
    legal_actions_to_text,
)
from agents import RandomAgent
from agents.agent_runner import AgentRunner
from .database import (
    init_db,
    create_game as create_game_in_db,
    get_game as get_game_from_db,
    get_latest_state,
    save_game_state,
    add_step,
    get_steps,
    get_step_count,
)

router = APIRouter()


class StepLog(BaseModel):
    """Log entry for a game step."""
    state_before: Dict[str, Any]
    action: Dict[str, Any]  # Serialized action
    state_after: Dict[str, Any]
    dice_roll: Optional[int]
    timestamp: str


# Request/Response models
class CreateGameRequest(BaseModel):
    """Request to create a new game."""
    player_names: List[str]
    rng_seed: Optional[int] = None  # Optional RNG seed for reproducibility


class CreateGameResponse(BaseModel):
    """Response when creating a new game."""
    game_id: str
    initial_state: Dict[str, Any]


class ActRequest(BaseModel):
    """Request to perform an action."""
    player_id: str
    action: Dict[str, Any]  # Serialized Action JSON


class ActResponse(BaseModel):
    """Response after performing an action."""
    new_state: Dict[str, Any]


class ReplayResponse(BaseModel):
    """Response containing game replay logs."""
    game_id: str
    steps: List[StepLog]


@router.post("/games", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest):
    """Create a new game and return the initial state."""
    if len(request.player_names) < 2 or len(request.player_names) > 4:
        raise HTTPException(
            status_code=400,
            detail="Game must have 2-4 players"
        )
    
    # Generate game ID
    game_id = str(uuid.uuid4())
    
    # Set RNG seed if provided
    rng_seed = request.rng_seed
    if rng_seed is not None:
        random.seed(rng_seed)
    
    # Generate random names for players
    # If all names are the same (or empty), generate unique random names
    num_players = len(request.player_names)
    if len(set(request.player_names)) <= 1:
        # All names are the same or empty, generate unique random names
        final_names = []
        used_names = set()
        for _ in range(num_players):
            name = generate_random_name()
            # Ensure uniqueness
            while name in used_names:
                name = generate_random_name()
            used_names.add(name)
            final_names.append(name)
    else:
        # Use provided names, but ensure uniqueness
        used_names = set()
        final_names = []
        for name in request.player_names:
            if not name or name in used_names:
                # Generate a random name if empty or duplicate
                name = generate_random_name()
                while name in used_names:
                    name = generate_random_name()
            used_names.add(name)
            final_names.append(name)
    
    # Create players with colors
    players = [
        Player(id=f"player_{i}", name=final_names[i], color=get_player_color(i))
        for i in range(len(final_names))
    ]
    
    # Create initial game state
    initial_state = GameState(
        game_id=game_id,
        players=players,
        current_player_index=0,
        phase="setup"
    )
    
    # Initialize the board (tiles, intersections, road edges)
    # This is needed even in setup phase so players can place settlements
    initial_state = initial_state._create_initial_board(initial_state)
    
    # Initialize robber on desert tile
    desert_tile = next((t for t in initial_state.tiles if t.resource_type is None), None)
    if desert_tile:
        initial_state.robber_tile_id = desert_tile.id
    
    # Serialize initial state
    serialized_state = serialize_game_state(initial_state)
    
    # Save game to database with initial state
    metadata = {
        "player_names": final_names,
        "num_players": len(final_names),
    }
    create_game_in_db(
        game_id,
        rng_seed=rng_seed,
        metadata=metadata,
        initial_state_json=serialized_state,
    )
    
    return CreateGameResponse(
        game_id=game_id,
        initial_state=serialized_state
    )


@router.get("/games/{game_id}")
async def get_game(game_id: str):
    """Get current game state (serialized JSON)."""
    # Check if game exists
    game_row = get_game_from_db(game_id)
    if not game_row:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get latest state from database
    state_json = get_latest_state(game_id)
    if state_json is None:
        raise HTTPException(
            status_code=404,
            detail="Game state not found."
        )
    
    return state_json


@router.get("/games/{game_id}/legal_actions")
async def get_legal_actions(game_id: str, player_id: str):
    """Get legal actions for a player in the current game state."""
    # Check if game exists
    game_row = get_game_from_db(game_id)
    if not game_row:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get latest state from database
    state_json = get_latest_state(game_id)
    if state_json is None:
        raise HTTPException(
            status_code=404,
            detail="Game state not found."
        )
    
    # Deserialize state
    current_state = deserialize_game_state(state_json)
    
    # Get legal actions
    legal_actions_list = legal_actions(current_state, player_id)
    
    # Serialize actions for JSON response
    serialized_actions = []
    for action, payload in legal_actions_list:
        action_dict = {
            "type": serialize_action(action),
        }
        if payload:
            action_dict["payload"] = serialize_action_payload(payload)
        serialized_actions.append(action_dict)
    
    return {"legal_actions": serialized_actions}


@router.post("/games/{game_id}/act", response_model=ActResponse)
async def act(game_id: str, request: ActRequest):
    """Apply an action to the game and return the new state."""
    # Check if game exists
    game_row = get_game_from_db(game_id)
    if not game_row:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get current state from database
    state_json = get_latest_state(game_id)
    if state_json is None:
        raise HTTPException(
            status_code=404,
            detail="Game state not found. Game may not have been initialized."
        )
    
    # Deserialize current state
    current_state = deserialize_game_state(state_json)
    
    # Restore RNG seed if present for reproducibility
    if game_row["rng_seed"] is not None:
        random.seed(game_row["rng_seed"])
    
    # Verify player exists
    player = next((p for p in current_state.players if p.id == request.player_id), None)
    if not player:
        raise HTTPException(status_code=400, detail=f"Player {request.player_id} not found in game")
    
    # Verify it's the player's turn (if in playing phase)
    # Exception: When a 7 is rolled, any player with 8+ resources can discard
    action_type = None
    if "type" in request.action:
        action_type = request.action.get("type")
    
    allow_out_of_turn = False
    if current_state.phase == "playing" and current_state.dice_roll == 7:
        if action_type == "discard_resources":
            # Any player can discard when 7 is rolled
            player = next((p for p in current_state.players if p.id == request.player_id), None)
            if player and sum(player.resources.values()) >= 8:
                allow_out_of_turn = True
    
    if not allow_out_of_turn:
        if current_state.phase == "playing":
            current_player = current_state.players[current_state.current_player_index]
            if current_player.id != request.player_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"It's not {request.player_id}'s turn. Current player: {current_player.id}"
                )
        elif current_state.phase == "setup":
            setup_player = current_state.players[current_state.setup_phase_player_index]
            if setup_player.id != request.player_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"It's not {request.player_id}'s turn in setup. Current player: {setup_player.id}"
                )
    
    # Deserialize action
    try:
        action_dict = request.action
        if not isinstance(action_dict, dict):
            raise ValueError("Action must be a JSON object")
        
        action_type = action_dict.get("type")
        if not action_type:
            raise ValueError("Action must have a 'type' field")
        
        # Deserialize action enum
        action = deserialize_action(action_type)
        
        # Deserialize payload if present
        payload: Optional[ActionPayload] = None
        if "payload" in action_dict:
            payload_data = action_dict["payload"]
            if payload_data is not None:
                if isinstance(payload_data, dict):
                    payload = deserialize_action_payload(payload_data)
                else:
                    raise ValueError("Payload must be a JSON object or null")
        
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid action format: {str(e)}")
    
    # Store state before
    state_before = serialize_game_state(current_state)
    
    # Get legal actions and text representations before applying action
    legal_actions_list = legal_actions(current_state, request.player_id)
    legal_actions_text = legal_actions_to_text(legal_actions_list)
    state_text = state_to_text(current_state, request.player_id)
    
    # Format chosen action text
    action_type_str = request.action.get("type", "")
    chosen_action_text = action_type_str.replace("_", " ").title()
    if "payload" in request.action and request.action["payload"]:
        payload_dict = request.action["payload"]
        if isinstance(payload_dict, dict):
            if "intersection_id" in payload_dict:
                chosen_action_text += f" at intersection {payload_dict['intersection_id']}"
            elif "road_edge_id" in payload_dict:
                chosen_action_text += f" on road edge {payload_dict['road_edge_id']}"
            elif "card_type" in payload_dict:
                chosen_action_text += f" ({payload_dict['card_type']})"
    
    # Apply step
    try:
        # Pass player_id for actions that can be done out of turn (like DISCARD_RESOURCES)
        new_state = current_state.step(action, payload, player_id=request.player_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")
    
    # Serialize new state
    state_after = serialize_game_state(new_state)
    
    # Update current state in games table
    save_game_state(game_id, state_after)
    
    # Get step index
    step_idx = get_step_count(game_id)
    
    # Save step to database
    add_step(
        game_id=game_id,
        step_idx=step_idx,
        player_id=request.player_id,
        state_before_json=state_before,
        state_after_json=state_after,
        action_json=request.action,
        dice_roll=new_state.dice_roll,
        state_text=state_text,
        legal_actions_text=legal_actions_text,
        chosen_action_text=chosen_action_text,
    )
    
    return ActResponse(new_state=state_after)


@router.get("/games/{game_id}/replay", response_model=ReplayResponse)
async def get_replay(game_id: str):
    """Get the sequence of logged steps for a game."""
    # Check if game exists
    game_row = get_game_from_db(game_id)
    if not game_row:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get steps from database
    step_rows = get_steps(game_id)
    
    # Convert to StepLog format
    steps = []
    for row in step_rows:
        step_log = StepLog(
            state_before=json.loads(row["state_before_json"]),
            action=json.loads(row["action_json"]),
            state_after=json.loads(row["state_after_json"]),
            dice_roll=row["dice_roll"],
            timestamp=row["timestamp"],
        )
        steps.append(step_log)
    
    return ReplayResponse(
        game_id=game_id,
        steps=steps
    )


@router.post("/games/{game_id}/restore")
async def restore_game_state(game_id: str, state: Dict[str, Any]):
    """Restore a game to a specific state (for debugging/resuming from replay).
    
    This allows you to set the game's current state to any state, useful for:
    - Resuming from a replay step
    - Debugging by restoring to a specific point in the game
    
    WARNING: This modifies the original game. Use /fork instead to create a copy.
    """
    # Check if game exists
    game_row = get_game_from_db(game_id)
    if not game_row:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Validate that the state has required fields
    if "game_id" not in state:
        raise HTTPException(status_code=400, detail="State must have game_id")
    if state["game_id"] != game_id:
        raise HTTPException(status_code=400, detail="State game_id must match URL game_id")
    
    # Validate the state by trying to deserialize it
    try:
        deserialize_game_state(state)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid game state: {str(e)}")
    
    # Save the state as the current game state
    save_game_state(game_id, state)
    
    return {"message": "Game state restored successfully", "game_id": game_id}


@router.post("/games/{game_id}/fork", response_model=CreateGameResponse)
async def fork_game(game_id: str, state: Dict[str, Any]):
    """Fork a game from a specific state, creating a new game with a new ID.
    
    This creates a copy of the game at the specified state, preserving the original.
    Useful for:
    - Testing different paths from a saved state
    - Debugging without modifying the original game
    - Creating branches for experimentation
    """
    # Check if source game exists
    source_game_row = get_game_from_db(game_id)
    if not source_game_row:
        raise HTTPException(status_code=404, detail="Source game not found")
    
    # Validate the state by trying to deserialize it
    try:
        deserialized_state = deserialize_game_state(state)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid game state: {str(e)}")
    
    # Generate new game ID
    new_game_id = str(uuid.uuid4())
    
    # Update the state to use the new game ID
    state["game_id"] = new_game_id
    deserialized_state.game_id = new_game_id
    
    # Copy metadata from source game
    source_metadata = {}
    if source_game_row["metadata"]:
        source_metadata = json.loads(source_game_row["metadata"])
    
    # Create metadata for the forked game
    metadata = {
        "player_names": [p.name for p in deserialized_state.players],
        "num_players": len(deserialized_state.players),
        "forked_from": game_id,
        "forked_at_step": None,  # Could be enhanced to track which step was forked
    }
    
    # Create the new game in the database
    create_game_in_db(
        new_game_id,
        rng_seed=source_game_row["rng_seed"],  # Preserve RNG seed for reproducibility
        metadata=metadata,
        initial_state_json=state,
    )
    
    return CreateGameResponse(
        game_id=new_game_id,
        initial_state=state
    )


class RunAgentsRequest(BaseModel):
    """Request to run agents automatically."""
    max_turns: int = 1000  # Maximum number of turns


class RunAgentsResponse(BaseModel):
    """Response from running agents automatically."""
    game_id: str
    completed: bool  # True if game finished normally, False if stopped early
    error: Optional[str] = None  # Error message if stopped early
    final_state: Dict[str, Any]  # Final game state
    turns_played: int  # Number of turns played


@router.post("/games/{game_id}/run_agents", response_model=RunAgentsResponse)
async def run_agents(game_id: str, request: RunAgentsRequest):
    """Run agents automatically until game ends, error, or max turns reached.
    
    This mode runs the game completely automatically with agents for all players.
    Returns the game ID so you can replay the game to see what happened.
    """
    # Check if game exists
    game_row = get_game_from_db(game_id)
    if not game_row:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get current state from database
    state_json = get_latest_state(game_id)
    if state_json is None:
        raise HTTPException(
            status_code=404,
            detail="Game state not found. Game may not have been initialized."
        )
    
    # Deserialize current state
    current_state = deserialize_game_state(state_json)
    
    # Restore RNG seed if present for reproducibility
    if game_row["rng_seed"] is not None:
        random.seed(game_row["rng_seed"])
    
    # Create agents for all players
    agents = {}
    for player in current_state.players:
        agents[player.id] = RandomAgent(player.id)
    
    # Create agent runner
    runner = AgentRunner(current_state, agents, max_turns=request.max_turns)
    
    # Callback to save state after each action
    def save_state_callback(game_id: str, state_before: GameState, state_after: GameState, action: Dict[str, Any], player_id: str):
        # Serialize states
        state_before_json = serialize_game_state(state_before)
        state_after_json = serialize_game_state(state_after)
        
        # Update current state in games table
        save_game_state(game_id, state_after)
        
        # Get step index
        step_idx = get_step_count(game_id)
        
        # Get legal actions and text representations
        legal_actions_list = legal_actions(state, player_id)
        legal_actions_text = legal_actions_to_text(legal_actions_list)
        state_text = state_to_text(state, player_id)
        
        # Format chosen action text
        action_type_str = action.get("type", "")
        chosen_action_text = action_type_str.replace("_", " ").title()
        if "payload" in action and action["payload"]:
            payload_dict = action["payload"]
            if isinstance(payload_dict, dict):
                if "intersection_id" in payload_dict:
                    chosen_action_text += f" at intersection {payload_dict['intersection_id']}"
                elif "road_edge_id" in payload_dict:
                    chosen_action_text += f" on road edge {payload_dict['road_edge_id']}"
                elif "card_type" in payload_dict:
                    chosen_action_text += f" ({payload_dict['card_type']})"
        
        # Save step to database
        add_step(
            game_id=game_id,
            step_idx=step_idx,
            player_id=player_id,
            state_before_json=state_before,
            state_after_json=state_after,
            action_json=action,
            dice_roll=state.dice_roll,
            state_text=state_text,
            legal_actions_text=legal_actions_text,
            chosen_action_text=chosen_action_text,
        )
    
    # Run the game automatically
    final_state, completed, error = runner.run_automatic(save_state_callback=save_state_callback)
    
    # Serialize final state
    final_state_json = serialize_game_state(final_state)
    
    return RunAgentsResponse(
        game_id=game_id,
        completed=completed,
        error=error,
        final_state=final_state_json,
        turns_played=runner.turn_count
    )


class WatchAgentsRequest(BaseModel):
    """Request to watch agents play (step-by-step mode)."""
    pass  # No parameters needed, just uses current game state


class WatchAgentsResponse(BaseModel):
    """Response from watching agents (single step)."""
    game_id: str
    game_continues: bool  # True if game should continue, False if finished/error
    error: Optional[str] = None  # Error message if stopped
    new_state: Dict[str, Any]  # New game state after action
    player_id: Optional[str] = None  # Player who took the action


@router.post("/games/{game_id}/watch_agents_step", response_model=WatchAgentsResponse)
async def watch_agents_step(game_id: str, request: WatchAgentsRequest):
    """Execute a single step with agents (for agent-watching mode).
    
    This endpoint executes one action and returns the new state.
    The frontend can call this repeatedly with a delay to watch agents play.
    """
    # Check if game exists
    game_row = get_game_from_db(game_id)
    if not game_row:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get current state from database
    state_json = get_latest_state(game_id)
    if state_json is None:
        raise HTTPException(
            status_code=404,
            detail="Game state not found. Game may not have been initialized."
        )
    
    # Deserialize current state
    current_state = deserialize_game_state(state_json)
    
    # Restore RNG seed if present for reproducibility
    if game_row["rng_seed"] is not None:
        random.seed(game_row["rng_seed"])
    
    # Create agents for all players
    agents = {}
    for player in current_state.players:
        agents[player.id] = RandomAgent(player.id)
    
    # Create agent runner
    runner = AgentRunner(current_state, agents, max_turns=1000)
    
    # Callback to save state after each action
    def save_state_callback(game_id: str, state_before: GameState, state_after: GameState, action: Dict[str, Any], player_id: str):
        # Serialize states
        state_before_json = serialize_game_state(state_before)
        state_after_json = serialize_game_state(state_after)
        
        # Update current state in games table
        save_game_state(game_id, state_after_json)
        
        # Get step index
        step_idx = get_step_count(game_id)
        
        # Get legal actions and text representations (use state_before for context)
        legal_actions_list = legal_actions(state_before, player_id)
        legal_actions_text = legal_actions_to_text(legal_actions_list)
        state_text = state_to_text(state_before, player_id)
        
        # Format chosen action text
        action_type_str = action.get("type", "")
        chosen_action_text = action_type_str.replace("_", " ").title()
        if "payload" in action and action["payload"]:
            payload_dict = action["payload"]
            if isinstance(payload_dict, dict):
                if "intersection_id" in payload_dict:
                    chosen_action_text += f" at intersection {payload_dict['intersection_id']}"
                elif "road_edge_id" in payload_dict:
                    chosen_action_text += f" on road edge {payload_dict['road_edge_id']}"
                elif "card_type" in payload_dict:
                    chosen_action_text += f" ({payload_dict['card_type']})"
        
        # Save step to database
        add_step(
            game_id=game_id,
            step_idx=step_idx,
            player_id=player_id,
            state_before_json=state_before_json,
            state_after_json=state_after_json,
            action_json=action,
            dice_roll=state_after.dice_roll,
            state_text=state_text,
            legal_actions_text=legal_actions_text,
            chosen_action_text=chosen_action_text,
        )
    
    # Run a single step
    new_state, game_continues, error, player_id = runner.run_step(save_state_callback=save_state_callback)
    
    # Serialize new state
    new_state_json = serialize_game_state(new_state)
    
    return WatchAgentsResponse(
        game_id=game_id,
        game_continues=game_continues,
        error=error,
        new_state=new_state_json,
        player_id=player_id
    )
