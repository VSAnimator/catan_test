"""API routes for the Catan game."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
from datetime import datetime
import asyncio
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

PLAYER_COLORS = ["#FF0000", "#00AA00", "#2196F3", "#FF8C00"]  # Red, Green, Blue, Yellow-Orange

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
    save_optimized_prompt,
    get_optimized_prompt,
    get_default_optimized_prompt,
    list_optimized_prompts,
    set_default_prompt,
    delete_optimized_prompt,
    delete_drill,
)

router = APIRouter()


class StepLog(BaseModel):
    """Log entry for a game step."""
    player_id: Optional[str] = None
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
    exclude_strategic_advice: bool = False  # Exclude strategic advice from LLM prompts
    exclude_higher_level_features: bool = False  # Exclude higher-level features from LLM prompts


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
        "exclude_strategic_advice": request.exclude_strategic_advice,
        "exclude_higher_level_features": request.exclude_higher_level_features,
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
    """Get current game state (serialized JSON) and metadata."""
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
    
    # Parse metadata to include agent_mapping
    metadata = {}
    if game_row["metadata"]:
        try:
            metadata = json.loads(game_row["metadata"])
        except:
            metadata = {}
    
    # Add metadata to response (including agent_mapping if present)
    response = state_json.copy()
    if metadata:
        response["_metadata"] = metadata
    
    return response


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
            player_id=row["player_id"],
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


# ============================================================================
# Drills API Endpoints (curated "best action" datasets)
# ============================================================================

from .database import (
    create_drill as create_drill_in_db,
    list_drills as list_drills_from_db,
    get_drill as get_drill_from_db,
    get_drill_steps as get_drill_steps_from_db,
    update_drill as update_drill_in_db,
    delete_drill,
)


class DrillStepCreate(BaseModel):
    player_id: str
    state: Dict[str, Any]
    expected_action: Dict[str, Any]  # For backward compatibility
    correct_actions: Optional[List[Dict[str, Any]]] = None
    incorrect_actions: Optional[List[Dict[str, Any]]] = None


class CreateDrillRequest(BaseModel):
    name: Optional[str] = None
    guideline_text: Optional[str] = None
    source_game_id: Optional[str] = None
    source_step_idx: Optional[int] = None
    player_id: str
    steps: List[DrillStepCreate]
    metadata: Optional[Dict[str, Any]] = None


class CreateDrillResponse(BaseModel):
    drill_id: int
    message: str


@router.get("/drills/test")
async def test_drills():
    """Simple test endpoint to verify routing works."""
    return {"status": "ok", "message": "Drills endpoint is reachable"}

@router.get("/drills")
async def list_drills(limit: int = 200):
    import time
    import asyncio
    try:
        print(f"[DEBUG] list_drills: Starting, limit={limit}", flush=True)
        start_time = time.time()
        # Run database query in thread pool to avoid blocking
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, list_drills_from_db, limit)
        query_time = time.time() - start_time
        print(f"[DEBUG] list_drills: Query completed in {query_time:.4f}s, got {len(rows)} rows", flush=True)
        
        start_time = time.time()
        drills = []
        for r in rows:
            # Direct access - columns are guaranteed by the SELECT statement
            metadata_val = r["metadata"]
            try:
                metadata_parsed = json.loads(metadata_val) if metadata_val else None
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse metadata for drill {r['id']}: {e}", flush=True)
                metadata_parsed = None
            
            drills.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "name": r["name"],
                    "guideline_text": r["guideline_text"],
                    "source_game_id": r["source_game_id"],
                    "source_step_idx": r["source_step_idx"],
                    "player_id": r["player_id"],
                    "metadata": metadata_parsed,
                    "num_steps": r["num_steps"],
                }
            )
        processing_time = time.time() - start_time
        total_time = query_time + processing_time
        
        print(f"[DEBUG] list_drills: Complete in {total_time:.4f}s (query={query_time:.4f}s, processing={processing_time:.4f}s)", flush=True)
        return {"drills": drills}
    except Exception as e:
        print(f"[ERROR] list_drills failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list drills: {str(e)}")


@router.get("/drills/{drill_id}")
async def get_drill(drill_id: int):
    drill_row = get_drill_from_db(drill_id)
    if not drill_row:
        raise HTTPException(status_code=404, detail="Drill not found")
    step_rows = get_drill_steps_from_db(drill_id)
    return {
        "drill": {
            "id": drill_row["id"],
            "created_at": drill_row["created_at"],
            "name": drill_row["name"],
            "guideline_text": drill_row["guideline_text"] if "guideline_text" in drill_row.keys() else None,
            "source_game_id": drill_row["source_game_id"],
            "source_step_idx": drill_row["source_step_idx"],
            "player_id": drill_row["player_id"],
            "metadata": json.loads(drill_row["metadata"]) if drill_row["metadata"] else None,
        },
        "steps": [
            {
                "idx": r["idx"],
                "player_id": r["player_id"],
                "state": json.loads(r["state_json"]),
                "expected_action": json.loads(r["expected_action_json"]),
                "state_text": r["state_text"],
                "legal_actions_text": r["legal_actions_text"],
                "correct_actions": json.loads(r["correct_actions_json"]) if "correct_actions_json" in r.keys() and r["correct_actions_json"] else None,
                "incorrect_actions": json.loads(r["incorrect_actions_json"]) if "incorrect_actions_json" in r.keys() and r["incorrect_actions_json"] else None,
            }
            for r in step_rows
        ],
    }


@router.post("/drills", response_model=CreateDrillResponse)
async def create_drill_endpoint(request: CreateDrillRequest):
    if not request.steps:
        raise HTTPException(status_code=400, detail="Drill must have at least 1 step")

    # Precompute state_text / legal_actions_text for better UX in drill viewer
    steps_for_db: List[Dict[str, Any]] = []
    for idx, s in enumerate(request.steps):
        try:
            state = deserialize_game_state(s.state)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid drill step state at idx={idx}: {str(e)}")

        la_list = legal_actions(state, s.player_id)
        
        # Validate correct/incorrect actions if provided
        correct_actions_json = None
        incorrect_actions_json = None
        
        if s.correct_actions is not None or s.incorrect_actions is not None:
            # Validate that at least one correct action exists
            if not s.correct_actions or len(s.correct_actions) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Drill step at idx={idx} must have at least one correct action"
                )
            
            # Validate all correct actions are legal
            for correct_action in s.correct_actions:
                if not _action_dict_matches_legal_action(correct_action, la_list):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Drill step at idx={idx} has a correct action that is not legal: {correct_action}"
                    )
            
            # Validate all incorrect actions are legal (if provided)
            if s.incorrect_actions:
                for incorrect_action in s.incorrect_actions:
                    if not _action_dict_matches_legal_action(incorrect_action, la_list):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Drill step at idx={idx} has an incorrect action that is not legal: {incorrect_action}"
                        )
            
            correct_actions_json = s.correct_actions
            incorrect_actions_json = s.incorrect_actions
        
        steps_for_db.append(
            {
                "idx": idx,
                "player_id": s.player_id,
                "state_json": s.state,
                "expected_action_json": s.expected_action,
                "state_text": state_to_text(state, s.player_id),
                "legal_actions_text": legal_actions_to_text(la_list),
                "correct_actions_json": correct_actions_json,
                "incorrect_actions_json": incorrect_actions_json,
            }
        )

    drill_id = create_drill_in_db(
        name=request.name,
        guideline_text=request.guideline_text,
        source_game_id=request.source_game_id,
        source_step_idx=request.source_step_idx,
        player_id=request.player_id,
        metadata=request.metadata,
        steps=steps_for_db,
    )
    return CreateDrillResponse(drill_id=drill_id, message="Drill created")


class UpdateDrillRequest(BaseModel):
    name: Optional[str] = None
    guideline_text: Optional[str] = None


@router.put("/drills/{drill_id}")
async def update_drill_endpoint(drill_id: int, request: UpdateDrillRequest):
    updated = update_drill_in_db(drill_id, **request.dict(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Drill not found or no fields to update")
    return {"message": "Drill updated"}


@router.delete("/drills/{drill_id}")
async def delete_drill_endpoint(drill_id: int):
    """Delete a drill and all its steps."""
    deleted = delete_drill(drill_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Drill not found")
    return {"message": "Drill deleted"}


# =============================================================================
# Drill Generation from Game Disagreements
# =============================================================================

class ExtractDrillCandidatesRequest(BaseModel):
    num_steps: int
    player_id: Optional[str] = None
    include_setup_actions: bool = True
    include_non_setup_actions: bool = True
    include_trade_proposals: bool = True
    include_building_actions: bool = True
    include_turn_ending_actions: bool = True
    include_play_dev_card_actions: bool = True


class ExtractDrillCandidatesResponse(BaseModel):
    candidates: List[Dict[str, Any]]


@router.post("/games/{game_id}/extract_drill_candidates", response_model=ExtractDrillCandidatesResponse)
async def extract_drill_candidates(
    game_id: str,
    request: ExtractDrillCandidatesRequest
):
    """
    Extract non-trivial steps from a game where the player had multiple legal actions.
    
    Returns: List of candidate steps with state_before, player_id, step_idx
    """
    import asyncio
    import time
    
    start_time = time.time()
    
    # Check if game exists
    game_row = get_game_from_db(game_id)
    if not game_row:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get all steps from the game
    steps = get_steps(game_id)
    if not steps:
        raise HTTPException(status_code=404, detail="No steps found for this game")
    
    print(f"[extract_drill_candidates] Processing {len(steps)} steps for game {game_id}", flush=True)
    
    # Run the heavy computation in a thread pool to avoid blocking the event loop
    def process_steps():
        candidates = []
        
        # Filter steps to find non-trivial ones (multiple legal actions)
        for step in steps:
            # Get player_id from step
            step_player_id = step["player_id"]
            if not step_player_id:
                continue
            
            # If player_id filter is specified, skip if doesn't match
            if request.player_id and step_player_id != request.player_id:
                continue
            
            # Get state_before_json
            state_before_json_str = step["state_before_json"] if "state_before_json" in step.keys() else None
            if not state_before_json_str:
                continue
            
            try:
                state_before_json = json.loads(state_before_json_str) if isinstance(state_before_json_str, str) else state_before_json_str
                state = deserialize_game_state(state_before_json)
            except Exception as e:
                # Skip steps with invalid state
                continue
            
            # Get legal actions for this player at this step
            legal_actions_list = legal_actions(state, step_player_id)
            
            # Only include if player had multiple legal actions (non-trivial)
            if len(legal_actions_list) <= 1:
                continue
            
            # Get the actual action that was taken to filter by action type
            action_json_str = step["action_json"] if "action_json" in step.keys() else None
            if not action_json_str:
                continue
            
            try:
                action_json = json.loads(action_json_str) if isinstance(action_json_str, str) else action_json_str
                action_type = action_json.get("type", "")
            except Exception:
                continue
            
            # Categorize action type
            setup_actions = {"setup_place_settlement", "setup_place_road", "start_game"}
            building_actions = {"build_road", "build_settlement", "build_city", "buy_dev_card"}
            trade_proposal_actions = {"propose_trade"}
            turn_ending_actions = {"end_turn"}
            play_dev_card_actions = {"play_dev_card"}
            
            is_setup = action_type in setup_actions
            is_non_setup = action_type not in setup_actions
            is_trade_proposal = action_type in trade_proposal_actions
            is_building = action_type in building_actions
            is_turn_ending = action_type in turn_ending_actions
            is_play_dev_card = action_type in play_dev_card_actions
            
            # Apply filters: include if action matches at least one enabled category
            # An action can belong to multiple categories (e.g., building + non-setup)
            matches_enabled_category = False
            
            if is_setup and request.include_setup_actions:
                matches_enabled_category = True
            if is_non_setup and request.include_non_setup_actions:
                matches_enabled_category = True
            if is_trade_proposal and request.include_trade_proposals:
                matches_enabled_category = True
            if is_building and request.include_building_actions:
                matches_enabled_category = True
            if is_turn_ending and request.include_turn_ending_actions:
                matches_enabled_category = True
            if is_play_dev_card and request.include_play_dev_card_actions:
                matches_enabled_category = True
            
            if not matches_enabled_category:
                continue
            
            candidates.append({
                "step_idx": step["step_idx"],
                "player_id": step_player_id,
                "state_before_json": state_before_json,
                "legal_actions_count": len(legal_actions_list)
            })
        
        # Randomly sample from candidates to get diverse distribution throughout the game
        if len(candidates) > request.num_steps:
            import random
            candidates = random.sample(candidates, request.num_steps)
            # Sort by step_idx to maintain chronological order in response
            candidates.sort(key=lambda x: x["step_idx"])
        
        return candidates
    
    # Run in executor to avoid blocking - use asyncio.to_thread if available (Python 3.9+)
    # Otherwise fall back to run_in_executor
    try:
        # Python 3.9+ has asyncio.to_thread which is cleaner
        candidates = await asyncio.to_thread(process_steps)
    except AttributeError:
        # Fallback for Python < 3.9
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        candidates = await loop.run_in_executor(None, process_steps)
    except Exception as e:
        print(f"[extract_drill_candidates] Error processing steps: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error processing steps: {str(e)}")
    
    elapsed = time.time() - start_time
    print(f"[extract_drill_candidates] Completed in {elapsed:.2f}s, found {len(candidates)} candidates", flush=True)
    
    return ExtractDrillCandidatesResponse(candidates=candidates)


class CompareLLMActionsRequest(BaseModel):
    candidates: List[Dict[str, Any]]
    good_model: str
    worse_model: str


class ComparePlayerVsLLMRequest(BaseModel):
    candidates: List[Dict[str, Any]]
    player_id: str  # The player whose actions we're comparing (e.g., "player_0")
    llm_model: str  # The LLM model to compare against


class CompareLLMActionsResponse(BaseModel):
    disagreements: List[Dict[str, Any]]
    agreements: List[Dict[str, Any]]  # Cases where both LLMs chose the same action


class ComparePlayerVsLLMResponse(BaseModel):
    disagreements: List[Dict[str, Any]]  # Player action != LLM action
    agreements: List[Dict[str, Any]]  # Player action == LLM action


def _actions_equal(action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
    """Compare two actions including full payload."""
    if action1.get("type") != action2.get("type"):
        return False
    payload1 = action1.get("payload", {})
    payload2 = action2.get("payload", {})
    # Deep equality check - compare all keys and values
    if isinstance(payload1, dict) and isinstance(payload2, dict):
        # Normalize by removing None values for comparison
        p1_clean = {k: v for k, v in payload1.items() if v is not None}
        p2_clean = {k: v for k, v in payload2.items() if v is not None}
        return p1_clean == p2_clean
    return payload1 == payload2

def _compare_states_for_differences(state1_json: Dict[str, Any], state2_json: Dict[str, Any], player_id: str, context: str = "") -> Dict[str, Any]:
    """
    Compare two game states and return a dict with differences.
    Returns dict with keys: 'are_different', 'differences', 'warnings'
    """
    if not state1_json or not state2_json:
        return {
            "are_different": False,
            "differences": [],
            "warnings": ["One or both states are None"],
            "player_intersections1": [],
            "player_intersections2": []
        }
    
    differences = []
    warnings = []
    
    # Compare intersections (settlements/cities)
    intersections1 = {i["id"]: (i.get("owner"), i.get("building_type")) for i in state1_json.get("intersections", [])}
    intersections2 = {i["id"]: (i.get("owner"), i.get("building_type")) for i in state2_json.get("intersections", [])}
    
    # Find intersections that differ
    all_intersection_ids = set(intersections1.keys()) | set(intersections2.keys())
    for inter_id in all_intersection_ids:
        owner1, building1 = intersections1.get(inter_id, (None, None))
        owner2, building2 = intersections2.get(inter_id, (None, None))
        if (owner1, building1) != (owner2, building2):
            differences.append(f"Intersection {inter_id}: State1 has owner={owner1}, building={building1}; State2 has owner={owner2}, building={building2}")
    
    # Check player-specific intersections
    player_intersections1 = [i["id"] for i in state1_json.get("intersections", []) 
                             if i.get("owner") == player_id]
    player_intersections2 = [i["id"] for i in state2_json.get("intersections", []) 
                             if i.get("owner") == player_id]
    
    if set(player_intersections1) == set(player_intersections2):
        warnings.append(f"Player {player_id} has same intersections in both states: {sorted(player_intersections1)}")
    
    # Compare roads
    roads1 = {(r.get("intersection1_id"), r.get("intersection2_id"), r.get("owner")) 
              for r in state1_json.get("road_edges", [])}
    roads2 = {(r.get("intersection1_id"), r.get("intersection2_id"), r.get("owner")) 
              for r in state2_json.get("road_edges", [])}
    
    if roads1 != roads2:
        diff_roads = roads1.symmetric_difference(roads2)
        differences.append(f"Roads differ: {len(diff_roads)} road(s) different")
    
    # Compare player resources (might differ due to action costs)
    players1 = {p["id"]: p.get("resources", {}) for p in state1_json.get("players", [])}
    players2 = {p["id"]: p.get("resources", {}) for p in state2_json.get("players", [])}
    
    if players1.get(player_id, {}) != players2.get(player_id, {}):
        differences.append(f"Player {player_id} resources differ")
    
    # Compare victory points
    vp1 = next((p.get("victory_points", 0) for p in state1_json.get("players", []) if p["id"] == player_id), 0)
    vp2 = next((p.get("victory_points", 0) for p in state2_json.get("players", []) if p["id"] == player_id), 0)
    
    if vp1 != vp2:
        differences.append(f"Player {player_id} victory points differ: {vp1} vs {vp2}")
    
    are_different = len(differences) > 0
    
    if not are_different and context:
        warnings.append(f"WARNING: States appear identical in {context} - this may indicate a bug!")
    
    return {
        "are_different": are_different,
        "differences": differences,
        "warnings": warnings,
        "player_intersections1": sorted(player_intersections1),
        "player_intersections2": sorted(player_intersections2)
    }


@router.post("/games/{game_id}/compare_llm_actions", response_model=CompareLLMActionsResponse)
async def compare_llm_actions(
    game_id: str,
    request: CompareLLMActionsRequest
):
    """
    Run two LLMs on a set of game states and compare their actions.
    
    Returns: List of disagreements with both LLM actions
    """
    import asyncio
    import traceback
    
    try:
        # Check if game exists
        game_row = get_game_from_db(game_id)
        if not game_row:
            raise HTTPException(status_code=404, detail="Game not found")
        
        from agents.llm_agent import LLMAgent
        
        disagreements = []
        agreements = []
        errors = []
        
        print(f"[compare_llm_actions] Processing {len(request.candidates)} candidates", flush=True)
        print(f"[compare_llm_actions] Good model: {request.good_model}, Worse model: {request.worse_model}", flush=True)
        
        if not request.candidates:
            return CompareLLMActionsResponse(disagreements=[])
        
        # Process candidates in parallel (up to 16 at a time)
        max_concurrency = 16
        
        def _process_one_candidate(candidate_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process a single candidate and return result dict with 'type' field: 'disagreement', 'agreement', or None/Exception on error."""
            try:
                step_idx = candidate_data.get("step_idx")
                player_id = candidate_data.get("player_id")
                state_before_json = candidate_data.get("state_before_json")
                
                if not all([step_idx is not None, player_id, state_before_json]):
                    print(f"[compare_llm_actions] Skipping invalid candidate: missing required fields", flush=True)
                    return None
                
                print(f"[compare_llm_actions] Processing candidate: step_idx={step_idx}, player_id={player_id}", flush=True)
                
                # Get state_after_json from the step to show the result of the action
                # We need to get it from the database since candidates only have state_before_json
                steps = get_steps(game_id)
                step_row = next((s for s in steps if s["step_idx"] == step_idx), None)
                state_after_json = None
                if step_row and "state_after_json" in step_row.keys():
                    state_after_json_str = step_row["state_after_json"]
                    if state_after_json_str:
                        try:
                            state_after_json = json.loads(state_after_json_str) if isinstance(state_after_json_str, str) else state_after_json_str
                        except Exception:
                            pass  # Fall back to state_before_json if state_after_json is invalid
                
                # Use state_after_json if available, otherwise fall back to state_before_json
                # Ensure we always have a valid state (never None)
                display_state_json = state_after_json if state_after_json is not None else state_before_json
                if display_state_json is None:
                    print(f"[compare_llm_actions] WARNING: Both state_after_json and state_before_json are None for step_idx={step_idx}", flush=True)
                    display_state_json = state_before_json  # Final fallback
                
                # Deserialize state (use state_before for getting legal actions)
                try:
                    state = deserialize_game_state(state_before_json)
                except Exception as e:
                    print(f"[compare_llm_actions] Failed to deserialize state for step_idx={step_idx}: {e}", flush=True)
                    raise Exception(f"Step {step_idx}: Failed to deserialize state: {str(e)}")
                
                # Get legal actions
                legal_actions_list = legal_actions(state, player_id)
                
                if not legal_actions_list:
                    print(f"[compare_llm_actions] No legal actions for step_idx={step_idx}, skipping", flush=True)
                    return None  # Skip if no legal actions
                
                # Create two LLM agents
                # LiteLLM automatically picks up the correct API key from environment based on model name
                try:
                    good_agent = LLMAgent(
                        player_id=player_id,
                        api_key=None,  # Let LiteLLM use environment variables automatically
                        model=request.good_model,
                        enable_retrieval=False
                    )
                    worse_agent = LLMAgent(
                        player_id=player_id,
                        api_key=None,  # Let LiteLLM use environment variables automatically
                        model=request.worse_model,
                        enable_retrieval=False
                    )
                except Exception as e:
                    error_msg = f"Step {step_idx}: Failed to create LLM agents: {str(e)}"
                    print(f"[compare_llm_actions] {error_msg}", flush=True)
                    raise Exception(error_msg)
                
                # Get actions from both agents (synchronous calls since we're in a thread)
                try:
                    good_result = good_agent.choose_action(state, legal_actions_list)
                    worse_result = worse_agent.choose_action(state, legal_actions_list)
                except Exception as e:
                    print(f"[compare_llm_actions] Failed to get actions from agents for step_idx={step_idx}: {e}", flush=True)
                    raise Exception(f"Step {step_idx}: Failed to get LLM actions: {str(e)}")
                
                # Extract action and payload from results (choose_action returns 4-tuple)
                # Handle both 3-tuple and 4-tuple return formats
                if len(good_result) >= 2:
                    good_action, good_payload = good_result[0], good_result[1]
                else:
                    print(f"[compare_llm_actions] Unexpected good_result format for step_idx={step_idx}: {good_result}", flush=True)
                    raise Exception(f"Step {step_idx}: Unexpected result format from good agent")
                
                if len(worse_result) >= 2:
                    worse_action, worse_payload = worse_result[0], worse_result[1]
                else:
                    print(f"[compare_llm_actions] Unexpected worse_result format for step_idx={step_idx}: {worse_result}", flush=True)
                    raise Exception(f"Step {step_idx}: Unexpected result format from worse agent")
                
                # Serialize actions to dict format
                try:
                    good_action_dict = {
                        "type": serialize_action(good_action)
                    }
                    if good_payload:
                        good_action_dict["payload"] = serialize_action_payload(good_payload)
                    
                    worse_action_dict = {
                        "type": serialize_action(worse_action)
                    }
                    if worse_payload:
                        worse_action_dict["payload"] = serialize_action_payload(worse_payload)
                except Exception as e:
                    print(f"[compare_llm_actions] Failed to serialize actions for step_idx={step_idx}: {e}", flush=True)
                    raise Exception(f"Step {step_idx}: Failed to serialize actions: {str(e)}")
                
                # Serialize legal actions for frontend (needed for both disagreements and agreements)
                serialized_legal_actions = []
                for action, payload in legal_actions_list:
                    action_dict = {
                        "type": serialize_action(action)
                    }
                    if payload:
                        action_dict["payload"] = serialize_action_payload(payload)
                    serialized_legal_actions.append(action_dict)
                
                # Compute states after both LLM actions for visualization
                state_after_good_action_json = None
                state_after_worse_action_json = None
                try:
                    from engine import Action, deserialize_action, deserialize_action_payload
                    # IMPORTANT: Deserialize fresh states from state_before_json for each action
                    # (the 'state' variable might have been modified by llm_agent.choose_action calls)
                    
                    # Compute state after good LLM's action
                    good_action_enum = deserialize_action(good_action_dict["type"])
                    good_payload_obj = deserialize_action_payload(good_action_dict.get("payload", {})) if good_action_dict.get("payload") else None
                    fresh_state_for_good = deserialize_game_state(state_before_json)
                    state_after_good = fresh_state_for_good.step(good_action_enum, good_payload_obj, player_id=player_id)
                    state_after_good_action_json = serialize_game_state(state_after_good)
                    
                    # Compute state after worse LLM's action (start from state_before again)
                    worse_action_enum = deserialize_action(worse_action_dict["type"])
                    worse_payload_obj = deserialize_action_payload(worse_action_dict.get("payload", {})) if worse_action_dict.get("payload") else None
                    fresh_state_for_worse = deserialize_game_state(state_before_json)
                    state_after_worse = fresh_state_for_worse.step(worse_action_enum, worse_payload_obj, player_id=player_id)
                    state_after_worse_action_json = serialize_game_state(state_after_worse)
                    
                    # VALIDATION: Verify states are actually different
                    comparison = _compare_states_for_differences(
                        state_after_good_action_json,
                        state_after_worse_action_json,
                        player_id,
                        f"good LLM action vs worse LLM action at step_idx={step_idx}"
                    )
                    if comparison["are_different"]:
                        print(f"[compare_llm_actions] ✓ States are different at step_idx={step_idx}: {len(comparison['differences'])} difference(s)", flush=True)
                        for diff in comparison["differences"][:3]:  # Log first 3 differences
                            print(f"  - {diff}", flush=True)
                    else:
                        print(f"[compare_llm_actions] ⚠️ WARNING: States are IDENTICAL at step_idx={step_idx}!", flush=True)
                        print(f"  Player intersections in state_after_good_action_json: {comparison['player_intersections1']}", flush=True)
                        print(f"  Player intersections in state_after_worse_action_json: {comparison['player_intersections2']}", flush=True)
                        for warning in comparison["warnings"]:
                            print(f"  ⚠️ {warning}", flush=True)
                    
                    good_intersection_id = good_payload_obj.intersection_id if hasattr(good_payload_obj, 'intersection_id') else None
                    worse_intersection_id = worse_payload_obj.intersection_id if hasattr(worse_payload_obj, 'intersection_id') else None
                    print(f"[compare_llm_actions] Good LLM intersection: {good_intersection_id}, Worse LLM intersection: {worse_intersection_id}", flush=True)
                except Exception as e:
                    import traceback
                    print(f"[compare_llm_actions] Failed to compute states after LLM actions for step_idx={step_idx}: {e}", flush=True)
                    print(f"[compare_llm_actions] Traceback: {traceback.format_exc()}", flush=True)
                    # DO NOT fall back to display_state_json - that would be wrong!
                    # If computation fails, we can't show the LLM states, so set to None
                    # The frontend should handle this gracefully
                    state_after_good_action_json = None
                    state_after_worse_action_json = None
                    print(f"[compare_llm_actions] WARNING: state_after_good_action_json and state_after_worse_action_json are None due to computation failure", flush=True)
                
                # Compare actions
                if not _actions_equal(good_action_dict, worse_action_dict):
                    # Actions differ - this is a disagreement
                    print(f"[compare_llm_actions] Found disagreement at step_idx={step_idx}", flush=True)
                    return {
                        "type": "disagreement",
                        "step_idx": step_idx,
                        "player_id": player_id,
                        "state_before_json": state_before_json,
                        "state_after_json": display_state_json,  # Keep for backward compatibility
                        "state_after_good_action_json": state_after_good_action_json,  # State after good LLM's action
                        "state_after_worse_action_json": state_after_worse_action_json,  # State after worse LLM's action
                        "good_action": good_action_dict,
                        "worse_action": worse_action_dict,
                        "legal_actions": serialized_legal_actions
                    }
                else:
                    # Actions agree - this is an agreement
                    print(f"[compare_llm_actions] Found agreement at step_idx={step_idx}", flush=True)
                    return {
                        "type": "agreement",
                        "step_idx": step_idx,
                        "player_id": player_id,
                        "state_before_json": state_before_json,
                        "state_after_json": display_state_json,  # State after action (same for both when they agree)
                        "state_after_good_action_json": state_after_good_action_json,  # Same as state_after_json when actions agree
                        "state_after_worse_action_json": state_after_worse_action_json,  # Same as state_after_json when actions agree
                        "agreed_action": good_action_dict,  # Both LLMs chose the same action
                        "legal_actions": serialized_legal_actions
                    }
            except Exception as e:
                # Return exception to be handled by caller
                return e
        
        # Use semaphore to limit concurrency
        sem = asyncio.Semaphore(max_concurrency)
        
        async def _run_one(candidate_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Async wrapper that uses semaphore to limit concurrency."""
            async with sem:
                return await asyncio.to_thread(_process_one_candidate, candidate_data)
        
        # Run all candidates in parallel
        print(f"[compare_llm_actions] Processing {len(request.candidates)} candidates in parallel (max {max_concurrency} concurrent)", flush=True)
        results = await asyncio.gather(*[_run_one(c) for c in request.candidates], return_exceptions=True)
        
        # Process results
        for candidate, result in zip(request.candidates, results):
            if isinstance(result, Exception):
                error_msg = f"Error processing candidate at step_idx={candidate.get('step_idx', 'unknown')}: {str(result)}"
                print(f"[compare_llm_actions] {error_msg}", flush=True)
                errors.append(error_msg)
            elif result is not None:
                # Result is a dict with 'type' field
                if result.get("type") == "disagreement":
                    disagreements.append(result)
                elif result.get("type") == "agreement":
                    agreements.append(result)
        
        print(f"[compare_llm_actions] Completed: {len(disagreements)} disagreements, {len(agreements)} agreements, {len(errors)} errors", flush=True)
        
        # If there were errors but we still got some results, log them but continue
        if len(errors) > 0:
            print(f"[compare_llm_actions] Errors encountered: {errors}", flush=True)
        
        # If all candidates failed, raise an error with details
        if len(disagreements) == 0 and len(errors) > 0:
            error_summary = "; ".join(errors[:3])  # Show first 3 errors
            if len(errors) > 3:
                error_summary += f" ... and {len(errors) - 3} more errors"
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process all candidates. First errors: {error_summary}"
            )
        
        # Return both disagreements and agreements
        return CompareLLMActionsResponse(disagreements=disagreements, agreements=agreements)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any other exceptions and return a proper error
        error_msg = f"Unexpected error in compare_llm_actions: {str(e)}"
        print(f"[compare_llm_actions] {error_msg}", flush=True)
        print(f"[compare_llm_actions] Traceback: {traceback.format_exc()}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@router.post("/games/{game_id}/compare_player_vs_llm", response_model=ComparePlayerVsLLMResponse)
async def compare_player_vs_llm(
    game_id: str,
    request: ComparePlayerVsLLMRequest
):
    """
    Compare a player's actual actions from game history against what an LLM would choose.
    
    Returns: List of disagreements and agreements
    """
    import asyncio
    import traceback
    import json
    
    try:
        # Check if game exists
        game_row = get_game_from_db(game_id)
        if not game_row:
            raise HTTPException(status_code=404, detail="Game not found")
        
        from agents.llm_agent import LLMAgent
        
        disagreements = []
        agreements = []
        errors = []
        
        print(f"[compare_player_vs_llm] Processing {len(request.candidates)} candidates", flush=True)
        print(f"[compare_player_vs_llm] Player: {request.player_id}, LLM model: {request.llm_model}", flush=True)
        
        if not request.candidates:
            return ComparePlayerVsLLMResponse(disagreements=[], agreements=[])
        
        # Get all steps to retrieve actual player actions
        steps = get_steps(game_id)
        step_dict = {step["step_idx"]: step for step in steps}
        
        def _process_one_candidate(candidate_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process a single candidate and return result dict with 'type' field: 'disagreement', 'agreement', or None/Exception on error."""
            try:
                step_idx = candidate_data.get("step_idx")
                player_id = candidate_data.get("player_id")
                state_before_json = candidate_data.get("state_before_json")
                
                if not all([step_idx is not None, player_id, state_before_json]):
                    print(f"[compare_player_vs_llm] Skipping invalid candidate: missing required fields", flush=True)
                    return None
                
                # Verify player_id matches
                if player_id != request.player_id:
                    return None
                
                print(f"[compare_player_vs_llm] Processing candidate: step_idx={step_idx}, player_id={player_id}", flush=True)
                
                # Get the actual action the player took
                step_row = step_dict.get(step_idx)
                if not step_row:
                    print(f"[compare_player_vs_llm] Step {step_idx} not found in game history", flush=True)
                    return None
                
                action_json_str = step_row["action_json"] if "action_json" in step_row.keys() else None
                if not action_json_str:
                    print(f"[compare_player_vs_llm] No action_json for step_idx={step_idx}", flush=True)
                    return None
                
                player_action_dict = json.loads(action_json_str) if isinstance(action_json_str, str) else action_json_str
                
                # Get state_after_json from the step to show the result of the action
                state_after_json = None
                if "state_after_json" in step_row.keys():
                    state_after_json_str = step_row["state_after_json"]
                    if state_after_json_str:
                        try:
                            state_after_json = json.loads(state_after_json_str) if isinstance(state_after_json_str, str) else state_after_json_str
                        except Exception:
                            pass  # Fall back to state_before_json if state_after_json is invalid
                
                # Use state_after_json if available, otherwise fall back to state_before_json
                # Ensure we always have a valid state (never None)
                display_state_json = state_after_json if state_after_json is not None else state_before_json
                if display_state_json is None:
                    print(f"[compare_player_vs_llm] WARNING: Both state_after_json and state_before_json are None for step_idx={step_idx}", flush=True)
                    display_state_json = state_before_json  # Final fallback
                
                # Deserialize state (use state_before for getting legal actions)
                try:
                    state = deserialize_game_state(state_before_json)
                except Exception as e:
                    print(f"[compare_player_vs_llm] Failed to deserialize state for step_idx={step_idx}: {e}", flush=True)
                    raise Exception(f"Step {step_idx}: Failed to deserialize state: {str(e)}")
                
                # Get legal actions
                legal_actions_list = legal_actions(state, player_id)
                
                if not legal_actions_list:
                    print(f"[compare_player_vs_llm] No legal actions for step_idx={step_idx}, skipping", flush=True)
                    return None  # Skip if no legal actions
                
                # Create LLM agent
                try:
                    llm_agent = LLMAgent(
                        player_id=player_id,
                        api_key=None,  # Let LiteLLM use environment variables automatically
                        model=request.llm_model,
                        enable_retrieval=False
                    )
                except Exception as e:
                    error_msg = f"Step {step_idx}: Failed to create LLM agent: {str(e)}"
                    print(f"[compare_player_vs_llm] {error_msg}", flush=True)
                    raise Exception(error_msg)
                
                # Get action from LLM
                try:
                    llm_result = llm_agent.choose_action(state, legal_actions_list)
                except Exception as e:
                    print(f"[compare_player_vs_llm] Failed to get action from LLM for step_idx={step_idx}: {e}", flush=True)
                    raise Exception(f"Step {step_idx}: Failed to get LLM action: {str(e)}")
                
                # Extract action and payload from LLM result
                if len(llm_result) >= 2:
                    llm_action, llm_payload = llm_result[0], llm_result[1]
                else:
                    print(f"[compare_player_vs_llm] Unexpected llm_result format for step_idx={step_idx}: {llm_result}", flush=True)
                    raise Exception(f"Step {step_idx}: Unexpected result format from LLM")
                
                # Serialize LLM action to dict format
                try:
                    llm_action_dict = {
                        "type": serialize_action(llm_action)
                    }
                    if llm_payload:
                        llm_action_dict["payload"] = serialize_action_payload(llm_payload)
                except Exception as e:
                    print(f"[compare_player_vs_llm] Failed to serialize LLM action for step_idx={step_idx}: {e}", flush=True)
                    raise Exception(f"Step {step_idx}: Failed to serialize LLM action: {str(e)}")
                
                # Serialize legal actions for frontend
                serialized_legal_actions = []
                for action, payload in legal_actions_list:
                    action_dict = {
                        "type": serialize_action(action)
                    }
                    if payload:
                        action_dict["payload"] = serialize_action_payload(payload)
                    serialized_legal_actions.append(action_dict)
                
                # Compute state after LLM's action for visualization
                state_after_llm_action_json = None
                try:
                    # Deserialize action and payload from dict format
                    from engine import Action, deserialize_action, deserialize_action_payload
                    llm_action_enum = deserialize_action(llm_action_dict["type"])
                    llm_payload_obj = deserialize_action_payload(llm_action_dict.get("payload", {})) if llm_action_dict.get("payload") else None
                    # IMPORTANT: Deserialize a fresh state from state_before_json to ensure we're starting from the original state
                    # (the 'state' variable might have been modified by llm_agent.choose_action)
                    fresh_state = deserialize_game_state(state_before_json)
                    # Apply LLM's action to fresh state to get state after LLM action
                    state_after_llm = fresh_state.step(llm_action_enum, llm_payload_obj, player_id=player_id)
                    state_after_llm_action_json = serialize_game_state(state_after_llm)
                    
                    # VALIDATION: Verify states are actually different
                    comparison = _compare_states_for_differences(
                        display_state_json, 
                        state_after_llm_action_json, 
                        player_id,
                        f"player action vs LLM action at step_idx={step_idx}"
                    )
                    if comparison["are_different"]:
                        print(f"[compare_player_vs_llm] ✓ States are different at step_idx={step_idx}: {len(comparison['differences'])} difference(s)", flush=True)
                        for diff in comparison["differences"][:3]:  # Log first 3 differences
                            print(f"  - {diff}", flush=True)
                    else:
                        print(f"[compare_player_vs_llm] ⚠️ WARNING: States are IDENTICAL at step_idx={step_idx}!", flush=True)
                        print(f"  Player intersections in state_after_json: {comparison['player_intersections1']}", flush=True)
                        print(f"  Player intersections in state_after_llm_action_json: {comparison['player_intersections2']}", flush=True)
                        for warning in comparison["warnings"]:
                            print(f"  ⚠️ {warning}", flush=True)
                    
                    llm_intersection_id = llm_payload_obj.intersection_id if hasattr(llm_payload_obj, 'intersection_id') else None
                    player_intersection_id = None
                    if player_action_dict.get("type") == "setup_place_settlement" and player_action_dict.get("payload"):
                        player_intersection_id = player_action_dict["payload"].get("intersection_id")
                    print(f"[compare_player_vs_llm] Player action intersection: {player_intersection_id}, LLM action intersection: {llm_intersection_id}", flush=True)
                except Exception as e:
                    import traceback
                    print(f"[compare_player_vs_llm] Failed to compute state after LLM action for step_idx={step_idx}: {e}", flush=True)
                    print(f"[compare_player_vs_llm] Traceback: {traceback.format_exc()}", flush=True)
                    # DO NOT fall back to display_state_json (player's state) - that would be wrong!
                    # If computation fails, we can't show the LLM's state, so set to None
                    # The frontend should handle this gracefully
                    state_after_llm_action_json = None
                    print(f"[compare_player_vs_llm] WARNING: state_after_llm_action_json is None due to computation failure", flush=True)
                
                # Compare actions
                if not _actions_equal(player_action_dict, llm_action_dict):
                    # Actions differ - this is a disagreement
                    print(f"[compare_player_vs_llm] Found disagreement at step_idx={step_idx}", flush=True)
                    return {
                        "type": "disagreement",
                        "step_idx": step_idx,
                        "player_id": player_id,
                        "state_before_json": state_before_json,
                        "state_after_json": display_state_json,  # State after player's action
                        "state_after_llm_action_json": state_after_llm_action_json,  # State after LLM's action
                        "player_action": player_action_dict,
                        "llm_action": llm_action_dict,
                        "legal_actions": serialized_legal_actions
                    }
                else:
                    # Actions agree
                    print(f"[compare_player_vs_llm] Found agreement at step_idx={step_idx}", flush=True)
                    return {
                        "type": "agreement",
                        "step_idx": step_idx,
                        "player_id": player_id,
                        "state_before_json": state_before_json,
                        "state_after_json": display_state_json,  # State after action (same for both)
                        "state_after_llm_action_json": state_after_llm_action_json,  # Same as state_after_json when actions agree
                        "agreed_action": player_action_dict,  # Player and LLM chose the same action
                        "legal_actions": serialized_legal_actions
                    }
            except Exception as e:
                # Return exception to be handled by caller
                return e
        
        # Use semaphore to limit concurrency
        max_concurrency = 16
        sem = asyncio.Semaphore(max_concurrency)
        
        async def _run_one(candidate_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Async wrapper that uses semaphore to limit concurrency."""
            async with sem:
                return await asyncio.to_thread(_process_one_candidate, candidate_data)
        
        # Run all candidates in parallel
        print(f"[compare_player_vs_llm] Processing {len(request.candidates)} candidates in parallel (max {max_concurrency} concurrent)", flush=True)
        results = await asyncio.gather(*[_run_one(c) for c in request.candidates], return_exceptions=True)
        
        # Process results
        for candidate, result in zip(request.candidates, results):
            if isinstance(result, Exception):
                error_msg = f"Error processing candidate at step_idx={candidate.get('step_idx', 'unknown')}: {str(result)}"
                print(f"[compare_player_vs_llm] {error_msg}", flush=True)
                errors.append(error_msg)
            elif result is not None:
                # Result is a dict with 'type' field
                if result.get("type") == "disagreement":
                    disagreements.append(result)
                elif result.get("type") == "agreement":
                    agreements.append(result)
        
        print(f"[compare_player_vs_llm] Completed: {len(disagreements)} disagreements, {len(agreements)} agreements, {len(errors)} errors", flush=True)
        
        # If there were errors but we still got some results, log them but continue
        if len(errors) > 0:
            print(f"[compare_player_vs_llm] Errors encountered: {errors}", flush=True)
        
        # Return both disagreements and agreements
        return ComparePlayerVsLLMResponse(disagreements=disagreements, agreements=agreements)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any other exceptions and return a proper error
        error_msg = f"Unexpected error in compare_player_vs_llm: {str(e)}"
        print(f"[compare_player_vs_llm] {error_msg}", flush=True)
        print(f"[compare_player_vs_llm] Traceback: {traceback.format_exc()}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


def _parse_llm_agent_spec(agent_type_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse frontend agent_type strings for LLM variants.

    Supported:
      - "llm" (legacy; use env vars)
      - "llm:<model>" (explicit model; thinking disabled)
      - "llm:<model>:thinking:<effort>" (explicit model + thinking effort)
      - "llm:<model>:thinking-<effort>" (alternate shorthand)

    Returns None if not an LLM spec.
    """
    if agent_type_str == "llm":
        return {"model": None, "thinking_mode": None, "thinking_effort": None}
    if not agent_type_str.startswith("llm:"):
        return None

    parts = agent_type_str.split(":")
    model = parts[1].strip() if len(parts) > 1 else None
    if not model:
        return {"model": None, "thinking_mode": False, "thinking_effort": "medium"}

    thinking_mode = False
    thinking_effort: Optional[str] = "medium"

    if len(parts) >= 3:
        p2 = parts[2].strip().lower()
        if p2 in ("thinking", "think", "reasoning"):
            thinking_mode = True
            if len(parts) >= 4 and parts[3].strip():
                thinking_effort = parts[3].strip().lower()
        elif p2.startswith("thinking-") or p2.startswith("think-") or p2.startswith("reasoning-"):
            thinking_mode = True
            _, _, effort = p2.partition("-")
            thinking_effort = (effort or "medium").strip().lower()

    return {"model": model, "thinking_mode": thinking_mode, "thinking_effort": thinking_effort}


def _make_agent(
    agent_type: str, 
    player_id: str, 
    *, 
    drill_guideline_text: Optional[str] = None,
    exclude_strategic_advice: bool = False,
    exclude_higher_level_features: bool = False
):
    # Mirror watch_agents_step behavior to keep a single mental model in the UI.
    from agents import RandomAgent, BehaviorTreeAgent
    try:
        from agents.llm_agent import LLMAgent
        from agents.variants import (
            BalancedAgent,
            AggressiveBuilderAgent,
            DevCardFocusedAgent,
            ExpansionAgent,
            DefensiveAgent,
            StateConditionedAgent,
            ImitationBehaviorTreeAgent,
        )

        AGENT_CLASSES = {
            "random": RandomAgent,
            "behavior_tree": BehaviorTreeAgent,
            "balanced": BalancedAgent,
            "aggressive_builder": AggressiveBuilderAgent,
            "dev_card_focused": DevCardFocusedAgent,
            "expansion": ExpansionAgent,
            "defensive": DefensiveAgent,
            "state_conditioned": StateConditionedAgent,
            "llm": LLMAgent,
            "imitation_bt": ImitationBehaviorTreeAgent,
        }
    except Exception:
        AGENT_CLASSES = {
            "random": RandomAgent,
            "behavior_tree": BehaviorTreeAgent,
        }
        try:
            from agents.llm_agent import LLMAgent
            AGENT_CLASSES["llm"] = LLMAgent
        except Exception:
            pass

    llm_spec = _parse_llm_agent_spec(agent_type)
    if llm_spec is not None:
        agent_class = AGENT_CLASSES.get("llm", RandomAgent)
        import os
        model = llm_spec["model"] or os.getenv("LLM_MODEL", "gpt-5.1")
        # LiteLLM automatically picks up the correct API key from environment based on model name
        # No need to pass api_key - just ensure environment variables are set (handled by main.py)
        return agent_class(
            player_id,
            api_key=None,  # Let LiteLLM use environment variables automatically
            model=model,
            enable_retrieval=False,
            thinking_mode=llm_spec["thinking_mode"],
            thinking_effort=llm_spec["thinking_effort"],
            drill_guideline_text=drill_guideline_text,
            exclude_strategic_advice=exclude_strategic_advice,
            exclude_higher_level_features=exclude_higher_level_features,
        )
    agent_class = AGENT_CLASSES.get(agent_type, RandomAgent)
    if agent_type == "imitation_bt":
        # Requires env var for now; this is primarily for offline evaluation.
        import os
        ref_game_id = os.getenv("IMITATION_GAME_ID")
        ref_player_id = os.getenv("IMITATION_PLAYER_ID", "player_0")
        if not ref_game_id:
            return RandomAgent(player_id)
        return agent_class(player_id, reference_game_id=ref_game_id, reference_player_id=ref_player_id)
    return agent_class(player_id)


def _canonical_action_dict(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize action dicts for comparison."""
    if not isinstance(action_dict, dict):
        return {"type": None, "payload": None}
    action_type = action_dict.get("type")
    payload = action_dict.get("payload", None)
    if payload is None:
        return {"type": action_type}
    if isinstance(payload, dict):
        # Remove nulls to avoid trivial mismatches between absent vs null
        cleaned = {k: v for k, v in payload.items() if v is not None}
        return {"type": action_type, "payload": cleaned}
    return {"type": action_type, "payload": payload}


def _action_dict_matches_legal_action(
    action_dict: Dict[str, Any],
    legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
) -> bool:
    """Check if an action dict matches any legal action."""
    canonical_action = _canonical_action_dict(action_dict)
    action_type = action_dict.get("type")
    
    for legal_action, legal_payload in legal_actions_list:
        legal_action_dict = {"type": serialize_action(legal_action)}
        if legal_payload is not None:
            legal_action_dict["payload"] = serialize_action_payload(legal_payload)
        canonical_legal = _canonical_action_dict(legal_action_dict)
        
        # Special case: PROPOSE_TRADE actions are legal even if they have a payload
        # because legal_actions returns (PROPOSE_TRADE, None) but agents construct the payload
        if action_type == "propose_trade" and legal_action == Action.PROPOSE_TRADE:
            # If PROPOSE_TRADE is legal, any propose_trade action with a valid payload is acceptable
            if canonical_action.get("type") == "propose_trade" and canonical_action.get("payload"):
                # Validate that the payload has required fields
                payload = canonical_action.get("payload", {})
                if isinstance(payload, dict):
                    if "target_player_ids" in payload and "give_resources" in payload and "receive_resources" in payload:
                        return True
        
        if canonical_action == canonical_legal:
            return True
    return False


def _filter_legal_actions(
    legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]],
    action_dicts: List[Dict[str, Any]]
) -> List[Tuple[Action, Optional[ActionPayload]]]:
    """Filter legal actions to only include those matching the given action dicts."""
    filtered = []
    for legal_action, legal_payload in legal_actions_list:
        legal_action_dict = {"type": serialize_action(legal_action)}
        if legal_payload is not None:
            legal_action_dict["payload"] = serialize_action_payload(legal_payload)
        canonical_legal = _canonical_action_dict(legal_action_dict)
        for action_dict in action_dicts:
            if _canonical_action_dict(action_dict) == canonical_legal:
                filtered.append((legal_action, legal_payload))
                break
    return filtered


class EvaluateDrillRequest(BaseModel):
    agent_type: str = "random"
    include_guidelines: bool = False
    exclude_strategic_advice: bool = False  # Exclude strategic advice from LLM prompt
    exclude_higher_level_features: bool = False  # Exclude computed features like production analysis


@router.post("/drills/{drill_id}/evaluate")
async def evaluate_drill(drill_id: int, request: EvaluateDrillRequest):
    drill_row = get_drill_from_db(drill_id)
    if not drill_row:
        raise HTTPException(status_code=404, detail="Drill not found")
    steps = get_drill_steps_from_db(drill_id)
    if not steps:
        raise HTTPException(status_code=400, detail="Drill has no steps")

    results = []
    passed_all = True
    drill_guideline_text = drill_row["guideline_text"] if "guideline_text" in drill_row.keys() else None
    for r in steps:
        player_id = r["player_id"]
        state_json = json.loads(r["state_json"])
        expected_action = json.loads(r["expected_action_json"])
        state = deserialize_game_state(state_json)
        la_list = legal_actions(state, player_id)
        
        # Check if drill step uses enhanced format (correct/incorrect actions)
        correct_actions = None
        incorrect_actions = None
        if "correct_actions_json" in r.keys() and r["correct_actions_json"] is not None:
            correct_actions = json.loads(r["correct_actions_json"])
        if "incorrect_actions_json" in r.keys() and r["incorrect_actions_json"] is not None:
            incorrect_actions = json.loads(r["incorrect_actions_json"])
        
        # Filter legal actions if correct/incorrect actions are specified
        if correct_actions is not None:
            # If incorrect_actions is not specified or empty, automatically set it to all other legal actions
            if not incorrect_actions:
                # Convert all legal actions to action dicts
                all_legal_action_dicts = []
                for legal_action, legal_payload in la_list:
                    action_dict = {"type": serialize_action(legal_action)}
                    if legal_payload is not None:
                        action_dict["payload"] = serialize_action_payload(legal_payload)
                    all_legal_action_dicts.append(action_dict)
                
                    # Filter out correct actions to get incorrect actions
                    # Convert canonical dicts to tuples for hashing
                    def dict_to_hashable(obj):
                        """Recursively convert dict to hashable tuple."""
                        if isinstance(obj, dict):
                            return tuple(sorted((k, dict_to_hashable(v)) for k, v in obj.items()))
                        elif isinstance(obj, list):
                            return tuple(dict_to_hashable(item) for item in obj)
                        else:
                            return obj
                    
                    def canonical_to_tuple(ca):
                        canonical = _canonical_action_dict(ca)
                        payload = canonical.get("payload")
                        if payload:
                            payload_tuple = dict_to_hashable(payload)
                        else:
                            payload_tuple = None
                        return (canonical.get("type"), payload_tuple)
                    
                    correct_action_set = {canonical_to_tuple(ca) for ca in correct_actions}
                    incorrect_actions = [
                        action_dict for action_dict in all_legal_action_dicts
                        if canonical_to_tuple(action_dict) not in correct_action_set
                    ]
            
            # Filter to only include correct + incorrect actions
            # This restricts the action space so the LLM can only choose between
            # the specified correct and incorrect actions, not all legal actions.
            action_dicts_to_include = correct_actions.copy()
            if incorrect_actions:
                action_dicts_to_include.extend(incorrect_actions)
            la_list = _filter_legal_actions(la_list, action_dicts_to_include)
            
            if not la_list:
                passed_all = False
                results.append(
                    {
                        "idx": r["idx"],
                        "player_id": player_id,
                        "match": False,
                        "error": "No legal actions match the specified correct/incorrect actions",
                        "expected_action": expected_action,
                        "actual_action": None,
                    }
                )
                continue

        agent = _make_agent(
            request.agent_type,
            player_id,
            drill_guideline_text=(drill_guideline_text if request.include_guidelines else None),
            exclude_strategic_advice=request.exclude_strategic_advice,
            exclude_higher_level_features=request.exclude_higher_level_features,
        )
        try:
            choice = agent.choose_action(state, la_list)
            if isinstance(choice, tuple) and len(choice) == 4:
                action, payload, reasoning, raw_llm_response = choice
            elif isinstance(choice, tuple) and len(choice) == 3:
                action, payload, reasoning = choice
                raw_llm_response = None
            else:
                action, payload = choice
                reasoning = None
                raw_llm_response = None
        except Exception as e:
            passed_all = False
            results.append(
                {
                    "idx": r["idx"],
                    "player_id": player_id,
                    "match": False,
                    "error": str(e),
                    "expected_action": expected_action,
                    "actual_action": None,
                }
            )
            continue

        actual_action_dict: Dict[str, Any] = {"type": serialize_action(action)}
        if payload is not None:
            actual_action_dict["payload"] = serialize_action_payload(payload)
        if reasoning:
            actual_action_dict["reasoning"] = reasoning
        if raw_llm_response is not None:
            actual_action_dict["raw_llm_response"] = raw_llm_response

        # Check match: if using enhanced format, check against correct_actions set
        if correct_actions is not None:
            # Check if actual action matches any correct action
            match = False
            for correct_action in correct_actions:
                if _canonical_action_dict(actual_action_dict) == _canonical_action_dict(correct_action):
                    match = True
                    break
        else:
            # Backward compatibility: check against single expected_action
            match = _canonical_action_dict(actual_action_dict) == _canonical_action_dict(expected_action)
        
        if not match:
            passed_all = False

        results.append(
            {
                "idx": r["idx"],
                "player_id": player_id,
                "match": match,
                "expected_action": expected_action,
                "actual_action": actual_action_dict,
                "correct_actions": correct_actions,
                "incorrect_actions": incorrect_actions,
            }
        )

    return {
        "drill_id": drill_id,
        "agent_type": request.agent_type,
        "passed": passed_all,
        "results": results,
    }


class EvaluateAllDrillsRequest(BaseModel):
    agent_type: str = "random"
    limit: int = 200
    include_step_results: bool = False
    max_concurrency: int = 4
    include_guidelines: bool = False
    drill_ids: Optional[List[int]] = None
    exclude_strategic_advice: bool = False  # Exclude strategic advice from LLM prompt
    exclude_higher_level_features: bool = False  # Exclude computed features like production analysis


@router.post("/drills/evaluate_all")
async def evaluate_all_drills(request: EvaluateAllDrillsRequest):
    if request.max_concurrency < 1 or request.max_concurrency > 32:
        raise HTTPException(status_code=400, detail="max_concurrency must be between 1 and 32")

    drill_rows = []
    if request.drill_ids is not None:
        # Evaluate only specified drills (preserve provided order)
        for did in request.drill_ids:
            try:
                drill_row = get_drill_from_db(int(did))
            except Exception:
                drill_row = None
            if drill_row is None:
                # Represent missing drill as a pseudo-row; handled below
                drill_rows.append(
                    {
                        "id": int(did),
                        "name": None,
                        "guideline_text": None,
                        "source_game_id": None,
                        "source_step_idx": None,
                        "player_id": None,
                        "num_steps": 0,
                        "_missing": True,
                    }
                )
            else:
                # Enrich with num_steps (for response parity with list_drills)
                steps_for_count = get_drill_steps_from_db(int(did))
                drill_rows.append(
                    {
                        **{k: drill_row[k] for k in drill_row.keys()},
                        "num_steps": len(steps_for_count),
                        "_missing": False,
                    }
                )
    else:
        drill_rows = list_drills_from_db(limit=request.limit)
    evaluated_at = datetime.utcnow().isoformat()
    run_id = str(uuid.uuid4())
    # Preload all drill steps in the main thread (read-only), then evaluate drills concurrently.
    prepared: List[Dict[str, Any]] = []
    for d in drill_rows:
        drill_id = int(d["id"])
        step_rows = get_drill_steps_from_db(drill_id) if hasattr(d, "__getitem__") else []
        steps_prepped = []
        for r in step_rows:
            steps_prepped.append(
                {
                    "idx": int(r["idx"]),
                    "player_id": r["player_id"],
                    "state_json": json.loads(r["state_json"]),
                    "expected_action": json.loads(r["expected_action_json"]),
                    "correct_actions": json.loads(r["correct_actions_json"]) if "correct_actions_json" in r.keys() and r["correct_actions_json"] is not None else None,
                    "incorrect_actions": json.loads(r["incorrect_actions_json"]) if "incorrect_actions_json" in r.keys() and r["incorrect_actions_json"] is not None else None,
                }
            )
        prepared.append(
            {
                "drill_id": drill_id,
                "name": d["name"],
                "guideline_text": d["guideline_text"] if "guideline_text" in d.keys() else None,
                "source_game_id": d["source_game_id"],
                "source_step_idx": d["source_step_idx"],
                "player_id": d["player_id"],
                "num_steps": d["num_steps"],
                "steps": steps_prepped,
                "missing": bool(d.get("_missing", False)) if isinstance(d, dict) else False,
            }
        )

    def _eval_one_drill(prep: Dict[str, Any]) -> Dict[str, Any]:
        if prep.get("missing"):
            return {
                "drill_id": prep["drill_id"],
                "name": prep["name"],
                "source_game_id": prep["source_game_id"],
                "source_step_idx": prep["source_step_idx"],
                "player_id": prep["player_id"],
                "num_steps": prep["num_steps"],
                "passed": False,
                "first_mismatch": {"idx": 0, "error": "Drill not found"},
                **({"step_results": []} if request.include_step_results else {}),
            }
        steps = prep["steps"]
        drill_guideline_text = prep.get("guideline_text")
        if not request.include_guidelines:
            drill_guideline_text = None
        if not steps:
            return {
                "drill_id": prep["drill_id"],
                "name": prep["name"],
                "source_game_id": prep["source_game_id"],
                "source_step_idx": prep["source_step_idx"],
                "player_id": prep["player_id"],
                "num_steps": prep["num_steps"],
                "passed": False,
                "first_mismatch": {"idx": 0, "error": "No steps"},
                **({"step_results": []} if request.include_step_results else {}),
            }

        passed = True
        first_mismatch = None
        step_results = [] if request.include_step_results else None

        for s in steps:
            player_id = s["player_id"]
            state_json = s["state_json"]
            expected_action = s["expected_action"]
            correct_actions = s.get("correct_actions")
            incorrect_actions = s.get("incorrect_actions")
            # Handle case where correct_actions/incorrect_actions come from JSON strings
            if isinstance(correct_actions, str):
                correct_actions = json.loads(correct_actions) if correct_actions else None
            if isinstance(incorrect_actions, str):
                incorrect_actions = json.loads(incorrect_actions) if incorrect_actions else None
            state = deserialize_game_state(state_json)
            la_list = legal_actions(state, player_id)
            
            # Filter legal actions if correct/incorrect actions are specified
            if correct_actions is not None:
                # If incorrect_actions is not specified or empty, automatically set it to all other legal actions
                if not incorrect_actions:
                    # Convert all legal actions to action dicts
                    all_legal_action_dicts = []
                    for legal_action, legal_payload in la_list:
                        action_dict = {"type": serialize_action(legal_action)}
                        if legal_payload is not None:
                            action_dict["payload"] = serialize_action_payload(legal_payload)
                        all_legal_action_dicts.append(action_dict)
                    
                    # Filter out correct actions to get incorrect actions
                    # Convert canonical dicts to tuples for hashing
                    def dict_to_hashable(obj):
                        """Recursively convert dict to hashable tuple."""
                        if isinstance(obj, dict):
                            return tuple(sorted((k, dict_to_hashable(v)) for k, v in obj.items()))
                        elif isinstance(obj, list):
                            return tuple(dict_to_hashable(item) for item in obj)
                        else:
                            return obj
                    
                    def canonical_to_tuple(ca):
                        canonical = _canonical_action_dict(ca)
                        payload = canonical.get("payload")
                        if payload:
                            payload_tuple = dict_to_hashable(payload)
                        else:
                            payload_tuple = None
                        return (canonical.get("type"), payload_tuple)
                    
                    correct_action_set = {canonical_to_tuple(ca) for ca in correct_actions}
                    incorrect_actions = [
                        action_dict for action_dict in all_legal_action_dicts
                        if canonical_to_tuple(action_dict) not in correct_action_set
                    ]
                
                action_dicts_to_include = correct_actions.copy()
                if incorrect_actions:
                    action_dicts_to_include.extend(incorrect_actions)
                la_list = _filter_legal_actions(la_list, action_dicts_to_include)
                
                if not la_list:
                    passed = False
                    first_mismatch = {
                        "idx": s["idx"],
                        "error": "No legal actions match the specified correct/incorrect actions"
                    }
                    if request.include_step_results:
                        step_results.append(
                            {
                                "idx": s["idx"],
                                "player_id": player_id,
                                "match": False,
                                "expected_action": expected_action,
                                "actual_action": None,
                                "error": "No legal actions match the specified correct/incorrect actions",
                            }
                        )
                    break
            
            agent = _make_agent(
                request.agent_type, 
                player_id, 
                drill_guideline_text=drill_guideline_text,
                exclude_strategic_advice=request.exclude_strategic_advice,
                exclude_higher_level_features=request.exclude_higher_level_features,
            )

            try:
                choice = agent.choose_action(state, la_list)
                if isinstance(choice, tuple) and len(choice) == 4:
                    action, payload, reasoning, raw_llm_response = choice
                elif isinstance(choice, tuple) and len(choice) == 3:
                    action, payload, reasoning = choice
                    raw_llm_response = None
                else:
                    action, payload = choice
                    reasoning = None
                    raw_llm_response = None
            except Exception as e:
                passed = False
                first_mismatch = {"idx": s["idx"], "error": str(e)}
                if request.include_step_results:
                    step_results.append(
                        {
                            "idx": s["idx"],
                            "player_id": player_id,
                            "match": False,
                            "expected_action": expected_action,
                            "actual_action": None,
                            "error": str(e),
                        }
                    )
                break

            actual_action_dict: Dict[str, Any] = {"type": serialize_action(action)}
            if payload is not None:
                actual_action_dict["payload"] = serialize_action_payload(payload)
            if reasoning:
                actual_action_dict["reasoning"] = reasoning
            if raw_llm_response is not None:
                actual_action_dict["raw_llm_response"] = raw_llm_response

            # Check match: if using enhanced format, check against correct_actions set
            if correct_actions is not None:
                match = False
                for correct_action in correct_actions:
                    if _canonical_action_dict(actual_action_dict) == _canonical_action_dict(correct_action):
                        match = True
                        break
            else:
                # Backward compatibility: check against single expected_action
                match = _canonical_action_dict(actual_action_dict) == _canonical_action_dict(expected_action)
            if request.include_step_results:
                step_results.append(
                    {
                        "idx": s["idx"],
                        "player_id": player_id,
                        "match": match,
                        "expected_action": expected_action,
                        "actual_action": actual_action_dict,
                    }
                )

            if not match:
                passed = False
                first_mismatch = {
                    "idx": s["idx"],
                    "expected_action": expected_action,
                    "actual_action": actual_action_dict,
                }
                break

        return {
            "drill_id": prep["drill_id"],
            "name": prep["name"],
            "source_game_id": prep["source_game_id"],
            "source_step_idx": prep["source_step_idx"],
            "player_id": prep["player_id"],
            "num_steps": prep["num_steps"],
            "passed": passed,
            "first_mismatch": first_mismatch,
            **({"step_results": step_results} if request.include_step_results else {}),
        }

    sem = asyncio.Semaphore(request.max_concurrency)

    async def _run_one(prep: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            return await asyncio.to_thread(_eval_one_drill, prep)

    results = await asyncio.gather(*[_run_one(p) for p in prepared], return_exceptions=True)
    summaries: List[Dict[str, Any]] = []
    for prep, res in zip(prepared, results):
        if isinstance(res, Exception):
            summaries.append(
                {
                    "drill_id": prep["drill_id"],
                    "name": prep["name"],
                    "source_game_id": prep["source_game_id"],
                    "source_step_idx": prep["source_step_idx"],
                    "player_id": prep["player_id"],
                    "num_steps": prep["num_steps"],
                    "passed": False,
                    "first_mismatch": {"idx": 0, "error": str(res)},
                    **({"step_results": []} if request.include_step_results else {}),
                }
            )
        else:
            summaries.append(res)

    return {
        "agent_type": request.agent_type,
        "run_id": run_id,
        "evaluated_at": evaluated_at,
        "max_concurrency": request.max_concurrency,
        "include_guidelines": request.include_guidelines,
        "results": summaries,
    }


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
    
    # Copy agent_mapping from source game if it exists
    if "agent_mapping" in source_metadata:
        metadata["agent_mapping"] = source_metadata["agent_mapping"]
    
    # Create the new game in the database
    create_game_in_db(
        new_game_id,
        rng_seed=source_game_row["rng_seed"],  # Preserve RNG seed for reproducibility
        metadata=metadata,
        initial_state_json=state,
    )
    
    # Add metadata to response state so frontend can restore agent_mapping
    response_state = state.copy()
    response_state["_metadata"] = metadata
    
    return CreateGameResponse(
        game_id=new_game_id,
        initial_state=response_state
    )


class RunAgentsRequest(BaseModel):
    """Request to run agents automatically."""
    max_turns: int = 1000  # Maximum number of turns
    exclude_strategic_advice: bool = False  # Exclude strategic advice from LLM prompt
    exclude_higher_level_features: bool = False  # Exclude computed features like production analysis


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
    agent_mapping: Optional[Dict[str, str]] = None  # player_id -> agent_type (e.g., "behavior_tree", "random")
    # If a player is not in agent_mapping, they are treated as human players
    exclude_strategic_advice: Optional[bool] = None  # Exclude strategic advice from LLM prompt (defaults to game metadata)
    exclude_higher_level_features: Optional[bool] = None  # Exclude computed features like production analysis (defaults to game metadata)


class WatchAgentsResponse(BaseModel):
    """Response from watching agents (single step)."""
    game_id: str
    game_continues: bool  # True if game should continue, False if finished/error
    error: Optional[str] = None  # Error message if stopped
    new_state: Dict[str, Any]  # New game state after action
    player_id: Optional[str] = None  # Player who took the action
    reasoning: Optional[str] = None  # Agent's reasoning for the action


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
    
    # Get flags from metadata if not provided in request (use metadata as defaults)
    current_metadata = {}
    if game_row["metadata"]:
        try:
            current_metadata = json.loads(game_row["metadata"])
        except:
            current_metadata = {}
    
    exclude_strategic_advice = request.exclude_strategic_advice
    if exclude_strategic_advice is None and "exclude_strategic_advice" in current_metadata:
        exclude_strategic_advice = current_metadata["exclude_strategic_advice"]
    else:
        exclude_strategic_advice = exclude_strategic_advice if exclude_strategic_advice is not None else False
    
    exclude_higher_level_features = request.exclude_higher_level_features
    if exclude_higher_level_features is None and "exclude_higher_level_features" in current_metadata:
        exclude_higher_level_features = current_metadata["exclude_higher_level_features"]
    else:
        exclude_higher_level_features = exclude_higher_level_features if exclude_higher_level_features is not None else False
    
    # Create agents only for players specified in agent_mapping
    agents = {}
    agent_mapping = request.agent_mapping or {}
    
    # Store agent_mapping in game metadata if provided (for future restoration)
    if agent_mapping:
        # Update agent_mapping in metadata (merge with existing)
        if "agent_mapping" not in current_metadata or current_metadata["agent_mapping"] != agent_mapping:
            current_metadata["agent_mapping"] = agent_mapping
            # Update metadata in database
            from api.database import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE games
                SET metadata = ?
                WHERE id = ?
            """, (json.dumps(current_metadata), game_id))
            conn.commit()
    
    # Import agent classes
    from agents import RandomAgent, BehaviorTreeAgent
    try:
        from agents.llm_agent import LLMAgent
        from agents.variants import (
            BalancedAgent,
            AggressiveBuilderAgent,
            DevCardFocusedAgent,
            ExpansionAgent,
            DefensiveAgent,
            StateConditionedAgent,
            PlayerStyleImitationAgent,
        )
        AGENT_CLASSES = {
            "random": RandomAgent,
            "behavior_tree": BehaviorTreeAgent,
            "balanced": BalancedAgent,
            "aggressive_builder": AggressiveBuilderAgent,
            "dev_card_focused": DevCardFocusedAgent,
            "expansion": ExpansionAgent,
            "defensive": DefensiveAgent,
            "state_conditioned": StateConditionedAgent,
            "player_style_imitation": PlayerStyleImitationAgent,
            "llm": LLMAgent,
        }
    except ImportError:
        AGENT_CLASSES = {
            "random": RandomAgent,
            "behavior_tree": BehaviorTreeAgent,
        }
        try:
            from agents.llm_agent import LLMAgent
            AGENT_CLASSES["llm"] = LLMAgent
        except ImportError:
            pass
    
    for player in current_state.players:
        if player.id in agent_mapping:
            agent_type = agent_mapping[player.id]
            llm_spec = _parse_llm_agent_spec(agent_type)
            agent_class = AGENT_CLASSES.get("llm" if llm_spec is not None else agent_type, RandomAgent)
            
            # Special handling for LLM agent
            if llm_spec is not None:
                import os
                model = llm_spec["model"] or os.getenv("LLM_MODEL", "gpt-5.1")  # Default to gpt-5.1 (latest model)
                # LiteLLM automatically picks up the correct API key from environment based on model name
                # No need to pass api_key - just ensure environment variables are set (handled by main.py)
                # Zero-shot mode: disable retrieval
                agents[player.id] = agent_class(
                    player.id,
                    api_key=None,  # Let LiteLLM use environment variables automatically
                    model=model,
                    enable_retrieval=False,  # Zero-shot mode
                    thinking_mode=llm_spec["thinking_mode"],
                    thinking_effort=llm_spec["thinking_effort"],
                    exclude_strategic_advice=exclude_strategic_advice,
                    exclude_higher_level_features=exclude_higher_level_features,
                )
            else:
                agents[player.id] = agent_class(player.id)
    
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
        
        # Extract reasoning and raw LLM response if present
        reasoning = action.get("reasoning", None)
        raw_llm_response = action.get("raw_llm_response", None)
        
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
            reasoning=reasoning,  # Store reasoning in database
            raw_llm_response=raw_llm_response,  # Store raw LLM response for parsing error analysis
        )
    
    # Get current player to check if they have an agent
    if current_state.phase == "setup":
        current_player = current_state.players[current_state.setup_phase_player_index]
    elif current_state.phase == "playing":
        current_player = current_state.players[current_state.current_player_index]
    else:
        # Game finished
        return WatchAgentsResponse(
            game_id=game_id,
            game_continues=False,
            error=None,
            new_state=state_json,
            player_id=None
        )
    
    # Decide whether we can/should run an agent step.
    #
    # Important: even if the *current* player is human, we may still need to run
    # an out-of-turn mandatory action for an agent (notably DISCARD_RESOURCES
    # after a 7 roll). AgentRunner.run_step supports this, so don't early-return.
    should_run = current_player.id in agents
    if not should_run and current_state.phase == "playing" and current_state.dice_roll == 7:
        # If any agent-controlled player still needs to discard, let the runner
        # process one discard action.
        for p in current_state.players:
            if p.id in agents and sum(p.resources.values()) >= 8 and p.id not in current_state.players_discarded:
                should_run = True
                break

    if not should_run:
        # No agent action to take right now.
        return WatchAgentsResponse(
            game_id=game_id,
            game_continues=True,
            error=None,
            new_state=state_json,
            player_id=None,
            reasoning=None,
        )
    
    # Run a single step (agent's turn)
    new_state, game_continues, error, player_id = runner.run_step(save_state_callback=save_state_callback)
    
    # Extract reasoning from the last saved action (if available)
    reasoning = None
    if player_id:
        # Get the last step to extract reasoning
        steps = get_steps(game_id)
        if steps:
            last_step = steps[-1]
            # sqlite3.Row objects use bracket notation, not .get()
            action_json_str = last_step['action_json'] if 'action_json' in last_step.keys() else None
            if action_json_str:
                action_json = json.loads(action_json_str) if isinstance(action_json_str, str) else action_json_str
                reasoning = action_json.get("reasoning") if isinstance(action_json, dict) else None
    
    # Serialize new state
    new_state_json = serialize_game_state(new_state)
    
    return WatchAgentsResponse(
        game_id=game_id,
        game_continues=game_continues,
        error=error,
        new_state=new_state_json,
        player_id=player_id,
        reasoning=reasoning
    )


class QueryEventsRequest(BaseModel):
    """Request model for querying game events."""
    num_games: int = 100
    action_type: Optional[str] = None
    card_type: Optional[str] = None
    dice_roll: Optional[int] = None
    player_id: Optional[str] = None
    min_turn: Optional[int] = None
    max_turn: Optional[int] = None
    analyze: Optional[str] = None
    limit: Optional[int] = None


@router.post("/games/query_events")
async def query_events(request: QueryEventsRequest):
    """Query game events across multiple games.
    
    Allows searching for specific events (e.g., monopoly card plays, 7-rolls)
    across multiple games and analyzing their correctness.
    """
    from scripts.query_game_events import GameEventQuery
    
    query = GameEventQuery()
    events = query.query_games(
        num_games=request.num_games,
        action_type=request.action_type,
        card_type=request.card_type,
        dice_roll=request.dice_roll,
        player_id=request.player_id,
        min_turn=request.min_turn,
        max_turn=request.max_turn,
    )
    
    if request.limit:
        events = events[:request.limit]
    
    # Get summary
    summary = query.get_event_summary()
    
    # Run analysis if requested
    analysis = None
    if request.analyze == "monopoly":
        analysis = query.analyze_monopoly_card()
    
    return {
        "events": [e.to_dict() for e in events],
        "summary": summary,
        "analysis": analysis,
    }


# ============================================================================
# Guidelines and Feedback API Endpoints
# ============================================================================

class AddGuidelineRequest(BaseModel):
    guideline_text: str
    player_id: Optional[str] = None
    context: Optional[str] = None
    priority: int = 0


class UpdateGuidelineRequest(BaseModel):
    guideline_text: Optional[str] = None
    context: Optional[str] = None
    priority: Optional[int] = None
    active: Optional[bool] = None


class AddFeedbackRequest(BaseModel):
    feedback_text: str
    step_idx: Optional[int] = None
    player_id: Optional[str] = None
    action_taken: Optional[str] = None
    feedback_type: str = "general"


@router.post("/guidelines")
async def add_guideline(request: AddGuidelineRequest):
    """Add a new guideline for LLM agents."""
    from api.guidelines_db import add_guideline
    guideline_id = add_guideline(
        guideline_text=request.guideline_text,
        player_id=request.player_id,
        context=request.context,
        priority=request.priority
    )
    return {"guideline_id": guideline_id, "message": "Guideline added successfully"}


@router.get("/guidelines")
async def get_guidelines_endpoint(
    player_id: Optional[str] = None,
    context: Optional[str] = None,
    active_only: bool = True
):
    """Get guidelines."""
    from api.guidelines_db import get_guidelines
    guidelines = get_guidelines(
        player_id=player_id,
        context=context,
        active_only=active_only
    )
    return {"guidelines": guidelines}


@router.put("/guidelines/{guideline_id}")
async def update_guideline_endpoint(guideline_id: int, request: UpdateGuidelineRequest):
    """Update a guideline."""
    from api.guidelines_db import update_guideline
    updated = update_guideline(
        guideline_id,
        **request.dict(exclude_unset=True)
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Guideline not found")
    return {"message": "Guideline updated successfully"}


@router.delete("/guidelines/{guideline_id}")
async def delete_guideline_endpoint(guideline_id: int):
    """Delete (deactivate) a guideline."""
    from api.guidelines_db import delete_guideline
    deleted = delete_guideline(guideline_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Guideline not found")
    return {"message": "Guideline deleted successfully"}


@router.post("/games/{game_id}/feedback")
async def add_feedback_endpoint(game_id: str, request: AddFeedbackRequest):
    """Add feedback for a specific move in a game."""
    from api.guidelines_db import add_feedback
    feedback_id = add_feedback(
        game_id=game_id,
        feedback_text=request.feedback_text,
        step_idx=request.step_idx,
        player_id=request.player_id,
        action_taken=request.action_taken,
        feedback_type=request.feedback_type
    )
    return {"feedback_id": feedback_id, "message": "Feedback added successfully"}


@router.get("/feedback")
async def get_feedback_endpoint(
    game_id: Optional[str] = None,
    player_id: Optional[str] = None,
    limit: int = 50
):
    """Get feedback."""
    from api.guidelines_db import get_feedback
    feedback = get_feedback(
        game_id=game_id,
        player_id=player_id,
        limit=limit
    )
    return {"feedback": feedback}


# =============================================================================
# Optimized Prompts API Endpoints
# =============================================================================

@router.get("/prompts")
async def list_prompts():
    """List all optimized prompts."""
    prompts = list_optimized_prompts()
    return {
        "prompts": [
            {
                "id": r["id"],
                "name": r["name"],
                "created_at": r["created_at"],
                "is_default": bool(r["is_default"]),
                "metadata": json.loads(r["metadata"]) if "metadata" in r.keys() and r["metadata"] else None,
            }
            for r in prompts
        ]
    }


@router.get("/prompts/{name}")
async def get_prompt(name: str):
    """Get a specific optimized prompt."""
    prompt_row = get_optimized_prompt(name)
    if not prompt_row:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return {
        "id": prompt_row["id"],
        "name": prompt_row["name"],
        "system_prompt": prompt_row["system_prompt"],
        "created_at": prompt_row["created_at"],
        "is_default": bool(prompt_row["is_default"]),
            "metadata": json.loads(prompt_row["metadata"]) if "metadata" in prompt_row.keys() and prompt_row["metadata"] else None,
    }


class CreatePromptRequest(BaseModel):
    name: str
    system_prompt: str
    metadata: Optional[Dict[str, Any]] = None
    is_default: bool = False


@router.post("/prompts")
async def create_prompt(request: CreatePromptRequest):
    """Create or update an optimized prompt."""
    prompt_id = save_optimized_prompt(
        name=request.name,
        system_prompt=request.system_prompt,
        metadata=request.metadata,
        is_default=request.is_default
    )
    return {
        "id": prompt_id,
        "name": request.name,
        "message": "Prompt saved"
    }


@router.put("/prompts/{name}/set_default")
async def set_default_prompt_endpoint(name: str):
    """Set a prompt as the default."""
    success = set_default_prompt(name)
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"message": f"Prompt '{name}' set as default"}


@router.delete("/prompts/{name}")
async def delete_prompt(name: str):
    """Delete an optimized prompt."""
    success = delete_optimized_prompt(name)
    if not success:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"message": f"Prompt '{name}' deleted"}
