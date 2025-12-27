"""API routes for the Catan game."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from typing import List, Dict, Optional, Any
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
)


class DrillStepCreate(BaseModel):
    player_id: str
    state: Dict[str, Any]
    expected_action: Dict[str, Any]


class CreateDrillRequest(BaseModel):
    name: Optional[str] = None
    source_game_id: Optional[str] = None
    source_step_idx: Optional[int] = None
    player_id: str
    steps: List[DrillStepCreate]
    metadata: Optional[Dict[str, Any]] = None


class CreateDrillResponse(BaseModel):
    drill_id: int
    message: str


@router.get("/drills")
async def list_drills(limit: int = 200):
    rows = list_drills_from_db(limit=limit)
    drills = []
    for r in rows:
        drills.append(
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "name": r["name"],
                "source_game_id": r["source_game_id"],
                "source_step_idx": r["source_step_idx"],
                "player_id": r["player_id"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
                "num_steps": r["num_steps"],
            }
        )
    return {"drills": drills}


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
        steps_for_db.append(
            {
                "idx": idx,
                "player_id": s.player_id,
                "state_json": s.state,
                "expected_action_json": s.expected_action,
                "state_text": state_to_text(state, s.player_id),
                "legal_actions_text": legal_actions_to_text(la_list),
            }
        )

    drill_id = create_drill_in_db(
        name=request.name,
        source_game_id=request.source_game_id,
        source_step_idx=request.source_step_idx,
        player_id=request.player_id,
        metadata=request.metadata,
        steps=steps_for_db,
    )
    return CreateDrillResponse(drill_id=drill_id, message="Drill created")


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


def _make_agent(agent_type: str, player_id: str):
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
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        model = llm_spec["model"] or os.getenv("LLM_MODEL", "gpt-5.1")
        return agent_class(
            player_id,
            api_key=api_key,
            model=model,
            enable_retrieval=False,
            thinking_mode=llm_spec["thinking_mode"],
            thinking_effort=llm_spec["thinking_effort"],
        )
    agent_class = AGENT_CLASSES.get(agent_type, RandomAgent)
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


class EvaluateDrillRequest(BaseModel):
    agent_type: str = "random"


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
    for r in steps:
        player_id = r["player_id"]
        state_json = json.loads(r["state_json"])
        expected_action = json.loads(r["expected_action_json"])
        state = deserialize_game_state(state_json)
        la_list = legal_actions(state, player_id)

        agent = _make_agent(request.agent_type, player_id)
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


@router.post("/drills/evaluate_all")
async def evaluate_all_drills(request: EvaluateAllDrillsRequest):
    if request.max_concurrency < 1 or request.max_concurrency > 32:
        raise HTTPException(status_code=400, detail="max_concurrency must be between 1 and 32")

    drill_rows = list_drills_from_db(limit=request.limit)
    evaluated_at = datetime.utcnow().isoformat()
    run_id = str(uuid.uuid4())
    # Preload all drill steps in the main thread (read-only), then evaluate drills concurrently.
    prepared: List[Dict[str, Any]] = []
    for d in drill_rows:
        drill_id = int(d["id"])
        step_rows = get_drill_steps_from_db(drill_id)
        steps_prepped = []
        for r in step_rows:
            steps_prepped.append(
                {
                    "idx": int(r["idx"]),
                    "player_id": r["player_id"],
                    "state_json": json.loads(r["state_json"]),
                    "expected_action": json.loads(r["expected_action_json"]),
                }
            )
        prepared.append(
            {
                "drill_id": drill_id,
                "name": d["name"],
                "source_game_id": d["source_game_id"],
                "source_step_idx": d["source_step_idx"],
                "player_id": d["player_id"],
                "num_steps": d["num_steps"],
                "steps": steps_prepped,
            }
        )

    def _eval_one_drill(prep: Dict[str, Any]) -> Dict[str, Any]:
        steps = prep["steps"]
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
            state = deserialize_game_state(state_json)
            la_list = legal_actions(state, player_id)
            agent = _make_agent(request.agent_type, player_id)

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
    
    # Create agents only for players specified in agent_mapping
    agents = {}
    agent_mapping = request.agent_mapping or {}
    
    # Store agent_mapping in game metadata if provided (for future restoration)
    if agent_mapping:
        # Get current metadata
        current_metadata = {}
        if game_row["metadata"]:
            try:
                current_metadata = json.loads(game_row["metadata"])
            except:
                current_metadata = {}
        
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
                # Get API key and model from env vars
                api_key = (
                    os.getenv("OPENAI_API_KEY") or
                    os.getenv("ANTHROPIC_API_KEY") or
                    os.getenv("GEMINI_API_KEY") or
                    os.getenv("LLM_API_KEY")
                )
                model = llm_spec["model"] or os.getenv("LLM_MODEL", "gpt-5.1")  # Default to gpt-5.1 (latest model)
                # Zero-shot mode: disable retrieval
                agents[player.id] = agent_class(
                    player.id,
                    api_key=api_key,
                    model=model,
                    enable_retrieval=False,  # Zero-shot mode
                    thinking_mode=llm_spec["thinking_mode"],
                    thinking_effort=llm_spec["thinking_effort"],
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
