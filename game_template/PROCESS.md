# Detailed Development Process

This document provides a step-by-step guide for building a game using this template.

## Overview

The process follows these phases:
1. **Game Engine** - Pure Python game logic
2. **REST API** - FastAPI wrapper around engine
3. **Frontend** - React UI for playing
4. **Agents** - AI players (random and LLM)
5. **Testing** - Automated testing and bug regression
6. **Agent Testing** - Running agents and analyzing logs

## Phase 1: Game Engine

### Step 1.1: Design Game State

In `backend/engine/engine.py`, define:

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional

# Action types
class Action(Enum):
    # Define your game's actions
    # Example: PLAY_CARD, DRAW_CARD, PASS, etc.
    pass

# Action payloads
@dataclass
class ActionPayload:
    # Base class for action payloads
    pass

# Game state
@dataclass
class GameState:
    # Core game state
    players: List[Player]
    current_player_index: int
    phase: str  # e.g., "setup", "playing", "finished"
    turn_number: int
    # Game-specific fields
    pass

@dataclass
class Player:
    id: str
    name: str
    # Player-specific fields (hand, score, etc.)
    pass
```

### Step 1.2: Implement Game Rules

Implement the core game logic:

```python
class GameState:
    def step(self, action: Action, payload: Optional[ActionPayload] = None) -> Dict[str, Any]:
        """
        Execute an action and return events.
        
        Returns:
            Dict with 'events' (list of events) and 'error' (optional error message)
        """
        # Validate action
        # Update game state
        # Return events
        pass
    
    def is_game_over(self) -> bool:
        """Check if game is finished."""
        pass
    
    def get_winner(self) -> Optional[str]:
        """Get winner player ID if game is over."""
        pass
```

### Step 1.3: Implement Serialization

In `backend/engine/serialization.py`:

```python
def serialize_game_state(state: GameState) -> Dict[str, Any]:
    """Convert GameState to JSON-serializable dict."""
    pass

def deserialize_game_state(data: Dict[str, Any]) -> GameState:
    """Convert dict to GameState."""
    pass

def state_to_text(state: GameState, player_id: Optional[str] = None) -> str:
    """
    Convert game state to human-readable text for LLM.
    
    Args:
        state: Game state
        player_id: Optional player ID to focus on their perspective
    
    Returns:
        Human-readable text description
    """
    pass

def legal_actions(state: GameState, player_id: str) -> List[Tuple[Action, Optional[ActionPayload]]]:
    """
    Get legal actions for a player.
    
    Returns:
        List of (Action, Optional[ActionPayload]) tuples
    """
    pass

def legal_actions_to_text(actions: List[Tuple[Action, Optional[ActionPayload]]]) -> str:
    """Convert legal actions to human-readable text."""
    pass
```

### Step 1.4: Write Unit Tests

In `backend/tests/test_engine_basic.py`:

```python
def test_game_creation():
    """Test creating a new game."""
    pass

def test_basic_action():
    """Test a basic game action."""
    pass

def test_game_rules():
    """Test game rules (e.g., illegal moves raise errors)."""
    pass
```

## Phase 2: REST API

### Step 2.1: Set Up FastAPI

In `backend/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.database import init_db

init_db()

app = FastAPI(title="Your Game API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
```

### Step 2.2: Create Database Schema

In `backend/api/database.py`:

```python
import sqlite3
from typing import Optional, Dict, Any

def init_db():
    """Initialize database tables."""
    conn = sqlite3.connect('your_game.db')
    cursor = conn.cursor()
    
    # Games table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rng_seed INTEGER,
            metadata TEXT
        )
    ''')
    
    # Game states table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            step_id INTEGER,
            state_json TEXT,
            state_before_json TEXT,
            action_json TEXT,
            dice_roll INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_game(player_names: List[str], rng_seed: Optional[int] = None) -> str:
    """Create a new game and return game_id."""
    pass

def save_game_state(game_id: str, step_id: int, state_json: str, 
                   state_before_json: Optional[str] = None,
                   action_json: Optional[str] = None,
                   dice_roll: Optional[int] = None):
    """Save game state to database."""
    pass

def get_latest_state(game_id: str) -> Optional[str]:
    """Get latest game state JSON."""
    pass

def get_state_at_step(game_id: str, step_id: int) -> Optional[str]:
    """Get game state at specific step."""
    pass
```

### Step 2.3: Create API Routes

In `backend/api/routes.py`:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

class CreateGameRequest(BaseModel):
    player_names: List[str]
    rng_seed: Optional[int] = None

class CreateGameResponse(BaseModel):
    game_id: str
    initial_state: Dict[str, Any]

@router.post("/games", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest):
    """Create a new game."""
    # Create game in database
    # Initialize game state
    # Save initial state
    # Return game_id and initial state
    pass

@router.get("/games/{game_id}")
async def get_game(game_id: str):
    """Get current game state."""
    # Load from database
    # Return state JSON
    pass

@router.post("/games/{game_id}/act")
async def act(game_id: str, request: ActRequest):
    """Perform an action."""
    # Load current state
    # Deserialize action
    # Execute action
    # Save new state
    # Return new state
    pass

@router.get("/games/{game_id}/legal-actions")
async def get_legal_actions(game_id: str, player_id: str):
    """Get legal actions for a player."""
    # Load current state
    # Get legal actions
    # Return actions
    pass
```

## Phase 3: Frontend

### Step 3.1: Set Up React App

Already configured in `frontend/`. Customize:

- `frontend/src/App.tsx` - Main game UI
- `frontend/src/api.ts` - API client functions
- `frontend/src/App.css` - Game-specific styles

### Step 3.2: Build Game UI

In `frontend/src/App.tsx`:

```typescript
function App() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [legalActions, setLegalActions] = useState<LegalAction[]>([]);
  
  // Load game state
  useEffect(() => {
    if (gameId) {
      loadGameState();
    }
  }, [gameId]);
  
  // Display game state
  // Show legal actions as buttons
  // Handle user actions
}
```

## Phase 4: Agents

### Step 4.1: Base Agent Interface

Already provided in `backend/agents/base_agent.py`. No changes needed.

### Step 4.2: Random Agent

Customize `backend/agents/random_agent.py`:

```python
class RandomAgent(BaseAgent):
    def choose_action(self, state: GameState, 
                     legal_actions_list: List[Tuple[Action, Optional[ActionPayload]]]
                     ) -> Tuple[Action, Optional[ActionPayload], Optional[str]]:
        # Filter out actions you don't want random agent to use
        # Randomly select from remaining actions
        # Generate payloads if needed
        pass
```

### Step 4.3: LLM Agent

Customize prompts in `backend/agents/llm_agent.py`:

```python
SYSTEM_PROMPT = """
You are playing [YOUR GAME NAME].

Game Rules:
[Describe your game rules here]

Your goal: [Win condition]
"""

ACTION_PROMPT = """
Current game state:
{observation}

Legal actions:
{legal_actions_text}

Choose the best action. Format your response as:
THOUGHT: [Your reasoning]
ACTION: [Action name]
PAYLOAD: [JSON payload if needed]
"""
```

## Phase 5: Automated Testing

### Step 5.1: Bug Regression System

The bug regression system is already set up in `backend/bug_tests/`.

When you discover a bug:

```bash
python -m bug_tests.manage_tests add \
  --game-id <game_id> \
  --step-id <step_id> \
  --description "Bug description" \
  --expected "Expected behavior" \
  --undesired "Actual bug behavior" \
  --test-action '{"type": "ACTION_NAME", "payload": {...}}'
```

Run tests:
```bash
python -m bug_tests.manage_tests run
```

### Step 5.2: Agent Testing Scripts

Use `backend/scripts/run_agents.py`:

```bash
python -m scripts.run_agents <game_id> --max-turns 1000
```

Batch testing:
```bash
python -m scripts.test_agents_batch --num-games 10
```

## Phase 6: Testing via Agents

### Step 6.1: Run Agents

```bash
# Create a game
python -m scripts.create_game

# Run agents on it
python -m scripts.run_agents <game_id>
```

### Step 6.2: Analyze Logs

- Check game logs for errors
- Verify agents follow rules
- Look for edge cases
- Identify bugs

### Step 6.3: Add Regression Tests

For each bug found:
1. Note game_id and step_id
2. Add regression test
3. Fix bug
4. Verify test passes

## Iteration Cycle

1. **Build feature** in game engine
2. **Test manually** via frontend
3. **Run agents** to test automatically
4. **Find bugs** from agent gameplay
5. **Add regression tests** for bugs
6. **Fix bugs**
7. **Repeat**

## Tips

- Start with minimal game engine, add features incrementally
- Use random agent first to test basic functionality
- Use LLM agent to find edge cases and complex bugs
- Keep regression tests - they prevent future bugs
- Document game rules clearly for LLM agent
- Use step-based RNG seeding for reproducible tests

