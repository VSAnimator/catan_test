# Game Development Template

This template provides a complete structure for building board/card games with:
1. **Frontend** (React + TypeScript)
2. **Backend** (FastAPI + Python game engine)
3. **LLM Agent** and **Random Agent** implementations
4. **Automated Test Generation** (bug regression testing system)
5. **Agent Testing** (automated agent runners with game log analysis)

## Quick Start

1. **Copy this template** into your new repository
2. **Customize** the game-specific code (engine, actions, state)
3. **Follow the process** outlined below

## Development Process

### Phase 1: Game Engine (Pure Python)

1. **Design the game state** in `backend/engine/engine.py`:
   - Define core classes (GameState, Player, etc.)
   - Implement game rules and logic
   - Keep it pure Python with no web framework dependencies

2. **Implement serialization** in `backend/engine/serialization.py`:
   - Serialize/deserialize game state to/from JSON
   - Convert state to human-readable text for LLM agents
   - List legal actions for a given state

3. **Test the engine** in `backend/tests/`:
   - Unit tests for game logic
   - Test serialization/deserialization
   - Test edge cases

### Phase 2: REST API (FastAPI)

1. **Set up FastAPI** in `backend/main.py`:
   - Initialize FastAPI app
   - Configure CORS for frontend
   - Include API routes

2. **Create API routes** in `backend/api/routes.py`:
   - `POST /api/games` - Create new game
   - `GET /api/games/{game_id}` - Get game state
   - `POST /api/games/{game_id}/act` - Perform action
   - `GET /api/games/{game_id}/legal-actions` - Get legal actions
   - Add game-specific endpoints as needed

3. **Set up database** in `backend/api/database.py`:
   - Store game states
   - Track game history (steps)
   - Support state restoration for testing

### Phase 3: Frontend (React + TypeScript)

1. **Set up React app** in `frontend/`:
   - Use Vite for fast development
   - TypeScript for type safety
   - Connect to backend API

2. **Build game UI** in `frontend/src/App.tsx`:
   - Display game state
   - Show legal actions
   - Handle user actions
   - Support game-specific UI elements

3. **API client** in `frontend/src/api.ts`:
   - TypeScript types for API responses
   - Functions to call backend endpoints

### Phase 4: Agents

1. **Base Agent Interface** in `backend/agents/base_agent.py`:
   - Abstract `choose_action()` method
   - Takes game state and legal actions
   - Returns action, payload, and optional reasoning

2. **Random Agent** in `backend/agents/random_agent.py`:
   - Randomly selects from legal actions
   - Useful for testing and baseline

3. **LLM Agent** in `backend/agents/llm_agent.py`:
   - Uses LLM (via LiteLLM) to choose actions
   - ReAct pattern (Reasoning and Acting)
   - Optional RAG (Retrieval-Augmented Generation) from past games
   - Supports multiple LLM providers (OpenAI, Anthropic, Google, etc.)

4. **Agent Runner** in `backend/agents/agent_runner.py`:
   - Runs agents automatically on games
   - Handles action execution
   - Tracks reasoning and errors

### Phase 5: Automated Testing

1. **Bug Regression Testing** in `backend/bug_tests/`:
   - When bugs are discovered during gameplay, capture them:
     ```bash
     python -m bug_tests.manage_tests add \
       --game-id <game_id> \
       --step-id <step_id> \
       --description "Bug description" \
       --expected "Expected behavior" \
       --undesired "Actual bug behavior"
     ```
   - Tests restore game state from any step
   - Reproducible RNG via step-based seeding
   - Optional LLM validation of test results

2. **Agent Testing Scripts** in `backend/scripts/`:
   - `run_agents.py` - Run agents on a game until completion
   - `test_agents_batch.py` - Run multiple games with agents
   - `test_llm_agents.py` - Test LLM agents specifically
   - Analyze game logs to verify agent behavior

### Phase 6: Testing via Agents

1. **Run agents automatically**:
   ```bash
   python -m scripts.run_agents <game_id> --max-turns 1000
   ```

2. **Watch agents play** via frontend or logs

3. **Analyze game logs**:
   - Check for rule violations
   - Verify agent decision-making
   - Identify bugs or edge cases

4. **Add regression tests** for any bugs found

## Project Structure

```
game_template/
├── .cursorrules              # Cursor AI rules
├── README.md                 # This file
├── PROCESS.md                # Detailed process documentation
├── Makefile                  # Development commands
│
├── backend/
│   ├── engine/               # Pure Python game engine
│   │   ├── __init__.py
│   │   ├── engine.py         # Core game logic (CUSTOMIZE)
│   │   └── serialization.py  # Serialization (CUSTOMIZE)
│   │
│   ├── api/                  # REST API layer
│   │   ├── __init__.py
│   │   ├── routes.py         # API endpoints (CUSTOMIZE)
│   │   ├── database.py       # Database utilities (CUSTOMIZE)
│   │   └── guidelines_db.py  # Optional: LLM guidelines storage
│   │
│   ├── agents/               # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Base agent interface
│   │   ├── random_agent.py  # Random agent (CUSTOMIZE)
│   │   ├── llm_agent.py      # LLM agent (CUSTOMIZE prompts)
│   │   ├── llm_retrieval.py  # RAG for LLM agent
│   │   └── agent_runner.py   # Agent execution engine
│   │
│   ├── bug_tests/            # Bug regression testing
│   │   ├── __init__.py
│   │   ├── test_registry.py  # Test case management
│   │   ├── test_runner.py    # Test execution
│   │   ├── llm_validator.py  # LLM validation
│   │   └── manage_tests.py   # CLI tool
│   │
│   ├── scripts/              # Utility scripts
│   │   ├── create_game.py    # Create new game
│   │   ├── run_agents.py     # Run agents on game
│   │   ├── test_agents_batch.py  # Batch agent testing
│   │   └── ...               # Other utility scripts
│   │
│   ├── tests/                # Unit tests
│   │   ├── __init__.py
│   │   ├── test_engine_basic.py
│   │   └── test_serialization.py
│   │
│   ├── main.py               # FastAPI app entry point
│   └── requirements.txt      # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── App.tsx           # Main React component (CUSTOMIZE)
    │   ├── App.css           # Styles (CUSTOMIZE)
    │   ├── api.ts            # API client (CUSTOMIZE)
    │   ├── main.tsx          # React entry point
    │   └── index.css      # Global styles
    │
    ├── index.html
    ├── package.json          # Node dependencies
    ├── tsconfig.json
    └── vite.config.ts
```

## Customization Guide

### 1. Game Engine (`backend/engine/engine.py`)

Replace with your game's core logic:
- GameState class
- Player class
- Game-specific classes (cards, board, etc.)
- Action enum
- ActionPayload classes
- Game rules implementation

### 2. Serialization (`backend/engine/serialization.py`)

Customize for your game:
- `serialize_game_state()` - Convert state to JSON
- `deserialize_game_state()` - Convert JSON to state
- `state_to_text()` - Human-readable state description (for LLM)
- `legal_actions()` - List valid actions for a player
- `legal_actions_to_text()` - Human-readable action list (for LLM)

### 3. API Routes (`backend/api/routes.py`)

Customize endpoints for your game:
- Game creation parameters
- Action types specific to your game
- Game-specific queries

### 4. Frontend (`frontend/src/App.tsx`)

Customize UI for your game:
- Game board/state visualization
- Action buttons/controls
- Game-specific UI elements

### 5. Random Agent (`backend/agents/random_agent.py`)

Customize action filtering:
- Which actions should be filtered out?
- How to generate action payloads?
- Special handling for specific actions?

### 6. LLM Agent (`backend/agents/llm_agent.py`)

Customize prompts:
- System prompt describing game rules
- Action selection prompt
- Observation format

## Key Features

### State Restoration for Testing
- Games store state at each step
- Tests can restore state from any step
- Reproducible RNG via step-based seeding

### LLM Agent with RAG
- Retrieves similar game states from past games
- Learns from examples and user feedback
- Supports multiple LLM providers

### Automated Agent Testing
- Run agents automatically on games
- Batch testing for multiple games
- Tournament-style comparisons

### Bug Regression Testing
- Capture bugs from gameplay
- Restore exact game state where bug occurred
- Prevent regressions

## Dependencies

### Backend
- Python 3.11+
- `uv` for package management
- FastAPI, uvicorn, pydantic
- LiteLLM for LLM support
- pytest for testing

### Frontend
- Node.js 18+
- React 18
- TypeScript
- Vite

## Installation

```bash
# Backend
cd backend && uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

## Running

```bash
# Both services
make dev

# Or separately
make dev-backend  # Terminal 1
make dev-frontend # Terminal 2
```

## Next Steps

1. Read `PROCESS.md` for detailed process documentation
2. Start with `backend/engine/engine.py` - build your game engine
3. Follow the phases outlined above
4. Use the bug testing system as you discover issues
5. Test agents by running them automatically and analyzing logs

## Tips

- **Start simple**: Build a minimal game engine first, then add features
- **Test early**: Use the bug regression system from the start
- **Iterate**: Run agents, find bugs, add tests, fix bugs, repeat
- **Document**: Keep notes on game rules and design decisions
- **Use LLM agent**: It can help discover edge cases and bugs

