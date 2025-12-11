# Getting Started with the Game Template

This template provides everything you need to build a board/card game with frontend, backend, agents, and automated testing.

## Quick Start

1. **Copy this template** to your new repository
2. **Rename files**: Remove `.template` extension from template files
3. **Customize** the game-specific code
4. **Follow the phases** in `PROCESS.md`

## Step-by-Step

### 1. Set Up Project Structure

```bash
# Copy template to new repo
cp -r game_template/* /path/to/new/repo/

# Remove .template extensions
find . -name "*.template" -exec sh -c 'mv "$1" "${1%.template}"' _ {} \;
```

### 2. Install Dependencies

```bash
# Backend
cd backend && uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

### 3. Start with Game Engine

Begin by customizing `backend/engine/engine.py`:

1. Define your `Action` enum
2. Define `ActionPayload` classes
3. Define `Player` class
4. Define `GameState` class
5. Implement `GameState.step()` method
6. Implement win condition checks

### 4. Implement Serialization

Customize `backend/engine/serialization.py`:

1. `serialize_game_state()` - Convert state to JSON
2. `deserialize_game_state()` - Convert JSON to state
3. `state_to_text()` - Human-readable state (for LLM)
4. `legal_actions()` - Get valid actions for a player
5. `legal_actions_to_text()` - Human-readable actions (for LLM)

### 5. Set Up Database

Customize `backend/api/database.py`:

1. Adjust table schema if needed
2. Implement game creation
3. Implement state saving/loading

### 6. Create API Routes

Customize `backend/api/routes.py`:

1. Implement `POST /api/games` - Create game
2. Implement `GET /api/games/{game_id}` - Get state
3. Implement `POST /api/games/{game_id}/act` - Perform action
4. Implement `GET /api/games/{game_id}/legal-actions` - Get actions

### 7. Build Frontend

Customize `frontend/src/`:

1. `api.ts` - API client types and functions
2. `App.tsx` - Main game UI
3. `App.css` - Game styles

### 8. Implement Agents

1. Customize `backend/agents/random_agent.py` - Random action selection
2. Customize `backend/agents/llm_agent.py` - LLM-based decisions (optional)
3. Customize `backend/agents/agent_runner.py` - Agent execution flow

### 9. Test Your Game

1. Write unit tests in `backend/tests/`
2. Run agents: `python -m scripts.run_agents <game_id>`
3. Add bug regression tests as you find issues

## Key Files to Customize

### Must Customize (Game-Specific)

- `backend/engine/engine.py` - Core game logic
- `backend/engine/serialization.py` - State serialization
- `backend/api/routes.py` - API endpoints
- `backend/api/database.py` - Database schema (if needed)
- `frontend/src/App.tsx` - Game UI
- `frontend/src/api.ts` - API client

### Optional Customize

- `backend/agents/random_agent.py` - Random agent behavior
- `backend/agents/llm_agent.py` - LLM prompts
- `backend/agents/agent_runner.py` - Agent execution flow
- `backend/scripts/create_game.py` - Game initialization
- `backend/scripts/run_agents.py` - Agent execution script

### Don't Need to Change

- `backend/agents/base_agent.py` - Base agent interface
- `backend/bug_tests/` - Bug regression testing (works as-is)
- `backend/main.py` - FastAPI setup
- Configuration files (package.json, requirements.txt, etc.)

## Testing Workflow

1. **Manual Testing**: Use frontend to play manually
2. **Random Agent**: Run random agent to test basic functionality
3. **LLM Agent**: Run LLM agent to find edge cases
4. **Bug Regression**: Add tests for bugs you find
5. **Iterate**: Fix bugs, add features, repeat

## Tips

- Start simple: Build minimal game engine first
- Test early: Use bug regression system from the start
- Use agents: They'll find bugs you might miss
- Document rules: Clear rules help LLM agent play correctly
- Iterate: Build, test, fix, repeat

## Next Steps

- Read `README.md` for overview
- Read `PROCESS.md` for detailed process
- Start customizing `backend/engine/engine.py`

