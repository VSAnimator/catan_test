# Catan Game - Monorepo

A Catan-like game implementation designed for human + LLM play and research. This monorepo contains a pure Python game engine, a FastAPI REST API, and a React + TypeScript frontend.

## Architecture

### Backend (`backend/`)

**⚠️ IMPORTANT: Backend Environment**
- The backend uses `uv` for package management with a virtual environment located at `backend/.venv/`
- **Always use `uv pip` instead of `pip`** for installing packages
- **Always activate the virtual environment** (`source backend/.venv/bin/activate`) before running tests or installing packages
- Example: `cd backend && source .venv/bin/activate && uv pip install <package>`
- For tests: `cd backend && source .venv/bin/activate && python -m pytest`

The backend is structured in three main layers:

1. **Game Engine** (`backend/engine/`)
   - Pure Python game logic with no web framework dependencies
   - Contains core game classes: `GameState`, `Player`, `Tile`, `Intersection`, `RoadEdge`, `ResourceType`
   - Can be used independently or imported by other modules
   - Located in `backend/engine/engine.py` and `backend/engine/serialization.py`
   - Includes text-based observation space for LLM agents (see `backend/agents/OBSERVATION_SPACE.md`)

2. **REST API Layer** (`backend/api/`)
   - FastAPI-based REST API that wraps the game engine
   - Provides HTTP endpoints for game operations
   - Located in `backend/api/routes.py`
   - Main application entry point: `backend/main.py`
   - SQLite database for game state persistence (`backend/api/database.py`)

3. **Agent System** (`backend/agents/`)
   - Multiple AI agent implementations for automated gameplay
   - **RandomAgent**: Random action selection
   - **BehaviorTreeAgent**: Rule-based strategic agent
   - **LLMAgent**: LLM-powered agent using ReAct pattern with RAG
   - **Agent Variants**: Specialized behavior tree agents (aggressive, defensive, balanced, etc.)
   - Agent runner for automated game execution
   - See `backend/agents/OBSERVATION_SPACE.md` for agent observation space documentation

**Key Features:**
- FastAPI with automatic OpenAPI documentation
- CORS enabled for frontend communication
- SQLite database for game state persistence
- RESTful endpoints for game creation, state retrieval, and game actions
- Multiple AI agent types for automated gameplay
- Game replay and history functionality
- Bug testing framework (`backend/bug_tests/`)

### Frontend (`frontend/`)

- React 18 with TypeScript
- Vite for fast development and building
- Minimal dependencies for clean, maintainable code
- Connects to backend API at `http://localhost:8000`

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- `uv` (for Python package management) - install from https://github.com/astral-sh/uv

### Installation

1. **Install backend dependencies:**
   ```bash
   make install-backend
   # or manually:
   cd backend && uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
   ```
   
   **Note:** Always use `uv pip` (not `pip`) when working in the backend directory. The virtual environment is located at `backend/.venv/` and must be activated before running any Python commands.

2. **Install frontend dependencies:**
   ```bash
   make install-frontend
   # or manually:
   cd frontend && npm install
   ```

### Running the Application

**Option 1: Run both services together**
```bash
make dev
```

**Option 2: Run services separately**

Terminal 1 - Backend:
```bash
make dev-backend
# or manually:
source ~/.zshrc && cd backend && source .venv/bin/activate && uvicorn main:app --reload
```
**Note:** Always source `~/.zshrc` first to ensure environment variables are loaded.

Terminal 2 - Frontend:
```bash
make dev-frontend
# or manually:
cd frontend && npm run dev
```

### Access Points

- **Frontend**: http://localhost:5173-5176 (Vite may use different ports)
- **Backend API**: http://localhost:8000 (or configured port)
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc

## Project Structure

```
catan_agent/
├── backend/
│   ├── engine/               # Pure game engine (no web deps)
│   │   ├── __init__.py
│   │   ├── engine.py         # Core game logic
│   │   └── serialization.py  # Serialization utilities
│   ├── api/                  # REST API layer
│   │   ├── __init__.py
│   │   ├── routes.py         # API endpoints
│   │   ├── database.py       # SQLite database utilities
│   │   └── guidelines_db.py  # Guidelines/feedback database
│   ├── agents/               # AI agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Base agent interface
│   │   ├── random_agent.py   # Random action agent
│   │   ├── behavior_tree_agent.py  # Rule-based agent
│   │   ├── llm_agent.py      # LLM-powered agent
│   │   ├── agent_runner.py   # Agent execution runner
│   │   ├── llm_retrieval.py  # RAG for LLM agent
│   │   ├── OBSERVATION_SPACE.md  # Agent observation docs
│   │   └── variants/         # Specialized agent variants
│   ├── bug_tests/            # Bug testing framework
│   ├── scripts/              # Utility scripts
│   ├── tests/                # Unit tests
│   ├── main.py               # FastAPI app entry point
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main React component
│   │   ├── api.ts            # API client
│   │   ├── App.css
│   │   ├── main.tsx          # React entry point
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── Makefile                  # Development commands
└── README.md                 # This file
```

## API Endpoints

### Game Management
- `POST /api/games` - Create a new game
- `GET /api/games/{game_id}` - Get current game state
- `POST /api/games/{game_id}/act` - Perform an action in the game
- `GET /api/games/{game_id}/legal_actions` - Get legal actions for a player
- `GET /api/games/{game_id}/replay` - Get game replay/history
- `POST /api/games/{game_id}/restore` - Restore game to a specific step
- `POST /api/games/{game_id}/fork` - Fork a game at a specific step

### Agent Operations
- `POST /api/games/{game_id}/run_agents` - Run agents automatically to completion
- `POST /api/games/{game_id}/watch_agents_step` - Step through agent gameplay

### Guidelines & Feedback
- `POST /api/guidelines` - Create a guideline
- `GET /api/guidelines` - Get all guidelines
- `PUT /api/guidelines/{guideline_id}` - Update a guideline
- `DELETE /api/guidelines/{guideline_id}` - Delete a guideline
- `POST /api/games/{game_id}/feedback` - Submit feedback for a game
- `GET /api/feedback` - Get feedback entries

### Utilities
- `POST /api/games/query_events` - Query game events
- `GET /health` - Health check
- `GET /` - API info

## Development Notes

### Backend Development
- **Always use `uv pip` (not `pip`)** for package management in the backend
- **Always activate the virtual environment** (`source backend/.venv/bin/activate`) before:
  - Installing packages: `uv pip install <package>`
  - Running tests: `python -m pytest`
  - Running the server: `uvicorn main:app --reload`
- The virtual environment is located at `backend/.venv/`

### General Notes
- The game engine is intentionally separated from the API layer for testability and reusability
- SQLite database is used for game state persistence
- CORS is configured to allow requests from the frontend development server
- TypeScript strict mode is enabled in the frontend
- Agent observation space is documented in `backend/agents/OBSERVATION_SPACE.md`

### Agent Development
- Agents implement the `BaseAgent` interface
- LLM agents use a text-based observation space (see `OBSERVATION_SPACE.md`)
- Multiple agent variants available for different strategies
- Agent runner supports automated gameplay with state persistence

## Features

✅ **Implemented:**
- Full Catan game rules implementation
- SQLite database for game state persistence
- Multiple AI agent types (Random, Behavior Tree, LLM)
- LLM agent with ReAct pattern and RAG
- Game history and replay functionality
- Bug testing framework
- Text-based observation space for LLM agents
- Compact graph representation of board state

## Future Enhancements

- WebSocket support for real-time updates
- Additional agent strategies and variants
- Performance optimizations for large-scale agent training
- Enhanced observation space features

