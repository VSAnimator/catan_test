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

The backend is structured in two layers:

1. **Game Engine** (`backend/engine/`)
   - Pure Python game logic with no web framework dependencies
   - Contains core game classes: `GameState`, `Player`, `Tile`, `Intersection`, `RoadEdge`, `ResourceType`
   - Can be used independently or imported by other modules
   - Located in `backend/engine/engine.py` and `backend/engine/serialization.py`

2. **REST API Layer** (`backend/api/`)
   - FastAPI-based REST API that wraps the game engine
   - Provides HTTP endpoints for game operations
   - Located in `backend/api/routes.py`
   - Main application entry point: `backend/main.py`

**Key Features:**
- FastAPI with automatic OpenAPI documentation
- CORS enabled for frontend communication
- In-memory game storage (can be replaced with database)
- RESTful endpoints for game creation, state retrieval, and game actions

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

- **Frontend**: http://localhost:5173 (Vite default)
- **Backend API**: http://localhost:8000
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
│   │   └── database.py       # Database utilities
│   ├── main.py               # FastAPI app entry point
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main React component
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

- `POST /api/games` - Create a new game
- `GET /api/games/{game_id}` - Get game state
- `POST /api/games/{game_id}/start` - Start a game
- `POST /api/games/{game_id}/turn/next` - Advance to next turn
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
- Current implementation uses in-memory storage for games (replace with database for production)
- CORS is configured to allow requests from the frontend development server
- TypeScript strict mode is enabled in the frontend

## Future Enhancements

- Database persistence for game state
- WebSocket support for real-time updates
- Full Catan game rules implementation
- LLM agent integration for AI players
- Game history and replay functionality

