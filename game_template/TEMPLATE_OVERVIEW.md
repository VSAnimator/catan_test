# Template Overview

This template captures the complete process for building a board/card game with:
1. Frontend (React + TypeScript)
2. Backend (FastAPI + Python game engine)
3. LLM Agent and Random Agent
4. Automated Test Generation (bug regression testing)
5. Agent Testing (automated agent runners)

## What's Included

### Documentation
- `README.md` - Overview and quick start
- `PROCESS.md` - Detailed step-by-step process
- `GETTING_STARTED.md` - Quick start guide
- `.cursorrules` - Cursor AI rules for development

### Backend Structure

#### Game Engine (`backend/engine/`)
- `engine.py.template` - Core game logic (CUSTOMIZE)
- `serialization.py.template` - State serialization (CUSTOMIZE)
- `__init__.py` - Module exports

#### API Layer (`backend/api/`)
- `routes.py.template` - REST API endpoints (CUSTOMIZE)
- `database.py.template` - Database utilities (CUSTOMIZE)
- `__init__.py` - Module exports

#### Agents (`backend/agents/`)
- `base_agent.py` - Base agent interface (reusable)
- `random_agent.py.template` - Random agent (CUSTOMIZE)
- `agent_runner.py.template` - Agent execution engine (CUSTOMIZE)
- `__init__.py` - Module exports

#### Testing (`backend/bug_tests/`)
- `test_registry.py` - Test case management (reusable)
- `test_runner.py` - Test execution (reusable)
- `llm_validator.py` - LLM validation (reusable)
- `manage_tests.py` - CLI tool (reusable)
- `README.md` - Bug testing documentation (reusable)

#### Scripts (`backend/scripts/`)
- `create_game.py.template` - Game creation script (CUSTOMIZE)
- `run_agents.py.template` - Agent execution script (CUSTOMIZE)
- `__init__.py` - Module exports

#### Tests (`backend/tests/`)
- `test_engine_basic.py.template` - Unit tests (CUSTOMIZE)
- `__init__.py` - Module exports

#### Configuration
- `main.py` - FastAPI app entry point
- `requirements.txt` - Python dependencies

### Frontend Structure

#### Source (`frontend/src/`)
- `App.tsx.template` - Main React component (CUSTOMIZE)
- `App.css.template` - Styles (CUSTOMIZE)
- `api.ts.template` - API client (CUSTOMIZE)
- `main.tsx` - React entry point
- `index.css` - Global styles

#### Configuration
- `package.json` - Node dependencies
- `tsconfig.json` - TypeScript config
- `tsconfig.node.json` - TypeScript node config
- `vite.config.ts` - Vite config
- `index.html` - HTML entry point

### Root Files
- `Makefile` - Development commands
- `.cursorrules` - Cursor AI rules

## File Naming Convention

Files with `.template` extension need to be:
1. Renamed (remove `.template`)
2. Customized for your game

Files without `.template` are:
- Reusable as-is (e.g., `base_agent.py`, bug testing files)
- Configuration files (package.json, requirements.txt, etc.)

## Customization Checklist

### Phase 1: Game Engine
- [ ] `backend/engine/engine.py` - Define game state, actions, rules
- [ ] `backend/engine/serialization.py` - Implement serialization

### Phase 2: API
- [ ] `backend/api/database.py` - Adjust database schema if needed
- [ ] `backend/api/routes.py` - Implement API endpoints

### Phase 3: Frontend
- [ ] `frontend/src/api.ts` - Define API types and functions
- [ ] `frontend/src/App.tsx` - Build game UI
- [ ] `frontend/src/App.css` - Style game UI

### Phase 4: Agents
- [ ] `backend/agents/random_agent.py` - Customize random agent
- [ ] `backend/agents/agent_runner.py` - Adjust agent execution flow
- [ ] `backend/agents/llm_agent.py` - Customize LLM prompts (optional)

### Phase 5: Scripts
- [ ] `backend/scripts/create_game.py` - Implement game creation
- [ ] `backend/scripts/run_agents.py` - Adjust agent execution script

### Phase 6: Tests
- [ ] `backend/tests/test_engine_basic.py` - Write unit tests

## Reusable Components

These components work as-is and don't need customization:

- `backend/agents/base_agent.py` - Base agent interface
- `backend/bug_tests/` - Bug regression testing system
- `backend/main.py` - FastAPI setup
- Configuration files (package.json, requirements.txt, Makefile, etc.)
- Frontend build configuration (vite.config.ts, tsconfig.json, etc.)

## Key Features

### State Restoration
- Games store state at each step
- Tests can restore state from any step
- Reproducible RNG via step-based seeding

### LLM Agent with RAG
- Retrieves similar game states from past games
- Learns from examples and user feedback
- Supports multiple LLM providers (OpenAI, Anthropic, Google, etc.)

### Automated Testing
- Bug regression testing system
- Agent testing scripts
- Batch testing support

## Usage

1. Copy template to new repository
2. Remove `.template` extensions from files
3. Follow customization checklist
4. Read `GETTING_STARTED.md` for detailed steps
5. Refer to `PROCESS.md` for detailed process documentation

## Support

- See `README.md` for overview
- See `PROCESS.md` for detailed process
- See `GETTING_STARTED.md` for quick start
- See `backend/bug_tests/README.md` for bug testing

