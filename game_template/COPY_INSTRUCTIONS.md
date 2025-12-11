# How to Use This Template

## Quick Copy Instructions

1. **Copy the entire `game_template` folder** to your new repository:
   ```bash
   cp -r game_template /path/to/new/repo/
   cd /path/to/new/repo/game_template
   ```

2. **Remove `.template` extensions** from all template files:
   ```bash
   find . -name "*.template" -exec sh -c 'mv "$1" "${1%.template}"' _ {} \;
   ```

3. **Start customizing** following the phases in `PROCESS.md`

## What You Get

### Complete Project Structure
- ✅ Frontend (React + TypeScript + Vite)
- ✅ Backend (FastAPI + Python game engine)
- ✅ Agent system (Base agent, Random agent, LLM agent support)
- ✅ Automated testing (Bug regression testing system)
- ✅ Agent testing scripts
- ✅ Development tools (Makefile, configuration files)

### Documentation
- ✅ `README.md` - Overview and architecture
- ✅ `PROCESS.md` - Detailed step-by-step process
- ✅ `GETTING_STARTED.md` - Quick start guide
- ✅ `TEMPLATE_OVERVIEW.md` - What's included
- ✅ `.cursorrules` - Cursor AI development rules

### Reusable Components
- ✅ Bug regression testing system (works as-is)
- ✅ Base agent interface (works as-is)
- ✅ Database utilities (customize schema if needed)
- ✅ Agent runner framework (customize execution flow)
- ✅ All configuration files (package.json, requirements.txt, etc.)

## Customization Required

Files marked with `.template` need customization:

### Game Engine (Required)
- `backend/engine/engine.py` - Core game logic
- `backend/engine/serialization.py` - State serialization

### API (Required)
- `backend/api/routes.py` - REST API endpoints
- `backend/api/database.py` - Database schema (if needed)

### Frontend (Required)
- `frontend/src/App.tsx` - Game UI
- `frontend/src/api.ts` - API client
- `frontend/src/App.css` - Styles

### Agents (Recommended)
- `backend/agents/random_agent.py` - Random agent behavior
- `backend/agents/agent_runner.py` - Agent execution flow

### Scripts (Recommended)
- `backend/scripts/create_game.py` - Game initialization
- `backend/scripts/run_agents.py` - Agent execution

### Tests (Recommended)
- `backend/tests/test_engine_basic.py` - Unit tests

## Next Steps

1. Read `GETTING_STARTED.md` for quick start
2. Read `PROCESS.md` for detailed process
3. Start with `backend/engine/engine.py` - build your game engine
4. Follow the phases in `PROCESS.md`
5. Use bug testing system as you find issues
6. Test agents by running them automatically

## Tips

- **Start simple**: Build minimal game engine first
- **Test early**: Use bug regression system from the start
- **Use agents**: They'll find bugs you might miss
- **Document rules**: Clear rules help LLM agent play correctly
- **Iterate**: Build, test, fix, repeat

## Example Games You Can Build

- **Card Games**: Love Letter, Coup, The Resistance
- **Board Games**: Ticket to Ride, Carcassonne, Splendor
- **Abstract Games**: Chess, Go, Checkers
- **Your Own Game**: Any turn-based game with discrete actions

The template is designed to work with any game that has:
- Discrete game states
- Turn-based gameplay
- Clear action space
- Win/loss conditions

