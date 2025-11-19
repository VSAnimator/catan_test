# Backend Development Guide

## ⚠️ CRITICAL: Using the UV Environment

This backend uses `uv` for package management with a virtual environment located at `backend/.venv/`.

### Rules for All Operations:

1. **Always use `uv pip` (NOT `pip`)** for installing packages
   - ✅ Correct: `uv pip install <package>`
   - ❌ Wrong: `pip install <package>`

2. **Always activate the virtual environment** before running any Python commands
   ```bash
   cd backend
   source .venv/bin/activate
   ```

3. **Examples:**
   - Installing packages: `source .venv/bin/activate && uv pip install <package>`
   - Running tests: `source .venv/bin/activate && python -m pytest`
   - Running server: `source .venv/bin/activate && uvicorn main:app --reload`

### Quick Reference

```bash
# Install dependencies
cd backend && source .venv/bin/activate && uv pip install -r requirements.txt

# Run tests
cd backend && source .venv/bin/activate && python -m pytest

# Run server
cd backend && source .venv/bin/activate && uvicorn main:app --reload
```

**Remember:** The virtual environment path is `backend/.venv/` - always activate it first!

