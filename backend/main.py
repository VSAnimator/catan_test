from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.database import init_db
import os
from pathlib import Path

# Load environment variables from ~/.zshrc for all API keys
# This ensures keys are available even if uvicorn workers don't inherit them from the shell
zshrc_path = Path.home() / ".zshrc"
api_keys_to_load = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
loaded_keys = {}

if zshrc_path.exists():
    try:
        with open(zshrc_path, "r") as f:
            for line in f:
                line = line.strip()
                for key_name in api_keys_to_load:
                    if line.startswith(f"export {key_name}="):
                        # Extract the key value
                        key_value = line.split("=", 1)[1].strip()
                        # Remove quotes if present
                        key_value = key_value.strip('"').strip("'")
                        # Validate key format before setting
                        if key_name == "ANTHROPIC_API_KEY" and not key_value.startswith("sk-ant-api"):
                            print(f"[main.py] WARNING: Skipping invalid ANTHROPIC_API_KEY format (starts with: {key_value[:20]}...)", flush=True)
                            continue
                        elif key_name == "OPENAI_API_KEY" and not key_value.startswith("sk-"):
                            print(f"[main.py] WARNING: Skipping invalid OPENAI_API_KEY format", flush=True)
                            continue
                        # Set the environment variable
                        os.environ[key_name] = key_value
                        loaded_keys[key_name] = len(key_value)
                        print(f"[main.py] Loaded {key_name} from ~/.zshrc (length: {len(key_value)})", flush=True)
                        break
    except Exception as e:
        print(f"[main.py] Warning: Could not load API keys from ~/.zshrc: {e}", flush=True)

# Log API key status on startup
for key_name in api_keys_to_load:
    key_value = os.getenv(key_name)
    if key_value:
        print(f"[main.py] {key_name} is available (length: {len(key_value)})", flush=True)
    else:
        print(f"[main.py] WARNING: {key_name} is NOT available", flush=True)

# Initialize database on startup
init_db()

app = FastAPI(title="Catan Game API", version="1.0.0")

# CORS middleware for frontend communication
# Allow all localhost ports for development (Vite may use different ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Catan Game API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

