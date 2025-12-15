from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.auth_routes import router as auth_router
from api.room_routes import router as room_router
from api.websocket_routes import router as websocket_router
from api.database import init_db

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
app.include_router(auth_router, prefix="/api")
app.include_router(room_router, prefix="/api")
app.include_router(websocket_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Catan Game API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

