import os
import time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import make_asgi_app
from api.routes import router
from api.auth_routes import router as auth_router
from api.room_routes import router as room_router
from api.websocket_routes import router as websocket_router
from api.database import init_db
from api.logging_config import configure_logging, get_logger
from api.monitoring import (
    http_requests_total,
    http_request_duration_seconds,
    track_performance
)
from api.security import init_security_tables, require_https, get_client_ip
from api.csrf import CSRFProtectionMiddleware

# Load environment variables
load_dotenv()

# Configure logging
environment = os.getenv("ENVIRONMENT", "development")
logger = configure_logging(environment)

# Initialize Sentry for error tracking
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=0.1 if environment == "production" else 1.0,
        environment=environment,
    )
    logger.info("sentry_initialized", environment=environment)

# Initialize database on startup
init_db()
init_security_tables()
logger.info("database_initialized")

# Rate limiting - using in-memory storage (for production, use Redis)
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
app = FastAPI(title="Catan Game API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Rate limiting is configured via slowapi and will be enforced per-endpoint
# Auth endpoints have stricter limits (5-10/minute), other endpoints use default limits

# CORS middleware for frontend communication
# Get allowed origins from environment variable, with fallback to localhost for development
cors_origins_str = os.getenv("CORS_ORIGINS", "")
if cors_origins_str:
    # Parse comma-separated list from environment variable
    allow_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
else:
    # Default to localhost ports for development
    allow_origins = [
        "http://localhost",
        "http://localhost:80",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:80",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSRF protection (only in production or when enabled)
if os.getenv("ENABLE_CSRF", "false").lower() == "true":
    app.add_middleware(CSRFProtectionMiddleware)

# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # HTTPS enforcement in production
    if environment == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        require_https(request)
    
    return response

# Request logging and metrics middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and track metrics."""
    start_time = time.time()
    client_ip = get_client_ip(request)
    
    # Skip logging for health checks and metrics
    if request.url.path in ["/health", "/metrics", "/"]:
        response = await call_next(request)
        return response
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Track metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Log request
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            client_ip=client_ip,
            user_agent=request.headers.get("user-agent")
        )
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "http_request_error",
            method=request.method,
            path=request.url.path,
            duration=duration,
            error=str(e),
            client_ip=client_ip
        )
        raise

# Initialize rate limiter in auth routes
from api.auth_routes import init_rate_limiter
init_rate_limiter(limiter)

app.include_router(router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(room_router, prefix="/api")
app.include_router(websocket_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Catan Game API", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "environment": environment}

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("application_started", environment=environment)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("application_shutdown")

