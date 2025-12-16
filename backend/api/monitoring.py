"""
Monitoring and metrics collection for the application.
"""
import time
from functools import wraps
from typing import Callable, Any
from prometheus_client import Counter, Histogram, Gauge
from .logging_config import get_logger

logger = get_logger("monitoring")

# Prometheus metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

websocket_connections = Gauge(
    'websocket_connections_total',
    'Current WebSocket connections',
    ['game_id']
)

active_games = Gauge(
    'active_games_total',
    'Number of active games'
)

user_sessions = Gauge(
    'user_sessions_total',
    'Number of active user sessions'
)

database_operations = Counter(
    'database_operations_total',
    'Database operations',
    ['operation', 'table']
)

database_operation_duration = Histogram(
    'database_operation_duration_seconds',
    'Database operation duration in seconds',
    ['operation', 'table']
)


def track_performance(func: Callable) -> Callable:
    """Decorator to track function performance."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(
                "function_performance",
                function=func.__name__,
                duration=duration,
                status="success"
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "function_performance",
                function=func.__name__,
                duration=duration,
                status="error",
                error=str(e)
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(
                "function_performance",
                function=func.__name__,
                duration=duration,
                status="success"
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "function_performance",
                function=func.__name__,
                duration=duration,
                status="error",
                error=str(e)
            )
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def track_database_operation(operation: str, table: str):
    """Decorator to track database operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                database_operations.labels(operation=operation, table=table).inc()
                database_operation_duration.labels(operation=operation, table=table).observe(duration)
                return result
            except Exception as e:
                database_operations.labels(operation=operation, table=table).inc()
                logger.error(
                    "database_operation_error",
                    operation=operation,
                    table=table,
                    error=str(e)
                )
                raise
        return wrapper
    return decorator

