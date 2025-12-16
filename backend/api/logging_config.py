"""
Structured logging configuration for the application.
"""
import structlog
import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict


def configure_logging(environment: str = "development"):
    """Configure structured logging based on environment."""
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO if environment == "production" else logging.DEBUG,
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if environment == "production" else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.INFO if environment == "production" else logging.DEBUG
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def get_logger(name: str = None):
    """Get a configured logger instance."""
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


class ActivityLogger:
    """Logger for user activity tracking."""
    
    def __init__(self):
        self.logger = get_logger("activity")
    
    def log_user_action(
        self,
        user_id: str,
        username: str,
        action: str,
        details: Dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log a user action."""
        self.logger.info(
            "user_action",
            user_id=user_id,
            username=username,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_game_action(
        self,
        game_id: str,
        user_id: str,
        player_id: str,
        action: str,
        details: Dict[str, Any] = None
    ):
        """Log a game action."""
        self.logger.info(
            "game_action",
            game_id=game_id,
            user_id=user_id,
            player_id=player_id,
            action=action,
            details=details or {},
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_websocket_event(
        self,
        event_type: str,
        game_id: str,
        user_id: str = None,
        details: Dict[str, Any] = None
    ):
        """Log a WebSocket event."""
        self.logger.info(
            "websocket_event",
            event_type=event_type,
            game_id=game_id,
            user_id=user_id,
            details=details or {},
            timestamp=datetime.utcnow().isoformat()
        )


# Global activity logger instance
activity_logger = ActivityLogger()

