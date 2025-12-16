"""
WebSocket routes for real-time game communication.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Optional
import json

from .websocket_manager import connection_manager
from .auth import get_current_user, verify_token
from .game_rooms import room_manager
from .database import get_latest_state, save_game_state, add_step, get_game as get_game_from_db
from .logging_config import get_logger, activity_logger
from .monitoring import websocket_connections
from engine import deserialize_game_state, serialize_game_state, legal_actions
import random

logger = get_logger("websocket")

router = APIRouter()


@router.websocket("/ws/game/{game_id}")
async def websocket_game_endpoint(websocket: WebSocket, game_id: str, token: Optional[str] = Query(None)):
    """WebSocket endpoint for game updates."""
    user_id = None
    user = None
    
    try:
        # Try to authenticate if token provided
        if token:
            try:
                user = get_current_user(token)
                if user:
                    user_id = user.id
            except Exception as e:
                print(f"WebSocket auth error: {e}")
        
        # Connect to game
        await connection_manager.connect(websocket, game_id, user_id)
        websocket_connections.labels(game_id=game_id).inc()
        logger.info("websocket_connected", game_id=game_id, user_id=user_id)
        activity_logger.log_websocket_event("connected", game_id, user_id)
        
        # Send initial game state (restore from database on reconnect)
        try:
            logger.debug("fetching_game_state", game_id=game_id)
            state_json = get_latest_state(game_id)
            if state_json:
                logger.debug("sending_initial_state", game_id=game_id, state_size=len(str(state_json)))
                await connection_manager.send_personal_message({
                    "type": "game_state",
                    "data": state_json
                }, websocket)
                logger.info("initial_state_sent", game_id=game_id)
                activity_logger.log_websocket_event("state_restored", game_id, user_id, {"restored": True})
            else:
                logger.warning("game_not_found", game_id=game_id)
                await connection_manager.send_personal_message({
                    "type": "error",
                    "message": "Game not found"
                }, websocket)
                await connection_manager.disconnect(websocket)
                websocket_connections.labels(game_id=game_id).dec()
                return
        except Exception as e:
            logger.error("error_sending_initial_state", game_id=game_id, error=str(e))
            import traceback
            traceback.print_exc()
            try:
                await connection_manager.send_personal_message({
                    "type": "error",
                    "message": f"Error loading game state: {str(e)}"
                }, websocket)
            except:
                pass
            await connection_manager.disconnect(websocket)
            return
        
        # Listen for messages
        while True:
            try:
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                if message_type == "ping":
                    # Heartbeat
                    await connection_manager.send_personal_message({
                        "type": "pong"
                    }, websocket)
                
                elif message_type == "get_state":
                    # Send current game state
                    state_json = get_latest_state(game_id)
                    if state_json:
                        await connection_manager.send_personal_message({
                            "type": "game_state",
                            "data": state_json
                        }, websocket)
                
                elif message_type == "action":
                    # Handle game action (will be implemented with proper validation)
                    await connection_manager.send_personal_message({
                        "type": "error",
                        "message": "Actions should be sent via REST API"
                    }, websocket)
                
                else:
                    await connection_manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }, websocket)
            except WebSocketDisconnect:
                logger.info("websocket_disconnected", game_id=game_id, user_id=user_id)
                activity_logger.log_websocket_event("disconnected", game_id, user_id)
                break
            except Exception as e:
                logger.error("websocket_message_error", game_id=game_id, error=str(e))
                import traceback
                traceback.print_exc()
                try:
                    await connection_manager.send_personal_message({
                        "type": "error",
                        "message": f"Error processing message: {str(e)}"
                    }, websocket)
                except:
                    pass
                break
    
    except WebSocketDisconnect:
        logger.info("websocket_disconnected_outer", game_id=game_id)
    except Exception as e:
        logger.error("websocket_error_outer", game_id=game_id, error=str(e))
        import traceback
        traceback.print_exc()
    finally:
        await connection_manager.disconnect(websocket)
        websocket_connections.labels(game_id=game_id).dec()
        logger.info("websocket_cleanup", game_id=game_id)


async def broadcast_game_state_update(game_id: str, state_json: dict):
    """Broadcast game state update to all connected clients."""
    await connection_manager.broadcast_to_game(game_id, {
        "type": "game_state_update",
        "data": state_json
    })


async def broadcast_game_event(game_id: str, event_type: str, event_data: dict):
    """Broadcast a game event to all connected clients."""
    await connection_manager.broadcast_to_game(game_id, {
        "type": "game_event",
        "event_type": event_type,
        "data": event_data
    })

