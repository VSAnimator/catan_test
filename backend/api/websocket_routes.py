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
from engine import deserialize_game_state, serialize_game_state, legal_actions
import random

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
        print(f"WebSocket connected for game {game_id}, user {user_id}")
        
        # Send initial game state
        try:
            print(f"Fetching game state for {game_id}")
            state_json = get_latest_state(game_id)
            print(f"Got game state: {state_json is not None}")
            if state_json:
                print(f"Sending initial game state, size: {len(str(state_json))}")
                await connection_manager.send_personal_message({
                    "type": "game_state",
                    "data": state_json
                }, websocket)
                print("Initial game state sent successfully")
            else:
                print(f"Game {game_id} not found")
                await connection_manager.send_personal_message({
                    "type": "error",
                    "message": "Game not found"
                }, websocket)
                await connection_manager.disconnect(websocket)
                return
        except Exception as e:
            print(f"Error sending initial state: {e}")
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
                print(f"WebSocket disconnected for game {game_id}")
                break
            except Exception as e:
                print(f"Error processing WebSocket message: {e}")
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
        print(f"WebSocket disconnected (outer) for game {game_id}")
    except Exception as e:
        print(f"WebSocket error (outer): {e}")
        import traceback
        traceback.print_exc()
    finally:
        await connection_manager.disconnect(websocket)


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

