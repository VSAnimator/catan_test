"""
WebSocket connection manager for real-time game updates.
"""
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from datetime import datetime


class ConnectionManager:
    """Manages WebSocket connections for games."""
    
    def __init__(self):
        # game_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # WebSocket -> (game_id, user_id)
        self.connection_info: Dict[WebSocket, tuple[str, Optional[str]]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, game_id: str, user_id: Optional[str] = None):
        """Connect a WebSocket to a game."""
        await websocket.accept()
        
        async with self._lock:
            if game_id not in self.active_connections:
                self.active_connections[game_id] = set()
            self.active_connections[game_id].add(websocket)
            self.connection_info[websocket] = (game_id, user_id)
    
    async def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket from a game."""
        async with self._lock:
            info = self.connection_info.get(websocket)
            if info:
                game_id, _ = info
                if game_id in self.active_connections:
                    self.active_connections[game_id].discard(websocket)
                    if not self.active_connections[game_id]:
                        del self.active_connections[game_id]
                del self.connection_info[websocket]
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending personal message: {e}")
            await self.disconnect(websocket)
    
    async def broadcast_to_game(self, game_id: str, message: dict, exclude: Optional[WebSocket] = None):
        """Broadcast a message to all connections in a game."""
        async with self._lock:
            connections = self.active_connections.get(game_id, set()).copy()
        
        disconnected = []
        for connection in connections:
            if connection == exclude:
                continue
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            await self.disconnect(conn)
    
    def get_game_connections(self, game_id: str) -> Set[WebSocket]:
        """Get all active connections for a game."""
        return self.active_connections.get(game_id, set()).copy()
    
    def get_connection_count(self, game_id: str) -> int:
        """Get the number of active connections for a game."""
        return len(self.active_connections.get(game_id, set()))


# Global connection manager instance
connection_manager = ConnectionManager()

