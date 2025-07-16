"""
WebSocket manager for real-time communication between server and clients.
"""

from typing import List, Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.surveillance_connections: Set[WebSocket] = set()
        self.alert_subscribers: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket, client_type: str = "general"):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if client_type == "surveillance":
            self.surveillance_connections.add(websocket)
        elif client_type == "alerts":
            self.alert_subscribers.add(websocket)
        
        logger.info(f"New {client_type} connection. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.surveillance_connections:
            self.surveillance_connections.discard(websocket)
        if websocket in self.alert_subscribers:
            self.alert_subscribers.discard(websocket)
        
        logger.info(f"Connection closed. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def broadcast(self, message: Dict, connection_type: str = "all"):
        """Broadcast a message to multiple clients."""
        message_str = json.dumps(message)
        
        if connection_type == "all":
            connections = self.active_connections
        elif connection_type == "surveillance":
            connections = self.surveillance_connections
        elif connection_type == "alerts":
            connections = self.alert_subscribers
        else:
            connections = []
        
        # Send to all relevant connections
        disconnected = []
        for connection in connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_detection_update(self, detection_data: Dict):
        """Send detection update to surveillance connections."""
        message = {
            "type": "detection",
            "timestamp": datetime.now().isoformat(),
            "data": detection_data
        }
        await self.broadcast(message, "surveillance")
    
    async def send_anomaly_alert(self, anomaly_data: Dict):
        """Send anomaly alert to subscribers."""
        message = {
            "type": "anomaly_alert",
            "timestamp": datetime.now().isoformat(),
            "severity": anomaly_data.get("severity", "medium"),
            "data": anomaly_data
        }
        await self.broadcast(message, "alerts")
    
    async def send_vehicle_update(self, vehicle_data: Dict):
        """Send vehicle profile update."""
        message = {
            "type": "vehicle_update",
            "timestamp": datetime.now().isoformat(),
            "data": vehicle_data
        }
        await self.broadcast(message, "all")
    
    def get_connection_stats(self) -> Dict:
        """Get statistics about current connections."""
        return {
            "total_connections": len(self.active_connections),
            "surveillance_connections": len(self.surveillance_connections),
            "alert_subscribers": len(self.alert_subscribers)
        }

# Global instance
manager = ConnectionManager()