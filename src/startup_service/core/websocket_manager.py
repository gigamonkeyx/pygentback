"""
WebSocket Manager for PyGent Factory Startup Service
Real-time communication for startup progress, logs, and system metrics.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
import uuid

from ..models.schemas import (
    WebSocketMessage, StartupProgressMessage, SystemMetricsMessage, LogMessage
)
from ..utils.logging_config import websocket_logger


class WebSocketConnection:
    """Represents a WebSocket connection with metadata."""
    
    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.connection_id = connection_id
        self.connected_at = datetime.utcnow()
        self.subscriptions: Set[str] = set()
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.last_ping = datetime.utcnow()
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to this connection."""
        try:
            await self.websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            raise ConnectionError(f"Failed to send message: {e}")
    
    def subscribe(self, channel: str):
        """Subscribe to a channel."""
        self.subscriptions.add(channel)
    
    def unsubscribe(self, channel: str):
        """Unsubscribe from a channel."""
        self.subscriptions.discard(channel)
    
    def is_subscribed(self, channel: str) -> bool:
        """Check if subscribed to a channel."""
        return channel in self.subscriptions


class WebSocketManager:
    """Manages WebSocket connections and real-time communication."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.channels: Dict[str, Set[str]] = {}  # channel -> set of connection_ids
        self.logger = websocket_logger
        
        # Background tasks
        self._ping_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.ping_interval = 30  # seconds
        self.connection_timeout = 300  # seconds
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self._ping_task = asyncio.create_task(self._ping_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def connect(self, websocket: WebSocket, user_id: str = None, session_id: str = None) -> str:
        """Accept a new WebSocket connection."""
        try:
            await websocket.accept()
            
            connection_id = str(uuid.uuid4())
            connection = WebSocketConnection(websocket, connection_id)
            connection.user_id = user_id
            connection.session_id = session_id
            
            self.connections[connection_id] = connection
            
            # Send welcome message
            welcome_message = {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat(),
                "server_info": {
                    "version": "1.0.0",
                    "features": ["startup_progress", "system_metrics", "log_streaming"]
                }
            }
            
            await connection.send_message(welcome_message)
            
            self.logger.info(f"WebSocket connection established: {connection_id}")
            return connection_id
            
        except Exception as e:
            self.logger.error(f"Failed to establish WebSocket connection: {e}")
            raise
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        try:
            # Find connection by websocket
            connection_id = None
            for conn_id, conn in self.connections.items():
                if conn.websocket == websocket:
                    connection_id = conn_id
                    break
            
            if connection_id:
                await self._remove_connection(connection_id)
                self.logger.info(f"WebSocket connection disconnected: {connection_id}")
            
        except Exception as e:
            self.logger.error(f"Error during WebSocket disconnection: {e}")
    
    async def disconnect_all(self):
        """Disconnect all WebSocket connections."""
        try:
            connection_ids = list(self.connections.keys())
            for connection_id in connection_ids:
                await self._remove_connection(connection_id)
            
            # Cancel background tasks
            if self._ping_task:
                self._ping_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            self.logger.info("All WebSocket connections disconnected")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting all connections: {e}")
    
    async def _remove_connection(self, connection_id: str):
        """Remove a connection and clean up subscriptions."""
        try:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                
                # Remove from all channels
                for channel in list(connection.subscriptions):
                    await self._unsubscribe_from_channel(connection_id, channel)
                
                # Close WebSocket if still open
                try:
                    await connection.websocket.close()
                except:
                    pass
                
                # Remove from connections
                del self.connections[connection_id]
                
        except Exception as e:
            self.logger.error(f"Error removing connection {connection_id}: {e}")
    
    async def subscribe_to_channel(self, connection_id: str, channel: str) -> bool:
        """Subscribe a connection to a channel."""
        try:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            connection.subscribe(channel)
            
            # Add to channel
            if channel not in self.channels:
                self.channels[channel] = set()
            self.channels[channel].add(connection_id)
            
            self.logger.debug(f"Connection {connection_id} subscribed to channel: {channel}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to channel: {e}")
            return False
    
    async def _unsubscribe_from_channel(self, connection_id: str, channel: str):
        """Unsubscribe a connection from a channel."""
        try:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                connection.unsubscribe(channel)
            
            # Remove from channel
            if channel in self.channels:
                self.channels[channel].discard(connection_id)
                
                # Clean up empty channels
                if not self.channels[channel]:
                    del self.channels[channel]
            
            self.logger.debug(f"Connection {connection_id} unsubscribed from channel: {channel}")
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from channel: {e}")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific connection."""
        try:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            await connection.send_message(message)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message to connection {connection_id}: {e}")
            return False
    
    async def send_to_client(self, websocket: WebSocket, message: Dict[str, Any]) -> bool:
        """Send message to a specific WebSocket client."""
        try:
            # Find connection by websocket
            for connection in self.connections.values():
                if connection.websocket == websocket:
                    await connection.send_message(message)
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to send message to client: {e}")
            return False
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]) -> int:
        """Broadcast message to all connections in a channel."""
        try:
            if channel not in self.channels:
                return 0
            
            sent_count = 0
            failed_connections = []
            
            for connection_id in self.channels[channel].copy():
                try:
                    if await self.send_to_connection(connection_id, message):
                        sent_count += 1
                    else:
                        failed_connections.append(connection_id)
                except Exception as e:
                    self.logger.warning(f"Failed to send to connection {connection_id}: {e}")
                    failed_connections.append(connection_id)
            
            # Clean up failed connections
            for connection_id in failed_connections:
                await self._remove_connection(connection_id)
            
            self.logger.debug(f"Broadcast to channel {channel}: {sent_count} messages sent")
            return sent_count
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast to channel {channel}: {e}")
            return 0
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected clients."""
        try:
            sent_count = 0
            failed_connections = []
            
            for connection_id in list(self.connections.keys()):
                try:
                    if await self.send_to_connection(connection_id, message):
                        sent_count += 1
                    else:
                        failed_connections.append(connection_id)
                except Exception as e:
                    self.logger.warning(f"Failed to send to connection {connection_id}: {e}")
                    failed_connections.append(connection_id)
            
            # Clean up failed connections
            for connection_id in failed_connections:
                await self._remove_connection(connection_id)
            
            self.logger.debug(f"Broadcast to all: {sent_count} messages sent")
            return sent_count
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast to all: {e}")
            return 0
    
    async def send_startup_progress(self, sequence_id: str, service: str, status: str, progress_percent: float, message: str, details: Dict[str, Any] = None):
        """Send startup progress update."""
        try:
            progress_message = StartupProgressMessage(
                sequence_id=sequence_id,
                service=service,
                status=status,
                progress_percent=progress_percent,
                message=message,
                details=details or {}
            )
            
            # Send to startup progress channel
            await self.broadcast_to_channel("startup_progress", progress_message.dict())
            
            # Send to sequence-specific channel
            sequence_channel = f"sequence_{sequence_id}"
            await self.broadcast_to_channel(sequence_channel, progress_message.dict())
            
        except Exception as e:
            self.logger.error(f"Failed to send startup progress: {e}")
    
    async def send_system_metrics(self, metrics: Dict[str, Any], services_status: Dict[str, str]):
        """Send system metrics update."""
        try:
            metrics_message = SystemMetricsMessage(
                metrics=metrics,
                services_status=services_status
            )
            
            await self.broadcast_to_channel("system_metrics", metrics_message.dict())
            
        except Exception as e:
            self.logger.error(f"Failed to send system metrics: {e}")
    
    async def send_log_message(self, level: str, logger: str, message: str, service: str = None, details: Dict[str, Any] = None):
        """Send log message."""
        try:
            log_message = LogMessage(
                level=level,
                logger=logger,
                message=message,
                service=service,
                details=details or {}
            )
            
            # Send to logs channel
            await self.broadcast_to_channel("logs", log_message.dict())
            
            # Send to service-specific log channel if service specified
            if service:
                service_channel = f"logs_{service}"
                await self.broadcast_to_channel(service_channel, log_message.dict())
            
        except Exception as e:
            self.logger.error(f"Failed to send log message: {e}")
    
    async def handle_client_message(self, connection_id: str, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe":
                channel = data.get("channel")
                if channel:
                    await self.subscribe_to_channel(connection_id, channel)
                    
                    # Send confirmation
                    response = {
                        "type": "subscription_confirmed",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await self.send_to_connection(connection_id, response)
            
            elif message_type == "unsubscribe":
                channel = data.get("channel")
                if channel:
                    await self._unsubscribe_from_channel(connection_id, channel)
                    
                    # Send confirmation
                    response = {
                        "type": "unsubscription_confirmed",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await self.send_to_connection(connection_id, response)
            
            elif message_type == "ping":
                # Update last ping time
                if connection_id in self.connections:
                    self.connections[connection_id].last_ping = datetime.utcnow()
                
                # Send pong response
                response = {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await self.send_to_connection(connection_id, response)
            
            else:
                self.logger.warning(f"Unknown message type from {connection_id}: {message_type}")
            
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON from connection {connection_id}: {message}")
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
    
    async def _ping_loop(self):
        """Background task to ping connections and check health."""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection in self.connections.items():
                    try:
                        # Check if connection is stale
                        time_since_ping = (current_time - connection.last_ping).total_seconds()
                        
                        if time_since_ping > self.connection_timeout:
                            stale_connections.append(connection_id)
                        else:
                            # Send ping
                            ping_message = {
                                "type": "ping",
                                "timestamp": current_time.isoformat()
                            }
                            await connection.send_message(ping_message)
                            
                    except Exception as e:
                        self.logger.warning(f"Ping failed for connection {connection_id}: {e}")
                        stale_connections.append(connection_id)
                
                # Remove stale connections
                for connection_id in stale_connections:
                    await self._remove_connection(connection_id)
                    self.logger.info(f"Removed stale connection: {connection_id}")
                
            except Exception as e:
                self.logger.error(f"Error in ping loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task for general cleanup."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up empty channels
                empty_channels = [channel for channel, connections in self.channels.items() if not connections]
                for channel in empty_channels:
                    del self.channels[channel]
                
                # Log connection statistics
                self.logger.info(f"WebSocket stats - Connections: {len(self.connections)}, Channels: {len(self.channels)}")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.connections)
    
    def get_channel_count(self) -> int:
        """Get total number of active channels."""
        return len(self.channels)
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific connection."""
        if connection_id not in self.connections:
            return None
        
        connection = self.connections[connection_id]
        return {
            "connection_id": connection_id,
            "connected_at": connection.connected_at.isoformat(),
            "user_id": connection.user_id,
            "session_id": connection.session_id,
            "subscriptions": list(connection.subscriptions),
            "last_ping": connection.last_ping.isoformat()
        }
