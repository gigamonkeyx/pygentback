"""
Protocol Manager

This module provides the main interface for managing multiple communication protocols
and coordinating message routing across different protocol implementations.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta

from .base import (
    CommunicationProtocol, ProtocolMessage, ProtocolType, ProtocolStats,
    BaseCommunicationProtocol, ProtocolError, MessagePriority, DeliveryMode
)


logger = logging.getLogger(__name__)


class ProtocolManager:
    """
    Main manager for communication protocols.
    
    Coordinates between different protocol implementations, handles message routing,
    and provides a unified interface for all communication operations.
    """
    
    def __init__(self, settings=None):
        """
        Initialize the protocol manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Protocol registry
        self.protocols: Dict[ProtocolType, CommunicationProtocol] = {}
        self.protocol_instances: Dict[str, CommunicationProtocol] = {}  # name -> protocol
        
        # Message routing
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.route_table: Dict[str, ProtocolType] = {}  # recipient -> protocol_type
        
        # Message queues for different priorities
        self.message_queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in MessagePriority
        }
        
        # State management
        self._running = False
        self._message_processor_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.global_stats = {
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_messages_failed": 0,
            "protocols_registered": 0,
            "active_connections": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the protocol manager"""
        try:
            self._running = True
            
            # Start message processor
            self._message_processor_task = asyncio.create_task(self._process_messages())
            
            # Start health monitor
            self._health_monitor_task = asyncio.create_task(self._monitor_health())
            
            logger.info("Protocol manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize protocol manager: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the protocol manager"""
        try:
            self._running = False
            
            # Cancel background tasks
            if self._message_processor_task:
                self._message_processor_task.cancel()
                try:
                    await self._message_processor_task
                except asyncio.CancelledError:
                    pass
            
            if self._health_monitor_task:
                self._health_monitor_task.cancel()
                try:
                    await self._health_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all protocols
            for protocol in self.protocols.values():
                await protocol.shutdown()
            
            for protocol in self.protocol_instances.values():
                await protocol.shutdown()
            
            logger.info("Protocol manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during protocol manager shutdown: {str(e)}")
    
    async def register_protocol(self, protocol_type: ProtocolType, 
                               protocol: CommunicationProtocol,
                               name: Optional[str] = None) -> str:
        """
        Register a communication protocol.
        
        Args:
            protocol_type: Type of protocol
            protocol: Protocol implementation
            name: Optional custom name for the protocol
            
        Returns:
            str: Protocol instance name
        """
        try:
            # Initialize protocol
            await protocol.initialize()
            
            # Register by type (primary protocol for this type)
            self.protocols[protocol_type] = protocol
            
            # Register by name (for multiple instances of same type)
            instance_name = name or f"{protocol_type.value}_{len(self.protocol_instances)}"
            self.protocol_instances[instance_name] = protocol
            
            # Update statistics
            self.global_stats["protocols_registered"] += 1
            if protocol.is_connected():
                self.global_stats["active_connections"] += 1
            
            logger.info(f"Registered protocol: {instance_name} ({protocol_type.value})")
            return instance_name
            
        except Exception as e:
            logger.error(f"Failed to register protocol {protocol_type.value}: {str(e)}")
            raise
    
    async def unregister_protocol(self, name: str) -> bool:
        """
        Unregister a communication protocol.
        
        Args:
            name: Protocol instance name
            
        Returns:
            bool: True if successful
        """
        try:
            if name not in self.protocol_instances:
                logger.warning(f"Protocol {name} not found")
                return False
            
            protocol = self.protocol_instances[name]
            
            # Shutdown protocol
            await protocol.shutdown()
            
            # Remove from registries
            del self.protocol_instances[name]
            
            # Remove from type registry if it's the primary
            for protocol_type, registered_protocol in list(self.protocols.items()):
                if registered_protocol is protocol:
                    del self.protocols[protocol_type]
                    break
            
            # Update statistics
            self.global_stats["protocols_registered"] -= 1
            
            logger.info(f"Unregistered protocol: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister protocol {name}: {str(e)}")
            return False
    
    async def send_message(self, message: ProtocolMessage, 
                          protocol_name: Optional[str] = None) -> bool:
        """
        Send a message through the appropriate protocol.
        
        Args:
            message: Message to send
            protocol_name: Optional specific protocol to use
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            # Determine which protocol to use
            protocol = self._select_protocol(message, protocol_name)
            if not protocol:
                logger.error(f"No suitable protocol found for message {message.id}")
                return False
            
            # Handle different delivery modes
            if message.delivery_mode == DeliveryMode.FIRE_AND_FORGET:
                # Send immediately
                success = await protocol.send_message(message)
            elif message.delivery_mode == DeliveryMode.AT_LEAST_ONCE:
                # Queue for reliable delivery
                await self._queue_message(message)
                success = True
            elif message.delivery_mode == DeliveryMode.REQUEST_RESPONSE:
                # Send and wait for response
                success = await self._send_with_response(message, protocol)
            else:
                # Default to immediate send
                success = await protocol.send_message(message)
            
            # Update global statistics
            if success:
                self.global_stats["total_messages_sent"] += 1
            else:
                self.global_stats["total_messages_failed"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message {message.id}: {str(e)}")
            self.global_stats["total_messages_failed"] += 1
            return False
    
    async def broadcast_message(self, message: ProtocolMessage,
                               protocol_types: Optional[List[ProtocolType]] = None,
                               exclude_protocols: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Broadcast a message to multiple protocols.
        
        Args:
            message: Message to broadcast
            protocol_types: Specific protocol types to target
            exclude_protocols: Protocol names to exclude
            
        Returns:
            Dict[str, bool]: Results by protocol name
        """
        results = {}
        exclude_set = set(exclude_protocols or [])
        
        # Determine target protocols
        if protocol_types:
            target_protocols = {
                name: protocol for name, protocol in self.protocol_instances.items()
                if any(protocol.protocol_type == pt for pt in protocol_types)
                and name not in exclude_set
            }
        else:
            target_protocols = {
                name: protocol for name, protocol in self.protocol_instances.items()
                if name not in exclude_set
            }
        
        # Send to all target protocols
        for name, protocol in target_protocols.items():
            try:
                # Create a copy of the message for each protocol
                message_copy = ProtocolMessage.from_dict(message.to_dict())
                message_copy.protocol = protocol.protocol_type
                
                success = await protocol.send_message(message_copy)
                results[name] = success
                
            except Exception as e:
                logger.error(f"Failed to broadcast to protocol {name}: {str(e)}")
                results[name] = False
        
        return results
    
    def add_route(self, recipient: str, protocol_type: ProtocolType) -> None:
        """Add a routing rule for a specific recipient"""
        self.route_table[recipient] = protocol_type
        logger.debug(f"Added route: {recipient} -> {protocol_type.value}")
    
    def remove_route(self, recipient: str) -> None:
        """Remove a routing rule"""
        if recipient in self.route_table:
            del self.route_table[recipient]
            logger.debug(f"Removed route for: {recipient}")
    
    def add_message_handler(self, message_type: str, handler: Callable) -> None:
        """Add a global message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def remove_message_handler(self, message_type: str, handler: Callable) -> None:
        """Remove a global message handler"""
        if message_type in self.message_handlers:
            if handler in self.message_handlers[message_type]:
                self.message_handlers[message_type].remove(handler)
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all protocols"""
        protocol_stats = {}
        
        for name, protocol in self.protocol_instances.items():
            try:
                stats = await protocol.get_stats()
                protocol_stats[name] = stats.to_dict()
            except Exception as e:
                logger.error(f"Failed to get stats for protocol {name}: {str(e)}")
                protocol_stats[name] = {"error": str(e)}
        
        return {
            "global": self.global_stats.copy(),
            "protocols": protocol_stats,
            "active_protocols": len([p for p in self.protocol_instances.values() if p.is_connected()]),
            "message_queue_sizes": {
                priority.value: queue.qsize() 
                for priority, queue in self.message_queues.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all protocols"""
        health = {
            "manager_status": "healthy" if self._running else "stopped",
            "protocols": {},
            "overall_status": "healthy"
        }
        
        unhealthy_count = 0
        
        for name, protocol in self.protocol_instances.items():
            try:
                is_connected = protocol.is_connected()
                stats = await protocol.get_stats()
                
                protocol_health = {
                    "connected": is_connected,
                    "last_activity": stats.last_activity.isoformat() if stats.last_activity else None,
                    "messages_sent": stats.messages_sent,
                    "messages_received": stats.messages_received,
                    "messages_failed": stats.messages_failed,
                    "status": "healthy" if is_connected else "disconnected"
                }
                
                if not is_connected:
                    unhealthy_count += 1
                    protocol_health["status"] = "unhealthy"
                
                health["protocols"][name] = protocol_health
                
            except Exception as e:
                unhealthy_count += 1
                health["protocols"][name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Determine overall status
        if unhealthy_count == 0:
            health["overall_status"] = "healthy"
        elif unhealthy_count < len(self.protocol_instances):
            health["overall_status"] = "degraded"
        else:
            health["overall_status"] = "unhealthy"
        
        return health
    
    def _select_protocol(self, message: ProtocolMessage, 
                        protocol_name: Optional[str] = None) -> Optional[CommunicationProtocol]:
        """Select the appropriate protocol for a message"""
        # Use specific protocol if requested
        if protocol_name and protocol_name in self.protocol_instances:
            return self.protocol_instances[protocol_name]
        
        # Use protocol specified in message
        if message.protocol in self.protocols:
            return self.protocols[message.protocol]
        
        # Use routing table
        if message.recipient in self.route_table:
            protocol_type = self.route_table[message.recipient]
            if protocol_type in self.protocols:
                return self.protocols[protocol_type]
        
        # Default to internal protocol
        return self.protocols.get(ProtocolType.INTERNAL)
    
    async def _queue_message(self, message: ProtocolMessage) -> None:
        """Queue a message for reliable delivery"""
        queue = self.message_queues[message.priority]
        await queue.put(message)
    
    async def _send_with_response(self, message: ProtocolMessage, 
                                 protocol: CommunicationProtocol) -> bool:
        """Send message and wait for response"""
        # Implement request-response pattern with timeout and correlation
        try:
            # Set up response correlation if message has correlation_id
            correlation_id = getattr(message, 'correlation_id', None)
            if correlation_id:
                # Store pending request for response correlation
                if not hasattr(self, '_pending_requests'):
                    self._pending_requests = {}
                    
                future = asyncio.Future()
                self._pending_requests[correlation_id] = future
                
                # Send the message
                success = await protocol.send_message(message)
                if not success:
                    self._pending_requests.pop(correlation_id, None)
                    return False
                
                # Wait for response with timeout
                try:
                    await asyncio.wait_for(future, timeout=30.0)  # 30 second timeout
                    return True
                except asyncio.TimeoutError:
                    logger.warning(f"Request {correlation_id} timed out")
                    self._pending_requests.pop(correlation_id, None)
                    return False
            else:
                # Fire-and-forget message
                return await protocol.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send message with response: {e}")
            return False
    
    async def _process_messages(self) -> None:
        """Process queued messages with priority handling"""
        while self._running:
            try:
                # Process messages by priority (urgent first)
                for priority in [MessagePriority.URGENT, MessagePriority.HIGH, 
                               MessagePriority.NORMAL, MessagePriority.LOW]:
                    queue = self.message_queues[priority]
                    
                    try:
                        message = queue.get_nowait()
                        
                        # Try to send the message
                        protocol = self._select_protocol(message)
                        if protocol and protocol.is_connected():
                            success = await protocol.send_message(message)
                            
                            if not success and message.can_retry():
                                # Retry failed messages
                                message.increment_retry()
                                await asyncio.sleep(message.retry_delay)
                                await queue.put(message)
                        
                    except asyncio.QueueEmpty:
                        continue
                
                # Brief pause to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processor: {str(e)}")
                await asyncio.sleep(1)
    
    async def _monitor_health(self) -> None:
        """Monitor protocol health and attempt reconnections"""
        while self._running:
            try:
                for name, protocol in list(self.protocol_instances.items()):
                    if not protocol.is_connected():
                        logger.warning(f"Protocol {name} disconnected, attempting reconnection")
                        
                        try:
                            await protocol.initialize()
                            if protocol.is_connected():
                                logger.info(f"Protocol {name} reconnected successfully")
                        except Exception as e:
                            logger.error(f"Failed to reconnect protocol {name}: {str(e)}")
                
                # Check every 30 seconds
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {str(e)}")
                await asyncio.sleep(10)
