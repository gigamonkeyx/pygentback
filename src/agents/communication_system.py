#!/usr/bin/env python3
"""
Multi-Agent Communication System

Advanced communication system for agent-to-agent messaging, coordination,
and collaborative task execution.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

from .base_agent import AgentMessage, MessageType

try:
    from ..cache.redis_manager import redis_manager, ensure_redis_initialized
except ImportError:
    redis_manager = None
    ensure_redis_initialized = None

try:
    from ..cache.cache_layers import cache_manager
except ImportError:
    cache_manager = None

logger = logging.getLogger(__name__)


class CommunicationProtocol(Enum):
    """Communication protocol types"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class CommunicationChannel:
    """Communication channel definition"""
    channel_id: str
    name: str
    protocol: CommunicationProtocol
    participants: Set[str] = field(default_factory=set)
    message_retention_hours: int = 24
    max_message_size: int = 1024 * 1024  # 1MB
    encryption_enabled: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MessageRoute:
    """Message routing information"""
    sender_id: str
    recipient_ids: List[str]
    channel_id: Optional[str] = None
    protocol: CommunicationProtocol = CommunicationProtocol.DIRECT
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_timeout: int = 30
    retry_count: int = 3


@dataclass
class CommunicationMetrics:
    """Communication system metrics"""
    total_messages: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    average_delivery_time: float = 0.0
    active_channels: int = 0
    active_subscriptions: int = 0
    bandwidth_usage: float = 0.0


class MultiAgentCommunicationSystem:
    """Advanced multi-agent communication system"""
    
    def __init__(self):
        # Communication infrastructure
        self.channels: Dict[str, CommunicationChannel] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> agent_ids
        self.message_queues: Dict[str, asyncio.Queue] = {}  # agent_id -> queue
        
        # Routing and delivery
        self.routing_table: Dict[str, str] = {}  # agent_id -> preferred_channel
        self.delivery_handlers: Dict[str, Callable] = {}
        self.message_filters: Dict[str, List[Callable]] = {}
        
        # Performance and monitoring
        self.metrics = CommunicationMetrics()
        self.message_history: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_queue_size = 1000
        self.default_timeout = 30
        self.enable_message_persistence = True
        
        # State
        self.is_initialized = False
        self.is_running = False
        
        logger.info("Multi-agent communication system created")
    
    async def initialize(self) -> bool:
        """Initialize communication system"""
        try:
            logger.info("Initializing multi-agent communication system...")

            # Ensure Redis is initialized if available
            if ensure_redis_initialized:
                await ensure_redis_initialized()

            # Initialize Redis pub/sub for distributed communication
            await self._initialize_redis_pubsub()

            # Create default channels
            await self._create_default_channels()

            # Start communication loops
            asyncio.create_task(self._message_processing_loop())
            asyncio.create_task(self._delivery_monitoring_loop())
            asyncio.create_task(self._cleanup_loop())

            self.is_initialized = True
            self.is_running = True

            logger.info("Multi-agent communication system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize communication system: {e}")
            return False
    
    async def _initialize_redis_pubsub(self):
        """Initialize Redis pub/sub for distributed messaging"""
        try:
            if redis_manager:
                # Try to initialize Redis if not already initialized
                if not redis_manager.is_initialized and ensure_redis_initialized:
                    await ensure_redis_initialized()

                if redis_manager.is_initialized:
                    # Subscribe to agent communication channels
                    try:
                        pubsub = await redis_manager.subscribe("agent_messages")
                        if pubsub:
                            asyncio.create_task(self._handle_redis_messages(pubsub))

                        pubsub_broadcast = await redis_manager.subscribe("agent_broadcasts")
                        if pubsub_broadcast:
                            asyncio.create_task(self._handle_redis_broadcasts(pubsub_broadcast))

                        logger.info("Redis pub/sub initialized successfully")
                    except Exception as e:
                        logger.warning(f"Redis pub/sub setup failed: {e}")
                else:
                    logger.warning("Redis not available for pub/sub messaging")
            else:
                logger.warning("Redis manager not available")

        except Exception as e:
            logger.error(f"Failed to initialize Redis pub/sub: {e}")
    
    async def _create_default_channels(self):
        """Create default communication channels"""
        try:
            # General broadcast channel
            await self.create_channel(
                "general_broadcast",
                "General Broadcast Channel",
                CommunicationProtocol.BROADCAST
            )
            
            # Coordination channel
            await self.create_channel(
                "coordination",
                "Agent Coordination Channel",
                CommunicationProtocol.MULTICAST
            )
            
            # Emergency channel
            await self.create_channel(
                "emergency",
                "Emergency Communication Channel",
                CommunicationProtocol.BROADCAST
            )
            
        except Exception as e:
            logger.error(f"Failed to create default channels: {e}")
    
    async def create_channel(self, channel_id: str, name: str, 
                           protocol: CommunicationProtocol) -> bool:
        """Create a new communication channel"""
        try:
            if channel_id in self.channels:
                logger.warning(f"Channel {channel_id} already exists")
                return False
            
            channel = CommunicationChannel(
                channel_id=channel_id,
                name=name,
                protocol=protocol
            )
            
            self.channels[channel_id] = channel
            self.metrics.active_channels += 1
            
            # Cache channel information if cache manager available
            if cache_manager:
                try:
                    await cache_manager.cache_performance_metric(
                        f"communication_channel:{channel_id}",
                        {
                            "name": name,
                            "protocol": protocol.value,
                            "created_at": channel.created_at.isoformat(),
                            "participants": list(channel.participants)
                        },
                        ttl=3600  # 1 hour
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache channel info: {e}")
            
            logger.info(f"Created communication channel: {name} ({channel_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create channel {channel_id}: {e}")
            return False
    
    async def join_channel(self, agent_id: str, channel_id: str) -> bool:
        """Add agent to communication channel"""
        try:
            if channel_id not in self.channels:
                logger.error(f"Channel {channel_id} does not exist")
                return False
            
            channel = self.channels[channel_id]
            channel.participants.add(agent_id)
            
            # Create message queue for agent if not exists
            if agent_id not in self.message_queues:
                self.message_queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
            
            logger.info(f"Agent {agent_id} joined channel {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to join channel {channel_id}: {e}")
            return False
    
    async def leave_channel(self, agent_id: str, channel_id: str) -> bool:
        """Remove agent from communication channel"""
        try:
            if channel_id not in self.channels:
                logger.error(f"Channel {channel_id} does not exist")
                return False
            
            channel = self.channels[channel_id]
            channel.participants.discard(agent_id)
            
            logger.info(f"Agent {agent_id} left channel {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to leave channel {channel_id}: {e}")
            return False
    
    async def subscribe_to_topic(self, agent_id: str, topic: str) -> bool:
        """Subscribe agent to a topic"""
        try:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            
            self.subscriptions[topic].add(agent_id)
            self.metrics.active_subscriptions += 1
            
            # Create message queue for agent if not exists
            if agent_id not in self.message_queues:
                self.message_queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
            
            logger.info(f"Agent {agent_id} subscribed to topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {e}")
            return False
    
    async def unsubscribe_from_topic(self, agent_id: str, topic: str) -> bool:
        """Unsubscribe agent from a topic"""
        try:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(agent_id)
                
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
                
                self.metrics.active_subscriptions -= 1
            
            logger.info(f"Agent {agent_id} unsubscribed from topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from topic {topic}: {e}")
            return False
    
    async def send_message(self, message: AgentMessage, route: MessageRoute) -> bool:
        """Send message using specified route"""
        try:
            start_time = datetime.utcnow()
            
            # Validate message
            if not self._validate_message(message):
                return False
            
            # Apply message filters
            if not await self._apply_message_filters(message):
                return False
            
            # Route message based on protocol
            success = False
            if route.protocol == CommunicationProtocol.DIRECT:
                success = await self._send_direct_message(message, route)
            elif route.protocol == CommunicationProtocol.BROADCAST:
                success = await self._send_broadcast_message(message, route)
            elif route.protocol == CommunicationProtocol.MULTICAST:
                success = await self._send_multicast_message(message, route)
            elif route.protocol == CommunicationProtocol.PUBLISH_SUBSCRIBE:
                success = await self._send_pubsub_message(message, route)
            elif route.protocol == CommunicationProtocol.REQUEST_RESPONSE:
                success = await self._send_request_response_message(message, route)
            
            # Update metrics
            self.metrics.total_messages += 1
            if success:
                self.metrics.messages_delivered += 1
                delivery_time = (datetime.utcnow() - start_time).total_seconds()
                self._update_delivery_metrics(delivery_time)
            else:
                self.metrics.messages_failed += 1
            
            # Store message history
            if self.enable_message_persistence:
                await self._store_message_history(message, route, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.metrics.messages_failed += 1
            return False
    
    def _validate_message(self, message: AgentMessage) -> bool:
        """Validate message format and content"""
        try:
            if not message.id:
                logger.error("Message ID is required")
                return False
            
            if not message.sender_id:
                logger.error("Sender ID is required")
                return False
            
            # Check message size
            message_size = len(json.dumps(message.content))
            if message_size > 1024 * 1024:  # 1MB limit
                logger.error(f"Message size {message_size} exceeds limit")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Message validation error: {e}")
            return False
    
    async def _apply_message_filters(self, message: AgentMessage) -> bool:
        """Apply message filters"""
        try:
            sender_filters = self.message_filters.get(message.sender_id, [])
            
            for filter_func in sender_filters:
                if not await filter_func(message):
                    logger.debug(f"Message {message.id} filtered out")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Message filter error: {e}")
            return True  # Allow message through on filter error
    
    async def _send_direct_message(self, message: AgentMessage, route: MessageRoute) -> bool:
        """Send direct message to specific recipient"""
        try:
            if not route.recipient_ids:
                logger.error("No recipients specified for direct message")
                return False
            
            recipient_id = route.recipient_ids[0]  # Direct message to first recipient
            
            if recipient_id not in self.message_queues:
                logger.error(f"Recipient {recipient_id} not found")
                return False
            
            # Add message to recipient's queue
            queue = self.message_queues[recipient_id]
            
            try:
                queue.put_nowait(message)
                logger.debug(f"Direct message {message.id} queued for {recipient_id}")
                return True
            except asyncio.QueueFull:
                logger.error(f"Message queue full for recipient {recipient_id}")
                return False
            
        except Exception as e:
            logger.error(f"Direct message delivery failed: {e}")
            return False
    
    async def _send_broadcast_message(self, message: AgentMessage, route: MessageRoute) -> bool:
        """Send broadcast message to all participants"""
        try:
            channel_id = route.channel_id or "general_broadcast"
            
            if channel_id not in self.channels:
                logger.error(f"Broadcast channel {channel_id} not found")
                return False
            
            channel = self.channels[channel_id]
            successful_deliveries = 0
            
            for participant_id in channel.participants:
                if participant_id == message.sender_id:
                    continue  # Don't send to sender
                
                if participant_id in self.message_queues:
                    try:
                        queue = self.message_queues[participant_id]
                        queue.put_nowait(message)
                        successful_deliveries += 1
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for participant {participant_id}")
            
            logger.debug(f"Broadcast message {message.id} delivered to {successful_deliveries} participants")
            return successful_deliveries > 0
            
        except Exception as e:
            logger.error(f"Broadcast message delivery failed: {e}")
            return False
    
    async def _send_multicast_message(self, message: AgentMessage, route: MessageRoute) -> bool:
        """Send multicast message to specific group"""
        try:
            successful_deliveries = 0
            
            for recipient_id in route.recipient_ids:
                if recipient_id in self.message_queues:
                    try:
                        queue = self.message_queues[recipient_id]
                        queue.put_nowait(message)
                        successful_deliveries += 1
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for recipient {recipient_id}")
            
            logger.debug(f"Multicast message {message.id} delivered to {successful_deliveries} recipients")
            return successful_deliveries > 0
            
        except Exception as e:
            logger.error(f"Multicast message delivery failed: {e}")
            return False
    
    async def _send_pubsub_message(self, message: AgentMessage, route: MessageRoute) -> bool:
        """Send publish-subscribe message"""
        try:
            # Use Redis pub/sub for distributed messaging
            topic = route.channel_id or "default_topic"
            
            message_data = {
                "id": message.id,
                "type": message.type.value,
                "sender_id": message.sender_id,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "priority": message.priority
            }
            
            await redis_manager.publish(f"agent_topic:{topic}", message_data)
            
            logger.debug(f"Published message {message.id} to topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Pub/sub message delivery failed: {e}")
            return False
    
    async def _send_request_response_message(self, message: AgentMessage, route: MessageRoute) -> bool:
        """Send request-response message with correlation tracking"""
        try:
            # Send as direct message but track for response
            message.requires_response = True
            
            success = await self._send_direct_message(message, route)
            
            if success and message.requires_response:
                # Set up response tracking
                await self._setup_response_tracking(message, route)
            
            return success
            
        except Exception as e:
            logger.error(f"Request-response message delivery failed: {e}")
            return False
    
    async def _setup_response_tracking(self, message: AgentMessage, route: MessageRoute):
        """Set up response tracking for request-response messages"""
        try:
            # Cache response tracking information if cache manager available
            if cache_manager:
                try:
                    await cache_manager.cache_performance_metric(
                        f"response_tracking:{message.id}",
                        {
                            "sender_id": message.sender_id,
                            "recipient_ids": route.recipient_ids,
                            "sent_at": datetime.utcnow().isoformat(),
                            "timeout": route.delivery_timeout
                        },
                        ttl=route.delivery_timeout
                    )
                except Exception as cache_error:
                    logger.warning(f"Failed to cache response tracking: {cache_error}")

        except Exception as e:
            logger.error(f"Response tracking setup failed: {e}")
    
    async def receive_message(self, agent_id: str) -> Optional[AgentMessage]:
        """Receive message for specific agent"""
        try:
            if agent_id not in self.message_queues:
                return None
            
            queue = self.message_queues[agent_id]
            
            try:
                # Get message with timeout
                message_data = await asyncio.wait_for(queue.get(), timeout=0.1)
                return message_data
            except asyncio.TimeoutError:
                return None
            
        except Exception as e:
            logger.error(f"Message receive failed for agent {agent_id}: {e}")
            return None
    
    async def _handle_redis_message(self, message_data: Dict[str, Any]):
        """Handle message received via Redis pub/sub"""
        try:
            # Reconstruct AgentMessage from Redis data
            message = AgentMessage(
                id=message_data["id"],
                type=MessageType(message_data["type"]),
                sender_id=message_data["sender_id"],
                content=message_data["content"],
                timestamp=datetime.fromisoformat(message_data["timestamp"]),
                priority=message_data.get("priority", 1)
            )
            
            # Route to local subscribers
            await self._route_redis_message(message)
            
        except Exception as e:
            logger.error(f"Redis message handling failed: {e}")
    
    async def _handle_redis_broadcast(self, message_data: Dict[str, Any]):
        """Handle broadcast message received via Redis"""
        try:
            # Similar to _handle_redis_message but for broadcasts
            await self._handle_redis_message(message_data)
            
        except Exception as e:
            logger.error(f"Redis broadcast handling failed: {e}")
    
    async def _route_redis_message(self, message: AgentMessage):
        """Route Redis message to local agents"""
        try:
            # Route to all local message queues
            for agent_id, queue in self.message_queues.items():
                if agent_id != message.sender_id:
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Redis message routing failed: {e}")
    
    def _update_delivery_metrics(self, delivery_time: float):
        """Update delivery time metrics"""
        try:
            total_deliveries = self.metrics.messages_delivered
            current_avg = self.metrics.average_delivery_time
            
            self.metrics.average_delivery_time = (
                (current_avg * (total_deliveries - 1) + delivery_time) / total_deliveries
            )
            
        except Exception as e:
            logger.error(f"Delivery metrics update failed: {e}")
    
    async def _store_message_history(self, message: AgentMessage, route: MessageRoute, success: bool):
        """Store message history for auditing"""
        try:
            history_entry = {
                "message_id": message.id,
                "sender_id": message.sender_id,
                "recipient_ids": route.recipient_ids,
                "protocol": route.protocol.value,
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
                "message_type": message.type.value
            }
            
            self.message_history[message.id] = history_entry
            
            # Cache message history if cache manager available
            if cache_manager:
                try:
                    await cache_manager.cache_performance_metric(
                        f"message_history:{message.id}",
                        history_entry,
                        ttl=24 * 3600  # 24 hours
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache message history: {e}")
            
        except Exception as e:
            logger.error(f"Message history storage failed: {e}")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.is_running:
            try:
                # Process any pending system messages
                await self._process_system_messages()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Message processing loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_system_messages(self):
        """Process system-level messages"""
        try:
            # Handle system coordination, health checks, etc.
            current_time = datetime.utcnow()

            # Process heartbeat messages
            for agent_id in list(self.message_queues.keys()):
                # Check if agent is still responsive
                last_activity_key = f"agent_activity:{agent_id}"
                if cache_manager:
                    last_activity = await cache_manager.get_cached_data(last_activity_key)
                    if last_activity:
                        last_time = datetime.fromisoformat(last_activity)
                        if (current_time - last_time).total_seconds() > 300:  # 5 minutes timeout
                            logger.warning(f"Agent {agent_id} appears inactive")

            # Process system coordination messages
            if redis_manager:
                # Check for system-wide coordination messages
                system_messages = await redis_manager.get_list("system_messages", 0, -1)
                for msg in system_messages:
                    await self._handle_system_coordination_message(msg)

        except Exception as e:
            logger.error(f"System message processing error: {e}")

    async def _handle_system_coordination_message(self, message: Dict[str, Any]):
        """Handle system coordination message"""
        try:
            msg_type = message.get("type")

            if msg_type == "agent_shutdown":
                agent_id = message.get("agent_id")
                if agent_id in self.message_queues:
                    del self.message_queues[agent_id]
                    logger.info(f"Removed message queue for shutdown agent {agent_id}")

            elif msg_type == "channel_cleanup":
                channel_id = message.get("channel_id")
                if channel_id in self.channels:
                    del self.channels[channel_id]
                    logger.info(f"Cleaned up channel {channel_id}")

        except Exception as e:
            logger.error(f"System coordination message handling error: {e}")
    
    async def _delivery_monitoring_loop(self):
        """Monitor message delivery and handle timeouts"""
        while self.is_running:
            try:
                # Check for message delivery timeouts
                await self._check_delivery_timeouts()
                
                # Update communication metrics
                await self._update_communication_metrics()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Delivery monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _check_delivery_timeouts(self):
        """Check for message delivery timeouts"""
        try:
            if not cache_manager:
                return

            current_time = datetime.utcnow()

            # Get all response tracking entries
            tracking_keys = await cache_manager.get_keys_pattern("response_tracking:*")

            for key in tracking_keys:
                tracking_data = await cache_manager.get_cached_data(key)
                if not tracking_data:
                    continue

                sent_time = datetime.fromisoformat(tracking_data["sent_at"])
                timeout_seconds = tracking_data.get("timeout", 30)

                if (current_time - sent_time).total_seconds() > timeout_seconds:
                    # Message has timed out
                    message_id = key.split(":")[-1]
                    sender_id = tracking_data["sender_id"]

                    logger.warning(f"Message {message_id} from {sender_id} timed out")

                    # Send timeout notification to sender
                    timeout_message = AgentMessage(
                        type=MessageType.RESPONSE,
                        sender_id="system",
                        recipient_id=sender_id,
                        content={
                            "type": "timeout",
                            "original_message_id": message_id,
                            "timeout_seconds": timeout_seconds
                        }
                    )

                    # Queue timeout notification
                    if sender_id in self.message_queues:
                        try:
                            self.message_queues[sender_id].put_nowait(timeout_message)
                        except asyncio.QueueFull:
                            logger.warning(f"Could not deliver timeout notification to {sender_id}")

                    # Remove tracking entry
                    await cache_manager.delete_cached_data(key)

        except Exception as e:
            logger.error(f"Delivery timeout check failed: {e}")
    
    async def _update_communication_metrics(self):
        """Update communication system metrics"""
        try:
            # Update bandwidth usage, queue sizes, etc.
            total_queue_size = sum(queue.qsize() for queue in self.message_queues.values())
            self.metrics.bandwidth_usage = total_queue_size * 0.001  # Rough estimate
            
            # Cache metrics if cache manager available
            if cache_manager:
                try:
                    await cache_manager.cache_performance_metric(
                        "communication_metrics",
                        {
                            "total_messages": self.metrics.total_messages,
                            "messages_delivered": self.metrics.messages_delivered,
                            "messages_failed": self.metrics.messages_failed,
                            "average_delivery_time": self.metrics.average_delivery_time,
                            "active_channels": self.metrics.active_channels,
                            "active_subscriptions": self.metrics.active_subscriptions,
                            "bandwidth_usage": self.metrics.bandwidth_usage,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        ttl=300  # 5 minutes
                    )
                except Exception as cache_error:
                    logger.warning(f"Failed to cache communication metrics: {cache_error}")
            
        except Exception as e:
            logger.error(f"Communication metrics update failed: {e}")
    
    async def _cleanup_loop(self):
        """Clean up old messages and channels"""
        while self.is_running:
            try:
                # Clean up old message history
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                to_remove = []
                for message_id, history in self.message_history.items():
                    timestamp = datetime.fromisoformat(history["timestamp"])
                    if timestamp < cutoff_time:
                        to_remove.append(message_id)
                
                for message_id in to_remove:
                    del self.message_history[message_id]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    def add_message_filter(self, agent_id: str, filter_func: Callable):
        """Add message filter for specific agent"""
        if agent_id not in self.message_filters:
            self.message_filters[agent_id] = []
        self.message_filters[agent_id].append(filter_func)
    
    def get_communication_status(self) -> Dict[str, Any]:
        """Get communication system status"""
        return {
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "metrics": {
                "total_messages": self.metrics.total_messages,
                "messages_delivered": self.metrics.messages_delivered,
                "messages_failed": self.metrics.messages_failed,
                "average_delivery_time": self.metrics.average_delivery_time,
                "active_channels": self.metrics.active_channels,
                "active_subscriptions": self.metrics.active_subscriptions,
                "bandwidth_usage": self.metrics.bandwidth_usage
            },
            "channels": {
                channel_id: {
                    "name": channel.name,
                    "protocol": channel.protocol.value,
                    "participants": len(channel.participants)
                }
                for channel_id, channel in self.channels.items()
            },
            "active_queues": len(self.message_queues),
            "total_queue_size": sum(queue.qsize() for queue in self.message_queues.values())
        }
    
    async def shutdown(self):
        """Shutdown communication system"""
        try:
            logger.info("Shutting down communication system...")
            
            self.is_running = False
            
            # Clear all message queues
            for queue in self.message_queues.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            self.message_queues.clear()
            self.channels.clear()
            self.subscriptions.clear()
            
            logger.info("Communication system shutdown completed")
            
        except Exception as e:
            logger.error(f"Communication system shutdown error: {e}")


# Global communication system instance
communication_system = MultiAgentCommunicationSystem()
