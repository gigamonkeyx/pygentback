"""
Integration Event System

Event-driven architecture for integration components with event bus,
handlers, routing, and persistence capabilities.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types"""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    COMPONENT_REGISTERED = "component_registered"
    COMPONENT_UNREGISTERED = "component_unregistered"
    COMPONENT_ERROR = "component_error"
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    HEALTH_CHECK_FAILED = "health_check_failed"
    CONFIGURATION_CHANGED = "configuration_changed"
    USER_ACTION = "user_action"
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """System event"""
    event_id: str
    event_type: EventType
    source: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['priority'] = self.priority.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        data = data.copy()
        data['event_type'] = EventType(data['event_type'])
        data['priority'] = EventPriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class EventSubscription:
    """Event subscription configuration"""
    subscription_id: str
    event_types: List[EventType]
    handler: Callable
    filter_func: Optional[Callable] = None
    priority_filter: Optional[EventPriority] = None
    source_filter: Optional[str] = None
    active: bool = True


class EventBus:
    """
    Event Bus System.
    
    Central event routing and distribution system for integration components.
    Supports event publishing, subscription, filtering, and persistence.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        
        # Event processing
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Subscriptions
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.type_subscriptions: Dict[EventType, List[str]] = {}
        
        # Event storage
        self.event_history: List[Event] = []
        self.max_history_size = 10000
        
        # Statistics
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'active_subscriptions': 0,
            'queue_size': 0
        }
        
        # Event persistence
        self.persistence_enabled = False
        self.persistence_handler: Optional[Callable] = None
    
    async def start(self):
        """Start the event bus"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus"""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
        logger.info("Event bus stopped")
    
    async def publish(self, event: Event):
        """Publish an event to the bus"""
        try:
            await self.event_queue.put(event)
            self.stats['events_published'] += 1
            self.stats['queue_size'] = self.event_queue.qsize()
            
            logger.debug(f"Published event: {event.event_type.value} from {event.source}")
            
        except asyncio.QueueFull:
            logger.error("Event queue full, dropping event")
            self.stats['events_failed'] += 1
    
    async def publish_event(self, event_type: EventType, source: str, 
                          data: Dict[str, Any] = None, 
                          priority: EventPriority = EventPriority.NORMAL,
                          correlation_id: Optional[str] = None,
                          tags: List[str] = None) -> str:
        """Convenience method to publish an event"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            timestamp=datetime.utcnow(),
            data=data or {},
            priority=priority,
            correlation_id=correlation_id,
            tags=tags or []
        )
        
        await self.publish(event)
        return event.event_id
    
    def subscribe(self, event_types: Union[EventType, List[EventType]], 
                 handler: Callable,
                 filter_func: Optional[Callable] = None,
                 priority_filter: Optional[EventPriority] = None,
                 source_filter: Optional[str] = None) -> str:
        """Subscribe to events"""
        if isinstance(event_types, EventType):
            event_types = [event_types]
        
        subscription_id = str(uuid.uuid4())
        subscription = EventSubscription(
            subscription_id=subscription_id,
            event_types=event_types,
            handler=handler,
            filter_func=filter_func,
            priority_filter=priority_filter,
            source_filter=source_filter
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Update type subscriptions
        for event_type in event_types:
            if event_type not in self.type_subscriptions:
                self.type_subscriptions[event_type] = []
            self.type_subscriptions[event_type].append(subscription_id)
        
        self.stats['active_subscriptions'] += 1
        logger.debug(f"Created subscription {subscription_id} for {[et.value for et in event_types]}")
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str):
        """Unsubscribe from events"""
        if subscription_id not in self.subscriptions:
            return
        
        subscription = self.subscriptions[subscription_id]
        
        # Remove from type subscriptions
        for event_type in subscription.event_types:
            if event_type in self.type_subscriptions:
                if subscription_id in self.type_subscriptions[event_type]:
                    self.type_subscriptions[event_type].remove(subscription_id)
                if not self.type_subscriptions[event_type]:
                    del self.type_subscriptions[event_type]
        
        del self.subscriptions[subscription_id]
        self.stats['active_subscriptions'] -= 1
        
        logger.debug(f"Removed subscription {subscription_id}")
    
    async def _process_events(self):
        """Main event processing loop"""
        while self.is_running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Store in history
                self._store_event(event)
                
                # Route to subscribers
                await self._route_event(event)
                
                # Persist if enabled
                if self.persistence_enabled and self.persistence_handler:
                    await self._persist_event(event)
                
                self.stats['events_processed'] += 1
                self.stats['queue_size'] = self.event_queue.qsize()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                self.stats['events_failed'] += 1
    
    def _store_event(self, event: Event):
        """Store event in history"""
        self.event_history.append(event)
        
        # Trim history if too large
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size//2:]
    
    async def _route_event(self, event: Event):
        """Route event to appropriate subscribers"""
        if event.event_type not in self.type_subscriptions:
            return
        
        subscription_ids = self.type_subscriptions[event.event_type]
        
        for subscription_id in subscription_ids:
            if subscription_id not in self.subscriptions:
                continue
            
            subscription = self.subscriptions[subscription_id]
            
            if not subscription.active:
                continue
            
            # Apply filters
            if not self._passes_filters(event, subscription):
                continue
            
            # Call handler
            try:
                if asyncio.iscoroutinefunction(subscription.handler):
                    await subscription.handler(event)
                else:
                    subscription.handler(event)
            except Exception as e:
                logger.error(f"Event handler error for subscription {subscription_id}: {e}")
    
    def _passes_filters(self, event: Event, subscription: EventSubscription) -> bool:
        """Check if event passes subscription filters"""
        # Priority filter
        if subscription.priority_filter and event.priority.value < subscription.priority_filter.value:
            return False
        
        # Source filter
        if subscription.source_filter and event.source != subscription.source_filter:
            return False
        
        # Custom filter
        if subscription.filter_func:
            try:
                return subscription.filter_func(event)
            except Exception as e:
                logger.error(f"Filter function error: {e}")
                return False
        
        return True
    
    async def _persist_event(self, event: Event):
        """Persist event using configured handler"""
        try:
            if asyncio.iscoroutinefunction(self.persistence_handler):
                await self.persistence_handler(event)
            else:
                self.persistence_handler(event)
        except Exception as e:
            logger.error(f"Event persistence error: {e}")
    
    def enable_persistence(self, persistence_handler: Callable):
        """Enable event persistence with custom handler"""
        self.persistence_enabled = True
        self.persistence_handler = persistence_handler
        logger.info("Event persistence enabled")
    
    def disable_persistence(self):
        """Disable event persistence"""
        self.persistence_enabled = False
        self.persistence_handler = None
        logger.info("Event persistence disabled")
    
    def get_events(self, event_type: Optional[EventType] = None,
                  source: Optional[str] = None,
                  since: Optional[datetime] = None,
                  limit: int = 100) -> List[Event]:
        """Get events from history with optional filtering"""
        events = self.event_history
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if source:
            events = [e for e in events if e.source == source]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        # Sort by timestamp (newest first) and limit
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return self.stats.copy()


class EventHandler:
    """
    Base Event Handler.
    
    Base class for creating event handlers with common functionality.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.handled_events = 0
        self.last_handled = None
    
    async def handle_event(self, event: Event):
        """Handle an event (override in subclasses)"""
        self.handled_events += 1
        self.last_handled = datetime.utcnow()
        logger.debug(f"Handler {self.name} processed event {event.event_type.value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            'name': self.name,
            'handled_events': self.handled_events,
            'last_handled': self.last_handled.isoformat() if self.last_handled else None
        }


class WorkflowEventHandler(EventHandler):
    """Event handler for workflow events"""
    
    def __init__(self):
        super().__init__("workflow_handler")
        self.workflow_states = {}
    
    async def handle_event(self, event: Event):
        """Handle workflow events"""
        await super().handle_event(event)
        
        if event.event_type == EventType.WORKFLOW_STARTED:
            workflow_id = event.data.get('workflow_id')
            if workflow_id:
                self.workflow_states[workflow_id] = {
                    'status': 'running',
                    'start_time': event.timestamp,
                    'events': [event]
                }
        
        elif event.event_type in [EventType.WORKFLOW_COMPLETED, EventType.WORKFLOW_FAILED]:
            workflow_id = event.data.get('workflow_id')
            if workflow_id and workflow_id in self.workflow_states:
                self.workflow_states[workflow_id]['status'] = (
                    'completed' if event.event_type == EventType.WORKFLOW_COMPLETED else 'failed'
                )
                self.workflow_states[workflow_id]['end_time'] = event.timestamp
                self.workflow_states[workflow_id]['events'].append(event)


class AlertEventHandler(EventHandler):
    """Event handler for system alerts"""
    
    def __init__(self):
        super().__init__("alert_handler")
        self.active_alerts = []
    
    async def handle_event(self, event: Event):
        """Handle alert events"""
        await super().handle_event(event)
        
        if event.event_type == EventType.SYSTEM_ALERT:
            alert_data = {
                'alert_id': event.event_id,
                'severity': event.data.get('severity', 'info'),
                'message': event.data.get('message', ''),
                'source': event.source,
                'timestamp': event.timestamp
            }
            self.active_alerts.append(alert_data)
            
            # Keep only recent alerts
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.active_alerts = [
                alert for alert in self.active_alerts
                if alert['timestamp'] > cutoff_time
            ]


class EventRouter:
    """
    Event Router.
    
    Routes events based on complex routing rules and patterns.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.routing_rules = []
    
    def add_routing_rule(self, condition: Callable, target_handler: Callable):
        """Add a routing rule"""
        self.routing_rules.append({
            'condition': condition,
            'handler': target_handler
        })
    
    async def route_event(self, event: Event):
        """Route event based on rules"""
        for rule in self.routing_rules:
            try:
                if rule['condition'](event):
                    if asyncio.iscoroutinefunction(rule['handler']):
                        await rule['handler'](event)
                    else:
                        rule['handler'](event)
            except Exception as e:
                logger.error(f"Routing rule error: {e}")


# Convenience functions for common event operations
async def publish_workflow_started(event_bus: EventBus, workflow_id: str, workflow_name: str):
    """Publish workflow started event"""
    await event_bus.publish_event(
        EventType.WORKFLOW_STARTED,
        source="workflow_engine",
        data={'workflow_id': workflow_id, 'workflow_name': workflow_name}
    )

async def publish_workflow_completed(event_bus: EventBus, workflow_id: str, result: Dict[str, Any]):
    """Publish workflow completed event"""
    await event_bus.publish_event(
        EventType.WORKFLOW_COMPLETED,
        source="workflow_engine",
        data={'workflow_id': workflow_id, 'result': result}
    )

async def publish_system_alert(event_bus: EventBus, severity: str, message: str, source: str):
    """Publish system alert event"""
    priority = EventPriority.CRITICAL if severity == 'critical' else EventPriority.HIGH
    await event_bus.publish_event(
        EventType.SYSTEM_ALERT,
        source=source,
        data={'severity': severity, 'message': message},
        priority=priority
    )
