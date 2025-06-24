"""
System Coordinators

Coordinators for managing different aspects of the integrated system.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .models import ComponentMetrics, IntegrationConfig, ResourceRequirement, ResourceType

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Current state of the integrated system"""
    overall_health: float = 1.0
    active_components: int = 0
    total_components: int = 0
    active_workflows: int = 0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AISystemCoordinator:
    """
    Coordinator for managing AI system components and their interactions.
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.system_state = SystemState()
        self.component_registry: Dict[str, Dict[str, Any]] = {}
        self.component_dependencies: Dict[str, Set[str]] = {}
        self.component_capabilities: Dict[str, List[str]] = {}
        
        # Coordination state
        self.coordination_rules: List[Dict[str, Any]] = []
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'coordination_requests': 0,
            'successful_coordinations': 0,
            'failed_coordinations': 0,
            'avg_coordination_time_ms': 0.0
        }
    
    def register_component(self, component_id: str, component_info: Dict[str, Any]):
        """Register an AI component with the coordinator"""
        self.component_registry[component_id] = {
            **component_info,
            'registered_at': datetime.utcnow(),
            'status': 'registered'
        }
        
        # Extract capabilities
        capabilities = component_info.get('capabilities', [])
        self.component_capabilities[component_id] = capabilities
        
        # Extract dependencies
        dependencies = set(component_info.get('dependencies', []))
        self.component_dependencies[component_id] = dependencies
        
        logger.info(f"Registered AI component: {component_id}")
    
    async def coordinate_components(self, coordination_request: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate interaction between AI components"""
        coordination_id = coordination_request.get('coordination_id', f"coord_{datetime.utcnow().timestamp()}")
        start_time = datetime.utcnow()
        
        try:
            self.stats['coordination_requests'] += 1
            
            # Analyze coordination requirements
            required_capabilities = coordination_request.get('required_capabilities', [])
            target_components = coordination_request.get('target_components', [])
            coordination_type = coordination_request.get('type', 'sequential')
            
            # Find suitable components
            suitable_components = self._find_suitable_components(required_capabilities, target_components)
            
            if not suitable_components:
                raise ValueError("No suitable components found for coordination")
            
            # Plan coordination sequence
            coordination_plan = await self._plan_coordination(suitable_components, coordination_type)
            
            # Execute coordination
            coordination_result = await self._execute_coordination(coordination_plan, coordination_request)
            
            # Update statistics
            coordination_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_coordination_stats(True, coordination_time)
            
            return {
                'coordination_id': coordination_id,
                'success': True,
                'result': coordination_result,
                'components_involved': suitable_components,
                'coordination_time_ms': coordination_time
            }
            
        except Exception as e:
            coordination_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_coordination_stats(False, coordination_time)
            
            logger.error(f"Component coordination failed: {e}")
            
            return {
                'coordination_id': coordination_id,
                'success': False,
                'error': str(e),
                'coordination_time_ms': coordination_time
            }
    
    def _find_suitable_components(self, required_capabilities: List[str], 
                                target_components: List[str]) -> List[str]:
        """Find components suitable for coordination"""
        suitable = []
        
        # If specific components are targeted, use them
        if target_components:
            for comp_id in target_components:
                if comp_id in self.component_registry:
                    comp_info = self.component_registry[comp_id]
                    if comp_info.get('status') == 'active':
                        suitable.append(comp_id)
        else:
            # Find components by capabilities
            for comp_id, capabilities in self.component_capabilities.items():
                if comp_id in self.component_registry:
                    comp_info = self.component_registry[comp_id]
                    if comp_info.get('status') == 'active':
                        # Check if component has required capabilities
                        if all(cap in capabilities for cap in required_capabilities):
                            suitable.append(comp_id)
        
        return suitable
    
    async def _plan_coordination(self, components: List[str], coordination_type: str) -> Dict[str, Any]:
        """Plan the coordination sequence"""
        plan = {
            'type': coordination_type,
            'components': components,
            'sequence': [],
            'dependencies': {}
        }
        
        if coordination_type == 'sequential':
            # Simple sequential execution
            plan['sequence'] = components
        
        elif coordination_type == 'parallel':
            # All components execute in parallel
            plan['sequence'] = [components]
        
        elif coordination_type == 'dependency_based':
            # Order based on component dependencies
            ordered_components = self._resolve_dependencies(components)
            plan['sequence'] = ordered_components
        
        else:
            # Default to sequential
            plan['sequence'] = components
        
        return plan
    
    def _resolve_dependencies(self, components: List[str]) -> List[str]:
        """Resolve component dependencies and return execution order"""
        # Simple topological sort
        ordered = []
        remaining = set(components)
        
        while remaining:
            # Find components with no unresolved dependencies
            ready = []
            for comp_id in remaining:
                dependencies = self.component_dependencies.get(comp_id, set())
                unresolved_deps = dependencies.intersection(remaining)
                if not unresolved_deps:
                    ready.append(comp_id)
            
            if not ready:
                # Circular dependency or missing component
                logger.warning("Circular dependency detected, using remaining components as-is")
                ordered.extend(list(remaining))
                break
            
            # Add ready components to order
            ordered.extend(ready)
            remaining -= set(ready)
        
        return ordered
    
    async def _execute_coordination(self, plan: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the coordination plan"""
        coordination_type = plan['type']
        sequence = plan['sequence']
        
        results = {}
        
        if coordination_type in ['sequential', 'dependency_based']:
            # Execute components sequentially
            for comp_id in sequence:
                result = await self._execute_component_action(comp_id, request)
                results[comp_id] = result
                
                # Pass result to next component if needed
                if 'pass_results' in request and request['pass_results']:
                    request['previous_result'] = result
        
        elif coordination_type == 'parallel':
            # Execute components in parallel
            if isinstance(sequence[0], list):
                # Parallel group
                parallel_group = sequence[0]
                tasks = [self._execute_component_action(comp_id, request) for comp_id in parallel_group]
                parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, comp_id in enumerate(parallel_group):
                    results[comp_id] = parallel_results[i]
        
        return results
    
    async def _execute_component_action(self, component_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action on a specific component"""
        try:
            # REAL component interface execution
            from core.agent_factory import AgentFactory

            # Get real component/agent for execution
            agent_factory = AgentFactory()
            component_agent = await agent_factory.get_agent(component_id)

            if component_agent:
                # Execute real action on component
                action_result = await component_agent.execute_action(request)
                return action_result
            else:
                # Component not found - try direct component execution
                from integration.component_manager import ComponentManager
                component_manager = ComponentManager()
                return await component_manager.execute_component_action(component_id, request)
            
            return {
                'component_id': component_id,
                'success': True,
                'result': f"Action executed on {component_id}",
                'execution_time_ms': 100
            }
            
        except Exception as e:
            return {
                'component_id': component_id,
                'success': False,
                'error': str(e)
            }
    
    def _update_coordination_stats(self, success: bool, coordination_time_ms: float):
        """Update coordination statistics"""
        if success:
            self.stats['successful_coordinations'] += 1
        else:
            self.stats['failed_coordinations'] += 1
        
        # Update average coordination time
        total_coordinations = self.stats['successful_coordinations'] + self.stats['failed_coordinations']
        if total_coordinations > 0:
            total_time = self.stats['avg_coordination_time_ms'] * (total_coordinations - 1) + coordination_time_ms
            self.stats['avg_coordination_time_ms'] = total_time / total_coordinations
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get overview of the AI system"""
        active_components = len([c for c in self.component_registry.values() if c.get('status') == 'active'])
        
        return {
            'total_components': len(self.component_registry),
            'active_components': active_components,
            'component_types': list(set(c.get('type', 'unknown') for c in self.component_registry.values())),
            'total_capabilities': len(set().union(*self.component_capabilities.values())),
            'coordination_stats': self.stats.copy()
        }


class ResourceCoordinator:
    """
    Coordinator for managing system resources across components.
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.resource_limits = config.default_resource_limits.copy()
        self.resource_allocations: Dict[str, Dict[str, float]] = {}
        self.resource_usage: Dict[str, float] = {}
        self.resource_reservations: Dict[str, Dict[str, float]] = {}
        
        # Resource monitoring
        self.resource_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            'cpu': 90.0,
            'memory': 90.0,
            'storage': 85.0,
            'network': 80.0
        }
        
        # Statistics
        self.stats = {
            'allocation_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'resource_conflicts': 0
        }
    
    async def allocate_resources(self, component_id: str, 
                               requirements: List[ResourceRequirement]) -> Dict[str, Any]:
        """Allocate resources for a component"""
        try:
            self.stats['allocation_requests'] += 1
            
            # Check if resources are available
            allocation_plan = self._plan_resource_allocation(component_id, requirements)
            
            if not allocation_plan['feasible']:
                self.stats['failed_allocations'] += 1
                return {
                    'success': False,
                    'error': 'Insufficient resources available',
                    'details': allocation_plan
                }
            
            # Perform allocation
            allocated_resources = {}
            for req in requirements:
                resource_type = req.resource_type.value
                amount = allocation_plan['allocations'][resource_type]
                
                # Update allocations
                if component_id not in self.resource_allocations:
                    self.resource_allocations[component_id] = {}
                
                self.resource_allocations[component_id][resource_type] = amount
                allocated_resources[resource_type] = amount
                
                # Update usage
                self.resource_usage[resource_type] = self.resource_usage.get(resource_type, 0) + amount
            
            self.stats['successful_allocations'] += 1
            
            logger.info(f"Allocated resources for component {component_id}: {allocated_resources}")
            
            return {
                'success': True,
                'allocated_resources': allocated_resources,
                'allocation_id': f"alloc_{component_id}_{datetime.utcnow().timestamp()}"
            }
            
        except Exception as e:
            self.stats['failed_allocations'] += 1
            logger.error(f"Resource allocation failed for {component_id}: {e}")
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def _plan_resource_allocation(self, component_id: str, 
                                requirements: List[ResourceRequirement]) -> Dict[str, Any]:
        """Plan resource allocation"""
        plan = {
            'feasible': True,
            'allocations': {},
            'conflicts': [],
            'recommendations': []
        }
        
        for req in requirements:
            resource_type = req.resource_type.value
            requested_amount = req.amount
            
            # Check current usage
            current_usage = self.resource_usage.get(resource_type, 0)
            limit = self.resource_limits.get(resource_type, float('inf'))
            available = limit - current_usage
            
            if requested_amount > available:
                plan['feasible'] = False
                plan['conflicts'].append({
                    'resource_type': resource_type,
                    'requested': requested_amount,
                    'available': available,
                    'current_usage': current_usage,
                    'limit': limit
                })
                
                # Suggest alternatives
                if req.max_amount and req.max_amount <= available:
                    plan['recommendations'].append({
                        'resource_type': resource_type,
                        'suggested_amount': min(available, req.max_amount),
                        'reason': 'Reduced to available amount'
                    })
            else:
                plan['allocations'][resource_type] = requested_amount
        
        return plan
    
    async def deallocate_resources(self, component_id: str) -> Dict[str, Any]:
        """Deallocate resources for a component"""
        try:
            if component_id not in self.resource_allocations:
                return {'success': True, 'message': 'No resources allocated'}
            
            # Release allocated resources
            allocated = self.resource_allocations[component_id]
            for resource_type, amount in allocated.items():
                self.resource_usage[resource_type] = max(0, self.resource_usage.get(resource_type, 0) - amount)
            
            # Remove allocation record
            del self.resource_allocations[component_id]
            
            logger.info(f"Deallocated resources for component {component_id}: {allocated}")
            
            return {
                'success': True,
                'deallocated_resources': allocated
            }
            
        except Exception as e:
            logger.error(f"Resource deallocation failed for {component_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        resource_status = {}
        
        for resource_type, limit in self.resource_limits.items():
            usage = self.resource_usage.get(resource_type, 0)
            utilization = (usage / limit) * 100 if limit > 0 else 0
            
            resource_status[resource_type] = {
                'limit': limit,
                'usage': usage,
                'available': limit - usage,
                'utilization_percent': utilization,
                'alert_threshold': self.alert_thresholds.get(resource_type, 90),
                'alert_triggered': utilization > self.alert_thresholds.get(resource_type, 90)
            }
        
        return {
            'resources': resource_status,
            'total_allocations': len(self.resource_allocations),
            'statistics': self.stats.copy()
        }
    
    async def monitor_resources(self):
        """Monitor resource usage and generate alerts"""
        current_status = self.get_resource_status()
        
        # Record history
        self.resource_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'resource_status': current_status['resources']
        })
        
        # Limit history size
        if len(self.resource_history) > 1000:
            self.resource_history = self.resource_history[-500:]
        
        # Check for alerts
        alerts = []
        for resource_type, status in current_status['resources'].items():
            if status['alert_triggered']:
                alerts.append({
                    'type': 'resource_threshold_exceeded',
                    'resource_type': resource_type,
                    'utilization': status['utilization_percent'],
                    'threshold': status['alert_threshold'],
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        if alerts:
            logger.warning(f"Resource alerts triggered: {len(alerts)} alerts")
            # In a real implementation, would send alerts to monitoring system
        
        return alerts


class EventCoordinator:
    """
    Coordinator for managing system events and event-driven workflows.
    """
    
    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict[str, Any]] = []
        self.event_subscriptions: Dict[str, Set[str]] = {}
        
        # Event processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'event_processing_errors': 0,
            'active_subscriptions': 0
        }
    
    async def start(self):
        """Start event processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._event_processing_loop())
        logger.info("Event coordinator started")
    
    async def stop(self):
        """Stop event processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event coordinator stopped")
    
    def subscribe(self, event_type: str, handler: Callable, subscriber_id: str):
        """Subscribe to events of a specific type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        
        if event_type not in self.event_subscriptions:
            self.event_subscriptions[event_type] = set()
        
        self.event_subscriptions[event_type].add(subscriber_id)
        self.stats['active_subscriptions'] = sum(len(subs) for subs in self.event_subscriptions.values())
        
        logger.debug(f"Subscribed {subscriber_id} to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, subscriber_id: str):
        """Unsubscribe from events"""
        if event_type in self.event_subscriptions:
            self.event_subscriptions[event_type].discard(subscriber_id)
            if not self.event_subscriptions[event_type]:
                del self.event_subscriptions[event_type]
                if event_type in self.event_handlers:
                    del self.event_handlers[event_type]
        
        self.stats['active_subscriptions'] = sum(len(subs) for subs in self.event_subscriptions.values())
        logger.debug(f"Unsubscribed {subscriber_id} from event type: {event_type}")
    
    async def publish_event(self, event_type: str, event_data: Dict[str, Any], 
                          source: str = "system"):
        """Publish an event"""
        event = {
            'event_id': f"evt_{datetime.utcnow().timestamp()}",
            'event_type': event_type,
            'data': event_data,
            'source': source,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.event_queue.put(event)
        self.stats['events_published'] += 1
        
        logger.debug(f"Published event: {event_type} from {source}")
    
    async def _event_processing_loop(self):
        """Main event processing loop"""
        while self.is_running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                self.stats['event_processing_errors'] += 1
    
    async def _process_event(self, event: Dict[str, Any]):
        """Process a single event"""
        event_type = event['event_type']
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-500:]
        
        # Call handlers
        if event_type in self.event_handlers:
            handlers = self.event_handlers[event_type]
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
        
        self.stats['events_processed'] += 1
    
    def get_event_status(self) -> Dict[str, Any]:
        """Get event coordinator status"""
        return {
            'is_running': self.is_running,
            'queue_size': self.event_queue.qsize(),
            'event_types': list(self.event_handlers.keys()),
            'total_subscriptions': self.stats['active_subscriptions'],
            'recent_events': self.event_history[-10:],  # Last 10 events
            'statistics': self.stats.copy()
        }
