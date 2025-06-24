"""
Agent Registry

Dynamic agent discovery, registration, and capability management system.
Provides intelligent agent selection, load monitoring, and performance tracking.
Enhanced with Darwinian evolution and A2A protocol integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from collections import defaultdict, deque
import uuid
from dataclasses import dataclass, field

from .coordination_models import (
    AgentCapability, AgentType, TaskRequest, PerformanceMetrics,
    OrchestrationConfig, AgentID, CapabilityName, AgentStatusCallback
)

# Import A2A protocol for evolutionary integration
try:
    from ..a2a import AgentCard, A2APeerInfo
except ImportError:
    # Fallback for development
    AgentCard = None
    A2APeerInfo = None

logger = logging.getLogger(__name__)


@dataclass
class EvolutionMetadata:
    """Evolutionary capabilities metadata for agents."""
    generation: int = 0
    fitness_score: float = 0.0
    parent_agents: List[str] = field(default_factory=list)
    mutation_rate: float = 0.1
    crossover_points: List[str] = field(default_factory=list)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    capability_mutations: Dict[str, float] = field(default_factory=dict)
    adaptation_traits: List[str] = field(default_factory=list)
    survival_metrics: Dict[str, float] = field(default_factory=dict)
    last_evolution: Optional[datetime] = None


@dataclass
class AgentLineage:
    """Agent lineage and genealogy tracking."""
    agent_id: str
    lineage_id: str
    ancestors: List[str] = field(default_factory=list)
    descendants: List[str] = field(default_factory=list)
    lineage_depth: int = 0
    creation_method: str = "spawn"  # spawn, crossover, mutation, migration
    evolutionary_branch: str = "main"
    genetic_diversity_score: float = 0.0
    genealogy_tree: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedReputation:
    """Distributed agent reputation via evolution results."""
    agent_id: str
    global_fitness: float = 0.0
    peer_ratings: Dict[str, float] = field(default_factory=dict)
    cross_network_performance: Dict[str, float] = field(default_factory=dict)
    reputation_consensus: float = 0.0
    trust_score: float = 0.0
    collaboration_history: List[Dict[str, Any]] = field(default_factory=list)
    validation_proofs: List[str] = field(default_factory=list)
    last_reputation_update: Optional[datetime] = None


class AgentRegistry:
    """
    Dynamic agent registry for multi-agent orchestration.
    Enhanced with Darwinian evolution and A2A protocol integration.
    
    Features:
    - Dynamic agent discovery and registration
    - Capability-based agent selection
    - Load monitoring and performance tracking
    - Health monitoring and failover
    - Agent lifecycle management
    - Evolutionary capabilities metadata tracking
    - A2A agent card generation with evolution history
    - Distributed agent capability evolution tracking
    - Agent lineage and genealogy tracking
    - Evolutionary performance metrics integration
    - Distributed agent reputation via evolution results
    """
    
    def __init__(self, config: OrchestrationConfig, a2a_server=None):
        self.config = config
        self.agents: Dict[AgentID, AgentCapability] = {}
        self.capability_index: Dict[CapabilityName, Set[AgentID]] = defaultdict(set)
        self.type_index: Dict[AgentType, Set[AgentID]] = defaultdict(set)
        
        # Performance tracking
        self.agent_metrics: Dict[AgentID, PerformanceMetrics] = {}
        self.task_history: Dict[AgentID, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Health monitoring
        self.health_check_tasks: Dict[AgentID, asyncio.Task] = {}
        self.is_running = False
        
        # Load balancing
        self.round_robin_counters: Dict[CapabilityName, int] = defaultdict(int)
        self.agent_weights: Dict[AgentID, float] = defaultdict(lambda: 1.0)
        
        # Callbacks
        self.status_callbacks: List[AgentStatusCallback] = []
        
        # Phase 1.6: Evolution Integration
        self.evolution_metadata: Dict[AgentID, EvolutionMetadata] = {}
        self.agent_lineages: Dict[AgentID, AgentLineage] = {}
        self.distributed_reputations: Dict[AgentID, DistributedReputation] = {}
        self.capability_evolution_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.lineage_trees: Dict[str, Dict[str, Any]] = {}
        self.a2a_server = a2a_server
        self.evolution_generation_counter = 0
        
        # A2A agent cards cache
        self.local_agent_cards: Dict[AgentID, Any] = {}
        self.peer_agent_cards: Dict[str, List[Any]] = {}
        
        logger.info("Agent Registry initialized with evolutionary capabilities")
    
    async def start(self):
        """Start the registry and begin monitoring."""
        self.is_running = True
        
        # Start health monitoring for all registered agents
        for agent_id in self.agents:
            await self._start_health_monitoring(agent_id)
        
        logger.info("Agent Registry started")
    
    async def stop(self):
        """Stop the registry and cleanup resources."""
        self.is_running = False
        
        # Stop health monitoring
        for task in self.health_check_tasks.values():
            task.cancel()
        
        self.health_check_tasks.clear()
        
        logger.info("Agent Registry stopped")
    
    async def register_agent(self, agent_capability: AgentCapability) -> bool:
        """Register a new agent."""
        try:
            agent_id = agent_capability.agent_id
            
            # Validate agent capability
            if not agent_id or not agent_capability.name:
                raise ValueError("Agent ID and name are required")
            
            # Check if agent already exists
            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} already registered")
                return False
            
            # Register agent
            self.agents[agent_id] = agent_capability
            
            # Update indexes
            for task_type in agent_capability.supported_tasks:
                self.capability_index[task_type].add(agent_id)
            
            self.type_index[agent_capability.agent_type].add(agent_id)
            
            # Initialize metrics
            self.agent_metrics[agent_id] = PerformanceMetrics()
            
            # Start health monitoring if registry is running
            if self.is_running:
                await self._start_health_monitoring(agent_id)
            
            # Notify callbacks
            await self._notify_status_callbacks(agent_capability)
            
            logger.info(f"Registered agent: {agent_capability.name} ({agent_capability.agent_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_capability.name}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: AgentID) -> bool:
        """Unregister an agent."""
        try:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found")
                return False
            
            agent_capability = self.agents[agent_id]
            
            # Stop health monitoring
            if agent_id in self.health_check_tasks:
                self.health_check_tasks[agent_id].cancel()
                del self.health_check_tasks[agent_id]
            
            # Remove from indexes
            for task_type in agent_capability.supported_tasks:
                self.capability_index[task_type].discard(agent_id)
            
            self.type_index[agent_capability.agent_type].discard(agent_id)
            
            # Remove agent
            del self.agents[agent_id]
            if agent_id in self.agent_metrics:
                del self.agent_metrics[agent_id]
            if agent_id in self.task_history:
                del self.task_history[agent_id]
            
            logger.info(f"Unregistered agent: {agent_capability.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def update_agent_status(self, agent_id: AgentID, **updates) -> bool:
        """Update agent status and capabilities."""
        try:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found")
                return False
            
            agent_capability = self.agents[agent_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(agent_capability, field):
                    setattr(agent_capability, field, value)
            
            agent_capability.last_activity = datetime.utcnow()
            
            # Notify callbacks
            await self._notify_status_callbacks(agent_capability)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {e}")
            return False
    
    async def select_agent(self, 
                          task_type: str,
                          agent_type: Optional[AgentType] = None,
                          preferred_agent: Optional[AgentID] = None,
                          exclude_agents: Optional[Set[AgentID]] = None) -> Optional[AgentID]:
        """Select the best agent for a task."""
        exclude_agents = exclude_agents or set()
        
        # Get candidate agents
        candidates = set()
        
        # Filter by task capability
        if task_type in self.capability_index:
            candidates.update(self.capability_index[task_type])
        
        # Filter by agent type if specified
        if agent_type and agent_type in self.type_index:
            candidates.intersection_update(self.type_index[agent_type])
        
        # Remove excluded agents
        candidates -= exclude_agents
        
        # Filter for available agents
        available_agents = []
        for agent_id in candidates:
            agent_capability = self.agents.get(agent_id)
            if agent_capability and agent_capability.can_handle_task(task_type):
                available_agents.append(agent_id)
        
        if not available_agents:
            return None
        
        # Use preferred agent if available
        if preferred_agent and preferred_agent in available_agents:
            return preferred_agent
        
        # Apply selection algorithm
        return self._select_best_agent(available_agents, task_type)
    
    async def select_multiple_agents(self,
                                   task_type: str,
                                   count: int,
                                   agent_type: Optional[AgentType] = None,
                                   exclude_agents: Optional[Set[AgentID]] = None) -> List[AgentID]:
        """Select multiple agents for parallel task execution."""
        selected_agents = []
        exclude_set = exclude_agents.copy() if exclude_agents else set()
        
        for _ in range(count):
            agent_id = await self.select_agent(
                task_type=task_type,
                agent_type=agent_type,
                exclude_agents=exclude_set
            )
            
            if agent_id:
                selected_agents.append(agent_id)
                exclude_set.add(agent_id)
            else:
                break  # No more available agents
        
        return selected_agents
    
    async def assign_task(self, agent_id: AgentID, task: TaskRequest) -> bool:
        """Assign a task to an agent."""
        try:
            agent_capability = self.agents.get(agent_id)
            if not agent_capability:
                return False
            
            if not agent_capability.can_handle_task(task.task_type):
                return False
            
            # Update agent load
            agent_capability.current_load += 1
            agent_capability.last_activity = datetime.utcnow()
            
            # Record task assignment
            self.task_history[agent_id].append({
                'task_id': task.task_id,
                'task_type': task.task_type,
                'assigned_at': datetime.utcnow(),
                'status': 'assigned'
            })
            
            # Update task
            task.assigned_agent_id = agent_id
            task.update_status(task.status.ASSIGNED)
            
            logger.debug(f"Assigned task {task.task_id} to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign task to agent {agent_id}: {e}")
            return False
    
    async def complete_task(self, agent_id: AgentID, task: TaskRequest, success: bool = True) -> bool:
        """Mark a task as completed by an agent."""
        try:
            agent_capability = self.agents.get(agent_id)
            if not agent_capability:
                return False
            
            # Update agent load
            if agent_capability.current_load > 0:
                agent_capability.current_load -= 1
            
            agent_capability.last_activity = datetime.utcnow()
            
            # Update task history
            task_history = self.task_history[agent_id]
            for task_record in reversed(task_history):
                if task_record['task_id'] == task.task_id:
                    task_record['completed_at'] = datetime.utcnow()
                    task_record['success'] = success
                    task_record['status'] = 'completed' if success else 'failed'
                    break
            
            # Update agent performance
            self._update_agent_performance(agent_id, success, task.execution_time or 0.0)
            
            logger.debug(f"Task {task.task_id} completed by agent {agent_id} (success: {success})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete task for agent {agent_id}: {e}")
            return False
    
    async def get_agent_status(self, agent_id: Optional[AgentID] = None) -> Dict[str, Any]:
        """Get status information for agents."""
        if agent_id:
            if agent_id not in self.agents:
                return {"error": f"Agent {agent_id} not found"}
            
            agent_capability = self.agents[agent_id]
            metrics = self.agent_metrics.get(agent_id, PerformanceMetrics())
            recent_tasks = list(self.task_history[agent_id])[-10:]  # Last 10 tasks
            
            return {
                "agent_id": agent_id,
                "name": agent_capability.name,
                "type": agent_capability.agent_type.value,
                "is_available": agent_capability.is_available,
                "current_load": agent_capability.current_load,
                "max_concurrent_tasks": agent_capability.max_concurrent_tasks,
                "utilization_rate": agent_capability.utilization_rate,
                "performance_score": agent_capability.performance_score,
                "supported_tasks": list(agent_capability.supported_tasks),
                "specializations": agent_capability.specializations,
                "last_activity": agent_capability.last_activity,
                "metrics": metrics,
                "recent_tasks": recent_tasks
            }
        else:
            # Return status for all agents
            status = {}
            for aid in self.agents:
                status[aid] = await self.get_agent_status(aid)
            return status
    
    async def get_agents_by_capability(self, capability: CapabilityName) -> List[AgentID]:
        """Get list of agents that support a specific capability."""
        return list(self.capability_index.get(capability, set()))
    
    async def get_agents_by_type(self, agent_type: AgentType) -> List[AgentID]:
        """Get list of agents of a specific type."""
        return list(self.type_index.get(agent_type, set()))
    
    def add_status_callback(self, callback: AgentStatusCallback):
        """Add a callback for agent status changes."""
        self.status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: AgentStatusCallback):
        """Remove a status callback."""
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
    
    def _select_best_agent(self, available_agents: List[AgentID], task_type: str) -> AgentID:
        """Select the best agent from available candidates."""
        if len(available_agents) == 1:
            return available_agents[0]
        
        # Score agents based on multiple criteria
        agent_scores = []
        
        for agent_id in available_agents:
            agent_capability = self.agents[agent_id]
            
            # Base score from performance
            score = agent_capability.performance_score
            
            # Adjust for current load (prefer less loaded agents)
            load_factor = 1.0 - agent_capability.utilization_rate
            score *= load_factor
            
            # Adjust for specialization (prefer specialized agents)
            if task_type in agent_capability.specializations:
                score *= 1.2
            
            # Adjust for recent activity (prefer recently active agents)
            if agent_capability.last_activity:
                time_since_activity = (datetime.utcnow() - agent_capability.last_activity).total_seconds()
                if time_since_activity < 300:  # 5 minutes
                    score *= 1.1
            
            agent_scores.append((score, agent_id))
        
        # Sort by score and return best agent
        agent_scores.sort(reverse=True)
        return agent_scores[0][1]
    
    async def _start_health_monitoring(self, agent_id: AgentID):
        """Start health monitoring for an agent."""
        async def health_check_loop():
            while self.is_running and agent_id in self.agents:
                try:
                    await self._perform_health_check(agent_id)
                    await asyncio.sleep(self.config.agent_health_check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check error for agent {agent_id}: {e}")
                    await asyncio.sleep(self.config.agent_health_check_interval)
        
        task = asyncio.create_task(health_check_loop())
        self.health_check_tasks[agent_id] = task
    
    async def _perform_health_check(self, agent_id: AgentID):
        """Perform health check on an agent."""
        agent_capability = self.agents.get(agent_id)
        if not agent_capability:
            return
        
        try:
            # Check if agent has been inactive for too long
            if agent_capability.last_activity:
                time_since_activity = (datetime.utcnow() - agent_capability.last_activity).total_seconds()
                if time_since_activity > self.config.agent_timeout:
                    agent_capability.is_available = False
                    logger.warning(f"Agent {agent_id} marked as unavailable due to inactivity")
                else:
                    agent_capability.is_available = True
            
            # Additional health checks could be added here
            # e.g., ping agent, check resource usage, etc.
            
        except Exception as e:
            logger.error(f"Health check failed for agent {agent_id}: {e}")
            agent_capability.is_available = False
    
    def _update_agent_performance(self, agent_id: AgentID, success: bool, execution_time: float):
        """Update agent performance metrics."""
        agent_capability = self.agents.get(agent_id)
        if not agent_capability:
            return
        
        # Update performance score using exponential moving average
        alpha = 0.1
        current_performance = 1.0 if success else 0.0
        agent_capability.performance_score = (
            alpha * current_performance + 
            (1 - alpha) * agent_capability.performance_score
        )
        
        # Update agent weight for load balancing
        self.agent_weights[agent_id] = agent_capability.performance_score
        
        # Update metrics
        metrics = self.agent_metrics.get(agent_id, PerformanceMetrics())
        metrics.total_tasks += 1
        if success:
            metrics.completed_tasks += 1
        else:
            metrics.failed_tasks += 1
        
        # Update average execution time
        if metrics.avg_task_execution_time == 0:
            metrics.avg_task_execution_time = execution_time
        else:
            metrics.avg_task_execution_time = (
                alpha * execution_time + 
                (1 - alpha) * metrics.avg_task_execution_time
            )
        
        metrics.task_success_rate = metrics.completed_tasks / max(metrics.total_tasks, 1)
        metrics.last_updated = datetime.utcnow()
    
    async def _notify_status_callbacks(self, agent_capability: AgentCapability):
        """Notify all status callbacks of agent changes."""
        for callback in self.status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_capability)
                else:
                    callback(agent_capability)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    async def get_registry_metrics(self) -> Dict[str, Any]:
        """Get comprehensive registry metrics."""
        total_agents = len(self.agents)
        available_agents = sum(1 for a in self.agents.values() if a.is_available)
        
        # Calculate average utilization
        if total_agents > 0:
            avg_utilization = sum(a.utilization_rate for a in self.agents.values()) / total_agents
            avg_performance = sum(a.performance_score for a in self.agents.values()) / total_agents
        else:
            avg_utilization = 0.0
            avg_performance = 0.0
        
        # Calculate task statistics
        total_tasks = sum(len(history) for history in self.task_history.values())
        recent_tasks = []
        for history in self.task_history.values():
            recent_tasks.extend([
                task for task in history 
                if 'completed_at' in task and 
                (datetime.utcnow() - task['completed_at']).total_seconds() < 300
            ])
        
        if recent_tasks:
            success_rate = sum(1 for task in recent_tasks if task.get('success', False)) / len(recent_tasks)
        else:
            success_rate = 1.0
        
        return {
            'total_agents': total_agents,
            'available_agents': available_agents,
            'agent_availability_rate': available_agents / max(total_agents, 1),
            'avg_agent_utilization': avg_utilization,
            'avg_agent_performance': avg_performance,
            'total_tasks_processed': total_tasks,
            'recent_success_rate': success_rate,
            'capabilities': list(self.capability_index.keys()),
            'agent_types': [t.value for t in self.type_index.keys()],
            'active_health_checks': len(self.health_check_tasks)
        }
    
    # Phase 1.6: Agent Registry Evolution Integration Methods
    
    async def register_agent_with_evolution(self, 
                                           agent_capability: AgentCapability,
                                           evolution_metadata: Optional[EvolutionMetadata] = None,
                                           parent_agents: Optional[List[str]] = None) -> bool:
        """Register agent with evolutionary capabilities metadata (1.6.1)."""
        try:
            # Register base agent first
            if not await self.register_agent(agent_capability):
                return False
            
            agent_id = agent_capability.agent_id
            
            # Initialize evolution metadata
            if evolution_metadata is None:
                evolution_metadata = EvolutionMetadata(
                    generation=self.evolution_generation_counter,
                    fitness_score=agent_capability.performance_score,
                    parent_agents=parent_agents or [],
                    last_evolution=datetime.utcnow()
                )
            
            self.evolution_metadata[agent_id] = evolution_metadata
            
            # Initialize lineage tracking (1.6.4)
            lineage = AgentLineage(
                agent_id=agent_id,
                lineage_id=str(uuid.uuid4()),
                ancestors=parent_agents or [],
                lineage_depth=len(parent_agents or []),
                creation_method="spawn" if not parent_agents else "evolution",
                evolutionary_branch="main"
            )
            self.agent_lineages[agent_id] = lineage
            
            # Initialize distributed reputation (1.6.6)
            reputation = DistributedReputation(
                agent_id=agent_id,
                global_fitness=evolution_metadata.fitness_score,
                reputation_consensus=agent_capability.performance_score,
                trust_score=1.0,  # Start with full trust
                last_reputation_update=datetime.utcnow()
            )
            self.distributed_reputations[agent_id] = reputation
            
            # Generate A2A agent card with evolution history (1.6.2)
            await self._generate_a2a_agent_card(agent_id)
            
            # Track capability evolution (1.6.3)
            await self._track_capability_evolution(agent_id, "registration", {
                "capabilities": list(agent_capability.supported_tasks),
                "generation": evolution_metadata.generation,
                "fitness": evolution_metadata.fitness_score
            })
            
            logger.info(f"Registered agent with evolution: {agent_id} (gen {evolution_metadata.generation})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent with evolution {agent_capability.name}: {e}")
            return False
    
    async def _generate_a2a_agent_card(self, agent_id: AgentID) -> Optional[Any]:
        """Generate A2A agent card with evolution history (1.6.2)."""
        try:
            if AgentCard is None:
                logger.warning("A2A AgentCard not available")
                return None
            
            agent_capability = self.agents.get(agent_id)
            evolution_meta = self.evolution_metadata.get(agent_id)
            lineage = self.agent_lineages.get(agent_id)
            reputation = self.distributed_reputations.get(agent_id)
            
            if not agent_capability:
                return None
            
            # Build performance metrics for A2A discovery (1.6.5)
            performance_metrics = {
                "performance_score": agent_capability.performance_score,
                "utilization_rate": agent_capability.utilization_rate,
                "success_rate": self.agent_metrics.get(agent_id, PerformanceMetrics()).task_success_rate,
                "avg_execution_time": self.agent_metrics.get(agent_id, PerformanceMetrics()).avg_task_execution_time,
                "evolution_fitness": evolution_meta.fitness_score if evolution_meta else 0.0,
                "reputation_score": reputation.reputation_consensus if reputation else 0.0,
                "trust_score": reputation.trust_score if reputation else 1.0,
                "genetic_diversity": lineage.genetic_diversity_score if lineage else 0.0
            }
            
            # Create A2A agent card
            agent_card = AgentCard(
                agent_id=agent_id,
                name=agent_capability.name,
                description=f"Evolved agent (gen {evolution_meta.generation if evolution_meta else 0})",
                capabilities=list(agent_capability.supported_tasks),
                communication_protocols=["A2A", "HTTP", "WebSocket"],
                supported_tasks=list(agent_capability.supported_tasks),
                performance_metrics=performance_metrics,
                evolution_generation=evolution_meta.generation if evolution_meta else 0,
                evolution_fitness=evolution_meta.fitness_score if evolution_meta else 0.0,
                evolution_lineage=lineage.ancestors if lineage else [],
                last_updated=datetime.utcnow(),
                availability_status="available" if agent_capability.is_available else "busy",
                endpoint_url=getattr(self.a2a_server, 'endpoint_url', '') if self.a2a_server else ""
            )
            
            self.local_agent_cards[agent_id] = agent_card
            
            # Register with A2A server if available
            if self.a2a_server and hasattr(self.a2a_server, 'register_local_agent'):
                await self.a2a_server.register_local_agent(agent_card)
            
            logger.debug(f"Generated A2A agent card for {agent_id}")
            return agent_card
            
        except Exception as e:
            logger.error(f"Failed to generate A2A agent card for {agent_id}: {e}")
            return None
    
    async def _track_capability_evolution(self, agent_id: AgentID, event_type: str, 
                                        event_data: Dict[str, Any]) -> None:
        """Track distributed agent capability evolution (1.6.3)."""
        try:
            evolution_event = {
                "agent_id": agent_id,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "event_data": event_data,
                "generation": self.evolution_metadata.get(agent_id, EvolutionMetadata()).generation
            }
            
            # Track in capability evolution history
            for capability in event_data.get("capabilities", []):
                self.capability_evolution_history[capability].append(evolution_event)
            
            # Update evolution metadata
            if agent_id in self.evolution_metadata:
                self.evolution_metadata[agent_id].evolution_history.append(evolution_event)
            
            # Broadcast to A2A peers if available
            if self.a2a_server and hasattr(self.a2a_server, 'broadcast_evolution_event'):
                await self.a2a_server.broadcast_evolution_event(evolution_event)
            
            logger.debug(f"Tracked capability evolution for {agent_id}: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to track capability evolution for {agent_id}: {e}")
    
    async def update_agent_lineage(self, agent_id: AgentID, 
                                 parent_ids: List[str],
                                 creation_method: str = "evolution") -> bool:
        """Update agent lineage and genealogy tracking (1.6.4)."""
        try:
            if agent_id not in self.agent_lineages:
                return False
            
            lineage = self.agent_lineages[agent_id]
            
            # Update lineage information
            lineage.ancestors.extend(parent_ids)
            lineage.lineage_depth = len(set(lineage.ancestors))
            lineage.creation_method = creation_method
            
            # Calculate genetic diversity score
            lineage.genetic_diversity_score = await self._calculate_genetic_diversity(agent_id)
            
            # Update genealogy tree
            await self._update_genealogy_tree(agent_id)
            
            # Update descendants for parent agents
            for parent_id in parent_ids:
                if parent_id in self.agent_lineages:
                    if agent_id not in self.agent_lineages[parent_id].descendants:
                        self.agent_lineages[parent_id].descendants.append(agent_id)
            
            # Track evolution event
            await self._track_capability_evolution(agent_id, "lineage_update", {
                "parents": parent_ids,
                "creation_method": creation_method,
                "genetic_diversity": lineage.genetic_diversity_score
            })
            
            logger.debug(f"Updated lineage for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent lineage for {agent_id}: {e}")
            return False
    
    async def update_distributed_reputation(self, agent_id: AgentID,
                                          peer_ratings: Optional[Dict[str, float]] = None,
                                          validation_proof: Optional[str] = None) -> bool:
        """Update distributed agent reputation via evolution results (1.6.6)."""
        try:
            if agent_id not in self.distributed_reputations:
                return False
            
            reputation = self.distributed_reputations[agent_id]
            evolution_meta = self.evolution_metadata.get(agent_id)
            agent_metrics = self.agent_metrics.get(agent_id)
            
            # Update peer ratings
            if peer_ratings:
                reputation.peer_ratings.update(peer_ratings)
            
            # Add validation proof
            if validation_proof:
                reputation.validation_proofs.append(validation_proof)
            
            # Calculate reputation consensus using various metrics
            fitness_weight = 0.3
            performance_weight = 0.3
            peer_weight = 0.2
            trust_weight = 0.2
            
            fitness_score = evolution_meta.fitness_score if evolution_meta else 0.0
            performance_score = agent_metrics.task_success_rate if agent_metrics else 0.0
            peer_score = sum(reputation.peer_ratings.values()) / max(len(reputation.peer_ratings), 1)
            
            reputation.reputation_consensus = (
                fitness_weight * fitness_score +
                performance_weight * performance_score +
                peer_weight * peer_score +
                trust_weight * reputation.trust_score
            )
            
            # Update global fitness
            reputation.global_fitness = reputation.reputation_consensus
            reputation.last_reputation_update = datetime.utcnow()
            
            # Broadcast reputation update to A2A network
            if self.a2a_server and hasattr(self.a2a_server, 'broadcast_reputation_update'):
                reputation_data = {
                    "agent_id": agent_id,
                    "reputation_consensus": reputation.reputation_consensus,
                    "global_fitness": reputation.global_fitness,
                    "peer_ratings_count": len(reputation.peer_ratings),
                    "validation_proofs_count": len(reputation.validation_proofs)
                }
                await self.a2a_server.broadcast_reputation_update(reputation_data)
            
            # Update A2A agent card with new reputation
            await self._generate_a2a_agent_card(agent_id)
            
            logger.debug(f"Updated distributed reputation for {agent_id}: {reputation.reputation_consensus:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update distributed reputation for {agent_id}: {e}")
            return False
    
    async def get_evolutionary_performance_metrics(self, agent_id: Optional[AgentID] = None) -> Dict[str, Any]:
        """Get evolutionary performance metrics for A2A discovery (1.6.5)."""
        try:
            if agent_id:
                # Single agent metrics
                if agent_id not in self.agents:
                    return {"error": f"Agent {agent_id} not found"}
                
                evolution_meta = self.evolution_metadata.get(agent_id, EvolutionMetadata())
                lineage = self.agent_lineages.get(agent_id, AgentLineage(agent_id=agent_id, lineage_id=""))
                reputation = self.distributed_reputations.get(agent_id, DistributedReputation(agent_id=agent_id))
                agent_metrics = self.agent_metrics.get(agent_id, PerformanceMetrics())
                
                return {
                    "agent_id": agent_id,
                    "evolution_generation": evolution_meta.generation,
                    "fitness_score": evolution_meta.fitness_score,
                    "genetic_diversity": lineage.genetic_diversity_score,
                    "reputation_consensus": reputation.reputation_consensus,
                    "global_fitness": reputation.global_fitness,
                    "trust_score": reputation.trust_score,
                    "performance_score": self.agents[agent_id].performance_score,
                    "success_rate": agent_metrics.task_success_rate,
                    "lineage_depth": lineage.lineage_depth,
                    "peer_ratings_count": len(reputation.peer_ratings),
                    "validation_proofs": len(reputation.validation_proofs),
                    "last_evolution": evolution_meta.last_evolution.isoformat() if evolution_meta.last_evolution else None
                }
            else:
                # All agents metrics
                all_metrics = {}
                for aid in self.agents:
                    all_metrics[aid] = await self.get_evolutionary_performance_metrics(aid)
                
                # Add aggregate metrics
                if all_metrics:
                    total_agents = len(all_metrics)
                    avg_fitness = sum(m.get("fitness_score", 0) for m in all_metrics.values()) / total_agents
                    avg_diversity = sum(m.get("genetic_diversity", 0) for m in all_metrics.values()) / total_agents
                    avg_reputation = sum(m.get("reputation_consensus", 0) for m in all_metrics.values()) / total_agents
                    max_generation = max(m.get("evolution_generation", 0) for m in all_metrics.values())
                    
                    all_metrics["_aggregate"] = {
                        "total_agents": total_agents,
                        "avg_fitness_score": avg_fitness,
                        "avg_genetic_diversity": avg_diversity,
                        "avg_reputation": avg_reputation,
                        "max_generation": max_generation,
                        "evolution_generation_counter": self.evolution_generation_counter
                    }
                
                return all_metrics
                
        except Exception as e:
            logger.error(f"Failed to get evolutionary performance metrics: {e}")
            return {"error": str(e)}
    
    async def _calculate_genetic_diversity(self, agent_id: AgentID) -> float:
        """Calculate genetic diversity score for agent lineage."""
        try:
            lineage = self.agent_lineages.get(agent_id)
            if not lineage or not lineage.ancestors:
                return 0.0
            
            # Count unique ancestor capabilities
            unique_capabilities = set()
            for ancestor_id in lineage.ancestors:
                if ancestor_id in self.agents:
                    unique_capabilities.update(self.agents[ancestor_id].supported_tasks)
            
            # Calculate diversity based on capability spread and lineage depth
            capability_diversity = len(unique_capabilities) / max(len(lineage.ancestors) * 10, 1)
            lineage_diversity = min(lineage.lineage_depth / 10.0, 1.0)
            
            return (capability_diversity + lineage_diversity) / 2.0
            
        except Exception as e:
            logger.error(f"Failed to calculate genetic diversity for {agent_id}: {e}")
            return 0.0
    
    async def _update_genealogy_tree(self, agent_id: AgentID) -> None:
        """Update genealogy tree structure."""
        try:
            lineage = self.agent_lineages.get(agent_id)
            if not lineage:
                return
            
            # Build tree structure
            tree_node = {
                "agent_id": agent_id,
                "generation": self.evolution_metadata.get(agent_id, EvolutionMetadata()).generation,
                "parents": lineage.ancestors,
                "children": lineage.descendants,
                "fitness": self.evolution_metadata.get(agent_id, EvolutionMetadata()).fitness_score,
                "creation_method": lineage.creation_method,
                "genetic_diversity": lineage.genetic_diversity_score
            }
            
            # Store in lineage trees
            if lineage.lineage_id not in self.lineage_trees:
                self.lineage_trees[lineage.lineage_id] = {"nodes": {}, "root": agent_id}
            
            self.lineage_trees[lineage.lineage_id]["nodes"][agent_id] = tree_node
            
        except Exception as e:
            logger.error(f"Failed to update genealogy tree for {agent_id}: {e}")
    
    async def get_agent_lineage_info(self, agent_id: AgentID) -> Dict[str, Any]:
        """Get comprehensive agent lineage information."""
        try:
            if agent_id not in self.agent_lineages:
                return {"error": f"No lineage info for agent {agent_id}"}
            
            lineage = self.agent_lineages[agent_id]
            evolution_meta = self.evolution_metadata.get(agent_id, EvolutionMetadata())
            
            return {
                "agent_id": agent_id,
                "lineage_id": lineage.lineage_id,
                "ancestors": lineage.ancestors,
                "descendants": lineage.descendants,
                "lineage_depth": lineage.lineage_depth,
                "creation_method": lineage.creation_method,
                "evolutionary_branch": lineage.evolutionary_branch,
                "genetic_diversity_score": lineage.genetic_diversity_score,
                "generation": evolution_meta.generation,
                "fitness_score": evolution_meta.fitness_score,
                "genealogy_tree": self.lineage_trees.get(lineage.lineage_id, {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get lineage info for agent {agent_id}: {e}")
            return {"error": str(e)}
    
    async def sync_a2a_agent_cards(self) -> Dict[str, Any]:
        """Sync agent cards with A2A network and update from peer discoveries."""
        try:
            if not self.a2a_server:
                return {"error": "A2A server not available"}
            
            # Update all local agent cards
            for agent_id in self.agents:
                await self._generate_a2a_agent_card(agent_id)
            
            # Discover peer agent cards
            if hasattr(self.a2a_server, 'discover_peer_agents'):
                peer_cards = await self.a2a_server.discover_peer_agents()
                
                # Process discovered peer cards
                for peer_id, cards in peer_cards.items():
                    self.peer_agent_cards[peer_id] = cards
                    
                    # Update reputation based on peer feedback
                    for card in cards:
                        if hasattr(card, 'performance_metrics') and 'reputation_feedback' in card.performance_metrics:
                            await self.update_distributed_reputation(
                                card.agent_id,
                                peer_ratings={peer_id: card.performance_metrics['reputation_feedback']}
                            )
            
            return {
                "local_cards": len(self.local_agent_cards),
                "peer_cards": {pid: len(cards) for pid, cards in self.peer_agent_cards.items()},
                "total_network_agents": len(self.local_agent_cards) + sum(len(cards) for cards in self.peer_agent_cards.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to sync A2A agent cards: {e}")
            return {"error": str(e)}