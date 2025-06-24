"""
Evolutionary Negotiation Protocols - Phase 2.3 Implementation
Aligned with Sakana AI Darwin Gödel Machine (DGM) research and A2A protocol integration.

This module implements evolutionary negotiation mechanisms that enable agents
to negotiate tasks, resources, and strategies based on their evolutionary fitness
and adaptive capabilities, coordinated through A2A protocols.
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class NegotiationType(Enum):
    """Types of negotiation."""
    TASK_ASSIGNMENT = "task_assignment"
    RESOURCE_ALLOCATION = "resource_allocation"
    COALITION_FORMATION = "coalition_formation"
    STRATEGY_SELECTION = "strategy_selection"
    CONFLICT_RESOLUTION = "conflict_resolution"
    WORKLOAD_DISTRIBUTION = "workload_distribution"


class NegotiationStrategy(Enum):
    """Negotiation strategies."""
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    ADAPTIVE = "adaptive"
    EVOLUTIONARY = "evolutionary"
    CONSENSUS_SEEKING = "consensus_seeking"
    FITNESS_BASED = "fitness_based"


class CoalitionType(Enum):
    """Types of coalitions."""
    TEMPORARY = "temporary"
    PERSISTENT = "persistent"
    TASK_SPECIFIC = "task_specific"
    CAPABILITY_BASED = "capability_based"
    FITNESS_BASED = "fitness_based"


@dataclass
class NegotiationProposal:
    """Represents a negotiation proposal."""
    id: str
    proposer_id: str
    negotiation_type: NegotiationType
    target_agents: List[str]
    proposal_details: Dict[str, Any]
    fitness_requirements: Dict[str, float]
    resource_requirements: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    deadline: Optional[float]
    priority: float
    timestamp: float
    status: str = "pending"
    responses: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.responses is None:
            self.responses = {}


@dataclass
class NegotiationResponse:
    """Response to a negotiation proposal."""
    proposal_id: str
    responder_id: str
    acceptance: bool
    counter_proposal: Optional[Dict[str, Any]]
    fitness_score: float
    capability_match: float
    resource_availability: Dict[str, float]
    conditions: List[str]
    timestamp: float


@dataclass
class Coalition:
    """Represents an agent coalition."""
    id: str
    coalition_type: CoalitionType
    members: List[str]
    leader_id: str
    formation_criteria: Dict[str, Any]
    objectives: List[str]
    fitness_distribution: Dict[str, float]  # member_id -> fitness
    resource_pool: Dict[str, Any]
    performance_history: List[Dict[str, Any]]
    status: str
    created_at: float
    expires_at: Optional[float] = None


@dataclass
class ConflictResolution:
    """Represents a conflict resolution process."""
    id: str
    conflict_type: str
    involved_agents: List[str]
    conflict_description: str
    resolution_strategy: str
    mediator_id: Optional[str]
    resolution_steps: List[Dict[str, Any]]
    outcome: Optional[Dict[str, Any]]
    satisfaction_scores: Dict[str, float]  # agent_id -> satisfaction
    timestamp: float


class EvolutionaryNegotiationProtocols:
    """
    Implements evolutionary negotiation mechanisms with A2A coordination.
    
    This class enables agents to negotiate using strategies that evolve based
    on fitness and success rates, aligned with DGM's adaptive optimization principles.
    """
    
    def __init__(self, orchestrator, fitness_weight: float = 0.7):
        self.orchestrator = orchestrator
        self.fitness_weight = fitness_weight
        self.active_negotiations: Dict[str, NegotiationProposal] = {}
        self.negotiation_history: List[Dict[str, Any]] = []
        self.coalitions: Dict[str, Coalition] = {}
        self.conflict_resolutions: Dict[str, ConflictResolution] = {}
        self.agent_fitness_cache: Dict[str, float] = {}
        self.negotiation_strategies: Dict[str, NegotiationStrategy] = {}
        self.success_metrics: Dict[str, List[float]] = {}
        self.adaptive_parameters: Dict[str, float] = {
            'cooperation_tendency': 0.5,
            'competition_threshold': 0.3,
            'coalition_formation_bias': 0.6,
            'fitness_influence': 0.7
        }
    
    async def start_negotiation_session(self) -> str:
        """Start a new negotiation session."""
        session_id = f"negotiation_{int(time.time())}_{hash(time.time()) % 10000}"
        
        # Broadcast negotiation session start to A2A network
        if hasattr(self.orchestrator, 'a2a_server'):
            await self.orchestrator.a2a_server.broadcast_evolution_event({
                'event_type': 'negotiation_session_start',
                'session_id': session_id,
                'coordinator': self.orchestrator.agent_id,
                'timestamp': time.time()
            })
        
        return session_id
    
    async def create_evolution_aware_task_negotiation(
        self,
        task_details: Dict[str, Any],
        candidate_agents: List[str] = None
    ) -> NegotiationProposal:
        """
        Create evolution-aware task negotiation strategies.
        
        Aligns with DGM's fitness-based decision making.
        """
        proposal_id = f"task_neg_{hashlib.md5(str(task_details).encode()).hexdigest()[:8]}"
        
        # Get candidate agents based on capabilities and fitness
        if candidate_agents is None:
            peers = await self.orchestrator.get_peer_agents()
            candidate_agents = await self._select_candidates_by_fitness(
                peers, task_details.get('required_capabilities', [])
            )
        
        # Calculate fitness requirements based on task complexity
        fitness_requirements = await self._calculate_fitness_requirements(task_details)
        
        proposal = NegotiationProposal(
            id=proposal_id,
            proposer_id=self.orchestrator.agent_id,
            negotiation_type=NegotiationType.TASK_ASSIGNMENT,
            target_agents=candidate_agents,
            proposal_details=task_details,
            fitness_requirements=fitness_requirements,
            resource_requirements=task_details.get('resources', {}),
            expected_outcome=task_details.get('expected_outcome', {}),
            deadline=task_details.get('deadline'),
            priority=task_details.get('priority', 0.5),
            timestamp=time.time()
        )
        
        self.active_negotiations[proposal_id] = proposal
        
        # Send negotiation requests via A2A
        await self._broadcast_negotiation_proposal(proposal)
        
        return proposal
    
    async def implement_fitness_based_negotiation(
        self,
        proposal: NegotiationProposal,
        agent_id: str
    ) -> NegotiationResponse:
        """
        Implement A2A negotiation based on evolutionary fitness.
        
        Uses DGM's empirical performance optimization.
        """
        # Get agent's current fitness
        agent_fitness = await self._get_agent_fitness(agent_id)
        
        # Calculate capability match
        capability_match = await self._calculate_capability_match(
            proposal.proposal_details, agent_id
        )
        
        # Determine acceptance based on fitness and capability
        acceptance_threshold = proposal.fitness_requirements.get('minimum_fitness', 0.5)
        capability_threshold = 0.6
        
        # Evolutionary decision making
        fitness_factor = agent_fitness * self.fitness_weight
        capability_factor = capability_match * (1 - self.fitness_weight)
        combined_score = fitness_factor + capability_factor
        
        acceptance = (
            agent_fitness >= acceptance_threshold and
            capability_match >= capability_threshold and
            combined_score >= 0.7
        )
        
        # Generate counter-proposal if not accepting
        counter_proposal = None
        if not acceptance and combined_score >= 0.5:
            counter_proposal = await self._generate_counter_proposal(proposal, agent_id)
        
        response = NegotiationResponse(
            proposal_id=proposal.id,
            responder_id=agent_id,
            acceptance=acceptance,
            counter_proposal=counter_proposal,
            fitness_score=agent_fitness,
            capability_match=capability_match,
            resource_availability=await self._get_resource_availability(agent_id),
            conditions=await self._generate_conditions(proposal, agent_id),
            timestamp=time.time()
        )
        
        # Update proposal with response
        proposal.responses[agent_id] = asdict(response)
        
        return response
    
    async def add_adaptive_negotiation_strategies(
        self,
        agent_id: str,
        performance_history: List[Dict[str, Any]]
    ) -> NegotiationStrategy:
        """
        Add adaptive negotiation strategies that evolve over time.
        
        Implements DGM's adaptive strategy evolution.
        """
        # Analyze historical performance
        success_rate = await self._calculate_negotiation_success_rate(performance_history)
        cooperation_benefit = await self._analyze_cooperation_benefits(performance_history)
        competition_effectiveness = await self._analyze_competition_effectiveness(performance_history)
        
        # Evolutionary strategy selection
        if success_rate < 0.4:
            # Low success rate - try adaptive approach
            strategy = NegotiationStrategy.ADAPTIVE
            self.adaptive_parameters['cooperation_tendency'] += 0.1
        elif cooperation_benefit > competition_effectiveness:
            # Cooperation is more beneficial
            strategy = NegotiationStrategy.COOPERATIVE
            self.adaptive_parameters['cooperation_tendency'] += 0.05
        elif competition_effectiveness > 0.7:
            # Competition is effective
            strategy = NegotiationStrategy.COMPETITIVE
            self.adaptive_parameters['competition_threshold'] -= 0.05
        else:
            # Use fitness-based approach
            strategy = NegotiationStrategy.FITNESS_BASED
        
        # Update agent's strategy
        self.negotiation_strategies[agent_id] = strategy
        
        # Broadcast strategy evolution to A2A network
        if hasattr(self.orchestrator, 'a2a_server'):
            await self.orchestrator.a2a_server.broadcast_evolution_event({
                'event_type': 'negotiation_strategy_evolved',
                'agent_id': agent_id,
                'new_strategy': strategy.value,
                'success_rate': success_rate,
                'timestamp': time.time()
            })
        
        return strategy
    
    async def create_evolutionary_coalition_formation(
        self,
        coalition_criteria: Dict[str, Any],
        target_size: int = None
    ) -> Coalition:
        """
        Create evolutionary coalition formation algorithms.
        
        Aligns with DGM's collaborative optimization.
        """
        coalition_id = f"coalition_{int(time.time())}_{hash(str(coalition_criteria)) % 10000}"
        
        # Get available agents
        peers = await self.orchestrator.get_peer_agents()
        candidate_agents = [peer['id'] for peer in peers]
        
        # Score agents for coalition membership
        agent_scores = {}
        for agent_id in candidate_agents:
            score = await self._calculate_coalition_fitness(agent_id, coalition_criteria)
            agent_scores[agent_id] = score
        
        # Select coalition members using evolutionary algorithm
        selected_members = await self._select_coalition_members(
            agent_scores, target_size or min(5, len(candidate_agents))
        )
        
        # Select leader based on highest fitness
        leader_id = max(selected_members, key=lambda aid: agent_scores[aid])
        
        # Calculate fitness distribution
        fitness_distribution = {
            member_id: await self._get_agent_fitness(member_id)
            for member_id in selected_members
        }
        
        coalition = Coalition(
            id=coalition_id,
            coalition_type=CoalitionType.FITNESS_BASED,
            members=selected_members,
            leader_id=leader_id,
            formation_criteria=coalition_criteria,
            objectives=coalition_criteria.get('objectives', []),
            fitness_distribution=fitness_distribution,
            resource_pool=await self._aggregate_coalition_resources(selected_members),
            performance_history=[],
            status="forming",
            created_at=time.time()
        )
        
        self.coalitions[coalition_id] = coalition
        
        # Notify members via A2A
        await self._notify_coalition_formation(coalition)
        
        return coalition
    
    async def implement_resource_allocation_evolution(
        self,
        available_resources: Dict[str, Any],
        requesting_agents: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Implement A2A-enabled resource allocation evolution.
        
        Uses DGM's adaptive resource optimization.
        """
        allocation_results = {
            'allocation_id': f"resource_alloc_{int(time.time())}",
            'total_resources': available_resources,
            'requesting_agents': requesting_agents,
            'allocations': {},
            'efficiency_score': 0.0,
            'fairness_score': 0.0,
            'fitness_impact': {},
            'timestamp': time.time()
        }
        
        # Get agent fitness and resource preferences
        agent_fitness = {}
        agent_preferences = {}
        
        for agent_id in requesting_agents:
            fitness = await self._get_agent_fitness(agent_id)
            preferences = await self._get_resource_preferences(agent_id)
            agent_fitness[agent_id] = fitness
            agent_preferences[agent_id] = preferences
        
        # Evolutionary allocation algorithm
        allocation = await self._evolutionary_resource_allocation(
            available_resources, agent_fitness, agent_preferences
        )
        
        allocation_results['allocations'] = allocation
        allocation_results['efficiency_score'] = await self._calculate_allocation_efficiency(allocation)
        allocation_results['fairness_score'] = await self._calculate_allocation_fairness(allocation, agent_fitness)
        
        # Calculate fitness impact
        for agent_id, resources in allocation.items():
            impact = await self._calculate_fitness_impact(agent_id, resources)
            allocation_results['fitness_impact'][agent_id] = impact
        
        # Broadcast allocation results via A2A
        if hasattr(self.orchestrator, 'a2a_server'):
            await self.orchestrator.a2a_server.broadcast_evolution_event({
                'event_type': 'resource_allocation_complete',
                'allocation_id': allocation_results['allocation_id'],
                'efficiency': allocation_results['efficiency_score'],
                'fairness': allocation_results['fairness_score'],
                'timestamp': time.time()
            })
        
        return allocation_results
    
    async def add_evolutionary_conflict_resolution(
        self,
        conflict_description: str,
        involved_agents: List[str],
        conflict_type: str = "resource_contention"
    ) -> ConflictResolution:
        """
        Add evolutionary conflict resolution mechanisms.
        
        Implements DGM's adaptive conflict resolution.
        """
        resolution_id = f"conflict_{hashlib.md5(conflict_description.encode()).hexdigest()[:8]}"
        
        # Select mediator based on fitness and neutrality
        mediator_id = await self._select_conflict_mediator(involved_agents)
        
        # Determine resolution strategy based on conflict type and agent fitness
        resolution_strategy = await self._determine_resolution_strategy(
            conflict_type, involved_agents
        )
        
        conflict_resolution = ConflictResolution(
            id=resolution_id,
            conflict_type=conflict_type,
            involved_agents=involved_agents,
            conflict_description=conflict_description,
            resolution_strategy=resolution_strategy,
            mediator_id=mediator_id,
            resolution_steps=[],
            outcome=None,
            satisfaction_scores={},
            timestamp=time.time()
        )
        
        # Execute resolution process
        resolution_outcome = await self._execute_conflict_resolution(conflict_resolution)
        conflict_resolution.outcome = resolution_outcome
        
        # Calculate satisfaction scores
        for agent_id in involved_agents:
            satisfaction = await self._calculate_satisfaction(agent_id, resolution_outcome)
            conflict_resolution.satisfaction_scores[agent_id] = satisfaction
        
        self.conflict_resolutions[resolution_id] = conflict_resolution
        
        # Broadcast resolution to A2A network
        if hasattr(self.orchestrator, 'a2a_server'):
            await self.orchestrator.a2a_server.broadcast_evolution_event({
                'event_type': 'conflict_resolved',
                'resolution_id': resolution_id,
                'strategy': resolution_strategy,
                'avg_satisfaction': np.mean(list(conflict_resolution.satisfaction_scores.values())),
                'timestamp': time.time()
            })
        
        return conflict_resolution
    
    # Helper methods
    
    async def _select_candidates_by_fitness(
        self,
        peers: List[Dict[str, Any]],
        required_capabilities: List[str]
    ) -> List[str]:
        """Select candidate agents based on fitness and capabilities."""
        candidates = []
        
        for peer in peers:
            if not required_capabilities or any(
                cap in peer.get('capabilities', []) for cap in required_capabilities
            ):
                fitness = await self._get_agent_fitness(peer['id'])
                if fitness >= 0.5:  # Minimum fitness threshold
                    candidates.append(peer['id'])
        
        # Sort by fitness and return top candidates
        candidates.sort(key=lambda aid: self.agent_fitness_cache.get(aid, 0.0), reverse=True)
        return candidates[:5]  # Limit to top 5 candidates
    
    async def _calculate_fitness_requirements(self, task_details: Dict[str, Any]) -> Dict[str, float]:
        """Calculate fitness requirements for a task."""
        complexity = task_details.get('complexity', 0.5)
        priority = task_details.get('priority', 0.5)
        
        return {
            'minimum_fitness': 0.3 + (complexity * 0.4),
            'preferred_fitness': 0.5 + (priority * 0.4),
            'leadership_fitness': 0.7 + (complexity * 0.2)
        }
    
    async def _broadcast_negotiation_proposal(self, proposal: NegotiationProposal):
        """Broadcast negotiation proposal via A2A."""
        if hasattr(self.orchestrator, 'a2a_server'):
            for agent_id in proposal.target_agents:
                await self.orchestrator.a2a_server.send_message(agent_id, {
                    'type': 'negotiation_proposal',
                    'proposal': asdict(proposal)
                })
    
    async def _get_agent_fitness(self, agent_id: str) -> float:
        """Get agent's current fitness score."""
        if agent_id in self.agent_fitness_cache:
            return self.agent_fitness_cache[agent_id]
          # Calculate fitness based on agent's actual performance metrics
        if self.a2a_server:
            try:
                # Query agent performance from A2A server
                performance_data = await self.a2a_server.get_agent_performance(agent_id)
                if performance_data:
                    # Calculate composite fitness from performance metrics
                    success_rate = performance_data.get('success_rate', 0.5)
                    efficiency = performance_data.get('efficiency', 0.5)
                    reliability = performance_data.get('reliability', 0.5)
                    
                    # Weighted combination of performance metrics
                    fitness = (success_rate * 0.4 + efficiency * 0.3 + reliability * 0.3)
                    self.agent_fitness_cache[agent_id] = fitness
                    return fitness
            except Exception as e:
                self.logger.warning(f"Failed to get performance data for {agent_id}: {e}")
        
        # Fallback: use cached data or default value
        fitness = 0.5  # Neutral fitness when no data available
        self.agent_fitness_cache[agent_id] = fitness
        return fitness
    
    async def _calculate_capability_match(
        self,
        task_details: Dict[str, Any],
        agent_id: str
    ) -> float:
        """Calculate how well agent capabilities match task requirements."""
        required_caps = set(task_details.get('required_capabilities', []))
          # Query agent capabilities from agent registry via A2A server
        agent_caps = set()
        if self.a2a_server:
            try:
                agent_info = await self.a2a_server.get_agent_capabilities(agent_id)
                if agent_info and 'capabilities' in agent_info:
                    agent_caps = set(agent_info['capabilities'])
            except Exception as e:
                self.logger.warning(f"Failed to get capabilities for {agent_id}: {e}")
        
        # Fallback to default capabilities if none retrieved
        if not agent_caps:
            agent_caps = set(['basic_processing', 'task_execution'])
        
        if not required_caps:
            return 1.0
        
        match_count = len(required_caps.intersection(agent_caps))
        return match_count / len(required_caps)
    
    async def _generate_counter_proposal(
        self,
        original_proposal: NegotiationProposal,
        agent_id: str
    ) -> Dict[str, Any]:
        """Generate a counter-proposal."""
        return {
            'modified_deadline': original_proposal.deadline + 3600 if original_proposal.deadline else None,
            'resource_adjustment': {'cpu': 0.8, 'memory': 0.9},
            'priority_adjustment': max(0.1, original_proposal.priority - 0.2),
            'collaboration_terms': ['require_backup_agent', 'shared_resources']
        }
    
    async def _get_resource_availability(self, agent_id: str) -> Dict[str, float]:
        """Get agent's resource availability."""
        return {
            'cpu': np.random.uniform(0.3, 1.0),
            'memory': np.random.uniform(0.2, 0.9),
            'storage': np.random.uniform(0.5, 1.0),
            'network': np.random.uniform(0.7, 1.0)
        }
    
    async def _generate_conditions(
        self,
        proposal: NegotiationProposal,
        agent_id: str
    ) -> List[str]:
        """Generate conditions for accepting the proposal."""
        conditions = []
        
        if proposal.priority > 0.8:
            conditions.append('high_priority_acknowledgment')
        
        if proposal.deadline and (proposal.deadline - time.time()) < 3600:
            conditions.append('urgent_task_bonus')
        
        return conditions
    
    async def _calculate_negotiation_success_rate(
        self,
        history: List[Dict[str, Any]]
    ) -> float:
        """Calculate negotiation success rate from history."""
        if not history:
            return 0.5
        
        successful = sum(1 for h in history if h.get('outcome') == 'success')
        return successful / len(history)
    
    async def _analyze_cooperation_benefits(
        self,
        history: List[Dict[str, Any]]
    ) -> float:
        """Analyze benefits of cooperative strategies."""
        cooperative_outcomes = [
            h.get('benefit_score', 0.0) for h in history 
            if h.get('strategy') == 'cooperative'
        ]
        return np.mean(cooperative_outcomes) if cooperative_outcomes else 0.5
    
    async def _analyze_competition_effectiveness(
        self,
        history: List[Dict[str, Any]]
    ) -> float:
        """Analyze effectiveness of competitive strategies."""
        competitive_outcomes = [
            h.get('benefit_score', 0.0) for h in history 
            if h.get('strategy') == 'competitive'
        ]
        return np.mean(competitive_outcomes) if competitive_outcomes else 0.5

    async def _calculate_coalition_fitness(
        self,
        agent_id: str,
        criteria: Dict[str, Any]
    ) -> float:
        """Calculate agent's fitness for coalition membership."""
        agent_fitness = await self._get_agent_fitness(agent_id)
        
        # Calculate capability match based on actual requirements
        capability_match = 1.0
        if 'required_capabilities' in criteria:
            capability_match = await self._calculate_capability_match(
                {'required_capabilities': criteria['required_capabilities']}, 
                agent_id
            )
        
        return (agent_fitness * 0.7) + (capability_match * 0.3)
    
    async def _select_coalition_members(
        self,
        agent_scores: Dict[str, float],
        target_size: int
    ) -> List[str]:
        """Select coalition members using evolutionary selection."""
        # Sort by score and select top candidates
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Add some diversity by including lower-scoring agents occasionally
        selected = []
        for i, (agent_id, score) in enumerate(sorted_agents):
            if len(selected) >= target_size:
                break
            
            if i < target_size * 0.7 or np.random.random() < 0.2:
                selected.append(agent_id)
        
        return selected[:target_size]
    
    async def _aggregate_coalition_resources(
        self,
        member_ids: List[str]
    ) -> Dict[str, Any]:
        """Aggregate resources from coalition members."""
        total_resources = {'cpu': 0.0, 'memory': 0.0, 'storage': 0.0}
        
        for member_id in member_ids:
            resources = await self._get_resource_availability(member_id)
            for resource, amount in resources.items():
                if resource in total_resources:
                    total_resources[resource] += amount
        
        return total_resources
    
    async def _notify_coalition_formation(self, coalition: Coalition):
        """Notify members of coalition formation via A2A."""
        if hasattr(self.orchestrator, 'a2a_server'):
            for member_id in coalition.members:
                await self.orchestrator.a2a_server.send_message(member_id, {
                    'type': 'coalition_invitation',
                    'coalition': asdict(coalition)
                })
    
    async def _get_resource_preferences(self, agent_id: str) -> Dict[str, float]:
        """Get agent's resource preferences."""
        return {
            'cpu': np.random.uniform(0.5, 1.0),
            'memory': np.random.uniform(0.3, 0.8),
            'storage': np.random.uniform(0.2, 0.6),
            'network': np.random.uniform(0.6, 1.0)
        }
    
    async def _evolutionary_resource_allocation(
        self,
        available_resources: Dict[str, Any],
        agent_fitness: Dict[str, float],
        agent_preferences: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform evolutionary resource allocation."""
        allocation = {}
        
        # Fitness-weighted allocation
        total_fitness = sum(agent_fitness.values())
        
        for agent_id, fitness in agent_fitness.items():
            fitness_ratio = fitness / total_fitness if total_fitness > 0 else 1 / len(agent_fitness)
            
            agent_allocation = {}
            for resource, total_amount in available_resources.items():
                if isinstance(total_amount, (int, float)):
                    # Apply preference weighting
                    preference = agent_preferences[agent_id].get(resource, 0.5)
                    allocated_amount = total_amount * fitness_ratio * preference
                    agent_allocation[resource] = allocated_amount
            
            allocation[agent_id] = agent_allocation
        
        return allocation
    
    async def _calculate_allocation_efficiency(self, allocation: Dict[str, Dict[str, Any]]) -> float:
        """Calculate allocation efficiency."""
        # Simple efficiency metric based on resource utilization
        total_allocated = 0
        total_possible = 0
        
        for agent_allocation in allocation.values():
            for resource, amount in agent_allocation.items():
                if isinstance(amount, (int, float)):
                    total_allocated += amount
                    total_possible += 1.0  # Normalized resource units
        
        return total_allocated / total_possible if total_possible > 0 else 0.0
    
    async def _calculate_allocation_fairness(
        self,
        allocation: Dict[str, Dict[str, Any]],
        agent_fitness: Dict[str, float]
    ) -> float:
        """Calculate allocation fairness score."""
        # Calculate variance in fitness-adjusted allocation
        fitness_adjusted_allocation = []
        
        for agent_id, agent_allocation in allocation.items():
            total_allocation = sum(
                amount for amount in agent_allocation.values() 
                if isinstance(amount, (int, float))
            )
            fitness = agent_fitness.get(agent_id, 0.5)
            adjusted = total_allocation / fitness if fitness > 0 else total_allocation
            fitness_adjusted_allocation.append(adjusted)
        
        # Lower variance = higher fairness
        variance = np.var(fitness_adjusted_allocation) if fitness_adjusted_allocation else 0
        return max(0.0, 1.0 - variance)
      async def _calculate_fitness_impact(
        self,
        agent_id: str,
        allocated_resources: Dict[str, Any]
    ) -> float:
        """Calculate fitness impact of resource allocation."""
        # Calculate impact based on agent performance and resource utilization
        current_fitness = await self._get_agent_fitness(agent_id)
        
        # Calculate resource utilization efficiency
        resource_sum = sum(
            amount for amount in allocated_resources.values() 
            if isinstance(amount, (int, float))
        )
        
        # Model impact based on agent's current capacity and resource needs
        if self.a2a_server:
            try:
                agent_capacity = await self.a2a_server.get_agent_capacity(agent_id)
                if agent_capacity:
                    utilization_ratio = resource_sum / agent_capacity.get('max_capacity', 100)
                    # Optimal utilization is around 70-80%
                    efficiency = 1.0 - abs(utilization_ratio - 0.75) / 0.75
                    return min(1.0, current_fitness * efficiency)
            except Exception as e:
                self.logger.warning(f"Failed to get capacity for {agent_id}: {e}")
        
        # Fallback calculation
        return min(1.0, current_fitness * min(1.0, resource_sum * 0.1))
    
    async def _select_conflict_mediator(self, involved_agents: List[str]) -> str:
        """Select a mediator for conflict resolution."""
        # Get available mediators (agents not involved in conflict)
        peers = await self.orchestrator.get_peer_agents()
        available_mediators = [
            peer['id'] for peer in peers 
            if peer['id'] not in involved_agents and peer['id'] != self.orchestrator.agent_id
        ]
        
        if not available_mediators:
            return self.orchestrator.agent_id  # Self-mediate if necessary
        
        # Select mediator with highest fitness
        best_mediator = available_mediators[0]
        best_fitness = await self._get_agent_fitness(best_mediator)
        
        for mediator_id in available_mediators[1:]:
            fitness = await self._get_agent_fitness(mediator_id)
            if fitness > best_fitness:
                best_fitness = fitness
                best_mediator = mediator_id
        
        return best_mediator
    
    async def _determine_resolution_strategy(
        self,
        conflict_type: str,
        involved_agents: List[str]
    ) -> str:
        """Determine resolution strategy based on conflict type."""
        if conflict_type == "resource_contention":
            return "resource_redistribution"
        elif conflict_type == "task_assignment":
            return "capability_based_assignment"
        elif conflict_type == "priority_conflict":
            return "fitness_based_prioritization"
        else:
            return "mediated_negotiation"
    
    async def _execute_conflict_resolution(
        self,
        conflict_resolution: ConflictResolution
    ) -> Dict[str, Any]:
        """Execute the conflict resolution process."""
        strategy = conflict_resolution.resolution_strategy
        
        if strategy == "resource_redistribution":
            return await self._redistribute_resources(conflict_resolution.involved_agents)
        elif strategy == "capability_based_assignment":
            return await self._reassign_based_on_capabilities(conflict_resolution.involved_agents)
        elif strategy == "fitness_based_prioritization":
            return await self._prioritize_by_fitness(conflict_resolution.involved_agents)
        else:
            return await self._mediated_resolution(conflict_resolution)
    
    async def _redistribute_resources(self, agents: List[str]) -> Dict[str, Any]:
        """Redistribute resources among conflicting agents."""
        return {
            'resolution_type': 'resource_redistribution',
            'new_allocation': {agent_id: {'cpu': 0.5, 'memory': 0.5} for agent_id in agents},
            'timestamp': time.time()
        }
    
    async def _reassign_based_on_capabilities(self, agents: List[str]) -> Dict[str, Any]:
        """Reassign tasks based on agent capabilities."""
        return {
            'resolution_type': 'capability_based_assignment',
            'new_assignments': {agent_id: f'task_suited_for_{agent_id}' for agent_id in agents},
            'timestamp': time.time()
        }
    
    async def _prioritize_by_fitness(self, agents: List[str]) -> Dict[str, Any]:
        """Prioritize agents based on fitness scores."""
        agent_fitness = {agent_id: await self._get_agent_fitness(agent_id) for agent_id in agents}
        sorted_agents = sorted(agent_fitness.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'resolution_type': 'fitness_based_prioritization',
            'priority_order': [agent_id for agent_id, _ in sorted_agents],
            'fitness_scores': agent_fitness,
            'timestamp': time.time()
        }
    
    async def _mediated_resolution(self, conflict_resolution: ConflictResolution) -> Dict[str, Any]:
        """Execute mediated resolution."""
        return {
            'resolution_type': 'mediated_negotiation',
            'mediator': conflict_resolution.mediator_id,
            'agreement': 'compromise_solution',
            'timestamp': time.time()
        }
    
    async def _calculate_satisfaction(
        self,
        agent_id: str,
        resolution_outcome: Dict[str, Any]
    ) -> float:
        """Calculate agent satisfaction with resolution outcome."""
        # Calculate satisfaction based on actual outcome metrics
        satisfaction = 0.5  # Base satisfaction
        
        # Factor in outcome success rate
        if resolution_outcome.get("success", False):
            satisfaction += 0.3
            
        # Factor in resource allocation fairness
        resource_efficiency = resolution_outcome.get("resource_efficiency", 0.5)
        satisfaction += resource_efficiency * 0.2
        
        # Factor in time to resolution
        resolution_time = resolution_outcome.get("resolution_time", 1.0)
        time_factor = max(0, 1 - (resolution_time / 10))  # Penalty for long resolution times
        satisfaction += time_factor * 0.1
        
        # Factor in compromise level (lower compromise = higher satisfaction)
        compromise_level = resolution_outcome.get("compromise_level", 0.5)
        satisfaction += (1 - compromise_level) * 0.1
        
        return min(1.0, max(0.0, satisfaction))
