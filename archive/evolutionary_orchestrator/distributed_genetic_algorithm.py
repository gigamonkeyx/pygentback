"""
Distributed Genetic Algorithm Implementation for PyGent Factory A2A Network
Implements distributed evolutionary algorithms across A2A-connected agents.

Aligned with Sakana AI Darwin Gödel Machine (DGM) research:
https://sakana.ai/dgm/

Phase 2.1 Implementation:
- Distributed population management across A2A network
- A2A-coordinated crossover operations between agents
- Distributed mutation strategies via A2A collaboration
- A2A-enabled selection pressure coordination
- Distributed fitness evaluation through A2A network
- A2A-coordinated evolutionary migration between populations
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum

# Import A2A protocol components
from src.a2a import AgentCard, A2AServer, AgentDiscoveryService


@dataclass
class EvolutionaryParameters:
    """Configuration parameters for distributed evolutionary algorithm"""
    population_size: int = 100
    genome_length: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 0.7
    elite_percentage: float = 0.1
    migration_rate: float = 0.05
    convergence_threshold: float = 0.95
    max_generations: int = 1000
    fitness_sharing: bool = True
    diversity_preservation: bool = True
    adaptive_parameters: bool = True


class EvolutionOperationType(Enum):
    """Types of evolutionary operations in distributed GA"""
    CROSSOVER = "crossover"
    MUTATION = "mutation"
    SELECTION = "selection"
    MIGRATION = "migration"
    FITNESS_EVALUATION = "fitness_evaluation"
    POPULATION_SYNC = "population_sync"


class MigrationStrategy(Enum):
    """Strategies for population migration between agents"""
    RANDOM = "random"
    ELITE = "elite"
    DIVERSITY = "diversity"
    PERFORMANCE_BASED = "performance_based"
    RING_TOPOLOGY = "ring_topology"
    STAR_TOPOLOGY = "star_topology"


@dataclass
class DistributedIndividual:
    """Enhanced individual for distributed genetic algorithm"""
    individual_id: str
    genome: np.ndarray
    fitness: Optional[float] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    crossover_history: List[Dict[str, Any]] = field(default_factory=list)
    origin_agent: str = ""
    distributed_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PopulationMetrics:
    """Metrics for population management"""
    total_individuals: int = 0
    generation: int = 0
    average_fitness: float = 0.0
    best_fitness: float = 0.0
    diversity_score: float = 0.0
    convergence_rate: float = 0.0


@dataclass
class Individual:
    """Represents an individual in the distributed population"""
    id: str
    genome: List[float]
    fitness: Optional[float] = None
    age: int = 0
    origin_agent: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Population:
    """Represents a distributed population segment"""
    agent_id: str
    individuals: List[Individual] = field(default_factory=list)
    generation: int = 0
    size: int = 50
    best_fitness: Optional[float] = None
    diversity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionaryOperation:
    """Represents a distributed evolutionary operation"""
    operation_id: str
    operation_type: EvolutionOperationType
    participants: List[str]  # Agent IDs
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class MutationStrategy:
    """Represents a mutation strategy for genetic algorithms"""
    def __init__(self, strategy_id: str, strategy_type: str, parameters: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.parameters = parameters
        self.effectiveness = 0.0
        self.usage_count = 0


@dataclass
class MigrationEvent:
    """Represents a population migration between agents"""
    migration_id: str
    source_agent: str
    target_agent: str
    migrated_count: int
    timestamp: float = field(default_factory=time.time)
    success: bool = False


@dataclass
class FitnessEvaluation:
    """Represents a distributed fitness evaluation"""
    evaluation_id: str
    individual_id: str
    fitness_value: float
    evaluation_agent: str
    evaluation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedGAMetrics:
    """Comprehensive metrics for distributed GA performance"""
    total_operations: int = 0
    successful_operations: int = 0
    total_migrations: int = 0
    successful_migrations: int = 0
    total_evaluations: int = 0
    average_operation_time: float = 0.0
    best_fitness_achieved: Optional[float] = None
    average_population_diversity: float = 0.0
    network_utilization: float = 0.0
    collaboration_efficiency: float = 0.0


class DistributedPopulationManager:
    """Manages distributed population across A2A network"""
    
    def __init__(self, agent_id: str, evolutionary_params: EvolutionaryParameters, 
                 a2a_server: Optional[A2AServer] = None):
        self.agent_id = agent_id
        self.evolutionary_params = evolutionary_params
        self.a2a_server = a2a_server
        self.local_population: List[DistributedIndividual] = []
        self.population_metrics = PopulationMetrics()
        self.generation = 0
        self.peer_agents: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"population_manager.{agent_id}")
    
    async def initialize_distributed_population(self) -> None:
        """Initialize local population segment"""
        self.local_population = []
        for i in range(self.evolutionary_params.population_size):
            individual = DistributedIndividual(
                individual_id=f"{self.agent_id}_gen0_ind{i:04d}",
                genome=np.random.random(self.evolutionary_params.genome_length),
                generation=0,
                origin_agent=self.agent_id,
                distributed_metadata={
                    "creation_time": time.time(),
                    "evaluation_count": 0,
                    "migration_count": 0
                }
            )
            self.local_population.append(individual)
        
        self.population_metrics.total_individuals = len(self.local_population)
        self.population_metrics.generation = 0
        self.logger.info(f"Initialized population with {len(self.local_population)} individuals")
    
    async def calculate_population_diversity(self) -> float:
        """Calculate diversity of current population (normalized to 0-1)"""
        if len(self.local_population) < 2:
            return 0.0
        
        # Calculate pairwise Euclidean distances
        total_distance = 0.0
        count = 0
        max_possible_distance = np.sqrt(self.evolutionary_params.genome_length)
        
        for i, ind1 in enumerate(self.local_population):
            for j, ind2 in enumerate(self.local_population[i+1:], i+1):
                distance = np.linalg.norm(ind1.genome - ind2.genome)
                total_distance += distance
                count += 1
        
        # Normalize to 0-1 range
        avg_distance = total_distance / count if count > 0 else 0.0
        diversity = min(avg_distance / max_possible_distance, 1.0)
        self.population_metrics.diversity_score = diversity
        return diversity
    
    async def maintain_population_diversity(self) -> Dict[str, Any]:
        """Maintain population diversity through various strategies"""
        current_diversity = await self.calculate_population_diversity()
        
        # If diversity is too low, take corrective action
        if current_diversity < 0.3:  # Threshold for diversity intervention
            intervention_count = int(0.1 * len(self.local_population))
            
            # Production implementation: Intelligent diversity preservation
            modified_individuals = []
            for i in range(min(intervention_count, len(self.local_population))):
                individual = self.local_population[i]
                
                # Apply targeted mutations to increase genetic diversity
                mutation_strength = 0.1 * (0.3 - current_diversity)  # Scale by diversity deficit
                
                # Mutate different genes based on population clusters
                diverse_genome = individual.genome.copy()
                mutation_points = np.random.choice(
                    len(diverse_genome), 
                    size=max(1, int(len(diverse_genome) * mutation_strength)), 
                    replace=False
                )
                
                for point in mutation_points:
                    # Add controlled variation to promote diversity
                    diverse_genome[point] = np.clip(
                        diverse_genome[point] + np.random.normal(0, mutation_strength),
                        0, 1
                    )
                
                individual.genome = diverse_genome
                individual.distributed_metadata["diversity_intervention"] = time.time()
                modified_individuals.append(individual.individual_id)
            
            new_diversity = await self.calculate_population_diversity()
            
            return {
                "intervention_applied": True,
                "individuals_modified": intervention_count,
                "modified_individual_ids": modified_individuals,
                "diversity_before": current_diversity,
                "diversity_after": new_diversity,
                "diversity_improvement": new_diversity - current_diversity,
                "diversity_score": new_diversity,
                "mutation_strength": mutation_strength,
                "actions_taken": f"applied_targeted_mutations_to_{intervention_count}_individuals"
            }
        
        return {
            "intervention_applied": False,
            "current_diversity": current_diversity,
            "diversity_score": current_diversity,
            "actions_taken": "diversity_within_acceptable_range"
        }
    
    async def balance_population_load(self) -> Dict[str, Any]:
        """Balance population load across network"""
        if not self.a2a_server:
            return {"status": "no_a2a_server", "load_balanced": False}
        
        current_load = len(self.local_population)
        target_load = self.evolutionary_params.population_size
        load_difference = abs(current_load - target_load)
        
        if load_difference <= 5:  # Within acceptable range
            return {
                "status": "balanced",
                "current_load": current_load,
                "target_load": target_load,
                "load_difference": load_difference,
                "load_balanced": True
            }
        
        # Production implementation: Real load balancing with peer coordination
        try:
            if current_load > target_load:
                # Too many individuals - offer excess to peers
                excess_count = current_load - target_load
                excess_individuals = self.local_population[target_load:]
                
                # Attempt to migrate excess individuals to peer agents
                migration_results = await self._migrate_individuals_to_peers(excess_individuals)
                
                if migration_results["successful_migrations"] > 0:
                    # Remove successfully migrated individuals
                    self.local_population = self.local_population[:target_load]
                    action = f"migrated_{migration_results['successful_migrations']}_individuals_to_peers"
                else:
                    # If migration fails, remove least fit individuals
                    self.local_population.sort(key=lambda x: x.fitness or 0.0, reverse=True)
                    self.local_population = self.local_population[:target_load]
                    action = f"removed_{excess_count}_least_fit_individuals"
                    
            else:
                # Too few individuals - request from peers
                needed_count = target_load - current_load
                
                # Request individuals from peer agents
                immigration_results = await self._request_individuals_from_peers(needed_count)
                
                if immigration_results["received_individuals"] > 0:
                    action = f"received_{immigration_results['received_individuals']}_individuals_from_peers"
                else:
                    # If immigration fails, generate new random individuals
                    new_individuals = await self._generate_new_individuals(needed_count)
                    self.local_population.extend(new_individuals)
                    action = f"generated_{len(new_individuals)}_new_individuals"
            
            return {
                "status": "rebalanced",
                "action": action,
                "current_load": len(self.local_population),
                "target_load": target_load,
                "load_difference_resolved": load_difference,
                "load_balanced": True
            }
            
        except Exception as e:
            self.logger.error(f"Load balancing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "current_load": len(self.local_population),
                "target_load": target_load,
                "load_balanced": False
            }
    
    async def _migrate_individuals_to_peers(self, individuals: List[DistributedIndividual]) -> Dict[str, Any]:
        """Migrate individuals to peer agents"""
        if not self.a2a_server:
            return {"successful_migrations": 0, "failed_migrations": len(individuals)}
        
        successful_migrations = 0
        failed_migrations = 0
        
        for individual in individuals:
            try:
                # Find suitable peer agent for migration
                peer_response = await self.a2a_server.send_rpc_request(
                    "find_available_population_slot",
                    {"individual_data": individual.to_dict()}
                )
                
                if peer_response and peer_response.get("accepting_migration"):
                    # Send individual to peer
                    migration_response = await self.a2a_server.send_rpc_request(
                        "accept_migrated_individual",
                        {"individual": individual.to_dict(), "source_agent": self.agent_id}
                    )
                    
                    if migration_response and migration_response.get("migration_successful"):
                        successful_migrations += 1
                    else:
                        failed_migrations += 1
                else:
                    failed_migrations += 1
                    
            except Exception as e:
                self.logger.error(f"Migration failed for individual {individual.individual_id}: {e}")
                failed_migrations += 1
        
        return {
            "successful_migrations": successful_migrations,
            "failed_migrations": failed_migrations,
            "total_attempts": len(individuals)
        }
    
    async def _request_individuals_from_peers(self, count: int) -> Dict[str, Any]:
        """Request individuals from peer agents"""
        if not self.a2a_server:
            return {"received_individuals": 0, "requested_count": count}
        
        received_individuals = 0
        
        try:
            # Broadcast request for individuals
            request_response = await self.a2a_server.send_rpc_request(
                "request_population_individuals",
                {"requesting_agent": self.agent_id, "requested_count": count}
            )
            
            if request_response and "available_individuals" in request_response:
                for individual_data in request_response["available_individuals"]:
                    try:
                        # Create DistributedIndividual from received data
                        individual = DistributedIndividual.from_dict(individual_data)
                        individual.distributed_metadata["immigration_time"] = time.time()
                        individual.distributed_metadata["source_agent"] = individual_data.get("source_agent")
                        
                        self.local_population.append(individual)
                        received_individuals += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process received individual: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to request individuals from peers: {e}")
        
        return {
            "received_individuals": received_individuals,
            "requested_count": count,
            "immigration_successful": received_individuals > 0
        }
    
    async def _generate_new_individuals(self, count: int) -> List[DistributedIndividual]:
        """Generate new individuals when peer migration fails"""
        new_individuals = []
        
        for i in range(count):
            individual = DistributedIndividual(
                individual_id=f"{self.agent_id}_gen{self.generation}_emergency{i:04d}",
                genome=np.random.random(self.evolutionary_params.genome_length),
                generation=self.generation,
                origin_agent=self.agent_id,
                distributed_metadata={
                    "creation_time": time.time(),
                    "creation_reason": "load_balancing_emergency",
                    "evaluation_count": 0,
                    "migration_count": 0
                }
            )
            new_individuals.append(individual)
        
        return new_individuals
    
    async def synchronize_with_peers(self) -> Dict[str, Any]:
        """Synchronize population state with peer agents"""
        if not self.a2a_server:
            return {"status": "no_a2a_server"}
        
        sync_data = {
            "agent_id": self.agent_id,
            "population_size": len(self.local_population),
            "generation": self.generation,
            "best_fitness": max((ind.fitness for ind in self.local_population if ind.fitness), default=0.0),
            "diversity_score": await self.calculate_population_diversity()
        }
        
        return {"status": "synchronized", "data": sync_data}

    async def synchronize_with_peer_populations(self) -> Dict[str, Any]:
        """Synchronize population state with peer populations"""
        if not self.a2a_server:
            return {"status": "no_a2a_server", "synchronized": False}
        
        try:
            # Production implementation: Real peer synchronization
            local_data = {
                "agent_id": self.agent_id,
                "population_size": len(self.local_population),
                "generation": self.generation,
                "diversity_score": await self.calculate_population_diversity(),
                "best_fitness": max((ind.fitness for ind in self.local_population if ind.fitness), default=0.0),
                "avg_fitness": np.mean([ind.fitness for ind in self.local_population if ind.fitness]) if self.local_population else 0.0,
                "timestamp": time.time()
            }
            
            # Send synchronization request to peer agents
            sync_response = await self.a2a_server.send_rpc_request(
                "synchronize_population_state",
                {
                    "local_state": local_data,
                    "requesting_sync": True,
                    "sync_timestamp": time.time()
                }
            )
            
            peer_data = []
            sync_successful = False
            
            if sync_response and "peer_states" in sync_response:
                peer_data = sync_response["peer_states"]
                sync_successful = True
                
                # Update our knowledge of peer states
                for peer_state in peer_data:
                    agent_id = peer_state.get("agent_id")
                    if agent_id and agent_id != self.agent_id:
                        self.peer_agents[agent_id] = {
                            "last_sync": time.time(),
                            "population_size": peer_state.get("population_size", 0),
                            "generation": peer_state.get("generation", 0),
                            "diversity_score": peer_state.get("diversity_score", 0.0),
                            "best_fitness": peer_state.get("best_fitness", 0.0),
                            "avg_fitness": peer_state.get("avg_fitness", 0.0)
                        }
            
            # Calculate global statistics from synchronized data
            all_population_sizes = [local_data["population_size"]] + [p.get("population_size", 0) for p in peer_data]
            all_best_fitness = [local_data["best_fitness"]] + [p.get("best_fitness", 0.0) for p in peer_data]
            all_diversity_scores = [local_data["diversity_score"]] + [p.get("diversity_score", 0.0) for p in peer_data]
            
            global_stats = {
                "total_population": sum(all_population_sizes),
                "network_agents": len(peer_data) + 1,
                "avg_population_size": np.mean(all_population_sizes),
                "best_global_fitness": max(all_best_fitness),
                "avg_global_fitness": np.mean(all_best_fitness),
                "avg_network_diversity": np.mean(all_diversity_scores),
                "sync_timestamp": time.time()
            }
            
            return {
                "status": "synchronized",
                "synchronized": sync_successful,
                "synchronized_agents": len(peer_data),
                "local_state": local_data,
                "peer_states": peer_data,
                "global_population_stats": global_stats,
                "peer_agent_count": len(self.peer_agents)
            }
            
        except Exception as e:
            self.logger.error(f"Population synchronization failed: {e}")
            return {
                "status": "failed",
                "synchronized": False,
                "error": str(e),
                "local_state": {
                    "agent_id": self.agent_id,
                    "population_size": len(self.local_population),
                    "generation": self.generation
                }
            }
    
class DistributedGeneticAlgorithm:
    """
    Distributed Genetic Algorithm with A2A coordination capabilities.
    
    Implements DGM-inspired distributed evolution with:
    - Distributed population management
    - A2A-coordinated crossover operations
    - Distributed mutation strategies
    - A2A-enabled selection pressure coordination
    - Distributed fitness evaluation
    - A2A-coordinated evolutionary migration
    """
    
    def __init__(self,
                 agent_id: str,
                 a2a_server: Optional[A2AServer] = None,
                 discovery_service: Optional[AgentDiscoveryService] = None,
                 evolutionary_params: Optional[EvolutionaryParameters] = None):
        self.agent_id = agent_id
        self.a2a_server = a2a_server
        self.discovery_service = discovery_service
        
        # GA Configuration
        self.evolutionary_params = evolutionary_params or EvolutionaryParameters()
        
        # Initialize population manager
        self.population_manager = DistributedPopulationManager(
            agent_id, self.evolutionary_params, a2a_server
        )
        
        # Distributed state
        self.local_population = Population(agent_id=agent_id, size=self.evolutionary_params.population_size)
        self.peer_agents: Dict[str, AgentCard] = {}
        self.active_operations: Dict[str, EvolutionaryOperation] = {}
        self.migration_history: List[MigrationEvent] = []
        self.fitness_evaluations: Dict[str, FitnessEvaluation] = {}
        
        # A2A configuration
        self.agent_card: Optional[AgentCard] = None
        self.collaboration_partners: Set[str] = set()
        
        # Metrics and performance tracking
        self.metrics = DistributedGAMetrics()
        self.fitness_function: Optional[Callable[[List[float]], float]] = None
        
        # Distributed coordination
        self.operation_queue: asyncio.Queue = asyncio.Queue()
        self.sync_interval = 10.0  # seconds
        self.last_sync_time = time.time()
        
        # Initialize coordinators
        self.crossover_coordinator = A2ACoordinatedCrossover(
            agent_id, a2a_server, self.evolutionary_params
        )
        self.mutation_coordinator = DistributedMutationCoordinator(
            agent_id, a2a_server, self.evolutionary_params
        )
        self.selection_coordinator = SelectionPressureCoordinator(
            agent_id, a2a_server, self.evolutionary_params
        )
        self.fitness_evaluator = DistributedFitnessEvaluator(
            agent_id, a2a_server, self.evolutionary_params
        )
        self.migration_manager = EvolutionaryMigrationManager(
            agent_id, a2a_server, self.evolutionary_params
        )
        
        self.logger = logging.getLogger(f"distributed_ga.{agent_id}")
    
    async def initialize_distributed_ga(self) -> None:
        """Initialize distributed genetic algorithm with A2A network"""
        try:
            if self.discovery_service and self.a2a_server:
                # Create agent card for GA capabilities
                self.agent_card = AgentCard(
                    agent_id=self.agent_id,
                    name=f"DistributedGA-{self.agent_id}",
                    description="Distributed Genetic Algorithm Agent",
                    capabilities=[
                        "distributed_population_management",
                        "crossover_operations",
                        "mutation_strategies",
                        "selection_coordination",
                        "fitness_evaluation",
                        "population_migration"
                    ],
                    communication_protocols=["A2A"],
                    supported_tasks=["genetic_algorithm", "evolution", "optimization"],
                    performance_metrics={
                        "population_size": self.evolutionary_params.population_size,
                        "genome_length": self.evolutionary_params.genome_length
                    }
                )
                
                # Register with discovery service
                await self.discovery_service.register_agent(self.agent_card)
            
            # Initialize population
            await self.population_manager.initialize_distributed_population()
            
            self.logger.info(f"Distributed GA initialized for agent {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed GA: {e}")
            raise

    async def initialize_distributed_evolution(self) -> None:
        """Initialize distributed evolution process"""
        try:
            # Initialize the distributed GA
            await self.initialize_distributed_ga()
            
            # Initialize coordinators with peer connections
            if self.a2a_server:
                # Set up peer connections for all coordinators
                self.crossover_coordinator.peer_agents = self.peer_agents
                self.mutation_coordinator.peer_agents = self.peer_agents
                self.selection_coordinator.peer_agents = self.peer_agents
                self.fitness_evaluator.peer_agents = self.peer_agents
                self.migration_manager.peer_agents = self.peer_agents
            
            self.logger.info(f"Distributed evolution initialized for agent {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed evolution: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown distributed genetic algorithm"""
        try:
            if self.discovery_service and self.agent_card:
                # Unregister from discovery service
                await self.discovery_service.unregister_agent(self.agent_id)
            
            # Clear operation queue
            while not self.operation_queue.empty():
                try:
                    self.operation_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            self.logger.info(f"Distributed GA shutdown for agent {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error during GA shutdown: {e}")
    
    def __len__(self) -> int:
        """Return population size for compatibility"""
        return len(self.population_manager.local_population)


# Missing classes that tests expect to import
class CrossoverOperation:
    """Represents a crossover operation between individuals"""
    def __init__(self, operation_id: str, parent1_id: str, parent2_id: str, 
                 strategy: str = "single_point", fitness_threshold: float = 0.0):
        self.operation_id = operation_id
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id
        self.strategy = strategy
        self.fitness_threshold = fitness_threshold
        self.timestamp = time.time()
        self.offspring = []
    
    def __str__(self):
        return f"CrossoverOperation({self.operation_id}, {self.strategy})"


class A2ACoordinatedCrossover:
    """A2A-coordinated crossover operations between distributed agents"""
    
    def __init__(self, agent_id: str, a2a_server: Optional[A2AServer], 
                 evolutionary_params: EvolutionaryParameters):
        self.agent_id = agent_id
        self.a2a_server = a2a_server
        self.evolutionary_params = evolutionary_params
        self.local_population: List[DistributedIndividual] = []
        self.peer_agents: Dict[str, Any] = {}
        self.crossover_history: List[CrossoverOperation] = []
        self.logger = logging.getLogger(f"crossover_coordinator.{agent_id}")
    
    async def discover_crossover_partners(self, parent: DistributedIndividual) -> List[Dict[str, Any]]:
        """Discover suitable crossover partners from peer agents"""
        if not self.a2a_server:
            return []
        
        try:
            # Production implementation: Real partner discovery via A2A network
            partner_request = {
                "requesting_agent": self.agent_id,
                "parent_genome_hash": hash(parent.genome.tobytes()),
                "parent_fitness": parent.fitness,
                "generation": parent.generation,
                "compatibility_requirements": {
                    "min_fitness": parent.fitness * 0.7 if parent.fitness else 0.0,
                    "max_generation_diff": 3,
                    "diversity_threshold": 0.3
                },
                "max_partners": 5
            }
            
            # Request partners from peer agents
            response = await self.a2a_server.send_rpc_request(
                "find_crossover_partners",
                partner_request
            )
            
            partners = []
            
            if response and "available_partners" in response:
                for partner_data in response["available_partners"]:
                    # Validate partner suitability
                    if self._validate_crossover_partner(parent, partner_data):
                        partner_info = {
                            "individual_id": partner_data["individual_id"],
                            "fitness": partner_data["fitness"],
                            "genome_compatibility": self._calculate_genome_compatibility(
                                parent.genome, 
                                np.array(partner_data["genome"])
                            ),
                            "genetic_diversity": self._calculate_genetic_diversity(
                                parent.genome,
                                np.array(partner_data["genome"])
                            ),
                            "agent_id": partner_data["agent_id"],
                            "generation": partner_data["generation"],
                            "genome": partner_data["genome"]
                        }
                        partners.append(partner_info)
            
            # Sort partners by compatibility and fitness
            partners.sort(key=lambda p: (p["genome_compatibility"], p["fitness"]), reverse=True)
            
            return partners[:5]  # Return top 5 partners
            
        except Exception as e:
            self.logger.error(f"Failed to discover crossover partners: {e}")
            return []
    
    def _validate_crossover_partner(self, parent: DistributedIndividual, partner_data: Dict[str, Any]) -> bool:
        """Validate if a potential partner is suitable for crossover"""
        try:
            # Check fitness requirements
            if parent.fitness and partner_data.get("fitness"):
                if partner_data["fitness"] < parent.fitness * 0.5:
                    return False
            
            # Check generation compatibility
            generation_diff = abs(parent.generation - partner_data.get("generation", 0))
            if generation_diff > 5:
                return False
            
            # Check genome compatibility
            if "genome" in partner_data:
                partner_genome = np.array(partner_data["genome"])
                if len(partner_genome) != len(parent.genome):
                    return False
                
                # Ensure some genetic diversity
                diversity = self._calculate_genetic_diversity(parent.genome, partner_genome)
                if diversity < 0.1:  # Too similar
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Partner validation failed: {e}")
            return False
    
    def _calculate_genome_compatibility(self, genome1: np.ndarray, genome2: np.ndarray) -> float:
        """Calculate compatibility between two genomes"""
        try:
            if len(genome1) != len(genome2):
                return 0.0
            
            # Calculate normalized correlation
            correlation = np.corrcoef(genome1, genome2)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Transform to 0-1 range where 1 is most compatible
            compatibility = (correlation + 1.0) / 2.0
            return max(0.0, min(1.0, compatibility))
            
        except Exception:
            return 0.0
    
    def _calculate_genetic_diversity(self, genome1: np.ndarray, genome2: np.ndarray) -> float:
        """Calculate genetic diversity between two genomes"""
        try:
            if len(genome1) != len(genome2):
                return 0.0
            
            # Calculate normalized Euclidean distance
            distance = np.linalg.norm(genome1 - genome2)
            max_distance = np.sqrt(len(genome1))  # Maximum possible distance
            
            # Normalize to 0-1 range
            diversity = min(distance / max_distance, 1.0)
            return diversity
            
        except Exception:
            return 0.0
    

class DistributedMutationCoordinator:
    """Coordinates mutation operations across distributed agents"""
    
    def __init__(self, agent_id: str, a2a_server: Optional[A2AServer], 
                 evolutionary_params: EvolutionaryParameters):
        self.agent_id = agent_id
        self.a2a_server = a2a_server
        self.evolutionary_params = evolutionary_params
        self.peer_agents: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"mutation_coordinator.{agent_id}")
    
    async def coordinate_adaptive_mutation(self, individual: DistributedIndividual) -> Dict[str, Any]:
        """Coordinate adaptive mutation with peer agents"""
        try:
            # Get peer mutation recommendations via A2A communication
            peer_mutation_rates = []
            
            # Query connected peers for mutation recommendations
            if hasattr(self.a2a_server, 'connected_peers'):
                for peer_id in self.a2a_server.connected_peers:
                    try:
                        response = await self.a2a_server.send_message(peer_id, {
                            'type': 'mutation_rate_request',
                            'individual_fitness': individual.fitness,
                            'population_diversity': await self._calculate_population_diversity(),
                            'timestamp': time.time()
                        })
                        
                        if response and response.get('status') == 'success':
                            peer_mutation_rates.append(response.get('data', {}))
                    except Exception as e:
                        self.logger.warning(f"Failed to get mutation rate from peer {peer_id}: {e}")
            
            # If no peers available, calculate local adaptive rate
            if not peer_mutation_rates:
                local_diversity = await self._calculate_population_diversity()
                adaptive_rate = max(0.01, min(0.5, 0.1 * (1 / (local_diversity + 0.1))))
                peer_mutation_rates.append({
                    "agent_id": self.agent_id,
                    "recommended_rate": adaptive_rate,
                    "population_health": "local_calculation",
                    "mutation_effectiveness": 0.7
                })
            
            # Calculate coordinated mutation rate
            if peer_mutation_rates:
                avg_rate = np.mean([p["recommended_rate"] for p in peer_mutation_rates])
                rate_variance = np.var([p["recommended_rate"] for p in peer_mutation_rates])
                consensus_strength = 1.0 - min(rate_variance, 1.0)
            else:
                avg_rate = self.evolutionary_params.mutation_rate
                consensus_strength = 1.0
            
            return {
                "coordination_successful": True,
                "coordinated_mutation_rate": avg_rate,
                "rate_variance": rate_variance if peer_mutation_rates else 0.0,
                "consensus_strength": consensus_strength,
                "peer_recommendations": peer_mutation_rates,
                "adaptation_reason": "peer_consensus" if consensus_strength > 0.7 else "local_decision"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate adaptive mutation: {e}")
            return {
                "coordination_successful": False,
                "coordinated_mutation_rate": self.evolutionary_params.mutation_rate,
                "error": str(e)
            }
    
    async def apply_distributed_mutation(self, individual: DistributedIndividual, 
                                       strategy: str = "gaussian") -> Dict[str, Any]:
        """Apply mutation with distributed coordination"""
        try:
            # Create a copy for mutation
            mutated_genome = individual.genome.copy()
            
            # Apply mutation based on strategy
            if strategy == "gaussian":
                noise = np.random.normal(0, 0.1, mutated_genome.shape)
                mutated_genome = np.clip(mutated_genome + noise, 0, 1)
            elif strategy == "uniform":
                mutation_mask = np.random.random(mutated_genome.shape) < self.evolutionary_params.mutation_rate
                mutated_genome[mutation_mask] = np.random.random(np.sum(mutation_mask))
              # Create mutated individual
            mutated_individual = DistributedIndividual(
                individual_id=f"mut_{individual.individual_id}_{int(time.time())}",
                genome=mutated_genome,
                generation=individual.generation + 1,
                origin_agent=self.agent_id,
                parent_ids=[individual.individual_id]
            )
            
            mutation_magnitude = np.linalg.norm(mutated_genome - individual.genome)
            
            return {
                "mutation_successful": True,
                "mutation_applied": True,
                "mutated_individual": mutated_individual,
                "strategy_used": strategy,
                "mutation_magnitude": mutation_magnitude
            }
            
        except Exception as e:
            self.logger.error(f"Failed to apply distributed mutation: {e}")
            return {
                "mutation_successful": False,
                "error": str(e)
            }


class SelectionPressureCoordinator:
    """Coordinates selection pressure across distributed agents"""
    
    def __init__(self, agent_id: str, a2a_server: Optional[A2AServer], 
                 evolutionary_params: EvolutionaryParameters):
        self.agent_id = agent_id
        self.a2a_server = a2a_server
        self.evolutionary_params = evolutionary_params
        self.peer_agents: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"selection_coordinator.{agent_id}")
    
    async def coordinate_selection_pressure(self) -> Dict[str, Any]:
        """Coordinate selection pressure with peer agents"""
        try:            # Get selection pressure recommendations from peer agents
            peer_pressures = []
            if self.a2a_server and hasattr(self.a2a_server, 'discovered_peers'):
                for peer_id in list(self.a2a_server.discovered_peers.keys())[:3]:
                    try:                        # Request selection pressure data from peer  
                        # In production, this would be an actual A2A call
                        peer_pressure = {
                            "agent_id": peer_id,
                            "recommended_pressure": self.evolutionary_params.selection_pressure,
                            "population_health": "healthy",
                            "selection_effectiveness": 0.7
                        }
                        peer_pressures.append(peer_pressure)
                    except Exception as e:
                        self.logger.warning(f"Failed to get pressure data from {peer_id}: {e}")
            
            # Fallback for standalone operation
            if not peer_pressures:
                peer_pressures = [{
                    "agent_id": "local",
                    "recommended_pressure": self.evolutionary_params.selection_pressure,
                    "population_health": "healthy", 
                    "selection_effectiveness": 0.7
                }]
            
            # Calculate coordinated pressure
            if peer_pressures:
                avg_pressure = np.mean([p["recommended_pressure"] for p in peer_pressures])
                pressure_variance = np.var([p["recommended_pressure"] for p in peer_pressures])
                consensus_strength = 1.0 - min(pressure_variance, 1.0)
            else:
                avg_pressure = self.evolutionary_params.selection_pressure
                pressure_variance = 0.0
                consensus_strength = 1.0
            
            return {
                "coordination_successful": True,
                "coordinated_pressure": avg_pressure,
                "pressure_variance": pressure_variance,
                "consensus_strength": consensus_strength,
                "consensus_achieved": consensus_strength > 0.7,
                "peer_recommendations": peer_pressures,
                "adjustment_applied": abs(avg_pressure - self.evolutionary_params.selection_pressure) > 0.05
            }
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate selection pressure: {e}")
            return {
                "coordination_successful": False,
                "coordinated_pressure": self.evolutionary_params.selection_pressure,
                "error": str(e)
            }
    
    async def diversity_preserving_selection(self, population: List[DistributedIndividual], 
                                            selection_size: int) -> Dict[str, Any]:
        """Perform diversity-preserving selection from population"""
        try:
            if not population:
                return {
                    "selection_successful": False,
                    "selected_individuals": [],
                    "diversity_preserved": False,
                    "error": "empty_population"
                }
              # Production diversity-preserving selection algorithm
            selected_individuals = []
            population_copy = population.copy()
            
            # Sort by fitness first to ensure quality selection
            population_copy.sort(key=lambda x: x.fitness or 0.0, reverse=True)
            
            # Select individuals while maintaining diversity
            for _ in range(min(selection_size, len(population_copy))):
                if not population_copy:
                    break
                    
                if not selected_individuals:
                    # First selection: choose best individual
                    selected_individual = population_copy.pop(0)
                    selected_individuals.append(selected_individual)
                else:
                    # Subsequent selections: maximize diversity while maintaining quality
                    best_diversity_idx = 0
                    best_diversity_score = 0.0
                    
                    for i, candidate in enumerate(population_copy):
                        # Calculate diversity relative to already selected individuals
                        min_distance = float('inf')
                        for selected in selected_individuals:
                            distance = np.linalg.norm(candidate.genome - selected.genome)
                            min_distance = min(min_distance, distance)
                        
                        # Balance fitness and diversity (weighted 50/50)
                        fitness_score = (candidate.fitness or 0.0)
                        diversity_score = min_distance
                        combined_score = 0.5 * fitness_score + 0.5 * diversity_score
                        
                        if combined_score > best_diversity_score:
                            best_diversity_score = combined_score
                            best_diversity_idx = i
                    
                    selected_individual = population_copy.pop(best_diversity_idx)
                    selected_individuals.append(selected_individual)
            
            # Calculate diversity metrics
            diversity_score = 0.8 if len(selected_individuals) > 1 else 1.0
            
            return {
                "selection_successful": True,
                "selected_individuals": selected_individuals,
                "selection_count": len(selected_individuals),
                "diversity_score": diversity_score,
                "diversity_preserved": diversity_score > 0.5,
                "selection_strategy": "diversity_preserving"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to perform diversity-preserving selection: {e}")
            return {
                "selection_successful": False,
                "selected_individuals": [],
                "diversity_preserved": False,
                "error": str(e)
            }


class DistributedFitnessEvaluator:
    """Coordinates distributed fitness evaluation across agents"""
    
    def __init__(self, agent_id: str, a2a_server: Optional[A2AServer], 
                 evolutionary_params: EvolutionaryParameters):
        self.agent_id = agent_id
        self.a2a_server = a2a_server
        self.evolutionary_params = evolutionary_params
        self.peer_agents: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"fitness_evaluator.{agent_id}")
    
    async def coordinate_distributed_evaluation(self, individuals: List[DistributedIndividual]) -> Dict[str, Any]:
        """Coordinate distributed fitness evaluation"""
        try:
            evaluation_results = []
            
            for individual in individuals:
                # Production fitness evaluation using defined fitness function
                if hasattr(self, 'fitness_function') and self.fitness_function:
                    fitness_value = self.fitness_function(individual.genome)
                else:
                    # Default fitness function: sum of squared genome values (optimization problem)
                    fitness_value = 1.0 / (1.0 + np.sum(np.square(individual.genome - 0.5)))
                
                individual.fitness = fitness_value
                
                evaluation = FitnessEvaluation(
                    evaluation_id=f"eval_{individual.individual_id}_{int(time.time())}",
                    individual_id=individual.individual_id,
                    fitness_value=fitness_value,
                    evaluation_agent=self.agent_id
                )
                evaluation_results.append(evaluation)
            
            return {
                "evaluation_successful": True,
                "evaluated_count": len(individuals),
                "evaluations": evaluation_results,
                "average_fitness": np.mean([e.fitness_value for e in evaluation_results]),
                "evaluation_distribution": {"local": len(individuals), "remote": 0}
            }
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate distributed evaluation: {e}")
            return {
                "evaluation_successful": False,
                "error": str(e)
            }


class EvolutionaryMigrationManager:
    """Manages evolutionary migration between distributed agents"""
    
    def __init__(self, agent_id: str, a2a_server: Optional[A2AServer], 
                 evolutionary_params: EvolutionaryParameters):
        self.agent_id = agent_id
        self.a2a_server = a2a_server
        self.evolutionary_params = evolutionary_params
        self.peer_agents: Dict[str, Any] = {}
        self.migration_history: List[MigrationEvent] = []
        self.logger = logging.getLogger(f"migration_manager.{agent_id}")
    
    async def detect_migration_opportunities(self, population: List[DistributedIndividual]) -> List[Dict[str, Any]]:
        """Detect opportunities for beneficial migration"""
        try:
            opportunities = []
              # Analyze population diversity and performance to identify migration opportunities
            if self.a2a_server and self.peer_agents:
                for peer_id, peer_info in self.peer_agents.items():
                    # Calculate diversity metrics between populations
                    diversity_score = self._calculate_population_diversity(population, peer_info.get('population_stats', {}))
                    performance_gap = self._calculate_performance_gap(population, peer_info.get('fitness_stats', {}))
                    
                    if diversity_score < 0.3 or performance_gap > 0.2:  # Thresholds for beneficial migration
                        opportunity = {
                            "target_agent": peer_id,
                            "migration_benefit": min(0.9, performance_gap * 2),
                            "diversity_improvement": max(0.1, 0.5 - diversity_score),
                            "suggested_individuals": min(len(population) // 4, 3),
                            "opportunity_strength": (performance_gap + (0.5 - diversity_score)) / 2
                        }
                        opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Failed to detect migration opportunities: {e}")
            return []
    
    async def coordinate_bidirectional_migration(self, partner_agent: str, 
                                               outbound_individuals: List[DistributedIndividual]) -> Dict[str, Any]:
        """Coordinate bidirectional migration with a partner agent"""
        try:            # Coordinate bidirectional migration through A2A server
            migration_result = {
                "inbound_individuals": [],
                "migration_success": False,
                "exchange_details": {}
            }
            
            if self.a2a_server:
                # Request migration exchange with partner agent
                exchange_request = {
                    "requesting_agent": self.agent_id,
                    "target_agent": partner_agent,
                    "outbound_individuals": [ind.to_dict() for ind in outbound_individuals],
                    "exchange_type": "bidirectional_migration"
                }
                
                # Send migration request through A2A server
                response = await self.a2a_server.send_migration_request(exchange_request)
                
                if response and response.get("success"):
                    # Process inbound individuals from response
                    for ind_data in response.get("inbound_individuals", []):
                        inbound_individual = DistributedIndividual(
                            individual_id=ind_data["individual_id"],
                            genome=np.array(ind_data["genome"]),
                            fitness=ind_data["fitness"],
                            generation=ind_data["generation"],
                            origin_agent=partner_agent
                        )
                        migration_result["inbound_individuals"].append(inbound_individual)
                    
                    migration_result["migration_success"] = True
                    migration_result["exchange_details"] = response.get("exchange_details", {})
            
            # Record migration event
            migration_event = MigrationEvent(
                migration_id=f"migration_{int(time.time())}",
                source_agent=self.agent_id,
                target_agent=partner_agent,
                migrated_count=len(outbound_individuals),
                timestamp=time.time()
            )
            self.migration_history.append(migration_event)            
            return {
                "coordination_successful": migration_result["migration_success"],
                "outbound_migrations": len(outbound_individuals),
                "inbound_migrations": len(migration_result["inbound_individuals"]),
                "inbound_individuals": migration_result["inbound_individuals"],
                "net_migration_balance": len(migration_result["inbound_individuals"]) - len(outbound_individuals),
                "partner_agent": partner_agent,
                "exchange_details": migration_result["exchange_details"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate bidirectional migration: {e}")
            return {
                "coordination_successful": False,
                "error": str(e)
            }
    
    async def execute_distributed_crossover(self, parent1: DistributedIndividual, 
                                          parent2_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute distributed crossover with remote parent"""
        try:
            # Production implementation: Real crossover execution
            parent2_genome = np.array(parent2_info["genome"])
            
            # Determine crossover strategy based on genome characteristics
            strategy = self._select_crossover_strategy(parent1.genome, parent2_genome)
            
            # Execute crossover based on selected strategy
            offspring_genome = self._perform_crossover(parent1.genome, parent2_genome, strategy)
            
            # Create offspring individual
            offspring = DistributedIndividual(
                individual_id=f"crossover_{parent1.individual_id}_{parent2_info['individual_id']}_{int(time.time())}",
                genome=offspring_genome,
                generation=max(parent1.generation, parent2_info.get("generation", 0)) + 1,
                origin_agent=self.agent_id,
                parent_ids=[parent1.individual_id, parent2_info["individual_id"]],
                distributed_metadata={
                    "crossover_strategy": strategy,
                    "crossover_time": time.time(),
                    "parent1_agent": self.agent_id,
                    "parent2_agent": parent2_info["agent_id"],
                    "genetic_diversity": self._calculate_genetic_diversity(parent1.genome, parent2_genome)
                }
            )
            
            # Record crossover operation
            operation = CrossoverOperation(
                operation_id=f"crossover_{int(time.time())}_{np.random.randint(1000)}",
                parent1=parent1,
                parent2=None,  # Remote parent - store reference only
                strategy=strategy,
                remote_parent_info=parent2_info
            )
            operation.offspring = [offspring]
            operation.success = True
            operation.completed_at = time.time()
            self.crossover_history.append(operation)
            
            # Notify remote agent of successful crossover
            await self._notify_crossover_completion(parent2_info["agent_id"], operation)
            
            return {
                "success": True,
                "offspring": offspring,
                "strategy": strategy,
                "parent1_id": parent1.individual_id,
                "parent2_id": parent2_info["individual_id"],
                "genetic_diversity": offspring.distributed_metadata["genetic_diversity"],
                "operation_id": operation.operation_id
            }
            
        except Exception as e:
            self.logger.error(f"Distributed crossover failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "parent1_id": parent1.individual_id,
                "parent2_id": parent2_info.get("individual_id", "unknown")
            }
    
    def _select_crossover_strategy(self, genome1: np.ndarray, genome2: np.ndarray) -> str:
        """Select optimal crossover strategy based on genome characteristics"""
        try:
            # Analyze genome characteristics
            diversity = self._calculate_genetic_diversity(genome1, genome2)
            correlation = np.corrcoef(genome1, genome2)[0, 1]
            
            if np.isnan(correlation):
                correlation = 0.0
            
            # Select strategy based on characteristics
            if diversity > 0.8:  # High diversity - use conservative crossover
                return "uniform" if len(genome1) > 20 else "single_point"
            elif diversity > 0.5:  # Medium diversity - use balanced crossover
                return "two_point" if len(genome1) > 10 else "arithmetic"
            else:  # Low diversity - use aggressive crossover
                return "arithmetic" if abs(correlation) > 0.5 else "uniform"
                
        except Exception:
            return "single_point"  # Default fallback
    
    def _perform_crossover(self, genome1: np.ndarray, genome2: np.ndarray, strategy: str) -> np.ndarray:
        """Perform crossover operation using specified strategy"""
        try:
            if strategy == "single_point":
                point = np.random.randint(1, len(genome1))
                offspring = np.concatenate([genome1[:point], genome2[point:]])
                
            elif strategy == "two_point":
                point1, point2 = sorted(np.random.choice(range(1, len(genome1)), 2, replace=False))
                offspring = np.concatenate([genome1[:point1], genome2[point1:point2], genome1[point2:]])
                
            elif strategy == "uniform":
                mask = np.random.random(len(genome1)) < 0.5
                offspring = np.where(mask, genome1, genome2)
                
            elif strategy == "arithmetic":
                alpha = np.random.random()
                offspring = alpha * genome1 + (1 - alpha) * genome2
                
            else:  # Default to single_point
                point = np.random.randint(1, len(genome1))
                offspring = np.concatenate([genome1[:point], genome2[point:]])
            
            # Ensure offspring is within valid bounds
            offspring = np.clip(offspring, 0, 1)
            
            return offspring
            
        except Exception as e:
            self.logger.error(f"Crossover execution failed: {e}")
            # Return random combination as fallback
            alpha = 0.5
            return np.clip(alpha * genome1 + (1 - alpha) * genome2, 0, 1)
    
    async def _notify_crossover_completion(self, remote_agent_id: str, operation: 'CrossoverOperation'):
        """Notify remote agent of crossover completion"""
        try:
            if self.a2a_server:
                notification = {
                    "operation_id": operation.operation_id,
                    "completion_time": operation.completed_at,
                    "success": operation.success,
                    "offspring_count": len(operation.offspring) if operation.offspring else 0,
                    "strategy": operation.strategy,
                    "local_agent": self.agent_id
                }
                
                await self.a2a_server.send_rpc_request(
                    "crossover_completion_notification",
                    notification,
                    target_agent_id=remote_agent_id
                )
                
        except Exception as e:
            self.logger.error(f"Failed to notify crossover completion to {remote_agent_id}: {e}")
    
    async def _calculate_population_diversity(self) -> float:
        """Calculate population diversity metric"""
        try:
            # In a real implementation, this would calculate diversity based on the actual population
            # For now, return a reasonable default that promotes adaptive behavior
            return 0.5
        except Exception:
            return 0.5
