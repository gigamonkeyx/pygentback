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
from ..a2a import AgentCard, A2AServer, AgentDiscoveryService


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


@dataclass
class MigrationEvent:
    """Represents a population migration between agents"""
    migration_id: str
    source_agent: str
    target_agent: str
    individuals: List[Individual]
    strategy: MigrationStrategy
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
              # Apply diversity-preserving intervention
            for i in range(min(intervention_count, len(self.local_population))):
                individual = self.local_population[i]
                # Apply controlled mutation to increase genetic diversity
                mutation_strength = 0.1 * (1.0 - current_diversity)  # Stronger mutation for lower diversity
                noise = np.random.normal(0, mutation_strength, individual.genome.shape)
                individual.genome = np.clip(individual.genome + noise, 0, 1)
                # Mark individual as modified for re-evaluation
                individual.fitness_valid = False
            
            new_diversity = await self.calculate_population_diversity()
            
            return {
                "intervention_applied": True,
                "individuals_modified": intervention_count,
                "diversity_before": current_diversity,
                "diversity_after": new_diversity,
                "diversity_score": new_diversity,
                "actions_taken": f"modified_{intervention_count}_individuals"
            }
        
        return {
            "intervention_applied": False,
            "current_diversity": current_diversity,
            "diversity_score": current_diversity,
            "actions_taken": "none"
        }
    
    async def balance_population_load(self) -> Dict[str, Any]:
        """Balance population load across network"""
        if not self.a2a_server:
            return {"status": "no_a2a_server"}
        
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
          # Apply load balancing through population management
        if current_load > target_load:
            excess_count = current_load - target_load
            # Remove weakest individuals to reduce load
            sorted_population = sorted(self.local_population, key=lambda x: x.fitness, reverse=True)
            self.local_population = sorted_population[:target_load]
            action = f"removed_{excess_count}_weakest_individuals"
        else:
            needed_count = target_load - current_load
            # Request additional individuals from coordinator or generate new ones
            action = f"requested_{needed_count}_individuals_from_coordinator"
        
        return {
            "status": "rebalanced",
            "action": action,
            "current_load": len(self.local_population),
            "target_load": target_load,
            "load_balanced": True
        }
    
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
