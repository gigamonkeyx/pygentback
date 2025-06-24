"""
Mutation Operators for Recipe Evolution

Implements various mutation strategies for introducing variation
in recipe genomes during evolutionary optimization.
"""

import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .recipe_genome import RecipeGenome

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of mutation operations"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"
    SEMANTIC = "semantic"
    CREEP = "creep"


@dataclass
class MutationConfig:
    """Configuration for mutation operations"""
    mutation_type: MutationType = MutationType.GAUSSIAN
    mutation_rate: float = 0.1
    gene_mutation_probability: float = 0.05
    gaussian_std: float = 0.1
    uniform_range: float = 0.2
    polynomial_eta: float = 20.0
    adaptive_factor: float = 0.1
    creep_step: float = 0.01
    repair_invalid: bool = True
    preserve_structure: bool = True


class MutationOperator(ABC):
    """Abstract base class for mutation operators"""
    
    def __init__(self, config: Optional[MutationConfig] = None):
        self.config = config or MutationConfig()
        self.genome_handler = RecipeGenome()
        self.mutation_history = []
    
    @abstractmethod
    def mutate(self, genome: List[float]) -> List[float]:
        """
        Perform mutation on a genome.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        pass
    
    def should_mutate_gene(self) -> bool:
        """Determine if a specific gene should be mutated"""
        return random.random() < self.config.gene_mutation_probability
    
    def should_mutate_genome(self) -> bool:
        """Determine if the genome should be mutated at all"""
        return random.random() < self.config.mutation_rate
    
    def repair_genome(self, genome: List[float]) -> List[float]:
        """Repair invalid genome values"""
        if not self.config.repair_invalid:
            return genome
        
        # Clamp values to valid range [0, 1]
        repaired = [max(0.0, min(1.0, val)) for val in genome]
        
        # Ensure proper genome length
        if len(repaired) != self.genome_handler.genome_length:
            repaired = self.genome_handler._normalize_genome_length(repaired)
        
        return repaired
    
    def record_mutation(self, original: List[float], mutated: List[float]):
        """Record mutation statistics"""
        changes = sum(1 for o, m in zip(original, mutated) if abs(o - m) > 1e-6)
        avg_change = np.mean([abs(o - m) for o, m in zip(original, mutated)])
        
        self.mutation_history.append({
            'genes_changed': changes,
            'avg_change_magnitude': avg_change,
            'total_genes': len(original)
        })
        
        # Keep only recent history
        if len(self.mutation_history) > 1000:
            self.mutation_history = self.mutation_history[-1000:]


class GaussianMutation(MutationOperator):
    """Gaussian (normal) mutation operator"""
    
    def mutate(self, genome: List[float]) -> List[float]:
        """Perform Gaussian mutation"""
        if not self.should_mutate_genome():
            return genome.copy()
        
        mutated = genome.copy()
        original = genome.copy()
        
        for i in range(len(mutated)):
            if self.should_mutate_gene():
                # Add Gaussian noise
                noise = np.random.normal(0, self.config.gaussian_std)
                mutated[i] += noise
        
        # Repair invalid values
        mutated = self.repair_genome(mutated)
        
        # Record mutation statistics
        self.record_mutation(original, mutated)
        
        logger.debug(f"Gaussian mutation applied with std={self.config.gaussian_std}")
        return mutated


class UniformMutation(MutationOperator):
    """Uniform mutation operator"""
    
    def mutate(self, genome: List[float]) -> List[float]:
        """Perform uniform mutation"""
        if not self.should_mutate_genome():
            return genome.copy()
        
        mutated = genome.copy()
        original = genome.copy()
        
        for i in range(len(mutated)):
            if self.should_mutate_gene():
                # Add uniform random change
                change = random.uniform(-self.config.uniform_range, self.config.uniform_range)
                mutated[i] += change
        
        # Repair invalid values
        mutated = self.repair_genome(mutated)
        
        # Record mutation statistics
        self.record_mutation(original, mutated)
        
        logger.debug(f"Uniform mutation applied with range=±{self.config.uniform_range}")
        return mutated


class PolynomialMutation(MutationOperator):
    """Polynomial mutation operator"""
    
    def mutate(self, genome: List[float]) -> List[float]:
        """Perform polynomial mutation"""
        if not self.should_mutate_genome():
            return genome.copy()
        
        mutated = genome.copy()
        original = genome.copy()
        eta = self.config.polynomial_eta
        
        for i in range(len(mutated)):
            if self.should_mutate_gene():
                # Polynomial mutation formula
                u = random.random()
                
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                mutated[i] += delta * 0.1  # Scale the change
        
        # Repair invalid values
        mutated = self.repair_genome(mutated)
        
        # Record mutation statistics
        self.record_mutation(original, mutated)
        
        logger.debug(f"Polynomial mutation applied with eta={eta}")
        return mutated


class CreepMutation(MutationOperator):
    """Creep mutation operator (small incremental changes)"""
    
    def mutate(self, genome: List[float]) -> List[float]:
        """Perform creep mutation"""
        if not self.should_mutate_genome():
            return genome.copy()
        
        mutated = genome.copy()
        original = genome.copy()
        
        for i in range(len(mutated)):
            if self.should_mutate_gene():
                # Small random step
                direction = random.choice([-1, 1])
                mutated[i] += direction * self.config.creep_step
        
        # Repair invalid values
        mutated = self.repair_genome(mutated)
        
        # Record mutation statistics
        self.record_mutation(original, mutated)
        
        logger.debug(f"Creep mutation applied with step={self.config.creep_step}")
        return mutated


class SemanticMutation(MutationOperator):
    """Semantic-aware mutation operator"""
    
    def mutate(self, genome: List[float]) -> List[float]:
        """Perform semantic-aware mutation"""
        if not self.should_mutate_genome():
            return genome.copy()
        
        mutated = genome.copy()
        original = genome.copy()
        
        # Get genome structure information
        genome_info = self.genome_handler.get_genome_info()
        structure = genome_info['structure']
        
        # Apply different mutation strategies to different semantic blocks
        pos = 0
        
        # Metadata block - conservative mutation
        end_pos = pos + structure['metadata_length']
        if random.random() < 0.3:  # Lower probability for metadata
            mutated[pos:end_pos] = self._mutate_block(
                mutated[pos:end_pos], mutation_strength=0.05
            )
        pos = end_pos
        
        # Agent requirements block - moderate mutation
        end_pos = pos + structure['agent_length']
        if random.random() < 0.5:
            mutated[pos:end_pos] = self._mutate_block(
                mutated[pos:end_pos], mutation_strength=0.1
            )
        pos = end_pos
        
        # MCP requirements block - moderate mutation
        end_pos = pos + structure['mcp_length']
        if random.random() < 0.5:
            mutated[pos:end_pos] = self._mutate_block(
                mutated[pos:end_pos], mutation_strength=0.1
            )
        pos = end_pos
        
        # Steps block - higher mutation (most variable part)
        end_pos = pos + structure['steps_length']
        if random.random() < 0.7:
            mutated[pos:end_pos] = self._mutate_block(
                mutated[pos:end_pos], mutation_strength=0.15
            )
        pos = end_pos
        
        # Performance block - conservative mutation
        end_pos = pos + structure['performance_length']
        if random.random() < 0.4:
            mutated[pos:end_pos] = self._mutate_block(
                mutated[pos:end_pos], mutation_strength=0.08
            )
        
        # Repair invalid values
        mutated = self.repair_genome(mutated)
        
        # Record mutation statistics
        self.record_mutation(original, mutated)
        
        logger.debug("Semantic mutation applied")
        return mutated
    
    def _mutate_block(self, block: List[float], mutation_strength: float) -> List[float]:
        """Mutate a semantic block with specified strength"""
        mutated_block = block.copy()
        
        for i in range(len(mutated_block)):
            if random.random() < self.config.gene_mutation_probability:
                noise = np.random.normal(0, mutation_strength)
                mutated_block[i] += noise
        
        return mutated_block


class AdaptiveMutation(MutationOperator):
    """Adaptive mutation operator that adjusts based on population diversity"""
    
    def __init__(self, config: Optional[MutationConfig] = None):
        super().__init__(config)
        self.population_diversity = 0.5  # Initial diversity estimate
        self.diversity_history = []
    
    def mutate(self, genome: List[float]) -> List[float]:
        """Perform adaptive mutation"""
        if not self.should_mutate_genome():
            return genome.copy()
        
        # Adjust mutation strength based on population diversity
        if self.population_diversity < 0.3:
            # Low diversity - increase mutation
            mutation_strength = self.config.gaussian_std * 2.0
        elif self.population_diversity > 0.7:
            # High diversity - decrease mutation
            mutation_strength = self.config.gaussian_std * 0.5
        else:
            # Normal diversity - standard mutation
            mutation_strength = self.config.gaussian_std
        
        mutated = genome.copy()
        original = genome.copy()
        
        for i in range(len(mutated)):
            if self.should_mutate_gene():
                noise = np.random.normal(0, mutation_strength)
                mutated[i] += noise
        
        # Repair invalid values
        mutated = self.repair_genome(mutated)
        
        # Record mutation statistics
        self.record_mutation(original, mutated)
        
        logger.debug(f"Adaptive mutation applied with strength={mutation_strength:.3f}")
        return mutated
    
    def update_diversity(self, population_genomes: List[List[float]]):
        """Update population diversity estimate"""
        if len(population_genomes) < 2:
            return
        
        # Calculate average pairwise distance
        total_distance = 0
        comparisons = 0
        
        for i in range(len(population_genomes)):
            for j in range(i + 1, len(population_genomes)):
                similarity = self.genome_handler.calculate_genome_similarity(
                    population_genomes[i], population_genomes[j]
                )
                distance = 1.0 - similarity
                total_distance += distance
                comparisons += 1
        
        if comparisons > 0:
            avg_distance = total_distance / comparisons
            self.population_diversity = avg_distance
            
            # Update history
            self.diversity_history.append(self.population_diversity)
            if len(self.diversity_history) > 100:
                self.diversity_history = self.diversity_history[-100:]


class MutationOperatorFactory:
    """Factory for creating mutation operators"""
    
    @staticmethod
    def create_operator(mutation_type: MutationType, 
                       config: Optional[MutationConfig] = None) -> MutationOperator:
        """Create a mutation operator of the specified type"""
        
        if mutation_type == MutationType.GAUSSIAN:
            return GaussianMutation(config)
        elif mutation_type == MutationType.UNIFORM:
            return UniformMutation(config)
        elif mutation_type == MutationType.POLYNOMIAL:
            return PolynomialMutation(config)
        elif mutation_type == MutationType.CREEP:
            return CreepMutation(config)
        elif mutation_type == MutationType.SEMANTIC:
            return SemanticMutation(config)
        elif mutation_type == MutationType.ADAPTIVE:
            return AdaptiveMutation(config)
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")
    
    @staticmethod
    def get_available_types() -> List[MutationType]:
        """Get list of available mutation types"""
        return list(MutationType)


def test_mutation_operators():
    """Test function for mutation operators"""
    # Create test genome
    genome_handler = RecipeGenome()
    genome_length = genome_handler.genome_length
    
    original_genome = [random.random() for _ in range(genome_length)]
    
    print(f"Testing mutation operators with genome length: {genome_length}")
    print(f"Original genome valid: {genome_handler.validate_genome(original_genome)}")
    
    # Test each mutation type
    for mutation_type in MutationType:
        try:
            operator = MutationOperatorFactory.create_operator(mutation_type)
            mutated_genome = operator.mutate(original_genome)
            
            # Validate mutated genome
            valid = genome_handler.validate_genome(mutated_genome)
            
            # Calculate change
            changes = sum(1 for o, m in zip(original_genome, mutated_genome) if abs(o - m) > 1e-6)
            avg_change = np.mean([abs(o - m) for o, m in zip(original_genome, mutated_genome)])
            
            print(f"{mutation_type.value}: Valid: {valid}, Changes: {changes}, Avg change: {avg_change:.4f}")
            
        except Exception as e:
            print(f"{mutation_type.value}: Error - {e}")


if __name__ == "__main__":
    test_mutation_operators()
