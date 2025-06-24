"""
Crossover Operators for Recipe Evolution

Implements various crossover strategies for combining recipe genomes
to create offspring with potentially improved characteristics.
"""

import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .recipe_genome import RecipeGenome

logger = logging.getLogger(__name__)


class CrossoverType(Enum):
    """Types of crossover operations"""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"


@dataclass
class CrossoverConfig:
    """Configuration for crossover operations"""
    crossover_type: CrossoverType = CrossoverType.UNIFORM
    crossover_rate: float = 0.8
    uniform_probability: float = 0.5
    arithmetic_alpha: float = 0.5
    adaptive_threshold: float = 0.1
    preserve_structure: bool = True
    repair_invalid: bool = True


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators"""
    
    def __init__(self, config: Optional[CrossoverConfig] = None):
        self.config = config or CrossoverConfig()
        self.genome_handler = RecipeGenome()
    
    @abstractmethod
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """
        Perform crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Tuple of two offspring genomes
        """
        pass
    
    def should_crossover(self) -> bool:
        """Determine if crossover should occur based on crossover rate"""
        return random.random() < self.config.crossover_rate
    
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
    
    def validate_parents(self, parent1: List[float], parent2: List[float]) -> bool:
        """Validate that parents are suitable for crossover"""
        if len(parent1) != len(parent2):
            logger.warning("Parent genomes have different lengths")
            return False
        
        if not self.genome_handler.validate_genome(parent1):
            logger.warning("Parent 1 genome is invalid")
            return False
        
        if not self.genome_handler.validate_genome(parent2):
            logger.warning("Parent 2 genome is invalid")
            return False
        
        return True


class SinglePointCrossover(CrossoverOperator):
    """Single-point crossover operator"""
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Perform single-point crossover"""
        if not self.validate_parents(parent1, parent2):
            return parent1.copy(), parent2.copy()
        
        if not self.should_crossover():
            return parent1.copy(), parent2.copy()
        
        # Choose crossover point
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # Create offspring
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # Repair if needed
        offspring1 = self.repair_genome(offspring1)
        offspring2 = self.repair_genome(offspring2)
        
        logger.debug(f"Single-point crossover at position {crossover_point}")
        return offspring1, offspring2


class TwoPointCrossover(CrossoverOperator):
    """Two-point crossover operator"""
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Perform two-point crossover"""
        if not self.validate_parents(parent1, parent2):
            return parent1.copy(), parent2.copy()
        
        if not self.should_crossover():
            return parent1.copy(), parent2.copy()
        
        # Choose two crossover points
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        
        # Create offspring
        offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        # Repair if needed
        offspring1 = self.repair_genome(offspring1)
        offspring2 = self.repair_genome(offspring2)
        
        logger.debug(f"Two-point crossover at positions {point1}, {point2}")
        return offspring1, offspring2


class UniformCrossover(CrossoverOperator):
    """Uniform crossover operator"""
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Perform uniform crossover"""
        if not self.validate_parents(parent1, parent2):
            return parent1.copy(), parent2.copy()
        
        if not self.should_crossover():
            return parent1.copy(), parent2.copy()
        
        offspring1 = []
        offspring2 = []
        
        # For each gene, randomly choose which parent to inherit from
        for i in range(len(parent1)):
            if random.random() < self.config.uniform_probability:
                offspring1.append(parent1[i])
                offspring2.append(parent2[i])
            else:
                offspring1.append(parent2[i])
                offspring2.append(parent1[i])
        
        # Repair if needed
        offspring1 = self.repair_genome(offspring1)
        offspring2 = self.repair_genome(offspring2)
        
        logger.debug("Uniform crossover completed")
        return offspring1, offspring2


class ArithmeticCrossover(CrossoverOperator):
    """Arithmetic crossover operator"""
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Perform arithmetic crossover"""
        if not self.validate_parents(parent1, parent2):
            return parent1.copy(), parent2.copy()
        
        if not self.should_crossover():
            return parent1.copy(), parent2.copy()
        
        alpha = self.config.arithmetic_alpha
        
        # Create offspring using arithmetic combination
        offspring1 = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
        offspring2 = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
        
        # Repair if needed
        offspring1 = self.repair_genome(offspring1)
        offspring2 = self.repair_genome(offspring2)
        
        logger.debug(f"Arithmetic crossover with alpha={alpha}")
        return offspring1, offspring2


class SemanticCrossover(CrossoverOperator):
    """Semantic-aware crossover operator"""
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Perform semantic-aware crossover"""
        if not self.validate_parents(parent1, parent2):
            return parent1.copy(), parent2.copy()
        
        if not self.should_crossover():
            return parent1.copy(), parent2.copy()
        
        # Get genome structure information
        genome_info = self.genome_handler.get_genome_info()
        structure = genome_info['structure']
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Crossover within semantic blocks
        pos = 0
        
        # Metadata block
        if random.random() < 0.5:
            end_pos = pos + structure['metadata_length']
            offspring1[pos:end_pos], offspring2[pos:end_pos] = \
                self._crossover_block(parent1[pos:end_pos], parent2[pos:end_pos])
        pos += structure['metadata_length']
        
        # Agent requirements block
        if random.random() < 0.5:
            end_pos = pos + structure['agent_length']
            offspring1[pos:end_pos], offspring2[pos:end_pos] = \
                self._crossover_block(parent1[pos:end_pos], parent2[pos:end_pos])
        pos += structure['agent_length']
        
        # MCP requirements block
        if random.random() < 0.5:
            end_pos = pos + structure['mcp_length']
            offspring1[pos:end_pos], offspring2[pos:end_pos] = \
                self._crossover_block(parent1[pos:end_pos], parent2[pos:end_pos])
        pos += structure['mcp_length']
        
        # Steps block
        if random.random() < 0.5:
            end_pos = pos + structure['steps_length']
            offspring1[pos:end_pos], offspring2[pos:end_pos] = \
                self._crossover_block(parent1[pos:end_pos], parent2[pos:end_pos])
        pos += structure['steps_length']
        
        # Performance block
        if random.random() < 0.5:
            end_pos = pos + structure['performance_length']
            offspring1[pos:end_pos], offspring2[pos:end_pos] = \
                self._crossover_block(parent1[pos:end_pos], parent2[pos:end_pos])
        
        # Repair if needed
        offspring1 = self.repair_genome(offspring1)
        offspring2 = self.repair_genome(offspring2)
        
        logger.debug("Semantic crossover completed")
        return offspring1, offspring2
    
    def _crossover_block(self, block1: List[float], block2: List[float]) -> Tuple[List[float], List[float]]:
        """Crossover within a semantic block"""
        if len(block1) <= 1:
            return block1.copy(), block2.copy()
        
        # Use uniform crossover within the block
        offspring1 = []
        offspring2 = []
        
        for i in range(len(block1)):
            if random.random() < 0.5:
                offspring1.append(block1[i])
                offspring2.append(block2[i])
            else:
                offspring1.append(block2[i])
                offspring2.append(block1[i])
        
        return offspring1, offspring2


class AdaptiveCrossover(CrossoverOperator):
    """Adaptive crossover operator that chooses strategy based on parent similarity"""
    
    def __init__(self, config: Optional[CrossoverConfig] = None):
        super().__init__(config)
        self.operators = {
            CrossoverType.SINGLE_POINT: SinglePointCrossover(config),
            CrossoverType.TWO_POINT: TwoPointCrossover(config),
            CrossoverType.UNIFORM: UniformCrossover(config),
            CrossoverType.ARITHMETIC: ArithmeticCrossover(config),
            CrossoverType.SEMANTIC: SemanticCrossover(config)
        }
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Perform adaptive crossover"""
        if not self.validate_parents(parent1, parent2):
            return parent1.copy(), parent2.copy()
        
        if not self.should_crossover():
            return parent1.copy(), parent2.copy()
        
        # Calculate parent similarity
        similarity = self.genome_handler.calculate_genome_similarity(parent1, parent2)
        
        # Choose crossover strategy based on similarity
        if similarity > 0.8:
            # High similarity - use more explorative crossover
            strategy = CrossoverType.UNIFORM
        elif similarity > 0.5:
            # Medium similarity - use semantic crossover
            strategy = CrossoverType.SEMANTIC
        elif similarity > 0.2:
            # Low similarity - use arithmetic crossover
            strategy = CrossoverType.ARITHMETIC
        else:
            # Very different - use two-point crossover
            strategy = CrossoverType.TWO_POINT
        
        operator = self.operators[strategy]
        offspring1, offspring2 = operator.crossover(parent1, parent2)
        
        logger.debug(f"Adaptive crossover used {strategy.value} (similarity={similarity:.3f})")
        return offspring1, offspring2


class CrossoverOperatorFactory:
    """Factory for creating crossover operators"""
    
    @staticmethod
    def create_operator(crossover_type: CrossoverType, 
                       config: Optional[CrossoverConfig] = None) -> CrossoverOperator:
        """Create a crossover operator of the specified type"""
        
        if crossover_type == CrossoverType.SINGLE_POINT:
            return SinglePointCrossover(config)
        elif crossover_type == CrossoverType.TWO_POINT:
            return TwoPointCrossover(config)
        elif crossover_type == CrossoverType.UNIFORM:
            return UniformCrossover(config)
        elif crossover_type == CrossoverType.ARITHMETIC:
            return ArithmeticCrossover(config)
        elif crossover_type == CrossoverType.SEMANTIC:
            return SemanticCrossover(config)
        elif crossover_type == CrossoverType.ADAPTIVE:
            return AdaptiveCrossover(config)
        else:
            raise ValueError(f"Unknown crossover type: {crossover_type}")
    
    @staticmethod
    def get_available_types() -> List[CrossoverType]:
        """Get list of available crossover types"""
        return list(CrossoverType)


def test_crossover_operators():
    """Test function for crossover operators"""
    # Create test genomes
    genome_handler = RecipeGenome()
    genome_length = genome_handler.genome_length
    
    parent1 = [random.random() for _ in range(genome_length)]
    parent2 = [random.random() for _ in range(genome_length)]
    
    print(f"Testing crossover operators with genome length: {genome_length}")
    print(f"Parent similarity: {genome_handler.calculate_genome_similarity(parent1, parent2):.3f}")
    
    # Test each crossover type
    for crossover_type in CrossoverType:
        try:
            operator = CrossoverOperatorFactory.create_operator(crossover_type)
            offspring1, offspring2 = operator.crossover(parent1, parent2)
            
            # Validate offspring
            valid1 = genome_handler.validate_genome(offspring1)
            valid2 = genome_handler.validate_genome(offspring2)
            
            print(f"{crossover_type.value}: Offspring valid: {valid1}, {valid2}")
            
        except Exception as e:
            print(f"{crossover_type.value}: Error - {e}")


if __name__ == "__main__":
    test_crossover_operators()
