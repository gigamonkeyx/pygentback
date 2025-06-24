"""
Tree of Thought - Core Data Models

This module defines the core data structures for Tree of Thought reasoning.
Based on comprehensive research including Princeton ToT, IBM TouT, and MCTS optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import uuid


class ThoughtState(Enum):
    """States that a thought can be in during reasoning"""
    PENDING = "pending"
    ACTIVE = "active"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    COMPLETED = "completed"


class TreeTraversalStrategy(Enum):
    """Tree traversal strategies for thought exploration"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    BEST_FIRST = "best_first"
    ADAPTIVE = "adaptive"


class EvaluationMethod(Enum):
    """Methods for evaluating thoughts"""
    VALUE = "value"
    VOTE = "vote"
    HYBRID = "hybrid"


class GenerationMethod(Enum):
    """Methods for generating thoughts"""
    PROPOSE = "propose"
    SAMPLE = "sample"
    GUIDED = "guided"


@dataclass
class ThoughtNode:
    """
    A single thought in the reasoning tree.
    
    Based on research findings from Princeton ToT and IBM TouT frameworks.
    Includes uncertainty quantification and evidence tracking.
    """
    id: str
    content: str
    depth: int
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    state: ThoughtState = ThoughtState.PENDING
    value_score: Optional[float] = None
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    reasoning: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    evaluated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0
    
    @property
    def is_evaluated(self) -> bool:
        """Check if this node has been evaluated"""
        return self.state == ThoughtState.EVALUATED and self.value_score is not None
    
    @property
    def quality_score(self) -> float:
        """
        Combined quality score considering value and confidence.
        Used for MCTS-style selection and pruning decisions.
        """
        if self.value_score is None or self.confidence is None:
            return 0.0
        return self.value_score * self.confidence
    
    @property
    def exploration_value(self) -> float:
        """
        Exploration value for adaptive search strategies.
        Balances exploitation vs exploration based on uncertainty.
        """
        if self.uncertainty is None:
            return self.quality_score
        # Higher uncertainty increases exploration value
        exploration_bonus = self.uncertainty * 0.1
        return self.quality_score + exploration_bonus


@dataclass
class ToTConfig:
    """
    Configuration for Tree of Thought reasoning.
    
    Based on research findings and production best practices.
    Supports adaptive strategies and memory optimization.
    """
    max_depth: int = 6
    max_children: int = 5
    search_strategy: TreeTraversalStrategy = TreeTraversalStrategy.ADAPTIVE
    evaluation_method: EvaluationMethod = EvaluationMethod.VALUE
    generation_method: GenerationMethod = GenerationMethod.PROPOSE
    pruning_threshold: float = 0.3
    parallel_exploration: bool = True
    max_parallel_thoughts: int = 4
    memory_limit: int = 1000  # Maximum nodes to keep in memory
    uncertainty_quantification: bool = True
    enable_backtracking: bool = True
    exploration_bonus: float = 0.1  # Bonus for exploring new paths
    confidence_threshold: float = 0.7  # Minimum confidence for path selection
    adaptive_threshold: float = 0.5  # Threshold for strategy switching
    cache_size: int = 500  # LRU cache size for nodes
    
    def __post_init__(self):
        # Validate configuration based on research constraints
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.max_children < 1:
            raise ValueError("max_children must be at least 1")
        if not 0 <= self.pruning_threshold <= 1:
            raise ValueError("pruning_threshold must be between 0 and 1")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.max_parallel_thoughts < 1:
            raise ValueError("max_parallel_thoughts must be at least 1")


@dataclass
class ThoughtEvaluation:
    """
    Evaluation result for a thought.
    
    Includes uncertainty quantification following TouT framework.
    """
    thought_id: str
    value_score: float
    confidence: float
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    uncertainty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TreeStatistics:
    """
    Statistics about the thought tree exploration.
    
    Used for performance monitoring and optimization.
    """
    total_nodes: int = 0
    max_depth_reached: int = 0
    avg_depth: float = 0.0
    total_evaluations: int = 0
    pruned_nodes: int = 0
    successful_paths: int = 0
    exploration_time: float = 0.0
    memory_usage: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    strategy_switches: int = 0
    backtrack_count: int = 0


@dataclass
class ReasoningPath:
    """
    A complete reasoning path from root to leaf.
    
    Represents a solution candidate with full reasoning chain.
    """
    path_id: str
    nodes: List[str]
    total_score: float
    confidence: float
    reasoning_chain: List[str]
    evidence_chain: List[List[str]] = field(default_factory=list)
    is_complete: bool = False
    is_successful: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.path_id:
            self.path_id = str(uuid.uuid4())


@dataclass
class ToTResult:
    """
    Final result of Tree of Thought reasoning.
    
    Contains the best path found and exploration statistics.
    """
    problem: str
    best_path: Optional[ReasoningPath]
    all_paths: List[ReasoningPath]
    statistics: TreeStatistics
    config: ToTConfig
    success: bool = False
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)