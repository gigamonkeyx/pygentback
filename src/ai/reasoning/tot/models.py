"""
Tree of Thought Data Models

Core data structures for the ToT framework including thought states,
trees, search results, and configuration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import uuid
from datetime import datetime


class GenerationStrategy(Enum):
    """Strategy for generating thoughts"""
    SAMPLE = "sample"      # Sample independent thoughts (Creative Writing)
    PROPOSE = "propose"    # Propose sequential thoughts (Game of 24)


class EvaluationMethod(Enum):
    """Method for evaluating thought states"""
    VALUE = "value"        # Evaluate states independently (Game of 24)
    VOTE = "vote"         # Vote on states together (Creative Writing)


class SearchMethod(Enum):
    """Search algorithm for exploring thought tree"""
    BFS = "bfs"           # Breadth-First Search
    DFS = "dfs"           # Depth-First Search
    ADAPTIVE = "adaptive" # Adaptive Search combining BFS and DFS


@dataclass
class ToTConfig:
    """Configuration for Tree of Thought reasoning"""
    # Generation settings
    generation_strategy: GenerationStrategy = GenerationStrategy.PROPOSE
    n_generate_sample: int = 1
    
    # Evaluation settings
    evaluation_method: EvaluationMethod = EvaluationMethod.VALUE
    n_evaluate_sample: int = 3
    
    # Search settings
    search_method: SearchMethod = SearchMethod.BFS
    n_select_sample: int = 5
    max_depth: int = 10
    max_iterations: int = 100
    value_threshold: float = 0.7
    confidence_threshold: float = 0.7
    
    # LLM settings
    model_name: str = ""  # Force explicit model selection
    temperature: float = 0.7
    evaluation_temperature: float = 0.3  # Lower temperature for more focused evaluation
    max_tokens: int = 1000
    
    # Task settings
    task_description: str = ""
    success_criteria: str = ""


@dataclass
class ThoughtState:
    """Represents a single thought state in the reasoning tree"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    depth: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Evaluation metrics
    value_score: float = 0.0
    vote_count: int = 0
    confidence: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_terminal: bool = False
    is_solution: bool = False
    is_pruned: bool = False
    
    def add_child(self, child_id: str) -> None:
        """Add a child state ID"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'content': self.content,
            'depth': self.depth,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'value_score': self.value_score,
            'vote_count': self.vote_count,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'is_terminal': self.is_terminal,
            'is_solution': self.is_solution,
            'is_pruned': self.is_pruned
        }


@dataclass
class ThoughtTree:
    """Represents the complete tree of thought states"""
    root_id: str
    states: Dict[str, ThoughtState] = field(default_factory=dict)
    config: ToTConfig = field(default_factory=ToTConfig)
    
    # Tree statistics
    total_nodes: int = 0
    max_depth_reached: int = 0
    solutions_found: int = 0
    
    def add_state(self, state: ThoughtState) -> None:
        """Add a state to the tree"""
        self.states[state.id] = state
        self.total_nodes += 1
        self.max_depth_reached = max(self.max_depth_reached, state.depth)
        
        if state.is_solution:
            self.solutions_found += 1
    
    def get_state(self, state_id: str) -> Optional[ThoughtState]:
        """Get a state by ID"""
        return self.states.get(state_id)
    
    def get_children(self, state_id: str) -> List[ThoughtState]:
        """Get all children of a state"""
        state = self.get_state(state_id)
        if not state:
            return []
        return [self.states[child_id] for child_id in state.children_ids 
                if child_id in self.states]
    
    def get_path_to_root(self, state_id: str) -> List[ThoughtState]:
        """Get the path from a state back to the root"""
        path = []
        current_id = state_id
        
        while current_id:
            state = self.get_state(current_id)
            if not state:
                break
            path.append(state)
            current_id = state.parent_id
        
        return list(reversed(path))
    
    def get_best_solution(self) -> Optional[ThoughtState]:
        """Get the best solution found so far"""
        solutions = [state for state in self.states.values() if state.is_solution]
        if not solutions:
            return None
        return max(solutions, key=lambda s: s.value_score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation"""
        return {
            'root_id': self.root_id,
            'states': {sid: state.to_dict() for sid, state in self.states.items()},
            'config': self.config.__dict__,
            'total_nodes': self.total_nodes,
            'max_depth_reached': self.max_depth_reached,
            'solutions_found': self.solutions_found
        }


@dataclass
class SearchResult:
    """Result of a Tree of Thought search"""
    tree: ThoughtTree
    best_solution: Optional[ThoughtState] = None
    all_solutions: List[ThoughtState] = field(default_factory=list)
    
    # Performance metrics
    total_time: float = 0.0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    
    # Search statistics
    nodes_explored: int = 0
    nodes_pruned: int = 0
    max_depth_reached: int = 0
    
    success: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation"""
        return {
            'tree': self.tree.to_dict(),
            'best_solution': self.best_solution.to_dict() if self.best_solution else None,
            'all_solutions': [sol.to_dict() for sol in self.all_solutions],
            'total_time': self.total_time,
            'total_llm_calls': self.total_llm_calls,
            'total_tokens_used': self.total_tokens_used,
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': self.nodes_pruned,
            'max_depth_reached': self.max_depth_reached,
            'success': self.success,
            'error_message': self.error_message
        }
