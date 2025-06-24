"""
Reasoning state management for Tree of Thought.

This module manages the overall state of a reasoning session,
tracking thoughts, evaluations, and session metadata.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from .thought import Thought, ThoughtType


class ReasoningState:
    """
    Manages the state of a ToT reasoning session.
    
    Tracks:
    - All thoughts generated during the session
    - Evaluation results and metrics
    - Session metadata and configuration
    - Timing and performance data
    """
    
    def __init__(self, problem: str, session_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize reasoning state.
        
        Args:
            problem: The original problem statement
            session_id: Unique identifier for this reasoning session
            metadata: Additional session metadata
        """
        self.problem = problem
        self.session_id = session_id
        self.thoughts: List[Thought] = []
        self.evaluations: Dict[str, Any] = {}
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
    def add_thought(self, thought: Thought):
        """Add a thought to the reasoning state"""
        self.thoughts.append(thought)
        self.updated_at = datetime.now()
        
    def remove_thought(self, thought_id: str) -> bool:
        """Remove a thought by ID. Returns True if found and removed."""
        for i, thought in enumerate(self.thoughts):
            if thought.id == thought_id:
                del self.thoughts[i]
                self.updated_at = datetime.now()
                return True
        return False
        
    def get_thought(self, thought_id: str) -> Optional[Thought]:
        """Get a thought by ID"""
        for thought in self.thoughts:
            if thought.id == thought_id:
                return thought
        return None
        
    def get_thoughts_by_type(self, thought_type: ThoughtType) -> List[Thought]:
        """Get all thoughts of a specific type"""
        return [t for t in self.thoughts if t.thought_type == thought_type]
        
    def get_thoughts_by_parent(self, parent_id: Optional[str]) -> List[Thought]:
        """Get all thoughts with the specified parent ID"""
        return [t for t in self.thoughts if t.parent_id == parent_id]
        
    def get_root_thoughts(self) -> List[Thought]:
        """Get all root thoughts (no parent)"""
        return self.get_thoughts_by_parent(None)
        
    def get_leaf_thoughts(self) -> List[Thought]:
        """Get all leaf thoughts (no children)"""
        leaf_thoughts = []
        parent_ids = {t.parent_id for t in self.thoughts if t.parent_id is not None}
        
        for thought in self.thoughts:
            if thought.id not in parent_ids:
                leaf_thoughts.append(thought)
                
        return leaf_thoughts
        
    def get_best_thoughts(self, limit: int = 5) -> List[Thought]:
        """Get the best thoughts by confidence score"""
        sorted_thoughts = sorted(self.thoughts, key=lambda t: t.confidence, reverse=True)
        return sorted_thoughts[:limit]
        
    def get_thoughts_by_depth(self, depth: int) -> List[Thought]:
        """Get all thoughts at a specific depth"""
        return [t for t in self.thoughts if t.depth == depth]
        
    def get_max_depth(self) -> int:
        """Get the maximum depth of thoughts in this state"""
        if not self.thoughts:
            return 0
        return max(t.depth for t in self.thoughts)
        
    def add_evaluation(self, evaluation_name: str, result: Any):
        """Add an evaluation result"""
        self.evaluations[evaluation_name] = result
        self.updated_at = datetime.now()
        
    def get_evaluation(self, evaluation_name: str) -> Any:
        """Get an evaluation result"""
        return self.evaluations.get(evaluation_name)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reasoning state"""
        type_counts = {}
        for thought_type in ThoughtType:
            type_counts[thought_type.value] = len(self.get_thoughts_by_type(thought_type))
            
        return {
            'total_thoughts': len(self.thoughts),
            'type_distribution': type_counts,
            'max_depth': self.get_max_depth(),
            'avg_confidence': sum(t.confidence for t in self.thoughts) / len(self.thoughts) if self.thoughts else 0.0,
            'session_duration': (self.updated_at - self.created_at).total_seconds(),
            'evaluations_count': len(self.evaluations)
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert reasoning state to dictionary"""
        return {
            'problem': self.problem,
            'session_id': self.session_id,
            'thoughts': [t.to_dict() for t in self.thoughts],
            'evaluations': self.evaluations,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningState':
        """Create reasoning state from dictionary"""
        state = cls(
            problem=data['problem'],
            session_id=data['session_id'],
            metadata=data.get('metadata', {})
        )
        
        # Restore thoughts
        for thought_data in data.get('thoughts', []):
            thought = Thought.from_dict(thought_data)
            state.thoughts.append(thought)
            
        state.evaluations = data.get('evaluations', {})
        
        if 'created_at' in data:
            state.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            state.updated_at = datetime.fromisoformat(data['updated_at'])
            
        return state
        
    def __repr__(self) -> str:
        return f"ReasoningState(session_id='{self.session_id}', thoughts={len(self.thoughts)})"
