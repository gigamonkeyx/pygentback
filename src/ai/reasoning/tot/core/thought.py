"""
Core thought data structure for Tree of Thought reasoning.

This module defines the fundamental Thought class that represents individual
reasoning steps in the ToT framework.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class ThoughtType(Enum):
    """Type classification for thoughts in the reasoning tree"""
    PROBLEM = "problem"      # Initial problem statement
    REASONING = "reasoning"  # Intermediate reasoning step
    SOLUTION = "solution"    # Final solution or answer
    CRITIQUE = "critique"    # Critical evaluation of other thoughts
    REFINEMENT = "refinement"  # Improved version of previous thought


@dataclass
class ThoughtMetrics:
    """Metrics and evaluation scores for a thought"""
    value_score: Optional[float] = None
    vote_score: Optional[float] = None
    evaluation_count: int = 0
    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    def update_score(self, metric_name: str, score: float):
        """Update a specific metric score"""
        setattr(self, metric_name, score)
    
    def get_score(self, metric_name: str) -> Optional[float]:
        """Get a specific metric score"""
        return getattr(self, metric_name, None)


class Thought:
    """
    Represents a single thought/reasoning step in the Tree of Thought.
    
    Each thought contains:
    - Content: The actual reasoning text
    - Type: Classification of the thought's role
    - Relationships: Parent/child connections in the tree
    - Evaluation: Confidence scores and metadata
    - Timestamps: Creation and update tracking
    """
    
    def __init__(
        self,
        content: str,
        thought_type: ThoughtType,
        parent_id: Optional[str] = None,
        depth: int = 0,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new thought.
        
        Args:
            content: The textual content of the thought
            thought_type: Classification of this thought's role
            parent_id: ID of the parent thought (None for root)
            depth: Depth level in the reasoning tree
            confidence: Initial confidence score (0.0-1.0)
            metadata: Additional metadata dictionary
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.thought_type = thought_type
        self.parent_id = parent_id
        self.depth = depth
        self.confidence = confidence
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.evaluation_scores = ThoughtMetrics()
        
    def update_confidence(self, new_confidence: float):
        """Update the confidence score for this thought"""
        self.confidence = max(0.0, min(1.0, new_confidence))
        
    def add_evaluation_score(self, metric: str, score: float):
        """Add an evaluation score for a specific metric"""
        self.evaluation_scores.update_score(metric, score)
        
    def get_evaluation_score(self, metric: str) -> Optional[float]:
        """Get evaluation score for a specific metric"""
        return self.evaluation_scores.get_score(metric)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert thought to dictionary representation"""
        return {
            'id': self.id,
            'content': self.content,
            'thought_type': self.thought_type.value,
            'parent_id': self.parent_id,
            'depth': self.depth,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'evaluation_scores': self.evaluation_scores.__dict__
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Thought':
        """Create thought from dictionary representation"""
        thought = cls(
            content=data['content'],
            thought_type=ThoughtType(data['thought_type']),
            parent_id=data.get('parent_id'),
            depth=data.get('depth', 0),
            confidence=data.get('confidence', 0.0),
            metadata=data.get('metadata', {})
        )
        thought.id = data['id']
        thought.evaluation_scores = ThoughtMetrics(**data.get('evaluation_scores', {}))
        if 'created_at' in data:
            thought.created_at = datetime.fromisoformat(data['created_at'])
        return thought
        
    def __repr__(self) -> str:
        return f"Thought(id='{self.id[:8]}...', type={self.thought_type.value}, confidence={self.confidence:.2f})"
        
    def __str__(self) -> str:
        return f"[{self.thought_type.value.upper()}] {self.content[:100]}{'...' if len(self.content) > 100 else ''}"
