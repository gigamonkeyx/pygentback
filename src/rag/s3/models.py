"""
s3 RAG Data Models

Core data structures for the s3 RAG framework including search states,
actions, rewards, and configuration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import uuid
from datetime import datetime


class SearchStrategy(Enum):
    """Search strategy for s3 agent"""
    ITERATIVE_REFINEMENT = "iterative_refinement"
    MULTI_QUERY = "multi_query"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class RewardType(Enum):
    """Type of reward signal"""
    GBR = "gain_beyond_rag"        # Gain Beyond RAG
    RELEVANCE = "relevance"        # Document relevance
    DIVERSITY = "diversity"        # Result diversity
    EFFICIENCY = "efficiency"      # Search efficiency


@dataclass
class S3Config:
    """Configuration for s3 RAG system"""
    # Search configuration
    search_strategy: SearchStrategy = SearchStrategy.ITERATIVE_REFINEMENT
    max_search_iterations: int = 5
    max_documents_per_iteration: int = 10
    similarity_threshold: float = 0.7
    
    # Agent configuration
    agent_model: str = "phi4-fast"
    generator_model: str = "phi4-fast"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Training configuration
    training_episodes: int = 1000
    batch_size: int = 32
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    
    # Reward configuration
    reward_types: List[RewardType] = field(default_factory=lambda: [RewardType.GBR])
    gbr_weight: float = 1.0
    relevance_weight: float = 0.5
    diversity_weight: float = 0.3
    efficiency_weight: float = 0.2
    
    # Memory configuration
    experience_buffer_size: int = 10000
    min_buffer_size: int = 1000
    
    # Evaluation configuration
    eval_frequency: int = 100
    eval_episodes: int = 50


@dataclass
class SearchAction:
    """Represents a search action taken by the s3 agent"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = "query"  # query, refine, expand, filter
    query: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Action metadata
    iteration: int = 0
    confidence: float = 0.0
    expected_improvement: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'action_id': self.action_id,
            'action_type': self.action_type,
            'query': self.query,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'iteration': self.iteration,
            'confidence': self.confidence,
            'expected_improvement': self.expected_improvement
        }


@dataclass
class SearchState:
    """Represents the current state of the search process"""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = ""
    current_query: str = ""
    iteration: int = 0
    
    # Retrieved documents
    documents: List[Dict[str, Any]] = field(default_factory=list)
    document_scores: List[float] = field(default_factory=list)
    
    # Search history
    action_history: List[SearchAction] = field(default_factory=list)
    query_history: List[str] = field(default_factory=list)
    
    # State metrics
    total_documents: int = 0
    unique_documents: int = 0
    average_relevance: float = 0.0
    search_efficiency: float = 0.0
    
    # Termination conditions
    is_terminal: bool = False
    termination_reason: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_action(self, action: SearchAction) -> None:
        """Add an action to the history"""
        self.action_history.append(action)
        self.query_history.append(action.query)
        self.iteration = len(self.action_history)
        self.updated_at = datetime.utcnow()
    
    def add_documents(self, docs: List[Dict[str, Any]], scores: List[float]) -> None:
        """Add retrieved documents with scores"""
        self.documents.extend(docs)
        self.document_scores.extend(scores)
        self.total_documents = len(self.documents)
        
        # Calculate unique documents (by ID or content hash)
        unique_ids = set()
        for doc in self.documents:
            doc_id = doc.get('id', hash(doc.get('content', '')))
            unique_ids.add(doc_id)
        self.unique_documents = len(unique_ids)
        
        # Update average relevance
        if self.document_scores:
            self.average_relevance = sum(self.document_scores) / len(self.document_scores)
        
        self.updated_at = datetime.utcnow()
    
    def get_top_documents(self, k: int = 5) -> List[Dict[str, Any]]:
        """Get top k documents by score"""
        if not self.documents:
            return []
        
        # Sort by score (descending)
        doc_score_pairs = list(zip(self.documents, self.document_scores))
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in sorted_pairs[:k]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'state_id': self.state_id,
            'original_query': self.original_query,
            'current_query': self.current_query,
            'iteration': self.iteration,
            'documents': self.documents,
            'document_scores': self.document_scores,
            'action_history': [action.to_dict() for action in self.action_history],
            'query_history': self.query_history,
            'total_documents': self.total_documents,
            'unique_documents': self.unique_documents,
            'average_relevance': self.average_relevance,
            'search_efficiency': self.search_efficiency,
            'is_terminal': self.is_terminal,
            'termination_reason': self.termination_reason,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class GBRReward:
    """Gain Beyond RAG reward calculation result"""
    reward_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Baseline performance
    baseline_score: float = 0.0
    baseline_method: str = "naive_rag"
    
    # s3 performance
    s3_score: float = 0.0
    s3_method: str = "s3_rag"
    
    # Gain calculation
    absolute_gain: float = 0.0
    relative_gain: float = 0.0
    normalized_gain: float = 0.0
    
    # Component scores
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    efficiency_score: float = 0.0
    coherence_score: float = 0.0
    
    # Metadata
    query: str = ""
    num_documents: int = 0
    search_iterations: int = 0
    computation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_gain(self) -> None:
        """Calculate gain metrics"""
        self.absolute_gain = self.s3_score - self.baseline_score
        
        if self.baseline_score > 0:
            self.relative_gain = self.absolute_gain / self.baseline_score
        else:
            self.relative_gain = 0.0
        
        # Normalize gain to [0, 1] range
        self.normalized_gain = max(0.0, min(1.0, (self.relative_gain + 1.0) / 2.0))
    
    def get_composite_reward(self, weights: Dict[str, float] = None) -> float:
        """Calculate composite reward from all components"""
        if weights is None:
            weights = {
                'gbr': 1.0,
                'relevance': 0.5,
                'diversity': 0.3,
                'efficiency': 0.2,
                'coherence': 0.4
            }
        
        composite = (
            weights.get('gbr', 1.0) * self.normalized_gain +
            weights.get('relevance', 0.5) * self.relevance_score +
            weights.get('diversity', 0.3) * self.diversity_score +
            weights.get('efficiency', 0.2) * self.efficiency_score +
            weights.get('coherence', 0.4) * self.coherence_score
        )
        
        # Normalize by total weights
        total_weight = sum(weights.values())
        return composite / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'reward_id': self.reward_id,
            'baseline_score': self.baseline_score,
            'baseline_method': self.baseline_method,
            's3_score': self.s3_score,
            's3_method': self.s3_method,
            'absolute_gain': self.absolute_gain,
            'relative_gain': self.relative_gain,
            'normalized_gain': self.normalized_gain,
            'relevance_score': self.relevance_score,
            'diversity_score': self.diversity_score,
            'efficiency_score': self.efficiency_score,
            'coherence_score': self.coherence_score,
            'query': self.query,
            'num_documents': self.num_documents,
            'search_iterations': self.search_iterations,
            'computation_time': self.computation_time,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class S3Experience:
    """Experience tuple for reinforcement learning"""
    # RL tuple components (required fields first)
    state: SearchState
    action: SearchAction
    reward: float
    next_state: SearchState
    done: bool

    # Additional metadata (optional fields with defaults)
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str = ""
    step: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'experience_id': self.experience_id,
            'state': self.state.to_dict(),
            'action': self.action.to_dict(),
            'reward': self.reward,
            'next_state': self.next_state.to_dict(),
            'done': self.done,
            'episode_id': self.episode_id,
            'step': self.step,
            'timestamp': self.timestamp.isoformat()
        }
