"""
s3 RAG Framework

Implementation of the s3 (Search, Select, Synthesize) RAG framework that
decouples search and generation, trains search agents with minimal data,
and uses Gain Beyond RAG (GBR) reward signals.

Based on the research paper that achieves superior performance with 90% less
training data compared to traditional RAG approaches.
"""

from .models import SearchState, SearchAction, GBRReward, S3Config
from .search_agent import S3SearchAgent, SearchStrategy
from .gbr_reward import GBRRewardCalculator, BaselineRAG
from .rl_trainer import S3RLTrainer, PPOConfig
from .s3_pipeline import S3Pipeline, S3Result

__all__ = [
    # Core models
    'SearchState',
    'SearchAction', 
    'GBRReward',
    'S3Config',
    
    # Search agent
    'S3SearchAgent',
    'SearchStrategy',
    
    # Reward system
    'GBRRewardCalculator',
    'BaselineRAG',
    
    # Training
    'S3RLTrainer',
    'PPOConfig',
    
    # Pipeline
    'S3Pipeline',
    'S3Result'
]
