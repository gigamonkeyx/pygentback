"""
Retrieval Scoring and Ranking

This module provides sophisticated scoring and ranking algorithms for retrieval results.
"""

import logging
import math
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import RetrievalQuery, RetrievalResult


logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Configuration for scoring weights"""
    similarity: float = 0.5
    temporal: float = 0.15
    authority: float = 0.15
    quality: float = 0.1
    keyword: float = 0.1
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.similarity + self.temporal + self.authority + self.quality + self.keyword
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Scoring weights sum to {total}, not 1.0. Consider normalizing.")


class RetrievalScorer:
    """
    Advanced scoring system for retrieval results.
    
    Combines multiple signals including semantic similarity, temporal relevance,
    source authority, content quality, and keyword matching to produce
    comprehensive relevance scores.
    """
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize the scorer with configurable weights.
        
        Args:
            weights: Scoring weights configuration
        """
        self.weights = weights or ScoringWeights()
        
        # Authority indicators for different domains
        self.authority_indicators = {
            'high': [
                'docs.', 'documentation', 'official', 'github.com',
                'stackoverflow.com', 'wikipedia.org', '.edu', '.gov',
                'microsoft.com', 'google.com', 'mozilla.org'
            ],
            'medium': [
                'medium.com', 'dev.to', 'hackernoon.com', 'towards',
                'blog.', 'tutorial', 'guide'
            ],
            'low': [
                'forum', 'discussion', 'comment', 'social'
            ]
        }
        
        # Quality indicators
        self.quality_indicators = {
            'positive': [
                'example', 'tutorial', 'guide', 'documentation',
                'explanation', 'overview', 'introduction'
            ],
            'negative': [
                'error', 'broken', 'deprecated', 'outdated',
                'todo', 'fixme', 'hack'
            ]
        }
    
    async def score_results(self, results: List[RetrievalResult], 
                           query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Score and rank retrieval results.
        
        Args:
            results: List of retrieval results to score
            query: Original retrieval query
            
        Returns:
            List[RetrievalResult]: Scored and ranked results
        """
        if not results:
            return results
        
        # Score each result
        for result in results:
            await self._score_result(result, query)
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply diversity if requested
        if query.diversify:
            results = self._apply_diversity(results, query)
        
        return results
    
    async def _score_result(self, result: RetrievalResult, query: RetrievalQuery) -> None:
        """Calculate comprehensive relevance score for a single result"""
        
        # Base similarity score (already calculated)
        similarity_score = result.similarity_score
        
        # Temporal relevance score
        temporal_score = self._calculate_temporal_score(result.metadata)
        
        # Source authority score
        authority_score = self._calculate_authority_score(result)
        
        # Content quality score
        quality_score = self._calculate_quality_score(result.content, result.metadata)
        
        # Keyword matching score
        keyword_score = self._calculate_keyword_score(result.content, query.text)
        
        # Store individual scores
        result.semantic_score = similarity_score
        result.temporal_score = temporal_score
        result.authority_score = authority_score
        result.quality_score = quality_score
        result.keyword_score = keyword_score
        
        # Calculate weighted relevance score
        relevance_score = (
            similarity_score * self.weights.similarity +
            temporal_score * self.weights.temporal +
            authority_score * self.weights.authority +
            quality_score * self.weights.quality +
            keyword_score * self.weights.keyword
        )
        
        # Apply query-specific boosts
        relevance_score = self._apply_query_boosts(relevance_score, result, query)
        
        result.relevance_score = min(1.0, max(0.0, relevance_score))
    
    def _calculate_temporal_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate score based on document recency"""
        try:
            created_at = metadata.get('created_at')
            if not created_at:
                return 0.5  # Neutral score for unknown dates
            
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            
            # Calculate age in days
            age_days = (datetime.utcnow() - created_at).days
            
            # Exponential decay function: newer documents get higher scores
            if age_days <= 1:
                return 1.0
            elif age_days <= 7:
                return 0.9
            elif age_days <= 30:
                return 0.8
            elif age_days <= 90:
                return 0.6
            elif age_days <= 365:
                return 0.4
            elif age_days <= 730:  # 2 years
                return 0.2
            else:
                return 0.1
                
        except Exception as e:
            logger.debug(f"Failed to calculate temporal score: {str(e)}")
            return 0.5
    
    def _calculate_authority_score(self, result: RetrievalResult) -> float:
        """Calculate score based on source authority"""
        source_url = result.source_url or ""
        source_path = result.source_path or ""
        source = (source_url + source_path).lower()
        
        # Check for high authority indicators
        for indicator in self.authority_indicators['high']:
            if indicator in source:
                return 0.9
        
        # Check for medium authority indicators
        for indicator in self.authority_indicators['medium']:
            if indicator in source:
                return 0.7
        
        # Check for low authority indicators
        for indicator in self.authority_indicators['low']:
            if indicator in source:
                return 0.3
        
        # Check metadata for authority signals
        metadata = result.metadata
        if metadata.get('verified_source'):
            return 0.8
        
        if metadata.get('peer_reviewed'):
            return 0.9
        
        # Default neutral score
        return 0.5
    
    def _calculate_quality_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate score based on content quality"""
        if not content:
            return 0.0
        
        content_lower = content.lower()
        
        # Length-based quality (prefer substantial content)
        length_score = min(1.0, len(content) / 500)
        
        # Structure indicators (punctuation, formatting)
        structure_indicators = ['.', ':', ';', '\n', '?', '!', '-', '*']
        structure_count = sum(1 for indicator in structure_indicators if indicator in content)
        structure_score = min(1.0, structure_count / 20)
        
        # Positive quality indicators
        positive_score = 0.0
        for indicator in self.quality_indicators['positive']:
            if indicator in content_lower:
                positive_score += 0.1
        positive_score = min(1.0, positive_score)
        
        # Negative quality indicators
        negative_score = 0.0
        for indicator in self.quality_indicators['negative']:
            if indicator in content_lower:
                negative_score += 0.2
        negative_score = min(1.0, negative_score)
        
        # Avoid very short content
        if len(content) < 50:
            return 0.3
        
        # Avoid very repetitive content
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        if total_words > 0:
            uniqueness_score = unique_words / total_words
        else:
            uniqueness_score = 0.0
        
        # Combine quality factors
        base_quality = (
            length_score * 0.3 +
            structure_score * 0.2 +
            positive_score * 0.3 +
            uniqueness_score * 0.2
        )
        
        # Apply negative penalty
        final_quality = base_quality * (1.0 - negative_score * 0.5)
        
        return max(0.0, min(1.0, final_quality))
    
    def _calculate_keyword_score(self, content: str, query_text: str) -> float:
        """Calculate score based on keyword matching"""
        if not content or not query_text:
            return 0.0
        
        content_lower = content.lower()
        query_words = query_text.lower().split()
        
        if not query_words:
            return 0.0
        
        # Count exact word matches
        exact_matches = 0
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                if f" {word} " in f" {content_lower} ":
                    exact_matches += 1
        
        # Count partial matches
        partial_matches = 0
        for word in query_words:
            if len(word) > 3:  # Only for longer words
                if word in content_lower and f" {word} " not in f" {content_lower} ":
                    partial_matches += 1
        
        # Calculate keyword score
        exact_score = exact_matches / len(query_words)
        partial_score = (partial_matches / len(query_words)) * 0.5
        
        return min(1.0, exact_score + partial_score)
    
    def _apply_query_boosts(self, base_score: float, result: RetrievalResult, 
                           query: RetrievalQuery) -> float:
        """Apply query-specific boosts to the score"""
        boosted_score = base_score
        
        # Context-based boosts
        if query.context:
            context_lower = query.context.lower()
            content_lower = result.content.lower()
            
            # Boost for context relevance
            if any(word in content_lower for word in context_lower.split()):
                boosted_score *= 1.1
        
        # Collection-based boosts
        if result.collection:
            if result.collection == "knowledge_base":
                boosted_score *= 1.05  # Slight boost for curated knowledge
            elif result.collection == "recent_documents":
                boosted_score *= 1.02  # Slight boost for recent content
        
        # Document type boosts
        doc_type = result.metadata.get('document_type', '')
        if doc_type in ['documentation', 'tutorial', 'guide']:
            boosted_score *= 1.1
        elif doc_type in ['code', 'example']:
            if 'code' in query.text.lower() or 'example' in query.text.lower():
                boosted_score *= 1.15
        
        return boosted_score
    
    def _apply_diversity(self, results: List[RetrievalResult], 
                        query: RetrievalQuery) -> List[RetrievalResult]:
        """Apply diversity to avoid too many similar results"""
        if len(results) <= 3:
            return results
        
        diverse_results = []
        seen_sources = set()
        seen_content_hashes = set()
        
        for result in results:
            # Check source diversity
            source_key = result.source_url or result.source_path or "unknown"
            
            # Check content diversity (simple hash of first 100 chars)
            content_hash = hash(result.content[:100])
            
            # Add if diverse enough or if we don't have many results yet
            if (len(diverse_results) < 3 or 
                (source_key not in seen_sources and content_hash not in seen_content_hashes)):
                diverse_results.append(result)
                seen_sources.add(source_key)
                seen_content_hashes.add(content_hash)
            
            # Stop when we have enough diverse results
            if len(diverse_results) >= query.max_results:
                break
        
        return diverse_results
