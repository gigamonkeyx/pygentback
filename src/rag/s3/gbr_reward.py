"""
Gain Beyond RAG (GBR) Reward Calculator

Implements the GBR reward signal that measures the improvement of s3 RAG
over baseline RAG approaches. This is the key innovation that enables
training with minimal data (2.4k vs 70k+ samples).
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import time

from .models import GBRReward, SearchState, S3Config

logger = logging.getLogger(__name__)


class BaselineRAG(ABC):
    """Abstract base class for baseline RAG implementations"""
    
    @abstractmethod
    async def retrieve_and_generate(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Retrieve documents and generate response"""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of this baseline method"""
        pass


class NaiveRAG(BaselineRAG):
    """Simple baseline RAG implementation"""
    
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    async def retrieve_and_generate(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Simple retrieve-then-generate approach"""
        try:
            # Retrieve documents
            start_time = time.time()
            documents = await self.retriever.retrieve(query, k=k)
            retrieval_time = time.time() - start_time
            
            # Generate response
            start_time = time.time()
            context = "\n".join([doc.get('content', '') for doc in documents])
            response = await self.generator.generate(query, context)
            generation_time = time.time() - start_time
            
            return {
                'response': response,
                'documents': documents,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': retrieval_time + generation_time,
                'num_documents': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Naive RAG failed: {e}")
            return {
                'response': "",
                'documents': [],
                'retrieval_time': 0.0,
                'generation_time': 0.0,
                'total_time': 0.0,
                'num_documents': 0
            }
    
    def get_method_name(self) -> str:
        return "naive_rag"


class DenseRAG(BaselineRAG):
    """Dense retrieval baseline"""
    
    def __init__(self, dense_retriever, generator):
        self.dense_retriever = dense_retriever
        self.generator = generator
    
    async def retrieve_and_generate(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Dense retrieval with reranking"""
        try:
            # Dense retrieval
            start_time = time.time()
            documents = await self.dense_retriever.retrieve(query, k=k*2)  # Retrieve more for reranking
            
            # Simple reranking (could use more sophisticated methods)
            if len(documents) > k:
                # Sort by score and take top k
                documents = sorted(documents, key=lambda x: x.get('score', 0), reverse=True)[:k]
            
            retrieval_time = time.time() - start_time
            
            # Generate response
            start_time = time.time()
            context = "\n".join([doc.get('content', '') for doc in documents])
            response = await self.generator.generate(query, context)
            generation_time = time.time() - start_time
            
            return {
                'response': response,
                'documents': documents,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': retrieval_time + generation_time,
                'num_documents': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Dense RAG failed: {e}")
            return {
                'response': "",
                'documents': [],
                'retrieval_time': 0.0,
                'generation_time': 0.0,
                'total_time': 0.0,
                'num_documents': 0
            }
    
    def get_method_name(self) -> str:
        return "dense_rag"


class GBRRewardCalculator:
    """
    Calculates Gain Beyond RAG (GBR) rewards
    
    The GBR reward measures how much better the s3 approach performs
    compared to baseline RAG methods. This enables training with minimal
    data by providing a strong learning signal.
    """
    
    def __init__(self, config: S3Config, baseline_methods: List[BaselineRAG] = None):
        self.config = config
        self.baseline_methods = baseline_methods or []
        
        # Evaluation metrics
        self.evaluator = ResponseEvaluator()
        
        # Statistics
        self.reward_history: List[GBRReward] = []
        self.baseline_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized GBR calculator with {len(self.baseline_methods)} baseline methods")
    
    async def calculate_gbr_reward(self, query: str, s3_state: SearchState, 
                                 s3_response: str) -> GBRReward:
        """
        Calculate GBR reward for an s3 search result
        
        Args:
            query: Original query
            s3_state: Final search state from s3 agent
            s3_response: Generated response using s3 results
            
        Returns:
            GBRReward object with detailed reward breakdown
        """
        reward = GBRReward(
            query=query,
            num_documents=s3_state.total_documents,
            search_iterations=s3_state.iteration
        )
        
        start_time = time.time()
        
        try:
            # Evaluate s3 performance
            s3_score = await self._evaluate_s3_performance(query, s3_state, s3_response)
            reward.s3_score = s3_score
            reward.s3_method = "s3_rag"
            
            # Get baseline performance
            baseline_score = await self._get_baseline_performance(query)
            reward.baseline_score = baseline_score
            
            # Calculate component scores
            await self._calculate_component_scores(reward, query, s3_state, s3_response)
            
            # Calculate gain metrics
            reward.calculate_gain()
            
            reward.computation_time = time.time() - start_time
            
            # Store in history
            self.reward_history.append(reward)
            
            logger.debug(f"GBR reward calculated: {reward.normalized_gain:.3f} "
                        f"(s3: {s3_score:.3f}, baseline: {baseline_score:.3f})")
            
            return reward
            
        except Exception as e:
            logger.error(f"Failed to calculate GBR reward: {e}")
            reward.computation_time = time.time() - start_time
            return reward
    
    async def _evaluate_s3_performance(self, query: str, state: SearchState, 
                                     response: str) -> float:
        """Evaluate the performance of s3 search and generation"""
        scores = []
        
        # Relevance score
        relevance = await self.evaluator.evaluate_relevance(query, response, state.documents)
        scores.append(relevance)
        
        # Diversity score
        diversity = self.evaluator.evaluate_diversity(state.documents)
        scores.append(diversity)
        
        # Efficiency score (inverse of search iterations)
        max_iterations = self.config.max_search_iterations
        efficiency = 1.0 - (state.iteration / max_iterations)
        scores.append(efficiency)
        
        # Coherence score
        coherence = await self.evaluator.evaluate_coherence(response)
        scores.append(coherence)
        
        # Weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # Relevance weighted highest
        return sum(score * weight for score, weight in zip(scores, weights))
    
    async def _get_baseline_performance(self, query: str) -> float:
        """Get baseline performance (cached or computed)"""
        # Check cache first
        if query in self.baseline_cache:
            return self.baseline_cache[query]['score']
        
        if not self.baseline_methods:
            # No baseline methods available, use default score
            return 0.5
        
        baseline_scores = []
        
        for baseline in self.baseline_methods:
            try:
                result = await baseline.retrieve_and_generate(query)
                score = await self.evaluator.evaluate_relevance(
                    query, result['response'], result['documents']
                )
                baseline_scores.append(score)
                
            except Exception as e:
                logger.warning(f"Baseline {baseline.get_method_name()} failed: {e}")
                baseline_scores.append(0.0)
        
        # Use best baseline performance
        best_score = max(baseline_scores) if baseline_scores else 0.5
        
        # Cache result
        self.baseline_cache[query] = {
            'score': best_score,
            'scores': baseline_scores,
            'timestamp': time.time()
        }
        
        return best_score
    
    async def _calculate_component_scores(self, reward: GBRReward, query: str,
                                        state: SearchState, response: str) -> None:
        """Calculate individual component scores"""
        try:
            # Relevance score
            reward.relevance_score = await self.evaluator.evaluate_relevance(
                query, response, state.documents
            )
            
            # Diversity score
            reward.diversity_score = self.evaluator.evaluate_diversity(state.documents)
            
            # Efficiency score
            max_iterations = self.config.max_search_iterations
            reward.efficiency_score = 1.0 - (state.iteration / max_iterations)
            
            # Coherence score
            reward.coherence_score = await self.evaluator.evaluate_coherence(response)
            
        except Exception as e:
            logger.error(f"Failed to calculate component scores: {e}")
            # Set default scores
            reward.relevance_score = 0.5
            reward.diversity_score = 0.5
            reward.efficiency_score = 0.5
            reward.coherence_score = 0.5
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward calculations"""
        if not self.reward_history:
            return {}
        
        gains = [r.normalized_gain for r in self.reward_history]
        s3_scores = [r.s3_score for r in self.reward_history]
        baseline_scores = [r.baseline_score for r in self.reward_history]
        
        return {
            'total_rewards': len(self.reward_history),
            'average_gain': np.mean(gains),
            'std_gain': np.std(gains),
            'max_gain': np.max(gains),
            'min_gain': np.min(gains),
            'average_s3_score': np.mean(s3_scores),
            'average_baseline_score': np.mean(baseline_scores),
            'improvement_rate': np.mean([r.s3_score > r.baseline_score for r in self.reward_history]),
            'cache_size': len(self.baseline_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear baseline performance cache"""
        self.baseline_cache.clear()
        logger.info("Baseline performance cache cleared")


class ResponseEvaluator:
    """Evaluates the quality of generated responses"""
    
    async def evaluate_relevance(self, query: str, response: str, 
                               documents: List[Dict[str, Any]]) -> float:
        """Evaluate relevance of response to query"""
        try:
            # Simple relevance scoring (could use more sophisticated methods)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            # Jaccard similarity
            intersection = len(query_words & response_words)
            union = len(query_words | response_words)
            
            if union == 0:
                return 0.0
            
            jaccard = intersection / union
            
            # Boost score if response uses document content
            doc_boost = 0.0
            if documents:
                doc_words = set()
                for doc in documents:
                    content = doc.get('content', '')
                    doc_words.update(content.lower().split())
                
                doc_intersection = len(response_words & doc_words)
                if len(response_words) > 0:
                    doc_boost = doc_intersection / len(response_words)
            
            # Combine scores
            relevance = 0.6 * jaccard + 0.4 * doc_boost
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            return 0.5
    
    def evaluate_diversity(self, documents: List[Dict[str, Any]]) -> float:
        """Evaluate diversity of retrieved documents"""
        if len(documents) <= 1:
            return 0.0
        
        try:
            # Simple diversity based on content overlap
            contents = [doc.get('content', '') for doc in documents]
            
            total_pairs = 0
            diverse_pairs = 0
            
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    total_pairs += 1
                    
                    # Calculate content similarity
                    words_i = set(contents[i].lower().split())
                    words_j = set(contents[j].lower().split())
                    
                    if len(words_i | words_j) > 0:
                        similarity = len(words_i & words_j) / len(words_i | words_j)
                        if similarity < 0.7:  # Consider diverse if similarity < 70%
                            diverse_pairs += 1
            
            return diverse_pairs / total_pairs if total_pairs > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Diversity evaluation failed: {e}")
            return 0.5
    
    async def evaluate_coherence(self, response: str) -> float:
        """Evaluate coherence of generated response"""
        try:
            # Simple coherence metrics
            sentences = response.split('.')
            if len(sentences) <= 1:
                return 0.8  # Short responses are generally coherent
            
            # Check for repetition
            sentence_words = [set(s.lower().split()) for s in sentences if s.strip()]
            
            if len(sentence_words) <= 1:
                return 0.8
            
            # Calculate average similarity between consecutive sentences
            similarities = []
            for i in range(len(sentence_words) - 1):
                words_curr = sentence_words[i]
                words_next = sentence_words[i + 1]
                
                if len(words_curr | words_next) > 0:
                    sim = len(words_curr & words_next) / len(words_curr | words_next)
                    similarities.append(sim)
            
            if not similarities:
                return 0.8
            
            # Good coherence has moderate similarity (not too high, not too low)
            avg_sim = np.mean(similarities)
            
            # Optimal similarity is around 0.3-0.7
            if 0.3 <= avg_sim <= 0.7:
                coherence = 1.0
            elif avg_sim < 0.3:
                coherence = avg_sim / 0.3  # Penalize low similarity
            else:
                coherence = 1.0 - (avg_sim - 0.7) / 0.3  # Penalize high similarity
            
            return max(0.0, min(1.0, coherence))
            
        except Exception as e:
            logger.error(f"Coherence evaluation failed: {e}")
            return 0.5
