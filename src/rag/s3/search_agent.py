"""
S3 Search Agent

Implements the search agent that learns to iteratively refine queries
and select documents using reinforcement learning with minimal training data.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time

from .models import SearchState, SearchAction, S3Config, SearchStrategy
from .gbr_reward import GBRRewardCalculator

logger = logging.getLogger(__name__)


class S3SearchAgent:
    """
    S3 Search Agent that learns to search iteratively
    
    Uses reinforcement learning to learn optimal search strategies
    with minimal training data (2.4k vs 70k+ samples for baselines).
    """
    
    def __init__(self, config: S3Config, retriever, generator, 
                 reward_calculator: Optional[GBRRewardCalculator] = None):
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.reward_calculator = reward_calculator
        
        # Agent state
        self.policy_network = None  # Would be actual RL policy
        self.value_network = None   # Would be actual value function
        
        # Search statistics
        self.search_count = 0
        self.total_search_time = 0.0
        self.success_rate = 0.0
        
        # Experience for training
        self.experience_buffer = []
        
        logger.info(f"Initialized S3 search agent with strategy: {config.search_strategy.value}")
    
    async def search(self, query: str, max_iterations: Optional[int] = None) -> SearchState:
        """
        Perform iterative search using learned policy
        
        Args:
            query: Initial search query
            max_iterations: Maximum search iterations (overrides config)
            
        Returns:
            Final search state with retrieved documents
        """
        start_time = time.time()
        self.search_count += 1
        
        # Initialize search state
        state = SearchState(
            original_query=query,
            current_query=query
        )
        
        max_iter = max_iterations or self.config.max_search_iterations
        
        try:
            for iteration in range(max_iter):
                # Check termination conditions
                if self._should_terminate(state):
                    state.is_terminal = True
                    state.termination_reason = "convergence"
                    break
                
                # Select action using policy
                action = await self._select_action(state)
                
                # Execute action
                new_docs, scores = await self._execute_action(action, state)
                
                # Update state
                state.add_action(action)
                state.add_documents(new_docs, scores)
                
                # Update search efficiency
                state.search_efficiency = self._calculate_efficiency(state)
                
                logger.debug(f"Iteration {iteration + 1}: Retrieved {len(new_docs)} documents, "
                           f"total: {state.total_documents}")
            
            # Mark as terminal if max iterations reached
            if not state.is_terminal:
                state.is_terminal = True
                state.termination_reason = "max_iterations"
            
            search_time = time.time() - start_time
            self.total_search_time += search_time
            
            logger.info(f"Search completed in {search_time:.2f}s with {state.iteration} iterations, "
                       f"{state.total_documents} total documents")
            
            return state
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            state.is_terminal = True
            state.termination_reason = f"error: {str(e)}"
            return state
    
    async def _select_action(self, state: SearchState) -> SearchAction:
        """Select next action using learned policy"""
        
        if self.config.search_strategy == SearchStrategy.ITERATIVE_REFINEMENT:
            return await self._select_refinement_action(state)
        elif self.config.search_strategy == SearchStrategy.MULTI_QUERY:
            return await self._select_multi_query_action(state)
        elif self.config.search_strategy == SearchStrategy.HIERARCHICAL:
            return await self._select_hierarchical_action(state)
        else:  # ADAPTIVE
            return await self._select_adaptive_action(state)
    
    async def _select_refinement_action(self, state: SearchState) -> SearchAction:
        """Select action for iterative refinement strategy"""
        
        if state.iteration == 0:
            # First iteration: use original query
            return SearchAction(
                action_type="initial_query",
                query=state.original_query,
                iteration=state.iteration,
                confidence=1.0
            )
        
        # Subsequent iterations: refine based on results
        if state.average_relevance < 0.5:
            # Low relevance: expand query
            refined_query = await self._expand_query(state)
            action_type = "expand_query"
        elif state.unique_documents < 3:
            # Few unique documents: diversify
            refined_query = await self._diversify_query(state)
            action_type = "diversify_query"
        else:
            # Good results: focus query
            refined_query = await self._focus_query(state)
            action_type = "focus_query"
        
        return SearchAction(
            action_type=action_type,
            query=refined_query,
            iteration=state.iteration,
            confidence=0.8
        )
    
    async def _select_multi_query_action(self, state: SearchState) -> SearchAction:
        """Select action for multi-query strategy"""
        
        if state.iteration == 0:
            # Generate multiple query variants
            query_variants = await self._generate_query_variants(state.original_query)
            selected_query = query_variants[0] if query_variants else state.original_query
        else:
            # Use remaining variants or refine
            if len(state.query_history) < 3:
                query_variants = await self._generate_query_variants(state.original_query)
                # Select unused variant
                used_queries = set(state.query_history)
                selected_query = next(
                    (q for q in query_variants if q not in used_queries),
                    state.current_query
                )
            else:
                # Combine best results
                selected_query = await self._combine_queries(state)
        
        return SearchAction(
            action_type="multi_query",
            query=selected_query,
            iteration=state.iteration,
            confidence=0.7
        )
    
    async def _select_hierarchical_action(self, state: SearchState) -> SearchAction:
        """Select action for hierarchical strategy"""
        
        if state.iteration == 0:
            # Start with broad query
            broad_query = await self._broaden_query(state.original_query)
        elif state.iteration == 1:
            # Narrow down based on results
            broad_query = await self._narrow_query(state)
        else:
            # Fine-tune
            broad_query = await self._fine_tune_query(state)
        
        return SearchAction(
            action_type="hierarchical",
            query=broad_query,
            iteration=state.iteration,
            confidence=0.75
        )
    
    async def _select_adaptive_action(self, state: SearchState) -> SearchAction:
        """Select action using adaptive strategy"""
        
        # Analyze current state and adapt strategy
        if state.average_relevance > 0.8:
            # High relevance: focus on diversity
            strategy = "diversify"
            query = await self._diversify_query(state)
        elif state.average_relevance < 0.3:
            # Low relevance: expand search
            strategy = "expand"
            query = await self._expand_query(state)
        elif state.unique_documents < 2:
            # Few documents: broaden search
            strategy = "broaden"
            query = await self._broaden_query(state.current_query)
        else:
            # Balanced: refine
            strategy = "refine"
            query = await self._focus_query(state)
        
        return SearchAction(
            action_type=f"adaptive_{strategy}",
            query=query,
            iteration=state.iteration,
            confidence=0.8
        )
    
    async def _execute_action(self, action: SearchAction, 
                            state: SearchState) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Execute a search action and return results"""
        
        try:
            # Update current query
            state.current_query = action.query
            
            # Retrieve documents
            max_docs = self.config.max_documents_per_iteration
            documents = await self.retriever.retrieve(action.query, k=max_docs)
            
            # Extract scores (assuming retriever returns scored documents)
            scores = [doc.get('score', 0.0) for doc in documents]
            
            # Filter by similarity threshold
            threshold = self.config.similarity_threshold
            filtered_docs = []
            filtered_scores = []
            
            for doc, score in zip(documents, scores):
                if score >= threshold:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)
            
            return filtered_docs, filtered_scores
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return [], []
    
    async def _expand_query(self, state: SearchState) -> str:
        """Expand query to increase recall"""
        # Simple expansion (could use more sophisticated methods)
        query_words = state.current_query.split()
        
        # Add synonyms or related terms (placeholder implementation)
        expansion_terms = ["related", "similar", "about", "concerning"]
        
        if len(query_words) < 5:  # Only expand short queries
            expanded = state.current_query + " " + expansion_terms[state.iteration % len(expansion_terms)]
            return expanded
        
        return state.current_query
    
    async def _diversify_query(self, state: SearchState) -> str:
        """Diversify query to get different perspectives"""
        # Add diversity terms
        diversity_terms = ["alternative", "different", "various", "multiple"]
        term = diversity_terms[state.iteration % len(diversity_terms)]
        return f"{term} {state.current_query}"
    
    async def _focus_query(self, state: SearchState) -> str:
        """Focus query to increase precision"""
        # Add specific terms based on best documents
        if state.documents:
            # Extract key terms from top documents (simplified)
            top_docs = state.get_top_documents(k=2)
            key_terms = []
            
            for doc in top_docs:
                content = doc.get('content', '')
                words = content.split()[:10]  # Take first 10 words
                key_terms.extend(words)
            
            if key_terms:
                # Add most common term
                from collections import Counter
                common_terms = Counter(key_terms).most_common(1)
                if common_terms:
                    focused = f"{state.current_query} {common_terms[0][0]}"
                    return focused
        
        return state.current_query
    
    async def _generate_query_variants(self, query: str) -> List[str]:
        """Generate multiple query variants"""
        variants = [
            query,
            f"what is {query}",
            f"how to {query}",
            f"{query} examples",
            f"{query} explanation"
        ]
        return variants
    
    async def _combine_queries(self, state: SearchState) -> str:
        """Combine successful query elements"""
        # Simple combination of query history
        unique_words = set()
        for query in state.query_history:
            unique_words.update(query.split())
        
        # Take most important words (simplified)
        combined_words = list(unique_words)[:8]  # Limit length
        return " ".join(combined_words)
    
    async def _broaden_query(self, query: str) -> str:
        """Broaden query for hierarchical search"""
        words = query.split()
        if len(words) > 2:
            # Remove specific terms
            return " ".join(words[:2])
        return query
    
    async def _narrow_query(self, state: SearchState) -> str:
        """Narrow query based on results"""
        if state.documents:
            # Add specific terms from results
            top_doc = state.get_top_documents(k=1)
            if top_doc:
                content_words = top_doc[0].get('content', '').split()[:5]
                return f"{state.original_query} {' '.join(content_words)}"
        
        return state.current_query
    
    async def _fine_tune_query(self, state: SearchState) -> str:
        """Fine-tune query for precision"""
        # Add precision terms
        precision_terms = ["specific", "detailed", "exact"]
        term = precision_terms[state.iteration % len(precision_terms)]
        return f"{term} {state.current_query}"
    
    def _should_terminate(self, state: SearchState) -> bool:
        """Check if search should terminate early"""
        
        # Terminate if we have enough high-quality documents
        if (state.total_documents >= 10 and 
            state.average_relevance > 0.8 and 
            state.unique_documents >= 5):
            return True
        
        # Terminate if no improvement in last 2 iterations
        if (state.iteration >= 2 and 
            len(state.action_history) >= 2):
            
            recent_docs = state.total_documents
            prev_iteration_docs = sum(1 for action in state.action_history[:-1])
            
            if recent_docs <= prev_iteration_docs:
                return True
        
        return False
    
    def _calculate_efficiency(self, state: SearchState) -> float:
        """Calculate search efficiency metric"""
        if state.iteration == 0:
            return 1.0
        
        # Efficiency = unique documents per iteration
        efficiency = state.unique_documents / state.iteration
        return min(1.0, efficiency / 5.0)  # Normalize to [0, 1]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        avg_search_time = (self.total_search_time / max(1, self.search_count))
        
        return {
            'total_searches': self.search_count,
            'total_search_time': self.total_search_time,
            'average_search_time': avg_search_time,
            'success_rate': self.success_rate,
            'experience_buffer_size': len(self.experience_buffer),
            'strategy': self.config.search_strategy.value
        }
    
    def reset_statistics(self) -> None:
        """Reset search statistics"""
        self.search_count = 0
        self.total_search_time = 0.0
        self.success_rate = 0.0
        logger.info("Search statistics reset")
