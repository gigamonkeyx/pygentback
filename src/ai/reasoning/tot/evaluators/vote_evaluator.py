"""
Vote Evaluator for Tree of Thoughts

Implements comparative voting system for thought evaluation in the ToT framework.
This evaluator compares thoughts against each other and determines relative rankings.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import itertools
import asyncio

from ..core.thought import Thought
from ..core.state import ReasoningState
from ..models import ToTConfig

logger = logging.getLogger(__name__)


class VoteEvaluator:
    """
    Comparative voting evaluator for thought states
    
    Evaluates thoughts by comparing them against each other rather than
    independently. Suitable for creative tasks where relative quality
    matters more than absolute scores.
    """
    
    def __init__(self, config: ToTConfig, llm_backend=None):
        self.config = config
        self.llm_backend = llm_backend
        self.evaluation_count = 0
        self.vote_cache = {}  # Cache for pairwise comparisons
        
    async def evaluate_thoughts(
        self,
        thoughts: List[Thought],
        current_state: ReasoningState,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Thought, float]]:
        """
        Evaluate thoughts using comparative voting
        
        Args:
            thoughts: List of thoughts to evaluate
            current_state: Current reasoning state
            context: Additional evaluation context
            
        Returns:
            List of (thought, score) tuples sorted by score descending
        """
        if context is None:
            context = {}
            
        if len(thoughts) == 0:
            return []
        
        if len(thoughts) == 1:
            # Single thought gets neutral score
            return [(thoughts[0], 0.5)]
            
        logger.info(f"Evaluating {len(thoughts)} thoughts using voting")
        
        # Perform pairwise comparisons
        vote_results = await self._conduct_pairwise_voting(
            thoughts, current_state, context
        )
        
        # Calculate final rankings
        ranked_thoughts = self._calculate_rankings(thoughts, vote_results)
        
        logger.info(f"Voting complete. Rankings: {[score for _, score in ranked_thoughts]}")
        return ranked_thoughts
    
    async def _conduct_pairwise_voting(
        self,
        thoughts: List[Thought],
        current_state: ReasoningState,
        context: Dict[str, Any]
    ) -> Dict[Tuple[int, int], int]:
        """
        Conduct pairwise voting between all thought combinations
        
        Returns:
            Dictionary mapping (thought_a_idx, thought_b_idx) -> winner_idx
        """
        vote_results = {}
        
        # Generate all pairwise combinations
        pairs = list(itertools.combinations(range(len(thoughts)), 2))
        
        logger.info(f"Conducting {len(pairs)} pairwise comparisons")
        
        # Perform comparisons (can be parallelized)
        comparison_tasks = []
        for i, j in pairs:
            task = self._compare_thoughts_pair(
                thoughts[i], thoughts[j], i, j, current_state, context
            )
            comparison_tasks.append(task)
        
        # Execute comparisons
        if len(comparison_tasks) <= 5:
            # Run in parallel for small numbers
            results = await asyncio.gather(*comparison_tasks, return_exceptions=True)
        else:
            # Run sequentially for large numbers to avoid overwhelming LLM
            results = []
            for task in comparison_tasks:
                result = await task
                results.append(result)
        
        # Process results
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Comparison {idx} failed: {result}")
                continue
                
            pair_idx, winner_idx = result
            vote_results[pair_idx] = winner_idx
        
        return vote_results
    
    async def _compare_thoughts_pair(
        self,
        thought_a: Thought,
        thought_b: Thought,
        idx_a: int,
        idx_b: int,
        current_state: ReasoningState,
        context: Dict[str, Any]
    ) -> Tuple[Tuple[int, int], int]:
        """
        Compare two thoughts and determine winner
        
        Returns:
            ((idx_a, idx_b), winner_idx)
        """
        self.evaluation_count += 1
        
        # Check cache first
        cache_key = (thought_a.id, thought_b.id)
        if cache_key in self.vote_cache:
            cached_result = self.vote_cache[cache_key]
            return ((idx_a, idx_b), cached_result)
        
        try:
            if self.llm_backend:
                winner_idx = await self._llm_compare_thoughts(
                    thought_a, thought_b, idx_a, idx_b, current_state, context
                )
            else:
                winner_idx = self._heuristic_compare_thoughts(
                    thought_a, thought_b, idx_a, idx_b, context
                )
            
            # Cache result
            self.vote_cache[cache_key] = winner_idx
            self.vote_cache[(thought_b.id, thought_a.id)] = idx_b if winner_idx == idx_a else idx_a
            
            return ((idx_a, idx_b), winner_idx)
            
        except Exception as e:
            logger.error(f"Error comparing thoughts {idx_a} vs {idx_b}: {e}")
            # Default to first thought in case of error
            return ((idx_a, idx_b), idx_a)
    
    async def _llm_compare_thoughts(
        self,
        thought_a: Thought,
        thought_b: Thought,
        idx_a: int,
        idx_b: int,
        current_state: ReasoningState,
        context: Dict[str, Any]
    ) -> int:
        """Compare thoughts using LLM"""
        prompt = self._build_comparison_prompt(
            thought_a, thought_b, current_state, context
        )
        
        response = await self.llm_backend.generate(
            prompt=prompt,
            temperature=self.config.evaluation_temperature,
            max_tokens=150
        )
        
        # Extract winner from response
        winner_idx = self._extract_winner_from_response(response, idx_a, idx_b)
        
        # Store comparison reasoning
        comparison_key = f"comparison_{thought_a.id}_{thought_b.id}"
        thought_a.metadata[comparison_key] = {
            'competitor': thought_b.id,
            'winner': winner_idx,
            'reasoning': response
        }
        thought_b.metadata[comparison_key] = {
            'competitor': thought_a.id,
            'winner': winner_idx,
            'reasoning': response
        }
        
        return winner_idx
    
    def _heuristic_compare_thoughts(
        self,
        thought_a: Thought,
        thought_b: Thought,
        idx_a: int,
        idx_b: int,
        context: Dict[str, Any]
    ) -> int:
        """Compare thoughts using heuristic methods"""
        
        # Calculate simple comparison metrics
        score_a = self._calculate_simple_score(thought_a, context)
        score_b = self._calculate_simple_score(thought_b, context)
        
        # Add metadata about the comparison
        comparison_key = f"heuristic_comparison_{thought_a.id}_{thought_b.id}"
        thought_a.metadata[comparison_key] = {
            'competitor': thought_b.id,
            'score_a': score_a,
            'score_b': score_b,
            'winner': idx_a if score_a > score_b else idx_b
        }
        
        return idx_a if score_a > score_b else idx_b
    
    def _calculate_simple_score(self, thought: Thought, context: Dict[str, Any]) -> float:
        """Calculate simple heuristic score for comparison"""
        content = thought.content.lower()
        
        # Length score (moderate length preferred)
        word_count = len(content.split())
        length_score = 1.0 - abs(word_count - 30) / 50  # Optimal around 30 words
        length_score = max(0.0, length_score)
        
        # Specificity score
        specific_words = [
            'specifically', 'exactly', 'precisely', 'implement',
            'algorithm', 'function', 'method', 'approach',
            'step', 'process', 'technique', 'strategy'
        ]
        specificity_score = sum(1 for word in specific_words if word in content)
        specificity_score = min(specificity_score / 5, 1.0)
        
        # Action orientation score
        action_words = [
            'create', 'build', 'implement', 'develop', 'design',
            'write', 'construct', 'establish', 'generate', 'produce'
        ]
        action_score = sum(1 for word in action_words if word in content)
        action_score = min(action_score / 3, 1.0)
        
        # Depth bonus based on thought depth
        depth_bonus = min(thought.depth * 0.1, 0.3)
        
        return (length_score * 0.3 + specificity_score * 0.4 + 
                action_score * 0.2 + depth_bonus * 0.1)
    
    def _calculate_rankings(
        self,
        thoughts: List[Thought],
        vote_results: Dict[Tuple[int, int], int]
    ) -> List[Tuple[Thought, float]]:
        """
        Calculate final rankings based on pairwise voting results
        
        Uses a simple win-loss record to determine relative rankings
        """
        n_thoughts = len(thoughts)
        
        # Count wins for each thought
        win_counts = [0] * n_thoughts
        total_comparisons = [0] * n_thoughts
        
        for (i, j), winner in vote_results.items():
            total_comparisons[i] += 1
            total_comparisons[j] += 1
            
            if winner == i:
                win_counts[i] += 1
            else:
                win_counts[j] += 1
        
        # Calculate win percentages
        win_percentages = []
        for i in range(n_thoughts):
            if total_comparisons[i] > 0:
                percentage = win_counts[i] / total_comparisons[i]
            else:
                percentage = 0.5  # Neutral for no comparisons
            win_percentages.append(percentage)
        
        # Create ranked list
        ranked_thoughts = []
        for i, thought in enumerate(thoughts):
            score = win_percentages[i]
            
            # Update thought metrics
            thought.metrics.vote_score = score
            thought.metrics.evaluation_count += total_comparisons[i]
            thought.metadata['vote_evaluation'] = {
                'wins': win_counts[i],
                'total_comparisons': total_comparisons[i],
                'win_percentage': score
            }
            
            ranked_thoughts.append((thought, score))
        
        # Sort by score descending
        ranked_thoughts.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_thoughts
    
    def _build_comparison_prompt(
        self,
        thought_a: Thought,
        thought_b: Thought,
        current_state: ReasoningState,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for pairwise thought comparison"""
        
        prompt_parts = [
            "# Thought Comparison",
            f"Problem: {current_state.problem}",
            "",
            "Compare the following two approaches and determine which is better:",
            "",
            "Approach A:",
            thought_a.content,
            "",
            "Approach B:",
            thought_b.content,
            "",
            "Evaluation criteria:",
            "- Quality of reasoning and logic",
            "- Progress toward solving the problem",
            "- Clarity and actionability",
            "- Innovation and creativity",
            "",
            "Which approach is better? Respond with 'A' or 'B' and briefly explain why.",
            "",
            "Winner: "
        ]
        
        return "\\n".join(prompt_parts)
    
    def _extract_winner_from_response(self, response: str, idx_a: int, idx_b: int) -> int:
        """Extract winner from LLM comparison response"""
        response_upper = response.upper()
        
        # Look for explicit A or B choice
        if 'A' in response_upper and 'B' not in response_upper:
            return idx_a
        elif 'B' in response_upper and 'A' not in response_upper:
            return idx_b
        elif response_upper.count('A') > response_upper.count('B'):
            return idx_a
        elif response_upper.count('B') > response_upper.count('A'):
            return idx_b
        
        # Look for qualitative indicators
        if any(word in response.lower() for word in ['first', 'approach a', 'option a']):
            return idx_a
        elif any(word in response.lower() for word in ['second', 'approach b', 'option b']):
            return idx_b
        
        # Default to first thought if unclear
        logger.warning(f"Unclear comparison result: {response}")
        return idx_a
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            'total_comparisons': self.evaluation_count,
            'evaluation_method': 'vote',
            'cache_size': len(self.vote_cache),
            'config': {
                'evaluation_temperature': self.config.evaluation_temperature,
                'n_evaluate_sample': self.config.n_evaluate_sample
            }
        }
    
    def clear_cache(self):
        """Clear the vote cache"""
        self.vote_cache.clear()
        logger.info("Vote cache cleared")
