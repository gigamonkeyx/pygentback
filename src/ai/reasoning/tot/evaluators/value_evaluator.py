"""
Value Evaluator for Tree of Thoughts

Implements scalar value assessment of thought states for the ToT framework.
This evaluator assigns numerical scores to thoughts to guide search decisions.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import re

from ..core.thought import Thought
from ..core.state import ReasoningState
from ..models import ToTConfig

logger = logging.getLogger(__name__)


class ValueEvaluator:
    """
    Scalar value evaluator for thought states
    
    Evaluates thoughts independently and assigns numerical scores
    based on various criteria including:
    - Logical consistency
    - Progress toward solution
    - Code quality (for coding tasks)
    - Completeness and clarity
    """
    
    def __init__(self, config: ToTConfig, llm_backend=None):
        self.config = config
        self.llm_backend = llm_backend
        self.evaluation_count = 0
        
        # Evaluation criteria weights
        self.criteria_weights = {
            'logical_consistency': 0.25,
            'solution_progress': 0.30,
            'implementation_quality': 0.20,
            'clarity_completeness': 0.15,
            'innovation_creativity': 0.10
        }
    
    async def evaluate_thoughts(
        self,
        thoughts: List[Thought],
        current_state: ReasoningState,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Thought, float]]:
        """
        Evaluate multiple thoughts and return them with scores
        
        Args:
            thoughts: List of thoughts to evaluate
            current_state: Current reasoning state
            context: Additional evaluation context
            
        Returns:
            List of (thought, score) tuples sorted by score descending
        """
        if context is None:
            context = {}
            
        logger.info(f"Evaluating {len(thoughts)} thoughts")
        
        scored_thoughts = []
        
        for thought in thoughts:
            score = await self.evaluate_single_thought(
                thought, current_state, context
            )
            scored_thoughts.append((thought, score))
            
        # Sort by score descending
        scored_thoughts.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Evaluation complete. Scores: {[score for _, score in scored_thoughts]}")
        return scored_thoughts
    
    async def evaluate_single_thought(
        self,
        thought: Thought,
        current_state: ReasoningState,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate a single thought and return its score
        
        Args:
            thought: Thought to evaluate
            current_state: Current reasoning state
            context: Additional evaluation context
            
        Returns:
            Numerical score (0.0 to 1.0)
        """
        if context is None:
            context = {}
            
        self.evaluation_count += 1
        
        try:
            # Use LLM-based evaluation if available, otherwise heuristic
            if self.llm_backend and context.get('use_llm_evaluation', True):
                score = await self._llm_evaluate_thought(thought, current_state, context)
            else:
                score = self._heuristic_evaluate_thought(thought, current_state, context)
            
            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
            
            # Update thought metrics
            thought.metrics.value_score = score
            thought.metrics.evaluation_count += 1
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating thought: {e}")
            return 0.0
    
    async def _llm_evaluate_thought(
        self,
        thought: Thought,
        current_state: ReasoningState,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate thought using LLM"""
        try:
            prompt = self._build_evaluation_prompt(thought, current_state, context)
            
            response = await self.llm_backend.generate(
                prompt=prompt,
                temperature=self.config.evaluation_temperature,
                max_tokens=200
            )
            
            # Extract numerical score from response
            score = self._extract_score_from_response(response)
            
            # Store LLM reasoning
            thought.metadata['llm_evaluation'] = response
            
            return score
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fall back to heuristic evaluation
            return self._heuristic_evaluate_thought(thought, current_state, context)
    
    def _heuristic_evaluate_thought(
        self,
        thought: Thought,
        current_state: ReasoningState,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate thought using heuristic methods"""
        
        scores = {}
        
        # Logical consistency (25%)
        scores['logical_consistency'] = self._evaluate_logical_consistency(thought, context)
        
        # Solution progress (30%)
        scores['solution_progress'] = self._evaluate_solution_progress(thought, current_state, context)
        
        # Implementation quality (20%)
        scores['implementation_quality'] = self._evaluate_implementation_quality(thought, context)
        
        # Clarity and completeness (15%)
        scores['clarity_completeness'] = self._evaluate_clarity_completeness(thought, context)
        
        # Innovation and creativity (10%)
        scores['innovation_creativity'] = self._evaluate_innovation_creativity(thought, context)
        
        # Calculate weighted score
        total_score = sum(
            scores[criterion] * weight
            for criterion, weight in self.criteria_weights.items()
        )
        
        # Store individual scores in metadata
        thought.metadata['evaluation_scores'] = scores
        thought.metadata['evaluation_method'] = 'heuristic'
        
        return total_score
    
    def _evaluate_logical_consistency(self, thought: Thought, context: Dict[str, Any]) -> float:
        """Evaluate logical consistency of the thought"""
        content = thought.content.lower()
        
        # Check for logical connectors and reasoning patterns
        logical_indicators = [
            'therefore', 'because', 'since', 'if', 'then', 'however',
            'moreover', 'furthermore', 'consequently', 'hence'
        ]
        
        logical_score = sum(1 for indicator in logical_indicators if indicator in content)
        logical_score = min(logical_score / 3, 1.0)  # Normalize to 0-1
        
        # Check for contradictions or inconsistencies
        contradiction_patterns = [
            r'but.*not.*but.*is',
            r'never.*always',
            r'impossible.*possible',
            r'cannot.*can.*same'
        ]
        
        contradiction_penalty = 0
        for pattern in contradiction_patterns:
            if re.search(pattern, content):
                contradiction_penalty += 0.3
        
        return max(0.0, logical_score - contradiction_penalty)
    
    def _evaluate_solution_progress(
        self,
        thought: Thought,
        current_state: ReasoningState,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate how much progress the thought makes toward solution"""
        content = thought.content.lower()
        
        # Progress indicators
        progress_indicators = [
            'implement', 'create', 'build', 'develop', 'solve',
            'algorithm', 'function', 'method', 'approach',
            'solution', 'answer', 'result', 'complete'
        ]
        
        progress_score = sum(1 for indicator in progress_indicators if indicator in content)
        progress_score = min(progress_score / 5, 1.0)
        
        # Bonus for concrete steps
        concrete_indicators = [
            'step 1', 'first', 'next', 'then', 'finally',
            'initialize', 'iterate', 'return', 'output'
        ]
        
        concrete_score = sum(1 for indicator in concrete_indicators if indicator in content)
        concrete_score = min(concrete_score / 3, 0.3)
        
        # Depth bonus - deeper thoughts that build on previous reasoning
        depth_bonus = min(thought.depth * 0.1, 0.2)
        
        return min(1.0, progress_score + concrete_score + depth_bonus)
    
    def _evaluate_implementation_quality(self, thought: Thought, context: Dict[str, Any]) -> float:
        """Evaluate implementation quality for coding tasks"""
        if context.get('task_type') != 'coding':
            return 0.7  # Neutral score for non-coding tasks
        
        content = thought.content.lower()
        
        # Code quality indicators
        quality_indicators = [
            'function', 'class', 'method', 'variable', 'parameter',
            'return', 'import', 'def', 'if', 'for', 'while',
            'try', 'except', 'with', 'assert'
        ]
        
        code_score = sum(1 for indicator in quality_indicators if indicator in content)
        code_score = min(code_score / 8, 1.0)
        
        # Best practices indicators
        best_practices = [
            'test', 'validate', 'check', 'error handling',
            'documentation', 'comment', 'readable', 'efficient',
            'optimize', 'clean', 'maintainable'
        ]
        
        practices_score = sum(1 for practice in best_practices if practice in content)
        practices_score = min(practices_score / 4, 0.3)
        
        return min(1.0, code_score + practices_score)
    
    def _evaluate_clarity_completeness(self, thought: Thought, context: Dict[str, Any]) -> float:
        """Evaluate clarity and completeness of the thought"""
        content = thought.content
        
        # Length-based completeness (reasonable length indicates thoroughness)
        length_score = min(len(content.split()) / 50, 1.0)
        
        # Structure indicators
        structure_indicators = [':', '.', ',', ';', '-', '1.', '2.', '3.']
        structure_score = sum(1 for indicator in structure_indicators if indicator in content)
        structure_score = min(structure_score / 10, 0.3)
        
        # Clarity indicators
        clarity_words = [
            'clear', 'simple', 'straightforward', 'obvious',
            'specifically', 'exactly', 'precisely', 'namely'
        ]
        
        clarity_score = sum(1 for word in clarity_words if word in content.lower())
        clarity_score = min(clarity_score / 3, 0.2)
        
        return min(1.0, length_score + structure_score + clarity_score)
    
    def _evaluate_innovation_creativity(self, thought: Thought, context: Dict[str, Any]) -> float:
        """Evaluate innovation and creativity of the thought"""
        content = thought.content.lower()
        
        # Innovation indicators
        innovation_words = [
            'novel', 'new', 'innovative', 'creative', 'unique',
            'different', 'alternative', 'unconventional', 'original',
            'breakthrough', 'improvement', 'optimization'
        ]
        
        innovation_score = sum(1 for word in innovation_words if word in content)
        innovation_score = min(innovation_score / 4, 1.0)
        
        # Penalty for very common/generic thoughts
        generic_patterns = [
            'use a loop', 'create a function', 'check if',
            'simple approach', 'basic implementation'
        ]
        
        generic_penalty = sum(0.2 for pattern in generic_patterns if pattern in content)
        
        return max(0.0, innovation_score - generic_penalty)
    
    def _build_evaluation_prompt(
        self,
        thought: Thought,
        current_state: ReasoningState,
        context: Dict[str, Any]
    ) -> str:
        """Build evaluation prompt for LLM"""
        
        prompt_parts = [
            "# Thought Evaluation",
            f"Problem: {current_state.problem}",
            f"Thought to evaluate: {thought.content}",
            "",
            "Evaluate this thought on the following criteria (each 0.0-1.0):",
            "1. Logical consistency and reasoning quality",
            "2. Progress toward solving the problem",
            "3. Implementation quality (for coding tasks)",
            "4. Clarity and completeness",
            "5. Innovation and creativity",
            "",
            "Provide a numerical score from 0.0 to 1.0 where:",
            "- 0.0-0.3: Poor quality, major issues",
            "- 0.4-0.6: Average quality, some issues",
            "- 0.7-0.8: Good quality, minor issues",
            "- 0.9-1.0: Excellent quality, highly valuable",
            "",
            "Score: "
        ]
        
        return "\\n".join(prompt_parts)
    
    def _extract_score_from_response(self, response: str) -> float:
        """Extract numerical score from LLM response"""
        # Look for decimal numbers in the response
        import re
        
        # Pattern to match decimal numbers
        pattern = r'([0-1]?\\.?\\d+)'
        matches = re.findall(pattern, response)
        
        if matches:
            try:
                score = float(matches[0])
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        # Fallback: look for qualitative assessments
        response_lower = response.lower()
        if any(word in response_lower for word in ['excellent', 'outstanding', 'perfect']):
            return 0.9
        elif any(word in response_lower for word in ['good', 'solid', 'strong']):
            return 0.7
        elif any(word in response_lower for word in ['average', 'okay', 'decent']):
            return 0.5
        elif any(word in response_lower for word in ['poor', 'weak', 'bad']):
            return 0.3
        else:
            return 0.5  # Default neutral score
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            'total_evaluations': self.evaluation_count,
            'evaluation_method': 'value',
            'criteria_weights': self.criteria_weights,
            'config': {
                'evaluation_temperature': self.config.evaluation_temperature,
                'value_threshold': self.config.value_threshold
            }
        }
