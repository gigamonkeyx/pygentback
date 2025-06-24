"""
Coding-Specific Evaluator for Tree of Thoughts

Specialized evaluator for software development tasks that understands
code quality, algorithmic complexity, and implementation considerations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import re
import ast

from ..core.thought import Thought
from ..core.state import ReasoningState
from ..models import ToTConfig

logger = logging.getLogger(__name__)


class CodingEvaluator:
    """
    Specialized evaluator for coding and software development tasks
    
    Evaluates thoughts based on:
    - Code quality and best practices
    - Algorithmic correctness and efficiency
    - Implementation feasibility
    - Testing and maintainability considerations
    """
    
    def __init__(self, config: ToTConfig, llm_backend=None):
        self.config = config
        self.llm_backend = llm_backend
        self.evaluation_count = 0
        
        # Coding-specific evaluation weights
        self.coding_weights = {
            'algorithmic_correctness': 0.30,
            'implementation_quality': 0.25,
            'efficiency_complexity': 0.20,
            'maintainability': 0.15,
            'testability': 0.10
        }
        
        # Code patterns for analysis
        self.code_patterns = {
            'functions': r'def\s+\w+\s*\(',
            'classes': r'class\s+\w+',
            'imports': r'import\s+\w+|from\s+\w+\s+import',
            'loops': r'for\s+\w+\s+in|while\s+\w+',
            'conditionals': r'if\s+\w+|elif\s+\w+|else:',
            'error_handling': r'try:|except\s+\w+:|finally:',
            'comments': r'#.*$|""".*?"""|\'\'\'.*?\'\'\'',
            'type_hints': r':\s*\w+\s*=|:\s*\w+\s*->'
        }
    
    async def evaluate_thoughts(
        self,
        thoughts: List[Thought],
        current_state: ReasoningState,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Thought, float]]:
        """
        Evaluate coding thoughts with specialized metrics
        
        Args:
            thoughts: List of thoughts to evaluate
            current_state: Current reasoning state
            context: Additional evaluation context
            
        Returns:
            List of (thought, score) tuples sorted by score descending
        """
        if context is None:
            context = {}
            
        logger.info(f"Evaluating {len(thoughts)} coding thoughts")
        
        scored_thoughts = []
        
        for thought in thoughts:
            score = await self.evaluate_coding_thought(
                thought, current_state, context
            )
            scored_thoughts.append((thought, score))
            
        # Sort by score descending
        scored_thoughts.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Coding evaluation complete. Scores: {[score for _, score in scored_thoughts]}")
        return scored_thoughts
    
    async def evaluate_coding_thought(
        self,
        thought: Thought,
        current_state: ReasoningState,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate a single coding thought
        
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
            # Analyze thought content for coding elements
            code_analysis = self._analyze_code_content(thought.content)
            
            # Calculate component scores
            scores = {}
            
            # Algorithmic correctness (30%)
            scores['algorithmic_correctness'] = self._evaluate_algorithmic_correctness(
                thought, code_analysis, context
            )
            
            # Implementation quality (25%)
            scores['implementation_quality'] = self._evaluate_implementation_quality(
                thought, code_analysis, context
            )
            
            # Efficiency and complexity (20%)
            scores['efficiency_complexity'] = self._evaluate_efficiency_complexity(
                thought, code_analysis, context
            )
            
            # Maintainability (15%)
            scores['maintainability'] = self._evaluate_maintainability(
                thought, code_analysis, context
            )
            
            # Testability (10%)
            scores['testability'] = self._evaluate_testability(
                thought, code_analysis, context
            )
            
            # Calculate weighted score
            total_score = sum(
                scores[criterion] * weight
                for criterion, weight in self.coding_weights.items()
            )
            
            # Ensure score is in valid range
            total_score = max(0.0, min(1.0, total_score))
            
            # Store detailed scores in metadata
            thought.metadata['coding_evaluation'] = {
                'component_scores': scores,
                'code_analysis': code_analysis,
                'total_score': total_score,
                'evaluation_method': 'coding_specialized'
            }
            
            # Update thought metrics
            thought.metrics.value_score = total_score
            thought.metrics.evaluation_count += 1
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error in coding evaluation: {e}")
            return 0.0
    
    def _analyze_code_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for coding elements and patterns"""
        analysis = {
            'has_code': False,
            'code_blocks': 0,
            'patterns': {},
            'complexity_indicators': [],
            'python_valid': False,
            'word_count': len(content.split()),
            'line_count': len(content.split('\\n'))
        }
        
        # Count code patterns
        for pattern_name, pattern in self.code_patterns.items():
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            analysis['patterns'][pattern_name] = len(matches)
            if matches:
                analysis['has_code'] = True
        
        # Count code blocks (markdown style)
        code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
        analysis['code_blocks'] = len(code_blocks)
        if code_blocks:
            analysis['has_code'] = True
        
        # Check for complexity indicators
        complexity_keywords = [
            'algorithm', 'complexity', 'big o', 'o(n)', 'o(log n)',
            'recursive', 'dynamic programming', 'optimization',
            'data structure', 'hash table', 'binary tree'
        ]
        
        content_lower = content.lower()
        for keyword in complexity_keywords:
            if keyword in content_lower:
                analysis['complexity_indicators'].append(keyword)
        
        # Try to validate Python syntax if code blocks present
        if code_blocks:
            for block in code_blocks:
                # Extract code from markdown block
                code_lines = block.split('\\n')[1:-1]  # Remove ``` lines
                if code_lines and code_lines[0].strip().startswith('python'):
                    code_lines = code_lines[1:]  # Remove language specifier
                
                code_text = '\\n'.join(code_lines)
                if self._is_valid_python(code_text):
                    analysis['python_valid'] = True
                    break
        
        return analysis
    
    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def _evaluate_algorithmic_correctness(
        self,
        thought: Thought,
        code_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Evaluate algorithmic correctness and logical flow"""
        content = thought.content.lower()
        score = 0.0
        
        # Logical flow indicators
        flow_indicators = [
            'step', 'first', 'then', 'next', 'finally',
            'algorithm', 'approach', 'method', 'procedure'
        ]
        
        flow_score = sum(1 for indicator in flow_indicators if indicator in content)
        flow_score = min(flow_score / 5, 0.4)
        
        # Problem-solving indicators
        problem_solving = [
            'solve', 'solution', 'answer', 'result',
            'implement', 'execute', 'process', 'handle'
        ]
        
        problem_score = sum(1 for indicator in problem_solving if indicator in content)
        problem_score = min(problem_score / 4, 0.3)
        
        # Complexity awareness
        complexity_score = min(len(code_analysis['complexity_indicators']) / 3, 0.3)
        
        # Bonus for valid Python code
        if code_analysis['python_valid']:
            score += 0.2
        
        return min(1.0, flow_score + problem_score + complexity_score + score)
    
    def _evaluate_implementation_quality(
        self,
        thought: Thought,
        code_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Evaluate implementation quality and code structure"""
        score = 0.0
        
        # Code structure indicators
        if code_analysis['patterns']['functions'] > 0:
            score += 0.2
        if code_analysis['patterns']['classes'] > 0:
            score += 0.15
        if code_analysis['patterns']['imports'] > 0:
            score += 0.1
        
        # Error handling
        if code_analysis['patterns']['error_handling'] > 0:
            score += 0.15
        
        # Type hints (modern Python practice)
        if code_analysis['patterns']['type_hints'] > 0:
            score += 0.1
        
        # Comments and documentation
        if code_analysis['patterns']['comments'] > 0:
            score += 0.1
        
        # Code blocks presence
        if code_analysis['code_blocks'] > 0:
            score += 0.2
        
        return min(1.0, score)
    
    def _evaluate_efficiency_complexity(
        self,
        thought: Thought,
        code_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Evaluate efficiency and complexity considerations"""
        content = thought.content.lower()
        
        # Efficiency keywords
        efficiency_keywords = [
            'efficient', 'optimize', 'performance', 'speed',
            'memory', 'space', 'time complexity', 'big o'
        ]
        
        efficiency_score = sum(1 for keyword in efficiency_keywords if keyword in content)
        efficiency_score = min(efficiency_score / 4, 0.5)
        
        # Data structure awareness
        data_structures = [
            'array', 'list', 'dictionary', 'hash', 'tree',
            'graph', 'stack', 'queue', 'heap', 'set'
        ]
        
        ds_score = sum(1 for ds in data_structures if ds in content)
        ds_score = min(ds_score / 3, 0.3)
        
        # Algorithm awareness
        algorithms = [
            'sort', 'search', 'binary search', 'dfs', 'bfs',
            'dynamic programming', 'greedy', 'divide and conquer'
        ]
        
        algo_score = sum(1 for algo in algorithms if algo in content)
        algo_score = min(algo_score / 3, 0.2)
        
        return min(1.0, efficiency_score + ds_score + algo_score)
    
    def _evaluate_maintainability(
        self,
        thought: Thought,
        code_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Evaluate maintainability and code quality aspects"""
        content = thought.content.lower()
        
        # Maintainability keywords
        maintainability_keywords = [
            'readable', 'clean', 'maintainable', 'modular',
            'reusable', 'documentation', 'comment', 'refactor'
        ]
        
        maint_score = sum(1 for keyword in maintainability_keywords if keyword in content)
        maint_score = min(maint_score / 4, 0.4)
        
        # Best practices indicators
        best_practices = [
            'naming convention', 'single responsibility', 'dry',
            'solid principles', 'code review', 'style guide'
        ]
        
        practices_score = sum(1 for practice in best_practices if practice in content)
        practices_score = min(practices_score / 3, 0.3)
        
        # Documentation indicators
        if code_analysis['patterns']['comments'] > 0:
            maint_score += 0.2
        
        # Reasonable length (not too short, not too verbose)
        length_score = 1.0 - abs(code_analysis['word_count'] - 40) / 60
        length_score = max(0.0, min(length_score, 0.1))
        
        return min(1.0, maint_score + practices_score + length_score)
    
    def _evaluate_testability(
        self,
        thought: Thought,
        code_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Evaluate testability considerations"""
        content = thought.content.lower()
        
        # Testing keywords
        testing_keywords = [
            'test', 'unit test', 'integration test', 'pytest',
            'assert', 'mock', 'validate', 'verify', 'check'
        ]
        
        test_score = sum(1 for keyword in testing_keywords if keyword in content)
        test_score = min(test_score / 3, 0.5)
        
        # Testable design indicators
        design_indicators = [
            'pure function', 'side effect', 'dependency injection',
            'interface', 'abstract', 'testable', 'isolate'
        ]
        
        design_score = sum(1 for indicator in design_indicators if indicator in content)
        design_score = min(design_score / 3, 0.3)
        
        # Edge case consideration
        edge_cases = [
            'edge case', 'boundary', 'null', 'empty', 'exception',
            'error case', 'validation', 'input validation'
        ]
        
        edge_score = sum(1 for case in edge_cases if case in content)
        edge_score = min(edge_score / 3, 0.2)
        
        return min(1.0, test_score + design_score + edge_score)
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get coding evaluation statistics"""
        return {
            'total_evaluations': self.evaluation_count,
            'evaluation_method': 'coding_specialized',
            'coding_weights': self.coding_weights,
            'code_patterns': list(self.code_patterns.keys()),
            'config': {
                'evaluation_temperature': self.config.evaluation_temperature,
                'task_type': 'coding'
            }
        }
