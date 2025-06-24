"""
General Problem Solving Task for Tree of Thought

Specialized ToT implementation for general problem solving
using multi-path reasoning to explore different solution approaches.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from ..tot_engine import ToTEngine

logger = logging.getLogger(__name__)


@dataclass
class ProblemSolvingConfig:
    """Configuration specific to problem solving"""
    problem_type: str = "general"  # "mathematical", "logical", "creative", "analytical"
    solution_requirements: List[str] = None
    constraints: List[str] = None
    evaluation_criteria: List[str] = None
    allow_multiple_solutions: bool = True


class ProblemSolvingTask:
    """
    Tree of Thought implementation for general problem solving
    
    Uses multi-path reasoning to explore different solution approaches,
    evaluate partial solutions, and find optimal problem-solving paths.
    """
    
    def __init__(self, config: Optional[ToTConfig] = None,
                 problem_config: Optional[ProblemSolvingConfig] = None):
        # Default ToT configuration for problem solving
        if config is None:
            config = ToTConfig(
                generation_strategy=GenerationStrategy.PROPOSE,
                evaluation_method=EvaluationMethod.VALUE,
                search_method=SearchMethod.BFS,
                n_generate_sample=3,
                n_evaluate_sample=2,
                n_select_sample=3,
                max_depth=10,
                temperature=0.7
            )
        
        self.config = config
        self.problem_config = problem_config or ProblemSolvingConfig()
        
        self.tot_engine = ToTEngine(config)
    
    async def solve_problem(self, problem_statement: str,
                          problem_type: Optional[str] = None,
                          constraints: Optional[List[str]] = None,
                          hints: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Solve a problem using Tree of Thought reasoning
        
        Args:
            problem_statement: The problem to solve
            problem_type: Type of problem (mathematical, logical, etc.)
            constraints: Problem constraints
            hints: Optional hints or guidance
            
        Returns:
            Dictionary containing solution results and reasoning paths
        """
        # Update problem configuration
        if problem_type:
            self.problem_config.problem_type = problem_type
        if constraints:
            self.problem_config.constraints = constraints
        
        # Prepare task context
        task_context = self._create_problem_context(
            problem_statement, problem_type, constraints, hints
        )
        
        # Execute ToT reasoning
        result = await self.tot_engine.solve(problem_statement, task_context)
        
        # Process and format results
        return self._process_problem_results(result, problem_statement)
    
    async def solve_mathematical_problem(self, problem: str,
                                       show_work: bool = True) -> Dict[str, Any]:
        """
        Solve a mathematical problem with step-by-step reasoning
        
        Args:
            problem: Mathematical problem statement
            show_work: Whether to show detailed work
            
        Returns:
            Mathematical solution with reasoning steps
        """
        # Configure for mathematical problem solving
        math_config = ProblemSolvingConfig(
            problem_type="mathematical",
            solution_requirements=["correct answer", "clear steps"],
            evaluation_criteria=["mathematical accuracy", "logical progression"]
        )
        
        # Create mathematical context
        task_context = self._create_mathematical_context(problem, show_work)
        
        # Execute reasoning
        result = await self.tot_engine.solve(problem, task_context)
        
        return self._process_mathematical_results(result, problem)
    
    async def solve_logical_puzzle(self, puzzle: str,
                                 puzzle_rules: List[str] = None) -> Dict[str, Any]:
        """
        Solve a logical puzzle using systematic reasoning
        
        Args:
            puzzle: Logical puzzle description
            puzzle_rules: Rules governing the puzzle
            
        Returns:
            Puzzle solution with logical reasoning
        """
        # Configure for logical problem solving
        logic_config = ProblemSolvingConfig(
            problem_type="logical",
            solution_requirements=["satisfies all constraints", "logical consistency"],
            constraints=puzzle_rules or [],
            evaluation_criteria=["logical validity", "constraint satisfaction"]
        )
        
        # Create logical context
        task_context = self._create_logical_context(puzzle, puzzle_rules)
        
        # Execute reasoning
        result = await self.tot_engine.solve(puzzle, task_context)
        
        return self._process_logical_results(result, puzzle)
    
    def _create_problem_context(self, problem_statement: str, problem_type: str,
                              constraints: List[str], hints: List[str]) -> Dict[str, Any]:
        """Create task-specific context for general problem solving"""
        return {
            "problem_statement": problem_statement,
            "problem_type": problem_type or self.problem_config.problem_type,
            "constraints": constraints or self.problem_config.constraints or [],
            "hints": hints or [],
            "solution_requirements": self.problem_config.solution_requirements or [],
            "evaluation_criteria": self.problem_config.evaluation_criteria or [],
            
            # Custom prompts for problem solving
            "propose_prompt": self._get_problem_propose_prompt(),
            "value_prompt": self._get_problem_value_prompt(),
            "solution_prompt": self._get_problem_solution_prompt()
        }
    
    def _create_mathematical_context(self, problem: str, show_work: bool) -> Dict[str, Any]:
        """Create context for mathematical problem solving"""
        return {
            "problem_statement": problem,
            "problem_type": "mathematical",
            "show_work": show_work,
            "constraints": ["must show mathematical steps", "must be mathematically correct"],
            
            "propose_prompt": self._get_math_propose_prompt(),
            "value_prompt": self._get_math_value_prompt(),
            "solution_prompt": self._get_math_solution_prompt()
        }
    
    def _create_logical_context(self, puzzle: str, rules: List[str]) -> Dict[str, Any]:
        """Create context for logical puzzle solving"""
        return {
            "problem_statement": puzzle,
            "problem_type": "logical",
            "puzzle_rules": rules or [],
            "constraints": rules or [],
            
            "propose_prompt": self._get_logic_propose_prompt(),
            "value_prompt": self._get_logic_value_prompt(),
            "solution_prompt": self._get_logic_solution_prompt()
        }
    
    def _get_problem_propose_prompt(self) -> str:
        """Get proposal prompt for general problem solving"""
        return """Problem Solving Task: {task_description}

Current solution step: {current_thought}
Depth: {current_depth}

Problem type: {problem_type}
Constraints: {constraints}
Hints: {hints}

Propose the next logical step toward solving this problem. Consider:
- Problem requirements and constraints
- Available information and hints
- Logical progression toward solution
- Alternative approaches if current path seems blocked

Next solution step:"""
    
    def _get_problem_value_prompt(self) -> str:
        """Get value evaluation prompt for general problem solving"""
        return """Problem Solving Evaluation

Solution step: {thought_content}
Problem type: {problem_type}

Evaluate this solution step based on:
1. Correctness: Is this step logically sound?
2. Progress: Does it move closer to the solution?
3. Efficiency: Is this an efficient approach?
4. Completeness: How much of the problem does this address?

Constraints: {constraints}
Evaluation criteria: {evaluation_criteria}

Rate this solution step on a scale of 0.0 to 1.0:

Score:"""
    
    def _get_problem_solution_prompt(self) -> str:
        """Get solution check prompt for general problem solving"""
        return """Problem Solution Check

Problem: {task_description}
Proposed solution: {thought_content}

Solution requirements: {solution_requirements}
Constraints: {constraints}

Is this a complete and correct solution that satisfies all requirements and constraints?

Answer yes or no:"""
    
    def _get_math_propose_prompt(self) -> str:
        """Get proposal prompt for mathematical problems"""
        return """Mathematical Problem: {problem_statement}

Current step: {current_thought}
Show work: {show_work}

Propose the next mathematical step. Be precise and show your work clearly.
Consider:
- Mathematical operations and rules
- Order of operations
- Algebraic manipulations
- Verification of steps

Next mathematical step:"""
    
    def _get_math_value_prompt(self) -> str:
        """Get value evaluation prompt for mathematical problems"""
        return """Mathematical Step Evaluation

Step: {thought_content}

Evaluate based on:
1. Mathematical accuracy: Is the math correct?
2. Clarity: Is the step clearly explained?
3. Progress: Does it advance the solution?
4. Methodology: Is the approach sound?

Score (0.0 to 1.0):"""
    
    def _get_math_solution_prompt(self) -> str:
        """Get solution check prompt for mathematical problems"""
        return """Mathematical Solution Check

Problem: {problem_statement}
Solution: {thought_content}

Is this mathematically correct and complete?

Answer yes or no:"""
    
    def _get_logic_propose_prompt(self) -> str:
        """Get proposal prompt for logical puzzles"""
        return """Logical Puzzle: {problem_statement}

Current reasoning: {current_thought}
Rules: {puzzle_rules}

Propose the next logical deduction or reasoning step.
Consider:
- Given constraints and rules
- Logical implications
- Process of elimination
- Consistency checks

Next logical step:"""
    
    def _get_logic_value_prompt(self) -> str:
        """Get value evaluation prompt for logical puzzles"""
        return """Logical Reasoning Evaluation

Reasoning step: {thought_content}
Puzzle rules: {puzzle_rules}

Evaluate based on:
1. Logical validity: Is the reasoning sound?
2. Rule compliance: Does it follow all rules?
3. Deductive power: How much does it narrow possibilities?
4. Consistency: Is it consistent with previous steps?

Score (0.0 to 1.0):"""
    
    def _get_logic_solution_prompt(self) -> str:
        """Get solution check prompt for logical puzzles"""
        return """Logical Puzzle Solution Check

Puzzle: {problem_statement}
Solution: {thought_content}
Rules: {puzzle_rules}

Does this solution satisfy all puzzle rules and constraints?

Answer yes or no:"""
    
    def _process_problem_results(self, result, problem_statement: str) -> Dict[str, Any]:
        """Process general problem solving results"""
        return self._create_solution_format(result, problem_statement, "general_problem")
    
    def _process_mathematical_results(self, result, problem: str) -> Dict[str, Any]:
        """Process mathematical problem results"""
        processed = self._create_solution_format(result, problem, "mathematical_problem")
        
        if result.best_solution:
            # Extract mathematical answer if possible
            solution_text = result.best_solution.content
            processed["mathematical_answer"] = self._extract_mathematical_answer(solution_text)
        
        return processed
    
    def _process_logical_results(self, result, puzzle: str) -> Dict[str, Any]:
        """Process logical puzzle results"""
        return self._create_solution_format(result, puzzle, "logical_puzzle")
    
    def _create_solution_format(self, result, problem: str, problem_type: str) -> Dict[str, Any]:
        """Create standardized solution format"""
        processed_result = {
            "problem_type": problem_type,
            "problem_statement": problem,
            "success": result.success,
            "solution_found": result.best_solution is not None,
            "total_time": result.total_time,
            "reasoning_stats": {
                "nodes_explored": result.nodes_explored,
                "max_depth_reached": result.max_depth_reached,
                "total_llm_calls": result.total_llm_calls
            }
        }
        
        if result.best_solution:
            processed_result["primary_solution"] = {
                "content": result.best_solution.content,
                "confidence_score": result.best_solution.value_score,
                "solution_path": self.tot_engine.format_solution_path(
                    result.best_solution, result.tree
                )
            }
            
            if len(result.all_solutions) > 1:
                processed_result["alternative_solutions"] = [
                    {
                        "content": sol.content,
                        "confidence_score": sol.value_score
                    }
                    for sol in sorted(result.all_solutions,
                                    key=lambda s: s.value_score, reverse=True)[1:]
                ]
        
        if not result.success:
            processed_result["error"] = result.error_message
        
        return processed_result
    
    def _extract_mathematical_answer(self, solution_text: str) -> Optional[str]:
        """Extract the final mathematical answer from solution text"""
        import re
        
        # Look for patterns like "= 42", "Answer: 42", "Result: 42"
        patterns = [
            r"=\s*([+-]?\d*\.?\d+)",
            r"answer[:\s]*([+-]?\d*\.?\d+)",
            r"result[:\s]*([+-]?\d*\.?\d+)",
            r"solution[:\s]*([+-]?\d*\.?\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution_text.lower())
            if match:
                return match.group(1)
        
        return None
