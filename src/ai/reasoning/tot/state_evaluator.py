"""
State Evaluator for Tree of Thought Framework

Evaluates thought states using different methods (value vs vote)
to guide the search process toward promising reasoning paths.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from .models import ThoughtState, ToTConfig, EvaluationMethod
from .thought_generator import LLMBackend, OllamaBackend
from ....utils.async_utils import run_with_timeout

logger = logging.getLogger(__name__)


class StateEvaluator:
    """Evaluates thought states for Tree of Thought reasoning"""
    
    def __init__(self, config: ToTConfig, llm_backend: Optional[LLMBackend] = None):
        self.config = config
        self.llm_backend = llm_backend or OllamaBackend(config.model_name)
        self.evaluation_count = 0
    
    async def evaluate_states(self, states: List[ThoughtState], 
                            task_context: Dict[str, Any]) -> List[ThoughtState]:
        """
        Evaluate multiple thought states
        
        Args:
            states: List of thought states to evaluate
            task_context: Task-specific context and prompts
            
        Returns:
            List of evaluated thought states with updated scores
        """
        if self.config.evaluation_method == EvaluationMethod.VALUE:
            return await self._evaluate_value_method(states, task_context)
        else:
            return await self._evaluate_vote_method(states, task_context)
    
    async def _evaluate_value_method(self, states: List[ThoughtState], 
                                   task_context: Dict[str, Any]) -> List[ThoughtState]:
        """Evaluate states independently using value scoring"""
        evaluated_states = []
        
        # Get the value prompt template
        value_prompt = task_context.get("value_prompt", self._default_value_prompt())
        
        # Evaluate each state independently
        for state in states:
            try:
                # Run multiple evaluations for robustness
                scores = []
                for i in range(self.config.n_evaluate_sample):
                    prompt = self._format_value_prompt(value_prompt, state, task_context)
                    score = await self._get_value_score(prompt)
                    if score is not None:
                        scores.append(score)
                
                # Calculate average score
                if scores:
                    state.value_score = sum(scores) / len(scores)
                    state.confidence = 1.0 - (max(scores) - min(scores)) / max(1.0, max(scores))
                else:
                    state.value_score = 0.0
                    state.confidence = 0.0
                
                # Check if this is a solution
                state.is_solution = await self._check_solution(state, task_context)
                
                evaluated_states.append(state)
                
            except Exception as e:
                logger.error(f"Error evaluating state {state.id}: {e}")
                state.value_score = 0.0
                state.confidence = 0.0
                evaluated_states.append(state)
        
        return evaluated_states
    
    async def _evaluate_vote_method(self, states: List[ThoughtState], 
                                  task_context: Dict[str, Any]) -> List[ThoughtState]:
        """Evaluate states collectively using voting"""
        if not states:
            return states
        
        # Get the vote prompt template
        vote_prompt = task_context.get("vote_prompt", self._default_vote_prompt())
        
        # Create voting prompt with all states
        prompt = self._format_vote_prompt(vote_prompt, states, task_context)
        
        try:
            # Run multiple voting rounds
            vote_results = []
            for i in range(self.config.n_evaluate_sample):
                votes = await self._get_vote_results(prompt, len(states))
                if votes:
                    vote_results.append(votes)
            
            # Aggregate votes
            if vote_results:
                for i, state in enumerate(states):
                    state.vote_count = sum(votes[i] for votes in vote_results if i < len(votes))
                    state.value_score = state.vote_count / len(vote_results)
                    
                    # Check if this is a solution
                    state.is_solution = await self._check_solution(state, task_context)
            
        except Exception as e:
            logger.error(f"Error in vote evaluation: {e}")
            # Fallback to equal scores
            for state in states:
                state.value_score = 0.5
                state.vote_count = 1
        
        return states
    
    async def _get_value_score(self, prompt: str) -> Optional[float]:
        """Get a numerical value score from LLM"""
        self.evaluation_count += 1
        
        try:
            response = await run_with_timeout(
                self.llm_backend.generate(
                    prompt,
                    temperature=0.1,  # Lower temperature for evaluation
                    max_tokens=100
                ),
                timeout=20.0
            )
            
            # Extract numerical score from response
            score = self._extract_score(response)
            return score
            
        except Exception as e:
            logger.error(f"Error getting value score: {e}")
            return None
    
    async def _get_vote_results(self, prompt: str, num_states: int) -> Optional[List[int]]:
        """Get voting results from LLM"""
        self.evaluation_count += 1
        
        try:
            response = await run_with_timeout(
                self.llm_backend.generate(
                    prompt,
                    temperature=0.1,
                    max_tokens=200
                ),
                timeout=20.0
            )
            
            # Extract vote rankings from response
            votes = self._extract_votes(response, num_states)
            return votes
            
        except Exception as e:
            logger.error(f"Error getting vote results: {e}")
            return None
    
    async def _check_solution(self, state: ThoughtState, task_context: Dict[str, Any]) -> bool:
        """Check if a state represents a valid solution"""
        solution_prompt = task_context.get("solution_prompt", self._default_solution_prompt())
        
        try:
            prompt = self._format_solution_prompt(solution_prompt, state, task_context)
            response = await run_with_timeout(
                self.llm_backend.generate(
                    prompt,
                    temperature=0.1,
                    max_tokens=50
                ),
                timeout=15.0
            )
            
            # Check for positive solution indicators
            response_lower = response.lower()
            return any(indicator in response_lower for indicator in 
                      ["yes", "correct", "solution", "valid", "true"])
            
        except Exception as e:
            logger.error(f"Error checking solution: {e}")
            return False
    
    def _extract_score(self, response: str) -> Optional[float]:
        """Extract numerical score from LLM response"""
        # Look for patterns like "Score: 0.8", "8/10", "80%", etc.
        patterns = [
            r"score[:\s]*([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)[/\s]*10",
            r"([0-9]*\.?[0-9]+)%",
            r"([0-9]*\.?[0-9]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to 0-1 range
                    if score > 1.0:
                        score = score / 10.0 if score <= 10.0 else score / 100.0
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue
        
        return None
    
    def _extract_votes(self, response: str, num_states: int) -> Optional[List[int]]:
        """Extract vote counts from LLM response"""
        votes = [0] * num_states
        
        # Look for patterns like "Option 1: 3 votes", "State A gets 2 votes", etc.
        lines = response.split('\n')
        for line in lines:
            for i in range(num_states):
                if f"option {i+1}" in line.lower() or f"state {i+1}" in line.lower():
                    vote_match = re.search(r"(\d+)", line)
                    if vote_match:
                        votes[i] = int(vote_match.group(1))
        
        return votes if sum(votes) > 0 else None
    
    def _format_value_prompt(self, template: str, state: ThoughtState, 
                           task_context: Dict[str, Any]) -> str:
        """Format value evaluation prompt"""
        return template.format(
            task_description=self.config.task_description,
            thought_content=state.content,
            success_criteria=self.config.success_criteria,
            **task_context
        )
    
    def _format_vote_prompt(self, template: str, states: List[ThoughtState], 
                          task_context: Dict[str, Any]) -> str:
        """Format vote evaluation prompt"""
        states_text = "\n".join([f"Option {i+1}: {state.content}" 
                                for i, state in enumerate(states)])
        
        return template.format(
            task_description=self.config.task_description,
            states_list=states_text,
            success_criteria=self.config.success_criteria,
            **task_context
        )
    
    def _format_solution_prompt(self, template: str, state: ThoughtState, 
                              task_context: Dict[str, Any]) -> str:
        """Format solution check prompt"""
        return template.format(
            task_description=self.config.task_description,
            thought_content=state.content,
            success_criteria=self.config.success_criteria,
            **task_context
        )
    
    def _default_value_prompt(self) -> str:
        """Default prompt for value evaluation"""
        return """Task: {task_description}

Evaluate this reasoning step: {thought_content}

Success criteria: {success_criteria}

Rate this step on a scale of 0.0 to 1.0 based on how likely it is to lead to a correct solution.
Consider: logical correctness, progress toward goal, and feasibility.

Score: """
    
    def _default_vote_prompt(self) -> str:
        """Default prompt for vote evaluation"""
        return """Task: {task_description}

Compare these reasoning options:
{states_list}

Success criteria: {success_criteria}

Rank these options by how promising they are for reaching the solution.
Give each option a vote count from 0-5.

Ranking:"""
    
    def _default_solution_prompt(self) -> str:
        """Default prompt for solution checking"""
        return """Task: {task_description}

Proposed solution: {thought_content}

Success criteria: {success_criteria}

Is this a valid and complete solution? Answer yes or no.

Answer:"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            "total_evaluations": self.evaluation_count,
            "method": self.config.evaluation_method.value,
            "model": self.config.model_name
        }
