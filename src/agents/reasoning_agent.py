"""
Reasoning Agent - Tree of Thought Implementation

Uses the sophisticated ToT reasoning system for complex multi-path reasoning tasks.
Replaces the mock "reasoning agent" with real AI-powered reasoning capabilities.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from core.agent import BaseAgent, AgentMessage, MessageType
from core.agent.config import AgentConfig
from ai.reasoning.tot.models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from ai.reasoning.tot.tot_engine import ToTEngine
from ai.reasoning.tot.thought_generator import OllamaBackend

logger = logging.getLogger(__name__)


class ReasoningAgent(BaseAgent):
    """
    Real Reasoning Agent using Tree of Thought (ToT) reasoning
    
    This agent uses the sophisticated ToT engine to perform multi-path reasoning,
    exploring different solution paths and selecting the best approach.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Configure ToT for reasoning tasks
        self.tot_config = ToTConfig(
            generation_strategy=GenerationStrategy.PROPOSE,
            evaluation_method=EvaluationMethod.VALUE,
            search_method=SearchMethod.BFS,
            n_generate_sample=3,
            n_evaluate_sample=2,
            n_select_sample=3,
            max_depth=4,
            temperature=0.7,
            model_name=config.get_custom_config("model_name", "")
        )

        # Initialize ToT engine with Ollama backend
        self.ollama_backend = OllamaBackend(
            model_name=self.tot_config.model_name,
            base_url=config.get_custom_config("ollama_url", "http://localhost:11434")
        )
        
        self.tot_engine = ToTEngine(
            self.tot_config, 
            self.ollama_backend
        )
        
        # Agent state
        self.reasoning_history = []
        self.total_reasoning_time = 0.0
        
        logger.info(f"ReasoningAgent initialized with ToT engine (model: {self.tot_config.model_name})")

    async def _agent_initialize(self) -> None:
        """Agent-specific initialization logic"""
        # Test ToT engine
        try:
            test_result = await self.tot_engine.solve("Test problem", {"max_depth": 1})
            if test_result.success:
                logger.info("ToT engine verified")
            else:
                logger.warning("ToT engine test failed")
        except Exception as e:
            logger.error(f"ToT engine initialization error: {e}")

    async def _agent_shutdown(self) -> None:
        """Agent-specific shutdown logic"""
        logger.info("ReasoningAgent shutting down")

    async def _handle_request(self, message: AgentMessage) -> AgentMessage:
        """Handle a request message"""
        return await self.process_message(message)

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process message using Tree of Thought reasoning
        
        Args:
            message: Incoming message to reason about
            
        Returns:
            AgentMessage: Response with reasoning results
        """
        start_time = datetime.now()
        
        try:
            # Extract the problem/question from the message
            problem = message.content.get("content", str(message.content))
            
            logger.info(f"Starting ToT reasoning for: {problem[:100]}...")
            
            # Use ToT engine to solve the problem
            result = await self.tot_engine.solve(problem)
            
            # Calculate reasoning time
            reasoning_time = (datetime.now() - start_time).total_seconds()
            self.total_reasoning_time += reasoning_time
            
            # Create response based on ToT results
            if result.success and result.best_solution:
                response_content = self._format_reasoning_response(result, reasoning_time)
                
                # Store in reasoning history
                self.reasoning_history.append({
                    "problem": problem,
                    "solution": result.best_solution.content,
                    "reasoning_time": reasoning_time,
                    "nodes_explored": result.nodes_explored,
                    "timestamp": start_time.isoformat()
                })
                
                logger.info(f"ToT reasoning completed successfully in {reasoning_time:.2f}s")
                
            else:
                # Fallback response if ToT fails
                response_content = self._create_fallback_response(problem, result.error_message)
                logger.warning(f"ToT reasoning failed: {result.error_message}")
            
            # Create response message
            response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content=response_content,
                correlation_id=message.id
            )
            
            # Update agent activity
            self.last_activity = datetime.utcnow()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in ReasoningAgent.process_message: {e}")
            
            # Create error response
            error_response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content={
                    "type": "error",
                    "message": f"Reasoning failed: {str(e)}",
                    "agent": "reasoning"
                },
                correlation_id=message.id
            )
            
            return error_response
    
    def _format_reasoning_response(self, result, reasoning_time: float) -> Dict[str, Any]:
        """Format the ToT reasoning result into a response"""
        
        # Get the reasoning path
        reasoning_path = []
        if result.tree and result.best_solution:
            path = self.tot_engine.get_solution_path(result.best_solution, result.tree)
            reasoning_path = [state.content for state in path]
        
        return {
            "type": "reasoning_response",
            "solution": result.best_solution.content,
            "confidence": result.best_solution.value_score if hasattr(result.best_solution, 'value_score') else 0.8,
            "reasoning_path": reasoning_path,
            "alternatives": [sol.content for sol in result.all_solutions[:3]] if result.all_solutions else [],
            "metrics": {
                "reasoning_time": reasoning_time,
                "nodes_explored": result.nodes_explored,
                "max_depth_reached": result.max_depth_reached,
                "total_llm_calls": result.total_llm_calls
            },
            "agent": "reasoning"
        }
    
    def _create_fallback_response(self, problem: str, error_message: Optional[str]) -> Dict[str, Any]:
        """Create a fallback response when ToT reasoning fails"""
        return {
            "type": "reasoning_response",
            "solution": f"I encountered an issue while reasoning about: {problem}. Let me provide a direct response instead.",
            "confidence": 0.3,
            "reasoning_path": ["Initial analysis", "Direct response due to reasoning system issue"],
            "alternatives": [],
            "metrics": {
                "reasoning_time": 0.0,
                "nodes_explored": 0,
                "max_depth_reached": 0,
                "total_llm_calls": 0
            },
            "error": error_message,
            "agent": "reasoning"
        }
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics for this agent"""
        return {
            "total_reasoning_sessions": len(self.reasoning_history),
            "total_reasoning_time": self.total_reasoning_time,
            "average_reasoning_time": self.total_reasoning_time / max(1, len(self.reasoning_history)),
            "tot_engine_stats": self.tot_engine.get_stats() if self.tot_engine else {},
            "recent_sessions": self.reasoning_history[-5:] if self.reasoning_history else []
        }
