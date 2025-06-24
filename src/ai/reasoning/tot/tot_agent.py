"""
Tree of Thought Agent - Orchestrates ToT reasoning processes

This agent integrates the ToT engine with the broader agent system,
providing a high-level interface for ToT-based problem solving.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .core.state import ReasoningState
from .tot_engine import ToTEngine, ToTSearchResult
from .models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from .thought_generator import OllamaBackend

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Represents a reasoning task assigned to the ToT agent"""
    task_id: str
    problem: str
    context: Dict[str, Any]
    priority: int = 1
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ToTAgent:
    """
    Tree of Thought Agent
    
    Provides high-level orchestration of ToT reasoning processes:
    - Task queue management
    - Session tracking
    - Result aggregation and reporting
    - Integration with agent communication systems
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        model_name: str = "qwen2.5-coder:7b",
        capabilities: Optional[List[str]] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.model_name = model_name
        self.capabilities = capabilities or ["reasoning", "problem_solving", "analysis"]
        
        # Agent state
        self.is_active = False
        self.current_sessions: Dict[str, ReasoningState] = {}
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        
        # ToT Engine (will be initialized when agent starts)
        self.tot_engine: Optional[ToTEngine] = None
        self.llm_backend: Optional[OllamaBackend] = None
        
        # Statistics
        self.total_sessions = 0
        self.successful_sessions = 0
        self.total_reasoning_time = 0.0
        
        logger.info(f"ToT Agent initialized: {self.name} (ID: {self.agent_id})")
        
    async def start(self) -> bool:
        """Start the ToT agent and initialize components"""
        try:
            # Initialize LLM backend
            self.llm_backend = OllamaBackend(model_name=self.model_name)
            
            # Test backend connection
            test_response = await self.llm_backend.generate("Test connection", max_tokens=10)
            if not test_response:
                raise Exception("LLM backend connection failed")
                
            # Create default ToT configuration
            default_config = ToTConfig(
                model_name=self.model_name,
                generation_strategy=GenerationStrategy.SAMPLE,
                evaluation_method=EvaluationMethod.VALUE,
                search_method=SearchMethod.ADAPTIVE,
                n_generate_sample=3,
                n_evaluate_sample=2,
                n_select_sample=2,
                max_depth=4,
                temperature=0.7,
                max_tokens=500,
                task_description="General problem solving",
                success_criteria="Find accurate and well-reasoned solutions"
            )
            
            # Initialize ToT engine
            self.tot_engine = ToTEngine(default_config, self.llm_backend)
            
            self.is_active = True
            logger.info(f"ToT Agent {self.name} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ToT Agent {self.name}: {e}")
            self.is_active = False
            return False
    
    async def stop(self) -> bool:
        """Stop the ToT agent and cleanup resources"""
        try:
            self.is_active = False
            
            # Cancel any ongoing sessions
            for session_id in list(self.current_sessions.keys()):
                await self.cancel_reasoning_session(session_id)
                
            # Clear task queue
            self.task_queue.clear()
            
            logger.info(f"ToT Agent {self.name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop ToT Agent {self.name}: {e}")
            return False
    
    async def solve_problem(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        custom_config: Optional[ToTConfig] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using ToT reasoning
        
        Args:
            problem: The problem statement
            context: Additional context for the problem
            custom_config: Custom ToT configuration for this problem
            
        Returns:
            Dictionary containing the reasoning result
        """
        if not self.is_active or not self.tot_engine:
            raise RuntimeError("ToT Agent is not active")
            
        session_id = f"session_{datetime.now().timestamp()}"
        self.total_sessions += 1
        
        try:
            logger.info(f"Starting ToT reasoning session {session_id}: {problem[:100]}...")
            
            # Use custom config if provided
            if custom_config:
                # Create new engine with custom config
                engine = ToTEngine(custom_config, self.llm_backend)
            else:
                engine = self.tot_engine
            
            # Create reasoning state for tracking
            reasoning_state = ReasoningState(
                problem=problem,
                session_id=session_id,
                metadata={
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "start_time": datetime.now().isoformat(),
                    "context": context or {}
                }
            )
            
            self.current_sessions[session_id] = reasoning_state
            
            # Execute ToT reasoning
            start_time = datetime.now()
            search_result: ToTSearchResult = await engine.solve(problem, context)
            end_time = datetime.now()
            
            reasoning_time = (end_time - start_time).total_seconds()
            self.total_reasoning_time += reasoning_time
            
            # Process results
            if search_result.success and search_result.solutions:
                self.successful_sessions += 1
                best_solution = search_result.solutions[0]
                
                result = {
                    "success": True,
                    "session_id": session_id,
                    "problem": problem,
                    "solution": best_solution.content,
                    "confidence": best_solution.confidence,
                    "reasoning_time": reasoning_time,
                    "total_thoughts": len(search_result.tree.nodes) if search_result.tree else 0,
                    "reasoning_depth": search_result.tree.get_max_depth() if search_result.tree else 0,
                    "alternative_solutions": [
                        {
                            "content": sol.content,
                            "confidence": sol.confidence
                        }
                        for sol in search_result.solutions[1:6]  # Top 5 alternatives
                    ],
                    "reasoning_tree": search_result.tree.to_dict() if search_result.tree else None,
                    "statistics": search_result.stats
                }
            else:
                result = {
                    "success": False,
                    "session_id": session_id,
                    "problem": problem,
                    "error": "No valid solutions found",
                    "reasoning_time": reasoning_time,
                    "statistics": search_result.stats
                }
            
            # Update reasoning state
            reasoning_state.metadata["end_time"] = end_time.isoformat()
            reasoning_state.metadata["result"] = result
            
            # Move session to completed
            del self.current_sessions[session_id]
            
            logger.info(f"ToT reasoning session {session_id} completed: {'SUCCESS' if result['success'] else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"ToT reasoning session {session_id} failed: {e}")
            
            # Cleanup failed session
            if session_id in self.current_sessions:
                del self.current_sessions[session_id]
                
            return {
                "success": False,
                "session_id": session_id,
                "problem": problem,
                "error": str(e),
                "reasoning_time": 0.0
            }
    
    async def add_task(self, problem: str, context: Dict[str, Any], priority: int = 1) -> str:
        """Add a reasoning task to the queue"""
        task = AgentTask(
            task_id=f"task_{datetime.now().timestamp()}",
            problem=problem,
            context=context,
            priority=priority
        )
        
        # Insert task in priority order
        inserted = False
        for i, existing_task in enumerate(self.task_queue):
            if task.priority > existing_task.priority:
                self.task_queue.insert(i, task)
                inserted = True
                break
                
        if not inserted:
            self.task_queue.append(task)
            
        logger.info(f"Added task {task.task_id} to queue (priority {priority})")
        return task.task_id
    
    async def process_task_queue(self) -> List[Dict[str, Any]]:
        """Process all tasks in the queue"""
        results = []
        
        while self.task_queue and self.is_active:
            task = self.task_queue.pop(0)
            
            try:
                result = await self.solve_problem(task.problem, task.context)
                result["task_id"] = task.task_id
                results.append(result)
                
                self.completed_tasks.append(task)
                
            except Exception as e:
                logger.error(f"Failed to process task {task.task_id}: {e}")
                results.append({
                    "success": False,
                    "task_id": task.task_id,
                    "error": str(e)
                })
        
        return results
    
    async def cancel_reasoning_session(self, session_id: str) -> bool:
        """Cancel an active reasoning session"""
        if session_id in self.current_sessions:
            del self.current_sessions[session_id]
            logger.info(f"Cancelled reasoning session {session_id}")
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "is_active": self.is_active,
            "model_name": self.model_name,
            "capabilities": self.capabilities,
            "current_sessions": len(self.current_sessions),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "statistics": {
                "total_sessions": self.total_sessions,
                "successful_sessions": self.successful_sessions,
                "success_rate": self.successful_sessions / max(1, self.total_sessions),
                "total_reasoning_time": self.total_reasoning_time,
                "avg_reasoning_time": self.total_reasoning_time / max(1, self.total_sessions)
            }
        }
    
    def __repr__(self) -> str:
        return f"ToTAgent(id='{self.agent_id}', name='{self.name}', active={self.is_active})"
