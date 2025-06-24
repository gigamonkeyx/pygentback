"""
AI Agent Orchestration System for Autonomous Historical Research
Coordinates multiple specialized AI agents for complex research tasks.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .ollama_integration import ollama_manager, ModelCapability
from .openrouter_integration import openrouter_manager

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of research agents."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"  
    FACT_CHECKER = "fact_checker"
    SUMMARIZER = "summarizer"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"
    DOCUMENT_ANALYZER = "document_analyzer"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentTask:
    """Individual task for an agent."""
    id: str
    agent_type: AgentType
    task_type: str
    input_data: Dict[str, Any]
    priority: int = 5  # 1-10, higher = more urgent
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

@dataclass
class AgentCapability:
    """Agent capability definition."""
    agent_type: AgentType
    tasks: List[str]
    model_preference: str  # "ollama" or "openrouter"
    model_capability: Optional[ModelCapability] = None
    concurrent_limit: int = 1

class ResearchAgent:
    """Base class for specialized research agents."""
    
    def __init__(self, agent_type: AgentType, agent_id: str):
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.is_busy = False
        self.task_count = 0
        self.success_count = 0
        
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a task (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def can_handle_task(self, task_type: str) -> bool:
        """Check if agent can handle specific task type."""
        raise NotImplementedError

class ResearchCoordinatorAgent(ResearchAgent):
    """Coordinates research workflow and delegates tasks."""
    
    def __init__(self):
        super().__init__(AgentType.COORDINATOR, "coordinator_001")
        
    def can_handle_task(self, task_type: str) -> bool:
        return task_type in ["research_planning", "task_delegation", "result_synthesis"]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process coordination tasks."""
        if task.task_type == "research_planning":
            return await self._plan_research(task.input_data)
        elif task.task_type == "task_delegation":
            return await self._delegate_tasks(task.input_data)
        elif task.task_type == "result_synthesis":
            return await self._synthesize_results(task.input_data)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def _plan_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Plan research workflow for a topic."""
        topic = input_data.get("topic", "")
        scope = input_data.get("scope", "comprehensive")
        
        prompt = f"""Plan a comprehensive historical research workflow for: {topic}

Scope: {scope}

Create a research plan with:
1. Key research questions to investigate
2. Types of sources to prioritize
3. Research phases and their sequence
4. Validation checkpoints
5. Expected deliverables

Return as JSON with structured plan."""
        
        response = await ollama_manager.generate(
            prompt=prompt,
            capability=ModelCapability.REASONING,
            temperature=0.3
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"plan": response.content, "format": "text"}
    
    async def _delegate_tasks(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate tasks to appropriate agents."""
        tasks = input_data.get("tasks", [])
        available_agents = input_data.get("available_agents", [])
        
        # Simple task assignment logic
        assignments = []
        for task in tasks:
            best_agent = self._select_best_agent(task, available_agents)
            assignments.append({
                "task": task,
                "assigned_agent": best_agent,
                "priority": task.get("priority", 5)
            })
        
        return {"assignments": assignments}
    
    async def _synthesize_results(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple agents."""
        results = input_data.get("results", [])
        
        # Combine results into coherent output
        combined_data = {}
        for result in results:
            agent_type = result.get("agent_type")
            data = result.get("data", {})
            combined_data[agent_type] = data
        
        return {"synthesized_results": combined_data}
    
    def _select_best_agent(self, task: Dict[str, Any], available_agents: List[str]) -> str:
        """Select best agent for a task."""
        task_type = task.get("type", "")
        
        # Simple mapping of task types to agent types
        task_agent_mapping = {
            "document_analysis": AgentType.DOCUMENT_ANALYZER,
            "fact_checking": AgentType.FACT_CHECKER,
            "summarization": AgentType.SUMMARIZER,
            "validation": AgentType.VALIDATOR,
            "synthesis": AgentType.SYNTHESIZER
        }
        
        preferred_agent = task_agent_mapping.get(task_type, AgentType.RESEARCHER)
        return preferred_agent.value

class DocumentAnalyzerAgent(ResearchAgent):
    """Analyzes historical documents for content and metadata."""
    
    def __init__(self):
        super().__init__(AgentType.DOCUMENT_ANALYZER, "doc_analyzer_001")
    
    def can_handle_task(self, task_type: str) -> bool:
        return task_type in ["analyze_document", "extract_entities", "classify_document"]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process document analysis tasks."""
        if task.task_type == "analyze_document":
            return await self._analyze_document(task.input_data)
        elif task.task_type == "extract_entities":
            return await self._extract_entities(task.input_data)
        elif task.task_type == "classify_document":
            return await self._classify_document(task.input_data)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def _analyze_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document content."""
        text = input_data.get("text", "")
        return await ollama_manager.analyze_document(text)
    
    async def _extract_entities(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract historical entities from document."""
        text = input_data.get("text", "")
        return await ollama_manager.extract_entities(text)
    
    async def _classify_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document type and significance."""
        text = input_data.get("text", "")
        metadata = input_data.get("metadata", {})
        
        prompt = f"""Classify this historical document:

Metadata: {json.dumps(metadata, indent=2)}
Text: {text[:1500]}

Classify:
1. Document type (letter, speech, government document, diary, newspaper, etc.)
2. Historical period
3. Significance level (1-10)
4. Primary topics/themes
5. Reliability assessment
6. Potential research value

Return as JSON."""
        
        response = await ollama_manager.generate(
            prompt=prompt,
            capability=ModelCapability.REASONING,
            temperature=0.2
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"classification": response.content}

class FactCheckerAgent(ResearchAgent):
    """Validates claims and checks facts against sources."""
    
    def __init__(self):
        super().__init__(AgentType.FACT_CHECKER, "fact_checker_001")
    
    def can_handle_task(self, task_type: str) -> bool:
        return task_type in ["fact_check", "verify_claim", "cross_reference"]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process fact-checking tasks."""
        if task.task_type == "fact_check":
            return await self._fact_check(task.input_data)
        elif task.task_type == "verify_claim":
            return await self._verify_claim(task.input_data)
        elif task.task_type == "cross_reference":
            return await self._cross_reference(task.input_data)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def _fact_check(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fact-check a claim using local and external models."""
        claim = input_data.get("claim", "")
        context = input_data.get("context", "")
        
        # Try local fact-checking first
        local_result = await ollama_manager.fact_check(claim, context)
        
        # If available and within budget, also check with external model
        try:
            external_result = await openrouter_manager.fact_check_with_search(claim, context)
            
            # Combine results
            return {
                "claim": claim,
                "local_check": local_result,
                "external_check": external_result,
                "consensus": self._analyze_consensus(local_result, external_result)
            }
        except Exception as e:
            logger.warning(f"External fact-check failed: {e}")
            return {
                "claim": claim,
                "local_check": local_result,
                "external_check": None,
                "consensus": local_result
            }
    
    async def _verify_claim(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a specific historical claim."""
        return await self._fact_check(input_data)
    
    async def _cross_reference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-reference information across multiple sources."""
        sources = input_data.get("sources", [])
        topic = input_data.get("topic", "")
        
        prompt = f"""Cross-reference information about '{topic}' across these sources:

{chr(10).join([f"Source {i+1}: {source[:500]}..." for i, source in enumerate(sources[:5])])}

Analyze:
1. Consistent information across sources
2. Contradictions or discrepancies
3. Unique information in each source
4. Overall reliability assessment
5. Gaps or missing information

Return detailed cross-reference analysis as JSON."""
        
        response = await ollama_manager.generate(
            prompt=prompt,
            capability=ModelCapability.REASONING,
            temperature=0.3
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"analysis": response.content}
    
    def _analyze_consensus(self, local_result: Dict, external_result: Dict) -> Dict[str, Any]:
        """Analyze consensus between local and external fact-checking."""
        if not external_result:
            return local_result
        
        local_supported = local_result.get("supported", None)
        external_supported = external_result.get("supported", None)
        
        if local_supported == external_supported:
            consensus = "agreement"
            confidence = min(
                local_result.get("confidence", 0),
                external_result.get("confidence", 0)
            ) + 10  # Boost confidence for agreement
        else:
            consensus = "disagreement"
            confidence = abs(
                local_result.get("confidence", 50) - 
                external_result.get("confidence", 50)
            )
        
        return {
            "consensus": consensus,
            "confidence": min(confidence, 100),
            "supported": local_supported if consensus == "agreement" else None,
            "local_confidence": local_result.get("confidence", 0),
            "external_confidence": external_result.get("confidence", 0)
        }

class AgentOrchestrator:
    """Orchestrates multiple AI agents for complex research tasks."""
    
    def __init__(self):
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.completed_tasks = {}
        self.running_tasks = {}
        self._running = False
          # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all available agents."""
        self.agents = {
            AgentType.COORDINATOR: ResearchCoordinatorAgent(),
            AgentType.DOCUMENT_ANALYZER: DocumentAnalyzerAgent(),
            AgentType.FACT_CHECKER: FactCheckerAgent(),
            # Additional agents can be added here
        }
        
        logger.info(f"Initialized {len(self.agents)} research agents")
    
    async def start(self):
        """Start the agent orchestration system."""
        self._running = True
        
        # Initialize Ollama if not already initialized
        if not ollama_manager._initialized:
            await ollama_manager.initialize()
        
        logger.info("Agent orchestrator started")
        
        # Start task processing loop
        asyncio.create_task(self._process_tasks())
    
    async def stop(self):
        """Stop the agent orchestration system."""
        self._running = False
        logger.info("Agent orchestrator stopped")
    
    async def submit_task(self, task: AgentTask) -> str:
        """Submit a task for processing."""
        await self.task_queue.put(task)
        logger.info(f"Task {task.id} submitted to queue")
        return task.id
    
    async def get_task_result(self, task_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Get result of a completed task."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].result
            await asyncio.sleep(0.1)
        
        return None
    
    async def _process_tasks(self):
        """Main task processing loop."""
        while self._running:
            try:
                # Get next task (with timeout to allow checking _running)
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Find appropriate agent
                agent = self._find_agent_for_task(task)
                if agent:
                    # Process task asynchronously
                    asyncio.create_task(self._execute_task(agent, task))
                else:
                    logger.warning(f"No agent available for task {task.id}")
                    task.status = TaskStatus.FAILED
                    task.error = "No suitable agent available"
                    self.completed_tasks[task.id] = task
                    
            except asyncio.TimeoutError:
                continue  # Check _running flag
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
    
    def _find_agent_for_task(self, task: AgentTask) -> Optional[ResearchAgent]:
        """Find the best agent for a task."""
        # Try to find agent by type first
        if task.agent_type in self.agents:
            agent = self.agents[task.agent_type]
            if not agent.is_busy and agent.can_handle_task(task.task_type):
                return agent
        
        # Fallback: find any available agent that can handle the task
        for agent in self.agents.values():
            if not agent.is_busy and agent.can_handle_task(task.task_type):
                return agent
        
        return None
    
    async def _execute_task(self, agent: ResearchAgent, task: AgentTask):
        """Execute a task with the assigned agent."""
        import time
        
        agent.is_busy = True
        task.status = TaskStatus.IN_PROGRESS
        self.running_tasks[task.id] = task
        start_time = time.time()
        
        try:
            # Process the task
            result = await agent.process_task(task)
            
            # Update task
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.processing_time = time.time() - start_time
            
            # Update agent stats
            agent.task_count += 1
            agent.success_count += 1
            
            logger.info(f"Task {task.id} completed by {agent.agent_id}")
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.processing_time = time.time() - start_time
            
            agent.task_count += 1
            
            logger.error(f"Task {task.id} failed: {e}")
        
        finally:
            # Cleanup
            agent.is_busy = False
            self.completed_tasks[task.id] = task
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "running": self._running,
            "agents": {
                agent_type.value: {
                    "busy": agent.is_busy,
                    "task_count": agent.task_count,
                    "success_rate": agent.success_count / max(agent.task_count, 1)
                }
                for agent_type, agent in self.agents.items()
            },
            "queue_size": self.task_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks)
        }

# Global agent orchestrator instance
agent_orchestrator = AgentOrchestrator()
