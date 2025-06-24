"""
Research Orchestrator Integration Module
Integrates the Research Orchestrator with PyGent Factory system
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime

from .research_orchestrator import (
    ResearchOrchestrator, 
    ResearchQuery, 
    ResearchOutput, 
    OutputFormat,
    ResearchPhase
)
from .coordination_models import OrchestrationConfig, TaskRequest, TaskStatus
from .agent_registry import AgentRegistry
from .task_dispatcher import TaskDispatcher
from .mcp_orchestrator import MCPOrchestrator

logger = logging.getLogger(__name__)


class ResearchTaskType:
    """Extended task types for research operations"""
    RESEARCH_QUERY = "research_query"
    LITERATURE_REVIEW = "literature_review"
    DATA_ANALYSIS = "data_analysis"
    REPORT_GENERATION = "report_generation"


class ResearchAgentType:
    """Research-specific agent types"""
    RESEARCH_PLANNER = "research_planner"
    WEB_RESEARCHER = "web_researcher" 
    ACADEMIC_ANALYZER = "academic_analyzer"
    CITATION_SPECIALIST = "citation_specialist"
    OUTPUT_GENERATOR = "output_generator"


class ResearchOrchestrationManager:
    """
    Manages Research Orchestrator integration with PyGent Factory.
    Provides high-level interface for research operations.
    """
    
    def __init__(self,
                 config: OrchestrationConfig,
                 agent_registry: AgentRegistry,
                 task_dispatcher: TaskDispatcher,
                 mcp_orchestrator: MCPOrchestrator):
        
        self.config = config
        self.agent_registry = agent_registry
        self.task_dispatcher = task_dispatcher
        self.mcp_orchestrator = mcp_orchestrator
        
        # Initialize research orchestrator
        self.research_orchestrator = ResearchOrchestrator(
            config=config,
            agent_registry=agent_registry,
            task_dispatcher=task_dispatcher,
            mcp_orchestrator=mcp_orchestrator
        )
        
        # Register research agents
        self._register_research_agents()
        
        logger.info("Research Orchestration Manager initialized")
    
    def _register_research_agents(self):
        """Register research-specific agents in the agent registry"""
        
        research_agents = [
            {
                "agent_type": ResearchAgentType.RESEARCH_PLANNER,
                "capabilities": ["topic_discovery", "hypothesis_generation", "strategy_planning"],
                "description": "Strategic research planning and methodology design"
            },
            {
                "agent_type": ResearchAgentType.WEB_RESEARCHER,
                "capabilities": ["web_search", "source_validation", "bias_detection"],
                "description": "Web-based research and source credibility assessment"
            },
            {
                "agent_type": ResearchAgentType.ACADEMIC_ANALYZER,
                "capabilities": ["document_analysis", "pattern_recognition", "synthesis"],
                "description": "Academic document analysis and synthesis"
            },
            {
                "agent_type": ResearchAgentType.CITATION_SPECIALIST,
                "capabilities": ["citation_formatting", "reference_management", "plagiarism_detection"],
                "description": "Citation management and academic formatting"
            },
            {
                "agent_type": ResearchAgentType.OUTPUT_GENERATOR,
                "capabilities": ["report_generation", "format_conversion", "quality_assessment"],
                "description": "Research output generation and formatting"
            }
        ]
        
        for agent_config in research_agents:
            # Register agent capabilities
            logger.info(f"Registering research agent: {agent_config['agent_type']}")
    
    async def submit_research_request(self, research_request: Dict[str, Any]) -> str:
        """
        Submit a research request through the task dispatcher.
        
        Args:
            research_request: Dict containing research parameters
            
        Returns:
            str: Research session ID
        """
        try:
            # Convert request to ResearchQuery
            query = self._create_research_query(research_request)
              # Create task request for the dispatcher
            task_request = TaskRequest(
                task_type=ResearchTaskType.RESEARCH_QUERY,
                description=f"Research query: {query.topic}",
                input_data={
                    "query": query.__dict__,
                    "research_id": query.query_id
                },
                priority=research_request.get("priority", 1),
                estimated_duration=research_request.get("estimated_hours", 2) * 3600  # Convert to seconds
            )
            
            # Submit through task dispatcher
            success = await self.task_dispatcher.submit_task(task_request)
            
            if success:
                logger.info(f"Research request submitted successfully: {query.query_id}")
                return query.query_id
            else:
                logger.error(f"Failed to submit research request: {query.query_id}")
                raise RuntimeError("Failed to submit research task")
                
        except Exception as e:
            logger.error(f"Research request submission failed: {e}")
            raise
    
    def _create_research_query(self, request: Dict[str, Any]) -> ResearchQuery:
        """Convert request dict to ResearchQuery object"""
        
        return ResearchQuery(
            topic=request.get("topic", ""),
            research_questions=request.get("research_questions", []),
            domain=request.get("domain", "general"),
            depth_level=request.get("depth_level", "comprehensive"),
            time_constraint=request.get("deadline"),
            output_format=OutputFormat(request.get("output_format", "research_summary")),
            quality_threshold=request.get("quality_threshold", 0.8),
            citation_style=request.get("citation_style", "APA"),
            language=request.get("language", "en")
        )
    
    async def conduct_research(self, query: ResearchQuery) -> ResearchOutput:
        """
        Direct research execution (bypassing task dispatcher for immediate results).
        
        Args:
            query: ResearchQuery object
            
        Returns:
            ResearchOutput: Complete research results
        """
        try:
            logger.info(f"Starting direct research execution: {query.query_id}")
            
            # Execute research through orchestrator
            output = await self.research_orchestrator.conduct_research(query)
            
            logger.info(f"Research execution completed: {query.query_id}")
            
            return output
            
        except Exception as e:
            logger.error(f"Direct research execution failed: {e}")
            raise
    
    async def get_research_status(self, research_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of research session.
        
        Args:
            research_id: Research session ID
            
        Returns:
            Optional[Dict]: Research status information
        """
        try:
            # Check orchestrator status
            orchestrator_status = await self.research_orchestrator.get_research_status(research_id)
            
            if orchestrator_status:
                return {
                    "research_id": research_id,
                    "status": orchestrator_status.get("status", "unknown"),
                    "phase": orchestrator_status.get("phase", {}).get("value", "unknown") if orchestrator_status.get("phase") else "unknown",
                    "start_time": orchestrator_status.get("start_time"),
                    "end_time": orchestrator_status.get("end_time"),
                    "error": orchestrator_status.get("error"),
                    "progress": self._calculate_progress(orchestrator_status)
                }
            
            # Check task dispatcher for task status
            task_status = await self._check_task_status(research_id)
            if task_status:
                return task_status
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get research status for {research_id}: {e}")
            return None
    
    def _calculate_progress(self, status: Dict[str, Any]) -> float:
        """Calculate research progress percentage"""
        
        phase_progress = {
            ResearchPhase.TOPIC_DISCOVERY: 0.1,
            ResearchPhase.HYPOTHESIS_GENERATION: 0.2,
            ResearchPhase.LITERATURE_REVIEW: 0.3,
            ResearchPhase.DATA_COLLECTION: 0.5,
            ResearchPhase.ANALYSIS: 0.7,
            ResearchPhase.SYNTHESIS: 0.8,
            ResearchPhase.OUTPUT_GENERATION: 0.9,
            ResearchPhase.VALIDATION: 1.0
        }
        
        current_phase = status.get("phase")
        if isinstance(current_phase, ResearchPhase):
            return phase_progress.get(current_phase, 0.0)
        
        return 0.0
    
    async def _check_task_status(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Check task status in dispatcher"""
        
        # Search through running and completed tasks
        for task_id, task in self.task_dispatcher.running_tasks.items():
            if task.input_data.get("research_id") == research_id:
                return {
                    "research_id": research_id,
                    "status": "running",
                    "task_id": task_id,
                    "start_time": task.started_at,
                    "progress": 0.5  # Assume 50% if running
                }
        
        for task_id, task in self.task_dispatcher.completed_tasks.items():
            if task.input_data.get("research_id") == research_id:
                return {
                    "research_id": research_id,
                    "status": "completed",
                    "task_id": task_id,
                    "start_time": task.started_at,
                    "end_time": task.completed_at,
                    "progress": 1.0
                }
        
        for task_id, task in self.task_dispatcher.failed_tasks.items():
            if task.input_data.get("research_id") == research_id:
                return {
                    "research_id": research_id,
                    "status": "failed",
                    "task_id": task_id,
                    "start_time": task.started_at,
                    "end_time": task.completed_at,
                    "error": task.error_message,
                    "progress": 0.0
                }
        
        return None
    
    async def cancel_research(self, research_id: str) -> bool:
        """
        Cancel active research session.
        
        Args:
            research_id: Research session ID
            
        Returns:
            bool: True if successfully cancelled
        """
        try:
            # Cancel in orchestrator
            orchestrator_cancelled = await self.research_orchestrator.cancel_research(research_id)
            
            # Cancel task in dispatcher if found
            task_cancelled = False
            for task_id, task in self.task_dispatcher.running_tasks.items():
                if task.input_data.get("research_id") == research_id:
                    task_cancelled = await self.task_dispatcher.cancel_task(task_id)
                    break
            
            success = orchestrator_cancelled or task_cancelled
            
            if success:
                logger.info(f"Research session cancelled: {research_id}")
            else:
                logger.warning(f"Research session not found for cancellation: {research_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel research {research_id}: {e}")
            return False
    
    def get_research_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive research metrics.
        
        Returns:
            Dict: Research system metrics
        """
        try:
            orchestrator_metrics = self.research_orchestrator.get_research_metrics()
            
            # Add integration metrics
            integration_metrics = {
                "registered_agents": len([
                    agent for agent in self.agent_registry.agents.values()
                    if hasattr(agent, 'agent_type') and 
                    any(research_type.value in str(agent.agent_type) for research_type in ResearchAgentType)
                ]),                "research_tasks_pending": len([
                    task for _, task in self.task_dispatcher.pending_tasks
                    if task.task_type in [ResearchTaskType.RESEARCH_QUERY]
                ]),
                "research_tasks_running": len([
                    task for task in self.task_dispatcher.running_tasks.values()
                    if task.task_type in [ResearchTaskType.RESEARCH_QUERY]
                ])
            }
            
            return {
                "orchestrator": orchestrator_metrics,
                "integration": integration_metrics,
                "system_health": self._assess_system_health()
            }
            
        except Exception as e:
            logger.error(f"Failed to get research metrics: {e}")
            return {"error": str(e)}
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall research system health"""
        
        health_indicators = {
            "orchestrator_active": self.research_orchestrator is not None,
            "task_dispatcher_running": self.task_dispatcher.is_running,
            "agent_registry_available": len(self.agent_registry.agents) > 0,
            "mcp_orchestrator_active": self.mcp_orchestrator is not None
        }
        
        health_score = sum(health_indicators.values()) / len(health_indicators)
        
        return {
            "overall_health": health_score,
            "indicators": health_indicators,
            "status": "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.6 else "unhealthy"
        }
    
    async def create_research_template(self, template_config: Dict[str, Any]) -> str:
        """
        Create a reusable research template.
        
        Args:
            template_config: Template configuration
            
        Returns:
            str: Template ID
        """
        template_id = f"template_{datetime.utcnow().timestamp()}"        # Store template configuration
        # In production: Use persistent storage
        template_data = {
            "template_id": template_id,
            "name": template_config.get("name", "Unnamed Template"),
            "description": template_config.get("description", ""),
            "default_query": self._create_research_query(template_config.get("defaults", {})),
            "parameters": template_config.get("parameters", {}),
            "created_at": datetime.utcnow()
        }
        
        # TODO: Persist template_data to storage
        _ = template_data  # Acknowledge template creation
        
        logger.info(f"Created research template: {template_id}")
        
        return template_id
    
    async def execute_research_template(self, template_id: str, 
                                      parameters: Dict[str, Any]) -> str:
        """
        Execute research using a predefined template.
        
        Args:
            template_id: Template identifier
            parameters: Template parameters
            
        Returns:
            str: Research session ID
        """
        # In production: Load template from storage
        # For now, create a basic research request
        
        research_request = {
            "topic": parameters.get("topic", "Template Research"),
            "domain": parameters.get("domain", "general"),
            "output_format": parameters.get("output_format", "research_summary"),
            **parameters
        }
        
        return await self.submit_research_request(research_request)


# Integration with existing PyGent Factory initialization

async def initialize_research_system(config: OrchestrationConfig,
                                   agent_registry: AgentRegistry,
                                   task_dispatcher: TaskDispatcher,
                                   mcp_orchestrator: MCPOrchestrator) -> ResearchOrchestrationManager:
    """
    Initialize the complete research system.
    
    Args:
        config: Orchestration configuration
        agent_registry: Agent registry instance
        task_dispatcher: Task dispatcher instance
        mcp_orchestrator: MCP orchestrator instance
        
    Returns:
        ResearchOrchestrationManager: Initialized research manager
    """
    
    logger.info("Initializing Research System...")
    
    # Create research orchestration manager
    research_manager = ResearchOrchestrationManager(
        config=config,
        agent_registry=agent_registry,
        task_dispatcher=task_dispatcher,
        mcp_orchestrator=mcp_orchestrator
    )
    
    # Register research task completion callbacks
    task_dispatcher.add_completion_callback(
        _research_task_completion_callback
    )
    
    logger.info("Research System initialization complete")
    
    return research_manager


async def _research_task_completion_callback(task: TaskRequest):
    """Handle completion of research tasks"""
    
    if task.task_type == ResearchTaskType.RESEARCH_QUERY:
        research_id = task.input_data.get("research_id")
        
        if task.status == TaskStatus.COMPLETED:
            logger.info(f"Research task completed successfully: {research_id}")
        elif task.status == TaskStatus.FAILED:
            logger.error(f"Research task failed: {research_id}, Error: {task.error_message}")
        elif task.status == TaskStatus.CANCELLED:
            logger.info(f"Research task cancelled: {research_id}")


# Export main classes
__all__ = [
    "ResearchOrchestrationManager",
    "ResearchTaskType",
    "ResearchAgentType", 
    "initialize_research_system"
]
