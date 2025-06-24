#!/usr/bin/env python3
"""
Agent Management API Endpoints

Comprehensive API endpoints for agent orchestration, workflow management,
and multi-agent coordination.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from pydantic import BaseModel, validator

# Import real authentication - no fallbacks allowed in production
from ..auth.authorization import get_auth_context, AuthorizationContext, Permission, require_permission

from ..agents.orchestration_manager import orchestration_manager, TaskDefinition
from ..agents.coordination_system import coordination_system, CoordinationPattern
from ..agents.communication_system import communication_system
from ..agents.specialized_agents import ResearchAgent, AnalysisAgent, GenerationAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agent Management"])


# Request/Response Models
class CreateAgentRequest(BaseModel):
    """Create agent request model"""
    agent_type: str
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        valid_types = ["research", "analysis", "generation", "coordination", "monitoring", "custom"]
        if v not in valid_types:
            raise ValueError(f"Agent type must be one of: {', '.join(valid_types)}")
        return v


class TaskSubmissionRequest(BaseModel):
    """Task submission request model"""
    task_id: Optional[str] = None
    task_type: str
    priority: int = 1
    required_capabilities: List[str] = []
    preferred_agent_type: Optional[str] = None
    timeout_seconds: int = 300
    retry_count: int = 3
    dependencies: List[str] = []
    parameters: Dict[str, Any] = {}


class WorkflowCreationRequest(BaseModel):
    """Workflow creation request model"""
    workflow_id: Optional[str] = None
    name: str
    description: str = ""
    coordination_pattern: str = "sequential"
    timeout_seconds: int = 3600
    max_parallel_tasks: int = 5
    enable_fault_tolerance: bool = True
    tasks: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    
    @validator('coordination_pattern')
    def validate_coordination_pattern(cls, v):
        valid_patterns = [pattern.value for pattern in CoordinationPattern]
        if v not in valid_patterns:
            raise ValueError(f"Coordination pattern must be one of: {', '.join(valid_patterns)}")
        return v


class AgentResponse(BaseModel):
    """Agent response model"""
    agent_id: str
    name: str
    agent_type: str
    status: str
    created_at: str
    capabilities: List[str]
    metrics: Dict[str, Any]


class WorkflowResponse(BaseModel):
    """Workflow response model"""
    workflow_id: str
    name: str
    description: str
    status: str
    coordination_pattern: str
    task_count: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@router.post("/create", response_model=Dict[str, str])
@require_permission(Permission.AGENT_CREATE)
async def create_agent(request: CreateAgentRequest, 
                      auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Create a new agent"""
    try:
        # Register agent types if not already registered
        if not hasattr(orchestration_manager, '_types_registered'):
            orchestration_manager.register_agent_type("research", ResearchAgent)
            orchestration_manager.register_agent_type("analysis", AnalysisAgent)
            orchestration_manager.register_agent_type("generation", GenerationAgent)
            orchestration_manager._types_registered = True
        
        # Create agent
        agent_id = await orchestration_manager.create_agent(
            agent_type=request.agent_type,
            name=request.name,
            config=request.config
        )
        
        if not agent_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create agent"
            )
        
        return {
            "agent_id": agent_id,
            "message": f"Agent created successfully",
            "created_by": auth_context.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent creation failed: {str(e)}"
        )


@router.get("/list", response_model=List[AgentResponse])
@require_permission(Permission.AGENT_LIST)
async def list_agents(auth_context: AuthorizationContext = Depends(get_auth_context)):
    """List all agents"""
    try:
        agents = []
        
        for agent_id, agent in orchestration_manager.agents.items():
            agent_status = agent.get_status()
            
            agents.append(AgentResponse(
                agent_id=agent_id,
                name=agent_status["name"],
                agent_type=agent_status["type"],
                status=agent_status["status"],
                created_at=agent_status["created_at"],
                capabilities=agent_status["capabilities"],
                metrics=agent_status["metrics"]
            ))
        
        return agents
        
    except Exception as e:
        logger.error(f"Agent listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.get("/{agent_id}", response_model=AgentResponse)
@require_permission(Permission.AGENT_READ)
async def get_agent(agent_id: str, 
                   auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Get specific agent details"""
    try:
        if agent_id not in orchestration_manager.agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        agent = orchestration_manager.agents[agent_id]
        agent_status = agent.get_status()
        
        return AgentResponse(
            agent_id=agent_id,
            name=agent_status["name"],
            agent_type=agent_status["type"],
            status=agent_status["status"],
            created_at=agent_status["created_at"],
            capabilities=agent_status["capabilities"],
            metrics=agent_status["metrics"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent: {str(e)}"
        )


@router.delete("/{agent_id}")
@require_permission(Permission.AGENT_DELETE)
async def delete_agent(agent_id: str,
                      auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Delete an agent"""
    try:
        success = await orchestration_manager.destroy_agent(agent_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found or could not be deleted"
            )
        
        return {
            "success": True,
            "message": f"Agent {agent_id} deleted successfully",
            "deleted_by": auth_context.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {str(e)}"
        )


@router.post("/tasks/submit")
@require_permission(Permission.TASK_CREATE)
async def submit_task(request: TaskSubmissionRequest,
                     auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Submit a task for execution"""
    try:
        task_id = request.task_id or f"task_{datetime.utcnow().timestamp()}"
        
        task_definition = TaskDefinition(
            task_id=task_id,
            task_type=request.task_type,
            priority=request.priority,
            required_capabilities=request.required_capabilities,
            timeout_seconds=request.timeout_seconds,
            retry_count=request.retry_count,
            dependencies=request.dependencies,
            parameters=request.parameters
        )
        
        success = await orchestration_manager.submit_task(task_definition)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to submit task"
            )
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "message": "Task submitted successfully",
            "submitted_by": auth_context.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task submission failed: {str(e)}"
        )


@router.post("/workflows/create", response_model=Dict[str, str])
@require_permission(Permission.TASK_CREATE)
async def create_workflow(request: WorkflowCreationRequest,
                         auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Create a new workflow"""
    try:
        workflow_definition = {
            "workflow_id": request.workflow_id,
            "name": request.name,
            "description": request.description,
            "coordination_pattern": request.coordination_pattern,
            "timeout_seconds": request.timeout_seconds,
            "max_parallel_tasks": request.max_parallel_tasks,
            "enable_fault_tolerance": request.enable_fault_tolerance,
            "tasks": request.tasks,
            "metadata": request.metadata,
            "created_by": auth_context.user_id
        }
        
        workflow_id = await coordination_system.create_workflow(workflow_definition)
        
        if not workflow_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create workflow"
            )
        
        return {
            "workflow_id": workflow_id,
            "message": "Workflow created successfully",
            "created_by": auth_context.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow creation failed: {str(e)}"
        )


@router.post("/workflows/{workflow_id}/start")
@require_permission(Permission.TASK_EXECUTE)
async def start_workflow(workflow_id: str,
                        auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Start workflow execution"""
    try:
        success = await coordination_system.start_workflow(workflow_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found or could not be started"
            )
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Workflow started successfully",
            "started_by": auth_context.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow start failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workflow: {str(e)}"
        )


@router.get("/workflows/list", response_model=List[WorkflowResponse])
@require_permission(Permission.TASK_LIST)
async def list_workflows(auth_context: AuthorizationContext = Depends(get_auth_context)):
    """List all workflows"""
    try:
        workflows = []
        
        for workflow_id, workflow in coordination_system.workflows.items():
            workflows.append(WorkflowResponse(
                workflow_id=workflow_id,
                name=workflow.name,
                description=workflow.description,
                status=workflow.status.value,
                coordination_pattern=workflow.coordination_pattern.value,
                task_count=len(workflow.tasks),
                created_at=workflow.created_at.isoformat(),
                started_at=workflow.started_at.isoformat() if workflow.started_at else None,
                completed_at=workflow.completed_at.isoformat() if workflow.completed_at else None
            ))
        
        return workflows
        
    except Exception as e:
        logger.error(f"Workflow listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )


@router.get("/status")
@require_permission(Permission.SYSTEM_MONITOR)
async def get_agent_system_status(auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Get comprehensive agent system status"""
    try:
        orchestration_status = orchestration_manager.get_orchestration_status()
        coordination_status = coordination_system.get_coordination_status()
        communication_status = communication_system.get_communication_status()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "orchestration": orchestration_status,
            "coordination": coordination_status,
            "communication": communication_status,
            "overall_status": "healthy" if all([
                orchestration_status["is_running"],
                coordination_status["is_running"],
                communication_status["is_running"]
            ]) else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Agent system status failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/metrics")
@require_permission(Permission.SYSTEM_MONITOR)
async def get_agent_metrics(auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Get agent system metrics"""
    try:
        orchestration_status = orchestration_manager.get_orchestration_status()
        coordination_status = coordination_system.get_coordination_status()
        communication_status = communication_system.get_communication_status()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "orchestration_metrics": orchestration_status["metrics"],
            "coordination_metrics": coordination_status["metrics"],
            "communication_metrics": communication_status["metrics"],
            "summary": {
                "total_agents": orchestration_status["metrics"]["total_agents"],
                "active_agents": orchestration_status["metrics"]["active_agents"],
                "total_workflows": coordination_status["metrics"]["total_workflows"],
                "active_workflows": coordination_status["metrics"]["active_workflows"],
                "total_messages": communication_status["metrics"]["total_messages"],
                "system_efficiency": round(
                    (orchestration_status["metrics"]["completed_tasks"] / 
                     max(orchestration_status["metrics"]["total_tasks"], 1)) * 100, 2
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Agent metrics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent metrics: {str(e)}"
        )


@router.post("/initialize")
@require_permission(Permission.SYSTEM_ADMIN)
async def initialize_agent_system(background_tasks: BackgroundTasks,
                                 auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Initialize the complete agent system"""
    try:
        # Initialize systems in background
        async def initialize_systems():
            try:
                # Initialize orchestration manager
                if not orchestration_manager.is_initialized:
                    await orchestration_manager.initialize()
                
                # Initialize coordination system
                if not coordination_system.is_initialized:
                    await coordination_system.initialize()
                
                # Initialize communication system
                if not communication_system.is_initialized:
                    await communication_system.initialize()
                
                logger.info("Agent system initialization completed")
                
            except Exception as e:
                logger.error(f"Agent system initialization failed: {e}")
        
        background_tasks.add_task(initialize_systems)
        
        return {
            "status": "initializing",
            "message": "Agent system initialization started",
            "initiated_by": auth_context.username
        }
        
    except Exception as e:
        logger.error(f"Agent system initialization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Initialization failed: {str(e)}"
        )


@router.post("/shutdown")
@require_permission(Permission.SYSTEM_ADMIN)
async def shutdown_agent_system(auth_context: AuthorizationContext = Depends(get_auth_context)):
    """Shutdown the agent system"""
    try:
        # Shutdown systems
        await coordination_system.shutdown()
        await communication_system.shutdown()
        await orchestration_manager.shutdown()
        
        return {
            "status": "shutdown",
            "message": "Agent system shutdown completed",
            "shutdown_by": auth_context.username
        }
        
    except Exception as e:
        logger.error(f"Agent system shutdown failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Shutdown failed: {str(e)}"
        )
