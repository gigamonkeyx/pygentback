"""
Agent Management API Routes

This module provides REST API endpoints for agent management including
creation, configuration, execution, and monitoring of agents with persistent
user association.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...core.agent_factory import AgentFactory
from ...security.auth import require_agent_permissions, User
from ...services.agent_service import get_agent_service, AgentService


logger = logging.getLogger(__name__)

router = APIRouter()

# Global agent factory instance (will be set by main.py)
_agent_factory: Optional[AgentFactory] = None

def set_agent_factory(factory: AgentFactory):
    """Set the global agent factory instance"""
    global _agent_factory
    _agent_factory = factory

def get_agent_factory() -> AgentFactory:
    """Get the agent factory dependency"""
    if _agent_factory is None:
        raise HTTPException(status_code=500, detail="Agent factory not initialized")
    return _agent_factory


# Request/Response models
class CreateAgentRequest(BaseModel):
    name: Optional[str] = None
    agent_type: str
    capabilities: List[str] = []
    mcp_tools: List[str] = []
    custom_config: Dict[str, Any] = {}


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    type: str
    status: str
    capabilities: List[str]
    mcp_tools: List[str]
    created_at: datetime
    last_activity: datetime


class ExecuteCapabilityRequest(BaseModel):
    capability: str
    parameters: Dict[str, Any] = {}


class AgentMessageRequest(BaseModel):
    content: Dict[str, Any]
    message_type: str = "request"


@router.post("/", response_model=AgentResponse)
async def create_agent(
    request: CreateAgentRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: User = Depends(require_agent_permissions)
):
    """Create a new agent associated with the current user"""
    try:
        agent_record = await agent_service.create_agent(
            user_id=current_user.id,
            agent_type=request.agent_type,
            name=request.name,
            capabilities=request.capabilities,
            mcp_tools=request.mcp_tools,
            custom_config=request.custom_config
        )
        
        return AgentResponse(
            agent_id=agent_record.id,
            name=agent_record.name,
            type=agent_record.type,
            status=agent_record.status,
            capabilities=agent_record.capabilities,
            mcp_tools=agent_record.config.get('mcp_tools', []),
            created_at=agent_record.created_at,
            last_activity=agent_record.updated_at
        )
        
    except ValueError as e:
        logger.error(f"Invalid request to create agent: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.get("/", response_model=List[AgentResponse])
async def list_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    agent_service: AgentService = Depends(get_agent_service),
    current_user: User = Depends(require_agent_permissions)
):
    """List all agents for the current user with optional filtering"""
    try:
        agents = agent_service.get_user_agents(
            user_id=current_user.id,
            agent_type=agent_type,
            status=status
        )
        
        return [
            AgentResponse(
                agent_id=agent.id,
                name=agent.name,
                type=agent.type,
                status=agent.status,
                capabilities=agent.capabilities,
                mcp_tools=agent.config.get('mcp_tools', []),
                created_at=agent.created_at,
                last_activity=agent.updated_at
            )
            for agent in agents
        ]
        
    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: User = Depends(require_agent_permissions)
):
    """Get agent by ID (only if owned by current user)"""
    try:
        agent = agent_service.get_agent_by_id(agent_id, current_user.id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse(
            agent_id=agent.id,
            name=agent.name,
            type=agent.type,
            status=agent.status,
            capabilities=agent.capabilities,
            mcp_tools=agent.config.get('mcp_tools', []),
            created_at=agent.created_at,
            last_activity=agent.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: User = Depends(require_agent_permissions)
):
    """Delete an agent (only if owned by current user)"""
    try:
        deleted = await agent_service.delete_agent(agent_id, current_user.id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {"message": f"Agent {agent_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")


@router.get("/{agent_id}/status")
async def get_agent_status(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service),
    agent_factory: AgentFactory = Depends(get_agent_factory),
    current_user: User = Depends(require_agent_permissions)
):
    """Get detailed agent status (only if owned by current user)"""
    try:
        # First verify the agent belongs to the user
        agent_record = agent_service.get_agent_by_id(agent_id, current_user.id)
        if not agent_record:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get runtime status from factory
        status = await agent_factory.get_agent_status(agent_id)
        if not status:
            # Return database status if runtime agent not active
            status = {
                "agent_id": agent_id,
                "status": agent_record.status,
                "runtime_active": False
            }
        else:
            status["runtime_active"] = True
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent status {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@router.post("/{agent_id}/execute")
async def execute_capability(
    agent_id: str,
    request: ExecuteCapabilityRequest,
    agent_service: AgentService = Depends(get_agent_service),
    agent_factory: AgentFactory = Depends(get_agent_factory),
    current_user: User = Depends(require_agent_permissions)
):
    """Execute an agent capability (only if owned by current user)"""
    try:
        # First verify the agent belongs to the user
        agent_record = agent_service.get_agent_by_id(agent_id, current_user.id)
        if not agent_record:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get runtime agent
        agent = await agent_factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not active")
        
        # Execute capability
        result = await agent.execute_capability(request.capability, request.parameters)
        
        return {
            "agent_id": agent_id,
            "capability": request.capability,
            "parameters": request.parameters,
            "result": result,
            "executed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute capability on agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute capability: {str(e)}")


@router.post("/{agent_id}/message")
async def send_message(
    agent_id: str,
    request: AgentMessageRequest,
    agent_service: AgentService = Depends(get_agent_service),
    agent_factory: AgentFactory = Depends(get_agent_factory),
    current_user: User = Depends(require_agent_permissions)
):
    """Send a message to an agent (only if owned by current user)"""
    try:
        # First verify the agent belongs to the user
        agent_record = agent_service.get_agent_by_id(agent_id, current_user.id)
        if not agent_record:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get runtime agent
        agent = await agent_factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not active")
        
        # Create and send message
        from ...core.agent import AgentMessage, MessageType
        message = AgentMessage(
            type=MessageType(request.message_type),
            sender=current_user.username,
            recipient=agent_id,
            content=request.content
        )
        
        response = await agent.process_message(message)
        
        return {
            "message_id": response.id,
            "response": response.content,
            "timestamp": response.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send message to agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.get("/{agent_id}/capabilities")
async def get_agent_capabilities(
    agent_id: str,
    agent_service: AgentService = Depends(get_agent_service),
    agent_factory: AgentFactory = Depends(get_agent_factory),
    current_user: User = Depends(require_agent_permissions)
):
    """Get agent capabilities (only if owned by current user)"""
    try:
        # First verify the agent belongs to the user
        agent_record = agent_service.get_agent_by_id(agent_id, current_user.id)
        if not agent_record:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Try to get from runtime agent, fallback to database record
        agent = await agent_factory.get_agent(agent_id)
        if agent:
            capabilities = agent.get_capabilities()
            return {
                "agent_id": agent_id,
                "capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "parameters": cap.parameters,
                        "required_tools": cap.required_tools,
                        "required_resources": cap.required_resources
                    }
                    for cap in capabilities
                ]
            }
        else:
            # Return capabilities from database record
            return {
                "agent_id": agent_id,
                "capabilities": agent_record.capabilities,
                "runtime_active": False
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent capabilities {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")


@router.get("/types/available")
async def get_available_agent_types(
    agent_factory: AgentFactory = Depends(get_agent_factory),
    current_user: User = Depends(require_agent_permissions)
):
    """Get available agent types"""
    try:
        types = agent_factory.get_available_agent_types()
        return {"available_types": types}
        
    except Exception as e:
        logger.error(f"Failed to get available agent types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent types: {str(e)}")


@router.get("/health/summary")
async def get_agents_health_summary(
    agent_factory: AgentFactory = Depends(get_agent_factory),
    current_user: User = Depends(require_agent_permissions)
):
    """Get agents health summary"""
    try:
        health_info = await agent_factory.health_check()
        return health_info
        
    except Exception as e:
        logger.error(f"Failed to get agents health summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get health summary: {str(e)}")
