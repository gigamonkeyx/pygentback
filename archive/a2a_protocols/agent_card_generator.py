from typing import Dict, Any, List
from datetime import datetime

from .models import AgentCard
from ...core.agent_factory import AgentFactory

class AgentCardGenerator:
    """Generate A2A agent cards from PyGent Factory agents"""
    
    def __init__(self, agent_factory: AgentFactory):
        self.agent_factory = agent_factory
    
    def generate_card(self, agent_id: str) -> AgentCard:
        """Generate agent card for specific agent"""
        # Get agent info from factory
        agent_info = self.agent_factory.get_agent_info(agent_id)
        
        if not agent_info:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Extract capabilities from agent
        capabilities = self._extract_capabilities(agent_info)
        
        # Generate card
        card = AgentCard(
            metadata={
                "name": agent_id,
                "displayName": agent_info.get("display_name", agent_id),
                "description": agent_info.get("description", "PyGent Factory Agent"),
                "version": "1.0.0",
                "tags": agent_info.get("tags", []),
                "created": datetime.utcnow().isoformat()
            },
            spec={
                "endpoints": {
                    "a2a": "/a2a/v1"
                },
                "capabilities": capabilities,
                "authentication": {
                    "type": "bearer",
                    "description": "Bearer token authentication"
                },
                "maxConcurrentTasks": agent_info.get("max_concurrent_tasks", 5),
                "supportedContentTypes": [
                    "text/plain",
                    "application/json",
                    "image/png",
                    "image/jpeg"
                ]
            }
        )
        
        return card
    
    def _extract_capabilities(self, agent_info: Dict[str, Any]) -> List[str]:
        """Extract capabilities from agent info"""
        capabilities = []
        
        # Add basic capabilities
        capabilities.append("reasoning")
        capabilities.append("task_execution")
        
        # Add specific capabilities based on agent type
        agent_type = agent_info.get("type", "")
        
        if "evolution" in agent_type.lower():
            capabilities.append("evolution")
            capabilities.append("self_improvement")
        
        if "multi" in agent_type.lower():
            capabilities.append("multi_agent")
            capabilities.append("coordination")
        
        if "learning" in agent_type.lower():
            capabilities.append("learning")
            capabilities.append("adaptation")
        
        # Add memory capabilities
        if agent_info.get("has_memory", True):
            capabilities.append("memory")
            capabilities.append("context_persistence")
        
        # Add tool capabilities
        if agent_info.get("tools", []):
            capabilities.append("tool_use")
            capabilities.extend([f"tool_{tool}" for tool in agent_info["tools"]])
        
        return capabilities
    
    def generate_factory_card(self) -> AgentCard:
        """Generate card for the factory itself"""
        card = AgentCard(
            metadata={
                "name": "pygent-factory",
                "displayName": "PyGent Factory",
                "description": "Advanced AI Agent Factory with Evolution and Multi-Agent Capabilities",
                "version": "1.0.0",
                "tags": ["factory", "evolution", "multi-agent", "ai"],
                "created": datetime.utcnow().isoformat()
            },
            spec={
                "endpoints": {
                    "a2a": "/a2a/v1"
                },
                "capabilities": [
                    "agent_creation",
                    "agent_evolution", 
                    "multi_agent_coordination",
                    "reasoning",
                    "learning",
                    "memory",
                    "tool_use",
                    "self_improvement"
                ],
                "authentication": {
                    "type": "bearer",
                    "description": "Bearer token authentication"
                },
                "maxConcurrentTasks": 10,
                "supportedContentTypes": [
                    "text/plain",
                    "application/json",
                    "image/png",
                    "image/jpeg",
                    "application/pdf"
                ]
            }
        )
        
        return card
