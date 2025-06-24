"""
Agent Registry and Factory

Registry for managing agent types and factory for creating agent instances.
"""

import logging
import uuid
from typing import Dict, List, Type, Any, Optional, Callable
from datetime import datetime

from .base import ConfigurableAgent
from .specialized import RecipeAgent, TestingAgent
from ..core import AgentCapability

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for managing available agent types and their configurations.
    """
    
    def __init__(self):
        self.agent_types: Dict[str, Type[ConfigurableAgent]] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.capability_index: Dict[str, List[str]] = {}  # capability -> agent_types
        
        # Register built-in agent types
        self._register_builtin_agents()
    
    def _register_builtin_agents(self):
        """Register built-in agent types"""
        self.register_agent_type("recipe_agent", RecipeAgent, {
            "description": "Agent specialized in recipe processing and management",
            "default_capabilities": ["recipe_parsing", "recipe_validation", "recipe_optimization"],
            "resource_requirements": {"memory_mb": 256, "cpu_cores": 1}
        })
        
        self.register_agent_type("testing_agent", TestingAgent, {
            "description": "Agent specialized in testing and validation tasks",
            "default_capabilities": ["test_execution", "test_analysis", "performance_testing"],
            "resource_requirements": {"memory_mb": 512, "cpu_cores": 2}
        })
    
    def register_agent_type(self, agent_type: str, agent_class: Type[ConfigurableAgent], 
                           config: Dict[str, Any]):
        """Register a new agent type"""
        self.agent_types[agent_type] = agent_class
        self.agent_configs[agent_type] = config
        
        # Update capability index
        capabilities = config.get("default_capabilities", [])
        for capability in capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            if agent_type not in self.capability_index[capability]:
                self.capability_index[capability].append(agent_type)
        
        logger.info(f"Registered agent type: {agent_type}")
    
    def unregister_agent_type(self, agent_type: str):
        """Unregister an agent type"""
        if agent_type in self.agent_types:
            # Remove from capability index
            config = self.agent_configs.get(agent_type, {})
            capabilities = config.get("default_capabilities", [])
            for capability in capabilities:
                if capability in self.capability_index:
                    self.capability_index[capability] = [
                        at for at in self.capability_index[capability] if at != agent_type
                    ]
                    if not self.capability_index[capability]:
                        del self.capability_index[capability]
            
            # Remove from registries
            del self.agent_types[agent_type]
            del self.agent_configs[agent_type]
            
            logger.info(f"Unregistered agent type: {agent_type}")
    
    def get_agent_types(self) -> List[str]:
        """Get list of registered agent types"""
        return list(self.agent_types.keys())
    
    def get_agent_config(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """Get configuration for agent type"""
        return self.agent_configs.get(agent_type)
    
    def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agent types that have a specific capability"""
        return self.capability_index.get(capability, [])
    
    def find_agents_by_capabilities(self, capabilities: List[str]) -> List[str]:
        """Find agent types that have all specified capabilities"""
        if not capabilities:
            return []
        
        # Start with agents that have the first capability
        candidate_agents = set(self.find_agents_by_capability(capabilities[0]))
        
        # Intersect with agents that have each subsequent capability
        for capability in capabilities[1:]:
            agents_with_capability = set(self.find_agents_by_capability(capability))
            candidate_agents = candidate_agents.intersection(agents_with_capability)
        
        return list(candidate_agents)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_agent_types": len(self.agent_types),
            "agent_types": list(self.agent_types.keys()),
            "total_capabilities": len(self.capability_index),
            "capabilities": list(self.capability_index.keys()),
            "capability_coverage": {
                capability: len(agent_types) 
                for capability, agent_types in self.capability_index.items()
            }
        }


class AgentFactory:
    """
    Factory for creating agent instances.
    """
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.creation_callbacks: List[Callable] = []
        
        # Creation statistics
        self.stats = {
            "agents_created": 0,
            "creation_failures": 0,
            "agents_by_type": {}
        }
    
    def create_agent(self, agent_type: str, agent_id: Optional[str] = None, 
                    name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> ConfigurableAgent:
        """Create an agent instance"""
        try:
            # Validate agent type
            if agent_type not in self.registry.agent_types:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Generate ID and name if not provided
            if agent_id is None:
                agent_id = str(uuid.uuid4())
            
            if name is None:
                name = f"{agent_type}_{agent_id[:8]}"
            
            # Get agent class
            agent_class = self.registry.agent_types[agent_type]
            
            # Create agent instance
            agent = agent_class(agent_id, name)
            
            # Apply configuration
            if config:
                agent.config.update(config)
            
            # Apply default configuration from registry
            default_config = self.registry.get_agent_config(agent_type)
            if default_config:
                # Merge configs (provided config takes precedence)
                merged_config = default_config.copy()
                merged_config.update(agent.config)
                agent.config = merged_config
            
            # Update statistics
            self.stats["agents_created"] += 1
            if agent_type not in self.stats["agents_by_type"]:
                self.stats["agents_by_type"][agent_type] = 0
            self.stats["agents_by_type"][agent_type] += 1
            
            # Call creation callbacks
            for callback in self.creation_callbacks:
                try:
                    callback(agent)
                except Exception as e:
                    logger.warning(f"Agent creation callback failed: {e}")
            
            logger.info(f"Created agent {name} of type {agent_type}")
            return agent
            
        except Exception as e:
            self.stats["creation_failures"] += 1
            logger.error(f"Failed to create agent of type {agent_type}: {e}")
            raise
    
    def create_agents_by_capability(self, capability: str, count: int = 1, 
                                  config: Optional[Dict[str, Any]] = None) -> List[ConfigurableAgent]:
        """Create agents that have a specific capability"""
        suitable_types = self.registry.find_agents_by_capability(capability)
        
        if not suitable_types:
            raise ValueError(f"No agent types found with capability: {capability}")
        
        # Use the first suitable type (could be enhanced with selection logic)
        agent_type = suitable_types[0]
        
        agents = []
        for i in range(count):
            agent = self.create_agent(agent_type, config=config)
            agents.append(agent)
        
        return agents
    
    def create_agents_by_capabilities(self, capabilities: List[str], count: int = 1,
                                    config: Optional[Dict[str, Any]] = None) -> List[ConfigurableAgent]:
        """Create agents that have all specified capabilities"""
        suitable_types = self.registry.find_agents_by_capabilities(capabilities)
        
        if not suitable_types:
            raise ValueError(f"No agent types found with all capabilities: {capabilities}")
        
        # Use the first suitable type
        agent_type = suitable_types[0]
        
        agents = []
        for i in range(count):
            agent = self.create_agent(agent_type, config=config)
            agents.append(agent)
        
        return agents
    
    def create_agent_pool(self, pool_config: Dict[str, Any]) -> List[ConfigurableAgent]:
        """Create a pool of agents based on configuration"""
        agents = []
        
        for agent_spec in pool_config.get("agents", []):
            agent_type = agent_spec.get("type")
            count = agent_spec.get("count", 1)
            config = agent_spec.get("config", {})
            
            for i in range(count):
                agent = self.create_agent(agent_type, config=config)
                agents.append(agent)
        
        return agents
    
    def add_creation_callback(self, callback: Callable):
        """Add callback to be called when agents are created"""
        self.creation_callbacks.append(callback)
    
    def get_creation_stats(self) -> Dict[str, Any]:
        """Get agent creation statistics"""
        return self.stats.copy()


class AgentTemplate:
    """
    Template for creating agents with predefined configurations.
    """
    
    def __init__(self, name: str, agent_type: str, config: Dict[str, Any], 
                 description: str = ""):
        self.name = name
        self.agent_type = agent_type
        self.config = config
        self.description = description
        self.created_at = datetime.utcnow()
        self.usage_count = 0
    
    def create_agent(self, factory: AgentFactory, agent_id: Optional[str] = None,
                    name: Optional[str] = None, additional_config: Optional[Dict[str, Any]] = None) -> ConfigurableAgent:
        """Create agent from template"""
        # Merge configurations
        final_config = self.config.copy()
        if additional_config:
            final_config.update(additional_config)
        
        # Create agent
        agent = factory.create_agent(
            self.agent_type, 
            agent_id=agent_id, 
            name=name or f"{self.name}_{uuid.uuid4().hex[:8]}", 
            config=final_config
        )
        
        self.usage_count += 1
        return agent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "config": self.config,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "usage_count": self.usage_count
        }


class TemplateManager:
    """
    Manager for agent templates.
    """
    
    def __init__(self):
        self.templates: Dict[str, AgentTemplate] = {}
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default agent templates"""
        # Recipe processing template
        self.add_template(AgentTemplate(
            name="recipe_processor",
            agent_type="recipe_agent",
            config={
                "max_recipe_complexity": 15,
                "enable_optimization": True,
                "validation_strict_mode": True
            },
            description="High-performance recipe processing agent"
        ))
        
        # Testing template
        self.add_template(AgentTemplate(
            name="test_runner",
            agent_type="testing_agent",
            config={
                "parallel_test_execution": True,
                "max_parallel_tests": 10,
                "test_timeout_seconds": 120
            },
            description="Parallel test execution agent"
        ))
        
        # Lightweight testing template
        self.add_template(AgentTemplate(
            name="quick_tester",
            agent_type="testing_agent",
            config={
                "parallel_test_execution": False,
                "max_test_duration_seconds": 60,
                "generate_detailed_reports": False
            },
            description="Quick testing agent for simple validation"
        ))
    
    def add_template(self, template: AgentTemplate):
        """Add agent template"""
        self.templates[template.name] = template
        logger.info(f"Added agent template: {template.name}")
    
    def remove_template(self, template_name: str):
        """Remove agent template"""
        if template_name in self.templates:
            del self.templates[template_name]
            logger.info(f"Removed agent template: {template_name}")
    
    def get_template(self, template_name: str) -> Optional[AgentTemplate]:
        """Get agent template by name"""
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """List available template names"""
        return list(self.templates.keys())
    
    def create_agent_from_template(self, template_name: str, factory: AgentFactory,
                                 agent_id: Optional[str] = None, name: Optional[str] = None,
                                 additional_config: Optional[Dict[str, Any]] = None) -> ConfigurableAgent:
        """Create agent from template"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        return template.create_agent(factory, agent_id, name, additional_config)
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get template usage statistics"""
        return {
            "total_templates": len(self.templates),
            "template_names": list(self.templates.keys()),
            "usage_stats": {
                name: template.usage_count 
                for name, template in self.templates.items()
            }
        }
