"""
Agent Factory - Agent Creation and Management (REFACTORED)

This module implements the Agent Factory pattern for creating, managing, and
orchestrating agents in the PyGent Factory system. It provides a centralized
way to create agents with proper MCP integration and memory management.

REFACTORED to use provider registry instead of direct provider management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type
import uuid
from dataclasses import dataclass

from src.core.agent import BaseAgent, AgentConfig, AgentStatus, AgentError
from src.memory.memory_manager import MemoryManager
from src.mcp.server_registry import MCPServerManager
from src.config.settings import Settings
from src.ai.providers.provider_registry import get_provider_registry

@dataclass
class AgentCreationRequest:
    """Request object for creating agents."""
    agent_type: str
    name: Optional[str] = None
    capabilities: Optional[List[str]] = None
    mcp_tools: Optional[List[str]] = None
    custom_config: Optional[Dict[str, Any]] = None

@dataclass
class AgentCreationResult:
    """Result object for agent creation."""
    success: bool
    agent_id: Optional[str] = None
    agent: Optional[BaseAgent] = None
    error: Optional[str] = None

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Registry for managing active agents."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_types: Dict[str, Type[BaseAgent]] = {}
        self._lock = asyncio.Lock()
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent in the registry."""
        async with self._lock:
            self._agents[agent.agent_id] = agent
            logger.info(f"Registered agent: {agent.agent_id} ({agent.type})")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the registry."""
        async with self._lock:
            if agent_id in self._agents:
                agent = self._agents.pop(agent_id)
                await agent.shutdown()
                logger.info(f"Unregistered agent: {agent_id}")
    
    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    async def list_agents(self, agent_type: Optional[str] = None, 
                         status: Optional[AgentStatus] = None) -> List[BaseAgent]:
        """List agents with optional filtering."""
        agents = list(self._agents.values())
        
        if agent_type:
            agents = [a for a in agents if a.type == agent_type]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        return agents
    
    def register_agent_type(self, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """Register an agent type class."""
        self._agent_types[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
    
    def get_agent_class(self, agent_type: str) -> Optional[Type[BaseAgent]]:
        """Get an agent class by type."""
        return self._agent_types.get(agent_type)
    
    def get_registered_types(self) -> List[str]:
        """Get list of registered agent types."""
        return list(self._agent_types.keys())

class AgentFactory:
    """
    Factory class for creating and managing agents.
    
    The AgentFactory is responsible for:
    - Creating agents with proper configuration
    - Setting up MCP tool integrations
    - Initializing agent memory
    - Managing agent lifecycle
    - Providing agent orchestration capabilities
    
    REFACTORED to use ProviderRegistry for all LLM provider management.
    """
    
    def __init__(self,
                 mcp_manager: Optional[MCPServerManager] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 settings: Optional[Settings] = None):
        """
        Initialize the Agent Factory.

        Args:
            mcp_manager: MCP server manager instance
            memory_manager: Memory manager instance
            settings: Application settings
        """
        self.mcp_manager = mcp_manager
        self.memory_manager = memory_manager
        self.settings = settings or self._create_default_settings()
        self.provider_registry = get_provider_registry()
        self.registry = AgentRegistry()
        self._initialized = False
        
        # Setup default agent types
        self._setup_default_agent_types()
    
    def _create_default_settings(self):
        """Create default settings."""
        class ProductionSettings:
            DEFAULT_AGENT_TIMEOUT = 300
            MAX_CONCURRENT_AGENTS = 10
            AGENT_CLEANUP_INTERVAL = 600
            HEALTH_CHECK_INTERVAL = 60
        return ProductionSettings()
    
    async def initialize(self, 
                        enable_ollama: bool = True,
                        enable_openrouter: bool = True,
                        ollama_config: Optional[Dict[str, Any]] = None,
                        openrouter_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent factory and providers."""
        if not self._initialized:
            # Initialize provider registry
            await self.provider_registry.initialize(
                enable_ollama=enable_ollama,
                enable_openrouter=enable_openrouter,
                ollama_config=ollama_config,
                openrouter_config=openrouter_config
            )
            self._initialized = True
            logger.info("Agent factory initialized")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the factory is initialized."""
        return self._initialized
    
    def _setup_default_agent_types(self) -> None:
        """Set up default agent types."""
        try:
            from ..agents.reasoning_agent import ReasoningAgent
            from ..agents.search_agent import SearchAgent
            from ..agents.general_agent import GeneralAgent
            from ..agents.evolution_agent import EvolutionAgent
            from ..agents.coding_agent import CodingAgent
            from ..agents.research_agent_adapter import ResearchAgentAdapter

            # Register the 6 real agent types
            self.registry.register_agent_type("reasoning", ReasoningAgent)
            self.registry.register_agent_type("search", SearchAgent)
            self.registry.register_agent_type("general", GeneralAgent)
            self.registry.register_agent_type("evolution", EvolutionAgent)
            self.registry.register_agent_type("coding", CodingAgent)
            self.registry.register_agent_type("research", ResearchAgentAdapter)
            
            # Add test-compatible aliases
            self.registry.register_agent_type("basic", GeneralAgent)
            self.registry.register_agent_type("nlp", GeneralAgent)

            logger.info("Registered 8 agent types")

        except ImportError as e:
            logger.error(f"Failed to import agent implementations: {e}")
            self.registry.register_agent_type("general", BaseAgent)
    
    async def create_agent(self, 
                          agent_type: str, 
                          name: Optional[str] = None,
                          capabilities: Optional[List[str]] = None,
                          mcp_tools: Optional[List[str]] = None,
                          custom_config: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """Create a new agent instance."""
        try:
            # Get agent class
            agent_class = self.registry.get_agent_class(agent_type)
            if not agent_class:
                raise AgentError(f"Unknown agent type: {agent_type}")
            
            # Generate agent ID and name
            agent_id = str(uuid.uuid4())
            if not name:
                name = f"{agent_type}_{agent_id[:8]}"
            
            # Create agent configuration
            config = AgentConfig(
                agent_id=agent_id,
                name=name,
                agent_type=agent_type,
                enabled_capabilities=capabilities or [],
                mcp_tools=mcp_tools or [],
                default_timeout=getattr(self.settings, 'DEFAULT_AGENT_TIMEOUT', 300),
                max_retries=3,
                custom_config=custom_config or {}
            )
            
            # Validate model availability
            await self._validate_model_availability(config)

            # Create agent instance
            agent = agent_class(config)

            # Configure MCP tools
            await self._configure_mcp_tools(agent, mcp_tools or [])

            # Initialize memory
            await self._initialize_agent_memory(agent)

            # Initialize agent
            await agent.initialize()
            
            # Register agent
            await self.registry.register_agent(agent)
            
            logger.info(f"Created agent: {agent_id} ({agent_type})")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent {agent_type}: {str(e)}")
            raise AgentError(f"Agent creation failed: {str(e)}")
    
    async def destroy_agent(self, agent_id: str) -> None:
        """Destroy an agent and clean up resources."""
        try:
            agent = await self.registry.get_agent(agent_id)
            if agent:
                # Cleanup memory
                if agent.memory:
                    await self.memory_manager.cleanup_memory_space(agent_id)
                
                # Unregister from registry
                await self.registry.unregister_agent(agent_id)
                
                logger.info(f"Destroyed agent: {agent_id}")
            else:
                logger.warning(f"Agent not found for destruction: {agent_id}")
                
        except Exception as e:
            logger.error(f"Failed to destroy agent {agent_id}: {str(e)}")
            raise AgentError(f"Agent destruction failed: {str(e)}")
    
    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return await self.registry.get_agent(agent_id)
    
    async def list_agents(self, 
                         agent_type: Optional[str] = None,
                         status: Optional[AgentStatus] = None) -> List[BaseAgent]:
        """List agents with optional filtering."""
        return await self.registry.list_agents(agent_type, status)
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status information."""
        agent = await self.registry.get_agent(agent_id)
        return agent.get_status() if agent else None
    
    def register_agent_type(self, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """Register a new agent type."""
        self.registry.register_agent_type(agent_type, agent_class)
    
    def get_available_agent_types(self) -> List[str]:
        """Get list of available agent types."""
        return self.registry.get_registered_types()

    # Provider-related methods (simplified)
    async def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models from all providers."""
        return await self.provider_registry.get_all_models()

    async def get_system_readiness(self) -> Dict[str, Any]:
        """Get comprehensive system readiness information."""
        provider_status = await self.provider_registry.get_system_status()
        
        readiness = {
            "providers": provider_status["providers"],
            "services": {
                "mcp_servers": self.mcp_manager is not None,
                "memory_manager": self.memory_manager is not None,
            },
            "agent_types_supported": {},
            "recommendations": []
        }
        
        # Check which agent types are supported
        ready_providers = await self.provider_registry.get_ready_providers()
        has_llm = len(ready_providers) > 0
        
        for agent_type in ["reasoning", "analysis", "coding", "search", "general", "research"]:
            requirements = self._get_agent_requirements(agent_type)
            supported = True
            
            if requirements["llm_provider"] and not has_llm:
                supported = False
            if requirements["memory"] and not self.memory_manager:
                supported = False
            if requirements["mcp_tools"] and not self.mcp_manager:
                supported = False
            
            readiness["agent_types_supported"][agent_type] = {
                "supported": supported,
                "requirements": requirements
            }
        
        # Simple recommendations
        if not has_llm:
            readiness["recommendations"].append(
                "⚠️  No LLM providers available. Install Ollama locally or configure OpenRouter API key."
            )
        if not self.memory_manager:
            readiness["recommendations"].append(
                "🧠 Memory manager not configured - agents won't have persistent memory."
            )
        if not self.mcp_manager:
            readiness["recommendations"].append(
                "🔧 MCP server manager not configured - agents won't have external tool access."
            )
        
        return readiness

    def _get_agent_requirements(self, agent_type: str) -> Dict[str, Any]:
        """Get requirements for a specific agent type."""
        base_requirements = {
            "llm_provider": False,
            "memory": False,
            "mcp_tools": False,
            "embeddings": False
        }
        
        if agent_type in ['reasoning', 'analysis', 'coding']:
            base_requirements["llm_provider"] = True
            base_requirements["memory"] = True
        elif agent_type == 'research':
            base_requirements["llm_provider"] = True
            base_requirements["memory"] = True
            base_requirements["mcp_tools"] = True
            base_requirements["embeddings"] = True
        elif agent_type == 'search':
            base_requirements["mcp_tools"] = True
        
        return base_requirements

    async def validate_agent_config(self, 
                                   agent_type: str, 
                                   custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent configuration before attempting creation."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Get requirements
        requirements = self._get_agent_requirements(agent_type)
        
        # Check provider configuration
        if requirements["llm_provider"]:
            provider = custom_config.get("provider", "ollama")
            model_name = custom_config.get("model_name", "")
            
            if not model_name:
                validation_result["valid"] = False
                validation_result["errors"].append("model_name is required for this agent type")
            else:
                # Check model availability using provider registry
                availability = await self.provider_registry.is_model_available(model_name, provider)
                if not availability.get(provider, False):
                    validation_result["valid"] = False
                    all_models = await self.provider_registry.get_all_models()
                    available_models = all_models.get(provider, [])
                    validation_result["errors"].append(
                        f"Model '{model_name}' not available on {provider}. Available: {', '.join(available_models[:3])}"
                    )
        
        # Check other requirements
        if requirements["memory"] and not self.memory_manager:
            validation_result["warnings"].append("Memory manager not available")
        
        if requirements["mcp_tools"] and not self.mcp_manager:
            validation_result["warnings"].append("MCP manager not available")
        
        return validation_result

    def get_default_model_config(self, agent_type: str, provider: str = None) -> Dict[str, Any]:
        """Get default model configuration for an agent type."""
        # Default to free OpenRouter model
        default_config = {
            "provider": "openrouter",
            "model_name": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Adjust settings based on agent type
        if agent_type == "reasoning":
            default_config["temperature"] = 0.3
            default_config["max_tokens"] = 2000
        elif agent_type == "coding":
            default_config["temperature"] = 0.1
            default_config["max_tokens"] = 3000
        elif agent_type == "research":
            default_config["temperature"] = 0.2
            default_config["max_tokens"] = 4000
        
        # Override provider if specified
        if provider == "ollama":
            default_config["provider"] = "ollama"
            default_config["model_name"] = "phi4-fast"
        
        return default_config

    async def get_recommended_models(self, include_paid: bool = True) -> Dict[str, List[str]]:
        """Get recommended models - free and paid options."""
        return await self.provider_registry.get_model_recommendations(
            agent_type="general", 
            include_free_only=not include_paid
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the agent factory."""
        agents = await self.list_agents()
        active_agents = [a for a in agents if a.status == AgentStatus.ACTIVE]
        error_agents = [a for a in agents if a.status == AgentStatus.ERROR]
        
        provider_status = await self.provider_registry.get_system_status()
        
        return {
            "total_agents": len(agents),
            "active_agents": len(active_agents),
            "error_agents": len(error_agents),
            "available_types": self.get_available_agent_types(),
            "providers": provider_status["providers"],
            "system_healthy": provider_status["system_healthy"]
        }
    
    async def shutdown(self) -> None:
        """Shutdown the agent factory and all agents."""
        logger.info("Shutting down Agent Factory...")
        
        agents = await self.list_agents()
        for agent in agents:
            try:
                await self.destroy_agent(agent.agent_id)
            except Exception as e:
                logger.error(f"Error shutting down agent {agent.agent_id}: {str(e)}")
        
        await self.provider_registry.shutdown()
        logger.info("Agent Factory shutdown complete")

    async def create_agent_from_request(self, request: AgentCreationRequest) -> AgentCreationResult:
        """Create a new agent instance from a request object."""
        try:
            agent = await self.create_agent(
                agent_type=request.agent_type,
                name=request.name,
                capabilities=request.capabilities,
                mcp_tools=request.mcp_tools,
                custom_config=request.custom_config
            )
            
            return AgentCreationResult(
                success=True,
                agent_id=agent.agent_id,
                agent=agent
            )
            
        except Exception as e:
            logger.error(f"Failed to create agent from request: {str(e)}")
            return AgentCreationResult(
                success=False,
                error=str(e)
            )
    
    async def _validate_model_availability(self, config: AgentConfig) -> None:
        """Validate that required models are available in the chosen provider."""
        provider = config.custom_config.get("provider", "ollama")
        model_name = config.custom_config.get("model_name", "")
        
        # For reasoning/analysis agents, a provider is required
        if config.agent_type in ['reasoning', 'analysis']:
            if not model_name:
                raise AgentError(f"Model name is required for {config.agent_type} agents")
            
            # Check if model is available
            availability = await self.provider_registry.is_model_available(model_name, provider)
            if not availability.get(provider, False):
                ready_providers = await self.provider_registry.get_ready_providers()
                if not ready_providers:
                    raise AgentError("No LLM providers are ready")
                
                # Get available models from the specified provider
                all_models = await self.provider_registry.get_all_models()
                available_models = all_models.get(provider, [])
                
                if not available_models:
                    raise AgentError(f"Provider '{provider}' has no available models")
                else:
                    raise AgentError(f"Model '{model_name}' not available on {provider}. Available: {', '.join(available_models[:3])}")
            
            logger.info(f"Model {model_name} validated for agent {config.agent_id} on provider {provider}")
        
        else:
            # Non-reasoning agents can work without models
            if model_name:
                logger.info(f"Model {model_name} specified for {config.agent_type} agent {config.agent_id}")
            else:
                logger.warning(f"No model specified for agent {config.agent_id}")
    
    async def _configure_mcp_tools(self, agent: BaseAgent, mcp_tools: List[str]) -> None:
        """Configure MCP tools for an agent."""
        # Real implementation would integrate with MCP manager
        # For now, log that this functionality is not yet implemented
        logger.info(f"MCP tool configuration not yet implemented for agent {agent.agent_id}")

    async def _initialize_agent_memory(self, agent: BaseAgent) -> None:
        """Initialize memory for an agent."""
        # Real implementation would integrate with memory manager
        # For now, log that this functionality is not yet implemented
        logger.info(f"Agent memory initialization not yet implemented for agent {agent.agent_id}")
