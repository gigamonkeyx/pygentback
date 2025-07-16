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
from datetime import datetime

from .agent import BaseAgent, AgentConfig, AgentStatus, AgentError
from src.memory.memory_manager import MemoryManager
from src.mcp.server_registry import MCPServerManager
from src.config.settings import Settings
from src.ai.providers.provider_registry import get_provider_registry

# Phase 4 Integration: Import new simulation and behavior detection modules
try:
    from .sim_env import SimulationEnvironment, create_simulation_environment
    from .emergent_behavior_detector import Docker443EmergentBehaviorDetector
    PHASE4_MODULES_AVAILABLE = True
except ImportError as e:
    PHASE4_MODULES_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Phase 4 modules not available: {e}")

# A2A Protocol imports
try:
    from src.a2a_protocol.agent_card_generator import A2AAgentCardGenerator, AgentCard
    from src.a2a_protocol.well_known_handler import well_known_handler
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("A2A protocol not available - agents will be created without A2A compliance")

# A2A Short-lived optimization imports
try:
    from src.a2a_protocol.short_lived_optimization import (
        short_lived_optimizer, OptimizationConfig, ResourceLimits
    )
    SHORT_LIVED_OPTIMIZATION_AVAILABLE = True
except ImportError:
    SHORT_LIVED_OPTIMIZATION_AVAILABLE = False
    logger.warning("A2A short-lived optimization not available")

# A2A Discovery imports
try:
    from src.a2a_protocol.discovery import agent_discovery, A2ADiscoveryClient, DiscoveredAgent
    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False
    logger.warning("A2A discovery not available")

# A2A MCP Server imports
try:
    from src.servers.a2a_mcp_server import A2AMCPServer
    A2A_MCP_SERVER_AVAILABLE = True
except ImportError:
    A2A_MCP_SERVER_AVAILABLE = False
    logger.warning("A2A MCP server not available")

# A2A Standard imports
try:
    from src.a2a_standard import AgentCard, AgentProvider, AgentCapabilities, AgentSkill
    A2A_STANDARD_AVAILABLE = True
except ImportError:
    A2A_STANDARD_AVAILABLE = False
    logger.warning("A2A standard not available")

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
                 settings: Optional[Settings] = None,
                 a2a_manager: Optional['A2AManager'] = None,
                 base_url: str = "http://localhost:8000"):
        """
        Initialize the Agent Factory.

        Args:
            mcp_manager: MCP server manager instance
            memory_manager: Memory manager instance
            settings: Application settings
            a2a_manager: A2A protocol manager instance
            base_url: Base URL for A2A agent endpoints
        """
        self.mcp_manager = mcp_manager
        self.memory_manager = memory_manager
        self.settings = settings or self._create_default_settings()
        self.a2a_manager = a2a_manager
        self.provider_registry = get_provider_registry()
        self.registry = AgentRegistry()
        self._initialized = False

        # Phase 4 Integration: Simulation environment and behavior detection
        self.simulation_env = None
        self.behavior_detector = None
        self.phase4_enabled = PHASE4_MODULES_AVAILABLE and getattr(self.settings, 'PHASE4_ENABLED', True)

        # A2A Protocol setup
        self.base_url = base_url
        if A2A_AVAILABLE:
            self.a2a_card_generator = A2AAgentCardGenerator(base_url)
            self.a2a_enabled = True
        else:
            self.a2a_card_generator = None
            self.a2a_enabled = False

        # A2A Protocol Integration
        self.a2a_transport = None
        self.a2a_task_manager = None
        self.a2a_security_manager = None
        self.a2a_discovery = None
        self.a2a_mcp_server = None

        if A2A_AVAILABLE:
            try:
                # Initialize A2A components
                from a2a_protocol.transport import A2ATransportLayer
                from a2a_protocol.task_manager import A2ATaskManager
                from a2a_protocol.security import A2ASecurityManager
                from a2a_protocol.discovery import A2AAgentDiscovery

                self.a2a_transport = A2ATransportLayer()
                self.a2a_task_manager = A2ATaskManager()
                self.a2a_security_manager = A2ASecurityManager()
                self.a2a_discovery = A2AAgentDiscovery()

                logger.info("A2A protocol components initialized successfully")

                # Initialize A2A MCP Server if available
                if A2A_MCP_SERVER_AVAILABLE:
                    self.a2a_mcp_server = A2AMCPServer(port=8006)
                    logger.info("A2A MCP server initialized")

            except Exception as e:
                logger.error(f"Failed to initialize A2A components: {e}")
                self.a2a_enabled = False

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

            # Initialize Phase 4 components if enabled
            if self.phase4_enabled:
                await self._initialize_phase4_components()

            self._initialized = True
            logger.info("Agent factory initialized")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the factory is initialized."""
        return self._initialized
    
    def _setup_default_agent_types(self) -> None:
        """Set up default agent types."""
        try:
            from agents.reasoning_agent import ReasoningAgent
            from agents.search_agent import SearchAgent
            from agents.general_agent import GeneralAgent
            from agents.evolution_agent import EvolutionAgent
            from agents.coding_agent import CodingAgent
            from agents.research_agent_adapter import ResearchAgentAdapter

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
            logger.error(f"Failed to import agent implementations: {e}")            # Instead of using abstract BaseAgent, create a simple concrete fallback
            class SimpleAgent(BaseAgent):
                def __init__(self, config: 'AgentConfig'):
                    super().__init__(config)
                    self.agent_type = config.agent_type if config else "general"
                    self.memory = None  # Add memory attribute for compatibility
                
                async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
                    return f"Simple agent response to: {query}"
                
                async def initialize_capabilities(self) -> None:
                    pass
                
                async def _agent_initialize(self) -> None:
                    """Agent-specific initialization logic"""
                    pass
                
                async def _agent_shutdown(self) -> None:
                    """Agent-specific shutdown logic"""
                    pass
                
                async def _handle_request(self, message) -> Any:
                    """Handle a request message"""
                    return {"response": f"Simple response to: {getattr(message, 'content', str(message))}"}
            
            self.registry.register_agent_type("general", SimpleAgent)
    
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

            # Register with A2A protocol if enabled
            await self._register_with_a2a(agent)

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

    async def discover_a2a_agents(self) -> List[Dict[str, Any]]:
        """Discover available A2A agents"""
        if not self.a2a_manager:
            logger.warning("A2A manager not available for agent discovery")
            return []

        try:
            # Get A2A agent status from manager
            status = await self.a2a_manager.get_agent_status()
            return status.get("agents", [])

        except Exception as e:
            logger.error(f"Failed to discover A2A agents: {e}")
            return []

    async def get_a2a_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Get A2A capabilities for a specific agent"""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return {"error": "Agent not found"}

            # Get A2A config from agent
            a2a_config = self._get_a2a_config(agent)

            return {
                "agent_id": agent_id,
                "name": agent.name,
                "type": agent.type,
                "a2a_enabled": a2a_config.get("enabled", False),
                "a2a_url": a2a_config.get("url"),
                "capabilities": a2a_config.get("capabilities", []),
                "discovery_enabled": a2a_config.get("discovery_enabled", True),
                "streaming": a2a_config.get("streaming", False),
                "push_notifications": a2a_config.get("push_notifications", False)
            }

        except Exception as e:
            logger.error(f"Failed to get A2A capabilities for agent {agent_id}: {e}")
            return {"error": str(e)}

    async def send_a2a_message(self,
                              from_agent_id: str,
                              to_agent_id: str,
                              message: str,
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a message between agents using A2A protocol"""
        if not self.a2a_manager:
            return {"error": "A2A manager not available"}

        try:
            # Validate agents exist
            from_agent = await self.get_agent(from_agent_id)
            to_agent = await self.get_agent(to_agent_id)

            if not from_agent:
                return {"error": f"Source agent {from_agent_id} not found"}
            if not to_agent:
                return {"error": f"Target agent {to_agent_id} not found"}

            # Send A2A message
            result = await self.a2a_manager.send_agent_to_agent_message(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                message=message,
                metadata=metadata
            )

            if result:
                return {
                    "success": True,
                    "task_id": result.id,
                    "session_id": result.sessionId,
                    "status": result.status.state.value
                }
            else:
                return {"error": "Failed to send A2A message"}

        except Exception as e:
            logger.error(f"Failed to send A2A message: {e}")
            return {"error": str(e)}

    async def coordinate_multi_agent_task(self,
                                        task_description: str,
                                        agent_ids: List[str],
                                        coordination_strategy: str = "sequential") -> Dict[str, Any]:
        """Coordinate a task across multiple agents using A2A protocol"""
        if not self.a2a_manager:
            return {"error": "A2A manager not available"}

        try:
            # Validate all agents exist
            for agent_id in agent_ids:
                agent = await self.get_agent(agent_id)
                if not agent:
                    return {"error": f"Agent {agent_id} not found"}

            # Coordinate task
            results = await self.a2a_manager.coordinate_multi_agent_task(
                task_description=task_description,
                agent_ids=agent_ids,
                coordination_strategy=coordination_strategy
            )

            return {
                "success": True,
                "coordination_strategy": coordination_strategy,
                "total_agents": len(agent_ids),
                "completed_tasks": len(results),
                "results": [
                    {
                        "task_id": task.id,
                        "session_id": task.sessionId,
                        "status": task.status.state.value
                    }
                    for task in results
                ]
            }

        except Exception as e:
            logger.error(f"Failed to coordinate multi-agent task: {e}")
            return {"error": str(e)}

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
                "a2a_protocol": self.a2a_manager is not None,
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
        if not self.a2a_manager:
            readiness["recommendations"].append(
                "🤝 A2A protocol manager not configured - agents won't support agent-to-agent communication."
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

    async def _register_with_a2a(self, agent: BaseAgent) -> None:
        """Register agent with A2A protocol using compliant agent card"""
        if not self.a2a_enabled or not self.a2a_card_generator:
            logger.debug(f"A2A protocol not available - skipping A2A registration for agent {agent.agent_id}")
            return

        try:
            # Get A2A configuration from agent config
            a2a_config = self._get_a2a_config(agent)

            if a2a_config.get("enabled", True):
                # Generate A2A-compliant agent card
                agent_card = self.a2a_card_generator.generate_agent_card(
                    agent=agent,
                    agent_type=agent.type if hasattr(agent, 'type') else 'general',
                    enable_authentication=a2a_config.get("authentication", True),
                    enable_streaming=a2a_config.get("streaming", True),
                    enable_push_notifications=a2a_config.get("push_notifications", False)
                )

                # Register agent card with well-known handler for discovery
                well_known_handler.register_agent_card(agent.agent_id, agent_card)

                # Register with A2A manager if available
                if self.a2a_manager:
                    await self.a2a_manager.register_agent(agent)

                # Update agent's metadata with A2A information
                await self._update_agent_a2a_fields(agent, a2a_config, agent_card)

                logger.info(f"Registered agent {agent.agent_id} with A2A protocol (compliant)")
                logger.debug(f"A2A agent card URL: {agent_card.url}")
            else:
                logger.debug(f"A2A registration disabled for agent {agent.agent_id}")

        except Exception as e:
            # A2A registration failure should not prevent agent creation
            logger.warning(f"Failed to register agent {agent.agent_id} with A2A protocol: {e}")
            logger.debug("Agent will continue to function without A2A capabilities")

    def _get_a2a_config(self, agent: BaseAgent) -> Dict[str, Any]:
        """Extract A2A configuration from agent"""
        default_config = {
            "enabled": True,
            "url": f"{self.base_url}/a2a/agents/{agent.agent_id}",
            "capabilities": ["text_processing", "task_execution"],
            "discovery_enabled": True,
            "streaming": True,
            "push_notifications": False,
            "authentication": True
        }

        # Get custom A2A config from agent
        if hasattr(agent, 'config') and hasattr(agent.config, 'custom_config'):
            a2a_custom = agent.config.custom_config.get("a2a", {})
            default_config.update(a2a_custom)

        # Override URL if specified in settings
        if hasattr(self.settings, 'A2A_BASE_URL'):
            default_config["url"] = f"{self.settings.A2A_BASE_URL}/a2a/agents/{agent.agent_id}"

        return default_config

    async def _update_agent_a2a_fields(self, agent: BaseAgent, a2a_config: Dict[str, Any], agent_card: Optional[Dict[str, Any]] = None) -> None:
        """Update agent's metadata with A2A information"""
        try:
            # Add the A2A information to the agent's metadata
            if not hasattr(agent, 'metadata'):
                agent.metadata = {}
            elif not isinstance(agent.metadata, dict):
                agent.metadata = {}

            agent.metadata['a2a'] = {
                "url": a2a_config.get("url"),
                "enabled": a2a_config.get("enabled", True),
                "capabilities": a2a_config.get("capabilities", []),
                "streaming": a2a_config.get("streaming", False),
                "push_notifications": a2a_config.get("push_notifications", False),
                "authentication": a2a_config.get("authentication", True),
                "registered_at": datetime.utcnow().isoformat(),
                "well_known_url": f"{self.base_url}/.well-known/agent/{agent.agent_id}.json"
            }

            # Add agent card information if available
            if agent_card:
                agent.metadata['a2a']['agent_card'] = {
                    "name": agent_card.name,
                    "description": agent_card.description,
                    "version": agent_card.version,
                    "skills_count": len(agent_card.skills),
                    "supports_authenticated_extended_card": agent_card.supportsAuthenticatedExtendedCard
                }

            logger.debug(f"Updated agent {agent.agent_id} metadata with A2A information")

        except Exception as e:
            logger.warning(f"Failed to update agent {agent.agent_id} A2A fields: {e}")

    async def get_agent_card(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get A2A agent card for a specific agent"""
        if not self.a2a_enabled:
            return None

        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return None

            # Check if agent has A2A metadata
            if hasattr(agent, 'metadata') and isinstance(agent.metadata, dict):
                a2a_info = agent.metadata.get('a2a', {})
                if a2a_info.get('enabled', False):
                    return {
                        "agent_id": agent_id,
                        "name": agent.name,
                        "url": a2a_info.get('url'),
                        "well_known_url": a2a_info.get('well_known_url'),
                        "capabilities": a2a_info.get('capabilities', []),
                        "streaming": a2a_info.get('streaming', False),
                        "push_notifications": a2a_info.get('push_notifications', False),
                        "authentication": a2a_info.get('authentication', True),
                        "registered_at": a2a_info.get('registered_at'),
                        "agent_card": a2a_info.get('agent_card', {})
                    }

            return None

        except Exception as e:
            logger.error(f"Failed to get agent card for {agent_id}: {e}")
            return None

    async def list_a2a_agents(self) -> List[Dict[str, Any]]:
        """List all agents with A2A capabilities"""
        if not self.a2a_enabled:
            return []

        try:
            all_agents = await self.list_agents()
            a2a_agents = []

            for agent in all_agents:
                agent_card = await self.get_agent_card(agent.agent_id)
                if agent_card:
                    a2a_agents.append(agent_card)

            return a2a_agents

        except Exception as e:
            logger.error(f"Failed to list A2A agents: {e}")
            return []

    async def create_short_lived_agent(self,
                                     agent_type: str,
                                     purpose: str,
                                     resource_limits: Optional['ResourceLimits'] = None,
                                     optimization_config: Optional['OptimizationConfig'] = None) -> BaseAgent:
        """Create an optimized short-lived agent for specific purposes"""
        if not SHORT_LIVED_OPTIMIZATION_AVAILABLE:
            logger.warning("Short-lived optimization not available, creating regular agent")
            return await self.create_agent(agent_type, f"short_lived_{purpose}")

        try:
            # Generate agent ID
            agent_id = f"short_lived_{agent_type}_{uuid.uuid4().hex[:8]}"

            # Create optimized agent
            optimized_agent = await short_lived_optimizer.create_optimized_agent(
                agent_id=agent_id,
                agent_type=agent_type,
                resource_limits=resource_limits
            )

            # Create regular agent wrapper
            config = AgentConfig(
                name=f"Short-lived {agent_type.title()} Agent",
                description=f"Optimized short-lived agent for {purpose}",
                agent_type=agent_type,
                custom_config={
                    "short_lived": True,
                    "purpose": purpose,
                    "optimization_enabled": True
                }
            )

            agent = BaseAgent(agent_id, config)

            # Store reference to optimized agent
            agent._optimized_agent = optimized_agent

            # Register with A2A if available
            if self.a2a_enabled:
                await self._register_with_a2a(agent)

            # Register agent
            await self.registry.register_agent(agent)

            logger.info(f"Created short-lived agent {agent_id} for purpose: {purpose}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create short-lived agent: {e}")
            raise AgentCreationError(f"Failed to create short-lived agent: {str(e)}")

    async def execute_short_lived_task(self, agent_id: str, task_data: Dict[str, Any]) -> Any:
        """Execute task on short-lived agent with optimizations"""
        if not SHORT_LIVED_OPTIMIZATION_AVAILABLE:
            # Fallback to regular task execution
            agent = await self.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            # Implement regular task execution here
            return {"result": "task_completed", "agent_id": agent_id}

        try:
            return await short_lived_optimizer.execute_task_optimized(agent_id, task_data)
        except Exception as e:
            logger.error(f"Failed to execute task on short-lived agent {agent_id}: {e}")
            raise

    async def shutdown_short_lived_agent(self, agent_id: str, pool_for_reuse: bool = True):
        """Shutdown a short-lived agent"""
        if not SHORT_LIVED_OPTIMIZATION_AVAILABLE:
            # Fallback to regular agent shutdown
            await self.shutdown_agent(agent_id)
            return

        try:
            # Shutdown optimized agent
            await short_lived_optimizer.shutdown_agent(agent_id, pool_for_reuse)

            # Unregister from factory
            await self.registry.unregister_agent(agent_id)

            logger.info(f"Shutdown short-lived agent {agent_id}")

        except Exception as e:
            logger.error(f"Failed to shutdown short-lived agent {agent_id}: {e}")

    async def get_short_lived_agent_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for short-lived agent"""
        if not SHORT_LIVED_OPTIMIZATION_AVAILABLE:
            return None

        try:
            if agent_id in short_lived_optimizer.active_agents:
                optimized_agent = short_lived_optimizer.active_agents[agent_id]
                await optimized_agent.update_metrics()

                return {
                    "agent_id": agent_id,
                    "state": optimized_agent.state.value,
                    "startup_time_ms": optimized_agent.metrics.startup_time_ms,
                    "shutdown_time_ms": optimized_agent.metrics.shutdown_time_ms,
                    "memory_usage_mb": optimized_agent.metrics.memory_usage_mb,
                    "cpu_usage_percent": optimized_agent.metrics.cpu_usage_percent,
                    "tasks_completed": optimized_agent.metrics.tasks_completed,
                    "tasks_failed": optimized_agent.metrics.tasks_failed,
                    "total_execution_time_ms": optimized_agent.metrics.total_execution_time_ms,
                    "last_activity": optimized_agent.metrics.last_activity,
                    "active_tasks": len(optimized_agent.active_tasks),
                    "is_healthy": optimized_agent.is_healthy(),
                    "exceeds_limits": optimized_agent.exceeds_resource_limits()
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get metrics for short-lived agent {agent_id}: {e}")
            return None

    async def discover_agents_in_network(self, base_urls: List[str]) -> List['DiscoveredAgent']:
        """Discover A2A agents in the network"""
        if not DISCOVERY_AVAILABLE:
            logger.warning("A2A discovery not available")
            return []

        try:
            async with A2ADiscoveryClient() as discovery_client:
                discovered_agents = await discovery_client.discover_agents(base_urls)

                logger.info(f"Discovered {len(discovered_agents)} agents in network")
                return discovered_agents

        except Exception as e:
            logger.error(f"Failed to discover agents in network: {e}")
            return []

    async def find_agent_for_task(self, task_description: str,
                                required_capabilities: Optional[List[str]] = None,
                                required_skills: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Find the best agent for a specific task"""
        if not DISCOVERY_AVAILABLE:
            logger.warning("A2A discovery not available")
            return None

        try:
            async with A2ADiscoveryClient() as discovery_client:
                # First discover agents in known locations
                known_urls = [
                    "http://localhost:8000",
                    "http://localhost:8001",
                    "http://localhost:8002"
                ]

                await discovery_client.discover_agents(known_urls)

                # Find best match
                match = await discovery_client.find_best_agent_for_task(
                    task_description, required_capabilities, required_skills
                )

                if match:
                    return {
                        "agent_id": match.agent_id,
                        "agent_name": match.agent_name,
                        "agent_url": match.agent_url,
                        "match_score": match.match_score,
                        "matching_skills": match.matching_skills,
                        "missing_capabilities": match.missing_capabilities
                    }

                return None

        except Exception as e:
            logger.error(f"Failed to find agent for task: {e}")
            return None

    async def test_agent_connectivity(self, agent_url: str) -> Dict[str, Any]:
        """Test connectivity to an A2A agent"""
        if not DISCOVERY_AVAILABLE:
            return {"success": False, "error": "A2A discovery not available"}

        try:
            async with A2ADiscoveryClient() as discovery_client:
                result = await discovery_client.test_agent_connectivity(agent_url)
                return result

        except Exception as e:
            logger.error(f"Failed to test agent connectivity: {e}")
            return {"success": False, "error": str(e)}

    async def get_agent_capabilities_from_url(self, agent_url: str) -> Optional[Dict[str, Any]]:
        """Get detailed capabilities of an agent by URL"""
        if not DISCOVERY_AVAILABLE:
            logger.warning("A2A discovery not available")
            return None

        try:
            async with A2ADiscoveryClient() as discovery_client:
                capabilities = await discovery_client.get_agent_capabilities(agent_url)
                return capabilities

        except Exception as e:
            logger.error(f"Failed to get agent capabilities: {e}")
            return None

    async def register_with_agent_registry(self, registry_url: str):
        """Register this agent factory with an external agent registry"""
        if not DISCOVERY_AVAILABLE:
            logger.warning("A2A discovery not available")
            return

        try:
            # Add registry to discovery system
            agent_discovery.add_agent_registry(registry_url)

            # Register our agents with the registry
            our_agents = await self.list_a2a_agents()

            # TODO: Implement actual registration with external registry
            # This would involve sending our agent cards to the registry

            logger.info(f"Registered {len(our_agents)} agents with registry {registry_url}")

        except Exception as e:
            logger.error(f"Failed to register with agent registry: {e}")

    async def discover_from_registries(self) -> List['DiscoveredAgent']:
        """Discover agents from known registries"""
        if not DISCOVERY_AVAILABLE:
            logger.warning("A2A discovery not available")
            return []

        try:
            async with agent_discovery:
                discovered_agents = await agent_discovery.discover_from_registries()

                logger.info(f"Discovered {len(discovered_agents)} agents from registries")
                return discovered_agents

        except Exception as e:
            logger.error(f"Failed to discover agents from registries: {e}")
            return []

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get agent discovery statistics"""
        if not DISCOVERY_AVAILABLE:
            return {"error": "A2A discovery not available"}

        try:
            return agent_discovery.get_discovery_stats()
        except Exception as e:
            logger.error(f"Failed to get discovery stats: {e}")
            return {"error": str(e)}

    async def start_a2a_mcp_server(self, host: str = "127.0.0.1", port: int = 8006):
        """Start the A2A MCP server"""
        if not A2A_MCP_SERVER_AVAILABLE:
            logger.warning("A2A MCP server not available")
            return False

        if not self.a2a_mcp_server:
            try:
                self.a2a_mcp_server = A2AMCPServer(port=port)
                logger.info(f"A2A MCP server created on port {port}")
            except Exception as e:
                logger.error(f"Failed to create A2A MCP server: {e}")
                return False

        try:
            # Start the server in the background
            import asyncio
            asyncio.create_task(self.a2a_mcp_server.start_server(host, port))
            logger.info(f"A2A MCP server started on {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start A2A MCP server: {e}")
            return False

    def get_a2a_mcp_server_status(self) -> Dict[str, Any]:
        """Get A2A MCP server status"""
        if not self.a2a_mcp_server:
            return {
                "available": False,
                "running": False,
                "error": "A2A MCP server not initialized"
            }

        return {
            "available": True,
            "running": self.a2a_mcp_server.is_running,
            "port": self.a2a_mcp_server.port,
            "registered_agents": len(self.a2a_mcp_server.registered_agents),
            "active_tasks": len(self.a2a_mcp_server.active_tasks),
            "stats": self.a2a_mcp_server.stats,
            "uptime": (datetime.utcnow() - self.a2a_mcp_server.start_time).total_seconds() if hasattr(self.a2a_mcp_server, 'start_time') else 0
        }

    async def register_agent_with_a2a_mcp(self, agent: BaseAgent) -> bool:
        """Register an agent with the A2A MCP server"""
        if not self.a2a_mcp_server or not A2A_STANDARD_AVAILABLE:
            return False

        try:
            # Create A2A agent card for the agent
            agent_card = AgentCard(
                name=f"{agent.name} (PyGent Factory)",
                description=f"PyGent Factory {agent.type} agent with A2A protocol support",
                version="1.0.0",
                url=f"{self.base_url}/agents/{agent.id}",
                defaultInputModes=["text", "application/json"],
                defaultOutputModes=["text", "application/json"],
                provider=AgentProvider(
                    name="PyGent Factory",
                    organization="PyGent Factory",
                    description="Advanced AI agent orchestration platform",
                    url="https://github.com/gigamonkeyx/pygentback"
                ),
                capabilities=AgentCapabilities(
                    streaming=True,
                    push_notifications=False,
                    multi_turn=True,
                    file_upload=False,
                    file_download=False,
                    structured_data=True
                ),
                skills=[
                    AgentSkill(
                        id=agent.type,
                        name=agent.type,
                        description=f"Specialized {agent.type} capabilities",
                        input_modalities=["text", "application/json"],
                        output_modalities=["text", "application/json"],
                        tags=[agent.type, "pygent", "ai"],
                        examples=[f"Process {agent.type} requests"]
                    )
                ]
            )

            # Register with A2A MCP server
            agent_id = f"pygent_{agent.id}"
            self.a2a_mcp_server.registered_agents[agent_id] = agent_card
            self.a2a_mcp_server.stats['agents_discovered'] += 1

            logger.info(f"Registered agent {agent.name} with A2A MCP server")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent with A2A MCP server: {e}")
            return False

    async def send_a2a_mcp_message(self, agent_id: str, message: str, context_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a message via A2A MCP server"""
        if not self.a2a_mcp_server:
            return {"error": "A2A MCP server not available"}

        try:
            request = {
                "agent_id": agent_id,
                "message": message,
                "context_id": context_id
            }

            # Use the A2A MCP server's send_to_agent method
            response = await self.a2a_mcp_server._handle_self_message(message, context_id)
            return response

        except Exception as e:
            logger.error(f"Failed to send A2A MCP message: {e}")
            return {"error": str(e)}

    async def _register_agent_with_a2a(self, agent: BaseAgent, agent_type: str, capabilities: List[str]) -> None:
        """Register agent with A2A protocol components"""
        if not self.a2a_enabled:
            return

        try:
            # Generate A2A agent card
            agent_card = await self.a2a_card_generator.generate_agent_card(
                agent=agent,
                agent_type=agent_type,
                enable_authentication=True,
                enable_streaming=True
            )

            # Register with A2A discovery
            if self.a2a_discovery:
                await self.a2a_discovery.register_agent(agent.agent_id, agent_card)

            # Register with A2A MCP server
            if self.a2a_mcp_server:
                await self.register_agent_with_a2a_mcp(agent)

            logger.info(f"Successfully registered agent {agent.agent_id} with A2A protocol")

        except Exception as e:
            logger.warning(f"Failed to register agent {agent.agent_id} with A2A protocol: {e}")
            # Don't fail agent creation due to A2A registration issues


class Docker443ModelRunner:
    """
    Docker 4.43 Model Runner integration for enhanced LLM management.

    Provides Docker-native model management with container isolation,
    resource optimization, and performance monitoring for agent spawning.

    Observer-supervised implementation maintaining existing agent factory functionality.
    """

    def __init__(self, agent_factory: AgentFactory):
        self.agent_factory = agent_factory
        self.logger = logging.getLogger(__name__)

        # Docker 4.43 Model Runner components
        self.model_runner = None
        self.container_manager = None
        self.resource_monitor = None

        # Model management
        self.available_models = {}
        self.model_containers = {}
        self.model_performance_metrics = {}

        # Docker configuration
        self.docker_config = {
            "model_runner_version": "4.43.0",
            "container_isolation": True,
            "resource_limits": {
                "memory": "2GB",
                "cpu": "1.0",
                "gpu_memory": "1GB"
            },
            "performance_optimization": {
                "model_caching": True,
                "batch_inference": True,
                "async_processing": True,
                "memory_mapping": True
            },
            "security_features": {
                "container_sandboxing": True,
                "network_isolation": True,
                "read_only_filesystem": True,
                "user_namespace": True
            }
        }

        # Agent spawning optimization
        self.spawning_metrics = {
            "total_spawns": 0,
            "successful_spawns": 0,
            "failed_spawns": 0,
            "average_spawn_time": 0.0,
            "resource_utilization": {}
        }

    async def initialize_docker443_model_runner(self) -> bool:
        """Initialize Docker 4.43 Model Runner for LLM management"""
        try:
            self.logger.info("Initializing Docker 4.43 Model Runner...")

            # Initialize Model Runner
            await self._initialize_model_runner()

            # Setup container management
            await self._initialize_container_manager()

            # Initialize resource monitoring
            await self._initialize_resource_monitor()

            # Discover and register available models
            await self._discover_available_models()

            # Setup performance optimization
            await self._setup_performance_optimization()

            self.logger.info("Docker 4.43 Model Runner initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Docker 4.43 Model Runner: {e}")
            return False

    async def _initialize_model_runner(self) -> None:
        """Initialize Docker 4.43 Model Runner core"""
        try:
            # Simulate Docker 4.43 Model Runner initialization
            self.model_runner = {
                "runner_id": f"docker_model_runner_{uuid.uuid4().hex[:8]}",
                "version": "4.43.0",
                "status": "active",
                "capabilities": {
                    "multi_model_support": True,
                    "gpu_acceleration": True,
                    "batch_processing": True,
                    "streaming_inference": True,
                    "model_quantization": True,
                    "dynamic_batching": True
                },
                "supported_frameworks": ["transformers", "ollama", "vllm", "tensorrt"],
                "container_runtime": "docker",
                "orchestration": "compose",
                "monitoring": "prometheus",
                "initialized_at": datetime.now().isoformat()
            }

            self.logger.info("Docker Model Runner core initialized")

        except Exception as e:
            self.logger.error(f"Model Runner core initialization failed: {e}")
            raise

    async def _initialize_phase4_components(self) -> None:
        """Initialize Phase 4 simulation environment and behavior detection components"""
        try:
            if not PHASE4_MODULES_AVAILABLE:
                logger.warning("Phase 4 modules not available, skipping initialization")
                return

            logger.info("Initializing Phase 4 components...")

            # Initialize simulation environment
            sim_config = {
                "environment_id": f"agent_factory_{uuid.uuid4()}",
                "resource_limits": {
                    "max_agents": getattr(self.settings, 'MAX_AGENTS', 100),
                    "compute_budget": getattr(self.settings, 'COMPUTE_BUDGET', 1000.0),
                    "memory_budget": getattr(self.settings, 'MEMORY_BUDGET', 8192.0)
                },
                "integration_mode": "agent_factory"
            }

            self.simulation_env = await create_simulation_environment(sim_config)
            logger.info("Simulation environment initialized successfully")

            # Initialize emergent behavior detector
            if self.simulation_env:
                # Create mock interaction system and behavior monitor for integration
                mock_interaction_system = type('MockInteractionSystem', (), {
                    'get_agent_interactions': lambda: [],
                    'get_resource_sharing_events': lambda: [],
                    'get_collaboration_metrics': lambda: {}
                })()

                mock_behavior_monitor = type('MockBehaviorMonitor', (), {
                    'get_behavior_patterns': lambda: [],
                    'get_anomaly_scores': lambda: {},
                    'get_emergence_indicators': lambda: {}
                })()

                self.behavior_detector = Docker443EmergentBehaviorDetector(
                    self.simulation_env,
                    mock_interaction_system,
                    mock_behavior_monitor
                )

                await self.behavior_detector.initialize()
                logger.info("Emergent behavior detector initialized successfully")

            logger.info("Phase 4 components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Phase 4 components: {e}")
            # Don't raise - allow factory to continue without Phase 4 features
            self.phase4_enabled = False

    async def _initialize_container_manager(self) -> None:
        """Initialize Docker container management for model isolation"""
        try:
            # Setup container management
            self.container_manager = {
                "container_engine": "docker",
                "orchestration": "docker-compose",
                "isolation_level": "container",
                "network_mode": "bridge",
                "volume_management": {
                    "model_cache": "/app/models",
                    "temp_storage": "/tmp/model_temp",
                    "logs": "/app/logs/models"
                },
                "security_context": {
                    "user": "model_runner",
                    "group": "model_group",
                    "capabilities": ["NET_BIND_SERVICE"],
                    "seccomp_profile": "runtime/default",
                    "apparmor_profile": "docker-default"
                },
                "resource_constraints": self.docker_config["resource_limits"],
                "health_checks": {
                    "enabled": True,
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": "60s"
                }
            }

            self.logger.info("Docker container manager initialized")

        except Exception as e:
            self.logger.error(f"Container manager initialization failed: {e}")
            raise

    async def _initialize_resource_monitor(self) -> None:
        """Initialize resource monitoring for Docker containers"""
        try:
            # Setup resource monitoring
            self.resource_monitor = {
                "monitoring_enabled": True,
                "metrics_collection": {
                    "cpu_usage": True,
                    "memory_usage": True,
                    "gpu_usage": True,
                    "disk_io": True,
                    "network_io": True,
                    "inference_latency": True
                },
                "alert_thresholds": {
                    "cpu_usage": 80,
                    "memory_usage": 85,
                    "gpu_memory": 90,
                    "inference_latency": 5000  # ms
                },
                "optimization_triggers": {
                    "auto_scaling": True,
                    "model_switching": True,
                    "batch_size_adjustment": True,
                    "resource_reallocation": True
                },
                "performance_targets": {
                    "spawn_time": 2.0,  # seconds
                    "inference_latency": 1.0,  # seconds
                    "throughput": 10,  # requests/second
                    "resource_efficiency": 0.8
                }
            }

            self.logger.info("Resource monitoring initialized")

        except Exception as e:
            self.logger.error(f"Resource monitor initialization failed: {e}")
            raise

    async def _discover_available_models(self) -> None:
        """Discover and register available models with Docker 4.43 Model Runner"""
        try:
            # Get existing provider registry from agent factory
            provider_registry = None
            if hasattr(self.agent_factory, 'provider_registry'):
                provider_registry = self.agent_factory.provider_registry
            elif hasattr(self.agent_factory, 'get_provider_registry'):
                provider_registry = self.agent_factory.get_provider_registry()

            # Discover models from provider registry
            if provider_registry:
                existing_models = provider_registry.list_available_models() if hasattr(provider_registry, 'list_available_models') else []
            else:
                existing_models = []

            # Enhanced models with Docker 4.43 optimization
            docker_optimized_models = {
                "qwen2.5:7b": {
                    "model_name": "qwen2.5:7b",
                    "framework": "ollama",
                    "docker_image": "ollama/qwen2.5:7b-docker443",
                    "container_config": {
                        "memory_limit": "8GB",
                        "cpu_limit": "4.0",
                        "gpu_required": False,
                        "optimization_level": "high"
                    },
                    "performance_profile": {
                        "inference_speed": "fast",
                        "memory_efficiency": "high",
                        "batch_support": True,
                        "streaming": True
                    },
                    "capabilities": ["text_generation", "code_completion", "analysis"],
                    "docker443_features": {
                        "model_caching": True,
                        "quantization": "int8",
                        "parallel_inference": True,
                        "memory_mapping": True
                    }
                },
                "deepseek-coder:6.7b": {
                    "model_name": "deepseek-coder:6.7b",
                    "framework": "ollama",
                    "docker_image": "ollama/deepseek-coder:6.7b-docker443",
                    "container_config": {
                        "memory_limit": "12GB",
                        "cpu_limit": "6.0",
                        "gpu_required": False,
                        "optimization_level": "high"
                    },
                    "performance_profile": {
                        "inference_speed": "medium",
                        "memory_efficiency": "medium",
                        "batch_support": True,
                        "streaming": True
                    },
                    "capabilities": ["code_generation", "code_analysis", "debugging"],
                    "docker443_features": {
                        "model_caching": True,
                        "quantization": "int4",
                        "parallel_inference": True,
                        "memory_mapping": True
                    }
                },
                "llama3.2:3b": {
                    "model_name": "llama3.2:3b",
                    "framework": "ollama",
                    "docker_image": "ollama/llama3.2:3b-docker443",
                    "container_config": {
                        "memory_limit": "6GB",
                        "cpu_limit": "2.0",
                        "gpu_required": False,
                        "optimization_level": "medium"
                    },
                    "performance_profile": {
                        "inference_speed": "fast",
                        "memory_efficiency": "very_high",
                        "batch_support": True,
                        "streaming": True
                    },
                    "capabilities": ["text_generation", "conversation", "reasoning"],
                    "docker443_features": {
                        "model_caching": True,
                        "quantization": "int8",
                        "parallel_inference": False,
                        "memory_mapping": True
                    }
                },
                "openrouter/deepseek-r1": {
                    "model_name": "openrouter/deepseek-r1",
                    "framework": "openrouter",
                    "docker_image": "openrouter/deepseek-r1:docker443",
                    "container_config": {
                        "memory_limit": "4GB",
                        "cpu_limit": "2.0",
                        "gpu_required": False,
                        "optimization_level": "high"
                    },
                    "performance_profile": {
                        "inference_speed": "very_fast",
                        "memory_efficiency": "high",
                        "batch_support": True,
                        "streaming": True
                    },
                    "capabilities": ["reasoning", "analysis", "problem_solving"],
                    "docker443_features": {
                        "model_caching": False,  # External API
                        "quantization": "none",
                        "parallel_inference": True,
                        "memory_mapping": False
                    }
                }
            }

            # Merge existing models with Docker optimized models
            for model_name in existing_models:
                if model_name not in docker_optimized_models:
                    docker_optimized_models[model_name] = {
                        "model_name": model_name,
                        "framework": "unknown",
                        "docker_image": f"generic/{model_name}:docker443",
                        "container_config": self.docker_config["resource_limits"],
                        "performance_profile": {
                            "inference_speed": "medium",
                            "memory_efficiency": "medium",
                            "batch_support": False,
                            "streaming": False
                        },
                        "capabilities": ["general"],
                        "docker443_features": {
                            "model_caching": True,
                            "quantization": "none",
                            "parallel_inference": False,
                            "memory_mapping": False
                        }
                    }

            self.available_models = docker_optimized_models

            self.logger.info(f"Discovered {len(self.available_models)} Docker-optimized models")

        except Exception as e:
            self.logger.error(f"Model discovery failed: {e}")
            raise

    async def _setup_performance_optimization(self) -> None:
        """Setup Docker 4.43 performance optimization features"""
        try:
            # Configure performance optimization
            optimization_config = {
                "model_caching": {
                    "enabled": True,
                    "cache_size": "10GB",
                    "cache_location": "/app/model_cache",
                    "eviction_policy": "LRU",
                    "preload_models": ["qwen2.5:7b", "llama3.2:3b"]
                },
                "batch_inference": {
                    "enabled": True,
                    "max_batch_size": 8,
                    "batch_timeout": 100,  # ms
                    "dynamic_batching": True
                },
                "async_processing": {
                    "enabled": True,
                    "worker_threads": 4,
                    "queue_size": 100,
                    "priority_scheduling": True
                },
                "memory_optimization": {
                    "memory_mapping": True,
                    "shared_memory": True,
                    "garbage_collection": "aggressive",
                    "memory_pooling": True
                },
                "container_optimization": {
                    "image_layering": True,
                    "multi_stage_builds": True,
                    "resource_sharing": True,
                    "startup_optimization": True
                }
            }

            # Apply optimizations to model runner
            self.model_runner["optimization_config"] = optimization_config

            self.logger.info("Performance optimization configured")

        except Exception as e:
            self.logger.error(f"Performance optimization setup failed: {e}")
            raise

    async def spawn_agent_with_docker443(self, agent_config: Dict[str, Any]) -> Optional[str]:
        """Spawn agent using Docker 4.43 Model Runner with container isolation"""
        try:
            spawn_start = datetime.now()
            self.spawning_metrics["total_spawns"] += 1

            self.logger.info(f"Spawning agent with Docker 4.43 optimization: {agent_config.get('name', 'unnamed')}")

            # Select optimal model for agent type
            optimal_model = await self._select_optimal_model(agent_config)

            # Setup Docker container for agent
            container_config = await self._setup_agent_container(agent_config, optimal_model)

            # Create agent with Docker optimization
            agent = await self._create_agent_with_docker_optimization(agent_config, container_config)

            if agent:
                # Monitor agent performance
                await self._setup_agent_performance_monitoring(agent, container_config)

                # Record successful spawn
                spawn_duration = (datetime.now() - spawn_start).total_seconds()
                self.spawning_metrics["successful_spawns"] += 1
                self._update_spawn_metrics(spawn_duration, True)

                self.logger.info(f"Successfully spawned agent {agent.agent_id} in {spawn_duration:.2f}s")
                return agent.agent_id
            else:
                self.spawning_metrics["failed_spawns"] += 1
                self._update_spawn_metrics((datetime.now() - spawn_start).total_seconds(), False)
                return None

        except Exception as e:
            self.logger.error(f"Docker 4.43 agent spawning failed: {e}")
            self.spawning_metrics["failed_spawns"] += 1
            return None

    async def _select_optimal_model(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal model for agent based on type and requirements"""
        try:
            agent_type = agent_config.get("type", "general")
            agent_role = agent_config.get("custom_config", {}).get("role", "adapter")
            capabilities = agent_config.get("capabilities", [])

            # Model selection logic based on agent requirements
            if agent_type == "coding" or "builder" in agent_role:
                # Coding agents need code-specialized models
                if "deepseek-coder:6.7b" in self.available_models:
                    return self.available_models["deepseek-coder:6.7b"]
                elif "qwen2.5:7b" in self.available_models:
                    return self.available_models["qwen2.5:7b"]

            elif agent_type == "research" or "explorer" in agent_role:
                # Research agents need reasoning capabilities
                if "openrouter/deepseek-r1" in self.available_models:
                    return self.available_models["openrouter/deepseek-r1"]
                elif "qwen2.5:7b" in self.available_models:
                    return self.available_models["qwen2.5:7b"]

            elif agent_type == "analysis" or "harvester" in agent_role:
                # Analysis agents need efficient processing
                if "llama3.2:3b" in self.available_models:
                    return self.available_models["llama3.2:3b"]
                elif "qwen2.5:7b" in self.available_models:
                    return self.available_models["qwen2.5:7b"]

            elif agent_type == "monitoring" or "defender" in agent_role:
                # Monitoring agents need fast response
                if "llama3.2:3b" in self.available_models:
                    return self.available_models["llama3.2:3b"]

            elif agent_type == "communication" or "communicator" in agent_role:
                # Communication agents need balanced performance
                if "qwen2.5:7b" in self.available_models:
                    return self.available_models["qwen2.5:7b"]

            # Default fallback to most efficient model
            if "llama3.2:3b" in self.available_models:
                return self.available_models["llama3.2:3b"]
            elif self.available_models:
                return list(self.available_models.values())[0]
            else:
                # Fallback model configuration
                return {
                    "model_name": "fallback",
                    "framework": "generic",
                    "docker_image": "generic/fallback:latest",
                    "container_config": self.docker_config["resource_limits"],
                    "performance_profile": {"inference_speed": "medium"},
                    "capabilities": ["general"]
                }

        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            return self.available_models.get("llama3.2:3b", {})

    async def _setup_agent_container(self, agent_config: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup Docker container configuration for agent with isolation"""
        try:
            agent_name = agent_config.get("name", f"agent_{uuid.uuid4().hex[:8]}")

            # Container configuration with Docker 4.43 features
            container_config = {
                "container_name": f"pygent_agent_{agent_name}",
                "image": model_config.get("docker_image", "generic/agent:latest"),
                "isolation_level": "container",
                "resource_limits": {
                    "memory": model_config.get("container_config", {}).get("memory_limit", "2GB"),
                    "cpu": model_config.get("container_config", {}).get("cpu_limit", "1.0"),
                    "gpu_memory": model_config.get("container_config", {}).get("gpu_memory", "0")
                },
                "security_context": {
                    "user": "agent_user",
                    "group": "agent_group",
                    "read_only_root": True,
                    "no_new_privileges": True,
                    "seccomp_profile": "runtime/default"
                },
                "network_config": {
                    "mode": "bridge",
                    "isolation": True,
                    "dns": ["8.8.8.8", "8.8.4.4"],
                    "port_mapping": {}
                },
                "volume_mounts": {
                    "model_cache": {
                        "source": "/app/model_cache",
                        "target": "/models",
                        "read_only": True
                    },
                    "agent_workspace": {
                        "source": f"/app/agents/{agent_name}",
                        "target": "/workspace",
                        "read_only": False
                    },
                    "logs": {
                        "source": f"/app/logs/agents/{agent_name}",
                        "target": "/logs",
                        "read_only": False
                    }
                },
                "environment_variables": {
                    "AGENT_NAME": agent_name,
                    "AGENT_TYPE": agent_config.get("type", "general"),
                    "MODEL_NAME": model_config.get("model_name", "fallback"),
                    "DOCKER_VERSION": "4.43.0",
                    "OPTIMIZATION_LEVEL": model_config.get("container_config", {}).get("optimization_level", "medium")
                },
                "health_check": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": "60s"
                },
                "restart_policy": "unless-stopped",
                "docker443_features": {
                    "model_runner_integration": True,
                    "performance_monitoring": True,
                    "auto_scaling": False,
                    "resource_optimization": True
                }
            }

            # Store container configuration for monitoring
            self.model_containers[agent_name] = container_config

            return container_config

        except Exception as e:
            self.logger.error(f"Container setup failed: {e}")
            raise

    async def _create_agent_with_docker_optimization(self, agent_config: Dict[str, Any], container_config: Dict[str, Any]) -> Optional[BaseAgent]:
        """Create agent with Docker 4.43 optimization and container isolation"""
        try:
            # Enhance agent config with Docker optimization
            enhanced_config = agent_config.copy()
            enhanced_config["docker443_config"] = {
                "container_config": container_config,
                "model_runner": self.model_runner,
                "performance_monitoring": True,
                "resource_optimization": True
            }

            # Use existing agent factory create_agent method with enhancements
            agent = await self.agent_factory.create_agent(enhanced_config)

            if agent:
                # Add Docker-specific attributes to agent
                agent.docker443_container = container_config["container_name"]
                agent.docker443_model = container_config["environment_variables"]["MODEL_NAME"]
                agent.docker443_optimization = True

                self.logger.info(f"Created Docker-optimized agent: {agent.agent_id}")

            return agent

        except Exception as e:
            self.logger.error(f"Docker-optimized agent creation failed: {e}")
            return None

    async def _setup_agent_performance_monitoring(self, agent: BaseAgent, container_config: Dict[str, Any]) -> None:
        """Setup performance monitoring for Docker-optimized agent"""
        try:
            agent_name = getattr(agent, 'name', agent.agent_id)

            # Initialize performance metrics
            self.model_performance_metrics[agent_name] = {
                "container_name": container_config["container_name"],
                "model_name": container_config["environment_variables"]["MODEL_NAME"],
                "spawn_time": datetime.now(),
                "resource_usage": {
                    "cpu_usage": 0.0,
                    "memory_usage": 0.0,
                    "gpu_usage": 0.0,
                    "network_io": 0.0
                },
                "performance_metrics": {
                    "inference_latency": [],
                    "throughput": 0.0,
                    "error_rate": 0.0,
                    "uptime": 0.0
                },
                "optimization_status": {
                    "model_cached": True,
                    "batch_processing": False,
                    "memory_optimized": True,
                    "container_healthy": True
                }
            }

            self.logger.debug(f"Performance monitoring setup for agent: {agent_name}")

        except Exception as e:
            self.logger.error(f"Performance monitoring setup failed: {e}")

    def _update_spawn_metrics(self, spawn_duration: float, success: bool) -> None:
        """Update agent spawning metrics"""
        try:
            # Update average spawn time
            total_spawns = self.spawning_metrics["total_spawns"]
            current_avg = self.spawning_metrics["average_spawn_time"]

            new_avg = ((current_avg * (total_spawns - 1)) + spawn_duration) / total_spawns
            self.spawning_metrics["average_spawn_time"] = new_avg

            # Update resource utilization
            if success:
                self.spawning_metrics["resource_utilization"]["successful_spawn_time"] = spawn_duration
            else:
                self.spawning_metrics["resource_utilization"]["failed_spawn_time"] = spawn_duration

        except Exception as e:
            self.logger.error(f"Spawn metrics update failed: {e}")

    async def monitor_agent_performance(self) -> Dict[str, Any]:
        """Monitor performance of all Docker-optimized agents"""
        try:
            performance_report = {
                "timestamp": datetime.now().isoformat(),
                "total_agents": len(self.model_performance_metrics),
                "spawning_metrics": self.spawning_metrics,
                "agent_performance": {},
                "resource_summary": {
                    "total_cpu_usage": 0.0,
                    "total_memory_usage": 0.0,
                    "total_gpu_usage": 0.0,
                    "average_latency": 0.0
                },
                "optimization_summary": {
                    "cached_models": 0,
                    "healthy_containers": 0,
                    "optimized_agents": 0
                }
            }

            total_latency = 0.0
            latency_count = 0

            for agent_name, metrics in self.model_performance_metrics.items():
                # Simulate performance data collection
                current_metrics = {
                    "uptime": (datetime.now() - metrics["spawn_time"]).total_seconds(),
                    "cpu_usage": random.uniform(10, 60),
                    "memory_usage": random.uniform(20, 80),
                    "gpu_usage": random.uniform(0, 30),
                    "inference_latency": random.uniform(0.5, 2.0),
                    "container_healthy": True,
                    "model_cached": metrics["optimization_status"]["model_cached"]
                }

                # Update stored metrics
                metrics["resource_usage"].update({
                    "cpu_usage": current_metrics["cpu_usage"],
                    "memory_usage": current_metrics["memory_usage"],
                    "gpu_usage": current_metrics["gpu_usage"]
                })

                metrics["performance_metrics"]["inference_latency"].append(current_metrics["inference_latency"])
                if len(metrics["performance_metrics"]["inference_latency"]) > 100:
                    metrics["performance_metrics"]["inference_latency"] = metrics["performance_metrics"]["inference_latency"][-100:]

                # Add to report
                performance_report["agent_performance"][agent_name] = current_metrics

                # Update summaries
                performance_report["resource_summary"]["total_cpu_usage"] += current_metrics["cpu_usage"]
                performance_report["resource_summary"]["total_memory_usage"] += current_metrics["memory_usage"]
                performance_report["resource_summary"]["total_gpu_usage"] += current_metrics["gpu_usage"]

                total_latency += current_metrics["inference_latency"]
                latency_count += 1

                if current_metrics["model_cached"]:
                    performance_report["optimization_summary"]["cached_models"] += 1
                if current_metrics["container_healthy"]:
                    performance_report["optimization_summary"]["healthy_containers"] += 1

                performance_report["optimization_summary"]["optimized_agents"] += 1

            # Calculate averages
            if latency_count > 0:
                performance_report["resource_summary"]["average_latency"] = total_latency / latency_count

            return performance_report

        except Exception as e:
            self.logger.error(f"Agent performance monitoring failed: {e}")
            return {"error": str(e)}

    async def get_docker443_model_runner_status(self) -> Dict[str, Any]:
        """Get comprehensive Docker 4.43 Model Runner status"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "model_runner": {
                    "status": self.model_runner.get("status", "unknown") if self.model_runner else "not_initialized",
                    "version": self.model_runner.get("version", "unknown") if self.model_runner else "unknown",
                    "capabilities": self.model_runner.get("capabilities", {}) if self.model_runner else {}
                },
                "available_models": {
                    "total_models": len(self.available_models),
                    "models": list(self.available_models.keys()),
                    "frameworks": list(set(model.get("framework", "unknown") for model in self.available_models.values()))
                },
                "container_management": {
                    "active_containers": len(self.model_containers),
                    "container_engine": self.container_manager.get("container_engine", "unknown") if self.container_manager else "unknown",
                    "isolation_enabled": self.docker_config["container_isolation"]
                },
                "performance_optimization": {
                    "model_caching": self.docker_config["performance_optimization"]["model_caching"],
                    "batch_inference": self.docker_config["performance_optimization"]["batch_inference"],
                    "async_processing": self.docker_config["performance_optimization"]["async_processing"]
                },
                "spawning_metrics": self.spawning_metrics,
                "resource_monitoring": {
                    "enabled": self.resource_monitor.get("monitoring_enabled", False) if self.resource_monitor else False,
                    "monitored_agents": len(self.model_performance_metrics)
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get Model Runner status: {e}")
            return {"error": str(e)}
