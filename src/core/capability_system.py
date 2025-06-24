"""
Agent Capability System

This module implements the capability management system for agents in PyGent Factory.
It provides a framework for defining, registering, and executing agent capabilities
with proper validation, dependency management, and MCP integration.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import inspect
import uuid

from .agent import AgentCapability, BaseAgent, AgentError


logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of capabilities"""
    CORE = "core"           # Built-in capabilities
    MCP = "mcp"            # MCP tool-based capabilities
    CUSTOM = "custom"      # Custom implementation capabilities
    COMPOSITE = "composite" # Capabilities that combine others


class CapabilityStatus(Enum):
    """Capability execution status"""
    AVAILABLE = "available"
    EXECUTING = "executing"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class CapabilityParameter:
    """Defines a capability parameter"""
    name: str
    type: Type
    description: str
    required: bool = True
    default: Any = None
    validation_func: Optional[Callable] = None


@dataclass
class CapabilityDefinition:
    """Complete capability definition"""
    name: str
    description: str
    capability_type: CapabilityType
    parameters: List[CapabilityParameter] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 3
    implementation: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityExecution:
    """Tracks capability execution"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    capability_name: str = ""
    agent_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: CapabilityStatus = CapabilityStatus.AVAILABLE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0


class CapabilityValidator:
    """Validates capability parameters and dependencies"""
    
    @staticmethod
    def validate_parameters(definition: CapabilityDefinition, 
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize capability parameters.
        
        Args:
            definition: Capability definition
            parameters: Input parameters
            
        Returns:
            Dict[str, Any]: Validated and normalized parameters
            
        Raises:
            ValueError: If validation fails
        """
        validated = {}
        
        for param_def in definition.parameters:
            param_name = param_def.name
            
            # Check required parameters
            if param_def.required and param_name not in parameters:
                if param_def.default is not None:
                    validated[param_name] = param_def.default
                else:
                    raise ValueError(f"Required parameter missing: {param_name}")
            elif param_name in parameters:
                value = parameters[param_name]
                
                # Type validation
                if not isinstance(value, param_def.type):
                    try:
                        # Attempt type conversion
                        value = param_def.type(value)
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Parameter {param_name} must be of type {param_def.type.__name__}"
                        )
                
                # Custom validation
                if param_def.validation_func:
                    if not param_def.validation_func(value):
                        raise ValueError(f"Validation failed for parameter: {param_name}")
                
                validated[param_name] = value
        
        return validated
    
    @staticmethod
    def check_dependencies(definition: CapabilityDefinition,
                          available_capabilities: List[str]) -> bool:
        """Check if all capability dependencies are available."""
        for dependency in definition.dependencies:
            if dependency not in available_capabilities:
                return False
        return True
    
    @staticmethod
    def check_tools(definition: CapabilityDefinition,
                   available_tools: List[str]) -> bool:
        """Check if all required MCP tools are available."""
        for tool in definition.required_tools:
            if tool not in available_tools:
                return False
        return True


class CapabilityRegistry:
    """Registry for managing capability definitions"""
    
    def __init__(self):
        self._capabilities: Dict[str, CapabilityDefinition] = {}
        self._lock = asyncio.Lock()
    
    async def register_capability(self, definition: CapabilityDefinition) -> None:
        """Register a capability definition."""
        async with self._lock:
            self._capabilities[definition.name] = definition
            logger.info(f"Registered capability: {definition.name}")
    
    async def unregister_capability(self, name: str) -> None:
        """Unregister a capability."""
        async with self._lock:
            if name in self._capabilities:
                del self._capabilities[name]
                logger.info(f"Unregistered capability: {name}")
    
    async def get_capability(self, name: str) -> Optional[CapabilityDefinition]:
        """Get a capability definition by name."""
        return self._capabilities.get(name)
    
    async def list_capabilities(self, 
                               capability_type: Optional[CapabilityType] = None) -> List[CapabilityDefinition]:
        """List all registered capabilities."""
        capabilities = list(self._capabilities.values())
        
        if capability_type:
            capabilities = [c for c in capabilities if c.capability_type == capability_type]
        
        return capabilities
    
    async def get_capability_names(self) -> List[str]:
        """Get list of all capability names."""
        return list(self._capabilities.keys())


class CapabilityExecutor:
    """Executes capabilities with proper error handling and monitoring"""
    
    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry
        self._active_executions: Dict[str, CapabilityExecution] = {}
        self._execution_history: List[CapabilityExecution] = []
        self._lock = asyncio.Lock()
    
    async def execute_capability(self,
                                agent: BaseAgent,
                                capability_name: str,
                                parameters: Dict[str, Any]) -> Any:
        """
        Execute a capability for an agent.
        
        Args:
            agent: The agent executing the capability
            capability_name: Name of the capability to execute
            parameters: Execution parameters
            
        Returns:
            Any: Capability execution result
            
        Raises:
            AgentError: If execution fails
        """
        # Get capability definition
        definition = await self.registry.get_capability(capability_name)
        if not definition:
            raise AgentError(f"Unknown capability: {capability_name}")
        
        # Create execution record
        execution = CapabilityExecution(
            capability_name=capability_name,
            agent_id=agent.agent_id,
            parameters=parameters
        )
        
        try:
            # Validate parameters
            validated_params = CapabilityValidator.validate_parameters(
                definition, parameters
            )
            execution.parameters = validated_params
            
            # Check dependencies
            agent_capabilities = agent.capabilities
            if not CapabilityValidator.check_dependencies(definition, agent_capabilities):
                raise AgentError(f"Missing dependencies for capability: {capability_name}")
            
            # Check required tools
            agent_tools = list(agent.mcp_tools.keys())
            if not CapabilityValidator.check_tools(definition, agent_tools):
                raise AgentError(f"Missing required tools for capability: {capability_name}")
            
            # Start execution
            execution.status = CapabilityStatus.EXECUTING
            execution.start_time = datetime.utcnow()
            
            async with self._lock:
                self._active_executions[execution.execution_id] = execution
            
            # Execute capability
            if definition.implementation:
                # Direct implementation
                result = await self._execute_direct(agent, definition, validated_params)
            else:
                # Delegate to agent
                result = await agent.execute_capability(capability_name, validated_params)
            
            # Record success
            execution.status = CapabilityStatus.AVAILABLE
            execution.result = result
            execution.end_time = datetime.utcnow()
            
            # Update agent activity
            agent.update_activity()
            agent.record_execution(
                capability_name,
                validated_params,
                result,
                (execution.end_time - execution.start_time).total_seconds(),
                True
            )
            
            return result
            
        except Exception as e:
            # Record failure
            execution.status = CapabilityStatus.ERROR
            execution.error = str(e)
            execution.end_time = datetime.utcnow()
            
            # Update agent
            if execution.start_time:
                duration = (execution.end_time - execution.start_time).total_seconds()
            else:
                duration = 0
            
            agent.record_execution(
                capability_name,
                parameters,
                None,
                duration,
                False
            )
            
            logger.error(f"Capability execution failed: {capability_name} - {str(e)}")
            raise AgentError(f"Capability execution failed: {str(e)}")
            
        finally:
            # Cleanup and record
            async with self._lock:
                self._active_executions.pop(execution.execution_id, None)
                self._execution_history.append(execution)
                
                # Keep only last 1000 executions
                if len(self._execution_history) > 1000:
                    self._execution_history = self._execution_history[-1000:]
    
    async def _execute_direct(self,
                             agent: BaseAgent,
                             definition: CapabilityDefinition,
                             parameters: Dict[str, Any]) -> Any:
        """Execute capability using direct implementation."""
        if not definition.implementation:
            raise AgentError("No implementation available for capability")
        
        # Check if implementation expects agent parameter
        sig = inspect.signature(definition.implementation)
        if 'agent' in sig.parameters:
            return await definition.implementation(agent, **parameters)
        else:
            return await definition.implementation(**parameters)
    
    async def get_active_executions(self, agent_id: Optional[str] = None) -> List[CapabilityExecution]:
        """Get list of active executions."""
        executions = list(self._active_executions.values())
        
        if agent_id:
            executions = [e for e in executions if e.agent_id == agent_id]
        
        return executions
    
    async def get_execution_history(self, 
                                   agent_id: Optional[str] = None,
                                   capability_name: Optional[str] = None,
                                   limit: int = 100) -> List[CapabilityExecution]:
        """Get execution history with optional filtering."""
        history = self._execution_history.copy()
        
        if agent_id:
            history = [e for e in history if e.agent_id == agent_id]
        
        if capability_name:
            history = [e for e in history if e.capability_name == capability_name]
        
        # Return most recent first
        history.reverse()
        return history[:limit]


class CapabilityManager:
    """
    Main capability management system.
    
    Coordinates capability registration, validation, and execution
    across the PyGent Factory system.
    """
    
    def __init__(self):
        self.registry = CapabilityRegistry()
        self.executor = CapabilityExecutor(self.registry)
        self._setup_core_capabilities()
    
    def _setup_core_capabilities(self) -> None:
        """Set up core system capabilities."""
        # This will be expanded with actual core capabilities
        pass
    
    async def register_capability(self, definition: CapabilityDefinition) -> None:
        """Register a new capability."""
        await self.registry.register_capability(definition)
    
    async def execute_capability(self,
                                agent: BaseAgent,
                                capability_name: str,
                                parameters: Dict[str, Any]) -> Any:
        """Execute a capability for an agent."""
        return await self.executor.execute_capability(agent, capability_name, parameters)
    
    async def get_available_capabilities(self, 
                                        agent: BaseAgent) -> List[CapabilityDefinition]:
        """Get capabilities available to an agent."""
        all_capabilities = await self.registry.list_capabilities()
        available = []
        
        for capability in all_capabilities:
            # Check if agent has this capability
            if capability.name in agent.capabilities:
                # Check dependencies and tools
                if (CapabilityValidator.check_dependencies(capability, agent.capabilities) and
                    CapabilityValidator.check_tools(capability, list(agent.mcp_tools.keys()))):
                    available.append(capability)
        
        return available
    
    async def validate_agent_capabilities(self, agent: BaseAgent) -> Dict[str, bool]:
        """Validate all capabilities for an agent."""
        results = {}
        
        for capability_name in agent.capabilities:
            definition = await self.registry.get_capability(capability_name)
            if definition:
                valid = (
                    CapabilityValidator.check_dependencies(definition, agent.capabilities) and
                    CapabilityValidator.check_tools(definition, list(agent.mcp_tools.keys()))
                )
                results[capability_name] = valid
            else:
                results[capability_name] = False
        
        return results
    
    async def get_capability_info(self, capability_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a capability."""
        definition = await self.registry.get_capability(capability_name)
        if not definition:
            return None
        
        return {
            "name": definition.name,
            "description": definition.description,
            "type": definition.capability_type.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.__name__,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default
                } for p in definition.parameters
            ],
            "required_tools": definition.required_tools,
            "required_resources": definition.required_resources,
            "dependencies": definition.dependencies,
            "timeout": definition.timeout,
            "metadata": definition.metadata
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get capability system metrics."""
        all_capabilities = await self.registry.list_capabilities()
        active_executions = await self.executor.get_active_executions()
        
        return {
            "total_capabilities": len(all_capabilities),
            "capabilities_by_type": {
                cap_type.value: len([c for c in all_capabilities if c.capability_type == cap_type])
                for cap_type in CapabilityType
            },
            "active_executions": len(active_executions),
            "execution_history_size": len(self.executor._execution_history)
        }


# Backward compatibility alias
CapabilitySystem = CapabilityManager

# Export all classes
__all__ = [
    "CapabilityType",
    "CapabilityStatus",
    "CapabilityParameter",
    "CapabilityDefinition",
    "CapabilityExecution",
    "CapabilityValidator",
    "CapabilityRegistry",
    "CapabilityExecutor",
    "CapabilityManager",
    "CapabilitySystem"  # Backward compatibility
]
