"""
Agent Capability System

This module defines agent capabilities and capability management functionality.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid


class CapabilityType(Enum):
    """Types of agent capabilities"""
    CORE = "core"                    # Core system capabilities
    COGNITIVE = "cognitive"          # Reasoning, planning, learning
    COMMUNICATION = "communication"  # Message handling, protocols
    TOOL = "tool"                   # Tool execution capabilities
    MEMORY = "memory"               # Memory management
    PERCEPTION = "perception"       # Input processing
    ACTION = "action"               # Output generation/actions
    INTEGRATION = "integration"     # External system integration
    CUSTOM = "custom"               # Custom domain-specific capabilities


@dataclass
class CapabilityParameter:
    """Defines a capability parameter"""
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    validation: Optional[Callable[[Any], bool]] = None
    
    def validate(self, value: Any) -> bool:
        """Validate parameter value"""
        if self.validation:
            return self.validation(value)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
            "default": self.default
        }


@dataclass
class AgentCapability:
    """
    Defines an agent capability.
    
    A capability represents a specific function or skill that an agent
    can perform, including the parameters it accepts and the resources
    it requires.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    capability_type: CapabilityType = CapabilityType.CUSTOM
    version: str = "1.0.0"
    
    # Parameters and validation
    parameters: List[CapabilityParameter] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    
    # Execution configuration
    timeout_seconds: int = 300
    max_retries: int = 3
    async_execution: bool = True
    
    # Capability metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Implementation details
    implementation: Optional[Callable] = None
    module_path: Optional[str] = None
    class_name: Optional[str] = None
    
    def add_parameter(self, name: str, param_type: str, description: str = "",
                     required: bool = True, default: Any = None,
                     validation: Optional[Callable[[Any], bool]] = None) -> None:
        """Add a parameter to the capability"""
        param = CapabilityParameter(
            name=name,
            type=param_type,
            description=description,
            required=required,
            default=default,
            validation=validation
        )
        self.parameters.append(param)
    
    def get_parameter(self, name: str) -> Optional[CapabilityParameter]:
        """Get parameter by name"""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate provided parameters against capability definition.
        
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Required parameter '{param.name}' is missing")
            elif param.name in params:
                # Validate parameter value
                if not param.validate(params[param.name]):
                    errors.append(f"Parameter '{param.name}' validation failed")
        
        # Check for unexpected parameters
        expected_params = {param.name for param in self.parameters}
        for param_name in params:
            if param_name not in expected_params:
                errors.append(f"Unexpected parameter '{param_name}'")
        
        return len(errors) == 0, errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters"""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in self.parameters:
            schema["properties"][param.name] = {
                "type": param.type,
                "description": param.description
            }
            
            if param.default is not None:
                schema["properties"][param.name]["default"] = param.default
            
            if param.required:
                schema["required"].append(param.name)
        
        return schema
    
    def has_required_tools(self, available_tools: List[str]) -> bool:
        """Check if all required tools are available"""
        return all(tool in available_tools for tool in self.required_tools)
    
    def has_required_resources(self, available_resources: List[str]) -> bool:
        """Check if all required resources are available"""
        return all(resource in available_resources for resource in self.required_resources)
    
    def can_execute(self, available_tools: List[str] = None, 
                   available_resources: List[str] = None) -> tuple[bool, List[str]]:
        """
        Check if capability can be executed with available resources.
        
        Returns:
            tuple: (can_execute, missing_requirements)
        """
        missing = []
        
        # Check tools
        if available_tools is not None:
            missing_tools = [tool for tool in self.required_tools if tool not in available_tools]
            missing.extend([f"tool:{tool}" for tool in missing_tools])
        
        # Check resources
        if available_resources is not None:
            missing_resources = [res for res in self.required_resources if res not in available_resources]
            missing.extend([f"resource:{res}" for res in missing_resources])
        
        return len(missing) == 0, missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capability_type": self.capability_type.value,
            "version": self.version,
            "parameters": [param.to_dict() for param in self.parameters],
            "required_tools": self.required_tools,
            "required_resources": self.required_resources,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "async_execution": self.async_execution,
            "tags": self.tags,
            "metadata": self.metadata,
            "module_path": self.module_path,
            "class_name": self.class_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCapability':
        """Create capability from dictionary"""
        capability = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            capability_type=CapabilityType(data.get("capability_type", "custom")),
            version=data.get("version", "1.0.0"),
            required_tools=data.get("required_tools", []),
            required_resources=data.get("required_resources", []),
            timeout_seconds=data.get("timeout_seconds", 300),
            max_retries=data.get("max_retries", 3),
            async_execution=data.get("async_execution", True),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            module_path=data.get("module_path"),
            class_name=data.get("class_name")
        )
        
        # Add parameters
        for param_data in data.get("parameters", []):
            param = CapabilityParameter(
                name=param_data["name"],
                type=param_data["type"],
                description=param_data.get("description", ""),
                required=param_data.get("required", True),
                default=param_data.get("default")
            )
            capability.parameters.append(param)
        
        return capability
    
    def __str__(self) -> str:
        """String representation"""
        return f"AgentCapability(name={self.name}, type={self.capability_type.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"AgentCapability(id={self.id}, name={self.name}, "
                f"type={self.capability_type.value}, version={self.version})")


# Factory functions for common capabilities
def create_text_processing_capability() -> AgentCapability:
    """Create a text processing capability"""
    capability = AgentCapability(
        name="text_processing",
        description="Process and analyze text content",
        capability_type=CapabilityType.COGNITIVE
    )
    
    capability.add_parameter("text", "string", "Text to process", required=True)
    capability.add_parameter("operation", "string", "Processing operation", required=True)
    capability.add_parameter("options", "object", "Processing options", required=False, default={})
    
    return capability


def create_memory_retrieval_capability() -> AgentCapability:
    """Create a memory retrieval capability"""
    capability = AgentCapability(
        name="memory_retrieval",
        description="Retrieve information from agent memory",
        capability_type=CapabilityType.MEMORY
    )
    
    capability.add_parameter("query", "string", "Search query", required=True)
    capability.add_parameter("memory_type", "string", "Type of memory to search", required=False)
    capability.add_parameter("limit", "integer", "Maximum results", required=False, default=10)
    
    capability.required_resources = ["memory_manager"]
    
    return capability


def create_tool_execution_capability() -> AgentCapability:
    """Create a tool execution capability"""
    capability = AgentCapability(
        name="tool_execution",
        description="Execute external tools and commands",
        capability_type=CapabilityType.TOOL
    )
    
    capability.add_parameter("tool_name", "string", "Name of tool to execute", required=True)
    capability.add_parameter("arguments", "object", "Tool arguments", required=False, default={})
    capability.add_parameter("timeout", "integer", "Execution timeout", required=False, default=60)
    
    capability.required_resources = ["mcp_manager"]
    
    return capability


def create_communication_capability() -> AgentCapability:
    """Create a communication capability"""
    capability = AgentCapability(
        name="communication",
        description="Send and receive messages",
        capability_type=CapabilityType.COMMUNICATION
    )
    
    capability.add_parameter("recipient", "string", "Message recipient", required=True)
    capability.add_parameter("content", "object", "Message content", required=True)
    capability.add_parameter("message_type", "string", "Type of message", required=False, default="request")
    
    capability.required_resources = ["message_bus"]
    
    return capability
