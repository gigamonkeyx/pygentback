"""
Agent Configuration

This module defines agent configuration classes and validation.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import timedelta
import uuid


@dataclass
class AgentConfig:
    """
    Configuration for an agent instance.
    
    This class defines all configurable aspects of an agent including
    behavior settings, resource limits, and integration parameters.
    """
    
    # Basic identification
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    agent_type: str = "base"
    version: str = "1.0.0"
    
    # Behavior configuration
    max_concurrent_tasks: int = 5
    default_timeout: int = 300  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Memory configuration
    memory_enabled: bool = True
    memory_limit: int = 1000  # number of entries
    memory_types: List[str] = field(default_factory=lambda: ["short_term", "long_term"])
    memory_consolidation_interval: int = 3600  # seconds
    
    # Communication configuration
    message_queue_size: int = 100
    message_timeout: int = 60  # seconds
    heartbeat_interval: int = 30  # seconds
    
    # Capability configuration
    enabled_capabilities: List[str] = field(default_factory=list)
    disabled_capabilities: List[str] = field(default_factory=list)
    capability_timeout: int = 300  # seconds
    
    # MCP configuration
    mcp_tools: List[str] = field(default_factory=list)
    mcp_servers: List[str] = field(default_factory=list)
    mcp_timeout: int = 30  # seconds
    
    # Resource limits
    max_memory_usage_mb: int = 512
    max_cpu_usage_percent: float = 80.0
    max_execution_time: int = 3600  # seconds
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_interval: int = 60  # seconds
    
    # Security settings
    require_authentication: bool = True
    allowed_users: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)
    
    # Custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Persistence settings
    persist_state: bool = True
    state_save_interval: int = 300  # seconds
    backup_enabled: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        if self.name is None:
            self.name = f"{self.agent_type}_{self.agent_id[:8]}"
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        
        if self.default_timeout <= 0:
            raise ValueError("default_timeout must be positive")
        
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        
        if self.memory_limit <= 0:
            raise ValueError("memory_limit must be positive")
        
        if self.message_queue_size <= 0:
            raise ValueError("message_queue_size must be positive")
        
        if self.max_memory_usage_mb <= 0:
            raise ValueError("max_memory_usage_mb must be positive")
        
        if not 0 <= self.max_cpu_usage_percent <= 100:
            raise ValueError("max_cpu_usage_percent must be between 0 and 100")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log_level")
    
    def get_timeout_timedelta(self) -> timedelta:
        """Get default timeout as timedelta"""
        return timedelta(seconds=self.default_timeout)
    
    def get_retry_delay_timedelta(self) -> timedelta:
        """Get retry delay as timedelta"""
        return timedelta(seconds=self.retry_delay)
    
    def is_capability_enabled(self, capability_name: str) -> bool:
        """Check if a capability is enabled"""
        if capability_name in self.disabled_capabilities:
            return False
        
        if self.enabled_capabilities:
            return capability_name in self.enabled_capabilities
        
        return True  # Default to enabled if no explicit list
    
    def enable_capability(self, capability_name: str) -> None:
        """Enable a capability"""
        if capability_name in self.disabled_capabilities:
            self.disabled_capabilities.remove(capability_name)
        
        if capability_name not in self.enabled_capabilities:
            self.enabled_capabilities.append(capability_name)
    
    def disable_capability(self, capability_name: str) -> None:
        """Disable a capability"""
        if capability_name in self.enabled_capabilities:
            self.enabled_capabilities.remove(capability_name)
        
        if capability_name not in self.disabled_capabilities:
            self.disabled_capabilities.append(capability_name)
    
    def add_mcp_tool(self, tool_name: str) -> None:
        """Add an MCP tool to the configuration"""
        if tool_name not in self.mcp_tools:
            self.mcp_tools.append(tool_name)
    
    def remove_mcp_tool(self, tool_name: str) -> None:
        """Remove an MCP tool from the configuration"""
        if tool_name in self.mcp_tools:
            self.mcp_tools.remove(tool_name)
    
    def add_mcp_server(self, server_name: str) -> None:
        """Add an MCP server to the configuration"""
        if server_name not in self.mcp_servers:
            self.mcp_servers.append(server_name)
    
    def remove_mcp_server(self, server_name: str) -> None:
        """Remove an MCP server from the configuration"""
        if server_name in self.mcp_servers:
            self.mcp_servers.remove(server_name)
    
    def set_custom_config(self, key: str, value: Any) -> None:
        """Set a custom configuration value"""
        self.custom_config[key] = value
    
    def get_custom_config(self, key: str, default: Any = None) -> Any:
        """Get a custom configuration value"""
        return self.custom_config.get(key, default)
    
    def set_environment_variable(self, key: str, value: str) -> None:
        """Set an environment variable"""
        self.environment_variables[key] = value
    
    def get_environment_variable(self, key: str, default: str = None) -> str:
        """Get an environment variable"""
        return self.environment_variables.get(key, default)
    
    def merge_config(self, other_config: 'AgentConfig') -> 'AgentConfig':
        """Merge this configuration with another, returning a new config"""
        # Create a copy of this config
        merged = AgentConfig(**self.__dict__)
        
        # Merge lists
        merged.enabled_capabilities = list(set(self.enabled_capabilities + other_config.enabled_capabilities))
        merged.disabled_capabilities = list(set(self.disabled_capabilities + other_config.disabled_capabilities))
        merged.mcp_tools = list(set(self.mcp_tools + other_config.mcp_tools))
        merged.mcp_servers = list(set(self.mcp_servers + other_config.mcp_servers))
        merged.memory_types = list(set(self.memory_types + other_config.memory_types))
        merged.allowed_users = list(set(self.allowed_users + other_config.allowed_users))
        merged.allowed_roles = list(set(self.allowed_roles + other_config.allowed_roles))
        
        # Merge dictionaries
        merged.custom_config = {**self.custom_config, **other_config.custom_config}
        merged.environment_variables = {**self.environment_variables, **other_config.environment_variables}
        
        # Override scalar values with other_config values if they differ from defaults
        if other_config.max_concurrent_tasks != 5:
            merged.max_concurrent_tasks = other_config.max_concurrent_tasks
        if other_config.default_timeout != 300:
            merged.default_timeout = other_config.default_timeout
        if other_config.memory_limit != 1000:
            merged.memory_limit = other_config.memory_limit
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type,
            "version": self.version,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "default_timeout": self.default_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "memory_enabled": self.memory_enabled,
            "memory_limit": self.memory_limit,
            "memory_types": self.memory_types,
            "memory_consolidation_interval": self.memory_consolidation_interval,
            "message_queue_size": self.message_queue_size,
            "message_timeout": self.message_timeout,
            "heartbeat_interval": self.heartbeat_interval,
            "enabled_capabilities": self.enabled_capabilities,
            "disabled_capabilities": self.disabled_capabilities,
            "capability_timeout": self.capability_timeout,
            "mcp_tools": self.mcp_tools,
            "mcp_servers": self.mcp_servers,
            "mcp_timeout": self.mcp_timeout,
            "max_memory_usage_mb": self.max_memory_usage_mb,
            "max_cpu_usage_percent": self.max_cpu_usage_percent,
            "max_execution_time": self.max_execution_time,
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "metrics_interval": self.metrics_interval,
            "require_authentication": self.require_authentication,
            "allowed_users": self.allowed_users,
            "allowed_roles": self.allowed_roles,
            "custom_config": self.custom_config,
            "environment_variables": self.environment_variables,
            "persist_state": self.persist_state,
            "state_save_interval": self.state_save_interval,
            "backup_enabled": self.backup_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from dictionary"""
        return cls(**data)
    
    def copy(self) -> 'AgentConfig':
        """Create a copy of this configuration"""
        return AgentConfig.from_dict(self.to_dict())
    
    def __str__(self) -> str:
        """String representation"""
        return f"AgentConfig(name={self.name}, type={self.agent_type})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"AgentConfig(agent_id={self.agent_id}, name={self.name}, "
                f"type={self.agent_type}, version={self.version})")


# Factory functions for common configurations
def create_basic_agent_config(agent_type: str = "basic", name: str = None) -> AgentConfig:
    """Create a basic agent configuration"""
    return AgentConfig(
        agent_type=agent_type,
        name=name,
        enabled_capabilities=["text_processing", "communication"],
        memory_enabled=True,
        require_authentication=False
    )


def create_research_agent_config(name: str = None) -> AgentConfig:
    """Create a research agent configuration"""
    return AgentConfig(
        agent_type="research",
        name=name,
        enabled_capabilities=["text_processing", "web_search", "document_analysis", "memory_retrieval"],
        mcp_tools=["brave_search", "web_scraper"],
        memory_limit=2000,
        max_concurrent_tasks=3
    )


def create_code_agent_config(name: str = None) -> AgentConfig:
    """Create a code generation agent configuration"""
    return AgentConfig(
        agent_type="code",
        name=name,
        enabled_capabilities=["code_generation", "code_analysis", "tool_execution"],
        mcp_tools=["github", "filesystem"],
        memory_limit=1500,
        max_execution_time=1800  # 30 minutes for code tasks
    )


def create_conversation_agent_config(name: str = None) -> AgentConfig:
    """Create a conversational agent configuration"""
    return AgentConfig(
        agent_type="conversation",
        name=name,
        enabled_capabilities=["text_processing", "communication", "memory_retrieval"],
        memory_enabled=True,
        memory_limit=5000,
        heartbeat_interval=15  # More frequent heartbeat for interactive use
    )
