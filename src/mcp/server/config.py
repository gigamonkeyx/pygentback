"""
MCP Server Configuration

This module defines configuration classes and types for MCP servers.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class MCPServerType(Enum):
    """Types of MCP servers"""
    FILESYSTEM = "filesystem"
    POSTGRES = "postgres"
    GITHUB = "github"
    BRAVE_SEARCH = "brave-search"
    WEB_SCRAPER = "web-scraper"
    CUSTOM = "custom"


class MCPTransportType(Enum):
    """MCP transport types"""
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    WEBSOCKET = "websocket"
    TCP = "tcp"


class MCPServerStatus(Enum):
    """MCP server status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class MCPServerConfig:
    """
    Configuration for an MCP server.
    
    This class defines all configurable aspects of an MCP server including
    connection parameters, capabilities, and runtime settings.
    """
    
    # Basic identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    server_type: MCPServerType = MCPServerType.CUSTOM
    version: str = "1.0.0"
    
    # Connection configuration
    command: List[str] = field(default_factory=list)
    transport: MCPTransportType = MCPTransportType.STDIO
    host: Optional[str] = None
    port: Optional[int] = None
    path: Optional[str] = None
    
    # Capabilities and tools
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    
    # Runtime configuration
    auto_start: bool = True
    restart_on_failure: bool = True
    max_restarts: int = 3
    restart_delay: float = 5.0
    timeout: int = 30
    
    # Environment and working directory
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_logging: bool = True
    log_file: Optional[str] = None
    
    # Security settings
    require_authentication: bool = False
    api_key: Optional[str] = None
    allowed_origins: List[str] = field(default_factory=list)
    
    # Custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        if not self.name:
            self.name = f"{self.server_type.value}_{self.id[:8]}"
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not self.command and self.transport == MCPTransportType.STDIO:
            raise ValueError("Command is required for STDIO transport")
        
        if self.transport in [MCPTransportType.HTTP, MCPTransportType.WEBSOCKET, MCPTransportType.TCP]:
            if not self.host:
                raise ValueError(f"Host is required for {self.transport.value} transport")
            if not self.port:
                raise ValueError(f"Port is required for {self.transport.value} transport")
        
        if self.max_restarts < 0:
            raise ValueError("max_restarts cannot be negative")
        
        if self.restart_delay < 0:
            raise ValueError("restart_delay cannot be negative")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
    
    def add_capability(self, capability: str) -> None:
        """Add a capability to the server"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            self.updated_at = datetime.utcnow()
    
    def remove_capability(self, capability: str) -> None:
        """Remove a capability from the server"""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
            self.updated_at = datetime.utcnow()
    
    def add_tool(self, tool: str) -> None:
        """Add a tool to the server"""
        if tool not in self.tools:
            self.tools.append(tool)
            self.updated_at = datetime.utcnow()
    
    def remove_tool(self, tool: str) -> None:
        """Remove a tool from the server"""
        if tool in self.tools:
            self.tools.remove(tool)
            self.updated_at = datetime.utcnow()
    
    def add_resource(self, resource: str) -> None:
        """Add a resource to the server"""
        if resource not in self.resources:
            self.resources.append(resource)
            self.updated_at = datetime.utcnow()
    
    def remove_resource(self, resource: str) -> None:
        """Remove a resource from the server"""
        if resource in self.resources:
            self.resources.remove(resource)
            self.updated_at = datetime.utcnow()
    
    def set_environment_variable(self, key: str, value: str) -> None:
        """Set an environment variable"""
        self.environment_variables[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_environment_variable(self, key: str, default: str = None) -> str:
        """Get an environment variable"""
        return self.environment_variables.get(key, default)
    
    def set_custom_config(self, key: str, value: Any) -> None:
        """Set a custom configuration value"""
        self.custom_config[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_custom_config(self, key: str, default: Any = None) -> Any:
        """Get a custom configuration value"""
        return self.custom_config.get(key, default)
    
    def get_connection_string(self) -> str:
        """Get connection string for the server"""
        if self.transport == MCPTransportType.STDIO:
            return f"stdio://{' '.join(self.command)}"
        elif self.transport == MCPTransportType.HTTP:
            return f"http://{self.host}:{self.port}{self.path or ''}"
        elif self.transport == MCPTransportType.WEBSOCKET:
            return f"ws://{self.host}:{self.port}{self.path or ''}"
        elif self.transport == MCPTransportType.TCP:
            return f"tcp://{self.host}:{self.port}"
        else:
            return f"{self.transport.value}://unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "server_type": self.server_type.value,
            "version": self.version,
            "command": self.command,
            "transport": self.transport.value,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "resources": self.resources,
            "auto_start": self.auto_start,
            "restart_on_failure": self.restart_on_failure,
            "max_restarts": self.max_restarts,
            "restart_delay": self.restart_delay,
            "timeout": self.timeout,
            "working_directory": self.working_directory,
            "environment_variables": self.environment_variables,
            "log_level": self.log_level,
            "enable_logging": self.enable_logging,
            "log_file": self.log_file,
            "require_authentication": self.require_authentication,
            "api_key": self.api_key,
            "allowed_origins": self.allowed_origins,
            "custom_config": self.custom_config,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPServerConfig':
        """Create configuration from dictionary"""
        config = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            server_type=MCPServerType(data.get("server_type", "custom")),
            version=data.get("version", "1.0.0"),
            command=data.get("command", []),
            transport=MCPTransportType(data.get("transport", "stdio")),
            host=data.get("host"),
            port=data.get("port"),
            path=data.get("path"),
            capabilities=data.get("capabilities", []),
            tools=data.get("tools", []),
            resources=data.get("resources", []),
            auto_start=data.get("auto_start", True),
            restart_on_failure=data.get("restart_on_failure", True),
            max_restarts=data.get("max_restarts", 3),
            restart_delay=data.get("restart_delay", 5.0),
            timeout=data.get("timeout", 30),
            working_directory=data.get("working_directory"),
            environment_variables=data.get("environment_variables", {}),
            log_level=data.get("log_level", "INFO"),
            enable_logging=data.get("enable_logging", True),
            log_file=data.get("log_file"),
            require_authentication=data.get("require_authentication", False),
            api_key=data.get("api_key"),
            allowed_origins=data.get("allowed_origins", []),
            custom_config=data.get("custom_config", {}),
            metadata=data.get("metadata", {})
        )
        
        # Parse timestamps
        if "created_at" in data:
            config.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            config.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return config
    
    def copy(self) -> 'MCPServerConfig':
        """Create a copy of this configuration"""
        return MCPServerConfig.from_dict(self.to_dict())
    
    def __str__(self) -> str:
        """String representation"""
        return f"MCPServerConfig(name={self.name}, type={self.server_type.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"MCPServerConfig(id={self.id}, name={self.name}, "
                f"type={self.server_type.value}, transport={self.transport.value})")


# Factory functions for common server configurations
def create_filesystem_server_config(name: str = None, root_path: str = ".") -> MCPServerConfig:
    """Create a filesystem server configuration"""
    return MCPServerConfig(
        name=name or "filesystem",
        server_type=MCPServerType.FILESYSTEM,
        command=["mcp-server-filesystem", root_path],
        capabilities=["file_read", "file_write", "directory_list"],
        tools=["read_file", "write_file", "list_directory", "create_directory"],
        custom_config={"root_path": root_path}
    )


def create_postgres_server_config(name: str = None, connection_string: str = None) -> MCPServerConfig:
    """Create a PostgreSQL server configuration"""
    return MCPServerConfig(
        name=name or "postgres",
        server_type=MCPServerType.POSTGRES,
        command=["mcp-server-postgres"],
        capabilities=["database_query", "database_execute"],
        tools=["query", "execute", "describe_table", "list_tables"],
        environment_variables={"DATABASE_URL": connection_string or "postgresql://localhost/postgres"},
        custom_config={"connection_string": connection_string}
    )


def create_github_server_config(name: str = None, token: str = None) -> MCPServerConfig:
    """Create a GitHub server configuration"""
    return MCPServerConfig(
        name=name or "github",
        server_type=MCPServerType.GITHUB,
        command=["mcp-server-github"],
        capabilities=["repository_access", "issue_management", "pull_requests"],
        tools=["create_repository", "get_file", "create_issue", "list_issues"],
        environment_variables={"GITHUB_TOKEN": token or ""},
        custom_config={"token": token}
    )


def create_brave_search_config(name: str = None, api_key: str = None) -> MCPServerConfig:
    """Create a Brave Search server configuration"""
    return MCPServerConfig(
        name=name or "brave_search",
        server_type=MCPServerType.BRAVE_SEARCH,
        command=["mcp-server-brave-search"],
        capabilities=["web_search", "news_search"],
        tools=["search", "news_search"],
        environment_variables={"BRAVE_API_KEY": api_key or ""},
        custom_config={"api_key": api_key}
    )
