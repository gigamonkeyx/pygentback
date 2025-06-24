"""
Database Schema for MCP Tool Storage

Creates the necessary database tables and models for storing MCP tool metadata
according to the MCP specification requirements.
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, JSON, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class MCPServerModel(Base):
    """Database model for MCP servers"""
    __tablename__ = 'mcp_servers'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, unique=True)
    command = Column(JSON, nullable=False)  # Store as JSON array
    server_type = Column(String(50), nullable=False)
    transport = Column(String(20), nullable=False, default='stdio')
    status = Column(String(20), nullable=False, default='stopped')
    capabilities = Column(JSON, nullable=True)  # Server capabilities
    custom_config = Column(JSON, nullable=True)
    auto_start = Column(Boolean, default=True)
    restart_on_failure = Column(Boolean, default=True)
    max_restarts = Column(Integer, default=3)
    timeout = Column(Integer, default=30)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_heartbeat = Column(DateTime, nullable=True)
    
    # Tracking
    start_count = Column(Integer, default=0)
    restart_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    
    # Relationships
    tools = relationship("MCPToolModel", back_populates="server", cascade="all, delete-orphan")
    resources = relationship("MCPResourceModel", back_populates="server", cascade="all, delete-orphan")
    prompts = relationship("MCPPromptModel", back_populates="server", cascade="all, delete-orphan")


class MCPToolModel(Base):
    """Database model for MCP tools discovered from servers"""
    __tablename__ = 'mcp_tools'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    server_id = Column(String(36), ForeignKey('mcp_servers.id'), nullable=False)
    
    # Tool definition from MCP spec
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    input_schema = Column(JSON, nullable=True)  # JSON Schema for tool inputs
    annotations = Column(JSON, nullable=True)  # Tool annotations
    
    # Discovery metadata
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_available = Column(Boolean, default=True)
    
    # Usage tracking
    call_count = Column(Integer, default=0)
    last_called = Column(DateTime, nullable=True)
    average_response_time = Column(Integer, nullable=True)  # milliseconds
    
    # Relationships
    server = relationship("MCPServerModel", back_populates="tools")
    
    # Unique constraint: one tool name per server
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )


class MCPResourceModel(Base):
    """Database model for MCP resources"""
    __tablename__ = 'mcp_resources'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    server_id = Column(String(36), ForeignKey('mcp_servers.id'), nullable=False)
    
    # Resource definition
    uri = Column(String(1000), nullable=False)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    mime_type = Column(String(100), nullable=True)
    annotations = Column(JSON, nullable=True)
    
    # Discovery metadata
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_available = Column(Boolean, default=True)
    
    # Relationships
    server = relationship("MCPServerModel", back_populates="resources")


class MCPPromptModel(Base):
    """Database model for MCP prompts"""
    __tablename__ = 'mcp_prompts'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    server_id = Column(String(36), ForeignKey('mcp_servers.id'), nullable=False)
    
    # Prompt definition
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    arguments = Column(JSON, nullable=True)  # Argument schema
    annotations = Column(JSON, nullable=True)
    
    # Discovery metadata
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_available = Column(Boolean, default=True)
    
    # Usage tracking
    use_count = Column(Integer, default=0)
    last_used = Column(DateTime, nullable=True)
    
    # Relationships
    server = relationship("MCPServerModel", back_populates="prompts")


class MCPToolCallLog(Base):
    """Log of MCP tool calls for analysis and evolution"""
    __tablename__ = 'mcp_tool_calls'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tool_id = Column(String(36), ForeignKey('mcp_tools.id'), nullable=False)
    server_id = Column(String(36), ForeignKey('mcp_servers.id'), nullable=False)
    
    # Call details
    agent_id = Column(String(255), nullable=True)  # Which agent made the call
    arguments = Column(JSON, nullable=True)  # Arguments passed to tool
    response_data = Column(JSON, nullable=True)  # Tool response
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Performance tracking
    response_time_ms = Column(Integer, nullable=True)
    memory_usage_mb = Column(Integer, nullable=True)
    
    # Timestamps
    called_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Evolution data
    usage_context = Column(JSON, nullable=True)  # Context when tool was used
    effectiveness_score = Column(Integer, nullable=True)  # 1-10 effectiveness rating


# SQL for creating indexes for performance
CREATE_INDEXES_SQL = """
-- Performance indexes for MCP tool discovery and usage
CREATE INDEX IF NOT EXISTS idx_mcp_servers_status ON mcp_servers(status);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_type ON mcp_servers(server_type);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_heartbeat ON mcp_servers(last_heartbeat);

CREATE INDEX IF NOT EXISTS idx_mcp_tools_server_id ON mcp_tools(server_id);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_name ON mcp_tools(name);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_available ON mcp_tools(is_available);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_server_name ON mcp_tools(server_id, name);

CREATE INDEX IF NOT EXISTS idx_mcp_resources_server_id ON mcp_resources(server_id);
CREATE INDEX IF NOT EXISTS idx_mcp_resources_uri ON mcp_resources(uri);
CREATE INDEX IF NOT EXISTS idx_mcp_resources_available ON mcp_resources(is_available);

CREATE INDEX IF NOT EXISTS idx_mcp_prompts_server_id ON mcp_prompts(server_id);
CREATE INDEX IF NOT EXISTS idx_mcp_prompts_name ON mcp_prompts(name);
CREATE INDEX IF NOT EXISTS idx_mcp_prompts_available ON mcp_prompts(is_available);

CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_tool_id ON mcp_tool_calls(tool_id);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_server_id ON mcp_tool_calls(server_id);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_agent_id ON mcp_tool_calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_called_at ON mcp_tool_calls(called_at);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_calls_success ON mcp_tool_calls(success);
"""

# Export all models
__all__ = [
    'Base',
    'MCPServerModel', 
    'MCPToolModel',
    'MCPResourceModel', 
    'MCPPromptModel',
    'MCPToolCallLog',
    'CREATE_INDEXES_SQL'
]
