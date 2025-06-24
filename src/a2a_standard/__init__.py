"""
Google A2A Protocol Standard Implementation

This module implements the official Google Agent2Agent (A2A) Protocol
as specified at https://google-a2a.github.io/A2A/latest/

The A2A protocol enables seamless communication and collaboration between
independent AI agents using standardized JSON-RPC 2.0 over HTTP(S).

Key Features:
- Agent Cards for capability discovery
- Task-based workflow management
- Multi-modal communication (text, files, structured data)
- Streaming support via Server-Sent Events
- Push notifications for long-running tasks
- Enterprise security with authentication/authorization
- MCP compatibility (complementary protocols)
"""

from .agent_card import AgentCard, AgentProvider, AgentCapabilities, AgentSkill, SecurityScheme
from .task import Task, TaskStatus, TaskState, TaskStatusUpdateEvent, TaskArtifactUpdateEvent
from .message import Message, Part, TextPart, FilePart, DataPart
from .artifact import Artifact

# Optional imports - only import if files exist
try:
    from .server import A2AServer
except ImportError:
    A2AServer = None

try:
    from .client import A2AClient
except ImportError:
    A2AClient = None

try:
    from .discovery import AgentDiscoveryService
except ImportError:
    AgentDiscoveryService = None

try:
    from .push_notifications import PushNotificationConfig, PushNotificationAuthenticationInfo
except ImportError:
    PushNotificationConfig = None
    PushNotificationAuthenticationInfo = None

try:
    from .streaming import StreamingHandler
except ImportError:
    StreamingHandler = None

try:
    from .errors import A2AError, TaskNotFoundError, TaskNotCancelableError
except ImportError:
    # Create basic error classes if not available
    class A2AError(Exception):
        def __init__(self, message: str, error_code: str = None):
            super().__init__(message)
            self.error_code = error_code

    class TaskNotFoundError(A2AError):
        pass

    class TaskNotCancelableError(A2AError):
        pass

__version__ = "0.2.0"  # Following Google A2A specification version

__all__ = [
    # Core Protocol Objects
    "AgentCard",
    "AgentProvider", 
    "AgentCapabilities",
    "AgentSkill",
    "SecurityScheme",
    
    # Task Management
    "Task",
    "TaskStatus",
    "TaskState",
    "TaskStatusUpdateEvent",
    "TaskArtifactUpdateEvent",
    
    # Communication
    "Message",
    "Part",
    "TextPart",
    "FilePart", 
    "DataPart",
    "Artifact",
    
    # Server/Client
    "A2AServer",
    "A2AClient",
    "AgentDiscoveryService",
    
    # Advanced Features
    "PushNotificationConfig",
    "PushNotificationAuthenticationInfo",
    "StreamingHandler",
    
    # Error Handling
    "A2AError",
    "TaskNotFoundError",
    "TaskNotCancelableError",
]
