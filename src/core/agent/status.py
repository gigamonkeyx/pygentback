"""
Agent Status Management

This module defines agent status types and status management functionality.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class AgentStatus(Enum):
    """Agent status enumeration"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"


@dataclass
class AgentStatusInfo:
    """Detailed agent status information"""
    status: AgentStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    
    def update_status(self, new_status: AgentStatus, message: str = "", 
                     details: Optional[Dict[str, Any]] = None) -> None:
        """Update the agent status"""
        self.status = new_status
        self.message = message
        if details:
            self.details.update(details)
        self.last_updated = datetime.utcnow()
    
    def add_error(self, error_message: str) -> None:
        """Add an error to the status"""
        self.error_count += 1
        self.last_error = error_message
        self.last_updated = datetime.utcnow()
    
    def clear_errors(self) -> None:
        """Clear error information"""
        self.error_count = 0
        self.last_error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "last_updated": self.last_updated.isoformat(),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "uptime_seconds": self.uptime_seconds
        }


class AgentStatusManager:
    """Manages agent status transitions and validation"""
    
    # Valid status transitions
    VALID_TRANSITIONS = {
        AgentStatus.INITIALIZING: [AgentStatus.ACTIVE, AgentStatus.ERROR, AgentStatus.STOPPED],
        AgentStatus.ACTIVE: [AgentStatus.IDLE, AgentStatus.BUSY, AgentStatus.ERROR, 
                           AgentStatus.STOPPING, AgentStatus.MAINTENANCE],
        AgentStatus.IDLE: [AgentStatus.ACTIVE, AgentStatus.BUSY, AgentStatus.ERROR, 
                         AgentStatus.STOPPING, AgentStatus.MAINTENANCE],
        AgentStatus.BUSY: [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.ERROR, 
                         AgentStatus.STOPPING],
        AgentStatus.ERROR: [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.STOPPING, 
                          AgentStatus.STOPPED, AgentStatus.MAINTENANCE],
        AgentStatus.STOPPING: [AgentStatus.STOPPED, AgentStatus.ERROR],
        AgentStatus.STOPPED: [AgentStatus.INITIALIZING],
        AgentStatus.MAINTENANCE: [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.STOPPING]
    }
    
    def __init__(self):
        self.status_info = AgentStatusInfo(AgentStatus.INITIALIZING)
        self.start_time = datetime.utcnow()
    
    def can_transition_to(self, new_status: AgentStatus) -> bool:
        """Check if transition to new status is valid"""
        current_status = self.status_info.status
        return new_status in self.VALID_TRANSITIONS.get(current_status, [])
    
    def transition_to(self, new_status: AgentStatus, message: str = "", 
                     details: Optional[Dict[str, Any]] = None) -> bool:
        """Transition to new status if valid"""
        if not self.can_transition_to(new_status):
            return False
        
        # Update uptime
        self.status_info.uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Update status
        self.status_info.update_status(new_status, message, details)
        return True
    
    def force_transition_to(self, new_status: AgentStatus, message: str = "", 
                           details: Optional[Dict[str, Any]] = None) -> None:
        """Force transition to new status (bypasses validation)"""
        self.status_info.uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        self.status_info.update_status(new_status, message, details)
    
    def get_status(self) -> AgentStatus:
        """Get current status"""
        return self.status_info.status
    
    def get_status_info(self) -> AgentStatusInfo:
        """Get detailed status information"""
        # Update uptime before returning
        self.status_info.uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        return self.status_info
    
    def is_active(self) -> bool:
        """Check if agent is in an active state"""
        return self.status_info.status in [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.BUSY]
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return self.status_info.status in [AgentStatus.ACTIVE, AgentStatus.IDLE]
    
    def is_error_state(self) -> bool:
        """Check if agent is in error state"""
        return self.status_info.status == AgentStatus.ERROR
    
    def add_error(self, error_message: str) -> None:
        """Add error and transition to error state if needed"""
        self.status_info.add_error(error_message)
        if self.can_transition_to(AgentStatus.ERROR):
            self.transition_to(AgentStatus.ERROR, f"Error occurred: {error_message}")
    
    def clear_errors(self) -> None:
        """Clear errors and transition out of error state if possible"""
        self.status_info.clear_errors()
        if self.status_info.status == AgentStatus.ERROR:
            if self.can_transition_to(AgentStatus.IDLE):
                self.transition_to(AgentStatus.IDLE, "Errors cleared")
    
    def reset(self) -> None:
        """Reset status manager"""
        self.status_info = AgentStatusInfo(AgentStatus.INITIALIZING)
        self.start_time = datetime.utcnow()


# Utility functions for status management
def is_terminal_status(status: AgentStatus) -> bool:
    """Check if status is terminal (no further transitions expected)"""
    return status in [AgentStatus.STOPPED]


def is_operational_status(status: AgentStatus) -> bool:
    """Check if status indicates agent is operational"""
    return status in [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.BUSY]


def get_status_description(status: AgentStatus) -> str:
    """Get human-readable description of status"""
    descriptions = {
        AgentStatus.INITIALIZING: "Agent is starting up and initializing",
        AgentStatus.ACTIVE: "Agent is active and ready to process tasks",
        AgentStatus.IDLE: "Agent is idle and waiting for tasks",
        AgentStatus.BUSY: "Agent is currently processing a task",
        AgentStatus.ERROR: "Agent has encountered an error",
        AgentStatus.STOPPING: "Agent is shutting down",
        AgentStatus.STOPPED: "Agent has been stopped",
        AgentStatus.MAINTENANCE: "Agent is in maintenance mode"
    }
    return descriptions.get(status, "Unknown status")


def get_status_color(status: AgentStatus) -> str:
    """Get color code for status (for UI display)"""
    colors = {
        AgentStatus.INITIALIZING: "yellow",
        AgentStatus.ACTIVE: "green",
        AgentStatus.IDLE: "blue",
        AgentStatus.BUSY: "orange",
        AgentStatus.ERROR: "red",
        AgentStatus.STOPPING: "orange",
        AgentStatus.STOPPED: "gray",
        AgentStatus.MAINTENANCE: "purple"
    }
    return colors.get(status, "gray")
