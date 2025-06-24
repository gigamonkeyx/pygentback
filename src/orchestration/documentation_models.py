"""
Documentation Orchestration Models

Data models and enums for documentation workflow orchestration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

class DocumentationWorkflowType(Enum):
    """Types of documentation workflows"""
    BUILD_AND_SYNC = "build_and_sync"
    DEVELOPMENT_MODE = "development_mode"
    PRODUCTION_DEPLOY = "production_deploy"
    HEALTH_CHECK = "health_check"
    CONFLICT_RESOLUTION = "conflict_resolution"

class DocumentationTaskType(Enum):
    """Types of documentation tasks"""
    VALIDATE_ENVIRONMENT = "validate_environment"
    RESOLVE_CONFLICTS = "resolve_conflicts"
    BUILD_VITEPRESS = "build_vitepress"
    SYNC_FRONTEND = "sync_frontend"
    VERIFY_INTEGRATION = "verify_integration"
    START_DEV_SERVER = "start_dev_server"
    WATCH_CHANGES = "watch_changes"
    HEALTH_MONITOR = "health_monitor"

class DocumentationTaskStatus(Enum):
    """Status of documentation tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"

@dataclass
class DocumentationConfig:
    """Configuration for documentation orchestration"""
    # Paths
    docs_source_path: Path = field(default_factory=lambda: Path("src/docs"))
    docs_build_path: Path = field(default_factory=lambda: Path("src/docs/dist"))
    frontend_public_path: Path = field(default_factory=lambda: Path("pygent-repo/public"))
    frontend_docs_path: Path = field(default_factory=lambda: Path("pygent-repo/public/docs"))
    
    # Build configuration
    vitepress_port: int = 3001
    build_timeout: int = 300  # 5 minutes
    sync_timeout: int = 60    # 1 minute
    
    # Conflict resolution
    auto_resolve_conflicts: bool = True
    backup_on_conflict: bool = True
    
    # Development mode
    enable_hot_reload: bool = True
    watch_file_patterns: List[str] = field(default_factory=lambda: ["*.md", "*.vue", "*.ts", "*.js"])
    
    # Health monitoring
    health_check_interval: int = 30  # seconds
    max_retry_attempts: int = 3
    
    # Integration
    integrate_with_ui: bool = True
    update_navigation: bool = True

@dataclass
class DocumentationTask:
    """Individual documentation task definition"""
    task_id: str
    task_type: DocumentationTaskType
    name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: DocumentationTaskStatus = DocumentationTaskStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class DocumentationWorkflow:
    """Documentation workflow definition"""
    workflow_id: str
    workflow_type: DocumentationWorkflowType
    name: str
    description: str
    tasks: List[DocumentationTask] = field(default_factory=list)
    config: DocumentationConfig = field(default_factory=DocumentationConfig)
    status: DocumentationTaskStatus = DocumentationTaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: float = 0.0
    current_task: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConflictResolutionResult:
    """Result of conflict resolution process"""
    conflicts_found: List[str]
    conflicts_resolved: List[str]
    resolution_actions: List[str]
    backup_created: bool = False
    backup_path: Optional[Path] = None
    success: bool = False
    error_message: Optional[str] = None

@dataclass
class BuildResult:
    """Result of VitePress build process"""
    success: bool
    build_time: float
    output_size: int
    files_generated: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    build_log: str = ""

@dataclass
class SyncResult:
    """Result of frontend synchronization"""
    success: bool
    files_copied: int
    sync_time: float
    target_path: Path
    manifest_created: bool = False
    errors: List[str] = field(default_factory=list)

@dataclass
class HealthCheckResult:
    """Result of health check process"""
    overall_health: str  # "healthy", "warning", "critical"
    checks_performed: List[str]
    issues_found: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DocumentationMetrics:
    """Metrics for documentation system"""
    total_builds: int = 0
    successful_builds: int = 0
    failed_builds: int = 0
    average_build_time: float = 0.0
    total_syncs: int = 0
    successful_syncs: int = 0
    conflicts_resolved: int = 0
    uptime_percentage: float = 100.0
    last_successful_build: Optional[datetime] = None
    last_health_check: Optional[datetime] = None

class ConflictType(Enum):
    """Types of configuration conflicts"""
    TAILWIND_CSS = "tailwind_css"
    POSTCSS = "postcss"
    VITE_CONFIG = "vite_config"
    DEPENDENCY = "dependency"
    PATH = "path"
    ENVIRONMENT = "environment"

@dataclass
class ConfigurationConflict:
    """Configuration conflict definition"""
    conflict_id: str
    conflict_type: ConflictType
    description: str
    affected_files: List[Path]
    severity: str  # "low", "medium", "high", "critical"
    auto_resolvable: bool
    resolution_strategy: str
    detected_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ResolutionAction:
    """Action taken to resolve a conflict"""
    action_id: str
    conflict_id: str
    action_type: str  # "backup", "modify", "delete", "create"
    target_file: Path
    description: str
    executed_at: datetime = field(default_factory=datetime.utcnow)
    success: bool = False
    error_message: Optional[str] = None

class DocumentationEventType(Enum):
    """Types of documentation events"""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    BUILD_STARTED = "build_started"
    BUILD_COMPLETED = "build_completed"
    SYNC_STARTED = "sync_started"
    SYNC_COMPLETED = "sync_completed"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    HEALTH_CHECK_COMPLETED = "health_check_completed"
    FILE_CHANGED = "file_changed"

@dataclass
class DocumentationEvent:
    """Documentation system event"""
    event_id: str
    event_type: DocumentationEventType
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "documentation_orchestrator"

# Workflow template definitions
WORKFLOW_TEMPLATES = {
    "build_and_sync": {
        "name": "Build and Sync Documentation",
        "description": "Complete build and synchronization workflow",
        "tasks": [
            "validate_environment",
            "resolve_conflicts", 
            "build_vitepress",
            "sync_frontend",
            "verify_integration"
        ]
    },
    "development_mode": {
        "name": "Development Mode",
        "description": "Start development server with hot reload",
        "tasks": [
            "validate_environment",
            "resolve_conflicts",
            "start_dev_server", 
            "watch_changes",
            "health_monitor"
        ]
    },
    "production_deploy": {
        "name": "Production Deployment",
        "description": "Production-ready build and deployment",
        "tasks": [
            "validate_environment",
            "resolve_conflicts",
            "build_vitepress",
            "sync_frontend", 
            "verify_integration"
        ]
    },
    "health_check": {
        "name": "System Health Check",
        "description": "Comprehensive system health verification",
        "tasks": [
            "validate_environment",
            "verify_integration",
            "health_monitor"
        ]
    }
}

# Default configuration values
DEFAULT_CONFIG = {
    "build_timeout": 300,
    "sync_timeout": 60,
    "health_check_interval": 30,
    "max_retry_attempts": 3,
    "auto_resolve_conflicts": True,
    "backup_on_conflict": True,
    "enable_hot_reload": True,
    "integrate_with_ui": True,
    "update_navigation": True
}
