"""
Documentation Orchestrator

Intelligent orchestration system for PyGent Factory documentation workflows.
Integrates with existing orchestration infrastructure for enterprise-grade
documentation management.
"""

import asyncio
import logging
import shutil
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from .documentation_models import (
    DocumentationWorkflow, DocumentationTask, DocumentationConfig,
    DocumentationWorkflowType, DocumentationTaskType, DocumentationTaskStatus,
    ConflictResolutionResult, BuildResult, SyncResult
)
from .coordination_models import TaskRequest, TaskPriority

# Handle integration imports gracefully
try:
    from ..integration.events import EventBus, Event, EventType
    INTEGRATION_EVENTS_AVAILABLE = True
except ImportError:
    EventBus = None
    Event = None
    EventType = None
    INTEGRATION_EVENTS_AVAILABLE = False

try:
    from ..integration.workflows import WorkflowManager, WorkflowExecutor
    INTEGRATION_WORKFLOWS_AVAILABLE = True
except ImportError:
    WorkflowManager = None
    WorkflowExecutor = None
    INTEGRATION_WORKFLOWS_AVAILABLE = False

from .build_coordinator import BuildCoordinator
from .sync_manager import SyncManager
from .conflict_resolver import ConflictResolver
from .health_monitor import HealthMonitor
from .intelligent_docs_builder import IntelligentDocsBuilder

logger = logging.getLogger(__name__)


class IntegrationError(Exception):
    """Raised when PyGent Factory integration components are missing"""
    pass





class DocumentationOrchestrator:
    """
    Main orchestrator for documentation workflows.
    
    Integrates with PyGent Factory's orchestration infrastructure to provide
    intelligent, self-healing documentation management.
    """
    
    def __init__(self, orchestration_manager, config: Optional[DocumentationConfig] = None):
        self.orchestration_manager = orchestration_manager
        self.config = config or DocumentationConfig()

        # Get references to existing components - NO FALLBACKS
        self.event_bus = orchestration_manager.event_bus
        self.workflow_manager = orchestration_manager.workflow_manager
        self.task_dispatcher = orchestration_manager.task_dispatcher
        self.metrics_collector = orchestration_manager.metrics_collector
        self.pygent_integration = orchestration_manager.pygent_integration

        # Validate all dependencies are present
        self._validate_dependencies()
        
        # Documentation-specific components
        self.build_coordinator = BuildCoordinator(self.config, self.event_bus)
        self.sync_manager = SyncManager(self.config, self.event_bus)
        self.conflict_resolver = ConflictResolver(self.config, self.event_bus)
        self.health_monitor = HealthMonitor(self.config, self.event_bus)

        # Initialize intelligent build system
        self.intelligent_builder = IntelligentDocsBuilder(
            docs_path=self.config.docs_source_path,
            output_path=self.config.docs_build_path,
            cache_path=self.config.docs_source_path / "public" / "diagrams"
        )
        
        # State management
        self.active_workflows: Dict[str, DocumentationWorkflow] = {}
        self.workflow_history: List[DocumentationWorkflow] = []
        self.is_running = False
        
        # Event handlers (will be registered during start())
        self._event_handlers_registered = False

        # Initialize workflow templates
        self._initialize_workflow_templates()

    def _validate_dependencies(self):
        """Validate all required dependencies are available"""
        required_components = {
            'event_bus': self.event_bus,
            'workflow_manager': self.workflow_manager,
            'task_dispatcher': self.task_dispatcher,
            'metrics_collector': self.metrics_collector,
            'pygent_integration': self.pygent_integration
        }

        missing = [name for name, component in required_components.items() if component is None]
        if missing:
            raise IntegrationError(f"Missing required PyGent Factory components: {missing}")

    async def start(self):
        """Start the documentation orchestrator"""
        try:
            logger.info("Starting Documentation Orchestrator...")
            
            # Start sub-components
            await self.build_coordinator.start()
            await self.sync_manager.start()
            await self.conflict_resolver.start()
            await self.health_monitor.start()
            
            # Register with orchestration manager
            await self._register_with_orchestration_manager()
            
            self.is_running = True
            logger.info("Documentation Orchestrator started successfully")
            
            # Emit startup event
            if INTEGRATION_EVENTS_AVAILABLE and Event and EventType:
                await self.event_bus.publish(Event(
                    event_id=f"docs_start_{int(datetime.utcnow().timestamp())}",
                    event_type=EventType.COMPONENT_REGISTERED,
                    source="documentation_orchestrator",
                    timestamp=datetime.utcnow(),
                    data={"component": "documentation_orchestrator", "status": "started"}
                ))
            else:
                await self.event_bus.publish_event(
                    event_type="component_registered",
                    source="documentation_orchestrator",
                    data={"component": "documentation_orchestrator", "status": "started"}
                )
            
        except Exception as e:
            logger.error(f"Failed to start Documentation Orchestrator: {e}")
            raise
    
    async def stop(self):
        """Stop the documentation orchestrator"""
        try:
            logger.info("Stopping Documentation Orchestrator...")
            
            # Cancel active workflows
            for workflow_id in list(self.active_workflows.keys()):
                await self.cancel_workflow(workflow_id)
            
            # Stop sub-components
            await self.health_monitor.stop()
            await self.conflict_resolver.stop()
            await self.sync_manager.stop()
            await self.build_coordinator.stop()
            
            self.is_running = False
            logger.info("Documentation Orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Documentation Orchestrator: {e}")
    
    async def execute_workflow(self, workflow_type: DocumentationWorkflowType, 
                             parameters: Optional[Dict[str, Any]] = None) -> str:
        """Execute a documentation workflow"""
        workflow_id = f"docs_{workflow_type.value}_{int(datetime.utcnow().timestamp())}"
        
        try:
            # Create workflow from template
            workflow = await self._create_workflow_from_template(
                workflow_id, workflow_type, parameters or {}
            )
            
            # Register workflow
            self.active_workflows[workflow_id] = workflow
            
            # Start execution
            await self._execute_workflow_async(workflow)
            
            logger.info(f"Started documentation workflow: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to start workflow {workflow_id}: {e}")
            raise
    
    async def _execute_workflow_async(self, workflow: DocumentationWorkflow):
        """Execute workflow asynchronously"""
        try:
            workflow.status = DocumentationTaskStatus.RUNNING
            workflow.started_at = datetime.utcnow()
            
            # Emit workflow started event
            if INTEGRATION_EVENTS_AVAILABLE and Event and EventType:
                await self.event_bus.publish(Event(
                    event_id=f"workflow_start_{workflow.workflow_id}",
                    event_type=EventType.WORKFLOW_STARTED,
                    source="documentation_orchestrator",
                    timestamp=datetime.utcnow(),
                    data={"workflow_id": workflow.workflow_id, "type": workflow.workflow_type.value}
                ))
            else:
                await self.event_bus.publish_event(
                    event_type="workflow_started",
                    source="documentation_orchestrator",
                    data={"workflow_id": workflow.workflow_id, "type": workflow.workflow_type.value}
                )
            
            # Execute tasks in dependency order
            for task in workflow.tasks:
                await self._execute_task(workflow, task)
                
                if task.status == DocumentationTaskStatus.FAILED:
                    if task.task_type in [DocumentationTaskType.VALIDATE_ENVIRONMENT, 
                                        DocumentationTaskType.BUILD_VITEPRESS]:
                        # Critical task failed, stop workflow
                        workflow.status = DocumentationTaskStatus.FAILED
                        break
            
            # Update workflow status
            if workflow.status != DocumentationTaskStatus.FAILED:
                workflow.status = DocumentationTaskStatus.COMPLETED
                workflow.progress_percentage = 100.0
            
            workflow.completed_at = datetime.utcnow()
            
            # Move to history
            self.workflow_history.append(workflow)
            if workflow.workflow_id in self.active_workflows:
                del self.active_workflows[workflow.workflow_id]
            
            # Emit completion event
            if INTEGRATION_EVENTS_AVAILABLE and Event and EventType:
                await self.event_bus.publish(Event(
                    event_id=f"workflow_complete_{workflow.workflow_id}",
                    event_type=EventType.WORKFLOW_COMPLETED,
                    source="documentation_orchestrator",
                    timestamp=datetime.utcnow(),
                    data={
                        "workflow_id": workflow.workflow_id,
                        "status": workflow.status.value,
                        "duration": (workflow.completed_at - workflow.started_at).total_seconds()
                    }
                ))
            else:
                await self.event_bus.publish_event(
                    event_type="workflow_completed",
                    source="documentation_orchestrator",
                    data={
                        "workflow_id": workflow.workflow_id,
                        "status": workflow.status.value,
                        "duration": (workflow.completed_at - workflow.started_at).total_seconds()
                    }
                )
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow.status = DocumentationTaskStatus.FAILED
            if INTEGRATION_EVENTS_AVAILABLE and Event and EventType:
                await self.event_bus.publish(Event(
                    event_id=f"workflow_failed_{workflow.workflow_id}",
                    event_type=EventType.WORKFLOW_FAILED,
                    source="documentation_orchestrator",
                    timestamp=datetime.utcnow(),
                    data={"workflow_id": workflow.workflow_id, "error": str(e)}
                ))
            else:
                await self.event_bus.publish_event(
                    event_type="workflow_failed",
                    source="documentation_orchestrator",
                    data={"workflow_id": workflow.workflow_id, "error": str(e)}
                )
    
    async def _execute_task(self, workflow: DocumentationWorkflow, task: DocumentationTask):
        """Execute a single task"""
        try:
            task.status = DocumentationTaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            workflow.current_task = task.task_id
            
            logger.info(f"Executing task: {task.name}")
            
            # Route task to appropriate handler
            if task.task_type == DocumentationTaskType.VALIDATE_ENVIRONMENT:
                result = await self._validate_environment(task)
            elif task.task_type == DocumentationTaskType.RESOLVE_CONFLICTS:
                result = await self.conflict_resolver.resolve_conflicts(task)
            elif task.task_type == DocumentationTaskType.BUILD_VITEPRESS:
                result = await self.build_coordinator.build_vitepress(task)
            elif task.task_type == DocumentationTaskType.SYNC_FRONTEND:
                result = await self.sync_manager.sync_to_frontend(task)
            elif task.task_type == DocumentationTaskType.VERIFY_INTEGRATION:
                result = await self._verify_integration(task)
            elif task.task_type == DocumentationTaskType.START_DEV_SERVER:
                result = await self.build_coordinator.start_dev_server(task)
            elif task.task_type == DocumentationTaskType.WATCH_CHANGES:
                result = await self._start_file_watcher(task)
            elif task.task_type == DocumentationTaskType.HEALTH_MONITOR:
                result = await self.health_monitor.start_monitoring(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.result = result
            task.status = DocumentationTaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # Update workflow progress
            completed_tasks = sum(1 for t in workflow.tasks if t.status == DocumentationTaskStatus.COMPLETED)
            workflow.progress_percentage = (completed_tasks / len(workflow.tasks)) * 100
            
        except Exception as e:
            logger.error(f"Task execution failed: {task.name} - {e}")
            task.status = DocumentationTaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = DocumentationTaskStatus.RETRYING
                logger.info(f"Retrying task: {task.name} (attempt {task.retry_count})")
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self._execute_task(workflow, task)

    async def _create_workflow_from_template(self, workflow_id: str, workflow_type: DocumentationWorkflowType,
                                           parameters: Dict[str, Any]) -> DocumentationWorkflow:
        """Create a workflow from predefined templates"""
        try:
            # Get template based on workflow type
            if workflow_type == DocumentationWorkflowType.BUILD_AND_SYNC:
                template = self._create_build_sync_template()
            elif workflow_type == DocumentationWorkflowType.DEVELOPMENT_MODE:
                template = self._create_dev_mode_template()
            elif workflow_type == DocumentationWorkflowType.PRODUCTION_DEPLOY:
                template = self._create_production_deploy_template()
            elif workflow_type == DocumentationWorkflowType.HEALTH_CHECK:
                template = self._create_health_check_template()
            elif workflow_type == DocumentationWorkflowType.CONFLICT_RESOLUTION:
                template = self._create_conflict_resolution_template()
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")

            # Create tasks from template
            tasks = []
            for i, task_def in enumerate(template["tasks"]):
                task = DocumentationTask(
                    task_id=f"{workflow_id}_task_{i}",
                    task_type=task_def["task_type"],
                    name=task_def["name"],
                    description=task_def["description"],
                    parameters=task_def.get("parameters", {})
                )
                tasks.append(task)

            # Create workflow
            workflow = DocumentationWorkflow(
                workflow_id=workflow_id,
                workflow_type=workflow_type,
                name=template["name"],
                description=template["description"],
                tasks=tasks,
                config=self.config
            )

            # Apply any custom parameters
            workflow.metadata.update(parameters)

            return workflow

        except Exception as e:
            logger.error(f"Failed to create workflow from template: {e}")
            raise

    def _initialize_workflow_templates(self):
        """Initialize predefined workflow templates"""
        if self.workflow_manager and INTEGRATION_WORKFLOWS_AVAILABLE:
            # Build and Sync Workflow
            build_sync_template = self._create_build_sync_template()
            self.workflow_manager.register_template("documentation_build_sync", build_sync_template)

            # Development Mode Workflow
            dev_mode_template = self._create_dev_mode_template()
            self.workflow_manager.register_template("documentation_dev_mode", dev_mode_template)

            # Production Deploy Workflow
            prod_deploy_template = self._create_production_deploy_template()
            self.workflow_manager.register_template("documentation_production", prod_deploy_template)
        else:
            logger.warning("WorkflowManager not available - workflow templates not registered")

    def _create_build_sync_template(self) -> Dict[str, Any]:
        """Create build and sync workflow template"""
        return {
            "name": "Documentation Build and Sync",
            "description": "Build VitePress documentation and sync to frontend",
            "tasks": [
                {
                    "task_type": DocumentationTaskType.VALIDATE_ENVIRONMENT,
                    "name": "Validate Environment",
                    "description": "Check dependencies and configuration"
                },
                {
                    "task_type": DocumentationTaskType.RESOLVE_CONFLICTS,
                    "name": "Resolve Conflicts",
                    "description": "Resolve Tailwind CSS and PostCSS conflicts"
                },
                {
                    "task_type": DocumentationTaskType.BUILD_VITEPRESS,
                    "name": "Build VitePress",
                    "description": "Build static documentation files"
                },
                {
                    "task_type": DocumentationTaskType.SYNC_FRONTEND,
                    "name": "Sync to Frontend",
                    "description": "Copy built files to frontend serving location"
                },
                {
                    "task_type": DocumentationTaskType.VERIFY_INTEGRATION,
                    "name": "Verify Integration",
                    "description": "Verify documentation is accessible from UI"
                }
            ]
        }

    def _create_dev_mode_template(self) -> Dict[str, Any]:
        """Create development mode workflow template"""
        return {
            "name": "Documentation Development Mode",
            "description": "Start development server with hot reload",
            "tasks": [
                {
                    "task_type": DocumentationTaskType.VALIDATE_ENVIRONMENT,
                    "name": "Validate Environment",
                    "description": "Check dependencies and configuration"
                },
                {
                    "task_type": DocumentationTaskType.RESOLVE_CONFLICTS,
                    "name": "Resolve Conflicts",
                    "description": "Resolve Tailwind CSS and PostCSS conflicts"
                },
                {
                    "task_type": DocumentationTaskType.START_DEV_SERVER,
                    "name": "Start Dev Server",
                    "description": "Start VitePress development server"
                },
                {
                    "task_type": DocumentationTaskType.WATCH_CHANGES,
                    "name": "Watch Changes",
                    "description": "Monitor file changes for hot reload"
                },
                {
                    "task_type": DocumentationTaskType.HEALTH_MONITOR,
                    "name": "Health Monitor",
                    "description": "Monitor server health and performance"
                }
            ]
        }

    def _create_production_deploy_template(self) -> Dict[str, Any]:
        """Create production deployment workflow template"""
        return {
            "name": "Documentation Production Deployment",
            "description": "Build and deploy documentation for production",
            "tasks": [
                {
                    "task_type": DocumentationTaskType.VALIDATE_ENVIRONMENT,
                    "name": "Validate Environment",
                    "description": "Check production dependencies"
                },
                {
                    "task_type": DocumentationTaskType.RESOLVE_CONFLICTS,
                    "name": "Resolve Conflicts",
                    "description": "Resolve all configuration conflicts"
                },
                {
                    "task_type": DocumentationTaskType.BUILD_VITEPRESS,
                    "name": "Production Build",
                    "description": "Build optimized documentation for production",
                    "parameters": {"production": True, "minify": True}
                },
                {
                    "task_type": DocumentationTaskType.SYNC_FRONTEND,
                    "name": "Deploy to Frontend",
                    "description": "Deploy built files to production frontend"
                },
                {
                    "task_type": DocumentationTaskType.VERIFY_INTEGRATION,
                    "name": "Production Verification",
                    "description": "Verify production deployment"
                }
            ]
        }

    def _create_health_check_template(self) -> Dict[str, Any]:
        """Create health check workflow template"""
        return {
            "name": "Documentation Health Check",
            "description": "Check documentation system health and status",
            "tasks": [
                {
                    "task_type": DocumentationTaskType.VALIDATE_ENVIRONMENT,
                    "name": "Environment Check",
                    "description": "Validate documentation environment and dependencies"
                },
                {
                    "task_type": DocumentationTaskType.VERIFY_INTEGRATION,
                    "name": "Integration Check",
                    "description": "Verify documentation integration with frontend"
                }
            ]
        }

    def _create_conflict_resolution_template(self) -> Dict[str, Any]:
        """Create conflict resolution workflow template"""
        return {
            "name": "Documentation Conflict Resolution",
            "description": "Detect and resolve documentation configuration conflicts",
            "tasks": [
                {
                    "task_type": DocumentationTaskType.RESOLVE_CONFLICTS,
                    "name": "Detect and Resolve Conflicts",
                    "description": "Scan for and resolve Tailwind CSS/PostCSS conflicts"
                },
                {
                    "task_type": DocumentationTaskType.VALIDATE_ENVIRONMENT,
                    "name": "Validate Resolution",
                    "description": "Verify conflicts have been resolved"
                }
            ]
        }

    async def _validate_environment(self, task: DocumentationTask) -> Dict[str, Any]:
        """Validate documentation environment"""
        validation_result = {
            "node_version": None,
            "npm_version": None,
            "vitepress_available": False,
            "paths_exist": {},
            "dependencies_installed": False,
            "conflicts_detected": []
        }

        try:
            # Check Node.js version
            node_result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            validation_result["node_version"] = node_result.stdout.strip()

            # Check npm version
            npm_result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
            validation_result["npm_version"] = npm_result.stdout.strip()

            # Check if VitePress is available
            vitepress_check = subprocess.run(
                ["npx", "vitepress", "--version"],
                capture_output=True, text=True, cwd=self.config.docs_source_path
            )
            validation_result["vitepress_available"] = vitepress_check.returncode == 0

            # Check required paths
            paths_to_check = {
                "docs_source": self.config.docs_source_path,
                "frontend_public": self.config.frontend_public_path,
                "package_json": self.config.docs_source_path / "package.json",
                "vitepress_config": self.config.docs_source_path / ".vitepress" / "config.ts"
            }

            for name, path in paths_to_check.items():
                validation_result["paths_exist"][name] = path.exists()

            # Check for dependency conflicts
            conflicts = await self.conflict_resolver.detect_conflicts()
            validation_result["conflicts_detected"] = conflicts

            return validation_result

        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            raise

    async def _verify_integration(self, task: DocumentationTask) -> Dict[str, Any]:
        """Verify documentation integration with frontend"""
        verification_result = {
            "frontend_files_exist": False,
            "manifest_valid": False,
            "routes_accessible": False,
            "ui_integration": False,
            "mermaid_support": False,
            "errors": []
        }

        try:
            # Check if frontend documentation files exist
            docs_index = self.config.frontend_docs_path / "index.html"
            verification_result["frontend_files_exist"] = docs_index.exists()

            # Validate manifest file
            manifest_path = self.config.frontend_docs_path / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    verification_result["manifest_valid"] = "routes" in manifest

            # Test route accessibility (if development server is running)
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:3000/docs/") as response:
                        verification_result["routes_accessible"] = response.status == 200
            except:
                verification_result["routes_accessible"] = False

            # Check UI integration components
            ui_components = [
                self.config.frontend_public_path.parent / "src" / "pages" / "DocumentationPage.tsx",
                self.config.frontend_public_path.parent / "src" / "components" / "layout" / "Sidebar.tsx"
            ]
            verification_result["ui_integration"] = all(comp.exists() for comp in ui_components)

            # Check for Mermaid diagram support
            assets_path = self.config.frontend_docs_path / "assets"
            if assets_path.exists():
                mermaid_files = list(assets_path.glob("*mermaid*"))
                verification_result["mermaid_support"] = len(mermaid_files) > 0

            return verification_result

        except Exception as e:
            verification_result["errors"].append(str(e))
            logger.error(f"Integration verification failed: {e}")
            return verification_result

    async def _start_file_watcher(self, task: DocumentationTask) -> Dict[str, Any]:
        """Start file watcher for development mode"""
        watcher_result = {
            "watcher_started": False,
            "watched_paths": [],
            "patterns": self.config.watch_file_patterns
        }

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class DocumentationEventHandler(FileSystemEventHandler):
                def __init__(self, orchestrator):
                    self.orchestrator = orchestrator

                def on_modified(self, event):
                    if not event.is_directory:
                        asyncio.create_task(self.orchestrator._handle_file_change(event.src_path))

            observer = Observer()
            event_handler = DocumentationEventHandler(self)

            # Watch documentation source directory
            observer.schedule(event_handler, str(self.config.docs_source_path), recursive=True)
            observer.start()

            watcher_result["watcher_started"] = True
            watcher_result["watched_paths"] = [str(self.config.docs_source_path)]

            # Store observer reference for cleanup
            task.parameters["observer"] = observer

            return watcher_result

        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            raise

    async def _handle_file_change(self, file_path: str):
        """Handle file change events"""
        try:
            logger.info(f"File changed: {file_path}")

            # Check if it's a documentation file
            path = Path(file_path)
            if any(path.match(pattern) for pattern in self.config.watch_file_patterns):
                # Trigger incremental sync
                await self.execute_workflow(
                    DocumentationWorkflowType.BUILD_AND_SYNC,
                    {"incremental": True, "changed_file": file_path}
                )

        except Exception as e:
            logger.error(f"Error handling file change: {e}")

    async def _register_event_handlers(self):
        """Register event handlers with the event bus"""
        # Register for system events
        await self.event_bus.subscribe(EventType.SYSTEM_STATUS, self._handle_system_event)
        await self.event_bus.subscribe(EventType.WORKFLOW_COMPLETED, self._handle_workflow_event)
        await self.event_bus.subscribe(EventType.WORKFLOW_FAILED, self._handle_workflow_event)

    async def _handle_system_event(self, event: Event):
        """Handle system events"""
        if event.data.get("component") == "frontend" and event.data.get("status") == "started":
            # Frontend started, verify documentation integration
            await self.execute_workflow(DocumentationWorkflowType.HEALTH_CHECK)

    async def _handle_workflow_event(self, event: Event):
        """Handle workflow events"""
        workflow_id = event.data.get("workflow_id")
        if workflow_id and workflow_id.startswith("docs_"):
            # Update metrics
            await self.metrics_collector.record_workflow_completion(
                workflow_id, event.data.get("status"), event.data.get("duration", 0)
            )

    async def _register_with_orchestration_manager(self):
        """Register documentation orchestrator with the main orchestration manager"""
        # For now, just log the registration - the OrchestrationManager already
        # has a reference to this DocumentationOrchestrator instance
        logger.info("Documentation Orchestrator registered with OrchestrationManager")

        # In the future, we could register with the integration system if needed:
        # component_info = {
        #     "component_id": "documentation_orchestrator",
        #     "component_type": "orchestrator",
        #     "name": "Documentation Orchestrator",
        #     "description": "Intelligent documentation workflow management",
        #     "capabilities": [
        #         "documentation_build", "frontend_sync", "conflict_resolution",
        #         "development_mode", "health_monitoring"
        #     ],
        #     "status": "active"
        # }

    async def cancel_workflow(self, workflow_id: str):
        """Cancel an active workflow"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = DocumentationTaskStatus.FAILED
            workflow.completed_at = datetime.utcnow()

            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]

            logger.info(f"Cancelled workflow: {workflow_id}")

    def get_workflow_status(self, workflow_id: str) -> Optional[DocumentationWorkflow]:
        """Get status of a workflow"""
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]

        # Check history
        for workflow in self.workflow_history:
            if workflow.workflow_id == workflow_id:
                return workflow

        return None

    def list_active_workflows(self) -> List[DocumentationWorkflow]:
        """List all active workflows"""
        return list(self.active_workflows.values())

    async def get_system_status(self) -> Dict[str, Any]:
        """Get documentation system status"""
        # Get intelligent build status
        build_status = await self.intelligent_builder.get_build_status()

        return {
            "orchestrator_running": self.is_running,
            "active_workflows": len(self.active_workflows),
            "total_workflows_executed": len(self.workflow_history),
            "intelligent_build": build_status,
            "components": {
                "build_coordinator": await self.build_coordinator.get_status(),
                "sync_manager": await self.sync_manager.get_status(),
                "conflict_resolver": await self.conflict_resolver.get_status(),
                "health_monitor": await self.health_monitor.get_status()
            }
        }

    async def execute_intelligent_build(self, force: bool = False, production: bool = True) -> Dict[str, Any]:
        """Execute intelligent documentation build with trigger detection and Mermaid caching"""
        logger.info(f"Executing intelligent documentation build (force={force}, production={production})")

        try:
            # Execute intelligent build
            build_result = await self.intelligent_builder.execute_intelligent_build(
                force=force,
                production=production
            )

            # Emit build events
            if build_result['success']:
                await self.event_bus.publish_event(
                    event_type="intelligent_build_completed",
                    source="documentation_orchestrator",
                    data={
                        "build_id": build_result['build_id'],
                        "duration": build_result['total_duration_seconds'],
                        "triggers": build_result.get('triggers', {}),
                        "mermaid_diagrams": len(build_result.get('mermaid_results', {}))
                    }
                )
            else:
                await self.event_bus.publish_event(
                    event_type="intelligent_build_failed",
                    source="documentation_orchestrator",
                    data={
                        "build_id": build_result['build_id'],
                        "error": build_result.get('error', 'Unknown error')
                    }
                )

            return build_result

        except Exception as e:
            logger.error(f"Intelligent build execution failed: {e}")
            await self.event_bus.publish_event(
                event_type="intelligent_build_failed",
                source="documentation_orchestrator",
                data={"error": str(e)}
            )
            raise

    async def check_build_triggers(self) -> Dict[str, Any]:
        """Check if documentation build should be triggered"""
        return await self.intelligent_builder.check_build_triggers()

    async def regenerate_mermaid_diagrams(self, force: bool = False) -> Dict[str, bool]:
        """Regenerate Mermaid diagrams with caching"""
        logger.info(f"Regenerating Mermaid diagrams (force={force})")
        return await self.intelligent_builder.prepare_mermaid_diagrams(force=force)

    async def get_mermaid_cache_status(self) -> Dict[str, Any]:
        """Get Mermaid diagram cache status"""
        return self.intelligent_builder.mermaid_manager.get_cache_status()

    async def force_complete_rebuild(self) -> Dict[str, Any]:
        """Force a complete rebuild of all documentation and diagrams"""
        logger.info("Forcing complete documentation rebuild")
        return await self.intelligent_builder.force_rebuild()
