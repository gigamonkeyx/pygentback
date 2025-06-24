"""
Integration Workflow Management

Comprehensive workflow management system for orchestrating complex AI workflows,
managing workflow state, and providing workflow templates and execution.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

# Local models to avoid circular imports
@dataclass
class WorkflowDefinition:
    """Local workflow definition"""
    name: str
    description: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ExecutionContext:
    """Local execution context"""
    workflow_id: str = ""
    execution_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)

@dataclass
class IntegrationResult:
    """Local integration result"""
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    context: Optional[ExecutionContext] = None
    error: Optional[str] = None

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowState:
    """Current state of a workflow execution"""
    workflow_id: str
    status: WorkflowStatus
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0


@dataclass
class WorkflowTemplate:
    """Template for creating workflows"""
    template_id: str
    name: str
    description: str
    category: str
    steps: List[Dict[str, Any]]
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    complexity_level: str = "medium"


class WorkflowManager:
    """
    Workflow Management System.
    
    Manages workflow definitions, templates, and execution state.
    Provides workflow lifecycle management and state persistence.
    """
    
    def __init__(self):
        # Workflow storage
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_states: Dict[str, WorkflowState] = {}
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        
        # Execution tracking
        self.active_workflows: Dict[str, asyncio.Task] = {}
        self.workflow_history: List[WorkflowState] = []
        
        # Event handlers
        self.status_change_handlers: List[Callable] = []
        self.completion_handlers: List[Callable] = []
        
        # Initialize default templates
        self._initialize_default_templates()
    
    def register_workflow(self, workflow: WorkflowDefinition) -> str:
        """Register a new workflow definition"""
        workflow_id = f"workflow_{len(self.workflows) + 1}"
        self.workflows[workflow_id] = workflow
        
        # Initialize state
        self.workflow_states[workflow_id] = WorkflowState(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING
        )
        
        logger.info(f"Registered workflow: {workflow.name} ({workflow_id})")
        return workflow_id
    
    def create_from_template(self, template_id: str, parameters: Dict[str, Any]) -> str:
        """Create a workflow from a template"""
        if template_id not in self.workflow_templates:
            raise ValueError(f"Template not found: {template_id}")
        
        template = self.workflow_templates[template_id]
        
        # Merge parameters with defaults
        merged_params = template.default_parameters.copy()
        merged_params.update(parameters)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            name=f"{template.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=template.description,
            steps=template.steps.copy()
        )
        
        # Substitute parameters in steps
        workflow.steps = self._substitute_parameters(workflow.steps, merged_params)
        
        return self.register_workflow(workflow)
    
    def get_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get current state of a workflow"""
        return self.workflow_states.get(workflow_id)
    
    def update_workflow_state(self, workflow_id: str, **updates):
        """Update workflow state"""
        if workflow_id not in self.workflow_states:
            return
        
        state = self.workflow_states[workflow_id]
        old_status = state.status
        
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        # Trigger status change handlers
        if 'status' in updates and updates['status'] != old_status:
            self._trigger_status_change_handlers(workflow_id, old_status, updates['status'])
        
        logger.debug(f"Updated workflow state: {workflow_id}")
    
    def list_workflows(self, status_filter: Optional[WorkflowStatus] = None) -> List[str]:
        """List workflows, optionally filtered by status"""
        if status_filter:
            return [
                wid for wid, state in self.workflow_states.items()
                if state.status == status_filter
            ]
        return list(self.workflows.keys())
    
    def get_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Get summary information about a workflow"""
        if workflow_id not in self.workflows:
            return {}
        
        workflow = self.workflows[workflow_id]
        state = self.workflow_states[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "status": state.status.value,
            "progress": state.progress_percentage,
            "current_step": state.current_step,
            "completed_steps": len(state.completed_steps),
            "total_steps": len(workflow.steps),
            "start_time": state.start_time.isoformat() if state.start_time else None,
            "duration": (
                (state.end_time or datetime.utcnow()) - state.start_time
            ).total_seconds() if state.start_time else 0
        }
    
    def add_template(self, template: WorkflowTemplate):
        """Add a workflow template"""
        self.workflow_templates[template.template_id] = template
        logger.info(f"Added workflow template: {template.name}")
    
    def list_templates(self, category: Optional[str] = None) -> List[WorkflowTemplate]:
        """List available workflow templates"""
        templates = list(self.workflow_templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates
    
    def _initialize_default_templates(self):
        """Initialize default workflow templates"""
        # Recipe optimization template
        recipe_optimization = WorkflowTemplate(
            template_id="recipe_optimization",
            name="Recipe Optimization",
            description="Optimize recipe using genetic algorithm and predictive models",
            category="optimization",
            steps=[
                {
                    'step_id': 'parse_recipe',
                    'component_type': 'nlp_system',
                    'action': 'parse_recipe',
                    'parameters': {'recipe_text': '${recipe_text}'},
                    'required': True
                },
                {
                    'step_id': 'predict_performance',
                    'component_type': 'predictive_optimization',
                    'action': 'predict_performance',
                    'parameters': {'recipe_data': '${parsed_recipe}'},
                    'dependencies': ['parse_recipe']
                },
                {
                    'step_id': 'genetic_optimization',
                    'component_type': 'genetic_algorithm',
                    'action': 'optimize_recipe',
                    'parameters': {
                        'recipe_data': '${parsed_recipe}',
                        'performance_prediction': '${predicted_performance}'
                    },
                    'dependencies': ['parse_recipe', 'predict_performance']
                }
            ],
            default_parameters={
                'optimization_iterations': 100,
                'population_size': 50,
                'mutation_rate': 0.1
            },
            required_capabilities=['nlp', 'genetic_algorithm', 'predictive'],
            estimated_duration=300.0,
            complexity_level="high"
        )
        
        # Multi-agent research template
        research_template = WorkflowTemplate(
            template_id="multi_agent_research",
            name="Multi-Agent Research",
            description="Conduct research using multiple AI agents",
            category="research",
            steps=[
                {
                    'step_id': 'initialize_agents',
                    'component_type': 'multi_agent',
                    'action': 'initialize_research_agents',
                    'parameters': {'research_topic': '${topic}'},
                    'required': True
                },
                {
                    'step_id': 'gather_sources',
                    'component_type': 'multi_agent',
                    'action': 'gather_sources',
                    'parameters': {'agents': '${initialized_agents}'},
                    'dependencies': ['initialize_agents']
                },
                {
                    'step_id': 'analyze_sources',
                    'component_type': 'nlp_system',
                    'action': 'analyze_sources',
                    'parameters': {'sources': '${gathered_sources}'},
                    'dependencies': ['gather_sources']
                },
                {
                    'step_id': 'synthesize_findings',
                    'component_type': 'multi_agent',
                    'action': 'synthesize_research',
                    'parameters': {
                        'analysis': '${source_analysis}',
                        'agents': '${initialized_agents}'
                    },
                    'dependencies': ['analyze_sources']
                }
            ],
            default_parameters={
                'agent_count': 3,
                'research_depth': 'comprehensive',
                'source_types': ['academic', 'primary', 'secondary']
            },
            required_capabilities=['multi_agent', 'nlp'],
            estimated_duration=600.0,
            complexity_level="high"
        )
        
        self.add_template(recipe_optimization)
        self.add_template(research_template)
    
    def _substitute_parameters(self, steps: List[Dict[str, Any]], 
                             parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Substitute parameters in workflow steps"""
        substituted_steps = []
        
        for step in steps:
            substituted_step = step.copy()
            
            # Substitute in parameters
            if 'parameters' in substituted_step:
                substituted_params = {}
                for key, value in substituted_step['parameters'].items():
                    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                        param_name = value[2:-1]
                        substituted_params[key] = parameters.get(param_name, value)
                    else:
                        substituted_params[key] = value
                substituted_step['parameters'] = substituted_params
            
            substituted_steps.append(substituted_step)
        
        return substituted_steps
    
    def _trigger_status_change_handlers(self, workflow_id: str, 
                                      old_status: WorkflowStatus, 
                                      new_status: WorkflowStatus):
        """Trigger status change event handlers"""
        for handler in self.status_change_handlers:
            try:
                handler(workflow_id, old_status, new_status)
            except Exception as e:
                logger.error(f"Status change handler error: {e}")
    
    def add_status_change_handler(self, handler: Callable):
        """Add a status change event handler"""
        self.status_change_handlers.append(handler)
    
    def add_completion_handler(self, handler: Callable):
        """Add a workflow completion event handler"""
        self.completion_handlers.append(handler)


class WorkflowExecutor:
    """
    Workflow Execution Engine.
    
    Executes workflows by coordinating with the integration engine
    and managing workflow state transitions.
    """
    
    def __init__(self, workflow_manager: WorkflowManager, integration_engine):
        self.workflow_manager = workflow_manager
        self.integration_engine = integration_engine
        
        # Execution configuration
        self.max_concurrent_workflows = 5
        self.step_timeout = 300.0  # 5 minutes
        self.retry_attempts = 3
        
        # Execution state
        self.running_workflows: Dict[str, asyncio.Task] = {}
    
    async def execute_workflow(self, workflow_id: str, 
                             context: Optional[ExecutionContext] = None) -> IntegrationResult:
        """Execute a workflow"""
        if workflow_id not in self.workflow_manager.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        if len(self.running_workflows) >= self.max_concurrent_workflows:
            raise RuntimeError("Maximum concurrent workflows reached")
        
        workflow = self.workflow_manager.workflows[workflow_id]
        
        # Create execution context if not provided
        if context is None:
            context = ExecutionContext(
                workflow_id=workflow_id,
                execution_id=f"exec_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                parameters={}
            )
        
        # Start execution
        execution_task = asyncio.create_task(
            self._execute_workflow_steps(workflow_id, workflow, context)
        )
        self.running_workflows[workflow_id] = execution_task
        
        try:
            result = await execution_task
            return result
        finally:
            if workflow_id in self.running_workflows:
                del self.running_workflows[workflow_id]
    
    async def _execute_workflow_steps(self, workflow_id: str, 
                                    workflow: WorkflowDefinition,
                                    context: ExecutionContext) -> IntegrationResult:
        """Execute workflow steps"""
        # Update state to running
        self.workflow_manager.update_workflow_state(
            workflow_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        step_results = {}
        completed_steps = []
        
        try:
            for i, step in enumerate(workflow.steps):
                step_id = step.get('step_id', f'step_{i}')
                
                # Update current step
                self.workflow_manager.update_workflow_state(
                    workflow_id,
                    current_step=step_id,
                    progress_percentage=(i / len(workflow.steps)) * 100
                )
                
                # Check dependencies
                dependencies = step.get('dependencies', [])
                if not all(dep in completed_steps for dep in dependencies):
                    missing_deps = [dep for dep in dependencies if dep not in completed_steps]
                    raise RuntimeError(f"Missing dependencies for step {step_id}: {missing_deps}")
                
                # Execute step
                step_result = await self._execute_step(step, step_results, context)
                step_results[step_id] = step_result
                completed_steps.append(step_id)
                
                # Update state
                self.workflow_manager.update_workflow_state(
                    workflow_id,
                    completed_steps=completed_steps.copy(),
                    step_results=step_results.copy()
                )
            
            # Workflow completed successfully
            self.workflow_manager.update_workflow_state(
                workflow_id,
                status=WorkflowStatus.COMPLETED,
                end_time=datetime.utcnow(),
                progress_percentage=100.0
            )
            
            return IntegrationResult(
                success=True,
                results=step_results,
                execution_time=(datetime.utcnow() - context.start_time).total_seconds(),
                context=context
            )
            
        except Exception as e:
            # Workflow failed
            self.workflow_manager.update_workflow_state(
                workflow_id,
                status=WorkflowStatus.FAILED,
                end_time=datetime.utcnow(),
                error_message=str(e)
            )
            
            return IntegrationResult(
                success=False,
                error=str(e),
                results=step_results,
                execution_time=(datetime.utcnow() - context.start_time).total_seconds(),
                context=context
            )
    
    async def _execute_step(self, step: Dict[str, Any], 
                          previous_results: Dict[str, Any],
                          context: ExecutionContext) -> Any:
        """Execute a single workflow step"""
        component_type = step.get('component_type')
        action = step.get('action')
        parameters = step.get('parameters', {})
        
        # Substitute previous results in parameters
        substituted_params = self._substitute_step_results(parameters, previous_results)
        
        # Execute through integration engine
        result = await self.integration_engine.execute_component_action(
            component_type, action, substituted_params, context
        )
        
        return result
    
    def _substitute_step_results(self, parameters: Dict[str, Any], 
                               results: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute previous step results in parameters"""
        substituted = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                result_key = value[2:-1]
                substituted[key] = results.get(result_key, value)
            else:
                substituted[key] = value
        
        return substituted
    
    async def cancel_workflow(self, workflow_id: str):
        """Cancel a running workflow"""
        if workflow_id in self.running_workflows:
            task = self.running_workflows[workflow_id]
            task.cancel()
            
            self.workflow_manager.update_workflow_state(
                workflow_id,
                status=WorkflowStatus.CANCELLED,
                end_time=datetime.utcnow()
            )
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        return {
            "running_workflows": len(self.running_workflows),
            "max_concurrent": self.max_concurrent_workflows,
            "active_workflow_ids": list(self.running_workflows.keys())
        }
