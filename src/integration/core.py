"""
Integration Core Components

Core engines for system integration and workflow orchestration.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .models import IntegrationConfig, WorkflowDefinition, ExecutionContext, IntegrationResult

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Integration status states"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class ComponentType(Enum):
    """Types of integrated components"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    NEURAL_SEARCH = "neural_search"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    MCP_INTELLIGENCE = "mcp_intelligence"
    NLP_SYSTEM = "nlp_system"
    MULTI_AGENT = "multi_agent"
    PREDICTIVE_OPTIMIZATION = "predictive_optimization"


@dataclass
class ComponentInfo:
    """Information about an integrated component"""
    component_id: str
    component_type: ComponentType
    name: str
    version: str
    status: str = "inactive"
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    health_score: float = 1.0
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntegrationEngine:
    """
    Main engine for integrating and coordinating all AI system components.
    """
    
    def __init__(self):
        self.status = IntegrationStatus.INITIALIZING
        self.components: Dict[str, ComponentInfo] = {}
        self.adapters: Dict[str, Any] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Integration configuration
        self.config = IntegrationConfig()
        
        # System state
        self.start_time: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        
        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.event_processor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'components_registered': 0,
            'workflows_executed': 0,
            'integration_errors': 0,
            'health_checks_performed': 0,
            'events_processed': 0
        }
    
    async def start(self):
        """Start the integration engine"""
        if self.status != IntegrationStatus.INITIALIZING:
            logger.warning("Integration engine already started")
            return
        
        try:
            self.start_time = datetime.utcnow()
            self.status = IntegrationStatus.RUNNING
            
            # Start background tasks
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            self.event_processor_task = asyncio.create_task(self._event_processor_loop())
            
            # Initialize all registered components
            await self._initialize_components()
            
            self.status = IntegrationStatus.READY
            logger.info("Integration engine started successfully")
            
        except Exception as e:
            self.status = IntegrationStatus.ERROR
            logger.error(f"Integration engine startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the integration engine"""
        if self.status == IntegrationStatus.SHUTDOWN:
            return
        
        logger.info("Shutting down integration engine")
        self.status = IntegrationStatus.SHUTDOWN
        
        # Cancel background tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
        if self.event_processor_task:
            self.event_processor_task.cancel()
        
        # Shutdown all components
        await self._shutdown_components()
        
        logger.info("Integration engine shutdown complete")

    async def stop(self):
        """Alias for shutdown() method for compatibility"""
        await self.shutdown()

    def register_component(self, component_info: ComponentInfo, adapter: Any):
        """Register a component with the integration engine"""
        self.components[component_info.component_id] = component_info
        self.adapters[component_info.component_id] = adapter
        self.stats['components_registered'] += 1
        
        logger.info(f"Registered component: {component_info.name} ({component_info.component_type.value})")
    
    def unregister_component(self, component_id: str):
        """Unregister a component"""
        if component_id in self.components:
            component_info = self.components[component_id]
            del self.components[component_id]
            del self.adapters[component_id]
            
            logger.info(f"Unregistered component: {component_info.name}")
    
    async def execute_workflow(self, workflow_definition: WorkflowDefinition, 
                             context: Optional[ExecutionContext] = None) -> IntegrationResult:
        """Execute a cross-component workflow"""
        if self.status != IntegrationStatus.READY:
            raise RuntimeError("Integration engine not ready")
        
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        if context is None:
            context = ExecutionContext(execution_id=execution_id)
        
        try:
            logger.info(f"Starting workflow execution: {workflow_definition.name}")
            
            # Validate workflow
            validation_result = await self._validate_workflow(workflow_definition)
            if not validation_result['valid']:
                raise ValueError(f"Workflow validation failed: {validation_result['errors']}")
            
            # Execute workflow steps
            results = {}
            for step in workflow_definition.steps:
                step_result = await self._execute_workflow_step(step, context, results)
                results[step['step_id']] = step_result
                
                # Check if step failed and workflow should stop
                if not step_result.get('success', False) and step.get('required', True):
                    raise RuntimeError(f"Required workflow step failed: {step['step_id']}")
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.stats['workflows_executed'] += 1
            
            return IntegrationResult(
                execution_id=execution_id,
                workflow_name=workflow_definition.name,
                success=True,
                execution_time_seconds=execution_time,
                results=results,
                metadata={'workflow_definition': workflow_definition.to_dict()}
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.stats['integration_errors'] += 1
            
            logger.error(f"Workflow execution failed: {e}")
            
            return IntegrationResult(
                execution_id=execution_id,
                workflow_name=workflow_definition.name,
                success=False,
                execution_time_seconds=execution_time,
                error_message=str(e),
                metadata={'workflow_definition': workflow_definition.to_dict()}
            )
    
    async def _initialize_components(self):
        """Initialize all registered components"""
        for component_id, adapter in self.adapters.items():
            try:
                if hasattr(adapter, 'initialize'):
                    await adapter.initialize()
                
                component_info = self.components[component_id]
                component_info.status = "active"
                
                logger.debug(f"Initialized component: {component_info.name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize component {component_id}: {e}")
                self.components[component_id].status = "error"
    
    async def _shutdown_components(self):
        """Shutdown all components"""
        for component_id, adapter in self.adapters.items():
            try:
                if hasattr(adapter, 'shutdown'):
                    await adapter.shutdown()
                
                self.components[component_id].status = "inactive"
                
            except Exception as e:
                logger.error(f"Failed to shutdown component {component_id}: {e}")
    
    async def _validate_workflow(self, workflow_definition: WorkflowDefinition) -> Dict[str, Any]:
        """Validate workflow definition"""
        errors = []
        warnings = []
        
        # Check if required components are available
        required_components = set()
        for step in workflow_definition.steps:
            component_type = step.get('component_type')
            if component_type:
                required_components.add(component_type)
        
        available_components = {comp.component_type.value for comp in self.components.values() 
                              if comp.status == "active"}
        
        missing_components = required_components - available_components
        if missing_components:
            errors.append(f"Missing required components: {missing_components}")
        
        # Check step dependencies
        step_ids = {step['step_id'] for step in workflow_definition.steps}
        for step in workflow_definition.steps:
            dependencies = step.get('dependencies', [])
            missing_deps = set(dependencies) - step_ids
            if missing_deps:
                errors.append(f"Step {step['step_id']} has missing dependencies: {missing_deps}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def _execute_workflow_step(self, step: Dict[str, Any], context: ExecutionContext, 
                                   previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step_id = step['step_id']
        component_type = step.get('component_type')
        action = step.get('action')
        parameters = step.get('parameters', {})
        
        logger.debug(f"Executing workflow step: {step_id}")
        
        try:
            # Find appropriate adapter
            adapter = None
            for comp_id, comp_info in self.components.items():
                if comp_info.component_type.value == component_type and comp_info.status == "active":
                    adapter = self.adapters[comp_id]
                    break
            
            if not adapter:
                raise RuntimeError(f"No active adapter found for component type: {component_type}")
            
            # Execute step action
            if hasattr(adapter, action):
                action_method = getattr(adapter, action)
                
                # Prepare parameters with context and previous results
                execution_params = {
                    **parameters,
                    'context': context,
                    'previous_results': previous_results
                }
                
                if asyncio.iscoroutinefunction(action_method):
                    result = await action_method(**execution_params)
                else:
                    result = action_method(**execution_params)
                
                return {
                    'success': True,
                    'result': result,
                    'step_id': step_id,
                    'component_type': component_type,
                    'action': action
                }
            else:
                raise AttributeError(f"Adapter does not support action: {action}")
                
        except Exception as e:
            logger.error(f"Workflow step {step_id} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_id': step_id,
                'component_type': component_type,
                'action': action
            }
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self.status in [IntegrationStatus.RUNNING, IntegrationStatus.READY]:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """Perform health checks on all components"""
        self.last_health_check = datetime.utcnow()
        self.stats['health_checks_performed'] += 1
        
        for component_id, component_info in self.components.items():
            try:
                adapter = self.adapters[component_id]
                
                # Check if adapter has health check method
                if hasattr(adapter, 'health_check'):
                    health_result = await adapter.health_check()
                    component_info.health_score = health_result.get('score', 1.0)
                    
                    if health_result.get('status') == 'unhealthy':
                        component_info.status = "error"
                        logger.warning(f"Component {component_info.name} is unhealthy")
                    elif component_info.status == "error":
                        component_info.status = "active"  # Recovered
                
                component_info.last_health_check = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Health check failed for component {component_id}: {e}")
                component_info.health_score = 0.0
                component_info.status = "error"
    
    async def _event_processor_loop(self):
        """Background event processing loop"""
        while self.status in [IntegrationStatus.RUNNING, IntegrationStatus.READY]:
            try:
                # Process events from event bus (simplified)
                await asyncio.sleep(0.1)
                self.stats['events_processed'] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processor error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        component_statuses = {}
        for comp_id, comp_info in self.components.items():
            component_statuses[comp_id] = {
                'name': comp_info.name,
                'type': comp_info.component_type.value,
                'status': comp_info.status,
                'health_score': comp_info.health_score,
                'capabilities': comp_info.capabilities,
                'last_health_check': comp_info.last_health_check.isoformat()
            }
        
        return {
            'integration_status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'total_components': len(self.components),
            'active_components': len([c for c in self.components.values() if c.status == "active"]),
            'components': component_statuses,
            'statistics': self.stats.copy()
        }


class OrchestrationEngine:
    """
    Engine for orchestrating complex multi-component workflows.
    """
    
    def __init__(self, integration_engine: IntegrationEngine):
        self.integration_engine = integration_engine
        self.workflow_templates: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # Orchestration configuration
        self.config = {
            'max_concurrent_workflows': 10,
            'workflow_timeout_seconds': 3600,
            'retry_failed_steps': True,
            'max_step_retries': 3
        }
        
        # Statistics
        self.stats = {
            'workflows_orchestrated': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'avg_workflow_duration_seconds': 0.0
        }
    
    def register_workflow_template(self, template: WorkflowDefinition):
        """Register a workflow template"""
        self.workflow_templates[template.name] = template
        logger.info(f"Registered workflow template: {template.name}")
    
    async def execute_template(self, template_name: str, parameters: Optional[Dict[str, Any]] = None,
                             context: Optional[ExecutionContext] = None) -> IntegrationResult:
        """Execute a workflow from template"""
        if template_name not in self.workflow_templates:
            raise ValueError(f"Workflow template not found: {template_name}")
        
        template = self.workflow_templates[template_name]
        
        # Create workflow instance from template
        workflow_instance = self._instantiate_workflow(template, parameters or {})
        
        # Execute workflow
        return await self.integration_engine.execute_workflow(workflow_instance, context)
    
    def _instantiate_workflow(self, template: WorkflowDefinition, 
                            parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create workflow instance from template with parameters"""
        # Simple parameter substitution (can be enhanced)
        instantiated_steps = []
        
        for step in template.steps:
            instantiated_step = step.copy()
            
            # Substitute parameters in step configuration
            if 'parameters' in instantiated_step:
                for param_name, param_value in instantiated_step['parameters'].items():
                    if isinstance(param_value, str) and param_value.startswith('${'):
                        # Parameter substitution
                        param_key = param_value[2:-1]  # Remove ${ and }
                        if param_key in parameters:
                            instantiated_step['parameters'][param_name] = parameters[param_key]
            
            instantiated_steps.append(instantiated_step)
        
        return WorkflowDefinition(
            name=f"{template.name}_instance_{uuid.uuid4().hex[:8]}",
            description=template.description,
            steps=instantiated_steps,
            metadata={**template.metadata, 'template_name': template.name, 'parameters': parameters}
        )
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get orchestration engine status"""
        return {
            'registered_templates': len(self.workflow_templates),
            'template_names': list(self.workflow_templates.keys()),
            'active_executions': len(self.active_executions),
            'statistics': self.stats.copy()
        }


class WorkflowOrchestrator:
    """
    High-level orchestrator for managing workflow execution and coordination.
    """
    
    def __init__(self, integration_engine: IntegrationEngine, orchestration_engine: OrchestrationEngine):
        self.integration_engine = integration_engine
        self.orchestration_engine = orchestration_engine
        
        # Pre-defined workflow templates
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register default workflow templates"""
        # Recipe optimization workflow
        recipe_optimization_workflow = WorkflowDefinition(
            name="recipe_optimization",
            description="Optimize recipe using genetic algorithm and predictive models",
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
                    'dependencies': ['parse_recipe'],
                    'required': False
                },
                {
                    'step_id': 'optimize_parameters',
                    'component_type': 'genetic_algorithm',
                    'action': 'optimize',
                    'parameters': {
                        'objective_function': '${objective_function}',
                        'parameter_space': '${parameter_space}'
                    },
                    'dependencies': ['parse_recipe'],
                    'required': True
                },
                {
                    'step_id': 'validate_optimized_recipe',
                    'component_type': 'multi_agent',
                    'action': 'validate_recipe',
                    'parameters': {'optimized_recipe': '${optimized_parameters}'},
                    'dependencies': ['optimize_parameters'],
                    'required': True
                }
            ]
        )
        
        self.orchestration_engine.register_workflow_template(recipe_optimization_workflow)
        
        # Multi-agent testing workflow
        testing_workflow = WorkflowDefinition(
            name="comprehensive_testing",
            description="Execute comprehensive testing using multiple agents",
            steps=[
                {
                    'step_id': 'setup_test_environment',
                    'component_type': 'multi_agent',
                    'action': 'setup_environment',
                    'parameters': {'test_config': '${test_config}'},
                    'required': True
                },
                {
                    'step_id': 'execute_tests',
                    'component_type': 'multi_agent',
                    'action': 'execute_tests',
                    'parameters': {'test_suite': '${test_suite}'},
                    'dependencies': ['setup_test_environment'],
                    'required': True
                },
                {
                    'step_id': 'analyze_results',
                    'component_type': 'nlp_system',
                    'action': 'interpret_test_results',
                    'parameters': {'test_results': '${test_results}'},
                    'dependencies': ['execute_tests'],
                    'required': True
                },
                {
                    'step_id': 'generate_recommendations',
                    'component_type': 'predictive_optimization',
                    'action': 'generate_recommendations',
                    'parameters': {'analysis_results': '${analysis_results}'},
                    'dependencies': ['analyze_results'],
                    'required': False
                }
            ]
        )
        
        self.orchestration_engine.register_workflow_template(testing_workflow)
    
    async def optimize_recipe(self, recipe_text: str, optimization_config: Dict[str, Any]) -> IntegrationResult:
        """Execute recipe optimization workflow"""
        parameters = {
            'recipe_text': recipe_text,
            'objective_function': optimization_config.get('objective_function'),
            'parameter_space': optimization_config.get('parameter_space', {})
        }
        
        return await self.orchestration_engine.execute_template('recipe_optimization', parameters)
    
    async def run_comprehensive_testing(self, test_config: Dict[str, Any]) -> IntegrationResult:
        """Execute comprehensive testing workflow"""
        parameters = {
            'test_config': test_config,
            'test_suite': test_config.get('test_suite', [])
        }
        
        return await self.orchestration_engine.execute_template('comprehensive_testing', parameters)
    
    def get_available_workflows(self) -> List[str]:
        """Get list of available workflow templates"""
        return list(self.orchestration_engine.workflow_templates.keys())
