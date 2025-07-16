"""
Orchestration Manager

Main integration point for the orchestration system. Coordinates all components
and provides a unified interface for multi-agent multi-MCP orchestration.
"""

from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
import asyncio
import logging
import json
from pathlib import Path

# Note: ToT imports removed as unused after A2A archival

from .coordination_models import (
    OrchestrationConfig, AgentCapability, MCPServerInfo, TaskRequest,
    PerformanceMetrics, AgentType, MCPServerType, TaskPriority
)
from .agent_registry import AgentRegistry
from .mcp_orchestrator import MCPOrchestrator
from .task_dispatcher import TaskDispatcher
from .metrics_collector import MetricsCollector
from .adaptive_load_balancer import AdaptiveLoadBalancer
from .transaction_coordinator import TransactionCoordinator
from .emergent_behavior_detector import EmergentBehaviorDetector
from .meta_learning_engine import MetaLearningEngine
from .predictive_optimizer import PredictiveOptimizer
from .pygent_integration import PyGentIntegration
from .production_config import ProductionConfig, ProductionDeployment
from .documentation_orchestrator import DocumentationOrchestrator

# Import Research Orchestrator
try:
    from .research_orchestrator import ResearchOrchestrator
    from .research_integration import ResearchOrchestrationManager, initialize_research_system
    RESEARCH_AVAILABLE = True
except ImportError:
    ResearchOrchestrator = None
    ResearchOrchestrationManager = None
    initialize_research_system = None
    RESEARCH_AVAILABLE = False

# Import A2A components
try:
    from ..a2a import A2AServer, AgentDiscoveryService, AgentCard
    A2A_AVAILABLE = True
except ImportError:
    A2AServer = None
    AgentDiscoveryService = None
    AgentCard = None
    A2A_AVAILABLE = False

# Import integration components
try:
    from ..integration.events import EventBus
    from ..integration.workflows import WorkflowManager
    INTEGRATION_AVAILABLE = True
except ImportError:
    EventBus = None
    WorkflowManager = None
    INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class OrchestrationManager:
    """
    Main orchestration manager that coordinates all orchestration components.
    
    Features:
    - Unified interface for orchestration operations
    - Component lifecycle management
    - Configuration management
    - Health monitoring and reporting
    - Integration with existing PyGent Factory systems
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None, a2a_manager: Optional['A2AManager'] = None):
        self.config = config or OrchestrationConfig()

        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid orchestration configuration")

        # A2A Protocol Integration
        self.a2a_manager = a2a_manager

        # A2A Coordination Engine
        if self.a2a_manager:
            from .a2a_coordination_strategies import A2ACoordinationEngine
            self.a2a_coordination_engine = A2ACoordinationEngine(self.a2a_manager, self)
        else:
            self.a2a_coordination_engine = None
        
        # Initialize core integration components
        if INTEGRATION_AVAILABLE:
            self.event_bus = EventBus()
            self.workflow_manager = WorkflowManager()
        else:
            self.event_bus = None
            self.workflow_manager = None
            logger.warning("Integration components not available - EventBus and WorkflowManager disabled")

        # Initialize components
        self.agent_registry = AgentRegistry(self.config)
        self.mcp_orchestrator = MCPOrchestrator(self.config)
        self.task_dispatcher = TaskDispatcher(
            self.config, self.agent_registry, self.mcp_orchestrator, self.a2a_manager
        )
        self.metrics_collector = MetricsCollector(
            self.config, self.agent_registry, self.mcp_orchestrator, self.task_dispatcher
        )
        
        # Phase 2 components
        self.adaptive_load_balancer = AdaptiveLoadBalancer(self.config)
        self.transaction_coordinator = TransactionCoordinator(self.config, self.mcp_orchestrator)
        self.emergent_behavior_detector = EmergentBehaviorDetector(self.config)
        
        # Phase 3 components
        self.meta_learning_engine = MetaLearningEngine(self.config)
        self.predictive_optimizer = PredictiveOptimizer(self.config)
        self.pygent_integration = PyGentIntegration(self.config)
        self.production_deployment = None  # Initialized when needed

        # Documentation orchestrator
        self.documentation_orchestrator = DocumentationOrchestrator(
            self, getattr(self.config, 'documentation_config', None)
        )

        # Research orchestrator (if available)
        if RESEARCH_AVAILABLE:
            self.research_orchestrator = ResearchOrchestrationManager(self.config)
            logger.info("Research orchestrator initialized")
        else:
            self.research_orchestrator = None
            logger.warning("Research orchestrator not available")

        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Integration callbacks
        self.status_callbacks: List[Callable] = []
        
        logger.info("Orchestration Manager initialized")
    
    async def start(self):
        """Start the orchestration system."""
        try:
            logger.info("Starting Orchestration Manager...")

            # Start integration components first
            if self.event_bus:
                await self.event_bus.start()
            if self.workflow_manager:
                # WorkflowManager doesn't have async start method, just initialize
                pass

            # Start components in order
            await self.agent_registry.start()
            await self.mcp_orchestrator.start()
            await self.task_dispatcher.start()
            await self.metrics_collector.start()

            # Start A2A manager if available
            if self.a2a_manager:
                await self.a2a_manager.initialize()
                logger.info("A2A manager started successfully")
            
            # Start Phase 2 components
            await self.transaction_coordinator.start()
            await self.emergent_behavior_detector.start()

            # Phase 4 Integration: Connect to agent factory simulation environment
            await self._integrate_phase4_components()
            
            # Start Phase 3 components
            await self.meta_learning_engine.start()
            await self.predictive_optimizer.start()
            await self.pygent_integration.start()

            # Start documentation orchestrator
            await self.documentation_orchestrator.start()

            # Start research orchestrator (if available)
            if RESEARCH_AVAILABLE:
                await self.research_orchestrator.start()

            self.is_running = True
            self.start_time = datetime.utcnow()
            
            logger.info("Orchestration Manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Orchestration Manager: {e}")
            await self.stop()  # Cleanup on failure
            raise
    
    async def stop(self):
        """Stop the orchestration system."""
        try:
            logger.info("Stopping Orchestration Manager...")
            
            self.is_running = False
            
            # Stop components in reverse order
            # Stop documentation orchestrator first
            await self.documentation_orchestrator.stop()

            # Stop research orchestrator (if available)
            if self.research_orchestrator:
                await self.research_orchestrator.stop()

            # Stop Phase 3 components
            await self.pygent_integration.stop()
            await self.predictive_optimizer.stop()
            await self.meta_learning_engine.stop()
            
            # Stop Phase 2 components
            await self.emergent_behavior_detector.stop()
            await self.transaction_coordinator.stop()

            # Stop A2A manager if available
            if self.a2a_manager:
                await self.a2a_manager.shutdown()
                logger.info("A2A manager stopped")

            await self.metrics_collector.stop()
            await self.task_dispatcher.stop()
            await self.mcp_orchestrator.stop()
            await self.agent_registry.stop()

            # Stop integration components last
            if self.event_bus:
                await self.event_bus.stop()

            logger.info("Orchestration Manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Orchestration Manager: {e}")
    
    # Agent Management
    async def register_agent(self, agent_capability: AgentCapability) -> bool:
        """Register a new agent."""
        return await self.agent_registry.register_agent(agent_capability)
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        return await self.agent_registry.unregister_agent(agent_id)
    
    async def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get agent status."""
        return await self.agent_registry.get_agent_status(agent_id)
    
    # MCP Server Management
    async def register_mcp_server(self, server_info: MCPServerInfo) -> bool:
        """Register a new MCP server."""
        return await self.mcp_orchestrator.register_server(server_info)
    
    async def unregister_mcp_server(self, server_id: str) -> bool:
        """Unregister an MCP server."""
        return await self.mcp_orchestrator.unregister_server(server_id)
    
    async def get_mcp_server_status(self, server_id: Optional[str] = None) -> Dict[str, Any]:
        """Get MCP server status."""
        return await self.mcp_orchestrator.get_server_status(server_id)
    
    # Task Management
    async def submit_task(self, task: TaskRequest) -> bool:
        """Submit a task for execution."""
        return await self.task_dispatcher.submit_task(task)
    
    async def submit_batch_tasks(self, tasks: List[TaskRequest]) -> List[bool]:
        """Submit multiple tasks."""
        return await self.task_dispatcher.submit_batch_tasks(tasks)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        return await self.task_dispatcher.cancel_task(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        return await self.task_dispatcher.get_task_status(task_id)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return await self.task_dispatcher.get_queue_status()
    
    # Metrics and Monitoring
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        return await self.metrics_collector.get_current_metrics()
    
    async def get_historical_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics."""
        return await self.metrics_collector.get_historical_metrics(hours)
    
    async def get_performance_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends."""
        return await self.metrics_collector.get_performance_trends(metric_name, hours)
    
    async def get_alerts(self, active_only: bool = True) -> Dict[str, Any]:
        """Get system alerts."""
        return await self.metrics_collector.get_alerts(active_only)
    
    # Evolution Management
    # Phase 3: Advanced Intelligence Methods
    async def execute_tot_reasoning(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute Tree of Thought reasoning via PyGent integration."""
        return await self.pygent_integration.execute_tot_reasoning(problem, context)
    
    async def execute_rag_workflow(self, query: str, domain: str = None) -> Dict[str, Any]:
        """Execute complete RAG workflow (retrieval + generation)."""
        # Retrieve documents
        retrieval_result = await self.pygent_integration.execute_rag_retrieval(query, domain)
        
        # Generate response
        context_docs = [doc.get('title', '') for doc in retrieval_result.get('documents', [])]
        generation_result = await self.pygent_integration.execute_rag_generation(context_docs, query)
        
        return {
            'retrieval': retrieval_result,
            'generation': generation_result,
            'total_time': retrieval_result.get('retrieval_time', 0) + generation_result.get('generation_time', 0)
        }
    
    async def execute_research_workflow(self, topic: str, workflow_type: str = "comprehensive") -> Dict[str, Any]:
        """Execute automated research workflow."""
        return await self.pygent_integration.execute_research_workflow(topic, workflow_type)
    
    async def predict_system_performance(self, hours_ahead: int = 1) -> Dict[str, float]:
        """Predict system performance metrics."""
        horizon = timedelta(hours=hours_ahead)
        return await self.predictive_optimizer.predict_metrics(horizon)
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get system optimization recommendations."""
        recommendations = await self.predictive_optimizer.generate_optimization_recommendations()
        return [
            {
                'recommendation_id': rec.recommendation_id,
                'objective': rec.objective.value,
                'predicted_improvement': rec.predicted_improvement,
                'confidence': rec.confidence,
                'actions': rec.actions,
                'estimated_cost': rec.estimated_cost,
                'risk_level': rec.risk_level,
                'implementation_time_minutes': rec.implementation_time.total_seconds() / 60
            }
            for rec in recommendations
        ]
    
    async def detect_system_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect current and predicted system bottlenecks."""
        return await self.predictive_optimizer.detect_bottlenecks()
    
    async def learn_from_task_execution(self, task_id: str, performance_data: Dict[str, Any]) -> bool:
        """Learn from task execution using meta-learning."""
        try:
            from .meta_learning_engine import MetaLearningTask
            
            # Create meta-learning task
            meta_task = MetaLearningTask(
                task_id=task_id,
                domain=performance_data.get('domain', 'general'),
                task_type=performance_data.get('task_type', 'unknown'),
                support_set=[],
                query_set=[],
                meta_features=performance_data.get('features', {}),
                difficulty_score=performance_data.get('difficulty', 0.5)
            )
            
            # Learn from task
            experience = await self.meta_learning_engine.learn_from_task(meta_task)
            
            return experience.final_performance > experience.initial_performance
            
        except Exception as e:
            logger.error(f"Meta-learning from task failed: {e}")
            return False
    
    async def deploy_to_production(self, production_config: Dict[str, Any] = None) -> bool:
        """Deploy system to production environment."""
        try:
            from .production_config import create_production_config
            
            # Create production configuration
            if production_config:
                config = ProductionConfig(**production_config)
            else:
                config = create_production_config()
            
            # Initialize deployment
            self.production_deployment = ProductionDeployment(config)
            
            # Deploy system
            success = await self.production_deployment.deploy()
            
            if success:
                logger.info("Production deployment successful")
            else:
                logger.error("Production deployment failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            return False
    
    async def get_pygent_component_health(self) -> Dict[str, Any]:
        """Get health status of PyGent Factory components."""
        return await self.pygent_integration.get_component_health()

    # Documentation Management
    async def build_documentation(self, production: bool = False) -> str:
        """Build documentation using orchestrated workflow."""
        from .documentation_models import DocumentationWorkflowType

        workflow_type = DocumentationWorkflowType.PRODUCTION_DEPLOY if production else DocumentationWorkflowType.BUILD_AND_SYNC
        return await self.documentation_orchestrator.execute_workflow(workflow_type, {"production": production})

    async def start_documentation_dev_mode(self) -> str:
        """Start documentation development mode with hot reload."""
        from .documentation_models import DocumentationWorkflowType

        return await self.documentation_orchestrator.execute_workflow(
            DocumentationWorkflowType.DEVELOPMENT_MODE
        )

    async def check_documentation_health(self) -> Dict[str, Any]:
        """Perform documentation system health check."""
        from .documentation_models import DocumentationWorkflowType

        workflow_id = await self.documentation_orchestrator.execute_workflow(
            DocumentationWorkflowType.HEALTH_CHECK
        )

        # Get workflow status
        workflow = self.documentation_orchestrator.get_workflow_status(workflow_id)
        if workflow and workflow.status.value == "completed":
            # Extract health check results from the last task
            health_task = next((task for task in workflow.tasks if task.task_type.value == "health_monitor"), None)
            if health_task and health_task.result:
                return health_task.result

        return await self.documentation_orchestrator.get_system_status()

    async def get_documentation_status(self) -> Dict[str, Any]:
        """Get comprehensive documentation system status."""
        return await self.documentation_orchestrator.get_system_status()

    async def list_documentation_workflows(self) -> List[Dict[str, Any]]:
        """List active documentation workflows."""
        workflows = self.documentation_orchestrator.list_active_workflows()
        return [
            {
                "workflow_id": w.workflow_id,
                "workflow_type": w.workflow_type.value,
                "name": w.name,
                "status": w.status.value,
                "progress_percentage": w.progress_percentage,
                "current_task": w.current_task,
                "started_at": w.started_at.isoformat() if w.started_at else None,
                "created_at": w.created_at.isoformat()
            }
            for w in workflows
        ]

    async def cancel_documentation_workflow(self, workflow_id: str) -> bool:
        """Cancel a documentation workflow."""
        try:
            await self.documentation_orchestrator.cancel_workflow(workflow_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel documentation workflow {workflow_id}: {e}")
            return False
    
    # Phase 2: Advanced Coordination Methods
    async def begin_distributed_transaction(self, 
                                          operations: List[Dict[str, Any]],
                                          timeout_minutes: int = 5) -> str:
        """Begin a distributed transaction across multiple MCP servers."""
        timeout = timedelta(minutes=timeout_minutes)
        return await self.transaction_coordinator.begin_transaction(operations, timeout)
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a distributed transaction."""
        return await self.transaction_coordinator.commit_transaction(transaction_id)
    
    async def abort_transaction(self, transaction_id: str, reason: str = "Manual abort") -> bool:
        """Abort a distributed transaction."""
        return await self.transaction_coordinator.abort_transaction(transaction_id, reason)
    
    async def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a distributed transaction."""
        return await self.transaction_coordinator.get_transaction_status(transaction_id)
    
    async def optimize_task_allocation(self, 
                                     task: TaskRequest,
                                     available_agents: List[str],
                                     available_servers: List[str]) -> Optional[Dict[str, Any]]:
        """Optimize task allocation using advanced load balancing."""
        # Get agent and server objects
        agents = [self.agent_registry.agents.get(aid) for aid in available_agents]
        servers = [self.mcp_orchestrator.servers.get(sid) for sid in available_servers]
        
        # Filter out None values
        agents = [a for a in agents if a is not None]
        servers = [s for s in servers if s is not None]
        
        if not agents or not servers:
            return None
        
        allocation = await self.adaptive_load_balancer.optimize_allocation(task, agents, servers)
        
        if allocation:
            return {
                'agent_id': allocation.agent_id,
                'server_id': allocation.server_id,
                'allocation_score': allocation.allocation_score,
                'expected_completion_time': allocation.expected_completion_time,
                'resource_cost': allocation.resource_cost
            }
        
        return None
    
    async def predict_agent_load(self, agent_id: str, hours_ahead: int = 1) -> Optional[Dict[str, Any]]:
        """Predict future load for an agent."""
        time_horizon = timedelta(hours=hours_ahead)
        prediction = await self.adaptive_load_balancer.predict_load(agent_id, time_horizon)
        
        return {
            'agent_id': prediction.agent_id,
            'predicted_load': prediction.predicted_load,
            'confidence': prediction.confidence,
            'time_horizon_hours': hours_ahead,
            'factors': prediction.factors
        }
    
    async def detect_emergent_behaviors(self) -> List[Dict[str, Any]]:
        """Get detected emergent behaviors."""
        patterns = list(self.emergent_behavior_detector.detected_patterns.values())
        
        return [
            {
                'pattern_id': pattern.pattern_id,
                'behavior_type': pattern.behavior_type.value,
                'significance': pattern.significance.value,
                'description': pattern.description,
                'performance_improvement': pattern.performance_improvement,
                'efficiency_gain': pattern.efficiency_gain,
                'detection_confidence': pattern.detection_confidence,
                'observation_count': pattern.observation_count,
                'reproduction_rate': pattern.reproduction_rate,
                'first_observed': pattern.first_observed.isoformat(),
                'last_observed': pattern.last_observed.isoformat()
            }
            for pattern in patterns
        ]
    
    async def observe_system_for_behaviors(self) -> bool:
        """Trigger system observation for emergent behavior detection."""
        try:
            # Get current system state
            agents_state = await self.get_agent_status()
            servers_state = await self.get_mcp_server_status()
            metrics = await self.metrics_collector.collect_metrics()
            
            # Get task data (simplified)
            task_data = {
                'queue_status': await self.get_queue_status()
            }
            
            # Observe system state
            await self.emergent_behavior_detector.observe_system_state(
                agents_state, servers_state, metrics, task_data
            )
            
            return True
            
        except Exception as e:
            logger.error(f"System observation failed: {e}")
            return False

    # A2A Protocol Integration Methods
    async def discover_a2a_agents(self) -> List[Dict[str, Any]]:
        """Discover available A2A agents across the orchestration system."""
        if not self.a2a_manager:
            logger.warning("A2A manager not available for agent discovery")
            return []

        try:
            # Get A2A agent status from manager
            status = await self.a2a_manager.get_agent_status()
            return status.get("agents", [])

        except Exception as e:
            logger.error(f"Failed to discover A2A agents: {e}")
            return []

    async def execute_a2a_workflow(self,
                                 workflow_description: str,
                                 agent_ids: List[str],
                                 coordination_strategy: str = "sequential",
                                 workflow_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow using A2A protocol for agent coordination."""
        if not self.a2a_manager:
            return {"error": "A2A manager not available"}

        try:
            # Validate agents exist in registry
            available_agents = []
            for agent_id in agent_ids:
                agent_status = await self.get_agent_status(agent_id)
                if agent_status and agent_status.get("status") == "available":
                    available_agents.append(agent_id)
                else:
                    logger.warning(f"Agent {agent_id} not available for A2A workflow")

            if not available_agents:
                return {"error": "No available agents for A2A workflow"}

            # Execute A2A coordination
            results = await self.a2a_manager.coordinate_multi_agent_task(
                task_description=workflow_description,
                agent_ids=available_agents,
                coordination_strategy=coordination_strategy
            )

            # Collect workflow metrics
            workflow_metrics = {
                "workflow_id": f"a2a_workflow_{datetime.utcnow().timestamp()}",
                "description": workflow_description,
                "coordination_strategy": coordination_strategy,
                "requested_agents": len(agent_ids),
                "available_agents": len(available_agents),
                "completed_tasks": len(results),
                "success_rate": len(results) / len(available_agents) if available_agents else 0,
                "metadata": workflow_metadata or {}
            }

            return {
                "success": True,
                "workflow_metrics": workflow_metrics,
                "task_results": [
                    {
                        "task_id": task.id,
                        "session_id": task.sessionId,
                        "status": task.status.state.value
                    }
                    for task in results
                ]
            }

        except Exception as e:
            logger.error(f"Failed to execute A2A workflow: {e}")
            return {"error": str(e)}

    async def coordinate_distributed_agents(self,
                                          task_description: str,
                                          local_agents: List[str],
                                          remote_agents: List[str] = None,
                                          coordination_mode: str = "hybrid") -> Dict[str, Any]:
        """Coordinate both local and distributed A2A agents."""
        if not self.a2a_manager:
            return {"error": "A2A manager not available"}

        try:
            all_agents = local_agents.copy()
            if remote_agents:
                all_agents.extend(remote_agents)

            # Execute coordination with mixed agent types
            if coordination_mode == "hybrid":
                # Execute local agents first, then remote
                local_results = await self.a2a_manager.coordinate_multi_agent_task(
                    task_description=task_description,
                    agent_ids=local_agents,
                    coordination_strategy="parallel"
                )

                remote_results = []
                if remote_agents:
                    remote_results = await self.a2a_manager.coordinate_multi_agent_task(
                        task_description=task_description,
                        agent_ids=remote_agents,
                        coordination_strategy="parallel"
                    )

                all_results = local_results + remote_results

            else:
                # Execute all agents with specified strategy
                all_results = await self.a2a_manager.coordinate_multi_agent_task(
                    task_description=task_description,
                    agent_ids=all_agents,
                    coordination_strategy=coordination_mode
                )

            return {
                "success": True,
                "coordination_mode": coordination_mode,
                "local_agents": len(local_agents),
                "remote_agents": len(remote_agents) if remote_agents else 0,
                "total_results": len(all_results),
                "results": all_results
            }

        except Exception as e:
            logger.error(f"Failed to coordinate distributed agents: {e}")
            return {"error": str(e)}

    async def get_a2a_orchestration_metrics(self) -> Dict[str, Any]:
        """Get A2A orchestration metrics."""
        if not self.a2a_manager:
            return {"error": "A2A manager not available"}

        try:
            # Get A2A manager status
            a2a_status = await self.a2a_manager.get_agent_status()

            # Get orchestration system metrics
            system_metrics = await self.get_system_metrics()

            # Combine metrics
            return {
                "a2a_enabled": True,
                "a2a_agents": a2a_status.get("total_agents", 0),
                "a2a_active_sessions": a2a_status.get("active_sessions", 0),
                "orchestration_agents": system_metrics.get("agent_registry", {}).get("total_agents", 0),
                "integration_health": "healthy" if self.a2a_manager else "unavailable",
                "last_updated": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get A2A orchestration metrics: {e}")
            return {"error": str(e)}

    async def execute_coordination_strategy(self,
                                          coordination_id: str,
                                          strategy: str,
                                          task_descriptions: List[str],
                                          agent_assignments: Optional[Dict[str, str]] = None,
                                          coordination_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a coordination strategy with multiple tasks."""
        if not self.a2a_coordination_engine:
            return {"error": "A2A coordination engine not available"}

        try:
            from .a2a_coordination_strategies import CoordinationStrategy, CoordinationTask

            # Convert strategy string to enum
            try:
                strategy_enum = CoordinationStrategy(strategy.lower())
            except ValueError:
                return {"error": f"Invalid coordination strategy: {strategy}"}

            # Create coordination tasks
            coordination_tasks = []
            for i, description in enumerate(task_descriptions):
                task_id = f"{coordination_id}_task_{i+1}"
                agent_id = agent_assignments.get(task_id) if agent_assignments else None

                coordination_task = CoordinationTask(
                    task_id=task_id,
                    description=description,
                    agent_id=agent_id,
                    priority=1,
                    timeout_seconds=300
                )
                coordination_tasks.append(coordination_task)

            # Execute coordination
            result = await self.a2a_coordination_engine.execute_coordination(
                coordination_id=coordination_id,
                strategy=strategy_enum,
                tasks=coordination_tasks,
                coordination_metadata=coordination_metadata
            )

            # Convert result to dictionary
            return {
                "success": True,
                "coordination_id": coordination_id,
                "strategy": result.strategy.value,
                "total_tasks": result.total_tasks,
                "successful_tasks": result.successful_tasks,
                "failed_tasks": result.failed_tasks,
                "success_rate": result.success_rate,
                "execution_time": result.execution_time,
                "results": result.results,
                "metadata": result.coordination_metadata
            }

        except Exception as e:
            logger.error(f"Coordination strategy execution failed: {e}")
            return {"error": str(e)}

    async def get_coordination_strategies(self) -> List[str]:
        """Get available coordination strategies."""
        try:
            from .a2a_coordination_strategies import CoordinationStrategy
            return [strategy.value for strategy in CoordinationStrategy]
        except Exception as e:
            logger.error(f"Failed to get coordination strategies: {e}")
            return []

    async def get_coordination_performance(self) -> Dict[str, Any]:
        """Get coordination strategy performance metrics."""
        if not self.a2a_coordination_engine:
            return {"error": "A2A coordination engine not available"}

        try:
            performance = self.a2a_coordination_engine.get_strategy_performance()
            history = self.a2a_coordination_engine.get_coordination_history(limit=10)

            return {
                "strategy_performance": performance,
                "recent_coordinations": history,
                "total_coordinations": len(self.a2a_coordination_engine.coordination_history),
                "active_coordinations": len(self.a2a_coordination_engine.active_coordinations)
            }

        except Exception as e:
            logger.error(f"Failed to get coordination performance: {e}")
            return {"error": str(e)}

    async def execute_multi_strategy_workflow(self,
                                            workflow_id: str,
                                            workflow_stages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a workflow with multiple coordination strategies."""
        if not self.a2a_coordination_engine:
            return {"error": "A2A coordination engine not available"}

        try:
            workflow_results = []
            overall_start_time = datetime.utcnow()

            for i, stage in enumerate(workflow_stages):
                stage_id = f"{workflow_id}_stage_{i+1}"
                strategy = stage.get("strategy", "sequential")
                tasks = stage.get("tasks", [])
                metadata = stage.get("metadata", {})

                logger.info(f"Executing workflow stage {i+1}/{len(workflow_stages)}: {strategy}")

                # Execute stage coordination
                stage_result = await self.execute_coordination_strategy(
                    coordination_id=stage_id,
                    strategy=strategy,
                    task_descriptions=tasks,
                    coordination_metadata=metadata
                )

                stage_result["stage_number"] = i + 1
                stage_result["stage_id"] = stage_id
                workflow_results.append(stage_result)

                # Check if stage failed and workflow should stop
                if not stage_result.get("success", False):
                    logger.warning(f"Workflow stage {i+1} failed, stopping workflow")
                    break

            # Calculate overall workflow metrics
            total_execution_time = (datetime.utcnow() - overall_start_time).total_seconds()
            successful_stages = sum(1 for r in workflow_results if r.get("success", False))

            return {
                "success": True,
                "workflow_id": workflow_id,
                "total_stages": len(workflow_stages),
                "completed_stages": len(workflow_results),
                "successful_stages": successful_stages,
                "total_execution_time": total_execution_time,
                "stage_results": workflow_results
            }

        except Exception as e:
            logger.error(f"Multi-strategy workflow execution failed: {e}")
            return {"error": str(e)}

    # System Status and Health
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                "is_running": self.is_running,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "uptime_seconds": (
                    (datetime.utcnow() - self.start_time).total_seconds()
                    if self.start_time else 0
                ),
                "config": {
                    "evolution_enabled": self.config.evolution_enabled,
                    "max_concurrent_tasks": self.config.max_concurrent_tasks,
                    "default_strategy": self.config.default_strategy.value,
                    "batch_processing_enabled": self.config.batch_processing_enabled
                }
            }
            
            # Add component status
            if self.is_running:
                status["components"] = {
                    "agent_registry": await self.agent_registry.get_registry_metrics(),
                    "mcp_orchestrator": await self.mcp_orchestrator.get_orchestration_metrics(),
                    "task_dispatcher": await self.task_dispatcher.get_dispatcher_metrics(),
                    "metrics_collector": {
                        "is_running": self.metrics_collector.is_running,
                        "historical_data_points": len(self.metrics_collector.historical_metrics)
                    }
                }
                
                # Phase 2 components
                status["components"]["adaptive_load_balancer"] = await self.adaptive_load_balancer.get_load_balancer_metrics()
                status["components"]["transaction_coordinator"] = await self.transaction_coordinator.get_coordinator_metrics()
                status["components"]["emergent_behavior_detector"] = await self.emergent_behavior_detector.get_detector_metrics()
                
                # Phase 3 components
                status["components"]["meta_learning_engine"] = await self.meta_learning_engine.get_meta_learning_metrics()
                status["components"]["predictive_optimizer"] = await self.predictive_optimizer.get_optimizer_metrics()
                status["components"]["pygent_integration"] = await self.pygent_integration.get_integration_metrics()

                # Documentation orchestrator
                status["components"]["documentation_orchestrator"] = await self.documentation_orchestrator.get_system_status()

                # A2A Protocol Integration
                if self.a2a_manager:
                    status["components"]["a2a_manager"] = await self.get_a2a_orchestration_metrics()
                else:
                    status["components"]["a2a_manager"] = {"status": "not_available"}

            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {
            "overall_health": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        issues = []
        
        try:
            # Check if system is running
            if not self.is_running:
                issues.append("System is not running")
                health["overall_health"] = "unhealthy"
            
            # Check component health
            if self.is_running:
                # Agent registry health
                agent_metrics = await self.agent_registry.get_registry_metrics()
                if agent_metrics["total_agents"] == 0:
                    issues.append("No agents registered")
                elif agent_metrics["agent_availability_rate"] < 0.5:
                    issues.append("Low agent availability")
                
                health["components"]["agent_registry"] = {
                    "status": "healthy" if agent_metrics["total_agents"] > 0 else "warning",
                    "total_agents": agent_metrics["total_agents"],
                    "available_agents": agent_metrics["available_agents"]
                }
                
                # MCP orchestrator health
                mcp_metrics = await self.mcp_orchestrator.get_orchestration_metrics()
                if mcp_metrics["total_servers"] == 0:
                    issues.append("No MCP servers registered")
                elif mcp_metrics["server_health_rate"] < 0.8:
                    issues.append("Poor MCP server health")
                
                health["components"]["mcp_orchestrator"] = {
                    "status": "healthy" if mcp_metrics["server_health_rate"] > 0.8 else "warning",
                    "total_servers": mcp_metrics["total_servers"],
                    "healthy_servers": mcp_metrics["healthy_servers"]
                }
                
                # Task dispatcher health
                queue_status = await self.task_dispatcher.get_queue_status()
                if queue_status["pending_tasks"] > 100:
                    issues.append("High task queue backlog")
                
                health["components"]["task_dispatcher"] = {
                    "status": "healthy" if queue_status["pending_tasks"] < 50 else "warning",
                    "pending_tasks": queue_status["pending_tasks"],
                    "running_tasks": queue_status["running_tasks"]
                }
                
                # Check for active alerts
                alerts = await self.metrics_collector.get_alerts(active_only=True)
                if alerts:
                    issues.extend([f"Active alert: {name}" for name in alerts.keys()])
                    if health["overall_health"] == "healthy":
                        health["overall_health"] = "warning"
            
            # Set overall health based on issues
            if issues:
                health["issues"] = issues
                if health["overall_health"] == "healthy":
                    health["overall_health"] = "warning"
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall_health": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Integration Helpers
    async def create_tot_agent(self, name: str, capabilities: List[str]) -> str:
        """Create a Tree of Thought agent."""
        agent_capability = AgentCapability(
            agent_id=f"tot_agent_{datetime.utcnow().timestamp()}",
            agent_type=AgentType.TOT_REASONING,
            name=name,
            description=f"Tree of Thought reasoning agent: {name}",
            supported_tasks=set(capabilities),
            max_concurrent_tasks=3,
            specializations=["reasoning", "problem_solving"]
        )
        
        success = await self.register_agent(agent_capability)
        return agent_capability.agent_id if success else ""
    
    async def create_rag_agent(self, name: str, agent_type: str = "retrieval") -> str:
        """Create a RAG agent."""
        if agent_type == "retrieval":
            agent_type_enum = AgentType.RAG_RETRIEVAL
            capabilities = {"document_retrieval", "knowledge_search", "context_gathering"}
        else:
            agent_type_enum = AgentType.RAG_GENERATION
            capabilities = {"text_generation", "content_creation", "response_synthesis"}
        
        agent_capability = AgentCapability(
            agent_id=f"rag_{agent_type}_agent_{datetime.utcnow().timestamp()}",
            agent_type=agent_type_enum,
            name=name,
            description=f"RAG {agent_type} agent: {name}",
            supported_tasks=capabilities,
            max_concurrent_tasks=5,
            specializations=["rag", agent_type]
        )
        
        success = await self.register_agent(agent_capability)
        return agent_capability.agent_id if success else ""
    
    async def create_evaluation_agent(self, name: str, evaluation_types: List[str]) -> str:
        """Create an evaluation agent."""
        agent_capability = AgentCapability(
            agent_id=f"eval_agent_{datetime.utcnow().timestamp()}",
            agent_type=AgentType.EVALUATION,
            name=name,
            description=f"Evaluation agent: {name}",
            supported_tasks=set(evaluation_types),
            max_concurrent_tasks=10,
            specializations=["evaluation", "assessment", "quality_control"]
        )
        
        success = await self.register_agent(agent_capability)
        return agent_capability.agent_id if success else ""
    
    async def register_existing_mcp_servers(self):
        """Register existing MCP servers from PyGent Factory."""
        # Filesystem server
        filesystem_server = MCPServerInfo(
            server_id="filesystem_server",
            server_type=MCPServerType.FILESYSTEM,
            name="Filesystem MCP Server",
            endpoint="npx -y @modelcontextprotocol/server-filesystem D:/mcp/pygent-factory",
            capabilities={"file_read", "file_write", "directory_list", "file_search"}
        )
        await self.register_mcp_server(filesystem_server)
        
        # PostgreSQL server
        postgresql_server = MCPServerInfo(
            server_id="postgresql_server",
            server_type=MCPServerType.POSTGRESQL,
            name="PostgreSQL MCP Server",
            endpoint="npx -y @modelcontextprotocol/server-postgres postgresql://postgres:postgres@localhost:54321/pygent_factory",
            capabilities={"database_query", "data_storage", "data_retrieval", "transaction_management"}
        )
        await self.register_mcp_server(postgresql_server)
        
        # GitHub server
        github_server = MCPServerInfo(
            server_id="github_server",
            server_type=MCPServerType.GITHUB,
            name="GitHub MCP Server",
            endpoint="npx -y @modelcontextprotocol/server-github",
            capabilities={"repository_access", "code_management", "version_control", "collaboration"}
        )
        await self.register_mcp_server(github_server)
        
        # Memory server
        memory_server = MCPServerInfo(
            server_id="memory_server",
            server_type=MCPServerType.MEMORY,
            name="Memory MCP Server",
            endpoint="npx -y @modelcontextprotocol/server-memory",
            capabilities={"knowledge_storage", "entity_management", "relationship_tracking", "memory_retrieval"}
        )
        await self.register_mcp_server(memory_server)
        
        logger.info("Registered existing MCP servers")
    
    async def create_sample_tasks(self) -> List[str]:
        """Create sample tasks for testing."""
        tasks = []
        
        # ToT reasoning task
        tot_task = TaskRequest(
            task_type="reasoning",
            priority=TaskPriority.HIGH,
            description="Solve complex problem using Tree of Thought reasoning",
            input_data={"problem": "How to optimize multi-agent coordination?"},
            required_capabilities={"reasoning", "problem_solving"},
            required_mcp_servers={MCPServerType.MEMORY, MCPServerType.FILESYSTEM}
        )
        
        if await self.submit_task(tot_task):
            tasks.append(tot_task.task_id)
        
        # RAG retrieval task
        rag_task = TaskRequest(
            task_type="document_retrieval",
            priority=TaskPriority.NORMAL,
            description="Retrieve relevant documents for query",
            input_data={"query": "multi-agent orchestration patterns"},
            required_capabilities={"document_retrieval", "knowledge_search"},
            required_mcp_servers={MCPServerType.FILESYSTEM, MCPServerType.POSTGRESQL}
        )
        
        if await self.submit_task(rag_task):
            tasks.append(rag_task.task_id)
        
        # Evaluation task
        eval_task = TaskRequest(
            task_type="quality_assessment",
            priority=TaskPriority.NORMAL,
            description="Evaluate system performance",
            input_data={"metrics": ["accuracy", "efficiency", "reliability"]},
            required_capabilities={"evaluation", "assessment"},
            required_mcp_servers={MCPServerType.MEMORY}
        )
        
        if await self.submit_task(eval_task):
            tasks.append(eval_task.task_id)
        
        logger.info(f"Created {len(tasks)} sample tasks")
        return tasks
    
    def add_status_callback(self, callback: Callable):
        """Add a status change callback."""
        self.status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable):
        """Remove a status callback."""
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
    
    async def export_configuration(self) -> str:
        """Export current configuration."""
        config_data = {
            "orchestration_config": {
                "evolution_enabled": self.config.evolution_enabled,
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "default_strategy": self.config.default_strategy.value,
                "batch_processing_enabled": self.config.batch_processing_enabled,
                "adaptive_threshold": self.config.adaptive_threshold
            },
            "system_status": await self.get_system_status(),
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        return json.dumps(config_data, indent=2, default=str)

    async def _integrate_phase4_components(self) -> None:
        """Integrate Phase 4 simulation environment and behavior detection"""
        try:
            logger.info("Integrating Phase 4 components with orchestration...")

            # Check if agent factory has Phase 4 components
            if hasattr(self.agent_registry, 'agent_factory'):
                agent_factory = self.agent_registry.agent_factory

                if hasattr(agent_factory, 'simulation_env') and agent_factory.simulation_env:
                    # Connect simulation environment to orchestration
                    self.simulation_env = agent_factory.simulation_env
                    logger.info("Connected to agent factory simulation environment")

                    # Enhance emergent behavior detector with simulation data
                    if hasattr(agent_factory, 'behavior_detector') and agent_factory.behavior_detector:
                        # Create bridge between orchestration and behavior detection
                        self.behavior_detector_bridge = agent_factory.behavior_detector
                        logger.info("Connected to emergent behavior detector")

                        # Set up periodic behavior monitoring
                        asyncio.create_task(self._monitor_emergent_behaviors())

            logger.info("Phase 4 integration completed successfully")

        except Exception as e:
            logger.error(f"Failed to integrate Phase 4 components: {e}")
            # Continue without Phase 4 integration

    async def _monitor_emergent_behaviors(self) -> None:
        """Monitor emergent behaviors from the simulation environment"""
        try:
            while self.is_running:
                if hasattr(self, 'behavior_detector_bridge'):
                    # Get behavior analysis from detector
                    behavior_summary = await self.behavior_detector_bridge.analyze_emergent_behaviors()

                    if behavior_summary.get('emergent_patterns'):
                        logger.info(f"Detected emergent behaviors: {behavior_summary['emergent_patterns']}")

                        # Integrate with orchestration decisions
                        await self._adapt_orchestration_to_behaviors(behavior_summary)

                # Check every 30 seconds
                await asyncio.sleep(30)

        except Exception as e:
            logger.error(f"Error in emergent behavior monitoring: {e}")

    async def _adapt_orchestration_to_behaviors(self, behavior_summary: Dict[str, Any]) -> None:
        """Adapt orchestration strategies based on emergent behaviors"""
        try:
            # Example adaptations based on emergent patterns
            patterns = behavior_summary.get('emergent_patterns', [])

            for pattern in patterns:
                if pattern.get('type') == 'resource_optimization':
                    # Adjust resource allocation strategies
                    logger.info("Adapting resource allocation based on emergent optimization patterns")

                elif pattern.get('type') == 'collaborative_efficiency':
                    # Enhance collaboration strategies
                    logger.info("Enhancing collaboration strategies based on emergent patterns")

                elif pattern.get('type') == 'adaptive_learning':
                    # Update learning parameters
                    logger.info("Updating learning parameters based on emergent adaptation")

        except Exception as e:
            logger.error(f"Error adapting orchestration to behaviors: {e}")