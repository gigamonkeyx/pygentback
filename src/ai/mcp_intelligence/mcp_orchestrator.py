"""
MCP Orchestrator

High-level orchestration system that coordinates all MCP intelligence components
for comprehensive server management, tool selection, and request routing.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

from .mcp_analyzer import MCPAnalyzer, MCPServerProfile, CapabilityType
from .tool_selector import ToolSelector, ToolSelection, SelectionCriteria
from .capability_matcher import CapabilityMatcher, MatchResult
from .server_monitor import ServerMonitor, ServerHealth, PerformanceMetrics
from .intelligent_router import IntelligentRouter, RoutingRequest, RoutingDecision

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationRequest:
    """High-level orchestration request"""
    request_id: str
    task_description: str
    required_capabilities: List[CapabilityType] = field(default_factory=list)
    preferred_capabilities: List[CapabilityType] = field(default_factory=list)
    
    # Performance requirements
    max_latency_ms: Optional[float] = None
    min_success_rate: Optional[float] = None
    max_cost: Optional[float] = None
    
    # Execution preferences
    allow_parallel_execution: bool = True
    require_consistency: bool = False
    max_retries: int = 3
    
    # Context
    user_context: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    """Execution plan for orchestrated request"""
    request_id: str
    selected_tools: List[str] = field(default_factory=list)
    selected_servers: List[str] = field(default_factory=list)
    
    # Execution strategy
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    fallback_options: List[str] = field(default_factory=list)
    
    # Performance estimates
    estimated_total_time_ms: float = 0.0
    estimated_success_probability: float = 0.0
    estimated_cost: float = 0.0
    
    # Plan metadata
    plan_confidence: float = 0.0
    plan_rationale: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionResult:
    """Result of orchestrated execution"""
    request_id: str
    success: bool
    
    # Execution details
    executed_tools: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    actual_cost: float = 0.0
    
    # Results
    tool_results: Dict[str, Any] = field(default_factory=dict)
    aggregated_result: Any = None
    
    # Performance metrics
    server_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error_details: List[str] = field(default_factory=list)
    
    # Metadata
    completed_at: datetime = field(default_factory=datetime.utcnow)


class MCPOrchestrator:
    """
    Main orchestration system for MCP intelligence.
    
    Coordinates all intelligence components to provide comprehensive
    MCP server management, intelligent tool selection, and optimized execution.
    """
    
    def __init__(self):
        # Initialize core components
        self.analyzer = MCPAnalyzer()
        self.tool_selector = ToolSelector()
        self.capability_matcher = CapabilityMatcher()
        self.server_monitor = ServerMonitor()
        self.intelligent_router = IntelligentRouter(
            self.server_monitor, self.capability_matcher, self.tool_selector
        )
        
        # Orchestration state
        self.registered_servers: Dict[str, MCPServerProfile] = {}
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # Orchestration history
        self.orchestration_history: List[Dict[str, Any]] = []
        self.performance_analytics: Dict[str, Any] = {}
        
        # Callbacks
        self.execution_callbacks: List[Callable] = []
        
        # Configuration
        self.config = {
            'max_concurrent_executions': 10,
            'default_timeout_ms': 30000,
            'enable_monitoring': True,
            'enable_analytics': True
        }
    
    async def initialize(self):
        """Initialize the orchestration system"""
        logger.info("Initializing MCP Orchestrator")
        
        # Start server monitoring if enabled
        if self.config['enable_monitoring']:
            asyncio.create_task(self.server_monitor.start_monitoring())
        
        logger.info("MCP Orchestrator initialized successfully")
    
    async def shutdown(self):
        """Shutdown the orchestration system"""
        logger.info("Shutting down MCP Orchestrator")
        
        # Stop monitoring
        self.server_monitor.stop_monitoring()
        
        # Cancel active executions
        for request_id in list(self.active_executions.keys()):
            await self.cancel_execution(request_id)
        
        logger.info("MCP Orchestrator shutdown complete")
    
    async def register_server(self, server_name: str, server_info: Dict[str, Any]) -> MCPServerProfile:
        """
        Register and analyze an MCP server.
        
        Args:
            server_name: Name of the server
            server_info: Server information and metadata
            
        Returns:
            Analyzed server profile
        """
        try:
            logger.info(f"Registering MCP server: {server_name}")
            
            # Analyze server capabilities
            profile = await self.analyzer.analyze_server(server_name, server_info)
            
            # Store profile
            self.registered_servers[server_name] = profile
            
            # Register with router
            self.intelligent_router.register_server(profile)
            
            # Add to monitoring
            if self.config['enable_monitoring']:
                self.server_monitor.add_server(server_name, server_info)
            
            logger.info(f"Successfully registered server {server_name} with "
                       f"{len(profile.capabilities)} capabilities")
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to register server {server_name}: {e}")
            raise
    
    async def unregister_server(self, server_name: str):
        """Unregister an MCP server"""
        try:
            if server_name in self.registered_servers:
                del self.registered_servers[server_name]
                
                # Remove from router
                self.intelligent_router.unregister_server(server_name)
                
                # Remove from monitoring
                self.server_monitor.remove_server(server_name)
                
                logger.info(f"Unregistered server: {server_name}")
            
        except Exception as e:
            logger.error(f"Failed to unregister server {server_name}: {e}")
    
    async def orchestrate_request(self, request: OrchestrationRequest) -> OrchestrationPlan:
        """
        Create orchestration plan for a request.
        
        Args:
            request: Orchestration request with requirements
            
        Returns:
            Detailed execution plan
        """
        try:
            logger.info(f"Creating orchestration plan for request {request.request_id}")
            
            # Create selection criteria
            criteria = SelectionCriteria(
                required_capabilities=request.required_capabilities,
                preferred_capabilities=request.preferred_capabilities,
                max_latency_ms=request.max_latency_ms,
                min_success_rate=request.min_success_rate,
                task_description=request.task_description,
                execution_context=request.execution_context
            )
            
            # Select optimal tools
            tool_selection = await self.tool_selector.select_tools(
                criteria, list(self.registered_servers.values())
            )
            
            # Create orchestration plan
            plan = OrchestrationPlan(
                request_id=request.request_id,
                selected_tools=tool_selection.selected_tools,
                selected_servers=tool_selection.selected_servers,
                execution_order=tool_selection.execution_order,
                parallel_groups=tool_selection.parallel_groups
            )
            
            # Calculate performance estimates
            await self._calculate_performance_estimates(plan, request)
            
            # Generate plan rationale
            plan.plan_rationale = self._generate_plan_rationale(tool_selection, request)
            
            # Calculate plan confidence
            plan.plan_confidence = self._calculate_plan_confidence(tool_selection, request)
            
            logger.info(f"Created orchestration plan for {request.request_id}: "
                       f"{len(plan.selected_tools)} tools, "
                       f"confidence: {plan.plan_confidence:.3f}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create orchestration plan for {request.request_id}: {e}")
            raise
    
    async def execute_plan(self, plan: OrchestrationPlan, 
                          request: OrchestrationRequest) -> ExecutionResult:
        """
        Execute an orchestration plan.
        
        Args:
            plan: Orchestration plan to execute
            request: Original request
            
        Returns:
            Execution result
        """
        try:
            logger.info(f"Executing orchestration plan for {request.request_id}")
            
            # Track execution
            execution_start = datetime.utcnow()
            self.active_executions[request.request_id] = {
                'plan': plan,
                'request': request,
                'start_time': execution_start,
                'status': 'executing'
            }
            
            # Execute tools
            result = await self._execute_tools(plan, request)
            
            # Update execution tracking
            execution_end = datetime.utcnow()
            result.execution_time_ms = (execution_end - execution_start).total_seconds() * 1000
            
            # Record execution
            self._record_execution(request, plan, result)
            
            # Clean up
            if request.request_id in self.active_executions:
                del self.active_executions[request.request_id]
            
            # Call execution callbacks
            await self._call_execution_callbacks(request, plan, result)
            
            logger.info(f"Completed execution for {request.request_id}: "
                       f"success={result.success}, time={result.execution_time_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Execution failed for {request.request_id}: {e}")
            
            # Create failed result
            result = ExecutionResult(
                request_id=request.request_id,
                success=False,
                error_details=[str(e)]
            )
            
            # Clean up
            if request.request_id in self.active_executions:
                del self.active_executions[request.request_id]
            
            return result
    
    async def _calculate_performance_estimates(self, plan: OrchestrationPlan, 
                                             request: OrchestrationRequest):
        """Calculate performance estimates for the plan"""
        total_time = 0.0
        total_success_prob = 1.0
        total_cost = 0.0
        
        for server_name in plan.selected_servers:
            # Get server metrics
            metrics = self.server_monitor.get_server_metrics(server_name)
            health = self.server_monitor.get_server_health(server_name)
            
            if metrics:
                # Add estimated execution time
                if plan.parallel_groups:
                    # For parallel execution, use maximum time in group
                    max_group_time = 0.0
                    for group in plan.parallel_groups:
                        group_time = len(group) * metrics.avg_response_time_ms
                        max_group_time = max(max_group_time, group_time)
                    total_time += max_group_time
                else:
                    # Sequential execution
                    server_tools = [t for t in plan.selected_tools 
                                  if self._get_server_for_tool(t) == server_name]
                    total_time += len(server_tools) * metrics.avg_response_time_ms
                
                # Calculate success probability
                server_success_rate = metrics.calculate_success_rate()
                total_success_prob *= server_success_rate
            
            # Estimate cost (simplified)
            if health:
                # Higher performance servers cost more
                server_cost = health.overall_score * 10.0  # Arbitrary cost unit
                total_cost += server_cost
        
        plan.estimated_total_time_ms = total_time
        plan.estimated_success_probability = total_success_prob
        plan.estimated_cost = total_cost
    
    def _generate_plan_rationale(self, tool_selection: ToolSelection, 
                               request: OrchestrationRequest) -> List[str]:
        """Generate rationale for the orchestration plan"""
        rationale = []
        
        # Tool selection rationale
        rationale.extend(tool_selection.selection_rationale)
        
        # Coverage analysis
        if tool_selection.unmet_requirements:
            unmet = [cap.value for cap in tool_selection.unmet_requirements]
            rationale.append(f"Unmet requirements: {', '.join(unmet)}")
        
        # Performance considerations
        if request.max_latency_ms:
            rationale.append(f"Optimized for latency constraint: {request.max_latency_ms}ms")
        
        if request.min_success_rate:
            rationale.append(f"Optimized for success rate: {request.min_success_rate:.1%}")
        
        return rationale
    
    def _calculate_plan_confidence(self, tool_selection: ToolSelection, 
                                 request: OrchestrationRequest) -> float:
        """Calculate confidence in the orchestration plan"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on coverage
        if not tool_selection.unmet_requirements:
            confidence += 0.3
        
        # Increase confidence based on tool selection quality
        confidence += tool_selection.total_score * 0.3
        
        # Increase confidence based on server health
        healthy_servers = 0
        for server_name in tool_selection.selected_servers:
            health = self.server_monitor.get_server_health(server_name)
            if health and health.overall_score > 0.8:
                healthy_servers += 1
        
        if tool_selection.selected_servers:
            health_ratio = healthy_servers / len(tool_selection.selected_servers)
            confidence += health_ratio * 0.2
        
        return min(1.0, confidence)
    
    async def _execute_tools(self, plan: OrchestrationPlan,
                           request: OrchestrationRequest) -> ExecutionResult:
        """Execute the tools in the plan using real MCP service calls"""
        result = ExecutionResult(request_id=request.request_id)

        try:
            # Execute tools using real MCP service calls
            for tool_name in plan.selected_tools:
                start_time = datetime.utcnow()

                # Get the server that provides this tool
                server_name = self._get_server_for_tool(tool_name)
                if not server_name:
                    raise RuntimeError(f"No server found for tool {tool_name}")

                server_profile = self.registered_servers.get(server_name)
                if not server_profile:
                    raise RuntimeError(f"Server profile not found for {server_name}")

                # Execute real MCP tool call
                try:
                    tool_result = await self._execute_real_mcp_tool(
                        server_profile, tool_name, request.parameters
                    )

                    execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                    # Record successful execution
                    result.executed_tools.append(tool_name)
                    result.tool_results[tool_name] = {
                        'status': 'success',
                        'result': tool_result,
                        'execution_time_ms': execution_time,
                        'server': server_name
                    }

                except Exception as tool_error:
                    execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                    # Record failed execution
                    result.tool_results[tool_name] = {
                        'status': 'error',
                        'error': str(tool_error),
                        'execution_time_ms': execution_time,
                        'server': server_name
                    }
                    result.error_details.append(f"Tool {tool_name} failed: {tool_error}")

            # Aggregate results
            successful_tools = [name for name, res in result.tool_results.items()
                              if res['status'] == 'success']

            result.aggregated_result = {
                'total_tools_executed': len(result.tool_results),
                'successful_tools': len(successful_tools),
                'failed_tools': len(result.tool_results) - len(successful_tools),
                'overall_status': 'success' if successful_tools else 'failed',
                'execution_summary': {
                    tool_name: res['result'] if res['status'] == 'success' else res['error']
                    for tool_name, res in result.tool_results.items()
                }
            }

            result.success = len(successful_tools) > 0

        except Exception as e:
            result.success = False
            result.error_details.append(f"Tool execution failed: {str(e)}")

        return result

    async def _execute_real_mcp_tool(self, server_profile: 'MCPServerProfile',
                                   tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a real MCP tool call"""
        try:
            # Try to get MCP client for the server
            from ..mcp_client import get_mcp_client

            client = await get_mcp_client(server_profile.server_name)
            if not client:
                raise RuntimeError(f"MCP client not available for server {server_profile.server_name}")

            # Execute the tool
            result = await client.call_tool(tool_name, parameters)

            if result.get('error'):
                raise RuntimeError(f"MCP tool error: {result['error']}")

            return result.get('content', result)

        except ImportError:
            # If MCP client not available, try direct HTTP call
            return await self._execute_mcp_tool_http(server_profile, tool_name, parameters)
        except Exception as e:
            logger.error(f"Real MCP tool execution failed: {e}")
            raise

    async def _execute_mcp_tool_http(self, server_profile: 'MCPServerProfile',
                                   tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute MCP tool via HTTP call"""
        try:
            import aiohttp

            # Construct MCP tool call URL
            base_url = getattr(server_profile, 'base_url', f"http://localhost:8000")
            tool_url = f"{base_url}/tools/{tool_name}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    tool_url,
                    json=parameters,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"HTTP {response.status}: {error_text}")

                    result = await response.json()
                    return result

        except Exception as e:
            logger.error(f"HTTP MCP tool execution failed: {e}")
            raise RuntimeError(f"Real MCP tool execution required but failed: {e}")
    
    def _get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server that provides a specific tool"""
        for server_name, profile in self.registered_servers.items():
            if tool_name in profile.available_tools:
                return server_name
        return None
    
    def _record_execution(self, request: OrchestrationRequest, 
                         plan: OrchestrationPlan, result: ExecutionResult):
        """Record execution for analytics"""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request.request_id,
            'task_description': request.task_description,
            'required_capabilities': [cap.value for cap in request.required_capabilities],
            'selected_tools': plan.selected_tools,
            'selected_servers': plan.selected_servers,
            'success': result.success,
            'execution_time_ms': result.execution_time_ms,
            'plan_confidence': plan.plan_confidence,
            'estimated_time_ms': plan.estimated_total_time_ms,
            'estimated_success_prob': plan.estimated_success_probability
        }
        
        self.orchestration_history.append(record)
        
        # Keep limited history
        if len(self.orchestration_history) > 10000:
            self.orchestration_history = self.orchestration_history[-10000:]
    
    async def _call_execution_callbacks(self, request: OrchestrationRequest,
                                      plan: OrchestrationPlan, result: ExecutionResult):
        """Call execution callbacks"""
        for callback in self.execution_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(request, plan, result)
                else:
                    callback(request, plan, result)
            except Exception as e:
                logger.error(f"Execution callback failed: {e}")
    
    async def cancel_execution(self, request_id: str):
        """Cancel an active execution"""
        if request_id in self.active_executions:
            execution_info = self.active_executions[request_id]
            execution_info['status'] = 'cancelled'

            # Cancel actual tool executions if they support cancellation
            try:
                # Send cancellation signal to MCP servers
                for server_name in execution_info.get('servers', []):
                    server_profile = self.registered_servers.get(server_name)
                    if server_profile:
                        await self._cancel_server_execution(server_profile, request_id)
            except Exception as e:
                logger.warning(f"Failed to cancel server executions: {e}")

            del self.active_executions[request_id]
            logger.info(f"Cancelled execution for request {request_id}")

    async def _cancel_server_execution(self, server_profile: 'MCPServerProfile', request_id: str):
        """Cancel execution on a specific MCP server"""
        try:
            # Try to send cancellation request to MCP server
            import aiohttp

            base_url = getattr(server_profile, 'base_url', f"http://localhost:8000")
            cancel_url = f"{base_url}/cancel/{request_id}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    cancel_url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully cancelled execution on {server_profile.server_name}")
                    else:
                        logger.warning(f"Server {server_profile.server_name} returned {response.status} for cancellation")

        except Exception as e:
            logger.warning(f"Failed to cancel execution on {server_profile.server_name}: {e}")
    
    def add_execution_callback(self, callback: Callable):
        """Add callback for execution events"""
        self.execution_callbacks.append(callback)
    
    def get_server_profiles(self) -> List[MCPServerProfile]:
        """Get all registered server profiles"""
        return list(self.registered_servers.values())
    
    def get_server_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all servers"""
        summary = {
            'total_servers': len(self.registered_servers),
            'healthy_servers': 0,
            'warning_servers': 0,
            'critical_servers': 0,
            'offline_servers': 0
        }
        
        for server_name in self.registered_servers:
            health = self.server_monitor.get_server_health(server_name)
            if health:
                if health.overall_score >= 0.8:
                    summary['healthy_servers'] += 1
                elif health.overall_score >= 0.6:
                    summary['warning_servers'] += 1
                elif health.overall_score >= 0.3:
                    summary['critical_servers'] += 1
                else:
                    summary['offline_servers'] += 1
        
        return summary
    
    def get_orchestration_analytics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration analytics"""
        if not self.orchestration_history:
            return {"error": "No orchestration history available"}
        
        # Success rate analysis
        total_executions = len(self.orchestration_history)
        successful_executions = sum(1 for record in self.orchestration_history if record['success'])
        success_rate = successful_executions / total_executions
        
        # Performance analysis
        execution_times = [record['execution_time_ms'] for record in self.orchestration_history]
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        # Capability analysis
        all_capabilities = []
        for record in self.orchestration_history:
            all_capabilities.extend(record['required_capabilities'])
        
        capability_counts = {}
        for capability in all_capabilities:
            capability_counts[capability] = capability_counts.get(capability, 0) + 1
        
        # Server utilization
        all_servers = []
        for record in self.orchestration_history:
            all_servers.extend(record['selected_servers'])
        
        server_counts = {}
        for server in all_servers:
            server_counts[server] = server_counts.get(server, 0) + 1
        
        return {
            'total_executions': total_executions,
            'success_rate': success_rate,
            'avg_execution_time_ms': avg_execution_time,
            'capability_usage': capability_counts,
            'server_utilization': server_counts,
            'active_executions': len(self.active_executions),
            'registered_servers': len(self.registered_servers)
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export orchestrator configuration"""
        return {
            'config': self.config.copy(),
            'registered_servers': [
                {
                    'name': profile.server_name,
                    'capabilities': [cap.capability_type.value for cap in profile.capabilities],
                    'tools_count': len(profile.available_tools)
                }
                for profile in self.registered_servers.values()
            ],
            'component_status': {
                'analyzer': 'active',
                'tool_selector': 'active',
                'capability_matcher': 'active',
                'server_monitor': 'active' if self.server_monitor.is_monitoring else 'inactive',
                'intelligent_router': 'active'
            }
        }
