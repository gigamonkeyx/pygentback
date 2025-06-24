"""
Tool Selector

Intelligent selection of MCP tools based on requirements, context, and performance.
Uses machine learning and heuristics to choose optimal tools for specific tasks.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import random

from .mcp_analyzer import MCPServerProfile, MCPCapability, CapabilityType, PerformanceLevel

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Strategies for tool selection"""
    PERFORMANCE_FIRST = "performance_first"
    RELIABILITY_FIRST = "reliability_first"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    CAPABILITY_COVERAGE = "capability_coverage"


class SelectionPriority(Enum):
    """Priority levels for tool selection"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SelectionCriteria:
    """Criteria for tool selection"""
    required_capabilities: List[CapabilityType] = field(default_factory=list)
    preferred_capabilities: List[CapabilityType] = field(default_factory=list)
    
    # Performance requirements
    max_latency_ms: Optional[float] = None
    min_success_rate: Optional[float] = None
    min_reliability_score: Optional[float] = None
    
    # Resource constraints
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    
    # Selection preferences
    strategy: SelectionStrategy = SelectionStrategy.BALANCED
    priority: SelectionPriority = SelectionPriority.MEDIUM
    max_tools: int = 5
    allow_redundancy: bool = False
    
    # Context information
    task_description: str = ""
    expected_workload: str = "medium"  # light, medium, heavy
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    # Constraints
    excluded_servers: Set[str] = field(default_factory=set)
    required_servers: Set[str] = field(default_factory=set)
    operating_system: Optional[str] = None
    
    def add_required_capability(self, capability: CapabilityType):
        """Add a required capability"""
        if capability not in self.required_capabilities:
            self.required_capabilities.append(capability)
    
    def add_preferred_capability(self, capability: CapabilityType):
        """Add a preferred capability"""
        if capability not in self.preferred_capabilities:
            self.preferred_capabilities.append(capability)


@dataclass
class ToolSelection:
    """Result of tool selection process"""
    selected_tools: List[str] = field(default_factory=list)
    selected_servers: List[str] = field(default_factory=list)
    server_profiles: List[MCPServerProfile] = field(default_factory=list)
    
    # Selection metrics
    total_score: float = 0.0
    coverage_score: float = 0.0
    performance_score: float = 0.0
    reliability_score: float = 0.0
    cost_score: float = 0.0
    
    # Selection details
    selection_rationale: List[str] = field(default_factory=list)
    unmet_requirements: List[CapabilityType] = field(default_factory=list)
    alternative_options: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution plan
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    
    def get_tools_for_capability(self, capability: CapabilityType) -> List[str]:
        """Get selected tools that support a specific capability"""
        matching_tools = []
        
        for profile in self.server_profiles:
            if profile.has_capability(capability):
                capability_tools = profile.get_tools_for_capability(capability)
                for tool in capability_tools:
                    if tool in self.selected_tools:
                        matching_tools.append(tool)
        
        return matching_tools
    
    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get server that provides a specific tool"""
        for profile in self.server_profiles:
            if tool_name in profile.available_tools:
                return profile.server_name
        return None


class ToolSelector:
    """
    Intelligent tool selector for MCP servers.
    
    Analyzes requirements and selects optimal tools based on various criteria
    including performance, reliability, capability coverage, and resource constraints.
    """
    
    def __init__(self):
        # Selection weights for different strategies
        self.strategy_weights = {
            SelectionStrategy.PERFORMANCE_FIRST: {
                'performance': 0.5,
                'reliability': 0.2,
                'coverage': 0.2,
                'cost': 0.1
            },
            SelectionStrategy.RELIABILITY_FIRST: {
                'performance': 0.2,
                'reliability': 0.5,
                'coverage': 0.2,
                'cost': 0.1
            },
            SelectionStrategy.BALANCED: {
                'performance': 0.25,
                'reliability': 0.25,
                'coverage': 0.25,
                'cost': 0.25
            },
            SelectionStrategy.COST_OPTIMIZED: {
                'performance': 0.1,
                'reliability': 0.2,
                'coverage': 0.2,
                'cost': 0.5
            },
            SelectionStrategy.LATENCY_OPTIMIZED: {
                'performance': 0.6,
                'reliability': 0.2,
                'coverage': 0.1,
                'cost': 0.1
            },
            SelectionStrategy.CAPABILITY_COVERAGE: {
                'performance': 0.1,
                'reliability': 0.2,
                'coverage': 0.6,
                'cost': 0.1
            }
        }
        
        # Performance level scores
        self.performance_scores = {
            PerformanceLevel.EXCELLENT: 1.0,
            PerformanceLevel.GOOD: 0.8,
            PerformanceLevel.AVERAGE: 0.6,
            PerformanceLevel.POOR: 0.3,
            PerformanceLevel.UNKNOWN: 0.5
        }
        
        # Selection history for learning
        self.selection_history = []
        self.feedback_data = []
    
    async def select_tools(self, 
                          criteria: SelectionCriteria,
                          available_servers: List[MCPServerProfile]) -> ToolSelection:
        """
        Select optimal tools based on criteria and available servers.
        
        Args:
            criteria: Selection criteria and requirements
            available_servers: List of available MCP server profiles
            
        Returns:
            Tool selection result with selected tools and rationale
        """
        try:
            logger.info(f"Selecting tools with strategy: {criteria.strategy.value}")
            
            # Filter servers based on constraints
            candidate_servers = self._filter_servers(available_servers, criteria)
            
            if not candidate_servers:
                logger.warning("No servers match the selection criteria")
                return ToolSelection(
                    unmet_requirements=criteria.required_capabilities,
                    selection_rationale=["No servers match the selection criteria"]
                )
            
            # Score servers for each required capability
            capability_scores = await self._score_servers_for_capabilities(
                candidate_servers, criteria
            )
            
            # Select optimal combination of tools
            selection = await self._select_optimal_combination(
                candidate_servers, capability_scores, criteria
            )
            
            # Generate execution plan
            self._generate_execution_plan(selection, criteria)
            
            # Record selection for learning
            self._record_selection(criteria, selection)
            
            logger.info(f"Selected {len(selection.selected_tools)} tools from "
                       f"{len(selection.selected_servers)} servers")
            
            return selection
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return ToolSelection(
                selection_rationale=[f"Selection failed: {e}"],
                unmet_requirements=criteria.required_capabilities
            )
    
    def _filter_servers(self, servers: List[MCPServerProfile], 
                       criteria: SelectionCriteria) -> List[MCPServerProfile]:
        """Filter servers based on constraints"""
        filtered_servers = []
        
        for server in servers:
            # Check excluded servers
            if server.server_name in criteria.excluded_servers:
                continue
            
            # Check operating system compatibility
            if (criteria.operating_system and 
                criteria.operating_system not in server.operating_systems):
                continue
            
            # Check resource constraints
            if (criteria.max_memory_mb and 
                server.memory_usage_mb > criteria.max_memory_mb):
                continue
            
            if (criteria.max_cpu_percent and 
                server.cpu_usage_percent > criteria.max_cpu_percent):
                continue
            
            # Check performance requirements
            if (criteria.max_latency_ms and 
                server.average_latency_ms > criteria.max_latency_ms):
                continue
            
            if (criteria.min_success_rate and 
                server.error_rate > (1.0 - criteria.min_success_rate)):
                continue
            
            filtered_servers.append(server)
        
        # Always include required servers if they exist
        for server in servers:
            if (server.server_name in criteria.required_servers and 
                server not in filtered_servers):
                filtered_servers.append(server)
        
        return filtered_servers
    
    async def _score_servers_for_capabilities(self, 
                                            servers: List[MCPServerProfile],
                                            criteria: SelectionCriteria) -> Dict[CapabilityType, List[Tuple[MCPServerProfile, float]]]:
        """Score servers for each required capability"""
        capability_scores = {}
        
        all_capabilities = criteria.required_capabilities + criteria.preferred_capabilities
        
        for capability_type in all_capabilities:
            scores = []
            
            for server in servers:
                score = self._calculate_server_capability_score(
                    server, capability_type, criteria
                )
                if score > 0:  # Only include servers that have the capability
                    scores.append((server, score))
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            capability_scores[capability_type] = scores
        
        return capability_scores
    
    def _calculate_server_capability_score(self, 
                                         server: MCPServerProfile,
                                         capability_type: CapabilityType,
                                         criteria: SelectionCriteria) -> float:
        """Calculate score for a server's capability"""
        capability = server.get_capability_by_type(capability_type)
        
        if not capability:
            return 0.0
        
        # Base score from performance level
        performance_score = self.performance_scores[capability.performance_level]
        
        # Reliability score
        reliability_score = capability.reliability_score
        
        # Tool count score (more tools = better coverage)
        tool_count_score = min(1.0, len(capability.tools) / 5.0)
        
        # Latency score
        latency_score = 1.0
        if capability.latency_ms > 0:
            if criteria.max_latency_ms:
                latency_score = max(0.0, 1.0 - (capability.latency_ms / criteria.max_latency_ms))
            else:
                # General latency scoring
                latency_score = max(0.0, 1.0 - (capability.latency_ms / 5000.0))
        
        # Success rate score
        success_rate_score = capability.success_rate
        
        # Resource efficiency score
        resource_score = 1.0
        if server.memory_usage_mb > 0:
            resource_score *= max(0.1, 1.0 - (server.memory_usage_mb / 2048.0))
        if server.cpu_usage_percent > 0:
            resource_score *= max(0.1, 1.0 - (server.cpu_usage_percent / 100.0))
        
        # Complexity support score
        complexity_scores = {"basic": 0.5, "intermediate": 0.75, "advanced": 1.0}
        complexity_score = complexity_scores.get(capability.complexity_support, 0.5)
        
        # Combine scores based on strategy
        weights = self.strategy_weights[criteria.strategy]
        
        total_score = (
            weights['performance'] * (performance_score * 0.4 + latency_score * 0.6) +
            weights['reliability'] * (reliability_score * 0.6 + success_rate_score * 0.4) +
            weights['coverage'] * (tool_count_score * 0.6 + complexity_score * 0.4) +
            weights['cost'] * resource_score
        )
        
        # Boost score for required servers
        if server.server_name in criteria.required_servers:
            total_score *= 1.5
        
        return total_score
    
    async def _select_optimal_combination(self, 
                                        servers: List[MCPServerProfile],
                                        capability_scores: Dict[CapabilityType, List[Tuple[MCPServerProfile, float]]],
                                        criteria: SelectionCriteria) -> ToolSelection:
        """Select optimal combination of tools"""
        selection = ToolSelection()
        selected_server_names = set()
        
        # First, ensure all required capabilities are covered
        for capability_type in criteria.required_capabilities:
            if capability_type not in capability_scores:
                selection.unmet_requirements.append(capability_type)
                continue
            
            scores = capability_scores[capability_type]
            if not scores:
                selection.unmet_requirements.append(capability_type)
                continue
            
            # Select best server for this capability
            best_server, best_score = scores[0]
            
            if best_server.server_name not in selected_server_names or criteria.allow_redundancy:
                self._add_server_to_selection(selection, best_server, capability_type)
                selected_server_names.add(best_server.server_name)
                
                selection.selection_rationale.append(
                    f"Selected {best_server.server_name} for {capability_type.value} "
                    f"(score: {best_score:.3f})"
                )
        
        # Add preferred capabilities if space allows
        for capability_type in criteria.preferred_capabilities:
            if len(selection.selected_servers) >= criteria.max_tools:
                break
            
            if capability_type not in capability_scores:
                continue
            
            scores = capability_scores[capability_type]
            for server, score in scores:
                if (server.server_name not in selected_server_names and 
                    len(selection.selected_servers) < criteria.max_tools):
                    
                    self._add_server_to_selection(selection, server, capability_type)
                    selected_server_names.add(server.server_name)
                    
                    selection.selection_rationale.append(
                        f"Added {server.server_name} for preferred {capability_type.value} "
                        f"(score: {score:.3f})"
                    )
                    break
        
        # Calculate overall scores
        self._calculate_selection_scores(selection, criteria)
        
        return selection
    
    def _add_server_to_selection(self, selection: ToolSelection, 
                               server: MCPServerProfile, 
                               capability_type: CapabilityType):
        """Add server and its relevant tools to selection"""
        if server.server_name not in selection.selected_servers:
            selection.selected_servers.append(server.server_name)
            selection.server_profiles.append(server)
        
        # Add tools for the specific capability
        capability_tools = server.get_tools_for_capability(capability_type)
        for tool in capability_tools:
            if tool not in selection.selected_tools:
                selection.selected_tools.append(tool)
    
    def _calculate_selection_scores(self, selection: ToolSelection, criteria: SelectionCriteria):
        """Calculate overall scores for the selection"""
        if not selection.server_profiles:
            return
        
        # Coverage score
        required_count = len(criteria.required_capabilities)
        covered_count = required_count - len(selection.unmet_requirements)
        selection.coverage_score = covered_count / max(required_count, 1)
        
        # Performance score
        performance_scores = []
        for server in selection.server_profiles:
            perf_score = self.performance_scores[server.overall_performance]
            performance_scores.append(perf_score)
        selection.performance_score = np.mean(performance_scores)
        
        # Reliability score
        reliability_scores = []
        for server in selection.server_profiles:
            # Average reliability across capabilities
            if server.capabilities:
                avg_reliability = np.mean([cap.reliability_score for cap in server.capabilities])
                reliability_scores.append(avg_reliability)
        selection.reliability_score = np.mean(reliability_scores) if reliability_scores else 0.0
        
        # Cost score (inverse of resource usage)
        cost_scores = []
        for server in selection.server_profiles:
            memory_cost = 1.0 - min(1.0, server.memory_usage_mb / 2048.0)
            cpu_cost = 1.0 - min(1.0, server.cpu_usage_percent / 100.0)
            cost_scores.append((memory_cost + cpu_cost) / 2.0)
        selection.cost_score = np.mean(cost_scores) if cost_scores else 1.0
        
        # Total score
        weights = self.strategy_weights[criteria.strategy]
        selection.total_score = (
            weights['performance'] * selection.performance_score +
            weights['reliability'] * selection.reliability_score +
            weights['coverage'] * selection.coverage_score +
            weights['cost'] * selection.cost_score
        )
    
    def _generate_execution_plan(self, selection: ToolSelection, criteria: SelectionCriteria):
        """Generate execution plan for selected tools"""
        # Simple execution order based on dependencies and performance
        tools_by_server = {}
        
        for tool in selection.selected_tools:
            server = selection.get_server_for_tool(tool)
            if server:
                if server not in tools_by_server:
                    tools_by_server[server] = []
                tools_by_server[server].append(tool)
        
        # Create execution order
        # For now, simple sequential order by server performance
        server_performance = {}
        for server_profile in selection.server_profiles:
            server_performance[server_profile.server_name] = self.performance_scores[
                server_profile.overall_performance
            ]
        
        sorted_servers = sorted(tools_by_server.keys(), 
                              key=lambda s: server_performance.get(s, 0), 
                              reverse=True)
        
        for server in sorted_servers:
            selection.execution_order.extend(tools_by_server[server])
        
        # Create parallel groups (tools from same server can run in parallel)
        for server, tools in tools_by_server.items():
            if len(tools) > 1:
                selection.parallel_groups.append(tools)
    
    def _record_selection(self, criteria: SelectionCriteria, selection: ToolSelection):
        """Record selection for learning and analysis"""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'criteria': {
                'strategy': criteria.strategy.value,
                'required_capabilities': [cap.value for cap in criteria.required_capabilities],
                'max_tools': criteria.max_tools
            },
            'selection': {
                'total_score': selection.total_score,
                'coverage_score': selection.coverage_score,
                'performance_score': selection.performance_score,
                'selected_tools_count': len(selection.selected_tools),
                'selected_servers_count': len(selection.selected_servers)
            }
        }
        
        self.selection_history.append(record)
        
        # Keep limited history
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]
    
    def add_feedback(self, selection: ToolSelection, 
                    success: bool, performance_rating: float, 
                    feedback_notes: str = ""):
        """Add feedback about a selection's performance"""
        feedback = {
            'timestamp': datetime.utcnow().isoformat(),
            'selection_id': id(selection),
            'success': success,
            'performance_rating': performance_rating,
            'feedback_notes': feedback_notes,
            'selected_tools': selection.selected_tools.copy(),
            'selected_servers': selection.selected_servers.copy()
        }
        
        self.feedback_data.append(feedback)
        
        # Keep limited feedback
        if len(self.feedback_data) > 1000:
            self.feedback_data = self.feedback_data[-1000:]
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about tool selections"""
        if not self.selection_history:
            return {"error": "No selection history available"}
        
        # Strategy usage
        strategy_counts = {}
        for record in self.selection_history:
            strategy = record['criteria']['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Average scores
        total_scores = [record['selection']['total_score'] for record in self.selection_history]
        coverage_scores = [record['selection']['coverage_score'] for record in self.selection_history]
        
        # Feedback analysis
        feedback_stats = {}
        if self.feedback_data:
            success_rate = sum(1 for f in self.feedback_data if f['success']) / len(self.feedback_data)
            avg_rating = np.mean([f['performance_rating'] for f in self.feedback_data])
            feedback_stats = {
                'success_rate': success_rate,
                'average_rating': avg_rating,
                'total_feedback': len(self.feedback_data)
            }
        
        return {
            'total_selections': len(self.selection_history),
            'strategy_distribution': strategy_counts,
            'average_total_score': np.mean(total_scores),
            'average_coverage_score': np.mean(coverage_scores),
            'feedback_statistics': feedback_stats
        }
