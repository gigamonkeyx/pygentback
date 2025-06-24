"""
Intelligent Router

Advanced routing system for MCP requests based on server capabilities,
performance, load balancing, and intelligent decision making.
"""

import logging
import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

from .mcp_analyzer import MCPServerProfile, CapabilityType
from .server_monitor import ServerMonitor, ServerHealth, PerformanceMetrics, HealthStatus
from .capability_matcher import CapabilityMatcher, MatchResult
from .tool_selector import ToolSelector, SelectionCriteria

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    FASTEST_RESPONSE = "fastest_response"
    HEALTH_BASED = "health_based"
    CAPABILITY_OPTIMIZED = "capability_optimized"
    LOAD_BALANCED = "load_balanced"
    INTELLIGENT = "intelligent"


class RoutingPriority(Enum):
    """Request priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RoutingRequest:
    """Request for routing"""
    request_id: str
    required_capabilities: List[CapabilityType] = field(default_factory=list)
    preferred_servers: List[str] = field(default_factory=list)
    excluded_servers: List[str] = field(default_factory=list)
    
    # Request characteristics
    priority: RoutingPriority = RoutingPriority.NORMAL
    expected_duration_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    requires_consistency: bool = False
    
    # Context information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_context: Dict[str, Any] = field(default_factory=dict)
    
    # Routing preferences
    strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT
    allow_fallback: bool = True
    max_retries: int = 3


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_server: str
    backup_servers: List[str] = field(default_factory=list)
    routing_score: float = 0.0
    
    # Decision rationale
    decision_factors: Dict[str, float] = field(default_factory=dict)
    routing_strategy_used: RoutingStrategy = RoutingStrategy.INTELLIGENT
    decision_confidence: float = 0.0
    
    # Execution plan
    estimated_latency_ms: float = 0.0
    estimated_success_probability: float = 0.0
    fallback_plan: List[str] = field(default_factory=list)
    
    # Metadata
    decision_timestamp: datetime = field(default_factory=datetime.utcnow)
    decision_time_ms: float = 0.0


@dataclass
class RoutingMetrics:
    """Metrics for routing performance"""
    total_requests: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    fallback_routes: int = 0
    
    # Timing metrics
    avg_decision_time_ms: float = 0.0
    avg_request_latency_ms: float = 0.0
    
    # Strategy effectiveness
    strategy_success_rates: Dict[str, float] = field(default_factory=dict)
    server_utilization: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    avg_routing_score: float = 0.0
    avg_decision_confidence: float = 0.0


class IntelligentRouter:
    """
    Intelligent routing system for MCP requests.
    
    Uses machine learning, server monitoring, and capability analysis
    to make optimal routing decisions for maximum performance and reliability.
    """
    
    def __init__(self, 
                 server_monitor: ServerMonitor,
                 capability_matcher: CapabilityMatcher,
                 tool_selector: ToolSelector):
        
        self.server_monitor = server_monitor
        self.capability_matcher = capability_matcher
        self.tool_selector = tool_selector
        
        # Routing state
        self.available_servers: Dict[str, MCPServerProfile] = {}
        self.server_connections: Dict[str, int] = {}  # Active connections per server
        self.server_load: Dict[str, float] = {}  # Current load per server
        
        # Round-robin state
        self.round_robin_index = 0
        
        # Routing history and metrics
        self.routing_history: List[Dict[str, Any]] = []
        self.routing_metrics = RoutingMetrics()
        
        # Learning data
        self.performance_feedback: List[Dict[str, Any]] = []
        
        # Strategy weights (learned over time)
        self.strategy_weights = {
            'health_score': 0.3,
            'performance_score': 0.25,
            'capability_match': 0.2,
            'load_balance': 0.15,
            'latency': 0.1
        }
    
    def register_server(self, server_profile: MCPServerProfile):
        """Register a server for routing"""
        self.available_servers[server_profile.server_name] = server_profile
        self.server_connections[server_profile.server_name] = 0
        self.server_load[server_profile.server_name] = 0.0
        
        logger.info(f"Registered server {server_profile.server_name} for routing")
    
    def unregister_server(self, server_name: str):
        """Unregister a server from routing"""
        if server_name in self.available_servers:
            del self.available_servers[server_name]
            del self.server_connections[server_name]
            del self.server_load[server_name]
            
            logger.info(f"Unregistered server {server_name} from routing")
    
    async def route_request(self, request: RoutingRequest) -> RoutingDecision:
        """
        Route a request to the optimal server.
        
        Args:
            request: Routing request with requirements and preferences
            
        Returns:
            Routing decision with selected server and rationale
        """
        start_time = datetime.utcnow()
        
        try:
            logger.debug(f"Routing request {request.request_id} with strategy {request.strategy.value}")
            
            # Filter available servers
            candidate_servers = await self._filter_candidate_servers(request)
            
            if not candidate_servers:
                logger.warning(f"No candidate servers for request {request.request_id}")
                return self._create_failed_decision(request, "No candidate servers available")
            
            # Apply routing strategy
            decision = await self._apply_routing_strategy(request, candidate_servers)
            
            # Calculate decision time
            decision_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            decision.decision_time_ms = decision_time
            
            # Update routing metrics
            self._update_routing_metrics(request, decision)
            
            # Record routing decision
            self._record_routing_decision(request, decision)
            
            logger.debug(f"Routed request {request.request_id} to {decision.selected_server} "
                        f"(score: {decision.routing_score:.3f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Routing failed for request {request.request_id}: {e}")
            return self._create_failed_decision(request, f"Routing error: {e}")
    
    async def _filter_candidate_servers(self, request: RoutingRequest) -> List[str]:
        """Filter servers based on request requirements"""
        candidates = []
        
        for server_name, server_profile in self.available_servers.items():
            # Check excluded servers
            if server_name in request.excluded_servers:
                continue
            
            # Check server health
            health = self.server_monitor.get_server_health(server_name)
            if health and health.status == HealthStatus.OFFLINE:
                continue
            
            # Check capability requirements
            if request.required_capabilities:
                has_all_capabilities = True
                for capability in request.required_capabilities:
                    if not server_profile.has_capability(capability):
                        has_all_capabilities = False
                        break
                
                if not has_all_capabilities:
                    continue
            
            # Check latency requirements
            if request.max_latency_ms:
                metrics = self.server_monitor.get_server_metrics(server_name)
                if metrics and metrics.avg_response_time_ms > request.max_latency_ms:
                    continue
            
            candidates.append(server_name)
        
        # Always include preferred servers if they exist
        for server_name in request.preferred_servers:
            if (server_name in self.available_servers and 
                server_name not in candidates):
                candidates.append(server_name)
        
        return candidates
    
    async def _apply_routing_strategy(self, 
                                    request: RoutingRequest,
                                    candidates: List[str]) -> RoutingDecision:
        """Apply the specified routing strategy"""
        
        if request.strategy == RoutingStrategy.ROUND_ROBIN:
            return await self._round_robin_routing(request, candidates)
        elif request.strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin_routing(request, candidates)
        elif request.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections_routing(request, candidates)
        elif request.strategy == RoutingStrategy.FASTEST_RESPONSE:
            return await self._fastest_response_routing(request, candidates)
        elif request.strategy == RoutingStrategy.HEALTH_BASED:
            return await self._health_based_routing(request, candidates)
        elif request.strategy == RoutingStrategy.CAPABILITY_OPTIMIZED:
            return await self._capability_optimized_routing(request, candidates)
        elif request.strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._load_balanced_routing(request, candidates)
        elif request.strategy == RoutingStrategy.INTELLIGENT:
            return await self._intelligent_routing(request, candidates)
        else:
            # Default to intelligent routing
            return await self._intelligent_routing(request, candidates)
    
    async def _round_robin_routing(self, request: RoutingRequest, candidates: List[str]) -> RoutingDecision:
        """Simple round-robin routing"""
        if not candidates:
            return self._create_failed_decision(request, "No candidates for round-robin")
        
        selected_server = candidates[self.round_robin_index % len(candidates)]
        self.round_robin_index += 1
        
        return RoutingDecision(
            selected_server=selected_server,
            backup_servers=candidates[:3],  # First 3 as backups
            routing_score=0.5,  # Neutral score
            routing_strategy_used=RoutingStrategy.ROUND_ROBIN,
            decision_confidence=0.7,
            decision_factors={'round_robin': 1.0}
        )
    
    async def _weighted_round_robin_routing(self, request: RoutingRequest, candidates: List[str]) -> RoutingDecision:
        """Weighted round-robin based on server performance"""
        if not candidates:
            return self._create_failed_decision(request, "No candidates for weighted round-robin")
        
        # Calculate weights based on server health
        weights = []
        for server_name in candidates:
            health = self.server_monitor.get_server_health(server_name)
            weight = health.overall_score if health else 0.5
            weights.append(weight)
        
        # Weighted selection
        total_weight = sum(weights)
        if total_weight == 0:
            return await self._round_robin_routing(request, candidates)
        
        weights = [w / total_weight for w in weights]
        selected_server = np.random.choice(candidates, p=weights)
        
        return RoutingDecision(
            selected_server=selected_server,
            backup_servers=[c for c in candidates if c != selected_server][:2],
            routing_score=weights[candidates.index(selected_server)],
            routing_strategy_used=RoutingStrategy.WEIGHTED_ROUND_ROBIN,
            decision_confidence=0.8,
            decision_factors={'weighted_selection': weights[candidates.index(selected_server)]}
        )
    
    async def _least_connections_routing(self, request: RoutingRequest, candidates: List[str]) -> RoutingDecision:
        """Route to server with least active connections"""
        if not candidates:
            return self._create_failed_decision(request, "No candidates for least connections")
        
        # Find server with minimum connections
        min_connections = float('inf')
        selected_server = candidates[0]
        
        for server_name in candidates:
            connections = self.server_connections.get(server_name, 0)
            if connections < min_connections:
                min_connections = connections
                selected_server = server_name
        
        return RoutingDecision(
            selected_server=selected_server,
            backup_servers=[c for c in candidates if c != selected_server][:2],
            routing_score=1.0 - (min_connections / 100.0),  # Normalize connections
            routing_strategy_used=RoutingStrategy.LEAST_CONNECTIONS,
            decision_confidence=0.8,
            decision_factors={'connections': min_connections}
        )
    
    async def _fastest_response_routing(self, request: RoutingRequest, candidates: List[str]) -> RoutingDecision:
        """Route to server with fastest response time"""
        if not candidates:
            return self._create_failed_decision(request, "No candidates for fastest response")
        
        fastest_time = float('inf')
        selected_server = candidates[0]
        
        for server_name in candidates:
            metrics = self.server_monitor.get_server_metrics(server_name)
            response_time = metrics.avg_response_time_ms if metrics else 1000.0
            
            if response_time < fastest_time:
                fastest_time = response_time
                selected_server = server_name
        
        # Calculate score (inverse of response time)
        score = max(0.1, 1.0 - (fastest_time / 5000.0))
        
        return RoutingDecision(
            selected_server=selected_server,
            backup_servers=[c for c in candidates if c != selected_server][:2],
            routing_score=score,
            routing_strategy_used=RoutingStrategy.FASTEST_RESPONSE,
            decision_confidence=0.9,
            estimated_latency_ms=fastest_time,
            decision_factors={'response_time_ms': fastest_time}
        )
    
    async def _health_based_routing(self, request: RoutingRequest, candidates: List[str]) -> RoutingDecision:
        """Route based on server health scores"""
        if not candidates:
            return self._create_failed_decision(request, "No candidates for health-based routing")
        
        best_health = 0.0
        selected_server = candidates[0]
        
        for server_name in candidates:
            health = self.server_monitor.get_server_health(server_name)
            health_score = health.overall_score if health else 0.0
            
            if health_score > best_health:
                best_health = health_score
                selected_server = server_name
        
        return RoutingDecision(
            selected_server=selected_server,
            backup_servers=[c for c in candidates if c != selected_server][:2],
            routing_score=best_health,
            routing_strategy_used=RoutingStrategy.HEALTH_BASED,
            decision_confidence=0.85,
            decision_factors={'health_score': best_health}
        )
    
    async def _capability_optimized_routing(self, request: RoutingRequest, candidates: List[str]) -> RoutingDecision:
        """Route based on capability matching"""
        if not candidates:
            return self._create_failed_decision(request, "No candidates for capability optimization")
        
        if not request.required_capabilities:
            # Fall back to health-based routing
            return await self._health_based_routing(request, candidates)
        
        # Get server profiles for candidates
        candidate_profiles = [self.available_servers[name] for name in candidates]
        
        # Match capabilities
        matches = await self.capability_matcher.match_capabilities(
            request.required_capabilities, candidate_profiles
        )
        
        if not matches:
            return await self._health_based_routing(request, candidates)
        
        # Select best match
        best_match = max(matches, key=lambda m: m.match_score.overall_score)
        selected_server = best_match.server_profile.server_name
        
        return RoutingDecision(
            selected_server=selected_server,
            backup_servers=[c for c in candidates if c != selected_server][:2],
            routing_score=best_match.match_score.overall_score,
            routing_strategy_used=RoutingStrategy.CAPABILITY_OPTIMIZED,
            decision_confidence=0.9,
            decision_factors={'capability_match': best_match.match_score.overall_score}
        )
    
    async def _load_balanced_routing(self, request: RoutingRequest, candidates: List[str]) -> RoutingDecision:
        """Route based on current server load"""
        if not candidates:
            return self._create_failed_decision(request, "No candidates for load balancing")
        
        # Find server with lowest load
        min_load = float('inf')
        selected_server = candidates[0]
        
        for server_name in candidates:
            load = self.server_load.get(server_name, 0.0)
            if load < min_load:
                min_load = load
                selected_server = server_name
        
        score = max(0.1, 1.0 - min_load)
        
        return RoutingDecision(
            selected_server=selected_server,
            backup_servers=[c for c in candidates if c != selected_server][:2],
            routing_score=score,
            routing_strategy_used=RoutingStrategy.LOAD_BALANCED,
            decision_confidence=0.8,
            decision_factors={'server_load': min_load}
        )
    
    async def _intelligent_routing(self, request: RoutingRequest, candidates: List[str]) -> RoutingDecision:
        """Intelligent routing using multiple factors"""
        if not candidates:
            return self._create_failed_decision(request, "No candidates for intelligent routing")
        
        best_score = 0.0
        selected_server = candidates[0]
        decision_factors = {}
        
        for server_name in candidates:
            score = await self._calculate_intelligent_score(server_name, request)
            
            if score > best_score:
                best_score = score
                selected_server = server_name
        
        # Calculate decision factors for selected server
        decision_factors = await self._get_decision_factors(selected_server, request)
        
        # Estimate performance
        metrics = self.server_monitor.get_server_metrics(selected_server)
        estimated_latency = metrics.avg_response_time_ms if metrics else 1000.0
        estimated_success = metrics.calculate_success_rate() if metrics else 0.9
        
        return RoutingDecision(
            selected_server=selected_server,
            backup_servers=sorted([c for c in candidates if c != selected_server],
                                 key=lambda c: asyncio.create_task(self._calculate_intelligent_score(c, request)),
                                 reverse=True)[:2],
            routing_score=best_score,
            routing_strategy_used=RoutingStrategy.INTELLIGENT,
            decision_confidence=0.95,
            estimated_latency_ms=estimated_latency,
            estimated_success_probability=estimated_success,
            decision_factors=decision_factors
        )
    
    async def _calculate_intelligent_score(self, server_name: str, request: RoutingRequest) -> float:
        """Calculate intelligent routing score for a server"""
        total_score = 0.0
        
        # Health score
        health = self.server_monitor.get_server_health(server_name)
        health_score = health.overall_score if health else 0.5
        total_score += health_score * self.strategy_weights['health_score']
        
        # Performance score
        metrics = self.server_monitor.get_server_metrics(server_name)
        if metrics:
            # Normalize response time (lower is better)
            response_score = max(0.0, 1.0 - (metrics.avg_response_time_ms / 5000.0))
            success_score = metrics.calculate_success_rate()
            performance_score = (response_score + success_score) / 2
        else:
            performance_score = 0.5
        
        total_score += performance_score * self.strategy_weights['performance_score']
        
        # Capability match score
        if request.required_capabilities:
            server_profile = self.available_servers[server_name]
            capability_score = 0.0
            
            for capability in request.required_capabilities:
                if server_profile.has_capability(capability):
                    capability_score += 1.0
            
            capability_score /= len(request.required_capabilities)
        else:
            capability_score = 1.0
        
        total_score += capability_score * self.strategy_weights['capability_match']
        
        # Load balance score
        current_load = self.server_load.get(server_name, 0.0)
        load_score = max(0.0, 1.0 - current_load)
        total_score += load_score * self.strategy_weights['load_balance']
        
        # Latency score
        if request.max_latency_ms and metrics:
            if metrics.avg_response_time_ms <= request.max_latency_ms:
                latency_score = 1.0
            else:
                latency_score = max(0.0, 1.0 - (metrics.avg_response_time_ms / request.max_latency_ms))
        else:
            latency_score = 1.0
        
        total_score += latency_score * self.strategy_weights['latency']
        
        # Priority boost for preferred servers
        if server_name in request.preferred_servers:
            total_score *= 1.2
        
        return min(1.0, total_score)
    
    async def _get_decision_factors(self, server_name: str, request: RoutingRequest) -> Dict[str, float]:
        """Get decision factors for a server"""
        factors = {}
        
        # Health factors
        health = self.server_monitor.get_server_health(server_name)
        if health:
            factors['health_score'] = health.overall_score
            factors['performance_health'] = health.performance_score
            factors['availability_health'] = health.availability_score
        
        # Performance factors
        metrics = self.server_monitor.get_server_metrics(server_name)
        if metrics:
            factors['response_time_ms'] = metrics.avg_response_time_ms
            factors['success_rate'] = metrics.calculate_success_rate()
            factors['error_rate'] = metrics.error_rate
        
        # Load factors
        factors['current_load'] = self.server_load.get(server_name, 0.0)
        factors['active_connections'] = self.server_connections.get(server_name, 0)
        
        return factors
    
    def _create_failed_decision(self, request: RoutingRequest, reason: str) -> RoutingDecision:
        """Create a failed routing decision"""
        return RoutingDecision(
            selected_server="",
            routing_score=0.0,
            routing_strategy_used=request.strategy,
            decision_confidence=0.0,
            decision_factors={'failure_reason': reason}
        )
    
    def _update_routing_metrics(self, request: RoutingRequest, decision: RoutingDecision):
        """Update routing metrics"""
        self.routing_metrics.total_requests += 1
        
        if decision.selected_server:
            self.routing_metrics.successful_routes += 1
        else:
            self.routing_metrics.failed_routes += 1
        
        # Update average decision time
        total_time = (self.routing_metrics.avg_decision_time_ms * 
                     (self.routing_metrics.total_requests - 1) + 
                     decision.decision_time_ms)
        self.routing_metrics.avg_decision_time_ms = total_time / self.routing_metrics.total_requests
        
        # Update average routing score
        total_score = (self.routing_metrics.avg_routing_score * 
                      (self.routing_metrics.total_requests - 1) + 
                      decision.routing_score)
        self.routing_metrics.avg_routing_score = total_score / self.routing_metrics.total_requests
    
    def _record_routing_decision(self, request: RoutingRequest, decision: RoutingDecision):
        """Record routing decision for analysis"""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request.request_id,
            'strategy': request.strategy.value,
            'selected_server': decision.selected_server,
            'routing_score': decision.routing_score,
            'decision_confidence': decision.decision_confidence,
            'decision_time_ms': decision.decision_time_ms,
            'required_capabilities': [cap.value for cap in request.required_capabilities],
            'decision_factors': decision.decision_factors
        }
        
        self.routing_history.append(record)
        
        # Keep limited history
        if len(self.routing_history) > 10000:
            self.routing_history = self.routing_history[-10000:]
    
    def update_server_load(self, server_name: str, load: float):
        """Update server load information"""
        self.server_load[server_name] = load
    
    def increment_connections(self, server_name: str):
        """Increment active connections for a server"""
        self.server_connections[server_name] = self.server_connections.get(server_name, 0) + 1
    
    def decrement_connections(self, server_name: str):
        """Decrement active connections for a server"""
        if server_name in self.server_connections:
            self.server_connections[server_name] = max(0, self.server_connections[server_name] - 1)
    
    def add_performance_feedback(self, request_id: str, server_name: str, 
                                success: bool, actual_latency_ms: float,
                                quality_score: float):
        """Add performance feedback for learning"""
        feedback = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request_id,
            'server_name': server_name,
            'success': success,
            'actual_latency_ms': actual_latency_ms,
            'quality_score': quality_score
        }
        
        self.performance_feedback.append(feedback)
        
        # Keep limited feedback
        if len(self.performance_feedback) > 10000:
            self.performance_feedback = self.performance_feedback[-10000:]
        
        # Update strategy weights based on feedback (simple learning)
        self._update_strategy_weights()
    
    def _update_strategy_weights(self):
        """Update strategy weights based on performance feedback"""
        # Simple learning algorithm - could be replaced with more sophisticated ML
        if len(self.performance_feedback) < 100:
            return
        
        recent_feedback = self.performance_feedback[-100:]
        
        # Analyze success rates by decision factors
        # This is a simplified implementation
        successful_decisions = [f for f in recent_feedback if f['success']]
        
        if len(successful_decisions) > 50:
            # Slightly increase weights for factors that correlate with success
            # This is a placeholder for more sophisticated learning
            pass
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        # Server utilization
        total_requests = sum(self.server_connections.values())
        server_utilization = {}
        for server_name, connections in self.server_connections.items():
            if total_requests > 0:
                server_utilization[server_name] = connections / total_requests
            else:
                server_utilization[server_name] = 0.0
        
        # Strategy success rates
        strategy_stats = {}
        for record in self.routing_history:
            strategy = record['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'total': 0, 'successful': 0}
            
            strategy_stats[strategy]['total'] += 1
            if record['selected_server']:
                strategy_stats[strategy]['successful'] += 1
        
        strategy_success_rates = {}
        for strategy, stats in strategy_stats.items():
            if stats['total'] > 0:
                strategy_success_rates[strategy] = stats['successful'] / stats['total']
            else:
                strategy_success_rates[strategy] = 0.0
        
        return {
            'total_servers': len(self.available_servers),
            'routing_metrics': {
                'total_requests': self.routing_metrics.total_requests,
                'success_rate': (self.routing_metrics.successful_routes / 
                               max(self.routing_metrics.total_requests, 1)),
                'avg_decision_time_ms': self.routing_metrics.avg_decision_time_ms,
                'avg_routing_score': self.routing_metrics.avg_routing_score
            },
            'server_utilization': server_utilization,
            'strategy_success_rates': strategy_success_rates,
            'current_strategy_weights': self.strategy_weights.copy(),
            'feedback_count': len(self.performance_feedback)
        }
