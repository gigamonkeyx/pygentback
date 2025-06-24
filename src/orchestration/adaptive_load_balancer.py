"""
Adaptive Load Balancer

Advanced load balancing system with predictive capabilities, multi-dimensional
optimization, and self-learning algorithms for optimal resource allocation.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics

from .coordination_models import (
    AgentCapability, MCPServerInfo, TaskRequest, PerformanceMetrics,
    OrchestrationConfig, AgentID, ServerID
)

logger = logging.getLogger(__name__)


@dataclass
class LoadPrediction:
    """Prediction for future load patterns."""
    agent_id: str
    predicted_load: float
    confidence: float
    time_horizon: timedelta
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Resource allocation decision."""
    agent_id: str
    server_id: str
    task_id: str
    allocation_score: float
    expected_completion_time: float
    resource_cost: float


class AdaptiveLoadBalancer:
    """
    Advanced load balancer with predictive capabilities and multi-objective optimization.
    
    Features:
    - Predictive load forecasting using historical patterns
    - Multi-dimensional optimization (latency, throughput, resource utilization)
    - Dynamic weight adjustment based on performance feedback
    - Fault-tolerant distribution with automatic failover
    - Learning-based optimization of allocation strategies
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        
        # Load tracking
        self.agent_load_history: Dict[AgentID, deque] = defaultdict(lambda: deque(maxlen=288))  # 24 hours
        self.server_load_history: Dict[ServerID, deque] = defaultdict(lambda: deque(maxlen=288))
        self.task_completion_history: deque = deque(maxlen=1000)
        
        # Prediction models
        self.load_predictors: Dict[str, Any] = {}
        self.prediction_accuracy: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Optimization weights
        self.optimization_weights = {
            'latency': 0.3,
            'throughput': 0.25,
            'resource_utilization': 0.2,
            'fairness': 0.15,
            'reliability': 0.1
        }
        
        # Performance tracking
        self.allocation_performance: Dict[str, List[float]] = defaultdict(list)
        self.strategy_effectiveness: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.prediction_window = timedelta(minutes=30)
        self.adaptation_threshold = 0.05
        
        logger.info("Adaptive Load Balancer initialized")
    
    async def predict_load(self, 
                          agent_id: AgentID, 
                          time_horizon: timedelta = None) -> LoadPrediction:
        """Predict future load for an agent."""
        if time_horizon is None:
            time_horizon = self.prediction_window
        
        try:
            # Get historical load data
            load_history = list(self.agent_load_history[agent_id])
            
            if len(load_history) < 10:
                # Insufficient data, return conservative prediction
                return LoadPrediction(
                    agent_id=agent_id,
                    predicted_load=0.5,
                    confidence=0.3,
                    time_horizon=time_horizon,
                    factors={'insufficient_data': 1.0}
                )
            
            # Simple trend analysis (can be enhanced with ML models)
            recent_loads = load_history[-10:]
            trend = self._calculate_trend(recent_loads)
            seasonal_factor = self._calculate_seasonal_factor(load_history)
            
            # Base prediction on recent average
            base_load = statistics.mean(recent_loads)
            
            # Apply trend and seasonal adjustments
            predicted_load = base_load + (trend * 0.3) + (seasonal_factor * 0.2)
            predicted_load = max(0.0, min(1.0, predicted_load))
            
            # Calculate confidence based on prediction stability
            variance = statistics.variance(recent_loads) if len(recent_loads) > 1 else 0.1
            confidence = max(0.1, min(0.9, 1.0 - variance))
            
            return LoadPrediction(
                agent_id=agent_id,
                predicted_load=predicted_load,
                confidence=confidence,
                time_horizon=time_horizon,
                factors={
                    'trend': trend,
                    'seasonal': seasonal_factor,
                    'base_load': base_load,
                    'variance': variance
                }
            )
            
        except Exception as e:
            logger.error(f"Load prediction failed for agent {agent_id}: {e}")
            return LoadPrediction(
                agent_id=agent_id,
                predicted_load=0.5,
                confidence=0.1,
                time_horizon=time_horizon,
                factors={'error': 1.0}
            )
    
    async def optimize_allocation(self, 
                                task: TaskRequest,
                                available_agents: List[AgentCapability],
                                available_servers: List[MCPServerInfo]) -> Optional[ResourceAllocation]:
        """Optimize resource allocation for a task using multi-objective optimization."""
        try:
            best_allocation = None
            best_score = -1.0
            
            # Evaluate all possible allocations
            for agent in available_agents:
                if not agent.can_handle_task(task.task_type):
                    continue
                
                # Get suitable servers for this task
                suitable_servers = [
                    server for server in available_servers
                    if any(req_type.value in server.capabilities for req_type in task.required_mcp_servers)
                ]
                
                if not suitable_servers:
                    continue
                
                for server in suitable_servers:
                    allocation = await self._evaluate_allocation(task, agent, server)
                    
                    if allocation and allocation.allocation_score > best_score:
                        best_score = allocation.allocation_score
                        best_allocation = allocation
            
            return best_allocation
            
        except Exception as e:
            logger.error(f"Allocation optimization failed: {e}")
            return None
    
    async def _evaluate_allocation(self, 
                                 task: TaskRequest,
                                 agent: AgentCapability,
                                 server: MCPServerInfo) -> Optional[ResourceAllocation]:
        """Evaluate a specific allocation option."""
        try:
            # Predict agent load
            load_prediction = await self.predict_load(agent.agent_id)
            
            # Calculate latency score
            latency_score = await self._calculate_latency_score(agent, server, task)
            
            # Calculate throughput score
            throughput_score = await self._calculate_throughput_score(agent, server)
            
            # Calculate resource utilization score
            utilization_score = await self._calculate_utilization_score(agent, server, load_prediction)
            
            # Calculate fairness score
            fairness_score = await self._calculate_fairness_score(agent)
            
            # Calculate reliability score
            reliability_score = await self._calculate_reliability_score(agent, server)
            
            # Combine scores using optimization weights
            total_score = (
                latency_score * self.optimization_weights['latency'] +
                throughput_score * self.optimization_weights['throughput'] +
                utilization_score * self.optimization_weights['resource_utilization'] +
                fairness_score * self.optimization_weights['fairness'] +
                reliability_score * self.optimization_weights['reliability']
            )
            
            # Estimate completion time
            base_time = task.estimated_duration or 60.0
            load_factor = 1.0 + (load_prediction.predicted_load * 0.5)
            performance_factor = agent.performance_score
            completion_time = base_time * load_factor / max(performance_factor, 0.1)
            
            # Calculate resource cost
            resource_cost = self._calculate_resource_cost(agent, server, completion_time)
            
            return ResourceAllocation(
                agent_id=agent.agent_id,
                server_id=server.server_id,
                task_id=task.task_id,
                allocation_score=total_score,
                expected_completion_time=completion_time,
                resource_cost=resource_cost
            )
            
        except Exception as e:
            logger.error(f"Allocation evaluation failed: {e}")
            return None
    
    async def _calculate_latency_score(self, 
                                     agent: AgentCapability,
                                     server: MCPServerInfo,
                                     task: TaskRequest) -> float:
        """Calculate latency optimization score."""
        try:
            # Base latency from server response time
            server_latency = server.response_time_avg or 0.1
            
            # Agent processing latency based on current load
            agent_latency = agent.utilization_rate * 2.0  # Higher load = higher latency
            
            # Task complexity factor
            complexity_factor = len(task.required_capabilities) * 0.1
            
            total_latency = server_latency + agent_latency + complexity_factor
            
            # Convert to score (lower latency = higher score)
            latency_score = max(0.0, 1.0 - (total_latency / 10.0))
            
            return latency_score
            
        except Exception as e:
            logger.error(f"Latency score calculation failed: {e}")
            return 0.5
    
    async def _calculate_throughput_score(self, 
                                        agent: AgentCapability,
                                        server: MCPServerInfo) -> float:
        """Calculate throughput optimization score."""
        try:
            # Agent throughput based on performance and available capacity
            agent_capacity = agent.max_concurrent_tasks - agent.current_load
            agent_throughput = agent.performance_score * agent_capacity
            
            # Server throughput based on connection availability
            server_capacity = server.max_connections - server.current_connections
            server_throughput = server.success_rate * server_capacity
            
            # Combined throughput score
            combined_throughput = min(agent_throughput, server_throughput)
            normalized_score = combined_throughput / max(agent.max_concurrent_tasks, 1)
            
            return max(0.0, min(1.0, normalized_score))
            
        except Exception as e:
            logger.error(f"Throughput score calculation failed: {e}")
            return 0.5
    
    async def _calculate_utilization_score(self, 
                                         agent: AgentCapability,
                                         server: MCPServerInfo,
                                         load_prediction: LoadPrediction) -> float:
        """Calculate resource utilization optimization score."""
        try:
            # Optimal utilization is around 70-80%
            optimal_agent_util = 0.75
            optimal_server_util = 0.75
            
            # Current utilization
            current_agent_util = agent.utilization_rate
            current_server_util = server.connection_utilization
            
            # Predicted utilization after task assignment
            predicted_agent_util = min(1.0, current_agent_util + (1.0 / agent.max_concurrent_tasks))
            predicted_server_util = min(1.0, current_server_util + (1.0 / server.max_connections))
            
            # Score based on distance from optimal
            agent_util_score = 1.0 - abs(predicted_agent_util - optimal_agent_util)
            server_util_score = 1.0 - abs(predicted_server_util - optimal_server_util)
            
            # Weight by prediction confidence
            confidence_weight = load_prediction.confidence
            utilization_score = (
                agent_util_score * confidence_weight +
                server_util_score * (1.0 - confidence_weight)
            )
            
            return max(0.0, min(1.0, utilization_score))
            
        except Exception as e:
            logger.error(f"Utilization score calculation failed: {e}")
            return 0.5
    
    async def _calculate_fairness_score(self, agent: AgentCapability) -> float:
        """Calculate fairness score to ensure equitable task distribution."""
        try:
            # Get recent task assignments for this agent
            recent_assignments = len([
                perf for perf in self.allocation_performance.get(agent.agent_id, [])
                if len(perf) > 0  # Has recent activity
            ])
            
            # Calculate average assignments across all agents
            all_assignments = [
                len(assignments) for assignments in self.allocation_performance.values()
            ]
            avg_assignments = statistics.mean(all_assignments) if all_assignments else 1.0
            
            # Fairness score favors agents with fewer recent assignments
            if avg_assignments > 0:
                fairness_ratio = recent_assignments / avg_assignments
                fairness_score = max(0.0, 1.0 - (fairness_ratio - 1.0))
            else:
                fairness_score = 1.0
            
            return max(0.0, min(1.0, fairness_score))
            
        except Exception as e:
            logger.error(f"Fairness score calculation failed: {e}")
            return 0.5
    
    async def _calculate_reliability_score(self, 
                                         agent: AgentCapability,
                                         server: MCPServerInfo) -> float:
        """Calculate reliability score based on historical performance."""
        try:
            # Agent reliability based on performance score and recent activity
            agent_reliability = agent.performance_score
            
            # Adjust for recent activity (more active = more reliable data)
            if agent.last_activity:
                time_since_activity = (datetime.utcnow() - agent.last_activity).total_seconds()
                activity_factor = max(0.5, 1.0 - (time_since_activity / 3600))  # 1 hour decay
                agent_reliability *= activity_factor
            
            # Server reliability based on success rate and health
            server_reliability = server.success_rate if server.is_healthy else 0.1
            
            # Combined reliability score
            combined_reliability = (agent_reliability + server_reliability) / 2.0
            
            return max(0.0, min(1.0, combined_reliability))
            
        except Exception as e:
            logger.error(f"Reliability score calculation failed: {e}")
            return 0.5
    
    def _calculate_resource_cost(self, 
                               agent: AgentCapability,
                               server: MCPServerInfo,
                               completion_time: float) -> float:
        """Calculate resource cost for the allocation."""
        try:
            # Agent cost based on utilization and performance
            agent_cost = (1.0 / max(agent.performance_score, 0.1)) * completion_time
            
            # Server cost based on response time and load
            server_cost = server.response_time_avg * server.connection_utilization
            
            # Total resource cost
            total_cost = agent_cost + server_cost
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Resource cost calculation failed: {e}")
            return 1.0
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend in data using simple linear regression."""
        if len(data) < 2:
            return 0.0
        
        try:
            x = list(range(len(data)))
            y = data
            
            n = len(data)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            # Linear regression slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            return slope
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {e}")
            return 0.0
    
    def _calculate_seasonal_factor(self, data: List[float]) -> float:
        """Calculate seasonal factor in load patterns."""
        if len(data) < 24:  # Need at least 24 data points for daily pattern
            return 0.0
        
        try:
            # Simple seasonal analysis - compare current hour to historical average
            current_hour = datetime.utcnow().hour
            hourly_averages = defaultdict(list)
            
            # Group data by hour (assuming 5-minute intervals)
            for i, value in enumerate(data):
                hour = (current_hour - (len(data) - i) // 12) % 24
                hourly_averages[hour].append(value)
            
            # Calculate seasonal factor
            if current_hour in hourly_averages and hourly_averages[current_hour]:
                current_hour_avg = statistics.mean(hourly_averages[current_hour])
                overall_avg = statistics.mean(data)
                seasonal_factor = current_hour_avg - overall_avg
            else:
                seasonal_factor = 0.0
            
            return seasonal_factor
            
        except Exception as e:
            logger.error(f"Seasonal factor calculation failed: {e}")
            return 0.0
    
    async def update_performance_feedback(self, 
                                        allocation: ResourceAllocation,
                                        actual_completion_time: float,
                                        success: bool):
        """Update performance feedback for learning."""
        try:
            # Calculate prediction accuracy
            prediction_error = abs(allocation.expected_completion_time - actual_completion_time)
            accuracy = max(0.0, 1.0 - (prediction_error / allocation.expected_completion_time))
            
            # Update allocation performance
            self.allocation_performance[allocation.agent_id].append(accuracy)
            
            # Limit history size
            if len(self.allocation_performance[allocation.agent_id]) > 100:
                self.allocation_performance[allocation.agent_id] = \
                    self.allocation_performance[allocation.agent_id][-100:]
            
            # Update strategy effectiveness
            strategy_key = f"{allocation.agent_id}_{allocation.server_id}"
            current_effectiveness = self.strategy_effectiveness[strategy_key]
            
            # Exponential moving average update
            success_score = 1.0 if success else 0.0
            new_effectiveness = (
                current_effectiveness * (1 - self.learning_rate) +
                success_score * self.learning_rate
            )
            self.strategy_effectiveness[strategy_key] = new_effectiveness
            
            # Adapt optimization weights based on performance
            await self._adapt_optimization_weights(accuracy, success)
            
        except Exception as e:
            logger.error(f"Performance feedback update failed: {e}")
    
    async def _adapt_optimization_weights(self, accuracy: float, success: bool):
        """Adapt optimization weights based on performance feedback."""
        try:
            # Only adapt if we have significant performance deviation
            if abs(accuracy - 0.8) < self.adaptation_threshold:
                return
            
            # Adjust weights based on performance
            if accuracy < 0.5:  # Poor prediction accuracy
                # Increase weight on reliability and decrease on latency
                self.optimization_weights['reliability'] = min(0.3, self.optimization_weights['reliability'] + 0.02)
                self.optimization_weights['latency'] = max(0.1, self.optimization_weights['latency'] - 0.02)
            
            elif accuracy > 0.9:  # Excellent prediction accuracy
                # Increase weight on throughput optimization
                self.optimization_weights['throughput'] = min(0.4, self.optimization_weights['throughput'] + 0.02)
                self.optimization_weights['fairness'] = max(0.05, self.optimization_weights['fairness'] - 0.02)
            
            # Normalize weights to sum to 1.0
            total_weight = sum(self.optimization_weights.values())
            for key in self.optimization_weights:
                self.optimization_weights[key] /= total_weight
            
        except Exception as e:
            logger.error(f"Weight adaptation failed: {e}")
    
    async def get_load_balancer_metrics(self) -> Dict[str, Any]:
        """Get comprehensive load balancer metrics."""
        try:
            # Calculate average prediction accuracy
            all_accuracies = []
            for agent_perfs in self.allocation_performance.values():
                all_accuracies.extend(agent_perfs)
            
            avg_accuracy = statistics.mean(all_accuracies) if all_accuracies else 0.0
            
            # Calculate strategy effectiveness
            avg_effectiveness = statistics.mean(self.strategy_effectiveness.values()) if self.strategy_effectiveness else 0.0
            
            return {
                'prediction_accuracy': avg_accuracy,
                'strategy_effectiveness': avg_effectiveness,
                'optimization_weights': dict(self.optimization_weights),
                'total_allocations': sum(len(perfs) for perfs in self.allocation_performance.values()),
                'active_agents': len(self.agent_load_history),
                'active_servers': len(self.server_load_history),
                'learning_rate': self.learning_rate
            }
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return {'error': str(e)}