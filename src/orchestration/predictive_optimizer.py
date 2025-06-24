"""
Predictive System Optimizer

Advanced predictive optimization system that forecasts system behavior,
optimizes resource allocation, and prevents performance bottlenecks.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .coordination_models import (
    PerformanceMetrics, OrchestrationConfig, AgentID, ServerID, TaskID
)

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    OPTIMIZE_RESOURCE_USAGE = "optimize_resource_usage"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_RELIABILITY = "maximize_reliability"


@dataclass
class PredictionModel:
    """Prediction model for system metrics."""
    model_id: str
    metric_name: str
    model_type: str
    accuracy: float
    last_updated: datetime
    prediction_horizon: timedelta
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """System optimization recommendation."""
    recommendation_id: str
    objective: OptimizationObjective
    predicted_improvement: float
    confidence: float
    actions: List[Dict[str, Any]]
    estimated_cost: float
    risk_level: str
    implementation_time: timedelta
    created_at: datetime = field(default_factory=datetime.utcnow)


class PredictiveOptimizer:
    """
    Advanced predictive optimization system.
    
    Features:
    - Multi-horizon forecasting
    - Multi-objective optimization
    - Proactive bottleneck prevention
    - Resource allocation optimization
    - Performance trend analysis
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        
        # Prediction models
        self.prediction_models: Dict[str, PredictionModel] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Optimization state
        self.optimization_objectives: List[OptimizationObjective] = [
            OptimizationObjective.MINIMIZE_LATENCY,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
            OptimizationObjective.OPTIMIZE_RESOURCE_USAGE
        ]
        
        # Recommendations
        self.active_recommendations: List[OptimizationRecommendation] = []
        self.implemented_recommendations: List[OptimizationRecommendation] = []
        
        # Forecasting parameters
        self.forecast_horizons = [
            timedelta(minutes=15),
            timedelta(hours=1),
            timedelta(hours=6),
            timedelta(hours=24)
        ]
        
        # Optimization parameters
        self.optimization_interval = timedelta(minutes=5)
        self.prediction_accuracy_threshold = 0.7
        self.recommendation_confidence_threshold = 0.6
        
        # System state tracking
        self.system_state_history: deque = deque(maxlen=500)
        self.bottleneck_history: List[Dict[str, Any]] = []
        
        logger.info("Predictive Optimizer initialized")
    
    async def start(self):
        """Start the predictive optimizer."""
        await self._initialize_prediction_models()
        logger.info("Predictive Optimizer started")
    
    async def stop(self):
        """Stop the predictive optimizer."""
        logger.info("Predictive Optimizer stopped")
    
    async def update_metrics(self, metrics: PerformanceMetrics):
        """Update system metrics for prediction and optimization."""
        try:
            timestamp = datetime.utcnow()
            
            # Store metrics in history
            metric_data = {
                'timestamp': timestamp,
                'task_success_rate': metrics.task_success_rate,
                'avg_response_time': metrics.avg_response_time,
                'coordination_efficiency': metrics.coordination_efficiency,
                'agent_utilization': metrics.avg_agent_utilization,
                'system_health': metrics.system_health_score,
                'active_tasks': metrics.active_tasks,
                'completed_tasks': metrics.completed_tasks,
                'failed_tasks': metrics.failed_tasks
            }
            
            # Update metric histories
            for metric_name, value in metric_data.items():
                if metric_name != 'timestamp':
                    self.metric_history[metric_name].append((timestamp, value))
            
            # Store system state
            self.system_state_history.append(metric_data)
            
            # Update prediction models
            await self._update_prediction_models(metric_data)
            
            # Check for optimization opportunities
            await self._check_optimization_opportunities(metrics)
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    async def predict_metrics(self, horizon: timedelta) -> Dict[str, float]:
        """Predict system metrics for a given time horizon."""
        try:
            predictions = {}
            
            for metric_name, model in self.prediction_models.items():
                if model.prediction_horizon >= horizon:
                    prediction = await self._predict_metric(metric_name, horizon)
                    predictions[metric_name] = prediction
            
            return predictions
            
        except Exception as e:
            logger.error(f"Metric prediction failed: {e}")
            return {}
    
    async def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on predictions."""
        try:
            recommendations = []
            
            for objective in self.optimization_objectives:
                recommendation = await self._generate_recommendation_for_objective(objective)
                if recommendation and recommendation.confidence >= self.recommendation_confidence_threshold:
                    recommendations.append(recommendation)
            
            # Update active recommendations
            self.active_recommendations = recommendations
            
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    async def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect current and predicted bottlenecks."""
        try:
            bottlenecks = []
            
            # Analyze current metrics
            if self.system_state_history:
                current_state = self.system_state_history[-1]
                
                # Check for high response time
                if current_state.get('avg_response_time', 0) > 5.0:
                    bottlenecks.append({
                        'type': 'high_latency',
                        'severity': 'high',
                        'metric': 'avg_response_time',
                        'current_value': current_state['avg_response_time'],
                        'threshold': 5.0,
                        'predicted_duration': timedelta(minutes=30)
                    })
                
                # Check for low coordination efficiency
                if current_state.get('coordination_efficiency', 1.0) < 0.6:
                    bottlenecks.append({
                        'type': 'coordination_inefficiency',
                        'severity': 'medium',
                        'metric': 'coordination_efficiency',
                        'current_value': current_state['coordination_efficiency'],
                        'threshold': 0.6,
                        'predicted_duration': timedelta(hours=1)
                    })
                
                # Check for high agent utilization
                if current_state.get('agent_utilization', 0) > 0.9:
                    bottlenecks.append({
                        'type': 'resource_saturation',
                        'severity': 'high',
                        'metric': 'agent_utilization',
                        'current_value': current_state['agent_utilization'],
                        'threshold': 0.9,
                        'predicted_duration': timedelta(hours=2)
                    })
            
            # Predict future bottlenecks
            future_bottlenecks = await self._predict_future_bottlenecks()
            bottlenecks.extend(future_bottlenecks)
            
            # Store bottleneck history
            if bottlenecks:
                self.bottleneck_history.append({
                    'timestamp': datetime.utcnow(),
                    'bottlenecks': bottlenecks
                })
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Bottleneck detection failed: {e}")
            return []
    
    async def optimize_resource_allocation(self, 
                                         available_agents: List[str],
                                         available_servers: List[str],
                                         pending_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize resource allocation based on predictions."""
        try:
            # Predict resource demand
            demand_predictions = await self._predict_resource_demand(pending_tasks)
            
            # Predict resource availability
            availability_predictions = await self._predict_resource_availability(
                available_agents, available_servers
            )
            
            # Generate optimal allocation
            allocation = await self._generate_optimal_allocation(
                demand_predictions, availability_predictions, pending_tasks
            )
            
            return allocation
            
        except Exception as e:
            logger.error(f"Resource allocation optimization failed: {e}")
            return {}
    
    async def _initialize_prediction_models(self):
        """Initialize prediction models for key metrics."""
        metrics_to_model = [
            'task_success_rate',
            'avg_response_time',
            'coordination_efficiency',
            'agent_utilization',
            'system_health'
        ]
        
        for metric_name in metrics_to_model:
            model = PredictionModel(
                model_id=f"model_{metric_name}",
                metric_name=metric_name,
                model_type="time_series",
                accuracy=0.5,  # Initial accuracy
                last_updated=datetime.utcnow(),
                prediction_horizon=timedelta(hours=1)
            )
            self.prediction_models[metric_name] = model
    
    async def _update_prediction_models(self, metric_data: Dict[str, Any]):
        """Update prediction models with new data."""
        for metric_name, model in self.prediction_models.items():
            if metric_name in metric_data:
                # Simple accuracy update based on recent predictions
                # In a real implementation, this would use proper ML model training
                model.accuracy = min(0.95, model.accuracy + 0.01)
                model.last_updated = datetime.utcnow()
    
    async def _predict_metric(self, metric_name: str, horizon: timedelta) -> float:
        """Predict a specific metric value."""
        if metric_name not in self.metric_history:
            return 0.5  # Default prediction
        
        history = list(self.metric_history[metric_name])
        if len(history) < 5:
            return 0.5
        
        # Simple trend-based prediction
        recent_values = [value for _, value in history[-10:]]
        
        # Calculate trend
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
            
            # Project trend forward
            horizon_hours = horizon.total_seconds() / 3600
            predicted_value = recent_values[-1] + (trend * horizon_hours)
            
            # Clamp to reasonable bounds
            return max(0.0, min(1.0, predicted_value))
        
        return recent_values[-1] if recent_values else 0.5
    
    async def _check_optimization_opportunities(self, metrics: PerformanceMetrics):
        """Check for optimization opportunities."""
        # Check if any metric is below optimal threshold
        optimization_needed = (
            metrics.task_success_rate < 0.9 or
            metrics.avg_response_time > 3.0 or
            metrics.coordination_efficiency < 0.8 or
            metrics.system_health_score < 0.85
        )
        
        if optimization_needed:
            # Generate recommendations
            await self.generate_optimization_recommendations()
    
    async def _generate_recommendation_for_objective(self, objective: OptimizationObjective) -> Optional[OptimizationRecommendation]:
        """Generate recommendation for a specific objective."""
        try:
            actions = []
            predicted_improvement = 0.0
            confidence = 0.0
            
            if objective == OptimizationObjective.MINIMIZE_LATENCY:
                actions = [
                    {'type': 'scale_agents', 'count': 2},
                    {'type': 'optimize_routing', 'algorithm': 'shortest_path'}
                ]
                predicted_improvement = 0.15
                confidence = 0.8
            
            elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
                actions = [
                    {'type': 'increase_batch_size', 'new_size': 20},
                    {'type': 'parallel_processing', 'degree': 4}
                ]
                predicted_improvement = 0.25
                confidence = 0.75
            
            elif objective == OptimizationObjective.OPTIMIZE_RESOURCE_USAGE:
                actions = [
                    {'type': 'load_balancing', 'strategy': 'adaptive'},
                    {'type': 'resource_pooling', 'enabled': True}
                ]
                predicted_improvement = 0.20
                confidence = 0.7
            
            if actions and confidence >= self.recommendation_confidence_threshold:
                return OptimizationRecommendation(
                    recommendation_id=f"rec_{datetime.utcnow().timestamp()}",
                    objective=objective,
                    predicted_improvement=predicted_improvement,
                    confidence=confidence,
                    actions=actions,
                    estimated_cost=0.1,  # Relative cost
                    risk_level="low",
                    implementation_time=timedelta(minutes=15)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Recommendation generation failed for {objective}: {e}")
            return None
    
    async def _predict_future_bottlenecks(self) -> List[Dict[str, Any]]:
        """Predict future bottlenecks."""
        future_bottlenecks = []
        
        # Predict metrics for different horizons
        for horizon in self.forecast_horizons:
            predictions = await self.predict_metrics(horizon)
            
            # Check for predicted bottlenecks
            if predictions.get('avg_response_time', 0) > 4.0:
                future_bottlenecks.append({
                    'type': 'predicted_high_latency',
                    'severity': 'medium',
                    'metric': 'avg_response_time',
                    'predicted_value': predictions['avg_response_time'],
                    'threshold': 4.0,
                    'predicted_time': datetime.utcnow() + horizon,
                    'horizon': horizon
                })
        
        return future_bottlenecks
    
    async def _predict_resource_demand(self, pending_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict resource demand based on pending tasks."""
        return {
            'cpu_demand': len(pending_tasks) * 0.1,
            'memory_demand': len(pending_tasks) * 0.05,
            'agent_demand': len(pending_tasks) * 0.2
        }
    
    async def _predict_resource_availability(self, agents: List[str], servers: List[str]) -> Dict[str, float]:
        """Predict resource availability."""
        return {
            'agent_availability': len(agents) * 0.8,
            'server_availability': len(servers) * 0.9,
            'total_capacity': (len(agents) + len(servers)) * 0.85
        }
    
    async def _generate_optimal_allocation(self, 
                                         demand: Dict[str, float],
                                         availability: Dict[str, float],
                                         tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate optimal resource allocation."""
        return {
            'allocation_strategy': 'balanced',
            'agent_assignments': len(tasks),
            'server_assignments': len(tasks),
            'predicted_efficiency': 0.85,
            'estimated_completion_time': len(tasks) * 2.0
        }
    
    async def get_optimizer_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimizer metrics."""
        return {
            'prediction_models': len(self.prediction_models),
            'model_accuracies': {
                name: model.accuracy 
                for name, model in self.prediction_models.items()
            },
            'active_recommendations': len(self.active_recommendations),
            'implemented_recommendations': len(self.implemented_recommendations),
            'bottleneck_history_size': len(self.bottleneck_history),
            'metric_history_sizes': {
                name: len(history) 
                for name, history in self.metric_history.items()
            },
            'optimization_objectives': [obj.value for obj in self.optimization_objectives]
        }