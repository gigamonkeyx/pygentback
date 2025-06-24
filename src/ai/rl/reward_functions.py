"""
Reward Functions for Recipe RL

Implements reward calculation for reinforcement learning agents
optimizing recipe performance and structure.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

try:
    from ..nas.architecture_encoder import RecipeArchitecture
    from ..nas.performance_predictor import PerformancePrediction
    from .action_space import RecipeAction
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    
    @dataclass
    class RecipeArchitecture:
        nodes: List[Any] = field(default_factory=list)
        edges: List[Any] = field(default_factory=list)
    
    @dataclass
    class PerformancePrediction:
        success_probability: float = 0.5
        execution_time_ms: float = 1000
        memory_usage_mb: float = 512
        complexity_score: float = 5.0
        confidence: float = 0.8
    
    @dataclass
    class RecipeAction:
        action_type: str = "modify"

logger = logging.getLogger(__name__)


class RewardComponent(Enum):
    """Components of the reward function"""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    COMPLEXITY = "complexity"
    STABILITY = "stability"
    NOVELTY = "novelty"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    DIVERSITY = "diversity"
    ROBUSTNESS = "robustness"


@dataclass
class RewardComponents:
    """Individual reward components"""
    performance: float = 0.0
    efficiency: float = 0.0
    complexity: float = 0.0
    stability: float = 0.0
    novelty: float = 0.0
    constraint_satisfaction: float = 0.0
    diversity: float = 0.0
    robustness: float = 0.0
    
    def total_reward(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate total weighted reward"""
        if weights is None:
            weights = {
                'performance': 0.3,
                'efficiency': 0.2,
                'complexity': 0.15,
                'stability': 0.15,
                'novelty': 0.1,
                'constraint_satisfaction': 0.05,
                'diversity': 0.03,
                'robustness': 0.02
            }
        
        total = 0.0
        for component, value in self.__dict__.items():
            weight = weights.get(component, 0.0)
            total += weight * value
        
        return total


class RewardFunction(ABC):
    """Abstract base class for reward functions"""
    
    @abstractmethod
    def calculate_reward(self, 
                        current_architecture: RecipeArchitecture,
                        previous_architecture: Optional[RecipeArchitecture],
                        action: RecipeAction,
                        performance_prediction: PerformancePrediction,
                        context: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        """
        Calculate reward for a state transition.
        
        Args:
            current_architecture: Current recipe architecture
            previous_architecture: Previous architecture (before action)
            action: Action that was taken
            performance_prediction: Predicted performance
            context: Additional context information
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        pass


class PerformanceReward(RewardFunction):
    """Reward based on recipe performance metrics"""
    
    def __init__(self, target_success_rate: float = 0.9):
        self.target_success_rate = target_success_rate
    
    def calculate_reward(self, 
                        current_architecture: RecipeArchitecture,
                        previous_architecture: Optional[RecipeArchitecture],
                        action: RecipeAction,
                        performance_prediction: PerformancePrediction,
                        context: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        
        components = RewardComponents()
        
        # Performance reward based on success probability
        success_rate = performance_prediction.success_probability
        components.performance = self._calculate_performance_reward(success_rate)
        
        # Efficiency reward based on execution time and memory
        components.efficiency = self._calculate_efficiency_reward(
            performance_prediction.execution_time_ms,
            performance_prediction.memory_usage_mb
        )
        
        # Complexity penalty
        components.complexity = self._calculate_complexity_reward(
            performance_prediction.complexity_score
        )
        
        # Stability reward based on confidence
        components.stability = self._calculate_stability_reward(
            performance_prediction.confidence
        )
        
        total_reward = components.total_reward()
        
        return total_reward, components
    
    def _calculate_performance_reward(self, success_rate: float) -> float:
        """Calculate performance-based reward"""
        # Exponential reward for high success rates
        if success_rate >= self.target_success_rate:
            return 1.0 + (success_rate - self.target_success_rate) * 5.0
        else:
            # Penalty for low success rates
            return success_rate ** 2
    
    def _calculate_efficiency_reward(self, execution_time: float, memory_usage: float) -> float:
        """Calculate efficiency-based reward"""
        # Normalize and invert (lower is better)
        time_score = max(0.0, 1.0 - (execution_time - 1000) / 10000)  # Normalize around 1s baseline
        memory_score = max(0.0, 1.0 - (memory_usage - 512) / 2048)   # Normalize around 512MB baseline
        
        return (time_score + memory_score) / 2.0
    
    def _calculate_complexity_reward(self, complexity_score: float) -> float:
        """Calculate complexity-based reward (penalty for high complexity)"""
        # Penalty for high complexity
        target_complexity = 5.0
        if complexity_score <= target_complexity:
            return 1.0
        else:
            return max(0.0, 1.0 - (complexity_score - target_complexity) / 10.0)
    
    def _calculate_stability_reward(self, confidence: float) -> float:
        """Calculate stability reward based on prediction confidence"""
        return confidence


class ImprovementReward(RewardFunction):
    """Reward based on improvement over previous architecture"""
    
    def calculate_reward(self, 
                        current_architecture: RecipeArchitecture,
                        previous_architecture: Optional[RecipeArchitecture],
                        action: RecipeAction,
                        performance_prediction: PerformancePrediction,
                        context: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        
        components = RewardComponents()
        
        if previous_architecture is None:
            # No previous architecture to compare
            return 0.0, components
        
        # Get previous performance from context
        previous_performance = context.get('previous_performance')
        if previous_performance is None:
            return 0.0, components
        
        # Calculate improvement in each metric
        performance_improvement = (
            performance_prediction.success_probability - 
            previous_performance.success_probability
        )
        
        efficiency_improvement = self._calculate_efficiency_improvement(
            performance_prediction, previous_performance
        )
        
        complexity_improvement = (
            previous_performance.complexity_score - 
            performance_prediction.complexity_score
        )  # Positive if complexity decreased
        
        # Assign rewards
        components.performance = max(-1.0, min(1.0, performance_improvement * 5.0))
        components.efficiency = max(-1.0, min(1.0, efficiency_improvement))
        components.complexity = max(-1.0, min(1.0, complexity_improvement * 0.2))
        
        total_reward = components.total_reward()
        
        return total_reward, components
    
    def _calculate_efficiency_improvement(self, current: PerformancePrediction, 
                                        previous: PerformancePrediction) -> float:
        """Calculate improvement in efficiency metrics"""
        time_improvement = (previous.execution_time_ms - current.execution_time_ms) / previous.execution_time_ms
        memory_improvement = (previous.memory_usage_mb - current.memory_usage_mb) / previous.memory_usage_mb
        
        return (time_improvement + memory_improvement) / 2.0


class NoveltyReward(RewardFunction):
    """Reward for exploring novel architectures"""
    
    def __init__(self, novelty_threshold: float = 0.7):
        self.novelty_threshold = novelty_threshold
        self.architecture_history = []
    
    def calculate_reward(self, 
                        current_architecture: RecipeArchitecture,
                        previous_architecture: Optional[RecipeArchitecture],
                        action: RecipeAction,
                        performance_prediction: PerformancePrediction,
                        context: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        
        components = RewardComponents()
        
        # Calculate novelty based on similarity to previous architectures
        novelty_score = self._calculate_novelty(current_architecture)
        components.novelty = novelty_score
        
        # Update history
        self._update_history(current_architecture)
        
        total_reward = components.total_reward({'novelty': 1.0})
        
        return total_reward, components
    
    def _calculate_novelty(self, architecture: RecipeArchitecture) -> float:
        """Calculate novelty score based on architecture history"""
        if not self.architecture_history:
            return 1.0  # First architecture is novel
        
        # Calculate similarity to all previous architectures
        similarities = []
        for prev_arch in self.architecture_history:
            similarity = self._calculate_architecture_similarity(architecture, prev_arch)
            similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity
        
        # Bonus for highly novel architectures
        if novelty > self.novelty_threshold:
            novelty *= 1.5
        
        return min(1.0, novelty)
    
    def _calculate_architecture_similarity(self, arch1: RecipeArchitecture, 
                                         arch2: RecipeArchitecture) -> float:
        """Calculate similarity between two architectures"""
        # Simple similarity based on node and edge counts
        node_diff = abs(len(arch1.nodes) - len(arch2.nodes))
        edge_diff = abs(len(arch1.edges) - len(arch2.edges))
        
        max_nodes = max(len(arch1.nodes), len(arch2.nodes), 1)
        max_edges = max(len(arch1.edges), len(arch2.edges), 1)
        
        node_similarity = 1.0 - (node_diff / max_nodes)
        edge_similarity = 1.0 - (edge_diff / max_edges)
        
        return (node_similarity + edge_similarity) / 2.0
    
    def _update_history(self, architecture: RecipeArchitecture):
        """Update architecture history"""
        self.architecture_history.append(architecture)
        
        # Keep limited history
        if len(self.architecture_history) > 100:
            self.architecture_history = self.architecture_history[-100:]


class ConstraintReward(RewardFunction):
    """Reward for satisfying architectural constraints"""
    
    def __init__(self, max_nodes: int = 30, max_edges: int = 60, 
                 max_complexity: float = 15.0):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_complexity = max_complexity
    
    def calculate_reward(self, 
                        current_architecture: RecipeArchitecture,
                        previous_architecture: Optional[RecipeArchitecture],
                        action: RecipeAction,
                        performance_prediction: PerformancePrediction,
                        context: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        
        components = RewardComponents()
        
        # Check constraint violations
        violations = []
        
        if len(current_architecture.nodes) > self.max_nodes:
            violations.append(f"Too many nodes: {len(current_architecture.nodes)}")
        
        if len(current_architecture.edges) > self.max_edges:
            violations.append(f"Too many edges: {len(current_architecture.edges)}")
        
        if performance_prediction.complexity_score > self.max_complexity:
            violations.append(f"Too complex: {performance_prediction.complexity_score}")
        
        # Calculate constraint satisfaction reward
        if not violations:
            components.constraint_satisfaction = 1.0
        else:
            # Penalty proportional to number of violations
            penalty = len(violations) * 0.5
            components.constraint_satisfaction = max(-1.0, -penalty)
        
        total_reward = components.total_reward({'constraint_satisfaction': 1.0})
        
        return total_reward, components


class RobustnessReward(RewardFunction):
    """Reward for robust architectures that handle errors well"""
    
    def calculate_reward(self, 
                        current_architecture: RecipeArchitecture,
                        previous_architecture: Optional[RecipeArchitecture],
                        action: RecipeAction,
                        performance_prediction: PerformancePrediction,
                        context: Dict[str, Any]) -> Tuple[float, RewardComponents]:
        
        components = RewardComponents()
        
        # Check for robustness features
        robustness_score = 0.0
        
        # Count error handling nodes
        error_handling_nodes = 0
        for node in current_architecture.nodes:
            if hasattr(node, 'operation') and 'error' in node.operation.lower():
                error_handling_nodes += 1
        
        if error_handling_nodes > 0:
            robustness_score += 0.3
        
        # Check for retry mechanisms
        retry_nodes = 0
        for node in current_architecture.nodes:
            if hasattr(node, 'parameters') and 'retry' in str(node.parameters).lower():
                retry_nodes += 1
        
        if retry_nodes > 0:
            robustness_score += 0.2
        
        # Check for fallback paths (multiple edges from decision nodes)
        decision_nodes_with_fallbacks = 0
        for node in current_architecture.nodes:
            if hasattr(node, 'node_type') and 'decision' in str(node.node_type).lower():
                outgoing_edges = [e for e in current_architecture.edges 
                                if hasattr(e, 'source_node') and e.source_node == node.id]
                if len(outgoing_edges) > 1:
                    decision_nodes_with_fallbacks += 1
        
        if decision_nodes_with_fallbacks > 0:
            robustness_score += 0.3
        
        # Check for timeout configurations
        timeout_configured = any(
            hasattr(node, 'parameters') and 'timeout' in str(node.parameters).lower()
            for node in current_architecture.nodes
        )
        
        if timeout_configured:
            robustness_score += 0.2
        
        components.robustness = min(1.0, robustness_score)
        
        total_reward = components.total_reward({'robustness': 1.0})
        
        return total_reward, components


class RewardCalculator:
    """
    Main reward calculator that combines multiple reward functions.
    
    Orchestrates different reward components and provides configurable
    weighting for multi-objective optimization.
    """
    
    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        self.reward_weights = reward_weights or {
            'performance': 0.4,
            'improvement': 0.25,
            'novelty': 0.15,
            'constraint': 0.1,
            'robustness': 0.1
        }
        
        # Initialize reward functions
        self.reward_functions = {
            'performance': PerformanceReward(),
            'improvement': ImprovementReward(),
            'novelty': NoveltyReward(),
            'constraint': ConstraintReward(),
            'robustness': RobustnessReward()
        }
        
        # Reward history for analysis
        self.reward_history = []
    
    def calculate_total_reward(self, 
                              current_architecture: RecipeArchitecture,
                              previous_architecture: Optional[RecipeArchitecture],
                              action: RecipeAction,
                              performance_prediction: PerformancePrediction,
                              context: Dict[str, Any]) -> Tuple[float, Dict[str, RewardComponents]]:
        """
        Calculate total reward from all reward functions.
        
        Args:
            current_architecture: Current recipe architecture
            previous_architecture: Previous architecture
            action: Action that was taken
            performance_prediction: Predicted performance
            context: Additional context
            
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        total_reward = 0.0
        reward_breakdown = {}
        
        # Calculate reward from each function
        for name, reward_function in self.reward_functions.items():
            try:
                reward, components = reward_function.calculate_reward(
                    current_architecture, previous_architecture, action,
                    performance_prediction, context
                )
                
                weight = self.reward_weights.get(name, 0.0)
                weighted_reward = reward * weight
                total_reward += weighted_reward
                
                reward_breakdown[name] = components
                
            except Exception as e:
                logger.warning(f"Reward calculation failed for {name}: {e}")
                reward_breakdown[name] = RewardComponents()
        
        # Record in history
        self.reward_history.append({
            'total_reward': total_reward,
            'breakdown': reward_breakdown,
            'action_type': action.action_type if hasattr(action, 'action_type') else 'unknown'
        })
        
        # Keep limited history
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        
        return total_reward, reward_breakdown
    
    def update_reward_weights(self, new_weights: Dict[str, float]):
        """Update reward function weights"""
        self.reward_weights.update(new_weights)
        logger.info(f"Updated reward weights: {self.reward_weights}")
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward history"""
        if not self.reward_history:
            return {"error": "No reward history available"}
        
        total_rewards = [entry['total_reward'] for entry in self.reward_history]
        
        stats = {
            "total_episodes": len(self.reward_history),
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "min_reward": np.min(total_rewards),
            "max_reward": np.max(total_rewards),
            "recent_mean": np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards),
            "reward_weights": self.reward_weights.copy()
        }
        
        return stats
    
    def analyze_reward_trends(self) -> Dict[str, Any]:
        """Analyze trends in reward components"""
        if len(self.reward_history) < 10:
            return {"error": "Insufficient reward history for trend analysis"}
        
        # Extract component trends
        component_trends = {}
        
        for component in RewardComponent:
            component_values = []
            for entry in self.reward_history:
                total_component_value = 0.0
                count = 0
                
                for reward_func_name, components in entry['breakdown'].items():
                    if hasattr(components, component.value):
                        total_component_value += getattr(components, component.value)
                        count += 1
                
                if count > 0:
                    component_values.append(total_component_value / count)
            
            if component_values:
                component_trends[component.value] = {
                    "mean": np.mean(component_values),
                    "trend": "improving" if len(component_values) > 5 and 
                            np.mean(component_values[-5:]) > np.mean(component_values[:5]) else "stable"
                }
        
        return component_trends
