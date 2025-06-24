"""
Recipe Environment for Reinforcement Learning

Implements the environment interface for RL agents to interact with
and optimize recipe architectures.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import copy

try:
    from ..nas.architecture_encoder import RecipeArchitecture, ArchitectureEncoder
    from ..nas.performance_predictor import PerformancePredictor, PerformancePrediction
    from ..nas.search_space import SearchSpace
    from .action_space import ActionSpace, RecipeAction
    from .reward_functions import RewardCalculator, RewardComponents
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
    
    class ActionSpace:
        def get_valid_actions(self, arch): return []
        def apply_action(self, arch, action): return arch
        def sample_random_action(self, arch): return None
    
    class RewardCalculator:
        def calculate_total_reward(self, *args): return 0.0, {}

logger = logging.getLogger(__name__)


class EnvironmentState(Enum):
    """States of the RL environment"""
    INITIAL = "initial"
    ACTIVE = "active"
    TERMINAL = "terminal"
    ERROR = "error"


@dataclass
class EnvironmentAction:
    """Action taken in the environment"""
    action: RecipeAction
    timestamp: float
    valid: bool = True
    applied: bool = False


@dataclass
class EnvironmentObservation:
    """Observation from the environment"""
    architecture_features: np.ndarray
    performance_features: np.ndarray
    action_mask: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert observation to flat vector"""
        return np.concatenate([
            self.architecture_features,
            self.performance_features,
            self.action_mask
        ])


class RecipeEnvironment:
    """
    Reinforcement Learning environment for recipe optimization.
    
    Provides the standard RL interface (reset, step, render) for agents
    to interact with and optimize recipe architectures.
    """
    
    def __init__(self, 
                 search_space: Optional[SearchSpace] = None,
                 performance_predictor: Optional[PerformancePredictor] = None,
                 reward_calculator: Optional[RewardCalculator] = None,
                 max_steps: int = 100):
        
        # Initialize components
        self.search_space = search_space or SearchSpace()
        self.performance_predictor = performance_predictor or PerformancePredictor(self.search_space)
        self.reward_calculator = reward_calculator or RewardCalculator()
        
        # Environment configuration
        self.max_steps = max_steps
        self.action_space = ActionSpace()
        self.architecture_encoder = ArchitectureEncoder()
        
        # Environment state
        self.current_architecture: Optional[RecipeArchitecture] = None
        self.previous_architecture: Optional[RecipeArchitecture] = None
        self.current_performance: Optional[PerformancePrediction] = None
        self.previous_performance: Optional[PerformancePrediction] = None
        
        self.state = EnvironmentState.INITIAL
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        # History tracking
        self.episode_history = []
        self.action_history = []
        self.reward_history = []
        
        # Observation space dimensions
        self.architecture_feature_dim = 50  # Features from architecture
        self.performance_feature_dim = 10   # Features from performance prediction
        self.action_mask_dim = self.action_space.get_action_space_size() if hasattr(self.action_space, 'get_action_space_size') else 20
        
        self.observation_dim = (self.architecture_feature_dim + 
                               self.performance_feature_dim + 
                               self.action_mask_dim)
    
    def reset(self, initial_architecture: Optional[RecipeArchitecture] = None) -> EnvironmentObservation:
        """
        Reset the environment to initial state.
        
        Args:
            initial_architecture: Optional starting architecture
            
        Returns:
            Initial observation
        """
        try:
            # Reset environment state
            self.state = EnvironmentState.ACTIVE
            self.step_count = 0
            self.total_reward = 0.0
            self.previous_architecture = None
            self.previous_performance = None
            
            # Initialize architecture
            if initial_architecture is not None:
                self.current_architecture = copy.deepcopy(initial_architecture)
            else:
                self.current_architecture = self.search_space.generate_random_architecture()
            
            # Get initial performance prediction
            self.current_performance = self._predict_performance(self.current_architecture)
            
            # Clear history for new episode
            self.action_history = []
            self.reward_history = []
            
            self.episode_count += 1
            
            logger.debug(f"Environment reset for episode {self.episode_count}")
            
            return self._get_observation()
            
        except Exception as e:
            logger.error(f"Environment reset failed: {e}")
            self.state = EnvironmentState.ERROR
            return self._get_error_observation()
    
    async def step(self, action: RecipeAction) -> Tuple[EnvironmentObservation, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        try:
            if self.state != EnvironmentState.ACTIVE:
                raise ValueError(f"Environment not active: {self.state}")
            
            self.step_count += 1
            
            # Store previous state
            self.previous_architecture = copy.deepcopy(self.current_architecture)
            self.previous_performance = self.current_performance
            
            # Validate and apply action
            action_valid = self.action_space.is_action_valid(self.current_architecture, action)
            
            if action_valid:
                # Apply action to architecture
                self.current_architecture = self.action_space.apply_action(
                    self.current_architecture, action
                )
                
                # Get new performance prediction
                self.current_performance = self._predict_performance(self.current_architecture)
                
                # Calculate reward
                reward, reward_components = self.reward_calculator.calculate_total_reward(
                    self.current_architecture,
                    self.previous_architecture,
                    action,
                    self.current_performance,
                    {'previous_performance': self.previous_performance}
                )
                
            else:
                # Invalid action penalty
                reward = -0.5
                reward_components = {}
                logger.warning(f"Invalid action attempted: {action.action_type}")
            
            # Record action and reward
            env_action = EnvironmentAction(
                action=action,
                timestamp=self.step_count,
                valid=action_valid,
                applied=action_valid
            )
            
            self.action_history.append(env_action)
            self.reward_history.append(reward)
            self.total_reward += reward
            
            # Check if episode is done
            done = self._is_episode_done()
            
            if done:
                self.state = EnvironmentState.TERMINAL
                self._finalize_episode()
            
            # Create observation
            observation = self._get_observation()
            
            # Create info dict
            info = {
                'step': self.step_count,
                'episode': self.episode_count,
                'action_valid': action_valid,
                'reward_components': reward_components,
                'performance': {
                    'success_probability': self.current_performance.success_probability,
                    'execution_time': self.current_performance.execution_time_ms,
                    'memory_usage': self.current_performance.memory_usage_mb,
                    'complexity': self.current_performance.complexity_score
                },
                'architecture_stats': {
                    'nodes': len(self.current_architecture.nodes),
                    'edges': len(self.current_architecture.edges)
                }
            }
            
            return observation, reward, done, info
            
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            self.state = EnvironmentState.ERROR
            return self._get_error_observation(), -1.0, True, {'error': str(e)}
    
    def _predict_performance(self, architecture: RecipeArchitecture) -> PerformancePrediction:
        """Predict performance for an architecture"""
        try:
            # This would be async in practice, but simplified for now
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            prediction = loop.run_until_complete(
                self.performance_predictor.predict_performance(architecture)
            )
            loop.close()
            return prediction
        except Exception as e:
            logger.warning(f"Performance prediction failed: {e}")
            # Return default prediction
            return PerformancePrediction(
                success_probability=0.5,
                execution_time_ms=1000,
                memory_usage_mb=512,
                complexity_score=5.0,
                confidence=0.1
            )
    
    def _get_observation(self) -> EnvironmentObservation:
        """Get current environment observation"""
        try:
            # Extract architecture features
            architecture_features = self._extract_architecture_features()
            
            # Extract performance features
            performance_features = self._extract_performance_features()
            
            # Create action mask
            action_mask = self._create_action_mask()
            
            return EnvironmentObservation(
                architecture_features=architecture_features,
                performance_features=performance_features,
                action_mask=action_mask,
                metadata={
                    'step': self.step_count,
                    'episode': self.episode_count,
                    'state': self.state.value
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create observation: {e}")
            return self._get_error_observation()
    
    def _extract_architecture_features(self) -> np.ndarray:
        """Extract features from current architecture"""
        features = np.zeros(self.architecture_feature_dim)
        
        if self.current_architecture is None:
            return features
        
        try:
            # Basic architecture statistics
            features[0] = len(self.current_architecture.nodes) / 30.0  # Normalized node count
            features[1] = len(self.current_architecture.edges) / 60.0  # Normalized edge count
            
            # Node type distribution
            node_types = {}
            for node in self.current_architecture.nodes:
                node_type = getattr(node, 'node_type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            features[2] = node_types.get('processing', 0) / max(len(self.current_architecture.nodes), 1)
            features[3] = node_types.get('agent', 0) / max(len(self.current_architecture.nodes), 1)
            features[4] = node_types.get('mcp_tool', 0) / max(len(self.current_architecture.nodes), 1)
            features[5] = node_types.get('decision', 0) / max(len(self.current_architecture.nodes), 1)
            
            # Edge type distribution
            edge_types = {}
            for edge in self.current_architecture.edges:
                edge_type = getattr(edge, 'edge_type', 'unknown')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            features[6] = edge_types.get('data_flow', 0) / max(len(self.current_architecture.edges), 1)
            features[7] = edge_types.get('control_flow', 0) / max(len(self.current_architecture.edges), 1)
            features[8] = edge_types.get('dependency', 0) / max(len(self.current_architecture.edges), 1)
            
            # Complexity metrics
            features[9] = self.architecture_encoder.calculate_architecture_complexity(self.current_architecture)
            
            # Connectivity metrics
            if len(self.current_architecture.nodes) > 1:
                max_edges = len(self.current_architecture.nodes) * (len(self.current_architecture.nodes) - 1)
                features[10] = len(self.current_architecture.edges) / max_edges  # Connectivity ratio
            
            # Fill remaining features with random architectural properties
            for i in range(11, self.architecture_feature_dim):
                features[i] = random.random() * 0.1  # Small random values
            
        except Exception as e:
            logger.warning(f"Architecture feature extraction failed: {e}")
        
        return features
    
    def _extract_performance_features(self) -> np.ndarray:
        """Extract features from performance prediction"""
        features = np.zeros(self.performance_feature_dim)
        
        if self.current_performance is None:
            return features
        
        try:
            features[0] = self.current_performance.success_probability
            features[1] = min(1.0, self.current_performance.execution_time_ms / 10000.0)  # Normalized
            features[2] = min(1.0, self.current_performance.memory_usage_mb / 2048.0)    # Normalized
            features[3] = min(1.0, self.current_performance.complexity_score / 20.0)     # Normalized
            features[4] = self.current_performance.confidence
            
            # Risk factors as binary features
            risk_factors = getattr(self.current_performance, 'risk_factors', [])
            features[5] = 1.0 if 'high_complexity' in risk_factors else 0.0
            features[6] = 1.0 if 'large_architecture' in risk_factors else 0.0
            features[7] = 1.0 if 'high_connectivity' in risk_factors else 0.0
            
            # Performance improvement (if previous performance available)
            if self.previous_performance:
                features[8] = (self.current_performance.success_probability - 
                              self.previous_performance.success_probability)
                features[9] = (self.previous_performance.execution_time_ms - 
                              self.current_performance.execution_time_ms) / 1000.0
            
        except Exception as e:
            logger.warning(f"Performance feature extraction failed: {e}")
        
        return features
    
    def _create_action_mask(self) -> np.ndarray:
        """Create mask for valid actions"""
        mask = np.zeros(self.action_mask_dim)
        
        if self.current_architecture is None:
            return mask
        
        try:
            # Get valid actions
            valid_actions = self.action_space.get_valid_actions(self.current_architecture)
            
            # Create mask based on action types
            action_type_counts = {}
            for action in valid_actions:
                action_type = action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type)
                action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
            
            # Map action types to mask indices (simplified)
            action_type_mapping = {
                'add_node': 0, 'remove_node': 1, 'modify_node': 2,
                'add_edge': 3, 'remove_edge': 4, 'modify_edge': 5,
                'change_parameter': 6, 'reorder_steps': 7,
                'split_node': 8, 'merge_nodes': 9,
                'add_parallel_branch': 10, 'add_conditional': 11
            }
            
            for action_type, count in action_type_counts.items():
                idx = action_type_mapping.get(action_type, -1)
                if 0 <= idx < len(mask):
                    mask[idx] = min(1.0, count / 10.0)  # Normalized count
            
        except Exception as e:
            logger.warning(f"Action mask creation failed: {e}")
        
        return mask
    
    def _is_episode_done(self) -> bool:
        """Check if episode should terminate"""
        # Maximum steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # High performance achieved
        if (self.current_performance and 
            self.current_performance.success_probability >= 0.95):
            return True
        
        # Architecture becomes invalid
        if self.current_architecture is None:
            return True
        
        # Too many constraint violations
        if (self.current_performance and 
            len(getattr(self.current_performance, 'risk_factors', [])) > 3):
            return True
        
        return False
    
    def _finalize_episode(self):
        """Finalize episode and record statistics"""
        episode_data = {
            'episode': self.episode_count,
            'steps': self.step_count,
            'total_reward': self.total_reward,
            'final_performance': {
                'success_probability': self.current_performance.success_probability,
                'execution_time': self.current_performance.execution_time_ms,
                'memory_usage': self.current_performance.memory_usage_mb,
                'complexity': self.current_performance.complexity_score
            },
            'final_architecture': {
                'nodes': len(self.current_architecture.nodes),
                'edges': len(self.current_architecture.edges)
            },
            'actions_taken': len(self.action_history),
            'valid_actions': sum(1 for a in self.action_history if a.valid)
        }
        
        self.episode_history.append(episode_data)
        
        # Keep limited history
        if len(self.episode_history) > 1000:
            self.episode_history = self.episode_history[-1000:]
        
        logger.info(f"Episode {self.episode_count} completed: "
                   f"{self.step_count} steps, reward: {self.total_reward:.3f}, "
                   f"final performance: {self.current_performance.success_probability:.3f}")
    
    def _get_error_observation(self) -> EnvironmentObservation:
        """Get observation for error state"""
        return EnvironmentObservation(
            architecture_features=np.zeros(self.architecture_feature_dim),
            performance_features=np.zeros(self.performance_feature_dim),
            action_mask=np.zeros(self.action_mask_dim),
            metadata={'error': True}
        )
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Render the environment state"""
        if mode == 'human':
            print(f"\n=== Recipe Environment State ===")
            print(f"Episode: {self.episode_count}, Step: {self.step_count}")
            print(f"State: {self.state.value}")
            print(f"Total Reward: {self.total_reward:.3f}")
            
            if self.current_architecture:
                print(f"Architecture: {len(self.current_architecture.nodes)} nodes, "
                      f"{len(self.current_architecture.edges)} edges")
            
            if self.current_performance:
                print(f"Performance: {self.current_performance.success_probability:.3f} success, "
                      f"{self.current_performance.execution_time_ms:.0f}ms, "
                      f"{self.current_performance.memory_usage_mb:.0f}MB")
            
            print("=" * 35)
            
        elif mode == 'rgb_array':
            # Could return visualization as numpy array
            return None
        
        return None
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the environment"""
        return {
            'observation_dim': self.observation_dim,
            'action_space_size': self.action_space.get_action_space_size() if hasattr(self.action_space, 'get_action_space_size') else 20,
            'max_steps': self.max_steps,
            'episode_count': self.episode_count,
            'current_state': self.state.value,
            'architecture_feature_dim': self.architecture_feature_dim,
            'performance_feature_dim': self.performance_feature_dim,
            'action_mask_dim': self.action_mask_dim
        }
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics about completed episodes"""
        if not self.episode_history:
            return {"error": "No completed episodes"}
        
        rewards = [ep['total_reward'] for ep in self.episode_history]
        steps = [ep['steps'] for ep in self.episode_history]
        final_performances = [ep['final_performance']['success_probability'] 
                            for ep in self.episode_history]
        
        return {
            "total_episodes": len(self.episode_history),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_steps": np.mean(steps),
            "mean_final_performance": np.mean(final_performances),
            "best_episode_reward": max(rewards),
            "best_episode_performance": max(final_performances),
            "recent_mean_reward": np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        }
