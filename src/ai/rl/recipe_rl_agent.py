"""
Recipe RL Agent

Main reinforcement learning agent for recipe optimization.
Combines policy networks, experience replay, and environment interaction.
"""

import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

try:
    from .recipe_environment import RecipeEnvironment, EnvironmentObservation
    from .action_space import ActionSpace, RecipeAction, ActionType
    from .reward_functions import RewardCalculator
    from .experience_replay import ExperienceReplay, Experience
    from .policy_networks import PolicyNetwork, ValueNetwork, ActorCriticNetwork, NetworkConfig
    from ..nas.architecture_encoder import RecipeArchitecture
    from ..nas.search_space import SearchSpace
    from ..nas.performance_predictor import PerformancePredictor
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    
    @dataclass
    class RecipeEnvironment:
        def reset(self): return None
        async def step(self, action): return None, 0.0, False, {}
    
    @dataclass
    class RecipeArchitecture:
        nodes: List[Any] = field(default_factory=list)
        edges: List[Any] = field(default_factory=list)

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL agent"""
    # Algorithm settings
    algorithm: str = "ppo"  # ppo, a2c, dqn
    learning_rate: float = 0.0003
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    
    # Training settings
    batch_size: int = 64
    update_frequency: int = 2048
    epochs_per_update: int = 10
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Exploration settings
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    
    # Experience replay settings
    replay_buffer_size: int = 100000
    replay_buffer_type: str = "uniform"  # uniform, prioritized
    min_replay_size: int = 1000
    
    # Network settings
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Training limits
    max_episodes: int = 1000
    max_steps_per_episode: int = 100
    target_reward: float = 10.0
    
    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 100
    evaluate_interval: int = 50


@dataclass
class TrainingResult:
    """Result of RL training"""
    total_episodes: int
    total_steps: int
    training_time_seconds: float
    final_reward: float
    best_reward: float
    convergence_episode: Optional[int]
    reward_history: List[float]
    loss_history: List[float]
    exploration_history: List[float]
    best_architecture: Optional[RecipeArchitecture]


class RecipeRLAgent:
    """
    Main reinforcement learning agent for recipe optimization.
    
    Implements PPO (Proximal Policy Optimization) algorithm with
    support for other RL algorithms like A2C and DQN.
    """
    
    def __init__(self, 
                 environment: RecipeEnvironment,
                 config: Optional[RLConfig] = None):
        
        self.environment = environment
        self.config = config or RLConfig()
        
        # Get environment dimensions
        env_info = self.environment.get_environment_info()
        self.observation_dim = env_info['observation_dim']
        self.action_dim = env_info['action_space_size']
        
        # Initialize networks
        if self.config.algorithm == "ppo":
            self.policy_network = ActorCriticNetwork(
                self.observation_dim, self.action_dim, self.config.network_config
            )
        elif self.config.algorithm == "a2c":
            self.policy_network = PolicyNetwork(
                self.observation_dim, self.action_dim, self.config.network_config
            )
            self.value_network = ValueNetwork(
                self.observation_dim, self.config.network_config
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        # Initialize experience replay
        self.experience_replay = ExperienceReplay(
            buffer_type=self.config.replay_buffer_type,
            capacity=self.config.replay_buffer_size
        )
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.training_start_time = None
        self.is_training = False
        
        # Training history
        self.reward_history = []
        self.loss_history = []
        self.exploration_history = []
        self.best_reward = float('-inf')
        self.best_architecture = None
        
        # Current exploration rate
        self.current_exploration_rate = self.config.exploration_rate
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
    
    async def train(self, 
                   initial_architectures: Optional[List[RecipeArchitecture]] = None,
                   progress_callback: Optional[Callable] = None) -> TrainingResult:
        """
        Train the RL agent to optimize recipes.
        
        Args:
            initial_architectures: Optional starting architectures
            progress_callback: Optional callback for progress updates
            
        Returns:
            Training result with statistics and best architecture
        """
        self.training_start_time = time.time()
        self.is_training = True
        
        if progress_callback:
            self.progress_callbacks.append(progress_callback)
        
        try:
            logger.info(f"Starting RL training with {self.config.algorithm} algorithm")
            
            # Training loop
            while (self.episode_count < self.config.max_episodes and 
                   self.best_reward < self.config.target_reward):
                
                # Run episode
                episode_reward, episode_steps = await self._run_episode(initial_architectures)
                
                # Update statistics
                self.reward_history.append(episode_reward)
                self.exploration_history.append(self.current_exploration_rate)
                
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    # Store best architecture from environment
                    if hasattr(self.environment, 'current_architecture'):
                        self.best_architecture = self.environment.current_architecture
                
                # Train networks if enough experience
                if self.experience_replay.is_ready_for_training(self.config.min_replay_size):
                    if self.total_steps % self.config.update_frequency == 0:
                        loss = await self._update_networks()
                        self.loss_history.append(loss)
                
                # Update exploration rate
                self._update_exploration_rate()
                
                # Progress callback
                if self.episode_count % self.config.log_interval == 0:
                    await self._call_progress_callbacks()
                
                # Evaluation
                if self.episode_count % self.config.evaluate_interval == 0:
                    await self._evaluate_agent()
                
                self.episode_count += 1
            
            # Create training result
            result = self._create_training_result()
            
            logger.info(f"RL training completed: {self.episode_count} episodes, "
                       f"best reward: {self.best_reward:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"RL training failed: {e}")
            raise
        finally:
            self.is_training = False
    
    async def _run_episode(self, initial_architectures: Optional[List[RecipeArchitecture]] = None) -> Tuple[float, int]:
        """Run a single episode"""
        # Reset environment
        initial_arch = None
        if initial_architectures and len(initial_architectures) > 0:
            initial_arch = initial_architectures[self.episode_count % len(initial_architectures)]
        
        observation = self.environment.reset(initial_arch)
        
        episode_reward = 0.0
        episode_steps = 0
        episode_experiences = []
        
        done = False
        while not done and episode_steps < self.config.max_steps_per_episode:
            # Select action
            action = await self._select_action(observation)
            
            # Take step in environment
            next_observation, reward, done, info = await self.environment.step(action)
            
            # Create experience
            experience = Experience(
                state=observation,
                action=action,
                reward=reward,
                next_state=next_observation,
                done=done,
                episode_id=self.episode_count,
                step_id=episode_steps,
                timestamp=time.time(),
                metadata=info
            )
            
            episode_experiences.append(experience)
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            observation = next_observation
        
        # Add experiences to replay buffer
        for experience in episode_experiences:
            self.experience_replay.add_experience(experience)
        
        return episode_reward, episode_steps
    
    async def _select_action(self, observation: EnvironmentObservation) -> RecipeAction:
        """Select action using current policy"""
        state_vector = observation.to_vector()
        
        # Get action probabilities from policy
        if self.config.algorithm == "ppo":
            action_probs, _ = self.policy_network.forward(state_vector)
        else:
            action_probs = self.policy_network.forward(state_vector)
        
        # Ensure action_probs is 1D
        if action_probs.ndim > 1:
            action_probs = action_probs.flatten()
        
        # Apply exploration
        if np.random.random() < self.current_exploration_rate:
            # Random action
            action_idx = np.random.randint(len(action_probs))
        else:
            # Sample from policy
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        # Convert action index to RecipeAction
        action = self._action_index_to_recipe_action(action_idx, observation)
        
        return action
    
    def _action_index_to_recipe_action(self, action_idx: int, observation: EnvironmentObservation) -> RecipeAction:
        """Convert action index to RecipeAction"""
        # Simplified mapping - in practice would be more sophisticated
        action_types = list(ActionType)
        action_type = action_types[action_idx % len(action_types)]
        
        # Create action with basic parameters
        action = RecipeAction(
            action_type=action_type,
            parameters={
                "action_index": action_idx,
                "exploration_rate": self.current_exploration_rate
            }
        )
        
        return action
    
    async def _update_networks(self) -> float:
        """Update policy and value networks"""
        # Sample batch from experience replay
        experiences, indices, weights = self.experience_replay.sample_batch(self.config.batch_size)
        
        if not experiences:
            return 0.0
        
        # Prepare training data
        states = np.array([exp.state.to_vector() for exp in experiences])
        actions = np.array([self._recipe_action_to_index(exp.action) for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state.to_vector() for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        # Calculate advantages and returns
        advantages, returns = self._calculate_advantages_and_returns(
            states, rewards, next_states, dones
        )
        
        # Prepare batch data
        batch_data = {
            'states': states,
            'actions': actions,
            'advantages': advantages,
            'returns': returns
        }
        
        # Update networks
        total_loss = 0.0
        for _ in range(self.config.epochs_per_update):
            if self.config.algorithm == "ppo":
                loss = self.policy_network.train_step(batch_data)
            else:
                # Separate policy and value updates for A2C
                policy_loss = self.policy_network.train_step(batch_data)
                value_loss = self.value_network.train_step(batch_data)
                loss = policy_loss + value_loss
            
            total_loss += loss
        
        avg_loss = total_loss / self.config.epochs_per_update
        
        # Update priorities for prioritized replay
        if self.config.replay_buffer_type == "prioritized" and indices is not None:
            # Calculate TD errors as priorities (simplified)
            td_errors = np.abs(advantages)
            self.experience_replay.update_priorities(indices, td_errors)
        
        return avg_loss
    
    def _recipe_action_to_index(self, action: RecipeAction) -> int:
        """Convert RecipeAction to action index"""
        # Simplified mapping
        action_types = list(ActionType)
        try:
            return action_types.index(action.action_type)
        except ValueError:
            return 0  # Default action
    
    def _calculate_advantages_and_returns(self, states: np.ndarray, rewards: np.ndarray,
                                        next_states: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate advantages and returns using GAE"""
        # Get value estimates
        if self.config.algorithm == "ppo":
            _, values = self.policy_network.forward(states)
            _, next_values = self.policy_network.forward(next_states)
        else:
            values = self.value_network.forward(states)
            next_values = self.value_network.forward(next_states)
        
        # Ensure values are 1D
        if values.ndim > 1:
            values = values.flatten()
        if next_values.ndim > 1:
            next_values = next_values.flatten()
        
        # Calculate TD errors
        td_targets = rewards + self.config.discount_factor * next_values * (1 - dones)
        td_errors = td_targets - values
        
        # Calculate GAE advantages
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] if not dones[t] else 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.discount_factor * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.discount_factor * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Calculate returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def _update_exploration_rate(self):
        """Update exploration rate with decay"""
        self.current_exploration_rate = max(
            self.config.min_exploration_rate,
            self.current_exploration_rate * self.config.exploration_decay
        )
    
    async def _call_progress_callbacks(self):
        """Call progress callbacks with current statistics"""
        try:
            progress_data = {
                'episode': self.episode_count,
                'total_steps': self.total_steps,
                'current_reward': self.reward_history[-1] if self.reward_history else 0.0,
                'best_reward': self.best_reward,
                'average_reward': np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(self.reward_history),
                'exploration_rate': self.current_exploration_rate,
                'replay_buffer_size': self.experience_replay.buffer.size(),
                'training_time': time.time() - self.training_start_time
            }
            
            for callback in self.progress_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress_data)
                else:
                    callback(progress_data)
                    
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
    
    async def _evaluate_agent(self):
        """Evaluate agent performance"""
        # Run evaluation episode without exploration
        old_exploration = self.current_exploration_rate
        self.current_exploration_rate = 0.0
        
        try:
            eval_reward, eval_steps = await self._run_episode()
            logger.info(f"Evaluation episode {self.episode_count}: "
                       f"reward={eval_reward:.3f}, steps={eval_steps}")
        finally:
            self.current_exploration_rate = old_exploration
    
    def _create_training_result(self) -> TrainingResult:
        """Create training result from current state"""
        training_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        # Find convergence episode (when reward stabilized)
        convergence_episode = None
        if len(self.reward_history) > 50:
            recent_rewards = self.reward_history[-50:]
            if np.std(recent_rewards) < 0.1 * np.mean(recent_rewards):
                convergence_episode = self.episode_count - 50
        
        return TrainingResult(
            total_episodes=self.episode_count,
            total_steps=self.total_steps,
            training_time_seconds=training_time,
            final_reward=self.reward_history[-1] if self.reward_history else 0.0,
            best_reward=self.best_reward,
            convergence_episode=convergence_episode,
            reward_history=self.reward_history.copy(),
            loss_history=self.loss_history.copy(),
            exploration_history=self.exploration_history.copy(),
            best_architecture=self.best_architecture
        )
    
    def save_agent(self, filepath: str):
        """Save agent state to file"""
        try:
            # Save networks
            if self.config.algorithm == "ppo":
                self.policy_network.save_model(f"{filepath}_policy.pth")
            else:
                self.policy_network.save_model(f"{filepath}_policy.pth")
                self.value_network.save_model(f"{filepath}_value.pth")
            
            # Save experience replay
            self.experience_replay.save_buffer(f"{filepath}_replay.json")
            
            # Save training state
            import json
            state_data = {
                'config': self.config.__dict__,
                'episode_count': self.episode_count,
                'total_steps': self.total_steps,
                'reward_history': self.reward_history,
                'loss_history': self.loss_history,
                'exploration_history': self.exploration_history,
                'best_reward': self.best_reward,
                'current_exploration_rate': self.current_exploration_rate
            }
            
            with open(f"{filepath}_state.json", 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info(f"Agent saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save agent: {e}")
    
    def load_agent(self, filepath: str):
        """Load agent state from file"""
        try:
            # Load networks
            if self.config.algorithm == "ppo":
                self.policy_network.load_model(f"{filepath}_policy.pth")
            else:
                self.policy_network.load_model(f"{filepath}_policy.pth")
                self.value_network.load_model(f"{filepath}_value.pth")
            
            # Load experience replay
            self.experience_replay.load_buffer(f"{filepath}_replay.json")
            
            # Load training state
            import json
            with open(f"{filepath}_state.json", 'r') as f:
                state_data = json.load(f)
            
            self.episode_count = state_data['episode_count']
            self.total_steps = state_data['total_steps']
            self.reward_history = state_data['reward_history']
            self.loss_history = state_data['loss_history']
            self.exploration_history = state_data['exploration_history']
            self.best_reward = state_data['best_reward']
            self.current_exploration_rate = state_data['current_exploration_rate']
            
            logger.info(f"Agent loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            'algorithm': self.config.algorithm,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'current_exploration_rate': self.current_exploration_rate,
            'replay_buffer_stats': self.experience_replay.get_replay_statistics(),
            'network_stats': {
                'policy_training_steps': getattr(self.policy_network, 'training_steps', 0),
                'value_training_steps': getattr(self.value_network, 'training_steps', 0) if hasattr(self, 'value_network') else 0
            },
            'performance_stats': {
                'mean_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
                'std_reward': np.std(self.reward_history) if self.reward_history else 0.0,
                'recent_mean_reward': np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else np.mean(self.reward_history) if self.reward_history else 0.0
            }
        }
