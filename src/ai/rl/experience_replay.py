"""
Experience Replay for Recipe RL

Implements experience replay buffer and sampling strategies
for reinforcement learning agents.
"""

import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import pickle
import json

try:
    from .recipe_environment import EnvironmentObservation
    from .action_space import RecipeAction
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    
    @dataclass
    class EnvironmentObservation:
        architecture_features: np.ndarray = field(default_factory=lambda: np.zeros(10))
        performance_features: np.ndarray = field(default_factory=lambda: np.zeros(5))
        action_mask: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    @dataclass
    class RecipeAction:
        action_type: str = "modify"

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple for replay buffer"""
    state: EnvironmentObservation
    action: RecipeAction
    reward: float
    next_state: EnvironmentObservation
    done: bool
    episode_id: int
    step_id: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience to dictionary for serialization"""
        return {
            'state_vector': self.state.to_vector().tolist(),
            'action_vector': self.action.to_vector().tolist() if hasattr(self.action, 'to_vector') else [],
            'reward': self.reward,
            'next_state_vector': self.next_state.to_vector().tolist(),
            'done': self.done,
            'episode_id': self.episode_id,
            'step_id': self.step_id,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create experience from dictionary"""
        # This is simplified - in practice would need proper reconstruction
        state = EnvironmentObservation(
            architecture_features=np.array(data['state_vector'][:50]),
            performance_features=np.array(data['state_vector'][50:60]),
            action_mask=np.array(data['state_vector'][60:])
        )
        
        next_state = EnvironmentObservation(
            architecture_features=np.array(data['next_state_vector'][:50]),
            performance_features=np.array(data['next_state_vector'][50:60]),
            action_mask=np.array(data['next_state_vector'][60:])
        )
        
        action = RecipeAction()  # Simplified
        
        return cls(
            state=state,
            action=action,
            reward=data['reward'],
            next_state=next_state,
            done=data['done'],
            episode_id=data['episode_id'],
            step_id=data['step_id'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
        )


class PrioritizedExperience(NamedTuple):
    """Experience with priority for prioritized replay"""
    experience: Experience
    priority: float
    weight: float = 1.0


class ReplayBuffer:
    """
    Basic replay buffer for storing and sampling experiences.
    
    Implements uniform random sampling from stored experiences.
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        # Statistics
        self.total_added = 0
        self.total_sampled = 0
    
    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.total_added += 1
        
        if len(self.buffer) == self.capacity:
            logger.debug(f"Replay buffer full, overwriting oldest experiences")
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences uniformly at random"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if batch_size == 0:
            return []
        
        batch = random.sample(list(self.buffer), batch_size)
        self.total_sampled += batch_size
        
        return batch
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int = 1000) -> bool:
        """Check if buffer has enough experiences for training"""
        return len(self.buffer) >= min_size
    
    def clear(self):
        """Clear all experiences from buffer"""
        self.buffer.clear()
        self.position = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            'capacity': self.capacity,
            'current_size': len(self.buffer),
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'utilization': len(self.buffer) / self.capacity
        }


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Samples experiences based on their temporal difference (TD) error,
    giving priority to experiences that are more surprising.
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
        # Statistics
        self.total_added = 0
        self.total_sampled = 0
    
    def add(self, experience: Experience, priority: Optional[float] = None):
        """Add experience with priority"""
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        
        self.position = (self.position + 1) % self.capacity
        self.total_added += 1
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        batch_size = min(batch_size, len(self.buffer))
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        self.total_sampled += batch_size
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority + 1e-6)
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int = 1000) -> bool:
        """Check if buffer has enough experiences for training"""
        return len(self.buffer) >= min_size
    
    def anneal_beta(self, step: int, total_steps: int):
        """Anneal importance sampling exponent"""
        self.beta = min(1.0, self.beta + step * (1.0 - 0.4) / total_steps)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        priorities = self.priorities[:len(self.buffer)]
        
        return {
            'capacity': self.capacity,
            'current_size': len(self.buffer),
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'alpha': self.alpha,
            'beta': self.beta,
            'max_priority': self.max_priority,
            'mean_priority': np.mean(priorities) if len(priorities) > 0 else 0.0,
            'std_priority': np.std(priorities) if len(priorities) > 0 else 0.0
        }


class EpisodeBuffer:
    """
    Buffer for storing complete episodes.
    
    Useful for algorithms that need full episode trajectories
    like Monte Carlo methods or trajectory optimization.
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
        
        # Statistics
        self.total_episodes = 0
        self.total_steps = 0
    
    def add_step(self, experience: Experience):
        """Add step to current episode"""
        self.current_episode.append(experience)
        
        if experience.done:
            self.finish_episode()
    
    def finish_episode(self):
        """Finish current episode and add to buffer"""
        if self.current_episode:
            episode_data = {
                'experiences': self.current_episode.copy(),
                'episode_id': self.current_episode[0].episode_id if self.current_episode else 0,
                'length': len(self.current_episode),
                'total_reward': sum(exp.reward for exp in self.current_episode),
                'final_reward': self.current_episode[-1].reward if self.current_episode else 0.0
            }
            
            self.episodes.append(episode_data)
            self.total_episodes += 1
            self.total_steps += len(self.current_episode)
            
            self.current_episode = []
    
    def sample_episodes(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample complete episodes"""
        if len(self.episodes) < batch_size:
            batch_size = len(self.episodes)
        
        if batch_size == 0:
            return []
        
        return random.sample(list(self.episodes), batch_size)
    
    def sample_steps(self, batch_size: int) -> List[Experience]:
        """Sample individual steps from all episodes"""
        all_experiences = []
        for episode in self.episodes:
            all_experiences.extend(episode['experiences'])
        
        if len(all_experiences) < batch_size:
            batch_size = len(all_experiences)
        
        if batch_size == 0:
            return []
        
        return random.sample(all_experiences, batch_size)
    
    def get_best_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get episodes with highest total reward"""
        if not self.episodes:
            return []
        
        sorted_episodes = sorted(self.episodes, 
                               key=lambda ep: ep['total_reward'], 
                               reverse=True)
        
        return sorted_episodes[:n]
    
    def size(self) -> int:
        """Get number of stored episodes"""
        return len(self.episodes)
    
    def total_experiences(self) -> int:
        """Get total number of experiences across all episodes"""
        return sum(len(episode['experiences']) for episode in self.episodes)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get episode buffer statistics"""
        if not self.episodes:
            return {
                'total_episodes': 0,
                'total_steps': 0,
                'mean_episode_length': 0,
                'mean_episode_reward': 0
            }
        
        episode_lengths = [ep['length'] for ep in self.episodes]
        episode_rewards = [ep['total_reward'] for ep in self.episodes]
        
        return {
            'capacity': self.capacity,
            'total_episodes': len(self.episodes),
            'total_steps': sum(episode_lengths),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'mean_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'best_episode_reward': max(episode_rewards),
            'worst_episode_reward': min(episode_rewards)
        }


class ExperienceReplay:
    """
    Main experience replay system that combines different buffer types.
    
    Provides unified interface for experience storage and sampling
    with support for different replay strategies.
    """
    
    def __init__(self, 
                 buffer_type: str = "uniform",
                 capacity: int = 100000,
                 prioritized_alpha: float = 0.6,
                 prioritized_beta: float = 0.4):
        
        self.buffer_type = buffer_type
        
        # Initialize appropriate buffer
        if buffer_type == "uniform":
            self.buffer = ReplayBuffer(capacity)
        elif buffer_type == "prioritized":
            self.buffer = PrioritizedReplayBuffer(capacity, prioritized_alpha, prioritized_beta)
        elif buffer_type == "episode":
            self.buffer = EpisodeBuffer(capacity)
        else:
            raise ValueError(f"Unknown buffer type: {buffer_type}")
        
        # Additional buffers for multi-buffer strategies
        self.episode_buffer = EpisodeBuffer(capacity // 10)  # Smaller episode buffer
        
        # Replay statistics
        self.replay_stats = {
            'total_experiences': 0,
            'successful_samples': 0,
            'failed_samples': 0
        }
    
    def add_experience(self, experience: Experience, priority: Optional[float] = None):
        """Add experience to replay buffer"""
        try:
            if self.buffer_type == "prioritized":
                self.buffer.add(experience, priority)
            else:
                self.buffer.add(experience)
            
            # Also add to episode buffer
            self.episode_buffer.add_step(experience)
            
            self.replay_stats['total_experiences'] += 1
            
        except Exception as e:
            logger.error(f"Failed to add experience: {e}")
    
    def sample_batch(self, batch_size: int) -> Tuple[List[Experience], Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample batch of experiences"""
        try:
            if self.buffer_type == "prioritized":
                experiences, indices, weights = self.buffer.sample(batch_size)
                self.replay_stats['successful_samples'] += len(experiences)
                return experiences, indices, weights
            else:
                experiences = self.buffer.sample(batch_size)
                self.replay_stats['successful_samples'] += len(experiences)
                return experiences, None, None
                
        except Exception as e:
            logger.error(f"Failed to sample batch: {e}")
            self.replay_stats['failed_samples'] += 1
            return [], None, None
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for prioritized replay"""
        if self.buffer_type == "prioritized" and hasattr(self.buffer, 'update_priorities'):
            self.buffer.update_priorities(indices, priorities)
    
    def sample_episodes(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample complete episodes"""
        return self.episode_buffer.sample_episodes(batch_size)
    
    def get_best_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get best performing episodes"""
        return self.episode_buffer.get_best_episodes(n)
    
    def is_ready_for_training(self, min_experiences: int = 1000) -> bool:
        """Check if enough experiences are available for training"""
        return self.buffer.is_ready(min_experiences)
    
    def save_buffer(self, filepath: str):
        """Save replay buffer to file"""
        try:
            # Convert experiences to serializable format
            if hasattr(self.buffer, 'buffer'):
                experiences_data = [exp.to_dict() for exp in self.buffer.buffer]
            else:
                experiences_data = []
            
            save_data = {
                'buffer_type': self.buffer_type,
                'experiences': experiences_data,
                'statistics': self.get_replay_statistics(),
                'buffer_stats': self.buffer.get_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"Replay buffer saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save replay buffer: {e}")
    
    def load_buffer(self, filepath: str):
        """Load replay buffer from file"""
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            # Reconstruct experiences
            for exp_data in save_data['experiences']:
                experience = Experience.from_dict(exp_data)
                self.add_experience(experience)
            
            logger.info(f"Replay buffer loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load replay buffer: {e}")
    
    def get_replay_statistics(self) -> Dict[str, Any]:
        """Get comprehensive replay statistics"""
        stats = {
            'buffer_type': self.buffer_type,
            'replay_stats': self.replay_stats.copy(),
            'buffer_stats': self.buffer.get_statistics(),
            'episode_stats': self.episode_buffer.get_statistics()
        }
        
        # Add sampling efficiency
        total_samples = self.replay_stats['successful_samples'] + self.replay_stats['failed_samples']
        if total_samples > 0:
            stats['sampling_efficiency'] = self.replay_stats['successful_samples'] / total_samples
        else:
            stats['sampling_efficiency'] = 0.0
        
        return stats
    
    def clear_buffers(self):
        """Clear all replay buffers"""
        self.buffer.clear()
        self.episode_buffer = EpisodeBuffer(self.episode_buffer.capacity)
        self.replay_stats = {
            'total_experiences': 0,
            'successful_samples': 0,
            'failed_samples': 0
        }
        
        logger.info("Replay buffers cleared")
    
    def analyze_experience_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of experiences in buffer"""
        if not hasattr(self.buffer, 'buffer') or not self.buffer.buffer:
            return {"error": "No experiences in buffer"}
        
        # Analyze reward distribution
        rewards = [exp.reward for exp in self.buffer.buffer]
        
        # Analyze episode distribution
        episodes = [exp.episode_id for exp in self.buffer.buffer]
        unique_episodes = len(set(episodes))
        
        # Analyze action distribution
        action_types = []
        for exp in self.buffer.buffer:
            if hasattr(exp.action, 'action_type'):
                action_types.append(str(exp.action.action_type))
        
        action_counts = {}
        for action_type in action_types:
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        return {
            'total_experiences': len(self.buffer.buffer),
            'unique_episodes': unique_episodes,
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            },
            'action_distribution': action_counts,
            'experiences_per_episode': len(self.buffer.buffer) / max(unique_episodes, 1)
        }
