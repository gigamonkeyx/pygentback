"""
S3 Reinforcement Learning Trainer

Implements PPO-based training for the s3 search agent with minimal data requirements.
This is a simplified implementation - a full version would use proper RL frameworks.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

from .models import S3Config, S3Experience, SearchState, SearchAction

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95


class S3RLTrainer:
    """
    Reinforcement Learning trainer for s3 search agent
    
    Uses PPO (Proximal Policy Optimization) to train the search agent
    with minimal data requirements (2.4k vs 70k+ samples).
    """
    
    def __init__(self, config: S3Config, search_agent, reward_calculator):
        self.config = config
        self.search_agent = search_agent
        self.reward_calculator = reward_calculator
        self.ppo_config = PPOConfig()
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.training_rewards = []
        self.validation_scores = []
        
        # Experience buffer
        self.experience_buffer = []
        self.max_buffer_size = config.experience_buffer_size
        
        logger.info("S3 RL trainer initialized")
    
    async def train(self, training_data: List[Dict[str, Any]], 
                   validation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Train the s3 search agent using reinforcement learning
        
        Args:
            training_data: Training examples with queries
            validation_data: Optional validation data
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting RL training with {len(training_data)} examples")
        
        start_time = time.time()
        best_validation_score = 0.0
        episodes_without_improvement = 0
        
        try:
            for episode in range(self.config.training_episodes):
                self.episode_count = episode
                
                # Sample training example
                example = np.random.choice(training_data)
                query = example['query']
                
                # Run episode
                episode_reward = await self._run_episode(query, episode)
                self.training_rewards.append(episode_reward)
                
                # Update policy if we have enough experience
                if len(self.experience_buffer) >= self.config.min_buffer_size:
                    await self._update_policy()
                
                # Validation
                if validation_data and episode % self.config.eval_frequency == 0:
                    validation_score = await self._validate(validation_data)
                    self.validation_scores.append(validation_score)
                    
                    if validation_score > best_validation_score:
                        best_validation_score = validation_score
                        episodes_without_improvement = 0
                    else:
                        episodes_without_improvement += 1
                    
                    logger.info(f"Episode {episode}: reward={episode_reward:.3f}, "
                              f"validation={validation_score:.3f}")
                
                # Early stopping
                if episodes_without_improvement >= 50:
                    logger.info("Early stopping due to no improvement")
                    break
            
            training_time = time.time() - start_time
            
            # Calculate final metrics
            final_reward = np.mean(self.training_rewards[-10:]) if self.training_rewards else 0.0
            converged = episodes_without_improvement < 50
            
            results = {
                'episodes': self.episode_count + 1,
                'training_time': training_time,
                'final_reward': final_reward,
                'best_validation_score': best_validation_score,
                'converged': converged,
                'total_experiences': len(self.experience_buffer)
            }
            
            logger.info(f"Training completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'episodes': self.episode_count,
                'training_time': time.time() - start_time,
                'error': str(e)
            }
    
    async def _run_episode(self, query: str, episode: int) -> float:
        """Run a single training episode"""
        
        # Initialize episode
        episode_experiences = []
        total_reward = 0.0
        
        try:
            # Run search with the agent
            search_state = await self.search_agent.search(query)
            
            # Generate response
            response = await self._generate_response(query, search_state)
            
            # Calculate reward
            if self.reward_calculator:
                gbr_reward = await self.reward_calculator.calculate_gbr_reward(
                    query, search_state, response
                )
                reward = gbr_reward.get_composite_reward()
            else:
                # Simple reward based on search results
                reward = self._calculate_simple_reward(search_state)
            
            total_reward = reward
            
            # Create experiences from the search trajectory
            for i, action in enumerate(search_state.action_history):
                # Create state representation (simplified)
                state = self._create_state_representation(search_state, i)
                next_state = self._create_state_representation(search_state, i + 1)
                
                # Create experience
                experience = S3Experience(
                    state=state,
                    action=action,
                    reward=reward / len(search_state.action_history),  # Distribute reward
                    next_state=next_state,
                    done=(i == len(search_state.action_history) - 1),
                    episode_id=f"episode_{episode}",
                    step=i
                )
                
                episode_experiences.append(experience)
            
            # Add experiences to buffer
            self.experience_buffer.extend(episode_experiences)
            
            # Limit buffer size
            if len(self.experience_buffer) > self.max_buffer_size:
                self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]
            
            self.total_steps += len(episode_experiences)
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Episode {episode} failed: {e}")
            return 0.0
    
    def _create_state_representation(self, search_state: SearchState, step: int) -> SearchState:
        """Create state representation for RL (simplified)"""
        
        # Create a simplified state representation
        # In practice, this would be a proper state encoding
        
        if step >= len(search_state.action_history):
            return search_state
        
        # Create partial state up to step
        partial_state = SearchState(
            original_query=search_state.original_query,
            current_query=search_state.query_history[step] if step < len(search_state.query_history) else search_state.current_query,
            iteration=step
        )
        
        # Add partial action history
        if step > 0:
            partial_state.action_history = search_state.action_history[:step]
            partial_state.query_history = search_state.query_history[:step]
        
        return partial_state
    
    def _calculate_simple_reward(self, search_state: SearchState) -> float:
        """Calculate simple reward when GBR calculator is not available"""
        
        # Reward based on search efficiency and results
        efficiency_reward = search_state.search_efficiency
        relevance_reward = search_state.average_relevance
        diversity_reward = min(1.0, search_state.unique_documents / 5.0)
        
        # Penalty for too many iterations
        iteration_penalty = max(0.0, 1.0 - (search_state.iteration / self.config.max_search_iterations))
        
        total_reward = (
            0.4 * relevance_reward +
            0.3 * efficiency_reward +
            0.2 * diversity_reward +
            0.1 * iteration_penalty
        )
        
        return total_reward
    
    async def _generate_response(self, query: str, search_state: SearchState) -> str:
        """Generate response for reward calculation"""
        
        if not search_state.documents:
            return "No relevant documents found."
        
        # Simple response generation (placeholder)
        top_docs = search_state.get_top_documents(k=3)
        
        response_parts = [f"Based on the search results for '{query}':"]
        
        for i, doc in enumerate(top_docs):
            content = doc.get('content', '')[:200]  # Truncate
            response_parts.append(f"{i+1}. {content}")
        
        return "\n".join(response_parts)
    
    async def _update_policy(self) -> None:
        """Update policy using PPO (simplified implementation)"""
        
        if len(self.experience_buffer) < self.ppo_config.batch_size:
            return
        
        try:
            # Sample batch
            batch_size = min(self.ppo_config.batch_size, len(self.experience_buffer))
            batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
            batch = [self.experience_buffer[i] for i in batch_indices]
            
            # In a real implementation, this would:
            # 1. Compute advantages using GAE
            # 2. Update policy network using PPO loss
            # 3. Update value network using MSE loss
            # 4. Apply gradient clipping
            
            # For now, just log the update
            avg_reward = np.mean([exp.reward for exp in batch])
            logger.debug(f"Policy update: batch_size={batch_size}, avg_reward={avg_reward:.3f}")
            
        except Exception as e:
            logger.error(f"Policy update failed: {e}")
    
    async def _validate(self, validation_data: List[Dict[str, Any]]) -> float:
        """Validate current policy on validation data"""
        
        if not validation_data:
            return 0.0
        
        try:
            # Sample validation examples
            sample_size = min(self.config.eval_episodes, len(validation_data))
            validation_sample = np.random.choice(validation_data, sample_size, replace=False)
            
            validation_rewards = []
            
            for example in validation_sample:
                query = example['query']
                
                # Run search
                search_state = await self.search_agent.search(query)
                
                # Calculate reward
                if self.reward_calculator:
                    response = await self._generate_response(query, search_state)
                    gbr_reward = await self.reward_calculator.calculate_gbr_reward(
                        query, search_state, response
                    )
                    reward = gbr_reward.get_composite_reward()
                else:
                    reward = self._calculate_simple_reward(search_state)
                
                validation_rewards.append(reward)
            
            return np.mean(validation_rewards)
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return 0.0
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        
        stats = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'experience_buffer_size': len(self.experience_buffer),
            'training_rewards': self.training_rewards,
            'validation_scores': self.validation_scores
        }
        
        if self.training_rewards:
            stats.update({
                'average_reward': np.mean(self.training_rewards),
                'best_reward': np.max(self.training_rewards),
                'recent_average_reward': np.mean(self.training_rewards[-10:])
            })
        
        if self.validation_scores:
            stats.update({
                'average_validation': np.mean(self.validation_scores),
                'best_validation': np.max(self.validation_scores),
                'recent_validation': self.validation_scores[-1] if self.validation_scores else 0.0
            })
        
        return stats
    
    def reset_training(self) -> None:
        """Reset training state"""
        self.episode_count = 0
        self.total_steps = 0
        self.training_rewards.clear()
        self.validation_scores.clear()
        self.experience_buffer.clear()
        
        logger.info("Training state reset")
