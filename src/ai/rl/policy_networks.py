"""
Policy Networks for Recipe RL

Implements neural networks for policy and value function approximation
in reinforcement learning agents.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random

# Set up UTF-8 logger
from ...utils.utf8_logger import get_pygent_logger
logger = get_pygent_logger("ai_rl_policy_networks")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using simplified networks")


@dataclass
class NetworkConfig:
    """Configuration for neural networks"""
    hidden_sizes: List[int] = None
    activation: str = "relu"
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_norm: bool = True
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128, 64]


class BaseNetwork(ABC):
    """Abstract base class for neural networks"""
    
    def __init__(self, input_dim: int, output_dim: int, config: Optional[NetworkConfig] = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or NetworkConfig()
        
        # Training statistics
        self.training_steps = 0
        self.loss_history = []
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        pass
    
    @abstractmethod
    def train_step(self, batch_data: Dict[str, np.ndarray]) -> float:
        """Perform one training step"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """Load model from file"""
        pass


if TORCH_AVAILABLE:
    class TorchPolicyNetwork(nn.Module, BaseNetwork):
        """PyTorch implementation of policy network"""
        
        def __init__(self, input_dim: int, output_dim: int, config: Optional[NetworkConfig] = None):
            nn.Module.__init__(self)
            BaseNetwork.__init__(self, input_dim, output_dim, config)
            
            # Build network layers
            layers = []
            prev_size = input_dim
            
            for hidden_size in self.config.hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                
                if self.config.batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                
                if self.config.activation == "relu":
                    layers.append(nn.ReLU())
                elif self.config.activation == "tanh":
                    layers.append(nn.Tanh())
                elif self.config.activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                
                if self.config.dropout_rate > 0:
                    layers.append(nn.Dropout(self.config.dropout_rate))
                
                prev_size = hidden_size
            
            # Output layer
            layers.append(nn.Linear(prev_size, output_dim))
            layers.append(nn.Softmax(dim=-1))  # Policy outputs probabilities
            
            self.network = nn.Sequential(*layers)
            
            # Optimizer
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Loss function
            self.criterion = nn.CrossEntropyLoss()
        
        def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
            """Forward pass through network"""
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
                return_numpy = True
            else:
                return_numpy = False
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            output = self.network(x)
            
            if return_numpy:
                return output.detach().numpy()
            else:
                return output
        
        def train_step(self, batch_data: Dict[str, np.ndarray]) -> float:
            """Perform one training step"""
            self.train()
            
            states = torch.FloatTensor(batch_data['states'])
            actions = torch.LongTensor(batch_data['actions'])
            advantages = torch.FloatTensor(batch_data['advantages'])
            
            # Forward pass
            action_probs = self.forward(states)
            
            # Calculate policy loss
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
            policy_loss = -(log_probs * advantages).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            policy_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            loss_value = policy_loss.item()
            self.training_steps += 1
            self.loss_history.append(loss_value)
            
            # Keep limited history
            if len(self.loss_history) > 1000:
                self.loss_history = self.loss_history[-1000:]
            
            return loss_value
        
        def save_model(self, filepath: str):
            """Save model to file"""
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'training_steps': self.training_steps,
                'loss_history': self.loss_history
            }, filepath)
        
        def load_model(self, filepath: str):
            """Load model from file"""
            checkpoint = torch.load(filepath)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_steps = checkpoint.get('training_steps', 0)
            self.loss_history = checkpoint.get('loss_history', [])
    
    
    class TorchValueNetwork(nn.Module, BaseNetwork):
        """PyTorch implementation of value network"""
        
        def __init__(self, input_dim: int, config: Optional[NetworkConfig] = None):
            nn.Module.__init__(self)
            BaseNetwork.__init__(self, input_dim, 1, config)  # Value network outputs single value
            
            # Build network layers
            layers = []
            prev_size = input_dim
            
            for hidden_size in self.config.hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                
                if self.config.batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                
                if self.config.activation == "relu":
                    layers.append(nn.ReLU())
                elif self.config.activation == "tanh":
                    layers.append(nn.Tanh())
                elif self.config.activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                
                if self.config.dropout_rate > 0:
                    layers.append(nn.Dropout(self.config.dropout_rate))
                
                prev_size = hidden_size
            
            # Output layer (no activation for value function)
            layers.append(nn.Linear(prev_size, 1))
            
            self.network = nn.Sequential(*layers)
            
            # Optimizer
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Loss function
            self.criterion = nn.MSELoss()
        
        def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
            """Forward pass through network"""
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
                return_numpy = True
            else:
                return_numpy = False
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            output = self.network(x)
            
            if return_numpy:
                return output.detach().numpy().squeeze()
            else:
                return output.squeeze()
        
        def train_step(self, batch_data: Dict[str, np.ndarray]) -> float:
            """Perform one training step"""
            self.train()
            
            states = torch.FloatTensor(batch_data['states'])
            targets = torch.FloatTensor(batch_data['targets'])
            
            # Forward pass
            values = self.forward(states)
            
            # Calculate value loss
            value_loss = self.criterion(values, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            value_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            loss_value = value_loss.item()
            self.training_steps += 1
            self.loss_history.append(loss_value)
            
            # Keep limited history
            if len(self.loss_history) > 1000:
                self.loss_history = self.loss_history[-1000:]
            
            return loss_value
        
        def save_model(self, filepath: str):
            """Save model to file"""
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'training_steps': self.training_steps,
                'loss_history': self.loss_history
            }, filepath)
        
        def load_model(self, filepath: str):
            """Load model from file"""
            checkpoint = torch.load(filepath)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_steps = checkpoint.get('training_steps', 0)
            self.loss_history = checkpoint.get('loss_history', [])
    
    
    class TorchActorCriticNetwork(nn.Module, BaseNetwork):
        """PyTorch implementation of Actor-Critic network"""
        
        def __init__(self, input_dim: int, action_dim: int, config: Optional[NetworkConfig] = None):
            nn.Module.__init__(self)
            BaseNetwork.__init__(self, input_dim, action_dim, config)
            
            # Shared layers
            shared_layers = []
            prev_size = input_dim
            
            for i, hidden_size in enumerate(self.config.hidden_sizes[:-1]):  # All but last layer
                shared_layers.append(nn.Linear(prev_size, hidden_size))
                
                if self.config.batch_norm:
                    shared_layers.append(nn.BatchNorm1d(hidden_size))
                
                if self.config.activation == "relu":
                    shared_layers.append(nn.ReLU())
                elif self.config.activation == "tanh":
                    shared_layers.append(nn.Tanh())
                elif self.config.activation == "leaky_relu":
                    shared_layers.append(nn.LeakyReLU())
                
                if self.config.dropout_rate > 0:
                    shared_layers.append(nn.Dropout(self.config.dropout_rate))
                
                prev_size = hidden_size
            
            self.shared_network = nn.Sequential(*shared_layers)
            
            # Actor head (policy)
            final_hidden = self.config.hidden_sizes[-1]
            self.actor_head = nn.Sequential(
                nn.Linear(prev_size, final_hidden),
                nn.ReLU(),
                nn.Linear(final_hidden, action_dim),
                nn.Softmax(dim=-1)
            )
            
            # Critic head (value function)
            self.critic_head = nn.Sequential(
                nn.Linear(prev_size, final_hidden),
                nn.ReLU(),
                nn.Linear(final_hidden, 1)
            )
            
            # Optimizer
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        def forward(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
            """Forward pass through network"""
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
                return_numpy = True
            else:
                return_numpy = False
            
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # Shared features
            shared_features = self.shared_network(x)
            
            # Actor and critic outputs
            action_probs = self.actor_head(shared_features)
            values = self.critic_head(shared_features).squeeze()
            
            if return_numpy:
                return action_probs.detach().numpy(), values.detach().numpy()
            else:
                return action_probs, values
        
        def train_step(self, batch_data: Dict[str, np.ndarray]) -> float:
            """Perform one training step"""
            self.train()
            
            states = torch.FloatTensor(batch_data['states'])
            actions = torch.LongTensor(batch_data['actions'])
            advantages = torch.FloatTensor(batch_data['advantages'])
            returns = torch.FloatTensor(batch_data['returns'])
            
            # Forward pass
            action_probs, values = self.forward(states)
            
            # Actor loss (policy gradient)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
            actor_loss = -(log_probs * advantages).mean()
            
            # Critic loss (value function)
            critic_loss = F.mse_loss(values, returns)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            loss_value = total_loss.item()
            self.training_steps += 1
            self.loss_history.append(loss_value)
            
            # Keep limited history
            if len(self.loss_history) > 1000:
                self.loss_history = self.loss_history[-1000:]
            
            return loss_value
        
        def save_model(self, filepath: str):
            """Save model to file"""
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'training_steps': self.training_steps,
                'loss_history': self.loss_history
            }, filepath)
        
        def load_model(self, filepath: str):
            """Load model from file"""
            checkpoint = torch.load(filepath)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_steps = checkpoint.get('training_steps', 0)
            self.loss_history = checkpoint.get('loss_history', [])


else:
    # Simplified implementations when PyTorch is not available
    class SimplePolicyNetwork(BaseNetwork):
        """Simplified policy network without PyTorch"""
        
        def __init__(self, input_dim: int, output_dim: int, config: Optional[NetworkConfig] = None):
            super().__init__(input_dim, output_dim, config)
            
            # Simple linear transformation
            self.weights = np.random.randn(input_dim, output_dim) * 0.1
            self.bias = np.zeros(output_dim)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Forward pass through network"""
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Linear transformation
            logits = np.dot(x, self.weights) + self.bias
            
            # Softmax activation
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            return probs
        
        def train_step(self, batch_data: Dict[str, np.ndarray]) -> float:
            """Perform one training step"""
            # Simplified training (gradient ascent on policy)
            states = batch_data['states']
            actions = batch_data['actions']
            advantages = batch_data['advantages']
            
            # Simple gradient update
            learning_rate = 0.001
            
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                advantage = advantages[i]
                
                # Simple gradient update
                probs = self.forward(state.reshape(1, -1))[0]
                
                # Update weights (simplified)
                grad = np.outer(state, np.zeros(self.output_dim))
                grad[:, action] = state * advantage * learning_rate
                
                self.weights += grad
            
            self.training_steps += 1
            return 0.1  # Dummy loss
        
        def save_model(self, filepath: str):
            """Save model to file"""
            np.savez(filepath, weights=self.weights, bias=self.bias)
        
        def load_model(self, filepath: str):
            """Load model from file"""
            data = np.load(filepath)
            self.weights = data['weights']
            self.bias = data['bias']
    
    
    class SimpleValueNetwork(BaseNetwork):
        """Simplified value network without PyTorch"""
        
        def __init__(self, input_dim: int, config: Optional[NetworkConfig] = None):
            super().__init__(input_dim, 1, config)
            
            # Simple linear transformation
            self.weights = np.random.randn(input_dim) * 0.1
            self.bias = 0.0
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Forward pass through network"""
            if x.ndim == 1:
                return np.dot(x, self.weights) + self.bias
            else:
                return np.dot(x, self.weights) + self.bias
        
        def train_step(self, batch_data: Dict[str, np.ndarray]) -> float:
            """Perform one training step"""
            states = batch_data['states']
            targets = batch_data['targets']
            
            # Simple gradient descent
            learning_rate = 0.001
            
            for i in range(len(states)):
                state = states[i]
                target = targets[i]
                
                # Prediction
                prediction = self.forward(state)
                error = target - prediction
                
                # Update weights
                self.weights += learning_rate * error * state
                self.bias += learning_rate * error
            
            self.training_steps += 1
            return 0.1  # Dummy loss
        
        def save_model(self, filepath: str):
            """Save model to file"""
            np.savez(filepath, weights=self.weights, bias=self.bias)
        
        def load_model(self, filepath: str):
            """Load model from file"""
            data = np.load(filepath)
            self.weights = data['weights']
            self.bias = data['bias']


# Factory functions
def PolicyNetwork(input_dim: int, output_dim: int, config: Optional[NetworkConfig] = None) -> BaseNetwork:
    """Create policy network"""
    if TORCH_AVAILABLE:
        return TorchPolicyNetwork(input_dim, output_dim, config)
    else:
        return SimplePolicyNetwork(input_dim, output_dim, config)


def ValueNetwork(input_dim: int, config: Optional[NetworkConfig] = None) -> BaseNetwork:
    """Create value network"""
    if TORCH_AVAILABLE:
        return TorchValueNetwork(input_dim, config)
    else:
        return SimpleValueNetwork(input_dim, config)


def ActorCriticNetwork(input_dim: int, action_dim: int, config: Optional[NetworkConfig] = None) -> BaseNetwork:
    """Create actor-critic network"""
    if TORCH_AVAILABLE:
        return TorchActorCriticNetwork(input_dim, action_dim, config)
    else:
        # Return separate networks for simplified implementation
        policy_net = SimplePolicyNetwork(input_dim, action_dim, config)
        value_net = SimpleValueNetwork(input_dim, config)
        
        # Create combined interface
        class SimpleActorCritic(BaseNetwork):
            def __init__(self):
                super().__init__(input_dim, action_dim, config)
                self.policy_net = policy_net
                self.value_net = value_net
            
            def forward(self, x):
                return self.policy_net.forward(x), self.value_net.forward(x)
            
            def train_step(self, batch_data):
                policy_loss = self.policy_net.train_step(batch_data)
                value_loss = self.value_net.train_step(batch_data)
                return policy_loss + value_loss
            
            def save_model(self, filepath):
                self.policy_net.save_model(filepath + "_policy.npz")
                self.value_net.save_model(filepath + "_value.npz")
            
            def load_model(self, filepath):
                self.policy_net.load_model(filepath + "_policy.npz")
                self.value_net.load_model(filepath + "_value.npz")
        
        return SimpleActorCritic()


def get_network_info() -> Dict[str, Any]:
    """Get information about available network implementations"""
    return {
        'torch_available': TORCH_AVAILABLE,
        'available_networks': ['PolicyNetwork', 'ValueNetwork', 'ActorCriticNetwork'],
        'supported_activations': ['relu', 'tanh', 'leaky_relu'],
        'features': {
            'batch_normalization': TORCH_AVAILABLE,
            'dropout': TORCH_AVAILABLE,
            'gradient_clipping': TORCH_AVAILABLE,
            'advanced_optimizers': TORCH_AVAILABLE
        }
    }
