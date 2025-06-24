"""
Action Space for Recipe RL

Defines the action space for reinforcement learning agents
to modify and optimize recipe structures.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import random

try:
    from ...testing.recipes.schema import RecipeDefinition
    from ..nas.architecture_encoder import RecipeArchitecture, ArchitectureNode, ArchitectureEdge, NodeType, EdgeType
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    from enum import Enum
    
    class NodeType(Enum):
        PROCESSING = "processing"
        AGENT = "agent"
        MCP_TOOL = "mcp_tool"
    
    class EdgeType(Enum):
        DATA_FLOW = "data_flow"
        DEPENDENCY = "dependency"
    
    @dataclass
    class RecipeDefinition:
        name: str = ""
    
    @dataclass
    class RecipeArchitecture:
        nodes: List[Any] = field(default_factory=list)
        edges: List[Any] = field(default_factory=list)

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the RL agent can take"""
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    MODIFY_NODE = "modify_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    MODIFY_EDGE = "modify_edge"
    CHANGE_PARAMETER = "change_parameter"
    REORDER_STEPS = "reorder_steps"
    SPLIT_NODE = "split_node"
    MERGE_NODES = "merge_nodes"
    ADD_PARALLEL_BRANCH = "add_parallel_branch"
    ADD_CONDITIONAL = "add_conditional"


@dataclass
class RecipeAction:
    """Action that can be taken on a recipe"""
    action_type: ActionType
    target_node_id: Optional[str] = None
    target_edge_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert action to numerical vector representation"""
        # Create one-hot encoding for action type
        action_vector = np.zeros(len(ActionType))
        action_vector[list(ActionType).index(self.action_type)] = 1.0
        
        # Add parameter values (simplified)
        param_vector = np.zeros(10)  # Fixed size for parameters
        if self.parameters:
            for i, (key, value) in enumerate(list(self.parameters.items())[:10]):
                if isinstance(value, (int, float)):
                    param_vector[i] = float(value)
                elif isinstance(value, bool):
                    param_vector[i] = 1.0 if value else 0.0
                else:
                    param_vector[i] = hash(str(value)) % 100 / 100.0  # Normalize hash
        
        return np.concatenate([action_vector, param_vector])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'RecipeAction':
        """Create action from numerical vector representation"""
        action_types = list(ActionType)
        action_type_idx = np.argmax(vector[:len(ActionType)])
        action_type = action_types[action_type_idx]
        
        # Extract parameters (simplified)
        param_vector = vector[len(ActionType):len(ActionType)+10]
        parameters = {}
        for i, value in enumerate(param_vector):
            if abs(value) > 0.01:  # Non-zero threshold
                parameters[f"param_{i}"] = float(value)
        
        return cls(
            action_type=action_type,
            parameters=parameters
        )


class ActionSpace:
    """
    Defines the action space for recipe reinforcement learning.
    
    Provides methods to generate valid actions, apply actions to recipes,
    and validate action feasibility.
    """
    
    def __init__(self, max_nodes: int = 30, max_edges: int = 60):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        # Action constraints
        self.action_constraints = {
            ActionType.ADD_NODE: {"max_nodes": max_nodes},
            ActionType.REMOVE_NODE: {"min_nodes": 3},  # Keep minimum viable recipe
            ActionType.ADD_EDGE: {"max_edges": max_edges},
            ActionType.REMOVE_EDGE: {"min_edges": 2}   # Keep minimum connectivity
        }
        
        # Available node operations
        self.node_operations = [
            "text_generation", "code_generation", "data_analysis",
            "sentiment_analysis", "format_conversion", "validation",
            "aggregation", "routing", "error_handling"
        ]
        
        # Available parameter modifications
        self.parameter_modifications = [
            "timeout", "retry_count", "memory_limit", "parallel_execution",
            "critical_flag", "temperature", "max_tokens", "threshold"
        ]
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space"""
        return len(ActionType) + 10  # Action type + parameter vector
    
    def get_valid_actions(self, architecture: RecipeArchitecture) -> List[RecipeAction]:
        """Get list of valid actions for the current architecture"""
        valid_actions = []
        
        # Add node actions
        if len(architecture.nodes) < self.max_nodes:
            for operation in self.node_operations:
                action = RecipeAction(
                    action_type=ActionType.ADD_NODE,
                    parameters={
                        "operation": operation,
                        "node_type": "processing",
                        "position": (len(architecture.nodes), 1)
                    }
                )
                valid_actions.append(action)
        
        # Remove node actions (except input/output)
        removable_nodes = [
            node for node in architecture.nodes 
            if node.node_type not in [NodeType.INPUT, NodeType.OUTPUT] if hasattr(node, 'node_type')
        ]
        
        if len(removable_nodes) > 0 and len(architecture.nodes) > 3:
            for node in removable_nodes[:5]:  # Limit to prevent explosion
                action = RecipeAction(
                    action_type=ActionType.REMOVE_NODE,
                    target_node_id=node.id if hasattr(node, 'id') else f"node_{random.randint(0, 1000)}"
                )
                valid_actions.append(action)
        
        # Modify node actions
        for node in architecture.nodes[:5]:  # Limit to prevent explosion
            for operation in self.node_operations:
                action = RecipeAction(
                    action_type=ActionType.MODIFY_NODE,
                    target_node_id=node.id if hasattr(node, 'id') else f"node_{random.randint(0, 1000)}",
                    parameters={"new_operation": operation}
                )
                valid_actions.append(action)
        
        # Add edge actions
        if len(architecture.edges) < self.max_edges:
            for i, source_node in enumerate(architecture.nodes[:3]):  # Limit combinations
                for j, target_node in enumerate(architecture.nodes[:3]):
                    if i != j:  # No self-loops
                        action = RecipeAction(
                            action_type=ActionType.ADD_EDGE,
                            parameters={
                                "source_node": source_node.id if hasattr(source_node, 'id') else f"node_{i}",
                                "target_node": target_node.id if hasattr(target_node, 'id') else f"node_{j}",
                                "edge_type": "data_flow",
                                "weight": 1.0
                            }
                        )
                        valid_actions.append(action)
        
        # Remove edge actions
        if len(architecture.edges) > 2:
            for edge in architecture.edges[:5]:  # Limit to prevent explosion
                action = RecipeAction(
                    action_type=ActionType.REMOVE_EDGE,
                    target_edge_id=edge.id if hasattr(edge, 'id') else f"edge_{random.randint(0, 1000)}"
                )
                valid_actions.append(action)
        
        # Parameter change actions
        for node in architecture.nodes[:3]:  # Limit to prevent explosion
            for param_name in self.parameter_modifications:
                action = RecipeAction(
                    action_type=ActionType.CHANGE_PARAMETER,
                    target_node_id=node.id if hasattr(node, 'id') else f"node_{random.randint(0, 1000)}",
                    parameters={
                        "parameter_name": param_name,
                        "parameter_value": random.uniform(0.1, 2.0)
                    }
                )
                valid_actions.append(action)
        
        # Advanced actions
        if len(architecture.nodes) < self.max_nodes - 2:
            # Add parallel branch
            action = RecipeAction(
                action_type=ActionType.ADD_PARALLEL_BRANCH,
                parameters={
                    "branch_operation": random.choice(self.node_operations),
                    "merge_strategy": "concatenate"
                }
            )
            valid_actions.append(action)
            
            # Add conditional
            action = RecipeAction(
                action_type=ActionType.ADD_CONDITIONAL,
                parameters={
                    "condition_type": "threshold",
                    "threshold": 0.5,
                    "true_operation": random.choice(self.node_operations),
                    "false_operation": random.choice(self.node_operations)
                }
            )
            valid_actions.append(action)
        
        return valid_actions
    
    def sample_random_action(self, architecture: RecipeArchitecture) -> RecipeAction:
        """Sample a random valid action"""
        valid_actions = self.get_valid_actions(architecture)
        if not valid_actions:
            # Fallback: parameter change action
            return RecipeAction(
                action_type=ActionType.CHANGE_PARAMETER,
                parameters={"parameter_name": "timeout", "parameter_value": random.uniform(30, 300)}
            )
        
        return random.choice(valid_actions)
    
    def apply_action(self, architecture: RecipeArchitecture, action: RecipeAction) -> RecipeArchitecture:
        """Apply an action to an architecture and return the modified architecture"""
        try:
            # Create a copy of the architecture
            new_nodes = [node for node in architecture.nodes]
            new_edges = [edge for edge in architecture.edges]
            new_metadata = architecture.metadata.copy() if hasattr(architecture, 'metadata') else {}
            
            if action.action_type == ActionType.ADD_NODE:
                new_node = self._create_new_node(action, len(new_nodes))
                new_nodes.append(new_node)
                
            elif action.action_type == ActionType.REMOVE_NODE:
                new_nodes = [node for node in new_nodes 
                           if (hasattr(node, 'id') and node.id != action.target_node_id)]
                # Remove associated edges
                new_edges = [edge for edge in new_edges 
                           if (hasattr(edge, 'source_node') and edge.source_node != action.target_node_id and
                               hasattr(edge, 'target_node') and edge.target_node != action.target_node_id)]
                
            elif action.action_type == ActionType.MODIFY_NODE:
                for i, node in enumerate(new_nodes):
                    if hasattr(node, 'id') and node.id == action.target_node_id:
                        new_nodes[i] = self._modify_node(node, action)
                        break
                        
            elif action.action_type == ActionType.ADD_EDGE:
                new_edge = self._create_new_edge(action, len(new_edges))
                new_edges.append(new_edge)
                
            elif action.action_type == ActionType.REMOVE_EDGE:
                new_edges = [edge for edge in new_edges 
                           if not (hasattr(edge, 'id') and edge.id == action.target_edge_id)]
                
            elif action.action_type == ActionType.CHANGE_PARAMETER:
                for i, node in enumerate(new_nodes):
                    if hasattr(node, 'id') and node.id == action.target_node_id:
                        new_nodes[i] = self._change_node_parameter(node, action)
                        break
                        
            elif action.action_type == ActionType.ADD_PARALLEL_BRANCH:
                parallel_nodes, parallel_edges = self._create_parallel_branch(action, len(new_nodes))
                new_nodes.extend(parallel_nodes)
                new_edges.extend(parallel_edges)
                
            elif action.action_type == ActionType.ADD_CONDITIONAL:
                conditional_nodes, conditional_edges = self._create_conditional(action, len(new_nodes))
                new_nodes.extend(conditional_nodes)
                new_edges.extend(conditional_edges)
            
            # Create new architecture
            new_architecture = RecipeArchitecture(
                nodes=new_nodes,
                edges=new_edges,
                metadata={**new_metadata, "last_action": action.action_type.value}
            )
            
            return new_architecture
            
        except Exception as e:
            logger.warning(f"Failed to apply action {action.action_type}: {e}")
            return architecture  # Return original if action fails
    
    def _create_new_node(self, action: RecipeAction, node_index: int):
        """Create a new node from action parameters"""
        # This is a simplified node creation - in practice would use proper node classes
        return type('Node', (), {
            'id': f"node_{node_index}",
            'node_type': action.parameters.get('node_type', 'processing'),
            'operation': action.parameters.get('operation', 'default_op'),
            'parameters': action.parameters.copy(),
            'position': action.parameters.get('position', (node_index, 1))
        })()
    
    def _modify_node(self, node, action: RecipeAction):
        """Modify an existing node"""
        # Create a modified copy of the node
        modified_node = type('Node', (), {
            'id': node.id if hasattr(node, 'id') else f"node_{random.randint(0, 1000)}",
            'node_type': getattr(node, 'node_type', 'processing'),
            'operation': action.parameters.get('new_operation', getattr(node, 'operation', 'default_op')),
            'parameters': {**getattr(node, 'parameters', {}), **action.parameters},
            'position': getattr(node, 'position', (0, 1))
        })()
        
        return modified_node
    
    def _create_new_edge(self, action: RecipeAction, edge_index: int):
        """Create a new edge from action parameters"""
        return type('Edge', (), {
            'id': f"edge_{edge_index}",
            'source_node': action.parameters.get('source_node', 'node_0'),
            'target_node': action.parameters.get('target_node', 'node_1'),
            'edge_type': action.parameters.get('edge_type', 'data_flow'),
            'weight': action.parameters.get('weight', 1.0),
            'parameters': action.parameters.copy()
        })()
    
    def _change_node_parameter(self, node, action: RecipeAction):
        """Change a parameter of an existing node"""
        new_parameters = getattr(node, 'parameters', {}).copy()
        param_name = action.parameters.get('parameter_name', 'timeout')
        param_value = action.parameters.get('parameter_value', 60)
        new_parameters[param_name] = param_value
        
        modified_node = type('Node', (), {
            'id': node.id if hasattr(node, 'id') else f"node_{random.randint(0, 1000)}",
            'node_type': getattr(node, 'node_type', 'processing'),
            'operation': getattr(node, 'operation', 'default_op'),
            'parameters': new_parameters,
            'position': getattr(node, 'position', (0, 1))
        })()
        
        return modified_node
    
    def _create_parallel_branch(self, action: RecipeAction, start_index: int) -> Tuple[List, List]:
        """Create nodes and edges for a parallel branch"""
        branch_node = type('Node', (), {
            'id': f"parallel_{start_index}",
            'node_type': 'parallel',
            'operation': action.parameters.get('branch_operation', 'parallel_op'),
            'parameters': action.parameters.copy(),
            'position': (start_index, 2)
        })()
        
        merge_node = type('Node', (), {
            'id': f"merge_{start_index}",
            'node_type': 'processing',
            'operation': 'merge_op',
            'parameters': {"strategy": action.parameters.get('merge_strategy', 'concatenate')},
            'position': (start_index + 1, 1)
        })()
        
        branch_edge = type('Edge', (), {
            'id': f"branch_edge_{start_index}",
            'source_node': f"parallel_{start_index}",
            'target_node': f"merge_{start_index}",
            'edge_type': 'data_flow',
            'weight': 1.0
        })()
        
        return [branch_node, merge_node], [branch_edge]
    
    def _create_conditional(self, action: RecipeAction, start_index: int) -> Tuple[List, List]:
        """Create nodes and edges for a conditional structure"""
        condition_node = type('Node', (), {
            'id': f"condition_{start_index}",
            'node_type': 'decision',
            'operation': 'conditional_op',
            'parameters': {
                "condition_type": action.parameters.get('condition_type', 'threshold'),
                "threshold": action.parameters.get('threshold', 0.5)
            },
            'position': (start_index, 1)
        })()
        
        true_node = type('Node', (), {
            'id': f"true_{start_index}",
            'node_type': 'processing',
            'operation': action.parameters.get('true_operation', 'true_op'),
            'parameters': {},
            'position': (start_index + 1, 0)
        })()
        
        false_node = type('Node', (), {
            'id': f"false_{start_index}",
            'node_type': 'processing',
            'operation': action.parameters.get('false_operation', 'false_op'),
            'parameters': {},
            'position': (start_index + 1, 2)
        })()
        
        true_edge = type('Edge', (), {
            'id': f"true_edge_{start_index}",
            'source_node': f"condition_{start_index}",
            'target_node': f"true_{start_index}",
            'edge_type': 'conditional',
            'weight': 1.0
        })()
        
        false_edge = type('Edge', (), {
            'id': f"false_edge_{start_index}",
            'source_node': f"condition_{start_index}",
            'target_node': f"false_{start_index}",
            'edge_type': 'conditional',
            'weight': 1.0
        })()
        
        return [condition_node, true_node, false_node], [true_edge, false_edge]
    
    def is_action_valid(self, architecture: RecipeArchitecture, action: RecipeAction) -> bool:
        """Check if an action is valid for the given architecture"""
        try:
            # Check basic constraints
            if action.action_type == ActionType.ADD_NODE:
                return len(architecture.nodes) < self.max_nodes
            
            elif action.action_type == ActionType.REMOVE_NODE:
                return len(architecture.nodes) > 3  # Minimum viable recipe
            
            elif action.action_type == ActionType.ADD_EDGE:
                return len(architecture.edges) < self.max_edges
            
            elif action.action_type == ActionType.REMOVE_EDGE:
                return len(architecture.edges) > 2  # Minimum connectivity
            
            elif action.action_type in [ActionType.MODIFY_NODE, ActionType.CHANGE_PARAMETER]:
                # Check if target node exists
                if action.target_node_id:
                    return any(hasattr(node, 'id') and node.id == action.target_node_id 
                             for node in architecture.nodes)
                return True
            
            elif action.action_type == ActionType.REMOVE_EDGE:
                # Check if target edge exists
                if action.target_edge_id:
                    return any(hasattr(edge, 'id') and edge.id == action.target_edge_id 
                             for edge in architecture.edges)
                return True
            
            else:
                return True  # Other actions are generally valid
                
        except Exception as e:
            logger.warning(f"Action validation failed: {e}")
            return False
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space"""
        return {
            "action_types": [action_type.value for action_type in ActionType],
            "action_space_size": self.get_action_space_size(),
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "node_operations": self.node_operations,
            "parameter_modifications": self.parameter_modifications,
            "constraints": self.action_constraints
        }
