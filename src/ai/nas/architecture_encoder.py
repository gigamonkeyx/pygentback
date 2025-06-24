"""
Architecture Encoder for Recipe NAS

Encodes recipe architectures as neural network representations
for neural architecture search optimization.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from ...testing.recipes.schema import RecipeDefinition, RecipeStep
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    
    @dataclass
    class RecipeDefinition:
        name: str = ""
        description: str = ""
        steps: List[Any] = field(default_factory=list)
    
    @dataclass
    class RecipeStep:
        name: str = ""
        action_type: str = ""

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in recipe architecture"""
    INPUT = "input"
    AGENT = "agent"
    MCP_TOOL = "mcp_tool"
    PROCESSING = "processing"
    DECISION = "decision"
    OUTPUT = "output"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class EdgeType(Enum):
    """Types of edges in recipe architecture"""
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    DEPENDENCY = "dependency"
    FEEDBACK = "feedback"
    CONDITIONAL = "conditional"


@dataclass
class ArchitectureNode:
    """Node in recipe architecture graph"""
    id: str
    node_type: NodeType
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[int, int] = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureEdge:
    """Edge in recipe architecture graph"""
    id: str
    source_node: str
    target_node: str
    edge_type: EdgeType
    weight: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecipeArchitecture:
    """Complete recipe architecture representation"""
    nodes: List[ArchitectureNode]
    edges: List[ArchitectureEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_node_by_id(self, node_id: str) -> Optional[ArchitectureNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_edges_from_node(self, node_id: str) -> List[ArchitectureEdge]:
        """Get all edges originating from a node"""
        return [edge for edge in self.edges if edge.source_node == node_id]
    
    def get_edges_to_node(self, node_id: str) -> List[ArchitectureEdge]:
        """Get all edges targeting a node"""
        return [edge for edge in self.edges if edge.target_node == node_id]


class ArchitectureEncoder:
    """
    Encodes recipe definitions as neural network architectures.
    
    Converts recipe steps, dependencies, and flow into graph representations
    suitable for neural architecture search optimization.
    """
    
    def __init__(self, max_nodes: int = 50, max_edges: int = 100):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        # Operation mappings
        self.operation_mapping = {
            'generate': 'generation_op',
            'analyze': 'analysis_op',
            'transform': 'transformation_op',
            'validate': 'validation_op',
            'execute': 'execution_op',
            'aggregate': 'aggregation_op',
            'filter': 'filtering_op',
            'route': 'routing_op'
        }
        
        # Reverse mapping
        self.reverse_operation_mapping = {v: k for k, v in self.operation_mapping.items()}
    
    def encode_recipe(self, recipe: RecipeDefinition) -> RecipeArchitecture:
        """
        Encode a recipe definition as an architecture graph.
        
        Args:
            recipe: Recipe definition to encode
            
        Returns:
            Recipe architecture representation
        """
        try:
            nodes = []
            edges = []
            
            # Create input node
            input_node = ArchitectureNode(
                id="input_0",
                node_type=NodeType.INPUT,
                operation="input_op",
                metadata={"recipe_name": recipe.name}
            )
            nodes.append(input_node)
            
            # Process recipe steps
            step_nodes = self._encode_steps(getattr(recipe, 'steps', []))
            nodes.extend(step_nodes)
            
            # Process agent requirements
            agent_nodes = self._encode_agent_requirements(getattr(recipe, 'agent_requirements', []))
            nodes.extend(agent_nodes)
            
            # Process MCP requirements
            mcp_nodes = self._encode_mcp_requirements(getattr(recipe, 'mcp_requirements', []))
            nodes.extend(mcp_nodes)
            
            # Create output node
            output_node = ArchitectureNode(
                id="output_0",
                node_type=NodeType.OUTPUT,
                operation="output_op",
                metadata={"recipe_name": recipe.name}
            )
            nodes.append(output_node)
            
            # Create edges based on dependencies and flow
            edges = self._create_edges(nodes, recipe)
            
            # Create architecture
            architecture = RecipeArchitecture(
                nodes=nodes,
                edges=edges,
                metadata={
                    "recipe_name": recipe.name,
                    "recipe_description": getattr(recipe, 'description', ''),
                    "encoding_timestamp": np.datetime64('now').astype(str)
                }
            )
            
            logger.debug(f"Encoded recipe '{recipe.name}' to architecture with "
                        f"{len(nodes)} nodes and {len(edges)} edges")
            
            return architecture
            
        except Exception as e:
            logger.error(f"Failed to encode recipe '{recipe.name}': {e}")
            return self._create_default_architecture()
    
    def decode_architecture(self, architecture: RecipeArchitecture) -> RecipeDefinition:
        """
        Decode an architecture back to a recipe definition.
        
        Args:
            architecture: Architecture to decode
            
        Returns:
            Decoded recipe definition
        """
        try:
            # Extract recipe metadata
            recipe_name = architecture.metadata.get("recipe_name", "Generated Recipe")
            recipe_description = architecture.metadata.get("recipe_description", "AI-generated recipe")
            
            # Decode steps from nodes
            steps = self._decode_steps_from_nodes(architecture.nodes)
            
            # Decode agent requirements
            agent_requirements = self._decode_agent_requirements_from_nodes(architecture.nodes)
            
            # Decode MCP requirements
            mcp_requirements = self._decode_mcp_requirements_from_nodes(architecture.nodes)
            
            # Create recipe definition
            recipe = RecipeDefinition(
                name=recipe_name,
                description=recipe_description
            )
            
            # Set attributes if they exist
            if hasattr(recipe, 'steps'):
                recipe.steps = steps
            if hasattr(recipe, 'agent_requirements'):
                recipe.agent_requirements = agent_requirements
            if hasattr(recipe, 'mcp_requirements'):
                recipe.mcp_requirements = mcp_requirements
            
            logger.debug(f"Decoded architecture to recipe '{recipe_name}'")
            return recipe
            
        except Exception as e:
            logger.error(f"Failed to decode architecture: {e}")
            return self._create_default_recipe()
    
    def _encode_steps(self, steps: List[Any]) -> List[ArchitectureNode]:
        """Encode recipe steps as nodes"""
        nodes = []
        
        for i, step in enumerate(steps):
            step_name = getattr(step, 'name', f'step_{i}')
            action_type = getattr(step, 'action_type', 'execute')
            
            # Map action type to operation
            operation = self.operation_mapping.get(action_type, 'execution_op')
            
            # Determine node type based on operation
            if 'generate' in action_type.lower():
                node_type = NodeType.PROCESSING
            elif 'decision' in action_type.lower() or 'validate' in action_type.lower():
                node_type = NodeType.DECISION
            elif 'parallel' in action_type.lower():
                node_type = NodeType.PARALLEL
            else:
                node_type = NodeType.SEQUENTIAL
            
            node = ArchitectureNode(
                id=f"step_{i}",
                node_type=node_type,
                operation=operation,
                parameters={
                    "step_name": step_name,
                    "action_type": action_type,
                    "timeout": getattr(step, 'timeout_seconds', 60),
                    "critical": getattr(step, 'critical', True)
                },
                position=(i, 1),
                metadata={"original_step": step_name}
            )
            
            nodes.append(node)
        
        return nodes
    
    def _encode_agent_requirements(self, agent_reqs: List[Any]) -> List[ArchitectureNode]:
        """Encode agent requirements as nodes"""
        nodes = []
        
        for i, req in enumerate(agent_reqs):
            agent_type = getattr(req, 'agent_type', f'agent_{i}')
            
            node = ArchitectureNode(
                id=f"agent_{i}",
                node_type=NodeType.AGENT,
                operation="agent_op",
                parameters={
                    "agent_type": agent_type,
                    "capabilities": getattr(req, 'required_capabilities', []),
                    "memory_limit": getattr(req, 'memory_limit_mb', 512),
                    "execution_time": getattr(req, 'max_execution_time', 300)
                },
                position=(i, 0),
                metadata={"agent_type": agent_type}
            )
            
            nodes.append(node)
        
        return nodes
    
    def _encode_mcp_requirements(self, mcp_reqs: List[Any]) -> List[ArchitectureNode]:
        """Encode MCP requirements as nodes"""
        nodes = []
        
        for i, req in enumerate(mcp_reqs):
            server_name = getattr(req, 'server_name', f'mcp_{i}')
            
            node = ArchitectureNode(
                id=f"mcp_{i}",
                node_type=NodeType.MCP_TOOL,
                operation="mcp_op",
                parameters={
                    "server_name": server_name,
                    "tools_needed": getattr(req, 'tools_needed', []),
                    "timeout": getattr(req, 'timeout_seconds', 30),
                    "retry_count": getattr(req, 'retry_count', 3)
                },
                position=(i, 2),
                metadata={"server_name": server_name}
            )
            
            nodes.append(node)
        
        return nodes
    
    def _create_edges(self, nodes: List[ArchitectureNode], recipe: RecipeDefinition) -> List[ArchitectureEdge]:
        """Create edges based on node relationships and dependencies"""
        edges = []
        edge_id = 0
        
        # Find input and output nodes
        input_nodes = [n for n in nodes if n.node_type == NodeType.INPUT]
        output_nodes = [n for n in nodes if n.node_type == NodeType.OUTPUT]
        step_nodes = [n for n in nodes if n.node_type in [NodeType.PROCESSING, NodeType.DECISION, NodeType.SEQUENTIAL]]
        agent_nodes = [n for n in nodes if n.node_type == NodeType.AGENT]
        mcp_nodes = [n for n in nodes if n.node_type == NodeType.MCP_TOOL]
        
        # Connect input to first step
        if input_nodes and step_nodes:
            edge = ArchitectureEdge(
                id=f"edge_{edge_id}",
                source_node=input_nodes[0].id,
                target_node=step_nodes[0].id,
                edge_type=EdgeType.DATA_FLOW
            )
            edges.append(edge)
            edge_id += 1
        
        # Connect steps sequentially
        for i in range(len(step_nodes) - 1):
            edge = ArchitectureEdge(
                id=f"edge_{edge_id}",
                source_node=step_nodes[i].id,
                target_node=step_nodes[i + 1].id,
                edge_type=EdgeType.CONTROL_FLOW
            )
            edges.append(edge)
            edge_id += 1
        
        # Connect agents to steps
        for agent_node in agent_nodes:
            for step_node in step_nodes:
                edge = ArchitectureEdge(
                    id=f"edge_{edge_id}",
                    source_node=agent_node.id,
                    target_node=step_node.id,
                    edge_type=EdgeType.DEPENDENCY,
                    weight=0.8
                )
                edges.append(edge)
                edge_id += 1
        
        # Connect MCP tools to steps
        for mcp_node in mcp_nodes:
            for step_node in step_nodes:
                edge = ArchitectureEdge(
                    id=f"edge_{edge_id}",
                    source_node=mcp_node.id,
                    target_node=step_node.id,
                    edge_type=EdgeType.DEPENDENCY,
                    weight=0.6
                )
                edges.append(edge)
                edge_id += 1
        
        # Connect last step to output
        if step_nodes and output_nodes:
            edge = ArchitectureEdge(
                id=f"edge_{edge_id}",
                source_node=step_nodes[-1].id,
                target_node=output_nodes[0].id,
                edge_type=EdgeType.DATA_FLOW
            )
            edges.append(edge)
            edge_id += 1
        
        return edges
    
    def _decode_steps_from_nodes(self, nodes: List[ArchitectureNode]) -> List[Dict[str, Any]]:
        """Decode steps from architecture nodes"""
        steps = []
        
        step_nodes = [n for n in nodes if n.node_type in [NodeType.PROCESSING, NodeType.DECISION, NodeType.SEQUENTIAL]]
        step_nodes.sort(key=lambda n: n.position[0])  # Sort by position
        
        for node in step_nodes:
            step_data = {
                "name": node.parameters.get("step_name", node.id),
                "action_type": node.parameters.get("action_type", "execute"),
                "timeout_seconds": node.parameters.get("timeout", 60),
                "critical": node.parameters.get("critical", True)
            }
            steps.append(step_data)
        
        return steps
    
    def _decode_agent_requirements_from_nodes(self, nodes: List[ArchitectureNode]) -> List[Dict[str, Any]]:
        """Decode agent requirements from architecture nodes"""
        requirements = []
        
        agent_nodes = [n for n in nodes if n.node_type == NodeType.AGENT]
        
        for node in agent_nodes:
            req_data = {
                "agent_type": node.parameters.get("agent_type", "default"),
                "required_capabilities": node.parameters.get("capabilities", []),
                "memory_limit_mb": node.parameters.get("memory_limit", 512),
                "max_execution_time": node.parameters.get("execution_time", 300)
            }
            requirements.append(req_data)
        
        return requirements
    
    def _decode_mcp_requirements_from_nodes(self, nodes: List[ArchitectureNode]) -> List[Dict[str, Any]]:
        """Decode MCP requirements from architecture nodes"""
        requirements = []
        
        mcp_nodes = [n for n in nodes if n.node_type == NodeType.MCP_TOOL]
        
        for node in mcp_nodes:
            req_data = {
                "server_name": node.parameters.get("server_name", "default"),
                "tools_needed": node.parameters.get("tools_needed", []),
                "timeout_seconds": node.parameters.get("timeout", 30),
                "retry_count": node.parameters.get("retry_count", 3)
            }
            requirements.append(req_data)
        
        return requirements
    
    def _create_default_architecture(self) -> RecipeArchitecture:
        """Create a default architecture for error cases"""
        input_node = ArchitectureNode(
            id="input_0",
            node_type=NodeType.INPUT,
            operation="input_op"
        )
        
        output_node = ArchitectureNode(
            id="output_0",
            node_type=NodeType.OUTPUT,
            operation="output_op"
        )
        
        edge = ArchitectureEdge(
            id="edge_0",
            source_node="input_0",
            target_node="output_0",
            edge_type=EdgeType.DATA_FLOW
        )
        
        return RecipeArchitecture(
            nodes=[input_node, output_node],
            edges=[edge],
            metadata={"default": True}
        )
    
    def _create_default_recipe(self) -> RecipeDefinition:
        """Create a default recipe for error cases"""
        return RecipeDefinition(
            name="Default Recipe",
            description="Default recipe created due to decoding failure"
        )
    
    def calculate_architecture_complexity(self, architecture: RecipeArchitecture) -> float:
        """Calculate complexity score for an architecture"""
        node_count = len(architecture.nodes)
        edge_count = len(architecture.edges)
        
        # Different node types have different complexity weights
        type_weights = {
            NodeType.INPUT: 0.1,
            NodeType.OUTPUT: 0.1,
            NodeType.AGENT: 0.3,
            NodeType.MCP_TOOL: 0.2,
            NodeType.PROCESSING: 0.4,
            NodeType.DECISION: 0.5,
            NodeType.PARALLEL: 0.6,
            NodeType.SEQUENTIAL: 0.3
        }
        
        weighted_node_complexity = sum(
            type_weights.get(node.node_type, 0.3) for node in architecture.nodes
        )
        
        # Edge complexity based on type
        edge_weights = {
            EdgeType.DATA_FLOW: 0.2,
            EdgeType.CONTROL_FLOW: 0.3,
            EdgeType.DEPENDENCY: 0.1,
            EdgeType.FEEDBACK: 0.4,
            EdgeType.CONDITIONAL: 0.5
        }
        
        weighted_edge_complexity = sum(
            edge_weights.get(edge.edge_type, 0.2) for edge in architecture.edges
        )
        
        # Normalize complexity
        total_complexity = weighted_node_complexity + weighted_edge_complexity
        normalized_complexity = min(1.0, total_complexity / 20.0)  # Normalize to 0-1
        
        return normalized_complexity
    
    def validate_architecture(self, architecture: RecipeArchitecture) -> Tuple[bool, List[str]]:
        """Validate architecture for correctness"""
        errors = []
        
        # Check for input and output nodes
        input_nodes = [n for n in architecture.nodes if n.node_type == NodeType.INPUT]
        output_nodes = [n for n in architecture.nodes if n.node_type == NodeType.OUTPUT]
        
        if not input_nodes:
            errors.append("No input node found")
        if not output_nodes:
            errors.append("No output node found")
        
        # Check for disconnected nodes
        node_ids = {node.id for node in architecture.nodes}
        connected_nodes = set()
        
        for edge in architecture.edges:
            connected_nodes.add(edge.source_node)
            connected_nodes.add(edge.target_node)
        
        disconnected = node_ids - connected_nodes
        if disconnected:
            errors.append(f"Disconnected nodes: {disconnected}")
        
        # Check for invalid edge references
        for edge in architecture.edges:
            if edge.source_node not in node_ids:
                errors.append(f"Edge {edge.id} references invalid source node: {edge.source_node}")
            if edge.target_node not in node_ids:
                errors.append(f"Edge {edge.id} references invalid target node: {edge.target_node}")
        
        return len(errors) == 0, errors
