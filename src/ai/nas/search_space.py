"""
Search Space Definition for Recipe NAS

Defines the search space for neural architecture search,
including valid operations, connections, and constraints.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import random

from .architecture_encoder import NodeType, EdgeType, ArchitectureNode, ArchitectureEdge, RecipeArchitecture

logger = logging.getLogger(__name__)


class OperationCategory(Enum):
    """Categories of operations in the search space"""
    GENERATION = "generation"
    ANALYSIS = "analysis"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    AGGREGATION = "aggregation"
    ROUTING = "routing"
    CONTROL = "control"


@dataclass
class OperationSpec:
    """Specification for an operation in the search space"""
    name: str
    category: OperationCategory
    node_type: NodeType
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    complexity_score: float = 1.0
    description: str = ""


@dataclass
class ConnectionSpec:
    """Specification for valid connections in the search space"""
    source_types: Set[NodeType]
    target_types: Set[NodeType]
    edge_type: EdgeType
    weight_range: Tuple[float, float] = (0.1, 1.0)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchConstraints:
    """Constraints for the search space"""
    max_nodes: int = 50
    max_edges: int = 100
    max_depth: int = 20
    min_nodes: int = 3
    min_edges: int = 2
    max_parallel_branches: int = 5
    max_agents: int = 10
    max_mcp_tools: int = 15
    complexity_budget: float = 10.0


class SearchSpace:
    """
    Defines the search space for recipe neural architecture search.
    
    Specifies valid operations, connections, and constraints for
    generating and mutating recipe architectures.
    """
    
    def __init__(self, constraints: Optional[SearchConstraints] = None):
        self.constraints = constraints or SearchConstraints()
        
        # Initialize operation specifications
        self.operations = self._initialize_operations()
        
        # Initialize connection specifications
        self.connections = self._initialize_connections()
        
        # Operation lookup by category
        self.operations_by_category = self._group_operations_by_category()
        
        # Valid transitions between node types
        self.valid_transitions = self._initialize_valid_transitions()
    
    def _initialize_operations(self) -> List[OperationSpec]:
        """Initialize available operations in the search space"""
        operations = [
            # Generation operations
            OperationSpec(
                name="text_generation",
                category=OperationCategory.GENERATION,
                node_type=NodeType.PROCESSING,
                parameters={"max_tokens": 1000, "temperature": 0.7},
                complexity_score=2.0,
                description="Generate text content"
            ),
            OperationSpec(
                name="code_generation",
                category=OperationCategory.GENERATION,
                node_type=NodeType.PROCESSING,
                parameters={"language": "python", "style": "clean"},
                complexity_score=2.5,
                description="Generate code"
            ),
            OperationSpec(
                name="data_generation",
                category=OperationCategory.GENERATION,
                node_type=NodeType.PROCESSING,
                parameters={"format": "json", "schema": "auto"},
                complexity_score=1.8,
                description="Generate structured data"
            ),
            
            # Analysis operations
            OperationSpec(
                name="sentiment_analysis",
                category=OperationCategory.ANALYSIS,
                node_type=NodeType.PROCESSING,
                parameters={"model": "transformer", "confidence_threshold": 0.8},
                complexity_score=1.5,
                description="Analyze sentiment"
            ),
            OperationSpec(
                name="code_analysis",
                category=OperationCategory.ANALYSIS,
                node_type=NodeType.PROCESSING,
                parameters={"metrics": ["complexity", "quality"], "deep_scan": True},
                complexity_score=2.2,
                description="Analyze code quality"
            ),
            OperationSpec(
                name="data_analysis",
                category=OperationCategory.ANALYSIS,
                node_type=NodeType.PROCESSING,
                parameters={"statistical": True, "visualization": False},
                complexity_score=1.7,
                description="Analyze data patterns"
            ),
            
            # Transformation operations
            OperationSpec(
                name="format_conversion",
                category=OperationCategory.TRANSFORMATION,
                node_type=NodeType.PROCESSING,
                parameters={"input_format": "auto", "output_format": "json"},
                complexity_score=1.2,
                description="Convert data formats"
            ),
            OperationSpec(
                name="data_cleaning",
                category=OperationCategory.TRANSFORMATION,
                node_type=NodeType.PROCESSING,
                parameters={"remove_duplicates": True, "normalize": True},
                complexity_score=1.4,
                description="Clean and normalize data"
            ),
            OperationSpec(
                name="text_processing",
                category=OperationCategory.TRANSFORMATION,
                node_type=NodeType.PROCESSING,
                parameters={"tokenize": True, "lemmatize": False},
                complexity_score=1.3,
                description="Process text data"
            ),
            
            # Validation operations
            OperationSpec(
                name="schema_validation",
                category=OperationCategory.VALIDATION,
                node_type=NodeType.DECISION,
                parameters={"strict": True, "auto_fix": False},
                complexity_score=1.1,
                description="Validate data schema"
            ),
            OperationSpec(
                name="quality_check",
                category=OperationCategory.VALIDATION,
                node_type=NodeType.DECISION,
                parameters={"threshold": 0.8, "metrics": ["accuracy", "completeness"]},
                complexity_score=1.6,
                description="Check quality metrics"
            ),
            OperationSpec(
                name="business_rules",
                category=OperationCategory.VALIDATION,
                node_type=NodeType.DECISION,
                parameters={"rules_engine": "simple", "fail_fast": True},
                complexity_score=1.8,
                description="Apply business rules"
            ),
            
            # Aggregation operations
            OperationSpec(
                name="data_merge",
                category=OperationCategory.AGGREGATION,
                node_type=NodeType.PROCESSING,
                parameters={"strategy": "union", "conflict_resolution": "latest"},
                complexity_score=1.5,
                description="Merge multiple data sources"
            ),
            OperationSpec(
                name="result_combination",
                category=OperationCategory.AGGREGATION,
                node_type=NodeType.PROCESSING,
                parameters={"method": "weighted_average", "weights": "auto"},
                complexity_score=1.7,
                description="Combine results"
            ),
            
            # Routing operations
            OperationSpec(
                name="conditional_routing",
                category=OperationCategory.ROUTING,
                node_type=NodeType.DECISION,
                parameters={"condition_type": "threshold", "threshold": 0.5},
                complexity_score=2.0,
                description="Route based on conditions"
            ),
            OperationSpec(
                name="load_balancing",
                category=OperationCategory.ROUTING,
                node_type=NodeType.PARALLEL,
                parameters={"strategy": "round_robin", "max_parallel": 3},
                complexity_score=2.3,
                description="Balance load across paths"
            ),
            
            # Control operations
            OperationSpec(
                name="loop_control",
                category=OperationCategory.CONTROL,
                node_type=NodeType.SEQUENTIAL,
                parameters={"max_iterations": 10, "break_condition": "convergence"},
                complexity_score=2.5,
                description="Control loop execution"
            ),
            OperationSpec(
                name="error_handling",
                category=OperationCategory.CONTROL,
                node_type=NodeType.DECISION,
                parameters={"retry_count": 3, "fallback_strategy": "default"},
                complexity_score=1.9,
                description="Handle errors and exceptions"
            )
        ]
        
        return operations
    
    def _initialize_connections(self) -> List[ConnectionSpec]:
        """Initialize valid connection specifications"""
        connections = [
            # Data flow connections
            ConnectionSpec(
                source_types={NodeType.INPUT, NodeType.PROCESSING},
                target_types={NodeType.PROCESSING, NodeType.DECISION, NodeType.OUTPUT},
                edge_type=EdgeType.DATA_FLOW,
                weight_range=(0.5, 1.0)
            ),
            
            # Control flow connections
            ConnectionSpec(
                source_types={NodeType.DECISION, NodeType.SEQUENTIAL},
                target_types={NodeType.PROCESSING, NodeType.DECISION, NodeType.PARALLEL},
                edge_type=EdgeType.CONTROL_FLOW,
                weight_range=(0.3, 0.8)
            ),
            
            # Dependency connections
            ConnectionSpec(
                source_types={NodeType.AGENT, NodeType.MCP_TOOL},
                target_types={NodeType.PROCESSING, NodeType.DECISION},
                edge_type=EdgeType.DEPENDENCY,
                weight_range=(0.2, 0.7)
            ),
            
            # Feedback connections
            ConnectionSpec(
                source_types={NodeType.PROCESSING, NodeType.DECISION},
                target_types={NodeType.PROCESSING, NodeType.DECISION},
                edge_type=EdgeType.FEEDBACK,
                weight_range=(0.1, 0.5),
                constraints={"no_self_loops": True}
            ),
            
            # Conditional connections
            ConnectionSpec(
                source_types={NodeType.DECISION},
                target_types={NodeType.PROCESSING, NodeType.PARALLEL, NodeType.OUTPUT},
                edge_type=EdgeType.CONDITIONAL,
                weight_range=(0.4, 0.9)
            )
        ]
        
        return connections
    
    def _group_operations_by_category(self) -> Dict[OperationCategory, List[OperationSpec]]:
        """Group operations by category for efficient lookup"""
        grouped = {}
        for category in OperationCategory:
            grouped[category] = [op for op in self.operations if op.category == category]
        return grouped
    
    def _initialize_valid_transitions(self) -> Dict[NodeType, Set[NodeType]]:
        """Initialize valid transitions between node types"""
        transitions = {
            NodeType.INPUT: {NodeType.PROCESSING, NodeType.AGENT, NodeType.MCP_TOOL},
            NodeType.AGENT: {NodeType.PROCESSING, NodeType.DECISION},
            NodeType.MCP_TOOL: {NodeType.PROCESSING, NodeType.DECISION},
            NodeType.PROCESSING: {NodeType.PROCESSING, NodeType.DECISION, NodeType.PARALLEL, NodeType.OUTPUT},
            NodeType.DECISION: {NodeType.PROCESSING, NodeType.PARALLEL, NodeType.SEQUENTIAL, NodeType.OUTPUT},
            NodeType.PARALLEL: {NodeType.PROCESSING, NodeType.DECISION, NodeType.OUTPUT},
            NodeType.SEQUENTIAL: {NodeType.PROCESSING, NodeType.DECISION, NodeType.OUTPUT},
            NodeType.OUTPUT: set()  # Output nodes don't connect to anything
        }
        
        return transitions
    
    def sample_operation(self, node_type: Optional[NodeType] = None, 
                        category: Optional[OperationCategory] = None) -> OperationSpec:
        """Sample a random operation from the search space"""
        candidates = self.operations
        
        # Filter by node type
        if node_type:
            candidates = [op for op in candidates if op.node_type == node_type]
        
        # Filter by category
        if category:
            candidates = [op for op in candidates if op.category == category]
        
        if not candidates:
            # Fallback to any operation
            candidates = self.operations
        
        return random.choice(candidates)
    
    def sample_connection(self, source_type: NodeType, target_type: NodeType) -> Optional[ConnectionSpec]:
        """Sample a valid connection between node types"""
        valid_connections = [
            conn for conn in self.connections
            if source_type in conn.source_types and target_type in conn.target_types
        ]
        
        if not valid_connections:
            return None
        
        return random.choice(valid_connections)
    
    def generate_random_architecture(self) -> RecipeArchitecture:
        """Generate a random architecture within the search space"""
        nodes = []
        edges = []
        
        # Generate random number of nodes within constraints
        num_nodes = random.randint(self.constraints.min_nodes, 
                                 min(self.constraints.max_nodes, 20))
        
        # Always start with input node
        input_node = ArchitectureNode(
            id="input_0",
            node_type=NodeType.INPUT,
            operation="input_op",
            position=(0, 0)
        )
        nodes.append(input_node)
        
        # Generate processing nodes
        for i in range(1, num_nodes - 1):
            # Sample operation
            operation = self.sample_operation()
            
            node = ArchitectureNode(
                id=f"node_{i}",
                node_type=operation.node_type,
                operation=operation.name,
                parameters=operation.parameters.copy(),
                position=(i, random.randint(0, 3))
            )
            nodes.append(node)
        
        # Always end with output node
        output_node = ArchitectureNode(
            id="output_0",
            node_type=NodeType.OUTPUT,
            operation="output_op",
            position=(num_nodes - 1, 0)
        )
        nodes.append(output_node)
        
        # Generate edges
        edges = self._generate_random_edges(nodes)
        
        architecture = RecipeArchitecture(
            nodes=nodes,
            edges=edges,
            metadata={"generated": "random", "search_space": "default"}
        )
        
        return architecture
    
    def _generate_random_edges(self, nodes: List[ArchitectureNode]) -> List[ArchitectureEdge]:
        """Generate random edges for the given nodes"""
        edges = []
        edge_id = 0
        
        # Ensure basic connectivity (input -> ... -> output)
        for i in range(len(nodes) - 1):
            source_node = nodes[i]
            target_node = nodes[i + 1]
            
            connection_spec = self.sample_connection(source_node.node_type, target_node.node_type)
            if connection_spec:
                weight = random.uniform(*connection_spec.weight_range)
                edge = ArchitectureEdge(
                    id=f"edge_{edge_id}",
                    source_node=source_node.id,
                    target_node=target_node.id,
                    edge_type=connection_spec.edge_type,
                    weight=weight
                )
                edges.append(edge)
                edge_id += 1
        
        # Add some random additional connections
        max_additional = min(10, self.constraints.max_edges - len(edges))
        for _ in range(random.randint(0, max_additional)):
            source_node = random.choice(nodes[:-1])  # Don't start from output
            target_node = random.choice(nodes[1:])   # Don't end at input
            
            if source_node.id != target_node.id:  # No self-loops
                connection_spec = self.sample_connection(source_node.node_type, target_node.node_type)
                if connection_spec:
                    weight = random.uniform(*connection_spec.weight_range)
                    edge = ArchitectureEdge(
                        id=f"edge_{edge_id}",
                        source_node=source_node.id,
                        target_node=target_node.id,
                        edge_type=connection_spec.edge_type,
                        weight=weight
                    )
                    edges.append(edge)
                    edge_id += 1
        
        return edges
    
    def mutate_architecture(self, architecture: RecipeArchitecture, 
                          mutation_rate: float = 0.1) -> RecipeArchitecture:
        """Mutate an architecture within the search space"""
        mutated_nodes = [node for node in architecture.nodes]  # Copy nodes
        mutated_edges = [edge for edge in architecture.edges]  # Copy edges
        
        # Mutate nodes
        for i, node in enumerate(mutated_nodes):
            if random.random() < mutation_rate:
                if node.node_type not in [NodeType.INPUT, NodeType.OUTPUT]:
                    # Sample new operation for this node type
                    new_operation = self.sample_operation(node.node_type)
                    mutated_nodes[i] = ArchitectureNode(
                        id=node.id,
                        node_type=node.node_type,
                        operation=new_operation.name,
                        parameters=new_operation.parameters.copy(),
                        position=node.position,
                        metadata=node.metadata
                    )
        
        # Mutate edges
        for i, edge in enumerate(mutated_edges):
            if random.random() < mutation_rate:
                # Find source and target nodes
                source_node = next((n for n in mutated_nodes if n.id == edge.source_node), None)
                target_node = next((n for n in mutated_nodes if n.id == edge.target_node), None)
                
                if source_node and target_node:
                    connection_spec = self.sample_connection(source_node.node_type, target_node.node_type)
                    if connection_spec:
                        new_weight = random.uniform(*connection_spec.weight_range)
                        mutated_edges[i] = ArchitectureEdge(
                            id=edge.id,
                            source_node=edge.source_node,
                            target_node=edge.target_node,
                            edge_type=connection_spec.edge_type,
                            weight=new_weight,
                            parameters=edge.parameters
                        )
        
        mutated_architecture = RecipeArchitecture(
            nodes=mutated_nodes,
            edges=mutated_edges,
            metadata={**architecture.metadata, "mutated": True}
        )
        
        return mutated_architecture
    
    def validate_architecture_constraints(self, architecture: RecipeArchitecture) -> Tuple[bool, List[str]]:
        """Validate that an architecture satisfies search space constraints"""
        violations = []
        
        # Check node count
        if len(architecture.nodes) > self.constraints.max_nodes:
            violations.append(f"Too many nodes: {len(architecture.nodes)} > {self.constraints.max_nodes}")
        
        if len(architecture.nodes) < self.constraints.min_nodes:
            violations.append(f"Too few nodes: {len(architecture.nodes)} < {self.constraints.min_nodes}")
        
        # Check edge count
        if len(architecture.edges) > self.constraints.max_edges:
            violations.append(f"Too many edges: {len(architecture.edges)} > {self.constraints.max_edges}")
        
        if len(architecture.edges) < self.constraints.min_edges:
            violations.append(f"Too few edges: {len(architecture.edges)} < {self.constraints.min_edges}")
        
        # Check complexity budget
        total_complexity = sum(
            next((op.complexity_score for op in self.operations if op.name == node.operation), 1.0)
            for node in architecture.nodes
        )
        
        if total_complexity > self.constraints.complexity_budget:
            violations.append(f"Complexity budget exceeded: {total_complexity} > {self.constraints.complexity_budget}")
        
        # Check agent and MCP tool limits
        agent_count = sum(1 for node in architecture.nodes if node.node_type == NodeType.AGENT)
        mcp_count = sum(1 for node in architecture.nodes if node.node_type == NodeType.MCP_TOOL)
        
        if agent_count > self.constraints.max_agents:
            violations.append(f"Too many agents: {agent_count} > {self.constraints.max_agents}")
        
        if mcp_count > self.constraints.max_mcp_tools:
            violations.append(f"Too many MCP tools: {mcp_count} > {self.constraints.max_mcp_tools}")
        
        return len(violations) == 0, violations
    
    def get_operation_by_name(self, name: str) -> Optional[OperationSpec]:
        """Get operation specification by name"""
        for operation in self.operations:
            if operation.name == name:
                return operation
        return None
    
    def get_operations_by_category(self, category: OperationCategory) -> List[OperationSpec]:
        """Get all operations in a category"""
        return self.operations_by_category.get(category, [])
    
    def get_search_space_summary(self) -> Dict[str, Any]:
        """Get summary of the search space"""
        return {
            "total_operations": len(self.operations),
            "operations_by_category": {
                cat.value: len(ops) for cat, ops in self.operations_by_category.items()
            },
            "total_connections": len(self.connections),
            "constraints": {
                "max_nodes": self.constraints.max_nodes,
                "max_edges": self.constraints.max_edges,
                "complexity_budget": self.constraints.complexity_budget
            },
            "node_types": [nt.value for nt in NodeType],
            "edge_types": [et.value for et in EdgeType]
        }
