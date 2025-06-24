"""
Recipe Genome Encoding/Decoding

Converts recipe definitions to genetic representations for evolutionary algorithms.
Handles encoding recipe structure, parameters, and configurations as numerical genomes.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from ...testing.recipes.schema import RecipeDefinition, RecipeCategory, RecipeDifficulty
    from ...testing.recipes.schema import AgentRequirement, MCPToolRequirement, RecipeStep
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    from enum import Enum
    
    class RecipeCategory(Enum):
        CODING = "coding"
        RESEARCH = "research"
        ANALYSIS = "analysis"
        AUTOMATION = "automation"
    
    class RecipeDifficulty(Enum):
        BASIC = "basic"
        INTERMEDIATE = "intermediate"
        ADVANCED = "advanced"
        EXPERT = "expert"
    
    @dataclass
    class RecipeDefinition:
        name: str = ""
        description: str = ""
        category: RecipeCategory = RecipeCategory.CODING
        difficulty: RecipeDifficulty = RecipeDifficulty.BASIC


logger = logging.getLogger(__name__)


@dataclass
class GenomeConfig:
    """Configuration for genome encoding parameters"""
    max_steps: int = 20
    max_agents: int = 5
    max_mcp_tools: int = 10
    max_name_length: int = 100
    max_description_length: int = 500
    parameter_precision: int = 1000  # For floating point encoding


class RecipeGenome:
    """
    Encodes recipe definitions as genetic representations for evolutionary algorithms.
    
    The genome structure:
    - Recipe metadata (category, difficulty, etc.)
    - Agent requirements (types, capabilities, configurations)
    - MCP tool requirements (servers, tools, parameters)
    - Recipe steps (actions, dependencies, parameters)
    - Performance targets and constraints
    """
    
    def __init__(self, config: Optional[GenomeConfig] = None):
        self.config = config or GenomeConfig()
        self.genome_length = self._calculate_genome_length()
        
        # Encoding mappings
        self.category_mapping = {cat: i for i, cat in enumerate(RecipeCategory)}
        self.difficulty_mapping = {diff: i for i, diff in enumerate(RecipeDifficulty)}
        
        # Reverse mappings for decoding
        self.reverse_category_mapping = {i: cat for cat, i in self.category_mapping.items()}
        self.reverse_difficulty_mapping = {i: diff for diff, i in self.difficulty_mapping.items()}
    
    def _calculate_genome_length(self) -> int:
        """Calculate total genome length based on configuration"""
        # Recipe metadata: category, difficulty, priority, timeout
        metadata_length = 4
        
        # Agent requirements: type, capability count, config parameters
        agent_length = self.config.max_agents * 5
        
        # MCP tool requirements: server type, tool count, config parameters
        mcp_length = self.config.max_mcp_tools * 4
        
        # Recipe steps: action type, dependency count, parameter count
        steps_length = self.config.max_steps * 6
        
        # Performance targets: success threshold, time budget, memory budget
        performance_length = 3
        
        return metadata_length + agent_length + mcp_length + steps_length + performance_length
    
    def encode_recipe(self, recipe: RecipeDefinition) -> List[float]:
        """
        Encode a recipe definition as a numerical genome.
        
        Args:
            recipe: Recipe definition to encode
            
        Returns:
            List of float values representing the genome
        """
        try:
            genome = []
            
            # Encode recipe metadata
            genome.extend(self._encode_metadata(recipe))
            
            # Encode agent requirements
            genome.extend(self._encode_agent_requirements(recipe))
            
            # Encode MCP tool requirements
            genome.extend(self._encode_mcp_requirements(recipe))
            
            # Encode recipe steps
            genome.extend(self._encode_steps(recipe))
            
            # Encode performance targets
            genome.extend(self._encode_performance_targets(recipe))
            
            # Pad or truncate to fixed length
            genome = self._normalize_genome_length(genome)
            
            logger.debug(f"Encoded recipe '{recipe.name}' to genome of length {len(genome)}")
            return genome
            
        except Exception as e:
            logger.error(f"Failed to encode recipe '{recipe.name}': {e}")
            return [0.0] * self.genome_length
    
    def decode_genome(self, genome: List[float]) -> RecipeDefinition:
        """
        Decode a numerical genome back to a recipe definition.
        
        Args:
            genome: List of float values representing the genome
            
        Returns:
            Decoded recipe definition
        """
        try:
            if len(genome) != self.genome_length:
                logger.warning(f"Genome length {len(genome)} doesn't match expected {self.genome_length}")
                genome = self._normalize_genome_length(genome)
            
            # Track position in genome
            pos = 0
            
            # Decode recipe metadata
            metadata, pos = self._decode_metadata(genome, pos)
            
            # Decode agent requirements
            agent_requirements, pos = self._decode_agent_requirements(genome, pos)
            
            # Decode MCP tool requirements
            mcp_requirements, pos = self._decode_mcp_requirements(genome, pos)
            
            # Decode recipe steps
            steps, pos = self._decode_steps(genome, pos)
            
            # Decode performance targets
            performance_targets, pos = self._decode_performance_targets(genome, pos)
            
            # Create recipe definition
            recipe = RecipeDefinition(
                name=metadata.get('name', 'Generated Recipe'),
                description=metadata.get('description', 'AI-generated recipe'),
                category=metadata.get('category', RecipeCategory.CODING),
                difficulty=metadata.get('difficulty', RecipeDifficulty.BASIC),
                agent_requirements=agent_requirements,
                mcp_requirements=mcp_requirements,
                steps=steps
            )
            
            # Apply performance targets if available
            if hasattr(recipe, 'validation_criteria') and performance_targets:
                recipe.validation_criteria.success_threshold = performance_targets.get('success_threshold', 0.8)
                recipe.validation_criteria.performance_budget_ms = int(performance_targets.get('time_budget', 5000))
                recipe.validation_criteria.memory_budget_mb = int(performance_targets.get('memory_budget', 1024))
            
            logger.debug(f"Decoded genome to recipe '{recipe.name}'")
            return recipe
            
        except Exception as e:
            logger.error(f"Failed to decode genome: {e}")
            return self._create_default_recipe()
    
    def _encode_metadata(self, recipe: RecipeDefinition) -> List[float]:
        """Encode recipe metadata"""
        metadata = []
        
        # Category (normalized to 0-1)
        category_idx = self.category_mapping.get(recipe.category, 0)
        metadata.append(category_idx / len(RecipeCategory))
        
        # Difficulty (normalized to 0-1)
        difficulty_idx = self.difficulty_mapping.get(recipe.difficulty, 0)
        metadata.append(difficulty_idx / len(RecipeDifficulty))
        
        # Priority (default 0.5)
        metadata.append(0.5)
        
        # Timeout (normalized, default 5000ms -> 0.5)
        timeout = getattr(recipe, 'timeout', 5000)
        metadata.append(min(timeout / 10000, 1.0))
        
        return metadata
    
    def _encode_agent_requirements(self, recipe: RecipeDefinition) -> List[float]:
        """Encode agent requirements"""
        encoded = []
        
        agent_reqs = getattr(recipe, 'agent_requirements', [])
        
        for i in range(self.config.max_agents):
            if i < len(agent_reqs):
                req = agent_reqs[i]
                # Agent type (hash normalized)
                agent_type_hash = hash(getattr(req, 'agent_type', 'default')) % 1000
                encoded.append(agent_type_hash / 1000)
                
                # Capability count
                capabilities = getattr(req, 'required_capabilities', [])
                encoded.append(min(len(capabilities) / 10, 1.0))
                
                # Memory limit (normalized)
                memory_limit = getattr(req, 'memory_limit_mb', 512)
                encoded.append(min(memory_limit / 2048, 1.0))
                
                # Execution time (normalized)
                exec_time = getattr(req, 'max_execution_time', 300)
                encoded.append(min(exec_time / 600, 1.0))
                
                # Configuration complexity (hash of config)
                config = getattr(req, 'configuration', {})
                config_hash = hash(str(config)) % 1000
                encoded.append(config_hash / 1000)
            else:
                # Empty slot
                encoded.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return encoded
    
    def _encode_mcp_requirements(self, recipe: RecipeDefinition) -> List[float]:
        """Encode MCP tool requirements"""
        encoded = []
        
        mcp_reqs = getattr(recipe, 'mcp_requirements', [])
        
        for i in range(self.config.max_mcp_tools):
            if i < len(mcp_reqs):
                req = mcp_reqs[i]
                # Server name (hash normalized)
                server_name = getattr(req, 'server_name', 'default')
                server_hash = hash(server_name) % 1000
                encoded.append(server_hash / 1000)
                
                # Tool count
                tools = getattr(req, 'tools_needed', [])
                encoded.append(min(len(tools) / 20, 1.0))
                
                # Timeout
                timeout = getattr(req, 'timeout_seconds', 30)
                encoded.append(min(timeout / 120, 1.0))
                
                # Retry count
                retry_count = getattr(req, 'retry_count', 3)
                encoded.append(min(retry_count / 10, 1.0))
            else:
                # Empty slot
                encoded.extend([0.0, 0.0, 0.0, 0.0])
        
        return encoded
    
    def _encode_steps(self, recipe: RecipeDefinition) -> List[float]:
        """Encode recipe steps"""
        encoded = []
        
        steps = getattr(recipe, 'steps', [])
        
        for i in range(self.config.max_steps):
            if i < len(steps):
                step = steps[i]
                # Action type (hash normalized)
                action = getattr(step, 'agent_action', 'default')
                action_hash = hash(action) % 1000
                encoded.append(action_hash / 1000)
                
                # MCP tools count
                mcp_tools = getattr(step, 'mcp_tools', [])
                encoded.append(min(len(mcp_tools) / 10, 1.0))
                
                # Dependencies count
                dependencies = getattr(step, 'dependencies', [])
                encoded.append(min(len(dependencies) / 5, 1.0))
                
                # Timeout
                timeout = getattr(step, 'timeout_seconds', 60)
                encoded.append(min(timeout / 300, 1.0))
                
                # Critical flag
                critical = getattr(step, 'critical', True)
                encoded.append(1.0 if critical else 0.0)
                
                # Retry flag
                retry = getattr(step, 'retry_on_failure', True)
                encoded.append(1.0 if retry else 0.0)
            else:
                # Empty slot
                encoded.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return encoded
    
    def _encode_performance_targets(self, recipe: RecipeDefinition) -> List[float]:
        """Encode performance targets"""
        encoded = []
        
        validation = getattr(recipe, 'validation_criteria', None)
        if validation:
            # Success threshold
            success_threshold = getattr(validation, 'success_threshold', 0.8)
            encoded.append(success_threshold)
            
            # Performance budget (normalized)
            perf_budget = getattr(validation, 'performance_budget_ms', 5000)
            encoded.append(min(perf_budget / 30000, 1.0))
            
            # Memory budget (normalized)
            memory_budget = getattr(validation, 'memory_budget_mb', 1024)
            encoded.append(min(memory_budget / 4096, 1.0))
        else:
            # Default values
            encoded.extend([0.8, 0.167, 0.25])  # 5s, 1GB defaults
        
        return encoded

    def _decode_metadata(self, genome: List[float], pos: int) -> Tuple[Dict[str, Any], int]:
        """Decode recipe metadata from genome"""
        metadata = {}

        # Category
        category_val = genome[pos]
        category_idx = int(category_val * len(RecipeCategory))
        metadata['category'] = self.reverse_category_mapping.get(category_idx, RecipeCategory.CODING)
        pos += 1

        # Difficulty
        difficulty_val = genome[pos]
        difficulty_idx = int(difficulty_val * len(RecipeDifficulty))
        metadata['difficulty'] = self.reverse_difficulty_mapping.get(difficulty_idx, RecipeDifficulty.BASIC)
        pos += 1

        # Skip priority and timeout for now
        pos += 2

        return metadata, pos

    def _decode_agent_requirements(self, genome: List[float], pos: int) -> Tuple[List[Any], int]:
        """Decode agent requirements from genome"""
        requirements = []

        for i in range(self.config.max_agents):
            # Check if this slot has data (non-zero values)
            slot_data = genome[pos:pos+5]
            if any(val > 0.01 for val in slot_data):  # Threshold for "active" slot
                # Create agent requirement (simplified for now)
                req_data = {
                    'agent_type': f'agent_type_{int(slot_data[0] * 1000)}',
                    'capability_count': int(slot_data[1] * 10),
                    'memory_limit': int(slot_data[2] * 2048),
                    'execution_time': int(slot_data[3] * 600)
                }
                requirements.append(req_data)

            pos += 5

        return requirements, pos

    def _decode_mcp_requirements(self, genome: List[float], pos: int) -> Tuple[List[Any], int]:
        """Decode MCP tool requirements from genome"""
        requirements = []

        for i in range(self.config.max_mcp_tools):
            # Check if this slot has data
            slot_data = genome[pos:pos+4]
            if any(val > 0.01 for val in slot_data):
                req_data = {
                    'server_name': f'server_{int(slot_data[0] * 1000)}',
                    'tool_count': int(slot_data[1] * 20),
                    'timeout': int(slot_data[2] * 120),
                    'retry_count': int(slot_data[3] * 10)
                }
                requirements.append(req_data)

            pos += 4

        return requirements, pos

    def _decode_steps(self, genome: List[float], pos: int) -> Tuple[List[Any], int]:
        """Decode recipe steps from genome"""
        steps = []

        for i in range(self.config.max_steps):
            # Check if this slot has data
            slot_data = genome[pos:pos+6]
            if any(val > 0.01 for val in slot_data):
                step_data = {
                    'action': f'action_{int(slot_data[0] * 1000)}',
                    'mcp_tool_count': int(slot_data[1] * 10),
                    'dependency_count': int(slot_data[2] * 5),
                    'timeout': int(slot_data[3] * 300),
                    'critical': slot_data[4] > 0.5,
                    'retry': slot_data[5] > 0.5
                }
                steps.append(step_data)

            pos += 6

        return steps, pos

    def _decode_performance_targets(self, genome: List[float], pos: int) -> Tuple[Dict[str, float], int]:
        """Decode performance targets from genome"""
        targets = {
            'success_threshold': genome[pos],
            'time_budget': genome[pos + 1] * 30000,
            'memory_budget': genome[pos + 2] * 4096
        }

        return targets, pos + 3

    def _normalize_genome_length(self, genome: List[float]) -> List[float]:
        """Normalize genome to expected length"""
        if len(genome) < self.genome_length:
            # Pad with zeros
            genome.extend([0.0] * (self.genome_length - len(genome)))
        elif len(genome) > self.genome_length:
            # Truncate
            genome = genome[:self.genome_length]

        return genome

    def _create_default_recipe(self) -> RecipeDefinition:
        """Create a default recipe for failed decoding"""
        return RecipeDefinition(
            name="Default Recipe",
            description="Default recipe created due to decoding failure",
            category=RecipeCategory.CODING,
            difficulty=RecipeDifficulty.BASIC
        )

    def get_genome_info(self) -> Dict[str, Any]:
        """Get information about the genome structure"""
        return {
            'genome_length': self.genome_length,
            'config': {
                'max_steps': self.config.max_steps,
                'max_agents': self.config.max_agents,
                'max_mcp_tools': self.config.max_mcp_tools,
                'parameter_precision': self.config.parameter_precision
            },
            'structure': {
                'metadata_length': 4,
                'agent_length': self.config.max_agents * 5,
                'mcp_length': self.config.max_mcp_tools * 4,
                'steps_length': self.config.max_steps * 6,
                'performance_length': 3
            }
        }

    def validate_genome(self, genome: List[float]) -> bool:
        """Validate that a genome is properly formatted"""
        if len(genome) != self.genome_length:
            return False

        # Check that all values are in valid range [0, 1]
        if not all(0.0 <= val <= 1.0 for val in genome):
            return False

        return True

    def calculate_genome_similarity(self, genome1: List[float], genome2: List[float]) -> float:
        """Calculate similarity between two genomes (0-1, higher = more similar)"""
        if len(genome1) != len(genome2) or len(genome1) != self.genome_length:
            return 0.0

        # Calculate Euclidean distance and convert to similarity
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(genome1, genome2)))
        max_distance = np.sqrt(self.genome_length)  # Maximum possible distance

        similarity = 1.0 - (distance / max_distance)
        return max(0.0, similarity)
