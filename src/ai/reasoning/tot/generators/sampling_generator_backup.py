"""
Sampling-based thought generator for Tree of Thought reasoning.

This generator uses random sampling to create diverse thoughts at each step.
"""

import logging
from typing import List, Dict, Any
from ..core.thought import Thought, ThoughtType
from ..models import ToTConfig
from ..thought_generator import ThoughtGenerator, LLMBackend

logger = logging.getLogger(__name__)


class SamplingGenerator(ThoughtGenerator):
    """
    Generates thoughts using random sampling strategy.
    
    This generator creates multiple diverse thoughts by sampling from the LLM
    with different temperature/randomness settings to explore various reasoning paths.
    """
    
    def __init__(self, config: ToTConfig, llm_backend: LLMBackend):
        super().__init__(config, llm_backend)
        
    async def generate_thoughts(self, parent_thought: Thought, context: Dict[str, Any]) -> List[Thought]:        """
        Generate thoughts using sampling strategy.
        
        Args:
            parent_thought: The parent thought to generate from
            context: Task context including problem description
            
        Returns:
            List of generated thoughts
        """
        try:
            # Ensure parent_thought is a Thought object
            if not isinstance(parent_thought, Thought):
                logger.error(f"parent_thought is not a Thought object: {type(parent_thought)}, value: {parent_thought}")
                return []
                
            thoughts = []
            problem = context.get('problem', '')
            
            # Create sampling prompt
            prompt = self._create_sampling_prompt(parent_thought, problem)
            
            # Generate multiple samples
            for i in range(self.config.n_generate_sample):
                try:
                    # Use higher temperature for more diversity
                    response = await self.llm_backend.generate(
                        prompt,
                        temperature=min(1.0, self.config.temperature + 0.2),
                        max_tokens=self.config.max_tokens
                    )
                    
                    if response and response.strip():
                        # Determine thought type based on depth and content
                        thought_type = self._determine_thought_type(parent_thought, response)
                        
                        thought = Thought(
                            content=response.strip(),
                            thought_type=thought_type,
                            parent_id=parent_thought.id,
                            depth=parent_thought.depth + 1
                        )
                        
                        thoughts.append(thought)
                        logger.debug(f"Generated sampling thought {i+1}: {response[:50]}...")
                        
                except Exception as e:
                    logger.warning(f"Failed to generate sampling thought {i+1}: {e}")
                    continue
                    
            return thoughts
            
        except Exception as e:
            logger.error(f"Sampling generation failed: {e}")
            return []
    
    def _create_sampling_prompt(self, parent_thought: Thought, problem: str) -> str:
        """Create prompt for sampling-based generation"""
        
        if parent_thought.thought_type == ThoughtType.PROBLEM:
            return f"""Problem: {problem}

Previous thought: {parent_thought.content}

Generate a reasoning step or potential solution approach for this problem. Be creative and explore different angles. Focus on one specific aspect or approach.

Reasoning step:"""
        
        elif parent_thought.thought_type == ThoughtType.REASONING:
            return f"""Problem: {problem}

Current reasoning: {parent_thought.content}

Continue the reasoning process. You can either:
1. Develop this reasoning further
2. Propose a specific solution
3. Consider alternative approaches

Next step:"""
        
        else:
            return f"""Problem: {problem}

Current thought: {parent_thought.content}

Refine or improve this thought, or provide an alternative perspective.

Refined thought:"""
    
    def _determine_thought_type(self, parent_thought: Thought, content: str) -> ThoughtType:
        """Determine the type of generated thought based on context"""
        
        # Simple heuristics for thought type classification
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['solution', 'answer', 'result', 'final']):
            return ThoughtType.SOLUTION
        elif any(word in content_lower for word in ['because', 'since', 'therefore', 'thus', 'reasoning']):
            return ThoughtType.REASONING
        elif any(word in content_lower for word in ['however', 'but', 'problem', 'issue', 'wrong']):
            return ThoughtType.CRITIQUE
        else:
            # Default based on parent type and depth
            if parent_thought.depth >= 2:
                return ThoughtType.SOLUTION
            else:
                return ThoughtType.REASONING
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'generator_type': 'sampling',
            'total_calls': self.generation_count,
            'config': {
                'n_generate_sample': self.config.n_generate_sample,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens
            }
        }
