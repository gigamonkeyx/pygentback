"""
Proposing-based thought generator for Tree of Thought reasoning.

This generator creates structured proposals for next reasoning steps.
"""

import logging
from typing import List, Dict, Any
from ..core.thought import Thought, ThoughtType
from ..models import ToTConfig
from ..thought_generator import ThoughtGenerator, LLMBackend

logger = logging.getLogger(__name__)


class ProposingGenerator(ThoughtGenerator):
    """
    Generates thoughts using proposing strategy.
    
    This generator creates structured proposals for the next reasoning steps,
    focusing on deliberate and coherent thought progression.
    """
    
    def __init__(self, config: ToTConfig, llm_backend: LLMBackend):
        super().__init__(config, llm_backend)
        
    async def generate_thoughts(self, parent_thought: Thought, context: Dict[str, Any]) -> List[Thought]:
        """
        Generate thoughts using proposing strategy.
        
        Args:
            parent_thought: The parent thought to generate from
            context: Task context including problem description
            
        Returns:
            List of generated thoughts
        """
        try:
            thoughts = []
            problem = context.get('problem', '')
            
            # Create proposing prompt
            prompt = self._create_proposing_prompt(parent_thought, problem)
            
            # Generate structured proposals
            response = await self.llm_backend.generate(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens * 2  # Allow more tokens for structured output
            )
            
            if response and response.strip():
                # Parse proposals from response
                proposals = self._parse_proposals(response)
                
                for i, proposal in enumerate(proposals[:self.config.n_generate_sample]):
                    if proposal.strip():
                        # Determine thought type
                        thought_type = self._determine_thought_type(parent_thought, proposal)
                        
                        thought = Thought(
                            content=proposal.strip(),
                            thought_type=thought_type,
                            parent_id=parent_thought.id,
                            depth=parent_thought.depth + 1
                        )
                        
                        thoughts.append(thought)
                        logger.debug(f"Generated proposal {i+1}: {proposal[:50]}...")
                        
            return thoughts
            
        except Exception as e:
            logger.error(f"Proposing generation failed: {e}")
            return []
    
    def _create_proposing_prompt(self, parent_thought: Thought, problem: str) -> str:
        """Create prompt for proposing-based generation"""
        
        if parent_thought.thought_type == ThoughtType.PROBLEM:
            return f"""Problem: {problem}

I need to propose {self.config.n_generate_sample} different approaches to solve this problem.

For each approach, I'll provide a clear proposal with reasoning.

Proposal 1:
Proposal 2:
Proposal 3:"""
        
        elif parent_thought.thought_type == ThoughtType.REASONING:
            return f"""Problem: {problem}

Current reasoning: {parent_thought.content}

I need to propose {self.config.n_generate_sample} possible next steps to continue this reasoning.

Proposal 1:
Proposal 2:
Proposal 3:"""
        
        else:
            return f"""Problem: {problem}

Current thought: {parent_thought.content}

I need to propose {self.config.n_generate_sample} ways to refine or build upon this thought.

Proposal 1:
Proposal 2:
Proposal 3:"""
    
    def _parse_proposals(self, response: str) -> List[str]:
        """Parse individual proposals from the response"""
        proposals = []
        
        # Split by proposal markers
        lines = response.split('\n')
        current_proposal = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Proposal '):
                if current_proposal:
                    proposals.append(current_proposal.strip())
                # Extract content after "Proposal N:"
                if ':' in line:
                    current_proposal = line.split(':', 1)[1].strip()
                else:
                    current_proposal = ""
            elif current_proposal:
                # Continue building current proposal
                current_proposal += " " + line
        
        # Add the last proposal
        if current_proposal:
            proposals.append(current_proposal.strip())
            
        # Fallback: split by numbered lists
        if not proposals and response:
            # Try to parse numbered lists
            import re
            matches = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', response, re.DOTALL)
            if matches:
                proposals = [match.strip() for match in matches]
            else:
                # If no structure found, return whole response as single proposal
                proposals = [response.strip()]
                
        return proposals
    
    def _determine_thought_type(self, parent_thought: Thought, content: str) -> ThoughtType:
        """Determine the type of generated thought based on context"""
        
        content_lower = content.lower()
        
        # Check for solution indicators
        if any(word in content_lower for word in ['solution', 'answer', 'final', 'conclude']):
            return ThoughtType.SOLUTION
        # Check for reasoning indicators  
        elif any(word in content_lower for word in ['analyze', 'consider', 'examine', 'explore']):
            return ThoughtType.REASONING
        # Check for critique indicators
        elif any(word in content_lower for word in ['however', 'alternatively', 'problem with', 'issue']):
            return ThoughtType.CRITIQUE
        else:
            # Default progression based on depth
            if parent_thought.depth >= 2:
                return ThoughtType.SOLUTION
            else:
                return ThoughtType.REASONING
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'generator_type': 'proposing',
            'total_calls': self.generation_count,
            'config': {
                'n_generate_sample': self.config.n_generate_sample,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens
            }
        }
