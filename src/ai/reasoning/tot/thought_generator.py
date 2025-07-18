"""
Thought Generator for Tree of Thought Framework

Generates multiple reasoning paths/thoughts from current states using
different strategies (sample vs propose) and LLM backends.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from .models import ThoughtState, ToTConfig, GenerationStrategy
from .core.thought import Thought
try:
    from ....utils.async_utils import run_with_timeout
except ImportError:
    # Fallback for direct execution
    import asyncio
    async def run_with_timeout(coro, timeout):
        return await asyncio.wait_for(coro, timeout=timeout)

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass


class OllamaBackend(LLMBackend):
    """Ollama backend for local LLM inference"""
    
    def __init__(self, model_name: str = "phi4-fast", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama"""
        session = None
        try:
            import aiohttp

            # Use a default model if none specified or if model doesn't exist
            model_to_use = self.model_name or "llama3:8b"

            payload = {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }

            session = aiohttp.ClientSession()
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                elif response.status == 404:
                    logger.warning(f"Model '{model_to_use}' not found, trying fallback")
                    # Try with a common model
                    fallback_payload = {
                        "model": "llama3:8b",
                        "prompt": prompt,
                        "stream": False,
                        **kwargs
                    }
                    async with session.post(f"{self.base_url}/api/generate", json=fallback_payload) as fallback_response:
                        if fallback_response.status == 200:
                            result = await fallback_response.json()
                            return result.get("response", "").strip()
                        else:
                            logger.error(f"Ollama API error with fallback: {fallback_response.status}")
                            return ""
                else:
                    logger.error(f"Ollama API error: {response.status}")
                    return ""
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return ""
        finally:
            if session:
                await session.close()


class ThoughtGenerator:
    """Generates thoughts for Tree of Thought reasoning"""
    
    def __init__(self, config: ToTConfig, llm_backend: Optional[LLMBackend] = None):
        self.config = config
        self.llm_backend = llm_backend or OllamaBackend(config.model_name)
        self.generation_count = 0
    
    async def generate_thoughts(self, parent_thought: Thought, 
                              context: Dict[str, Any]) -> List[Thought]:
        """
        Generate multiple thoughts from the parent thought
        
        Args:
            parent_thought: Parent thought to generate from
            context: Task-specific context and prompts
            
        Returns:
            List of generated thoughts
        """
        if self.config.generation_strategy == GenerationStrategy.SAMPLE:
            return await self._generate_sample_thoughts(parent_thought, context)
        else:
            return await self._generate_propose_thoughts(parent_thought, context)
    
    async def _generate_sample_thoughts(self, parent_thought: Thought, 
                                      context: Dict[str, Any]) -> List[Thought]:
        """Generate independent sample thoughts (for creative tasks)"""
        thoughts = []
        
        # Get the sampling prompt template
        sample_prompt = context.get("sample_prompt", self._default_sample_prompt())
        
        # Generate multiple independent thoughts
        for i in range(self.config.n_generate_sample):
            prompt = self._format_prompt(sample_prompt, parent_thought, context, i)
            
            try:
                thought_text = await self._call_llm(prompt)
                if thought_text and thought_text not in [t.content for t in thoughts]:
                    # Create new thought from parent
                    new_thought = Thought(
                        content=thought_text,
                        parent_id=parent_thought.id if parent_thought else None,
                        depth=(parent_thought.depth + 1) if parent_thought else 0
                    )
                    thoughts.append(new_thought)
            except Exception as e:
                logger.error(f"Error generating sample thought {i}: {e}")
        
        return thoughts
    
    async def _generate_propose_thoughts(self, parent_thought: Thought, 
                                       context: Dict[str, Any]) -> List[Thought]:
        """Generate sequential proposal thoughts (for step-by-step tasks)"""
        thoughts = []
        
        # Get the proposal prompt template
        propose_prompt = context.get("propose_prompt", self._default_propose_prompt())
        
        # Generate sequential thoughts
        for i in range(self.config.n_generate_sample):
            prompt = self._format_prompt(propose_prompt, parent_thought, context, i)
            
            try:
                thought_text = await self._call_llm(prompt)
                if thought_text:
                    # Create new thought from parent
                    new_thought = Thought(
                        content=thought_text,
                        parent_id=parent_thought.id if parent_thought else None,
                        depth=(parent_thought.depth + 1) if parent_thought else 0
                    )
                    thoughts.append(new_thought)
            except Exception as e:
                logger.error(f"Error generating proposal thought {i}: {e}")
        
        return thoughts
    
    def _format_prompt(self, template: str, parent_thought: Thought, 
                      context: Dict[str, Any], iteration: int) -> str:
        """Format the prompt template with current context"""
        return template.format(
            task_description=self.config.task_description,
            current_thought=parent_thought.content if parent_thought else "",
            current_depth=parent_thought.depth if parent_thought else 0,
            iteration=iteration,
            **context
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM backend with timeout and error handling"""
        self.generation_count += 1
        
        try:
            result = await run_with_timeout(
                self.llm_backend.generate(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                ),
                timeout=30.0
            )
            return result.strip() if result else ""
        except asyncio.TimeoutError:
            logger.error("LLM call timed out")
            return ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""
    
    def _default_sample_prompt(self) -> str:
        """Default prompt template for sampling strategy"""
        return """Task: {task_description}

Current thought: {current_thought}

Generate a creative and diverse continuation or alternative approach to this thought.
Be imaginative and explore different possibilities.

Thought:"""
    
    def _default_propose_prompt(self) -> str:
        """Default prompt template for proposal strategy"""
        return """Task: {task_description}

Current thought: {current_thought}
Depth: {current_depth}

Propose the next logical step or reasoning move to progress toward the solution.
Be specific and actionable.

Next step:"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "total_generations": self.generation_count,
            "strategy": self.config.generation_strategy.value,
            "model": self.config.model_name
        }
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            'generation_count': self.generation_count,
            'config': {
                'model_name': self.config.model_name,
                'generation_strategy': self.config.generation_strategy.value,
                'n_generate_sample': self.config.n_generate_sample,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens
            }
        }
