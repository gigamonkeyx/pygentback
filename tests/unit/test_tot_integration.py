"""
Quick test of the new ToT system
"""
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.ai.reasoning.tot.models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from src.ai.reasoning.tot.tot_engine import ToTEngine
from src.ai.reasoning.tot.thought_generator import LLMBackend


class MockLLMBackend(LLMBackend):
    """Simple mock for testing"""
    
    def __init__(self):
        self.call_count = 0
        
    async def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        if "evaluate" in prompt.lower() or "score" in prompt.lower():
            return f"Score: 0.{7 + (self.call_count % 3)}"  # Return 0.7, 0.8, 0.9
        else:
            return f"Mock reasoning step {self.call_count}: This is a test response to {prompt[:30]}..."


async def test_tot_engine():
    """Test the ToT engine with mock backend"""
    print("Testing ToT Engine...")
    
    # Configure ToT
    config = ToTConfig(
        model_name="mock-model",
        generation_strategy=GenerationStrategy.SAMPLE,
        evaluation_method=EvaluationMethod.VALUE,
        search_method=SearchMethod.BFS,
        n_generate_sample=2,
        n_evaluate_sample=1,
        n_select_sample=1,
        max_depth=2,
        temperature=0.7
    )
    
    # Create engine
    mock_backend = MockLLMBackend()
    engine = ToTEngine(config, mock_backend)
    
    # Test problem
    problem = "What is 2 + 2?"
    
    print(f"Solving problem: {problem}")
    result = await engine.solve(problem)
    
    print(f"Result success: {result.success}")
    print(f"Solutions found: {len(result.solutions) if result.solutions else 0}")
    print(f"Search time: {result.search_time:.2f}s")
    print(f"Tree nodes: {len(result.tree.nodes) if result.tree else 0}")
    
    if result.solutions:
        for i, solution in enumerate(result.solutions[:3]):
            print(f"Solution {i+1}: {solution.content[:100]}... (confidence: {solution.confidence:.2f})")
    
    print(f"Engine stats - Total runs: {engine.total_runs}, Successful: {engine.successful_runs}")
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(test_tot_engine())
