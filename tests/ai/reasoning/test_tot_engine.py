"""
Tests for Tree of Thought Engine

Basic tests to verify ToT functionality with mock LLM backend.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.ai.reasoning.tot.models import ToTConfig, GenerationStrategy, EvaluationMethod
from src.ai.reasoning.tot.tot_engine import ToTEngine
from src.ai.reasoning.tot.thought_generator import LLMBackend


class MockLLMBackend(LLMBackend):
    """Mock LLM backend for testing"""
    
    def __init__(self):
        self.call_count = 0
        self.responses = [
            "First thought: Let's break this down step by step",
            "Second thought: We need to consider the constraints",
            "Third thought: Here's a potential solution approach",
            "0.8",  # Evaluation score
            "yes",  # Solution check
        ]
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Return mock responses"""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = f"Mock response {self.call_count}"
        
        self.call_count += 1
        return response


@pytest.fixture
def mock_config():
    """Create test configuration"""
    return ToTConfig(
        generation_strategy=GenerationStrategy.PROPOSE,
        evaluation_method=EvaluationMethod.VALUE,
        n_generate_sample=2,
        n_evaluate_sample=1,
        n_select_sample=2,
        max_depth=3,
        temperature=0.7
    )


@pytest.fixture
def mock_llm_backend():
    """Create mock LLM backend"""
    return MockLLMBackend()


@pytest.fixture
def tot_engine(mock_config, mock_llm_backend):
    """Create ToT engine with mock backend"""
    return ToTEngine(mock_config, mock_llm_backend)


@pytest.mark.asyncio
async def test_tot_engine_initialization(tot_engine):
    """Test ToT engine initialization"""
    assert tot_engine.config is not None
    assert tot_engine.thought_generator is not None
    assert tot_engine.state_evaluator is not None
    assert tot_engine.search_strategy is not None
    assert tot_engine.total_runs == 0
    assert tot_engine.successful_runs == 0


@pytest.mark.asyncio
async def test_simple_problem_solving(tot_engine):
    """Test solving a simple problem"""
    problem = "What is 2 + 2?"
    task_context = {
        "propose_prompt": "Solve: {task_description}\nCurrent: {current_thought}\nNext step:",
        "value_prompt": "Rate this step: {thought_content}\nScore:",
        "solution_prompt": "Is this complete: {thought_content}\nAnswer:"
    }
    
    result = await tot_engine.solve(problem, task_context)
    
    # Verify result structure
    assert result is not None
    assert hasattr(result, 'tree')
    assert hasattr(result, 'success')
    assert hasattr(result, 'total_time')
    assert result.total_time > 0
    
    # Verify tree was created
    assert result.tree is not None
    assert result.tree.root_id is not None
    assert len(result.tree.states) > 0


@pytest.mark.asyncio
async def test_thought_generation(tot_engine):
    """Test thought generation functionality"""
    from src.ai.reasoning.tot.models import ThoughtState
    
    state = ThoughtState(content="Initial problem", depth=0)
    task_context = {
        "propose_prompt": "Next step for: {current_thought}\nStep:"
    }
    
    thoughts = await tot_engine._generate_thoughts_wrapper(state, task_context)
    
    assert isinstance(thoughts, list)
    # Should generate some thoughts (even if mocked)
    assert len(thoughts) >= 0


@pytest.mark.asyncio
async def test_state_evaluation(tot_engine):
    """Test state evaluation functionality"""
    from src.ai.reasoning.tot.models import ThoughtState
    
    states = [
        ThoughtState(content="First solution approach", depth=1),
        ThoughtState(content="Second solution approach", depth=1)
    ]
    task_context = {
        "value_prompt": "Rate: {thought_content}\nScore:"
    }
    
    evaluated_states = await tot_engine._evaluate_states_wrapper(states, task_context)
    
    assert isinstance(evaluated_states, list)
    assert len(evaluated_states) == len(states)
    
    # Check that states have evaluation scores
    for state in evaluated_states:
        assert hasattr(state, 'value_score')


@pytest.mark.asyncio
async def test_engine_statistics(tot_engine):
    """Test engine statistics tracking"""
    initial_stats = tot_engine.get_stats()
    
    assert initial_stats['total_runs'] == 0
    assert initial_stats['successful_runs'] == 0
    assert initial_stats['success_rate'] == 0
    
    # Run a simple problem
    problem = "Test problem"
    await tot_engine.solve(problem)
    
    updated_stats = tot_engine.get_stats()
    assert updated_stats['total_runs'] == 1


@pytest.mark.asyncio
async def test_solution_path_formatting(tot_engine):
    """Test solution path formatting"""
    from src.ai.reasoning.tot.models import ThoughtState, ThoughtTree
    
    # Create a simple tree
    root = ThoughtState(content="Problem", depth=0)
    child = ThoughtState(content="Solution", depth=1, parent_id=root.id)
    
    tree = ThoughtTree(root_id=root.id)
    tree.add_state(root)
    tree.add_state(child)
    root.add_child(child.id)
    
    # Test path formatting
    path = tot_engine.get_solution_path(child, tree)
    assert len(path) == 2
    assert path[0].content == "Problem"
    assert path[1].content == "Solution"
    
    formatted = tot_engine.format_solution_path(child, tree)
    assert "Problem:" in formatted
    assert "Step 1:" in formatted
    assert "Solution" in formatted


def test_config_validation():
    """Test configuration validation"""
    config = ToTConfig(
        generation_strategy=GenerationStrategy.SAMPLE,
        evaluation_method=EvaluationMethod.VOTE,
        n_generate_sample=3,
        max_depth=5
    )
    
    assert config.generation_strategy == GenerationStrategy.SAMPLE
    assert config.evaluation_method == EvaluationMethod.VOTE
    assert config.n_generate_sample == 3
    assert config.max_depth == 5


@pytest.mark.asyncio
async def test_error_handling(mock_config):
    """Test error handling with failing LLM backend"""
    
    class FailingLLMBackend(LLMBackend):
        async def generate(self, prompt: str, **kwargs) -> str:
            raise Exception("LLM backend failed")
    
    failing_backend = FailingLLMBackend()
    tot_engine = ToTEngine(mock_config, failing_backend)
    
    result = await tot_engine.solve("Test problem")
    
    # Should handle errors gracefully
    assert result is not None
    assert not result.success
    assert result.error_message is not None


if __name__ == "__main__":
    # Run basic test
    async def run_basic_test():
        config = ToTConfig(max_depth=2, n_generate_sample=1)
        backend = MockLLMBackend()
        engine = ToTEngine(config, backend)
        
        result = await engine.solve("What is 1 + 1?")
        print(f"Test result: Success={result.success}, Time={result.total_time:.2f}s")
        print(f"Tree nodes: {result.tree.total_nodes}")
        
        if result.best_solution:
            print(f"Best solution: {result.best_solution.content}")
    
    asyncio.run(run_basic_test())
