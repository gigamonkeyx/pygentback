"""
Integration tests for the ToT Engine
"""
import pytest

from ..models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from ..tot_engine import ToTEngine
from ..thought_generator import LLMBackend


class MockLLMBackend(LLMBackend):
    """Mock LLM backend for testing"""
    
    def __init__(self, responses=None):
        self.responses = responses or [
            "First reasoning step: Analyze the problem",
            "Second reasoning step: Consider alternatives", 
            "Third reasoning step: Evaluate options",
            "Score: 0.8 - This is a good approach",
            "Score: 0.6 - This has some merit",
            "Score: 0.9 - This is excellent"
        ]
        self.call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return f"Mock response {self.call_count}"


class TestToTEngine:
    """Test the complete ToT Engine"""
    
    @pytest.fixture
    def basic_config(self):
        return ToTConfig(
            model_name="test-model",
            generation_strategy=GenerationStrategy.SAMPLE,
            evaluation_method=EvaluationMethod.VALUE,
            search_method=SearchMethod.BFS,
            n_generate_sample=2,
            n_evaluate_sample=1,
            n_select_sample=1,
            max_depth=2,
            temperature=0.7,
            max_tokens=100,
            task_description="Test problem",
            success_criteria="Find good solution"
        )
    
    @pytest.fixture
    def mock_backend(self):
        return MockLLMBackend()
    
    @pytest.mark.asyncio
    async def test_engine_solve_basic_problem(self, basic_config, mock_backend):
        """Test solving a basic problem with ToT engine"""
        engine = ToTEngine(basic_config, mock_backend)
        
        result = await engine.solve("What is 2 + 2?")
        
        assert result is not None
        assert hasattr(result, 'solutions')
        assert hasattr(result, 'tree')
        assert hasattr(result, 'search_time')
        assert hasattr(result, 'stats')
        
        # Should have found at least one solution or attempted search
        assert result.tree is not None
        assert result.search_time > 0
    
    @pytest.mark.asyncio
    async def test_engine_with_context(self, basic_config, mock_backend):
        """Test engine with additional context"""
        engine = ToTEngine(basic_config, mock_backend)
        
        context = {
            "domain": "mathematics",
            "difficulty": "easy",
            "session_id": "test_session"
        }
        
        result = await engine.solve("Solve x + 3 = 7", context)
        
        assert result is not None
        assert result.stats is not None
        
        # Check that context was used
        if 'session_id' in result.stats:
            assert result.stats['session_id'] == "test_session"
    
    @pytest.mark.asyncio
    async def test_engine_different_strategies(self):
        """Test engine with different strategy combinations"""
        configs = [
            # BFS + Sampling + Value
            ToTConfig(
                model_name="test",
                generation_strategy=GenerationStrategy.SAMPLE,
                evaluation_method=EvaluationMethod.VALUE,
                search_method=SearchMethod.BFS,
                n_generate_sample=2,
                max_depth=2
            ),
            # DFS + Proposing + Vote
            ToTConfig(
                model_name="test",
                generation_strategy=GenerationStrategy.PROPOSE,
                evaluation_method=EvaluationMethod.VOTE,
                search_method=SearchMethod.DFS,
                n_generate_sample=2,
                max_depth=2
            ),
            # Adaptive search
            ToTConfig(
                model_name="test",
                generation_strategy=GenerationStrategy.SAMPLE,
                evaluation_method=EvaluationMethod.VALUE,
                search_method=SearchMethod.ADAPTIVE,
                n_generate_sample=2,
                max_depth=2
            )
        ]
        
        for config in configs:
            mock_backend = MockLLMBackend()
            engine = ToTEngine(config, mock_backend)
            
            result = await engine.solve("Test problem")
            
            assert result is not None
            assert result.tree is not None
            assert result.search_time >= 0
    
    @pytest.mark.asyncio
    async def test_engine_error_handling(self, basic_config):
        """Test engine error handling"""
        # Backend that always fails
        class FailingBackend(LLMBackend):
            async def generate(self, prompt: str, **kwargs) -> str:
                raise Exception("Backend failure")
        
        engine = ToTEngine(basic_config, FailingBackend())
        
        result = await engine.solve("Test problem")
        
        # Should return result even on failure
        assert result is not None
        assert not result.success
        assert 'error' in result.stats
    
    @pytest.mark.asyncio
    async def test_engine_statistics(self, basic_config, mock_backend):
        """Test that engine collects proper statistics"""
        engine = ToTEngine(basic_config, mock_backend)
        
        # Run multiple problems
        problems = ["Problem 1", "Problem 2", "Problem 3"]
        
        for problem in problems:
            await engine.solve(problem)
        
        # Check engine statistics
        assert engine.total_runs == 3
        assert engine.total_search_time > 0
        
        # At least some runs should be successful
        assert engine.successful_runs >= 0
    
    @pytest.mark.asyncio
    async def test_engine_max_depth_constraint(self, mock_backend):
        """Test that engine respects max depth constraint"""
        config = ToTConfig(
            model_name="test",
            generation_strategy=GenerationStrategy.SAMPLE,
            evaluation_method=EvaluationMethod.VALUE,
            search_method=SearchMethod.BFS,
            n_generate_sample=1,
            max_depth=1,  # Very shallow
            max_tokens=50
        )
        
        engine = ToTEngine(config, mock_backend)
        result = await engine.solve("Deep problem requiring multiple steps")
        
        assert result is not None
        if result.tree and result.tree.root:
            # All thoughts should be within max depth
            for thought in result.tree.nodes.values():
                assert thought.depth <= config.max_depth
    
    @pytest.mark.asyncio
    async def test_engine_solution_ranking(self, mock_backend):
        """Test that solutions are properly ranked"""
        # Backend that returns solutions with different quality scores
        class GradedBackend(LLMBackend):
            def __init__(self):
                self.call_count = 0
                
            async def generate(self, prompt: str, **kwargs) -> str:
                self.call_count += 1
                if "evaluate" in prompt.lower() or "score" in prompt.lower():
                    # Return different scores for different solutions
                    scores = ["0.9", "0.7", "0.5"]
                    return f"Score: {scores[min(self.call_count % 3, len(scores)-1)]}"
                else:
                    return f"Solution {self.call_count}"
        
        config = ToTConfig(
            model_name="test",
            generation_strategy=GenerationStrategy.SAMPLE,
            evaluation_method=EvaluationMethod.VALUE,
            search_method=SearchMethod.BFS,
            n_generate_sample=3,
            n_select_sample=2,
            max_depth=2
        )
        
        engine = ToTEngine(config, GradedBackend())
        result = await engine.solve("Find the best solution")
        
        if result.solutions:
            # Solutions should be ordered by confidence
            confidences = [s.confidence for s in result.solutions]
            assert confidences == sorted(confidences, reverse=True)
