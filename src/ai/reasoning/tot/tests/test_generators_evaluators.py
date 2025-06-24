"""
Test suite for ToT generators and evaluators
"""
import pytest

from ..models import ToTConfig, GenerationStrategy, EvaluationMethod
from ..core.thought import Thought, ThoughtType
from ..generators import SamplingGenerator, ProposingGenerator
from ..evaluators import ValueEvaluator, VoteEvaluator, CodingEvaluator


class MockLLMBackend:
    """Mock LLM backend for testing"""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "Mock response"


class TestSamplingGenerator:
    """Test the SamplingGenerator"""
    
    @pytest.fixture
    def config(self):
        return ToTConfig(
            model_name="test-model",
            generation_strategy=GenerationStrategy.SAMPLE,
            n_generate_sample=3
        )
    
    @pytest.fixture
    def mock_backend(self):
        return MockLLMBackend([
            "First thought",
            "Second thought", 
            "Third thought"
        ])
    
    @pytest.mark.asyncio
    async def test_generate_thoughts(self, config, mock_backend):
        """Test generating thoughts with sampling strategy"""
        generator = SamplingGenerator(config, mock_backend)
        parent = Thought("Problem: Solve 2+2", ThoughtType.PROBLEM)
        
        thoughts = await generator.generate_thoughts(parent, {"problem": "Solve 2+2"})
        
        assert len(thoughts) == 3
        assert all(isinstance(t, Thought) for t in thoughts)
        assert all(t.parent_id == parent.id for t in thoughts)
        assert all(t.depth == parent.depth + 1 for t in thoughts)
        
        contents = [t.content for t in thoughts]
        assert "First thought" in contents
        assert "Second thought" in contents
        assert "Third thought" in contents


class TestProposingGenerator:
    """Test the ProposingGenerator"""
    
    @pytest.fixture
    def config(self):
        return ToTConfig(
            model_name="test-model",
            generation_strategy=GenerationStrategy.PROPOSE,
            n_generate_sample=2
        )
    
    @pytest.fixture  
    def mock_backend(self):
        return MockLLMBackend([
            "Proposal 1: Use algebra",
            "Proposal 2: Use mental math"
        ])
    
    @pytest.mark.asyncio
    async def test_generate_thoughts(self, config, mock_backend):
        """Test generating thoughts with proposing strategy"""
        generator = ProposingGenerator(config, mock_backend)
        parent = Thought("Problem: Solve 2+2", ThoughtType.PROBLEM)
        
        thoughts = await generator.generate_thoughts(parent, {"problem": "Solve 2+2"})
        
        assert len(thoughts) == 2
        assert all(isinstance(t, Thought) for t in thoughts)
        assert all(t.parent_id == parent.id for t in thoughts)
        
        contents = [t.content for t in thoughts]
        assert any("algebra" in c.lower() for c in contents)
        assert any("mental" in c.lower() for c in contents)


class TestValueEvaluator:
    """Test the ValueEvaluator"""
    
    @pytest.fixture
    def config(self):
        return ToTConfig(
            model_name="test-model",
            evaluation_method=EvaluationMethod.VALUE
        )
    
    @pytest.fixture
    def mock_backend(self):
        return MockLLMBackend([
            "This approach is very promising. Score: 0.8",
            "This approach has some merit. Score: 0.6",
            "This approach is flawed. Score: 0.3"
        ])
    
    @pytest.mark.asyncio
    async def test_evaluate_thoughts(self, config, mock_backend):
        """Test evaluating thoughts with value method"""
        evaluator = ValueEvaluator(config, mock_backend)
        
        thoughts = [
            Thought("Use algebra", ThoughtType.REASONING),
            Thought("Use mental math", ThoughtType.REASONING),
            Thought("Random guess", ThoughtType.REASONING)
        ]
        
        await evaluator.evaluate_thoughts(thoughts, {"problem": "Solve 2+2"})
        
        # Check that all thoughts got confidence scores
        for thought in thoughts:
            assert hasattr(thought, 'confidence')
            assert 0.0 <= thought.confidence <= 1.0
        
        # Check that scores were assigned appropriately
        assert thoughts[0].confidence >= thoughts[1].confidence >= thoughts[2].confidence


class TestVoteEvaluator:
    """Test the VoteEvaluator"""
    
    @pytest.fixture
    def config(self):
        return ToTConfig(
            model_name="test-model",
            evaluation_method=EvaluationMethod.VOTE,
            n_evaluate_sample=3
        )
    
    @pytest.fixture
    def mock_backend(self):
        return MockLLMBackend([
            "Vote: A",  # Votes for option A
            "Vote: A",  # Votes for option A  
            "Vote: B"   # Votes for option B
        ])
    
    @pytest.mark.asyncio
    async def test_evaluate_thoughts(self, config, mock_backend):
        """Test evaluating thoughts with vote method"""
        evaluator = VoteEvaluator(config, mock_backend)
        
        thoughts = [
            Thought("Option A: Use algebra", ThoughtType.REASONING),
            Thought("Option B: Use mental math", ThoughtType.REASONING)
        ]
        
        await evaluator.evaluate_thoughts(thoughts, {"problem": "Solve 2+2"})
        
        # Option A should have higher confidence (2/3 votes)
        # Option B should have lower confidence (1/3 votes)
        assert thoughts[0].confidence > thoughts[1].confidence
        assert abs(thoughts[0].confidence - 2/3) < 0.1
        assert abs(thoughts[1].confidence - 1/3) < 0.1


class TestCodingEvaluator:
    """Test the CodingEvaluator"""
    
    @pytest.fixture
    def config(self):
        return ToTConfig(
            model_name="test-model",
            evaluation_method=EvaluationMethod.CODING
        )
    
    @pytest.fixture
    def mock_backend(self):
        return MockLLMBackend([
            "Syntax: CORRECT, Logic: CORRECT, Efficiency: GOOD",  # High score
            "Syntax: CORRECT, Logic: INCORRECT, Efficiency: POOR", # Low score
        ])
    
    @pytest.mark.asyncio 
    async def test_evaluate_coding_thoughts(self, config, mock_backend):
        """Test evaluating coding thoughts"""
        evaluator = CodingEvaluator(config, mock_backend)
        
        thoughts = [
            Thought("def add(a, b): return a + b", ThoughtType.SOLUTION),
            Thought("def add(a, b): return a * b", ThoughtType.SOLUTION)  # Wrong operation
        ]
        
        await evaluator.evaluate_thoughts(thoughts, {"problem": "Write function to add two numbers"})
        
        # First thought should have higher confidence
        assert thoughts[0].confidence > thoughts[1].confidence
        assert thoughts[0].confidence > 0.7  # Should be high for correct solution
        assert thoughts[1].confidence < 0.5  # Should be low for incorrect solution
        
        # Check that evaluation scores were recorded
        for thought in thoughts:
            assert 'evaluation_scores' in thought.metadata
            scores = thought.metadata['evaluation_scores']
            assert 'syntax' in scores
            assert 'logic' in scores
            assert 'efficiency' in scores
