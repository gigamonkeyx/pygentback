"""
Test suite for ToT search algorithms.

Tests BFS, DFS, and Adaptive search implementations.
"""

import pytest
from ..core.thought import Thought, ThoughtType
from ..core.tree import ThoughtTree
from ..models import ToTConfig, SearchMethod
from ..search import BFSSearch, DFSSearch, AdaptiveSearch


class TestBFSSearch:
    """Test the BFS search algorithm"""
    
    @pytest.fixture
    def config(self):
        return ToTConfig(
            model_name="test-model",
            search_method=SearchMethod.BFS,
            max_depth=3,
            n_select_sample=2
        )
    
    @pytest.fixture
    def sample_tree(self):
        """Create a sample tree for testing"""
        tree = ThoughtTree(max_depth=3)
        
        # Create root problem
        root = Thought("Test problem", ThoughtType.PROBLEM, depth=0)
        tree.add_thought(root)
        
        # Create some reasoning thoughts
        thought1 = Thought("Reasoning 1", ThoughtType.REASONING, parent_id=root.id, depth=1, confidence=0.6)
        thought2 = Thought("Reasoning 2", ThoughtType.REASONING, parent_id=root.id, depth=1, confidence=0.8)
        tree.add_thought(thought1)
        tree.add_thought(thought2)
          # Create solution thoughts
        solution1 = Thought("Solution 1", ThoughtType.SOLUTION, parent_id=thought1.id, depth=2, confidence=0.9)
        solution2 = Thought("Solution 2", ThoughtType.SOLUTION, parent_id=thought2.id, depth=2, confidence=0.7)
        tree.add_thought(solution1)
        tree.add_thought(solution2)
        
        return tree
    
    @pytest.mark.asyncio
    async def test_bfs_search(self, config, sample_tree):
        """Test BFS search algorithm"""
        search = BFSSearch(config)
        
        # Mock generator and evaluator
        async def mock_generator(parent, context):
            if parent.depth < 2:
                return [
                    Thought(f"Solution: Generated solution from {parent.content[:10]}", 
                           ThoughtType.SOLUTION, 
                           parent_id=parent.id, 
                           depth=parent.depth + 1,
                           confidence=0.8)
                ]
            return []
        
        async def mock_evaluator(thoughts, reasoning_state, context):
            for thought in thoughts:
                thought.confidence = 0.8
            return [(thought, thought.confidence) for thought in thoughts]
        
        solutions = await search.search(
            sample_tree,
            mock_generator,
            mock_evaluator,
            {"problem": "Test problem"}
        )
        
        # Should find solutions
        assert len(solutions) > 0
        assert all(isinstance(s, Thought) for s in solutions)
        assert all(s.confidence > 0 for s in solutions)


class TestDFSSearch:
    """Test the DFS search algorithm"""
    
    @pytest.fixture
    def config(self):
        return ToTConfig(            model_name="test-model",
            search_method=SearchMethod.DFS,
            max_depth=3,
            n_select_sample=2
        )
    
    @pytest.mark.asyncio
    async def test_dfs_search(self, config):
        """Test DFS search algorithm"""
        # Create a minimal tree for DFS
        tree = ThoughtTree(max_depth=2)
        root = Thought("Problem", ThoughtType.PROBLEM, depth=0)
        tree.add_thought(root)
        
        search = DFSSearch(config)
        
        # Mock generator that creates one path
        call_count = 0
        async def mock_generator(parent, context):
            nonlocal call_count
            call_count += 1
            if parent.depth < 1:
                return [
                    Thought(f"Child {call_count}", 
                           ThoughtType.SOLUTION, 
                           parent_id=parent.id, 
                           depth=parent.depth + 1,
                           confidence=0.8)
                ]
            return []
        
        async def mock_evaluator(thoughts, reasoning_state, context):
            for thought in thoughts:
                thought.confidence = 0.8
            return [(thought, thought.confidence) for thought in thoughts]
        
        solutions = await search.search(
            tree,
            mock_generator,
            mock_evaluator,
            {"problem": "Test problem"}
        )
        
        # Should find solutions
        assert len(solutions) > 0
        assert all(isinstance(s, Thought) for s in solutions)
        assert all(s.confidence > 0 for s in solutions)


class TestAdaptiveSearch:
    """Test the Adaptive search algorithm"""
    
    @pytest.fixture
    def config(self):
        return ToTConfig(
            model_name="test-model",
            search_method=SearchMethod.ADAPTIVE,
            max_depth=3,
            n_select_sample=2
        )
    
    @pytest.fixture
    def sample_tree(self):
        """Create a sample tree for testing"""
        tree = ThoughtTree(max_depth=3)
        
        # Create root problem
        root = Thought("Test problem", ThoughtType.PROBLEM, depth=0)
        tree.add_thought(root)
        
        # Create some reasoning thoughts
        thought1 = Thought("Reasoning 1", ThoughtType.REASONING, parent_id=root.id, depth=1, confidence=0.6)
        thought2 = Thought("Reasoning 2", ThoughtType.REASONING, parent_id=root.id, depth=1, confidence=0.8)
        tree.add_thought(thought1)
        tree.add_thought(thought2)
        
        return tree
    
    @pytest.mark.asyncio
    async def test_adaptive_search_switches_strategy(self, config, sample_tree):
        """Test adaptive search switches between BFS and DFS"""
        search = AdaptiveSearch(config)
        
        # Mock generator and evaluator
        async def mock_generator(parent, context):
            if parent.depth < 2:
                return [
                    Thought(f"Generated from {parent.content[:10]}", 
                           ThoughtType.SOLUTION, 
                           parent_id=parent.id, 
                           depth=parent.depth + 1,
                           confidence=0.8)
                ]
            return []
        
        async def mock_evaluator(thoughts, reasoning_state, context):
            for thought in thoughts:
                thought.confidence = 0.8
            return [(thought, thought.confidence) for thought in thoughts]
        
        solutions = await search.search(
            sample_tree,
            mock_generator,
            mock_evaluator,
            {"problem": "Test problem"}
        )
        
        # Should find solutions using adaptive strategy
        assert len(solutions) > 0
        assert all(isinstance(s, Thought) for s in solutions)


def create_test_tree():
    """Helper function to create a test tree"""
    tree = ThoughtTree(max_depth=3)
    
    # Create root problem
    root = Thought("Test problem", ThoughtType.PROBLEM, depth=0)
    tree.add_thought(root)
    
    # Create some reasoning thoughts
    thought1 = Thought("Reasoning 1", ThoughtType.REASONING, parent_id=root.id, depth=1, confidence=0.6)
    thought2 = Thought("Reasoning 2", ThoughtType.REASONING, parent_id=root.id, depth=1, confidence=0.8)
    tree.add_thought(thought1)
    tree.add_thought(thought2)
    
    # Create solution thoughts
    solution1 = Thought("Solution 1", ThoughtType.SOLUTION, parent_id=thought1.id, depth=2, confidence=0.9)
    solution2 = Thought("Solution 2", ThoughtType.SOLUTION, parent_id=thought2.id, depth=2, confidence=0.7)
    tree.add_thought(solution1)
    tree.add_thought(solution2)
    
    return tree
