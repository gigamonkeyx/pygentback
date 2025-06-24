"""
Test suite for ToT core components
"""
from datetime import datetime

from ..core.thought import Thought, ThoughtType
from ..core.state import ReasoningState
from ..core.tree import ThoughtTree


class TestThought:
    """Test the Thought data structure"""
    
    def test_thought_creation(self):
        """Test basic thought creation"""
        thought = Thought(
            content="Test thought",
            thought_type=ThoughtType.REASONING,
            parent_id=None,
            depth=0
        )
        
        assert thought.content == "Test thought"
        assert thought.thought_type == ThoughtType.REASONING
        assert thought.parent_id is None
        assert thought.depth == 0
        assert thought.confidence == 0.0
        assert isinstance(thought.created_at, datetime)
        assert thought.id is not None
    
    def test_thought_with_parent(self):
        """Test thought creation with parent"""
        parent = Thought("Parent thought", ThoughtType.PROBLEM)
        child = Thought(
            content="Child thought",
            thought_type=ThoughtType.REASONING,
            parent_id=parent.id,
            depth=1
        )
        
        assert child.parent_id == parent.id
        assert child.depth == 1
    
    def test_thought_serialization(self):
        """Test thought to_dict method"""
        thought = Thought("Test", ThoughtType.SOLUTION, confidence=0.8)
        data = thought.to_dict()
        
        expected_keys = [
            'id', 'content', 'thought_type', 'parent_id', 'depth',
            'confidence', 'metadata', 'created_at', 'evaluation_scores'
        ]
        
        for key in expected_keys:
            assert key in data
        
        assert data['content'] == "Test"
        assert data['thought_type'] == ThoughtType.SOLUTION.value
        assert data['confidence'] == 0.8


class TestReasoningState:
    """Test the ReasoningState management"""
    
    def test_state_creation(self):
        """Test basic state creation"""
        state = ReasoningState(
            problem="Test problem",
            session_id="test_session"
        )
        
        assert state.problem == "Test problem"
        assert state.session_id == "test_session"
        assert len(state.thoughts) == 0
        assert len(state.evaluations) == 0
        assert isinstance(state.created_at, datetime)
    
    def test_add_thought(self):
        """Test adding thoughts to state"""
        state = ReasoningState("Test problem", "session")
        thought = Thought("Test thought", ThoughtType.REASONING)
        
        state.add_thought(thought)
        
        assert len(state.thoughts) == 1
        assert state.thoughts[0] == thought
    
    def test_get_thoughts_by_type(self):
        """Test filtering thoughts by type"""
        state = ReasoningState("Test problem", "session")
        
        problem_thought = Thought("Problem", ThoughtType.PROBLEM)
        reasoning_thought = Thought("Reasoning", ThoughtType.REASONING)
        solution_thought = Thought("Solution", ThoughtType.SOLUTION)
        
        state.add_thought(problem_thought)
        state.add_thought(reasoning_thought)
        state.add_thought(solution_thought)
        
        reasoning_thoughts = state.get_thoughts_by_type(ThoughtType.REASONING)
        assert len(reasoning_thoughts) == 1
        assert reasoning_thoughts[0] == reasoning_thought
    
    def test_get_best_thoughts(self):
        """Test getting best thoughts by confidence"""
        state = ReasoningState("Test problem", "session")
        
        low_conf = Thought("Low", ThoughtType.SOLUTION, confidence=0.3)
        high_conf = Thought("High", ThoughtType.SOLUTION, confidence=0.9)
        mid_conf = Thought("Mid", ThoughtType.SOLUTION, confidence=0.6)
        
        state.add_thought(low_conf)
        state.add_thought(high_conf)
        state.add_thought(mid_conf)
        
        best_thoughts = state.get_best_thoughts(limit=2)
        assert len(best_thoughts) == 2
        assert best_thoughts[0] == high_conf
        assert best_thoughts[1] == mid_conf
    
    def test_state_serialization(self):
        """Test state to_dict method"""
        state = ReasoningState("Test problem", "session")
        thought = Thought("Test", ThoughtType.REASONING)
        state.add_thought(thought)
        
        data = state.to_dict()
        
        expected_keys = [
            'problem', 'session_id', 'thoughts', 'evaluations',
            'metadata', 'created_at', 'updated_at'
        ]
        
        for key in expected_keys:
            assert key in data
        
        assert data['problem'] == "Test problem"
        assert len(data['thoughts']) == 1


class TestThoughtTree:
    """Test the ThoughtTree structure"""
    
    def test_tree_creation(self):
        """Test basic tree creation"""
        tree = ThoughtTree(max_depth=5)
        
        assert tree.max_depth == 5
        assert tree.root is None
        assert len(tree.nodes) == 0
    
    def test_add_root_thought(self):
        """Test adding root thought"""
        tree = ThoughtTree()
        root = Thought("Root", ThoughtType.PROBLEM)
        
        tree.add_thought(root)
        
        assert tree.root == root
        assert root.id in tree.nodes
        assert len(tree.nodes) == 1
    
    def test_add_child_thought(self):
        """Test adding child thoughts"""
        tree = ThoughtTree()
        root = Thought("Root", ThoughtType.PROBLEM)
        child = Thought("Child", ThoughtType.REASONING, parent_id=root.id, depth=1)
        
        tree.add_thought(root)
        tree.add_thought(child)
        
        assert len(tree.nodes) == 2
        assert child.id in tree.nodes
        
        children = tree.get_children(root.id)
        assert len(children) == 1
        assert children[0] == child
    
    def test_get_path_to_root(self):
        """Test getting path from thought to root"""
        tree = ThoughtTree()
        
        root = Thought("Root", ThoughtType.PROBLEM)
        child1 = Thought("Child1", ThoughtType.REASONING, parent_id=root.id, depth=1)
        child2 = Thought("Child2", ThoughtType.SOLUTION, parent_id=child1.id, depth=2)
        
        tree.add_thought(root)
        tree.add_thought(child1)
        tree.add_thought(child2)
        
        path = tree.get_path_to_root(child2.id)
        assert len(path) == 3
        assert path[0] == child2
        assert path[1] == child1
        assert path[2] == root
    
    def test_get_leaves(self):
        """Test getting leaf thoughts"""
        tree = ThoughtTree()
        
        root = Thought("Root", ThoughtType.PROBLEM)
        child1 = Thought("Child1", ThoughtType.REASONING, parent_id=root.id, depth=1)
        child2 = Thought("Child2", ThoughtType.REASONING, parent_id=root.id, depth=1)
        leaf = Thought("Leaf", ThoughtType.SOLUTION, parent_id=child1.id, depth=2)
        
        tree.add_thought(root)
        tree.add_thought(child1)
        tree.add_thought(child2)
        tree.add_thought(leaf)
        
        leaves = tree.get_leaves()
        # child2 and leaf should be leaves
        assert len(leaves) == 2
        leaf_ids = [t.id for t in leaves]
        assert child2.id in leaf_ids
        assert leaf.id in leaf_ids
    
    def test_max_depth_constraint(self):
        """Test max depth constraint"""
        tree = ThoughtTree(max_depth=2)
        
        root = Thought("Root", ThoughtType.PROBLEM, depth=0)
        child1 = Thought("Child1", ThoughtType.REASONING, parent_id=root.id, depth=1)
        child2 = Thought("Child2", ThoughtType.REASONING, parent_id=child1.id, depth=2)
        too_deep = Thought("TooDeep", ThoughtType.SOLUTION, parent_id=child2.id, depth=3)
        
        tree.add_thought(root)
        tree.add_thought(child1)
        tree.add_thought(child2)
        
        # Should not add thought that exceeds max depth
        tree.add_thought(too_deep)
        assert too_deep.id not in tree.nodes
        assert len(tree.nodes) == 3
    
    def test_tree_serialization(self):
        """Test tree to_dict method"""
        tree = ThoughtTree()
        root = Thought("Root", ThoughtType.PROBLEM)
        tree.add_thought(root)
        
        data = tree.to_dict()
        
        expected_keys = ['root_id', 'max_depth', 'nodes', 'edges']
        for key in expected_keys:
            assert key in data
        
        assert data['root_id'] == root.id
        assert len(data['nodes']) == 1
        assert len(data['edges']) == 0
