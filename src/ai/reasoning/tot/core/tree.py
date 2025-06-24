"""
Tree data structure for Tree of Thought reasoning.

This module implements the tree structure that organizes thoughts
hierarchically and provides tree traversal and manipulation methods.
"""

from typing import Dict, List, Optional
from .thought import Thought


class ThoughtTree:
    """
    Tree structure for organizing thoughts in hierarchical reasoning.
    
    Manages:
    - Tree structure with parent-child relationships
    - Tree traversal and navigation
    - Depth constraints and validation
    - Serialization and persistence
    """
    
    def __init__(self, max_depth: int = 10):
        """
        Initialize thought tree.
        
        Args:
            max_depth: Maximum allowed depth for thoughts in the tree
        """
        self.max_depth = max_depth
        self.root: Optional[Thought] = None
        self.nodes: Dict[str, Thought] = {}  # thought_id -> Thought
        self.edges: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        
    def add_thought(self, thought: Thought) -> bool:
        """
        Add a thought to the tree.
        
        Args:
            thought: The thought to add
            
        Returns:
            True if successfully added, False if rejected (e.g., too deep)
        """
        # Check depth constraint
        if thought.depth > self.max_depth:
            return False
            
        # Add to nodes
        self.nodes[thought.id] = thought
        
        # Handle root thought
        if thought.parent_id is None:
            if self.root is None:
                self.root = thought
            
        # Update parent-child relationships
        if thought.parent_id is not None:
            if thought.parent_id not in self.edges:
                self.edges[thought.parent_id] = []
            self.edges[thought.parent_id].append(thought.id)
            
        return True
        
    def remove_thought(self, thought_id: str, remove_subtree: bool = True) -> bool:
        """
        Remove a thought from the tree.
        
        Args:
            thought_id: ID of the thought to remove
            remove_subtree: If True, also remove all descendant thoughts
            
        Returns:
            True if successfully removed
        """
        if thought_id not in self.nodes:
            return False
            
        thought = self.nodes[thought_id]
        
        # Remove from parent's children list
        if thought.parent_id and thought.parent_id in self.edges:
            self.edges[thought.parent_id].remove(thought_id)
            
        # Handle children
        if remove_subtree:
            # Remove all descendants recursively
            children = self.get_children(thought_id)
            for child in children:
                self.remove_thought(child.id, remove_subtree=True)
        else:
            # Reconnect children to grandparent
            children = self.get_children(thought_id)
            for child in children:
                child.parent_id = thought.parent_id
                child.depth = child.depth - 1  # Adjust depth
                
                # Update edges
                if thought.parent_id:
                    if thought.parent_id not in self.edges:
                        self.edges[thought.parent_id] = []
                    self.edges[thought.parent_id].append(child.id)
                    
        # Remove from data structures
        del self.nodes[thought_id]
        if thought_id in self.edges:
            del self.edges[thought_id]
            
        # Update root if necessary
        if self.root and self.root.id == thought_id:
            self.root = None
            
        return True
        
    def get_thought(self, thought_id: str) -> Optional[Thought]:
        """Get a thought by ID"""
        return self.nodes.get(thought_id)
        
    def get_children(self, thought_id: str) -> List[Thought]:
        """Get direct children of a thought"""
        child_ids = self.edges.get(thought_id, [])
        return [self.nodes[child_id] for child_id in child_ids if child_id in self.nodes]
        
    def get_parent(self, thought_id: str) -> Optional[Thought]:
        """Get the parent of a thought"""
        thought = self.nodes.get(thought_id)
        if thought and thought.parent_id:
            return self.nodes.get(thought.parent_id)
        return None
        
    def get_siblings(self, thought_id: str) -> List[Thought]:
        """Get siblings of a thought (thoughts with same parent)"""
        thought = self.nodes.get(thought_id)
        if not thought or not thought.parent_id:
            return []
            
        siblings = self.get_children(thought.parent_id)
        return [s for s in siblings if s.id != thought_id]
        
    def get_path_to_root(self, thought_id: str) -> List[Thought]:
        """Get the path from a thought to the root"""
        path = []
        current_id = thought_id
        
        while current_id and current_id in self.nodes:
            thought = self.nodes[current_id]
            path.append(thought)
            current_id = thought.parent_id
            
        return path
        
    def get_leaves(self) -> List[Thought]:
        """Get all leaf thoughts (thoughts with no children)"""
        leaves = []
        for thought_id, thought in self.nodes.items():
            if thought_id not in self.edges or len(self.edges[thought_id]) == 0:
                leaves.append(thought)
        return leaves
        
    def get_thoughts_at_depth(self, depth: int) -> List[Thought]:
        """Get all thoughts at a specific depth"""
        return [thought for thought in self.nodes.values() if thought.depth == depth]
        
    def get_max_depth(self) -> int:
        """Get the maximum depth of any thought in the tree"""
        if not self.nodes:
            return 0
        return max(thought.depth for thought in self.nodes.values())
        
    def get_max_breadth(self) -> int:
        """Get the maximum breadth (number of children) at any level"""
        if not self.nodes:
            return 0
            
        max_breadth = 0
        for depth in range(self.get_max_depth() + 1):
            breadth = len(self.get_thoughts_at_depth(depth))
            max_breadth = max(max_breadth, breadth)
        return max_breadth
    
    def get_statistics(self) -> Dict:
        """Get tree statistics"""
        stats = {
            'total_nodes': len(self.nodes),
            'max_depth': self.get_max_depth(),
            'max_breadth': self.get_max_breadth(),
            'leaf_nodes': len(self.get_leaves()),
            'avg_branching_factor': 0.0
        }
        
        if len(self.nodes) > 1:
            total_children = sum(len(children) for children in self.edges.values())
            non_leaf_nodes = len(self.nodes) - len(self.get_leaves())
            if non_leaf_nodes > 0:
                stats['avg_branching_factor'] = total_children / non_leaf_nodes
        
        return stats

    def get_subtree(self, thought_id: str) -> 'ThoughtTree':
        """Get subtree rooted at the specified thought"""
        if thought_id not in self.nodes:
            return ThoughtTree()
            
        subtree = ThoughtTree(self.max_depth)
        root_thought = self.nodes[thought_id]
        
        # BFS to collect all descendants
        queue = [thought_id]
        visited = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
                
            visited.add(current_id)
            current_thought = self.nodes[current_id]
            
            # Adjust depth relative to new root
            adjusted_thought = Thought(
                content=current_thought.content,
                thought_type=current_thought.thought_type,
                parent_id=current_thought.parent_id if current_thought.id != thought_id else None,
                depth=current_thought.depth - root_thought.depth,
                confidence=current_thought.confidence,
                metadata=current_thought.metadata.copy()
            )
            adjusted_thought.id = current_thought.id
            adjusted_thought.created_at = current_thought.created_at
            adjusted_thought.evaluation_scores = current_thought.evaluation_scores.copy()
            
            subtree.add_thought(adjusted_thought)
            
            # Add children to queue
            children = self.get_children(current_id)
            for child in children:
                queue.append(child.id)
                
        return subtree
        
    def validate_tree(self) -> List[str]:
        """Validate tree structure and return list of issues found"""
        issues = []
        
        # Check for orphaned nodes
        for thought_id, thought in self.nodes.items():
            if thought.parent_id and thought.parent_id not in self.nodes:
                issues.append(f"Thought {thought_id} has missing parent {thought.parent_id}")
                
        # Check for circular references
        visited = set()
        for thought_id in self.nodes:
            if thought_id not in visited:
                path = []
                current_id = thought_id
                
                while current_id and current_id not in visited:
                    if current_id in path:
                        issues.append(f"Circular reference detected: {' -> '.join(path + [current_id])}")
                        break
                    path.append(current_id)
                    thought = self.nodes.get(current_id)
                    current_id = thought.parent_id if thought else None
                    
                visited.update(path)
                
        # Check depth consistency
        for thought in self.nodes.values():
            if thought.parent_id:
                parent = self.nodes.get(thought.parent_id)
                if parent and thought.depth != parent.depth + 1:
                    issues.append(f"Depth inconsistency: thought {thought.id} depth {thought.depth}, parent depth {parent.depth}")
                    
        return issues
        
    def to_dict(self) -> Dict:
        """Convert tree to dictionary representation"""
        return {
            'root_id': self.root.id if self.root else None,
            'max_depth': self.max_depth,
            'nodes': {tid: thought.to_dict() for tid, thought in self.nodes.items()},
            'edges': self.edges
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ThoughtTree':
        """Create tree from dictionary representation"""
        tree = cls(max_depth=data.get('max_depth', 10))
        
        # Restore thoughts
        for thought_data in data.get('nodes', {}).values():
            thought = Thought.from_dict(thought_data)
            tree.add_thought(thought)
            
        # Root will be set automatically during add_thought
        return tree
        
    def __len__(self) -> int:
        """Return number of thoughts in tree"""
        return len(self.nodes)
        
    def __repr__(self) -> str:
        return f"ThoughtTree(nodes={len(self.nodes)}, max_depth={self.max_depth})"
