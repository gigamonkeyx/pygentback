"""
Vector Search Integration for Tree of Thought

Integrates GPU-accelerated vector search with ToT reasoning to enable
semantic similarity search over thought states and solution paths.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..models import ThoughtState, ThoughtTree
import sys
from pathlib import Path

# Add search module to path
search_path = Path(__file__).parent.parent.parent.parent.parent / "search"
sys.path.insert(0, str(search_path))

# Set up UTF-8 logger
# Use absolute import with fallback
try:
    from .....utils.utf8_logger import get_pygent_logger
    logger = get_pygent_logger("ai_reasoning_vector_search")
except ImportError:
    # Fallback to absolute import
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
        from src.utils.utf8_logger import get_pygent_logger
        logger = get_pygent_logger("ai_reasoning_tot_integrations_vector_search")
    except ImportError:
        import logging
        logger = logging.getLogger("ai_reasoning_tot_integrations_vector_search")

try:
    from gpu_search import create_vector_index, VectorSearchConfig, IndexType, SearchResult
except ImportError:
    # Fallback imports for when GPU search is not available
    VectorSearchConfig = None
    IndexType = None
    SearchResult = None
    def create_vector_index(*args, **kwargs):
        raise NotImplementedError("Vector search not available")

# Logger already defined above with UTF-8 support


@dataclass
class ThoughtEmbedding:
    """Embedding representation of a thought state"""
    thought_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class VectorSearchIntegration:
    """
    Integrates vector search capabilities with Tree of Thought reasoning
    
    Enables:
    - Semantic similarity search over thought states
    - Clustering of similar reasoning paths
    - Retrieval of relevant past solutions
    - Thought state deduplication
    """
    
    def __init__(self, embedding_dim: int = 768, use_gpu: bool = True):
        self.embedding_dim = embedding_dim
        
        # Configure vector search
        self.search_config = VectorSearchConfig(
            index_type=IndexType.IVF_FLAT,
            dimension=embedding_dim,
            nlist=100,
            nprobe=10,
            use_gpu=use_gpu,
            use_float16=True,  # Memory optimization
            batch_size=1000,
            metric_type="L2"
        )
        
        # Create vector index
        self.vector_index = create_vector_index(self.search_config)
        self.thought_embeddings: Dict[str, ThoughtEmbedding] = {}
        self.embedding_to_thought: Dict[int, str] = {}
        self.next_index = 0
        
        # Text encoder (placeholder - would use actual embedding model)
        self.text_encoder = None
        
        logger.info(f"Initialized vector search integration with {embedding_dim}D embeddings")
    
    def encode_thought(self, thought: ThoughtState) -> np.ndarray:
        """
        Encode a thought state into a vector embedding
        
        In a real implementation, this would use a pre-trained embedding model
        like sentence-transformers, OpenAI embeddings, or a custom encoder.
        """
        # Placeholder implementation - would use actual embedding model
        # For now, create a simple hash-based embedding
        text = f"{thought.content} depth:{thought.depth} score:{thought.value_score}"
        
        # Simple hash-based embedding (replace with real embedder)
        hash_val = hash(text)
        embedding = np.random.RandomState(hash_val % (2**31)).normal(0, 1, self.embedding_dim)
        return embedding.astype(np.float32)
    
    def add_thought(self, thought: ThoughtState) -> bool:
        """Add a thought state to the vector index"""
        try:
            # Generate embedding
            embedding = self.encode_thought(thought)
            
            # Create thought embedding object
            thought_embedding = ThoughtEmbedding(
                thought_id=thought.id,
                embedding=embedding,
                metadata={
                    'depth': thought.depth,
                    'value_score': thought.value_score,
                    'is_solution': thought.is_solution,
                    'parent_id': thought.parent_id,
                    'content_length': len(thought.content)
                }
            )
            
            # Store embedding
            self.thought_embeddings[thought.id] = thought_embedding
            self.embedding_to_thought[self.next_index] = thought.id
            
            # Add to vector index
            success = self.vector_index.add(embedding.reshape(1, -1))
            
            if success:
                self.next_index += 1
                logger.debug(f"Added thought {thought.id} to vector index")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add thought to vector index: {e}")
            return False
    
    def add_thought_tree(self, tree: ThoughtTree) -> int:
        """Add all thoughts from a tree to the vector index"""
        added_count = 0
        
        # Train index if not already trained
        if not self.vector_index.trained and len(tree.states) > 0:
            # Collect sample embeddings for training
            sample_thoughts = list(tree.states.values())[:min(1000, len(tree.states))]
            sample_embeddings = []
            
            for thought in sample_thoughts:
                embedding = self.encode_thought(thought)
                sample_embeddings.append(embedding)
            
            if sample_embeddings:
                training_data = np.vstack(sample_embeddings)
                self.vector_index.train(training_data)
        
        # Add all thoughts
        for thought in tree.states.values():
            if self.add_thought(thought):
                added_count += 1
        
        logger.info(f"Added {added_count}/{len(tree.states)} thoughts to vector index")
        return added_count
    
    def find_similar_thoughts(self, query_thought: ThoughtState, 
                            k: int = 5, 
                            min_similarity: float = 0.0) -> List[Tuple[ThoughtState, float]]:
        """Find thoughts similar to the query thought"""
        try:
            # Encode query thought
            query_embedding = self.encode_thought(query_thought)
            
            # Search for similar embeddings
            search_result = self.vector_index.search(
                query_embedding.reshape(1, -1), k=k
            )
            
            # Convert results to thought states
            similar_thoughts = []
            
            if len(search_result.distances) > 0 and len(search_result.indices) > 0:
                distances = search_result.distances[0]
                indices = search_result.indices[0]
                
                for distance, idx in zip(distances, indices):
                    if idx in self.embedding_to_thought:
                        thought_id = self.embedding_to_thought[idx]
                        if thought_id in self.thought_embeddings:
                            # Convert distance to similarity (for L2 distance)
                            similarity = 1.0 / (1.0 + distance)
                            
                            if similarity >= min_similarity:
                                thought_embedding = self.thought_embeddings[thought_id]
                                # Note: We don't have the original ThoughtState here
                                # In practice, you'd store a reference or reconstruct it
                                similar_thoughts.append((thought_embedding, similarity))
            
            return similar_thoughts
            
        except Exception as e:
            logger.error(f"Failed to find similar thoughts: {e}")
            return []
    
    def find_similar_solutions(self, query_thought: ThoughtState, 
                             k: int = 3) -> List[Tuple[str, float]]:
        """Find solution thoughts similar to the query"""
        # Filter for solution thoughts only
        solution_embeddings = [
            emb for emb in self.thought_embeddings.values() 
            if emb.metadata.get('is_solution', False)
        ]
        
        if not solution_embeddings:
            return []
        
        try:
            # Create temporary index for solutions only
            solution_vectors = np.vstack([emb.embedding for emb in solution_embeddings])
            solution_ids = [emb.thought_id for emb in solution_embeddings]
            
            # Encode query
            query_embedding = self.encode_thought(query_thought)
            
            # Simple similarity search (could use separate index)
            distances = np.linalg.norm(solution_vectors - query_embedding, axis=1)
            top_k_indices = np.argpartition(distances, min(k, len(distances)-1))[:k]
            top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
            
            similar_solutions = []
            for idx in top_k_indices:
                distance = distances[idx]
                similarity = 1.0 / (1.0 + distance)
                thought_id = solution_ids[idx]
                similar_solutions.append((thought_id, similarity))
            
            return similar_solutions
            
        except Exception as e:
            logger.error(f"Failed to find similar solutions: {e}")
            return []
    
    def cluster_thoughts(self, thought_ids: List[str], 
                        n_clusters: int = 5) -> Dict[int, List[str]]:
        """Cluster thoughts by semantic similarity"""
        try:
            # Get embeddings for specified thoughts
            embeddings = []
            valid_ids = []
            
            for thought_id in thought_ids:
                if thought_id in self.thought_embeddings:
                    embeddings.append(self.thought_embeddings[thought_id].embedding)
                    valid_ids.append(thought_id)
            
            if len(embeddings) < n_clusters:
                # Not enough thoughts to cluster
                return {0: valid_ids}
            
            # Simple k-means clustering (could use more sophisticated methods)
            from sklearn.cluster import KMeans
            
            embeddings_array = np.vstack(embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Group thoughts by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_ids[i])
            
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to cluster thoughts: {e}")
            return {0: thought_ids}
    
    def deduplicate_thoughts(self, thoughts: List[ThoughtState], 
                           similarity_threshold: float = 0.95) -> List[ThoughtState]:
        """Remove duplicate or highly similar thoughts"""
        if len(thoughts) <= 1:
            return thoughts
        
        try:
            # Encode all thoughts
            embeddings = []
            for thought in thoughts:
                embedding = self.encode_thought(thought)
                embeddings.append(embedding)
            
            embeddings_array = np.vstack(embeddings)
            
            # Find duplicates using pairwise similarity
            unique_thoughts = []
            used_indices = set()
            
            for i, thought in enumerate(thoughts):
                if i in used_indices:
                    continue
                
                unique_thoughts.append(thought)
                current_embedding = embeddings_array[i]
                
                # Mark similar thoughts as used
                for j in range(i + 1, len(thoughts)):
                    if j in used_indices:
                        continue
                    
                    other_embedding = embeddings_array[j]
                    distance = np.linalg.norm(current_embedding - other_embedding)
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= similarity_threshold:
                        used_indices.add(j)
            
            logger.info(f"Deduplicated {len(thoughts)} thoughts to {len(unique_thoughts)}")
            return unique_thoughts
            
        except Exception as e:
            logger.error(f"Failed to deduplicate thoughts: {e}")
            return thoughts
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get vector search statistics"""
        base_stats = self.vector_index.get_stats()
        
        integration_stats = {
            'total_thoughts': len(self.thought_embeddings),
            'solution_thoughts': sum(
                1 for emb in self.thought_embeddings.values() 
                if emb.metadata.get('is_solution', False)
            ),
            'embedding_dimension': self.embedding_dim,
            'search_config': {
                'index_type': self.search_config.index_type.value,
                'use_gpu': self.search_config.use_gpu,
                'nlist': self.search_config.nlist,
                'nprobe': self.search_config.nprobe
            }
        }
        
        return {**base_stats, **integration_stats}
    
    def save_index(self, filepath: str) -> bool:
        """Save the vector index to disk"""
        return self.vector_index.save(filepath)
    
    def load_index(self, filepath: str) -> bool:
        """Load the vector index from disk"""
        return self.vector_index.load(filepath)
    
    def cleanup(self):
        """Clean up resources"""
        self.vector_index.cleanup()
        self.thought_embeddings.clear()
        self.embedding_to_thought.clear()
