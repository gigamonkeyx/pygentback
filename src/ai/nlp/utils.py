"""
NLP Utilities Module

Utility components for semantic analysis, embeddings, and similarity calculations.
"""

import logging
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .core import CacheManager, ConfidenceCalculator
from .models import SemanticEmbedding, SimilarityScore

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Simple embedding engine for text representation.
    
    In production, this would integrate with models like BERT, GPT, or sentence-transformers.
    For now, implements basic TF-IDF style embeddings.
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.vocabulary: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.cache = CacheManager(max_size=1000)
        
        # Document frequency tracking
        self.document_count = 0
        self.term_document_freq: Dict[str, int] = {}
    
    def fit_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        all_terms = set()
        
        for text in texts:
            terms = self._tokenize(text)
            unique_terms = set(terms)
            
            # Update document frequency
            for term in unique_terms:
                self.term_document_freq[term] = self.term_document_freq.get(term, 0) + 1
            
            all_terms.update(terms)
        
        # Build vocabulary
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
        self.document_count = len(texts)
        
        # Calculate IDF scores
        self._calculate_idf_scores()
        
        logger.info(f"Built vocabulary with {len(self.vocabulary)} terms from {len(texts)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _calculate_idf_scores(self):
        """Calculate IDF scores for terms"""
        for term, doc_freq in self.term_document_freq.items():
            # IDF = log(total_docs / doc_freq)
            self.idf_scores[term] = math.log(self.document_count / doc_freq)
    
    def create_embedding(self, text: str, model_name: str = "tfidf") -> SemanticEmbedding:
        """Create embedding for text"""
        # Check cache
        cache_key = f"{model_name}:{hash(text)}"
        cached_embedding = self.cache.get(cache_key)
        
        if cached_embedding:
            return cached_embedding
        
        # Tokenize text
        terms = self._tokenize(text)
        
        # Create TF-IDF vector
        embedding = self._create_tfidf_vector(terms)
        
        # Create embedding object
        semantic_embedding = SemanticEmbedding(
            text=text,
            embedding=embedding.tolist(),
            model_name=model_name,
            embedding_dimension=len(embedding)
        )
        
        # Cache result
        self.cache.set(cache_key, semantic_embedding)
        
        return semantic_embedding
    
    def _create_tfidf_vector(self, terms: List[str]) -> np.ndarray:
        """Create TF-IDF vector from terms"""
        # Initialize vector
        vector = np.zeros(self.embedding_dim)
        
        if not self.vocabulary:
            # If no vocabulary, create simple hash-based embedding
            return self._create_hash_embedding(terms)
        
        # Calculate term frequencies
        term_freq = {}
        for term in terms:
            term_freq[term] = term_freq.get(term, 0) + 1
        
        # Create TF-IDF vector
        for term, freq in term_freq.items():
            if term in self.vocabulary:
                vocab_idx = self.vocabulary[term]
                # Map vocabulary index to embedding dimension
                embed_idx = vocab_idx % self.embedding_dim
                
                # TF-IDF score
                tf = freq / len(terms)  # Term frequency
                idf = self.idf_scores.get(term, 1.0)  # Inverse document frequency
                tfidf = tf * idf
                
                vector[embed_idx] += tfidf
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _create_hash_embedding(self, terms: List[str]) -> np.ndarray:
        """Create simple hash-based embedding"""
        vector = np.zeros(self.embedding_dim)
        
        for term in terms:
            # Simple hash function
            hash_val = hash(term) % self.embedding_dim
            vector[hash_val] += 1.0
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def batch_create_embeddings(self, texts: List[str], model_name: str = "tfidf") -> List[SemanticEmbedding]:
        """Create embeddings for multiple texts"""
        embeddings = []
        
        for text in texts:
            embedding = self.create_embedding(text, model_name)
            embeddings.append(embedding)
        
        return embeddings


class SimilarityCalculator:
    """
    Calculates similarity between texts and embeddings.
    """
    
    def __init__(self):
        self.cache = CacheManager(max_size=500)
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def calculate_euclidean_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate Euclidean similarity between embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
        
        # Convert to similarity (0-1 range)
        # Using exponential decay: similarity = exp(-distance)
        similarity = math.exp(-distance)
        return float(similarity)
    
    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between texts"""
        # Tokenize texts
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_text_similarity(self, text1: str, text2: str, 
                                similarity_type: str = "cosine",
                                embedding_engine: Optional[EmbeddingEngine] = None) -> SimilarityScore:
        """Calculate similarity between two texts"""
        # Check cache
        cache_key = f"{similarity_type}:{hash(text1)}:{hash(text2)}"
        cached_score = self.cache.get(cache_key)
        
        if cached_score:
            return cached_score
        
        # Calculate similarity based on type
        if similarity_type == "jaccard":
            score = self.calculate_jaccard_similarity(text1, text2)
        elif similarity_type in ["cosine", "euclidean"] and embedding_engine:
            # Create embeddings
            emb1 = embedding_engine.create_embedding(text1)
            emb2 = embedding_engine.create_embedding(text2)
            
            if similarity_type == "cosine":
                score = self.calculate_cosine_similarity(emb1.embedding, emb2.embedding)
            else:  # euclidean
                score = self.calculate_euclidean_similarity(emb1.embedding, emb2.embedding)
        else:
            # Default to simple word overlap
            score = self._calculate_word_overlap(text1, text2)
        
        # Create similarity score object
        similarity_score = SimilarityScore(
            text1=text1,
            text2=text2,
            score=score,
            similarity_type=similarity_type,
            metadata={'calculated_at': datetime.utcnow().isoformat()}
        )
        
        # Cache result
        self.cache.set(cache_key, similarity_score)
        
        return similarity_score
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total if total > 0 else 0.0
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str],
                         similarity_type: str = "cosine",
                         embedding_engine: Optional[EmbeddingEngine] = None,
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar texts to query"""
        similarities = []
        
        for candidate in candidate_texts:
            similarity_score = self.calculate_text_similarity(
                query_text, candidate, similarity_type, embedding_engine
            )
            similarities.append((candidate, similarity_score.score))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class SemanticAnalyzer:
    """
    High-level semantic analysis combining embeddings and similarity.
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_engine = EmbeddingEngine(embedding_dim)
        self.similarity_calculator = SimilarityCalculator()
        
        # Analysis cache
        self.cache = CacheManager(max_size=200)
        
        # Semantic clusters
        self.clusters: Dict[str, List[str]] = {}
    
    def analyze_text_collection(self, texts: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze a collection of texts for semantic patterns"""
        if not texts:
            return {'error': 'No texts provided'}
        
        # Fit vocabulary
        self.embedding_engine.fit_vocabulary(texts)
        
        # Create embeddings
        embeddings = self.embedding_engine.batch_create_embeddings(texts)
        
        # Calculate pairwise similarities
        similarity_matrix = self._calculate_similarity_matrix(embeddings)
        
        # Find clusters
        clusters = self._find_semantic_clusters(texts, similarity_matrix)
        
        # Calculate statistics
        stats = self._calculate_collection_stats(texts, embeddings, similarity_matrix)
        
        return {
            'text_count': len(texts),
            'vocabulary_size': len(self.embedding_engine.vocabulary),
            'clusters': clusters,
            'statistics': stats,
            'embeddings': [emb.to_dict() if hasattr(emb, 'to_dict') else emb.__dict__ for emb in embeddings]
        }
    
    def _calculate_similarity_matrix(self, embeddings: List[SemanticEmbedding]) -> np.ndarray:
        """Calculate similarity matrix for embeddings"""
        n = len(embeddings)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = self.similarity_calculator.calculate_cosine_similarity(
                        embeddings[i].embedding, embeddings[j].embedding
                    )
                
                matrix[i, j] = similarity
                matrix[j, i] = similarity  # Symmetric matrix
        
        return matrix
    
    def _find_semantic_clusters(self, texts: List[str], similarity_matrix: np.ndarray,
                              threshold: float = 0.7) -> Dict[str, List[str]]:
        """Find semantic clusters using simple threshold-based clustering"""
        n = len(texts)
        clusters = {}
        assigned = set()
        cluster_id = 0
        
        for i in range(n):
            if i in assigned:
                continue
            
            # Start new cluster
            cluster_name = f"cluster_{cluster_id}"
            cluster_texts = [texts[i]]
            assigned.add(i)
            
            # Find similar texts
            for j in range(i + 1, n):
                if j not in assigned and similarity_matrix[i, j] >= threshold:
                    cluster_texts.append(texts[j])
                    assigned.add(j)
            
            if len(cluster_texts) > 1:  # Only keep clusters with multiple texts
                clusters[cluster_name] = cluster_texts
                cluster_id += 1
        
        return clusters
    
    def _calculate_collection_stats(self, texts: List[str], embeddings: List[SemanticEmbedding],
                                  similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for text collection"""
        # Text length statistics
        text_lengths = [len(text.split()) for text in texts]
        
        # Similarity statistics
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        # Embedding statistics
        embedding_norms = [np.linalg.norm(emb.embedding) for emb in embeddings]
        
        return {
            'text_length': {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths)
            },
            'similarity': {
                'mean': np.mean(upper_triangle),
                'std': np.std(upper_triangle),
                'min': np.min(upper_triangle),
                'max': np.max(upper_triangle)
            },
            'embedding_norms': {
                'mean': np.mean(embedding_norms),
                'std': np.std(embedding_norms)
            }
        }
    
    def find_semantic_matches(self, query: str, candidate_texts: List[str],
                            threshold: float = 0.5, max_results: int = 10) -> List[Dict[str, Any]]:
        """Find semantically similar texts to query"""
        if not candidate_texts:
            return []
        
        # Ensure vocabulary includes all texts
        all_texts = [query] + candidate_texts
        self.embedding_engine.fit_vocabulary(all_texts)
        
        # Find similar texts
        similar_texts = self.similarity_calculator.find_most_similar(
            query, candidate_texts, "cosine", self.embedding_engine, max_results
        )
        
        # Filter by threshold and format results
        results = []
        for text, score in similar_texts:
            if score >= threshold:
                results.append({
                    'text': text,
                    'similarity_score': score,
                    'confidence': self._calculate_match_confidence(score, query, text)
                })
        
        return results
    
    def _calculate_match_confidence(self, similarity_score: float, query: str, match_text: str) -> float:
        """Calculate confidence in semantic match"""
        factors = []
        
        # Similarity score factor
        factors.append(similarity_score)
        
        # Text length similarity factor
        query_len = len(query.split())
        match_len = len(match_text.split())
        length_ratio = min(query_len, match_len) / max(query_len, match_len) if max(query_len, match_len) > 0 else 0
        factors.append(length_ratio)
        
        # Word overlap factor
        query_words = set(query.lower().split())
        match_words = set(match_text.lower().split())
        overlap = len(query_words.intersection(match_words))
        total_unique = len(query_words.union(match_words))
        overlap_ratio = overlap / total_unique if total_unique > 0 else 0
        factors.append(overlap_ratio)
        
        return ConfidenceCalculator.combined_confidence(factors, [0.5, 0.2, 0.3])
    
    def get_semantic_summary(self, texts: List[str]) -> Dict[str, Any]:
        """Get semantic summary of text collection"""
        if not texts:
            return {'error': 'No texts provided'}
        
        analysis = self.analyze_text_collection(texts)
        
        # Generate summary
        summary = {
            'total_texts': len(texts),
            'vocabulary_size': analysis['vocabulary_size'],
            'cluster_count': len(analysis['clusters']),
            'avg_similarity': analysis['statistics']['similarity']['mean'],
            'diversity_score': 1.0 - analysis['statistics']['similarity']['mean'],  # Inverse of similarity
            'complexity_score': analysis['statistics']['text_length']['mean'] / 100.0  # Normalized
        }
        
        # Add insights
        insights = []
        
        if summary['cluster_count'] > len(texts) * 0.3:
            insights.append("High semantic diversity - texts cover many different topics")
        elif summary['cluster_count'] < len(texts) * 0.1:
            insights.append("Low semantic diversity - texts are very similar")
        
        if summary['avg_similarity'] > 0.8:
            insights.append("Very high similarity between texts")
        elif summary['avg_similarity'] < 0.3:
            insights.append("Low similarity between texts - diverse content")
        
        summary['insights'] = insights
        
        return summary
