#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code-Specific Retriever - Phase 2.1
Observer-approved code documentation retrieval with Python/JavaScript/TypeScript indexing

Provides specialized retrieval for coding tasks with language-specific
documentation indexing and code pattern matching.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

# Core imports
from ...storage.vector import VectorStoreManager, VectorQuery, VectorDocument
from ...utils.embedding import get_embedding_service
from ...rag.document_processor import DocumentProcessor, ProcessedDocument

logger = logging.getLogger(__name__)


@dataclass
class CodeDocument:
    """Code-specific document with language and pattern metadata"""
    content: str
    title: str
    language: str
    doc_type: str  # 'function', 'class', 'module', 'example', 'documentation'
    patterns: List[str]  # Code patterns found in document
    complexity_score: float
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/retrieval"""
        return {
            'content': self.content,
            'title': self.title,
            'language': self.language,
            'doc_type': self.doc_type,
            'patterns': self.patterns,
            'complexity_score': self.complexity_score,
            'relevance_score': self.relevance_score,
            'metadata': self.metadata or {}
        }


class CodeRetriever:
    """
    Code-Specific Document Retriever
    
    Specialized retriever for coding tasks that understands programming
    languages, code patterns, and provides context-aware documentation.
    """
    
    def __init__(self, 
                 vector_store_manager: VectorStoreManager,
                 embedding_service=None):
        """
        Initialize code retriever
        
        Args:
            vector_store_manager: Vector store for document storage/retrieval
            embedding_service: Service for generating embeddings
        """
        self.vector_store_manager = vector_store_manager
        self.embedding_service = embedding_service or get_embedding_service()
        
        # Supported programming languages
        self.supported_languages = {
            'python': {
                'extensions': ['.py', '.pyx', '.pyi'],
                'patterns': [
                    r'def\s+(\w+)',      # Functions
                    r'class\s+(\w+)',    # Classes
                    r'import\s+(\w+)',   # Imports
                    r'from\s+(\w+)',     # From imports
                    r'@(\w+)',           # Decorators
                ]
            },
            'javascript': {
                'extensions': ['.js', '.jsx', '.mjs'],
                'patterns': [
                    r'function\s+(\w+)',     # Functions
                    r'const\s+(\w+)\s*=',    # Constants
                    r'let\s+(\w+)\s*=',      # Variables
                    r'class\s+(\w+)',        # Classes
                    r'import\s+.*from',      # Imports
                ]
            },
            'typescript': {
                'extensions': ['.ts', '.tsx'],
                'patterns': [
                    r'function\s+(\w+)',     # Functions
                    r'const\s+(\w+):\s*',    # Typed constants
                    r'interface\s+(\w+)',    # Interfaces
                    r'type\s+(\w+)\s*=',     # Type aliases
                    r'class\s+(\w+)',        # Classes
                ]
            }
        }
        
        # Code documentation collections
        self.code_collections = {
            'python_docs': 'python_documentation',
            'js_docs': 'javascript_documentation', 
            'ts_docs': 'typescript_documentation',
            'code_examples': 'code_examples',
            'api_docs': 'api_documentation'
        }
        
        # Document processor for code files
        self.document_processor = None
        
        # Performance tracking
        self.retrieval_stats = {
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "average_retrieval_time_ms": 0.0,
            "language_distribution": {},
            "pattern_matches": {}
        }
        
        # Initialization flag
        self._initialized = False
        
        logger.info("Code Retriever initialized")
    
    async def initialize(self) -> None:
        """Initialize code retriever components"""
        if self._initialized:
            return
        
        try:
            # Initialize embedding service
            if hasattr(self.embedding_service, 'initialize'):
                await self.embedding_service.initialize()
            
            # Initialize vector store manager
            if not self.vector_store_manager._initialized:
                await self.vector_store_manager.initialize()
            
            # Initialize document processor
            self.document_processor = DocumentProcessor(
                embedding_service=self.embedding_service
            )
            await self.document_processor.initialize()
            
            # Ensure code collections exist
            await self._ensure_collections_exist()
            
            self._initialized = True
            logger.info("Code Retriever initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Code Retriever: {e}")
            raise
    
    async def retrieve(self, query: str, k: int = 5, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve code-relevant documents for a query
        
        Args:
            query: Search query (coding question/task)
            k: Number of documents to retrieve
            language: Specific programming language to focus on
            
        Returns:
            List of relevant code documents with scores
        """
        import time
        start_time = time.time()
        self.retrieval_stats["total_retrievals"] += 1
        
        try:
            # Ensure initialization
            if not self._initialized:
                await self.initialize()
            
            # Detect language from query if not specified
            if not language:
                language = self._detect_language_from_query(query)
            
            # Update language distribution stats
            if language:
                self.retrieval_stats["language_distribution"][language] = (
                    self.retrieval_stats["language_distribution"].get(language, 0) + 1
                )
            
            # Extract code patterns from query
            query_patterns = self._extract_code_patterns(query, language)
            
            # Generate embedding for query
            query_embedding = await self.embedding_service.get_embedding(query)
            if not query_embedding:
                logger.warning("Failed to generate embedding for query")
                return []
            
            # Determine collections to search
            collections = self._get_relevant_collections(language)
            
            # Perform vector search across relevant collections
            all_results = []
            for collection in collections:
                try:
                    vector_query = VectorQuery(
                        query_vector=query_embedding,
                        collection=collection,
                        limit=k * 2,  # Get more results for filtering
                        similarity_threshold=0.3
                    )
                    
                    results = await self.vector_store_manager.search_similar(vector_query)
                    
                    # Convert to code documents and add collection info
                    for result in results:
                        doc_dict = result.document.to_dict() if hasattr(result.document, 'to_dict') else {
                            'content': result.document.content,
                            'title': getattr(result.document, 'title', 'Unknown'),
                            'metadata': getattr(result.document, 'metadata', {})
                        }
                        doc_dict['score'] = result.similarity_score
                        doc_dict['collection'] = collection
                        all_results.append(doc_dict)
                        
                except Exception as e:
                    logger.warning(f"Search failed for collection {collection}: {e}")
                    continue
            
            # Rank and filter results
            ranked_results = self._rank_code_results(all_results, query, query_patterns, language)
            
            # Limit to requested number
            final_results = ranked_results[:k]
            
            # Update statistics
            retrieval_time_ms = (time.time() - start_time) * 1000
            self._update_retrieval_stats(retrieval_time_ms, len(final_results))
            self.retrieval_stats["successful_retrievals"] += 1
            
            logger.debug(f"Code retrieval completed: {len(final_results)} documents in {retrieval_time_ms:.2f}ms")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Code retrieval failed: {e}")
            return []
    
    def _detect_language_from_query(self, query: str) -> Optional[str]:
        """Detect programming language from query text"""
        query_lower = query.lower()
        
        # Language-specific keywords
        language_keywords = {
            'python': ['python', 'def ', 'import ', 'class ', 'pip', 'django', 'flask', 'pandas'],
            'javascript': ['javascript', 'js', 'function', 'const ', 'let ', 'var ', 'npm', 'node'],
            'typescript': ['typescript', 'ts', 'interface', 'type ', 'generic', 'angular', 'react']
        }
        
        # Score each language
        language_scores = {}
        for lang, keywords in language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                language_scores[lang] = score
        
        # Return language with highest score
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        return None
    
    def _extract_code_patterns(self, query: str, language: Optional[str]) -> List[str]:
        """Extract code patterns from query"""
        patterns = []
        
        if not language or language not in self.supported_languages:
            return patterns
        
        lang_config = self.supported_languages[language]
        for pattern_regex in lang_config['patterns']:
            matches = re.findall(pattern_regex, query, re.IGNORECASE)
            patterns.extend(matches)
        
        # Update pattern match statistics
        for pattern in patterns:
            self.retrieval_stats["pattern_matches"][pattern] = (
                self.retrieval_stats["pattern_matches"].get(pattern, 0) + 1
            )
        
        return patterns
    
    def _get_relevant_collections(self, language: Optional[str]) -> List[str]:
        """Get relevant collections based on language"""
        collections = []
        
        if language == 'python':
            collections.extend(['python_docs', 'code_examples', 'api_docs'])
        elif language == 'javascript':
            collections.extend(['js_docs', 'code_examples', 'api_docs'])
        elif language == 'typescript':
            collections.extend(['ts_docs', 'js_docs', 'code_examples', 'api_docs'])
        else:
            # Default: search all collections
            collections = list(self.code_collections.keys())
        
        # Map to actual collection names
        return [self.code_collections.get(col, col) for col in collections]
    
    def _rank_code_results(self, 
                          results: List[Dict[str, Any]], 
                          query: str, 
                          query_patterns: List[str],
                          language: Optional[str]) -> List[Dict[str, Any]]:
        """Rank code results based on relevance factors"""
        
        for result in results:
            base_score = result.get('score', 0.0)
            
            # Language match bonus
            doc_language = result.get('metadata', {}).get('language', '')
            if language and doc_language.lower() == language.lower():
                base_score += 0.2
            
            # Pattern match bonus
            content = result.get('content', '').lower()
            pattern_matches = sum(1 for pattern in query_patterns if pattern.lower() in content)
            if pattern_matches > 0:
                base_score += 0.1 * pattern_matches
            
            # Document type bonus (prefer examples and documentation)
            doc_type = result.get('metadata', {}).get('doc_type', '')
            if doc_type in ['example', 'documentation', 'tutorial']:
                base_score += 0.15
            elif doc_type in ['function', 'class']:
                base_score += 0.1
            
            # Update final score
            result['relevance_score'] = base_score
        
        # Sort by relevance score (descending)
        return sorted(results, key=lambda x: x.get('relevance_score', 0.0), reverse=True)
    
    async def _ensure_collections_exist(self) -> None:
        """Ensure all code collections exist in vector store"""
        try:
            for collection_name in self.code_collections.values():
                # This would typically create collections if they don't exist
                # For now, we'll just log the collections we expect
                logger.debug(f"Ensuring collection exists: {collection_name}")
                
        except Exception as e:
            logger.warning(f"Failed to ensure collections exist: {e}")
    
    def _update_retrieval_stats(self, retrieval_time_ms: float, result_count: int) -> None:
        """Update retrieval performance statistics"""
        total = self.retrieval_stats["total_retrievals"]
        
        # Update average retrieval time
        current_avg_time = self.retrieval_stats["average_retrieval_time_ms"]
        self.retrieval_stats["average_retrieval_time_ms"] = (
            (current_avg_time * (total - 1) + retrieval_time_ms) / total
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current retrieval statistics"""
        return {
            **self.retrieval_stats,
            "success_rate": (
                self.retrieval_stats["successful_retrievals"] / 
                max(1, self.retrieval_stats["total_retrievals"])
            ),
            "supported_languages": list(self.supported_languages.keys()),
            "collections": list(self.code_collections.values()),
            "initialized": self._initialized
        }
    
    async def shutdown(self) -> None:
        """Shutdown code retriever and cleanup resources"""
        try:
            if self.document_processor:
                await self.document_processor.shutdown()
            
            logger.info("Code Retriever shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Code Retriever shutdown: {e}")
