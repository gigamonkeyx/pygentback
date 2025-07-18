#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple RAG Augmentation Engine - Phase 2.1 (Dependency-Free)
Observer-approved RAG integration without heavy dependencies

Provides RAG augmentation functionality without requiring transformers library
or other heavy dependencies that may have compatibility issues.
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SimpleRAGResult:
    """Result of simple RAG augmentation process"""
    original_prompt: str
    augmented_prompt: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_time_ms: float
    relevance_scores: List[float]
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleEmbeddingService:
    """Simple embedding service that doesn't require transformers"""
    
    def __init__(self):
        self._initialized = False
    
    async def initialize(self):
        """Initialize the embedding service"""
        self._initialized = True
        logger.info("Simple embedding service initialized")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate a simple hash-based embedding"""
        if not self._initialized:
            await self.initialize()
        
        # Create a simple hash-based embedding
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to normalized float vector
        embedding = []
        for byte in hash_bytes:
            embedding.append((byte / 255.0) * 2.0 - 1.0)  # Normalize to [-1, 1]
        
        # Pad to standard embedding size (384 dimensions)
        while len(embedding) < 384:
            embedding.extend(embedding[:min(len(embedding), 384 - len(embedding))])
        
        return embedding[:384]
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings


class SimpleCodeRetriever:
    """Simple code retriever without heavy dependencies"""
    
    def __init__(self):
        self._initialized = False
        self.mock_documents = self._create_mock_code_documents()
        self.retrieval_stats = {
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "average_retrieval_time_ms": 0.0
        }
    
    def _create_mock_code_documents(self) -> List[Dict[str, Any]]:
        """Create mock code documents for testing"""
        return [
            {
                "content": """def reverse_string(s):
    \"\"\"Reverse a string using slicing\"\"\"
    return s[::-1]

# Example usage
text = "hello world"
reversed_text = reverse_string(text)
print(reversed_text)  # Output: dlrow olleh""",
                "title": "String Reversal Function",
                "language": "python",
                "doc_type": "function",
                "score": 0.9
            },
            {
                "content": """def fibonacci(n):
    \"\"\"Calculate fibonacci number using recursion\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    \"\"\"Calculate fibonacci number iteratively\"\"\"
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
                "title": "Fibonacci Functions",
                "language": "python",
                "doc_type": "function",
                "score": 0.85
            },
            {
                "content": """class BinaryTree:
    \"\"\"Simple binary tree implementation\"\"\"
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)""",
                "title": "Binary Tree Class",
                "language": "python",
                "doc_type": "class",
                "score": 0.8
            },
            {
                "content": """function validateEmail(email) {
    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return emailRegex.test(email);
}

// Example usage
console.log(validateEmail("test@example.com")); // true
console.log(validateEmail("invalid-email"));    // false""",
                "title": "Email Validation Function",
                "language": "javascript",
                "doc_type": "function",
                "score": 0.75
            },
            {
                "content": """interface User {
    id: number;
    name: string;
    email: string;
    isActive: boolean;
}

class UserManager {
    private users: User[] = [];
    
    addUser(user: User): void {
        this.users.push(user);
    }
    
    getUserById(id: number): User | undefined {
        return this.users.find(user => user.id === id);
    }
}""",
                "title": "TypeScript User Interface",
                "language": "typescript",
                "doc_type": "interface",
                "score": 0.7
            }
        ]
    
    async def initialize(self):
        """Initialize the code retriever"""
        self._initialized = True
        logger.info("Simple code retriever initialized")
    
    async def retrieve(self, query: str, k: int = 5, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant code documents"""
        start_time = time.time()
        self.retrieval_stats["total_retrievals"] += 1
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Simple keyword-based matching
            query_lower = query.lower()
            scored_docs = []
            
            for doc in self.mock_documents:
                score = 0.0
                
                # Language matching
                if language and doc.get("language", "").lower() == language.lower():
                    score += 0.3
                
                # Content matching
                content_lower = doc["content"].lower()
                title_lower = doc["title"].lower()
                
                # Simple keyword scoring
                keywords = query_lower.split()
                for keyword in keywords:
                    if keyword in content_lower:
                        score += 0.2
                    if keyword in title_lower:
                        score += 0.3
                
                # Base relevance score
                score += doc.get("score", 0.0) * 0.2
                
                if score > 0.1:  # Minimum relevance threshold
                    doc_copy = doc.copy()
                    doc_copy["relevance_score"] = score
                    scored_docs.append(doc_copy)
            
            # Sort by relevance and limit results
            scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = scored_docs[:k]
            
            # Update statistics
            retrieval_time_ms = (time.time() - start_time) * 1000
            self._update_stats(retrieval_time_ms)
            self.retrieval_stats["successful_retrievals"] += 1
            
            logger.debug(f"Retrieved {len(results)} documents in {retrieval_time_ms:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _update_stats(self, retrieval_time_ms: float):
        """Update retrieval statistics"""
        total = self.retrieval_stats["total_retrievals"]
        current_avg = self.retrieval_stats["average_retrieval_time_ms"]
        self.retrieval_stats["average_retrieval_time_ms"] = (
            (current_avg * (total - 1) + retrieval_time_ms) / total
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            **self.retrieval_stats,
            "success_rate": (
                self.retrieval_stats["successful_retrievals"] / 
                max(1, self.retrieval_stats["total_retrievals"])
            ),
            "initialized": self._initialized,
            "document_count": len(self.mock_documents)
        }


class SimpleRAGAugmenter:
    """Simple RAG augmenter without heavy dependencies"""
    
    def __init__(self):
        self.code_retriever = SimpleCodeRetriever()
        self.embedding_service = SimpleEmbeddingService()
        self._initialized = False
        
        # Performance tracking
        self.augmentation_stats = {
            "total_augmentations": 0,
            "successful_augmentations": 0,
            "average_retrieval_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Simple cache
        self.cache = {}
        self.cache_max_size = 50
    
    async def initialize(self):
        """Initialize the RAG augmenter"""
        if self._initialized:
            return
        
        await self.embedding_service.initialize()
        await self.code_retriever.initialize()
        self._initialized = True
        logger.info("Simple RAG augmenter initialized")
    
    async def augment_prompt(self, 
                           prompt: str, 
                           context: Optional[Dict[str, Any]] = None,
                           max_documents: int = 3) -> SimpleRAGResult:
        """Augment a prompt with relevant context"""
        start_time = time.time()
        self.augmentation_stats["total_augmentations"] += 1
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Check cache
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
            if cache_key in self.cache:
                self.augmentation_stats["cache_hits"] += 1
                self.augmentation_stats["successful_augmentations"] += 1  # Count cached results as successful
                return self.cache[cache_key]
            
            self.augmentation_stats["cache_misses"] += 1
            
            # Detect language
            language = self._detect_language(prompt)
            
            # Retrieve relevant documents
            documents = await self.code_retriever.retrieve(
                prompt, k=max_documents, language=language
            )
            
            # Create augmented prompt
            augmented_prompt = self._create_augmented_prompt(prompt, documents)
            
            # Calculate metrics
            retrieval_time_ms = (time.time() - start_time) * 1000
            relevance_scores = [doc.get("relevance_score", 0.0) for doc in documents]
            
            result = SimpleRAGResult(
                original_prompt=prompt,
                augmented_prompt=augmented_prompt,
                retrieved_documents=documents,
                retrieval_time_ms=retrieval_time_ms,
                relevance_scores=relevance_scores,
                success=True,
                metadata={
                    "detected_language": language,
                    "document_count": len(documents),
                    "cache_key": cache_key
                }
            )
            
            # Cache result
            if len(self.cache) < self.cache_max_size:
                self.cache[cache_key] = result
            
            # Update stats
            self._update_stats(retrieval_time_ms)
            self.augmentation_stats["successful_augmentations"] += 1
            
            return result
            
        except Exception as e:
            error_msg = f"Augmentation failed: {e}"
            logger.error(error_msg)
            
            return SimpleRAGResult(
                original_prompt=prompt,
                augmented_prompt=prompt,
                retrieved_documents=[],
                retrieval_time_ms=(time.time() - start_time) * 1000,
                relevance_scores=[],
                success=False,
                error_message=error_msg
            )
    
    def _detect_language(self, prompt: str) -> Optional[str]:
        """Simple language detection"""
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in ["python", "def ", "import ", "class "]):
            return "python"
        elif any(keyword in prompt_lower for keyword in ["javascript", "function", "const ", "let "]):
            return "javascript"
        elif any(keyword in prompt_lower for keyword in ["typescript", "interface", "type "]):
            return "typescript"
        
        return None
    
    def _create_augmented_prompt(self, original_prompt: str, documents: List[Dict[str, Any]]) -> str:
        """Create augmented prompt with context"""
        if not documents:
            return original_prompt
        
        context_parts = []
        for i, doc in enumerate(documents[:3]):  # Limit to top 3
            title = doc.get("title", f"Document {i+1}")
            content = doc.get("content", "")
            context_parts.append(f"[{title}]\n{content}")
        
        context_section = "\n\n".join(context_parts)
        
        return f"""Context Information:
{context_section}

Based on the above context, please respond to the following:
{original_prompt}"""
    
    def _update_stats(self, retrieval_time_ms: float):
        """Update performance statistics"""
        total = self.augmentation_stats["total_augmentations"]
        current_avg = self.augmentation_stats["average_retrieval_time_ms"]
        self.augmentation_stats["average_retrieval_time_ms"] = (
            (current_avg * (total - 1) + retrieval_time_ms) / total
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get augmentation statistics"""
        return {
            **self.augmentation_stats,
            "success_rate": (
                self.augmentation_stats["successful_augmentations"] / 
                max(1, self.augmentation_stats["total_augmentations"])
            ),
            "cache_size": len(self.cache),
            "initialized": self._initialized
        }
    
    async def shutdown(self):
        """Shutdown the augmenter"""
        self.cache.clear()
        logger.info("Simple RAG augmenter shutdown complete")
