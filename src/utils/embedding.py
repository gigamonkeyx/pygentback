"""
Embedding Generation Utilities

This module provides utilities for generating embeddings using various models
including OpenAI, Sentence Transformers, and local models. It supports
batch processing, caching, and multiple embedding providers.
"""

import asyncio
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

import openai
import ollama
from sentence_transformers import SentenceTransformer

# Use try/except for flexible import
try:
    from config.settings import Settings
except ImportError:
    try:
        from ..config.settings import Settings
    except ImportError:
        # Fallback for when config is not available
        class Settings:
            class ai:
                EMBEDDING_MODEL = "all-MiniLM-L6-v2"
                OPENAI_API_KEY = None
                EMBEDDING_DIMENSION = 384


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text: str
    embedding: List[float]
    model: str
    timestamp: datetime
    token_count: Optional[int] = None
    processing_time: Optional[float] = None


class EmbeddingCache:
    """Simple in-memory cache for embeddings"""
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[List[float], datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model"""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        key = self._get_cache_key(text, model)
        
        if key in self.cache:
            embedding, timestamp = self.cache[key]
            
            # Check if expired
            if datetime.utcnow() - timestamp > self.ttl:
                del self.cache[key]
                return None
            
            return embedding
        
        return None
    
    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Set embedding in cache"""
        key = self._get_cache_key(text, model)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (embedding, datetime.utcnow())
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class OpenAIEmbeddingProvider:
    """OpenAI and compatible embedding providers (like OpenRouter)"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"Initialized OpenAI-compatible provider with model: {model}, base_url: {base_url or 'Default'}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings using OpenAI API"""
        try:
            start_time = datetime.utcnow()
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            results = []
            for i, text in enumerate(texts):
                embedding_data = response.data[i]
                
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding_data.embedding,
                    model=self.model,
                    timestamp=datetime.utcnow(),
                    token_count=response.usage.total_tokens if hasattr(response, 'usage') else None,
                    processing_time=processing_time / len(texts)
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {str(e)}")
            raise


class FastEmbeddingProvider:
    """Fast embedding provider using deterministic algorithms (no external dependencies)"""

    def __init__(self, model_name: str = "fast-deterministic-v1"):
        self.model_name = model_name
        self.dimension = 384
        self._initialized = True
        logger.info(f"Fast embedding provider initialized: {self.model_name}")

    def _generate_fast_embedding(self, text: str) -> List[float]:
        """Generate fast deterministic embedding without external dependencies"""
        import hashlib
        import math

        # Create multiple hash variants for better distribution
        hashes = [
            hashlib.sha256(text.encode()).digest(),
            hashlib.sha256((text + "_variant1").encode()).digest(),
            hashlib.sha256((text + "_variant2").encode()).digest(),
            hashlib.md5(text.encode()).digest()
        ]

        # Convert to float values
        embedding = []
        for hash_bytes in hashes:
            for byte in hash_bytes:
                if len(embedding) >= self.dimension:
                    break
                # Convert byte to float in range [-1, 1]
                embedding.append((byte / 127.5) - 1.0)

        # Ensure exact dimension
        while len(embedding) < self.dimension:
            embedding.append(0.0)
        embedding = embedding[:self.dimension]

        # Add text-length based features for better semantic representation
        text_features = [
            len(text) / 1000.0,  # Text length feature
            len(text.split()) / 100.0,  # Word count feature
            text.count(' ') / 100.0,  # Space count feature
            sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
        ]

        # Blend text features into embedding
        for i, feature in enumerate(text_features):
            if i < len(embedding):
                embedding[i] = (embedding[i] + feature) / 2.0

        # Normalize to unit vector for cosine similarity
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding
    
    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate fast embeddings using deterministic algorithms"""
        try:
            start_time = datetime.now(datetime.timezone.utc)

            # Generate embeddings using fast deterministic method
            results = []
            for text in texts:
                embedding = self._generate_fast_embedding(text)
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=self.model_name,
                    timestamp=datetime.now(datetime.timezone.utc),
                    processing_time=0.001  # Very fast processing
                ))

            total_time = (datetime.now(datetime.timezone.utc) - start_time).total_seconds()
            logger.debug(f"Generated {len(texts)} embeddings in {total_time:.3f}s")

            return results

        except Exception as e:
            logger.error(f"Fast embedding generation failed: {str(e)}")
            # Return zero embeddings on failure
            return [EmbeddingResult(
                text=text,
                embedding=[0.0] * self.dimension,
                model=self.model_name,
                timestamp=datetime.now(datetime.timezone.utc),
                processing_time=0.0,
                error=str(e)
            ) for text in texts]


class SentenceTransformerProvider:
    """Sentence Transformer embedding provider"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized SentenceTransformer with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            # Fallback to fast embedding
            self.model = None

    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings using SentenceTransformer"""
        try:
            if self.model is None:
                # Fallback to fast embedding
                fast_provider = FastEmbeddingProvider(self.model_name)
                return await fast_provider.generate_embeddings(texts)

            start_time = datetime.utcnow()

            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_tensor=False)

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            results = []
            for i, text in enumerate(texts):
                result = EmbeddingResult(
                    text=text,
                    embedding=embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else list(embeddings[i]),
                    model=self.model_name,
                    timestamp=datetime.utcnow(),
                    processing_time=processing_time / len(texts)
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"SentenceTransformer embedding generation failed: {e}")
            # Fallback to fast embedding
            fast_provider = FastEmbeddingProvider(self.model_name)
            return await fast_provider.generate_embeddings(texts)


class OllamaEmbeddingProvider:
    """Ollama embedding provider"""

    def __init__(self, model_name: str, base_url: Optional[str] = None):
        self.model_name = model_name
        self.client = ollama.AsyncClient(host=base_url)
        logger.info(f"Initialized OllamaEmbeddingProvider with model: {model_name} and base_url: {base_url or 'Default'}")

    async def generate_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings using Ollama API in parallel."""

        async def _embed(text: str) -> EmbeddingResult:
            """Helper function to embed a single text and handle errors."""
            try:
                start_time = datetime.utcnow()
                response = await self.client.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                return EmbeddingResult(
                    text=text,
                    embedding=response["embedding"],
                    model=self.model_name,
                    timestamp=datetime.utcnow(),
                    processing_time=processing_time
                )
            except Exception as e:
                logger.error(f"Ollama embedding generation failed for text: '{text[:100]}...'. Error: {e}")
                # Return a result with an empty embedding on failure
                return EmbeddingResult(
                    text=text,
                    embedding=[],
                    model=self.model_name,
                    timestamp=datetime.utcnow()
                )

        # Create a list of coroutine tasks
        tasks = [_embed(text) for text in texts]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        return results


class EmbeddingService:
    """
    Main embedding service that coordinates different providers.
    
    Provides a unified interface for embedding generation with
    automatic provider selection, caching, and batch processing.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        # Import here to avoid circular imports
        try:
            from ..config.settings import get_settings
            self.settings = settings or get_settings()
        except ImportError:
            # Fallback to default settings if config not available
            self.settings = Settings()
        
        self.cache = EmbeddingCache()
        self.providers: Dict[str, Any] = {}
        self.default_provider: Optional[str] = None
        
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize embedding providers"""
        try:
            # Initialize SentenceTransformer provider
            # Recommended models for software development context (set in .env as EMBEDDING_MODEL):
            # - BAAI/bge-small-en-v1.5: Fast, excellent performance, 384 dim. A great default.
            # - nomic-ai/nomic-embed-text-v1: High performance, 768 dim.
            # - thenlper/gte-large: High performance, 1024 dim. (Used by OpenRouter)
            # - all-MiniLM-L6-v2: Fast, small, but older. Good for resource-constrained systems.
            self.providers["fast_embedding"] = FastEmbeddingProvider(
                self.settings.ai.EMBEDDING_MODEL
            )
            
            # Initialize OpenAI provider if API key is available
            if self.settings.ai.OPENAI_API_KEY:
                self.providers["openai"] = OpenAIEmbeddingProvider(
                    api_key=self.settings.ai.OPENAI_API_KEY,
                    model="text-embedding-ada-002",
                    base_url=self.settings.ai.OPENAI_API_BASE
                )

            # Initialize OpenRouter provider if API key is available
            if self.settings.ai.OPENROUTER_API_KEY:
                self.providers["openrouter"] = OpenAIEmbeddingProvider(
                    api_key=self.settings.ai.OPENROUTER_API_KEY,
                    model="thenlper/gte-large", # Recommended embedding model on OpenRouter
                    base_url=self.settings.ai.OPENROUTER_API_BASE
                )

            # Initialize Ollama provider if base URL is available
            if self.settings.ai.OLLAMA_BASE_URL and self.settings.ai.OLLAMA_EMBED_MODEL:
                self.providers["ollama"] = OllamaEmbeddingProvider(
                    model_name=self.settings.ai.OLLAMA_EMBED_MODEL,
                    base_url=self.settings.ai.OLLAMA_BASE_URL
                )

            # Determine the default provider based on settings and availability
            # Priority:
            # 1. Explicitly set DEFAULT_EMBEDDING_PROVIDER from settings
            # 2. Ollama (local, free)
            # 3. OpenRouter (free models available)
            # 4. SentenceTransformer (local, free)
            # 5. OpenAI (paid)
            
            explicit_provider = getattr(self.settings.ai, 'DEFAULT_EMBEDDING_PROVIDER', None)

            if explicit_provider and explicit_provider in self.providers:
                self.default_provider = explicit_provider
            elif "fast_embedding" in self.providers:
                self.default_provider = "fast_embedding"
            elif "ollama" in self.providers:
                self.default_provider = "ollama"
            elif "openrouter" in self.providers:
                self.default_provider = "openrouter"
            elif "openai" in self.providers:
                self.default_provider = "openai"
            else:
                # Fallback to the first available provider if the logic above fails
                if self.providers:
                    self.default_provider = next(iter(self.providers))

            logger.info(f"Initialized embedding providers: {list(self.providers.keys())}")
            if self.default_provider:
                logger.info(f"Default embedding provider set to: {self.default_provider}")
            else:
                logger.warning("No embedding providers could be initialized. Embedding generation will fail.")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding providers: {str(e)}", exc_info=True)
            raise
    
    async def generate_embedding(self, 
                                text: str, 
                                provider: Optional[str] = None,
                                use_cache: bool = True) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            provider: Embedding provider to use
            use_cache: Whether to use cache
            
        Returns:
            EmbeddingResult: Embedding result
        """
        results = await self.generate_embeddings([text], provider, use_cache)
        return results[0]
    
    async def generate_embeddings(self, 
                                 texts: List[str], 
                                 provider: Optional[str] = None,
                                 use_cache: bool = True,
                                 batch_size: int = 100) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            provider: Embedding provider to use
            use_cache: Whether to use cache
            batch_size: Batch size for processing
            
        Returns:
            List[EmbeddingResult]: List of embedding results
        """
        if not texts:
            return []
        
        provider = provider or self.default_provider
        
        if not provider or provider not in self.providers:
            raise ValueError(f"Unknown or unconfigured embedding provider: {provider}")
        
        # Check cache first
        results = []
        texts_to_process = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if use_cache:
                cached_embedding = self.cache.get(text, provider)
                if cached_embedding:
                    result = EmbeddingResult(
                        text=text,
                        embedding=cached_embedding,
                        model=provider,
                        timestamp=datetime.utcnow()
                    )
                    results.append((i, result))
                    continue
            
            texts_to_process.append(text)
            text_indices.append(i)
        
        # Process remaining texts in batches
        if texts_to_process:
            embedding_provider = self.providers[provider]
            
            for batch_start in range(0, len(texts_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(texts_to_process))
                batch_texts = texts_to_process[batch_start:batch_end]
                
                try:
                    batch_results = await embedding_provider.generate_embeddings(batch_texts)
                    
                    for j, result in enumerate(batch_results):
                        original_index = text_indices[batch_start + j]
                        results.append((original_index, result))
                        
                        # Cache result
                        if use_cache:
                            self.cache.set(result.text, provider, result.embedding)
                
                except Exception as e:
                    logger.error(f"Batch embedding generation failed: {str(e)}")
                    # Create error results for failed batch
                    for j in range(len(batch_texts)):
                        original_index = text_indices[batch_start + j]
                        error_result = EmbeddingResult(
                            text=batch_texts[j],
                            embedding=[0.0] * self.get_embedding_dimension(provider),
                            model=provider,
                            timestamp=datetime.utcnow()
                        )
                        results.append((original_index, error_result))
        
        # Sort results by original index and return
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    async def preload_ollama_models(self, model_names: List[str]):
        """Pre-load models for the Ollama provider."""
        if "ollama" in self.providers:
            provider = self.providers["ollama"]
            logger.info(f"Preloading Ollama models: {model_names}")
            for model_name in model_names:
                try:
                    # Check if model exists locally
                    await provider.client.show(model_name)
                    logger.info(f"Ollama model '{model_name}' is already available locally.")
                except ollama.ResponseError as e:
                    if e.status_code == 404:
                        logger.info(f"Ollama model '{model_name}' not found. Pulling from hub...")
                        try:
                            # The ollama client shows progress, which is fine for interactive use.
                            # Here we just await the completion.
                            await provider.client.pull(model_name)
                            logger.info(f"Successfully pulled Ollama model '{model_name}'.")
                        except Exception as pull_error:
                            logger.error(f"Failed to pull Ollama model '{model_name}': {pull_error}")
                    else:
                        logger.error(f"Error checking for Ollama model '{model_name}': {e}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred while preloading Ollama model '{model_name}': {e}")
        else:
            logger.warning("Ollama provider not initialized. Cannot preload models.")

    def get_embedding_dimension(self, provider: Optional[str] = None) -> int:
        """Get embedding dimension for a provider"""
        provider = provider or self.default_provider
        
        if provider == "openai":
            return 1536  # text-embedding-ada-002 dimension
        elif provider == "openrouter":
            # This depends on the model used. gte-large is 1024.
            return 1024
        elif provider == "ollama":
            if provider in self.providers:
                model_name = self.providers[provider].model_name.lower()
                if "deepseek" in model_name:
                    return 4096
                if "nomic-embed-text" in model_name:
                    return 768
                if "all-minilm" in model_name:
                    return 384
                if "mxbai-embed-large" in model_name:
                    return 1024
            return 4096  # Fallback for other ollama models
        elif provider == "sentence_transformer":
            if provider in self.providers:
                provider_instance = self.providers[provider]
                model_name = provider_instance.model_name.lower()
                
                # Check for known model dimensions for performance and accuracy
                if "bge-large" in model_name:
                    return 1024
                if "bge-base" in model_name:
                    return 768
                if "bge-small" in model_name:
                    return 384
                if "gte-large" in model_name:  # e.g. thenlper/gte-large
                    return 1024
                if "gte-base" in model_name:
                    return 768
                if "nomic-embed-text" in model_name:  # nomic-ai/nomic-embed-text-v1
                    return 768
                if "all-minilm-l6-v2" in model_name:
                    return 384
                if "mxbai-embed-large" in model_name:
                    return 1024

                # Fallback to asking the model object, which is the ideal case
                model = provider_instance.model
                if model and hasattr(model, 'get_sentence_embedding_dimension'):
                    return model.get_sentence_embedding_dimension()

            return self.settings.ai.EMBEDDING_DIMENSION # Fallback to settings
        
        return self.settings.ai.EMBEDDING_DIMENSION
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    async def similarity_search(self, 
                               query_embedding: List[float],
                               candidate_embeddings: List[List[float]],
                               top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform similarity search between query and candidate embeddings.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List[Tuple[int, float]]: List of (index, similarity_score) tuples
        """
        try:
            # Convert to numpy arrays for efficient computation
            query_vec = np.array(query_embedding)
            candidate_vecs = np.array(candidate_embeddings)
            
            # Compute cosine similarities
            similarities = np.dot(candidate_vecs, query_vec) / (
                np.linalg.norm(candidate_vecs, axis=1) * np.linalg.norm(query_vec)
            )
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return results with similarity scores
            results = [(int(idx), float(similarities[idx])) for idx in top_indices]
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {str(e)}")
            return 0.0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": self.cache.size(),
            "max_cache_size": self.cache.max_size,
            "cache_ttl_hours": self.cache.ttl.total_seconds() / 3600
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def get_current_provider(self) -> str:
        """Get the name of the current default provider"""
        return self.default_provider or "unknown"


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(settings: Optional[Settings] = None) -> EmbeddingService:
    """
    Get the global embedding service instance.
    
    Args:
        settings: Application settings (required for first call)
        
    Returns:
        EmbeddingService: Global embedding service instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        if settings is None:
            try:
                from config.settings import get_settings
                settings = get_settings()
            except ImportError:
                try:
                    from ..config.settings import get_settings
                    settings = get_settings()
                except ImportError:
                    # Create default settings
                    settings = Settings()
        
        _embedding_service = EmbeddingService(settings)
    
    return _embedding_service
