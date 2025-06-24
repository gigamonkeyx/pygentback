"""
Enhanced embedding service integration for historical research with batch processing and caching.
"""
import logging
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

from ..utils.embedding import EmbeddingService

logger = logging.getLogger(__name__)

class HistoricalResearchEmbeddingService:
    """Enhanced embedding service for historical research with specialized features."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: str = "data/embeddings_cache",
                 batch_size: int = 32,
                 settings=None):
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.settings = settings
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base embedding service
        if settings:
            self.embedding_service = EmbeddingService(settings)
        else:
            # Create minimal settings for standalone use
            from ..config.settings import Settings
            minimal_settings = Settings()
            minimal_settings.ai.EMBEDDING_MODEL = model_name
            self.embedding_service = EmbeddingService(minimal_settings)
        self._initialized = False
          # Performance tracking
        self.stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the embedding service."""
        try:
            # The base embedding service is already initialized in constructor
            self._initialized = True
            logger.info(f"Historical research embedding service initialized with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            return False
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text with caching."""
        if not self._initialized:
            logger.error("Embedding service not initialized")
            return None
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(text)
            cached_embedding = await self._get_cached_embedding(cache_key)
            
            if cached_embedding:
                self.stats['cache_hits'] += 1
                return cached_embedding
            
            # Generate new embedding using the base service
            result = await self.embedding_service.generate_embedding(text)
            embedding = result.embedding if result else None
            
            if embedding:
                # Cache the result
                await self._cache_embedding(cache_key, embedding)
                self.stats['cache_misses'] += 1
                self.stats['total_embeddings'] += 1
                
                # Update performance stats
                processing_time = time.time() - start_time
                self.stats['total_processing_time'] += processing_time
                self.stats['average_processing_time'] = (
                    self.stats['total_processing_time'] / self.stats['total_embeddings']
                )
                
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts with batch processing."""
        if not self._initialized:
            logger.error("Embedding service not initialized")
            return [None] * len(texts)
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._process_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _process_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Process a batch of texts for embeddings."""
        embeddings = []
        
        # Check cache for each text
        cache_keys = [self._get_cache_key(text) for text in texts]
        cached_results = await self._get_cached_embeddings_batch(cache_keys)
        
        # Identify texts that need new embeddings
        texts_to_process = []
        indices_to_process = []
        
        for i, (text, cached_embedding) in enumerate(zip(texts, cached_results)):
            if cached_embedding:
                embeddings.append(cached_embedding)
                self.stats['cache_hits'] += 1
            else:
                embeddings.append(None)  # Placeholder
                texts_to_process.append(text)
                indices_to_process.append(i)
          # Generate embeddings for uncached texts
        if texts_to_process:
            try:
                # Use the base embedding service for batch processing
                results = await self.embedding_service.generate_embeddings(texts_to_process)
                new_embeddings = [result.embedding if result else None for result in results]
                
                # Update results and cache
                for idx, new_embedding in zip(indices_to_process, new_embeddings):
                    if new_embedding:
                        embeddings[idx] = new_embedding
                        # Cache the result
                        cache_key = cache_keys[idx]
                        await self._cache_embedding(cache_key, new_embedding)
                        
                        self.stats['cache_misses'] += 1
                        self.stats['total_embeddings'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process embedding batch: {e}")
        
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        
        # Include model name in cache key to avoid conflicts
        cache_input = f"{self.model_name}:{text}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()
    
    async def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            import json
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if cache is valid (could add expiration logic here)
            return data.get('embedding')
            
        except Exception as e:
            logger.warning(f"Failed to read cached embedding {cache_key}: {e}")
            return None
    
    async def _get_cached_embeddings_batch(self, cache_keys: List[str]) -> List[Optional[List[float]]]:
        """Get cached embeddings for multiple keys."""
        embeddings = []
        
        for cache_key in cache_keys:
            embedding = await self._get_cached_embedding(cache_key)
            embeddings.append(embedding)
        
        return embeddings
    
    async def _cache_embedding(self, cache_key: str, embedding: List[float]):
        """Cache an embedding."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            import json
            cache_data = {
                'embedding': embedding,
                'model': self.model_name,
                'timestamp': time.time(),
                'dimension': len(embedding)
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to cache embedding {cache_key}: {e}")
    
    async def embed_document_chunks(self, 
                                  chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed multiple document chunks and return enhanced chunks."""
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = [chunk.get('text', '') for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.get_embeddings_batch(texts)
        
        # Enhance chunks with embeddings
        enhanced_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                enhanced_chunk = chunk.copy()
                enhanced_chunk['embedding'] = embedding
                enhanced_chunk['embedding_model'] = self.model_name
                enhanced_chunk['embedding_dimension'] = len(embedding)
                enhanced_chunks.append(enhanced_chunk)
            else:
                logger.warning(f"Failed to generate embedding for chunk: {chunk.get('id', 'unknown')}")
        
        return enhanced_chunks
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the embedding service."""
        cache_hit_rate = 0.0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        
        return {
            'model_name': self.model_name,
            'total_embeddings_generated': self.stats['total_embeddings'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'average_processing_time': self.stats['average_processing_time'],
            'total_processing_time': self.stats['total_processing_time']
        }
    
    async def clear_cache(self, older_than_days: Optional[int] = None):
        """Clear embedding cache, optionally only entries older than specified days."""
        try:
            import time
            
            current_time = time.time()
            cutoff_time = None
            
            if older_than_days:
                cutoff_time = current_time - (older_than_days * 24 * 60 * 60)
            
            cleared_count = 0
            
            for cache_file in self.cache_dir.glob("*.json"):
                should_clear = False
                
                if cutoff_time:
                    # Check file modification time
                    file_mtime = cache_file.stat().st_mtime
                    if file_mtime < cutoff_time:
                        should_clear = True
                else:
                    # Clear all
                    should_clear = True
                
                if should_clear:
                    try:
                        cache_file.unlink()
                        cleared_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to clear cache file {cache_file}: {e}")
            
            logger.info(f"Cleared {cleared_count} cache files")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    async def warmup_cache(self, texts: List[str]):
        """Warm up cache by pre-generating embeddings for common texts."""
        logger.info(f"Warming up embedding cache with {len(texts)} texts...")
        
        start_time = time.time()
        embeddings = await self.get_embeddings_batch(texts)
        processing_time = time.time() - start_time
        
        successful_embeddings = sum(1 for e in embeddings if e is not None)
        
        logger.info(f"Cache warmup complete: {successful_embeddings}/{len(texts)} embeddings generated in {processing_time:.2f}s")
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.embedding_service:
                await self.embedding_service.cleanup()
            
            logger.info("Historical research embedding service cleaned up")
            
        except Exception as e:
            logger.error(f"Error during embedding service cleanup: {e}")
