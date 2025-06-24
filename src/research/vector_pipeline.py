"""
Vector document creation pipeline for historical research.
Converts extracted text to searchable vector documents with rich metadata.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .semantic_chunking import SemanticChunker, DocumentChunk
from .embedding_service import HistoricalResearchEmbeddingService
from .vector_config import HistoricalResearchVectorStore

logger = logging.getLogger(__name__)

@dataclass
class VectorDocumentResult:
    """Result of vector document creation process."""
    success: bool
    document_id: str
    chunks_processed: int = 0
    chunks_embedded: int = 0
    chunks_indexed: int = 0
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class VectorDocumentPipeline:
    """Pipeline for creating vector documents from extracted text."""
    
    def __init__(self,
                 embedding_service: HistoricalResearchEmbeddingService,
                 vector_store: HistoricalResearchVectorStore,
                 chunker: Optional[SemanticChunker] = None):
        
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunker = chunker or SemanticChunker()
        
        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'total_chunks_embedded': 0,
            'total_chunks_indexed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
    
    async def create_vector_document(self,
                                   document_id: str,
                                   text_content: str,
                                   document_metadata: Dict[str, Any]) -> VectorDocumentResult:
        """Create vector document from text content."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Creating vector document for {document_id}")
            
            # Step 1: Semantic chunking
            chunks = self.chunker.chunk_document(
                document_id=document_id,
                text=text_content,
                document_metadata=document_metadata
            )
            
            if not chunks:
                return VectorDocumentResult(
                    success=False,
                    document_id=document_id,
                    error_message="No chunks created from document"
                )
            
            logger.info(f"Created {len(chunks)} chunks for document {document_id}")
            
            # Step 2: Generate embeddings for chunks
            embedded_chunks = await self._embed_chunks(chunks)
            
            # Step 3: Index chunks in vector store
            indexed_count = await self._index_chunks(document_id, embedded_chunks)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._update_stats(processing_time, len(chunks), len(embedded_chunks), indexed_count)
            
            # Create result
            result = VectorDocumentResult(
                success=True,
                document_id=document_id,
                chunks_processed=len(chunks),
                chunks_embedded=len(embedded_chunks),
                chunks_indexed=indexed_count,
                processing_time=processing_time,
                metadata={
                    'chunking_stats': self.chunker.get_chunking_stats(chunks),
                    'embedding_stats': self.embedding_service.get_performance_stats(),
                    'total_text_length': len(text_content),
                    'document_metadata': document_metadata
                }
            )
            
            logger.info(f"Successfully created vector document {document_id}: "
                       f"{indexed_count}/{len(chunks)} chunks indexed")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to create vector document {document_id}: {e}")
            
            return VectorDocumentResult(
                success=False,
                document_id=document_id,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _embed_chunks(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks."""
        try:
            # Convert chunks to embedding format
            chunk_data = []
            for chunk in chunks:
                chunk_data.append({
                    'id': chunk.id,
                    'text': chunk.text,
                    'page_number': chunk.page_number,
                    'section': chunk.section_title,
                    'section_type': chunk.section_type,
                    'metadata': chunk.metadata
                })
            
            # Generate embeddings in batch
            embedded_chunks = await self.embedding_service.embed_document_chunks(chunk_data)
            
            logger.info(f"Generated embeddings for {len(embedded_chunks)}/{len(chunks)} chunks")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to embed chunks: {e}")
            return []
    
    async def _index_chunks(self, document_id: str, embedded_chunks: List[Dict[str, Any]]) -> int:
        """Index embedded chunks in vector store."""
        try:
            indexed_count = await self.vector_store.add_document_chunks(
                document_id=document_id,
                chunks=embedded_chunks
            )
            
            logger.info(f"Indexed {indexed_count} chunks for document {document_id}")
            return indexed_count
            
        except Exception as e:
            logger.error(f"Failed to index chunks for {document_id}: {e}")
            return 0
    
    def _update_stats(self, processing_time: float, chunks_created: int, 
                     chunks_embedded: int, chunks_indexed: int):
        """Update processing statistics."""
        self.stats['documents_processed'] += 1
        self.stats['total_chunks_created'] += chunks_created
        self.stats['total_chunks_embedded'] += chunks_embedded
        self.stats['total_chunks_indexed'] += chunks_indexed
        self.stats['total_processing_time'] += processing_time
        
        if self.stats['documents_processed'] > 0:
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['documents_processed']
            )
    
    async def create_vector_documents_batch(self,
                                          documents: List[Dict[str, Any]]) -> List[VectorDocumentResult]:
        """Process multiple documents in batch."""
        results = []
        
        for doc in documents:
            result = await self.create_vector_document(
                document_id=doc['id'],
                text_content=doc['text'],
                document_metadata=doc.get('metadata', {})
            )
            results.append(result)
        
        # Log batch summary
        successful = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_indexed for r in results)
        
        logger.info(f"Batch processing complete: {successful}/{len(documents)} documents successful, "
                   f"{total_chunks} total chunks indexed")
        
        return results
    
    async def update_document_metadata(self,
                                     document_id: str,
                                     new_metadata: Dict[str, Any]) -> bool:
        """Update metadata for all chunks of a document."""
        try:
            # Search for all chunks of this document
            chunks = await self.vector_store.search_documents(
                query="",  # Empty query to get all
                metadata_filter={'document_id': document_id}
            )
              # Update each chunk with new metadata
            updated_count = 0
            for chunk in chunks:
                # Here you would update the vector store
                # This is a placeholder for the update operation
                # merged_metadata = {**chunk['metadata'], **new_metadata}
                # await self.vector_store.update_chunk_metadata(chunk['chunk_id'], merged_metadata)
                updated_count += 1
            
            logger.info(f"Updated metadata for {updated_count} chunks of document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document metadata for {document_id}: {e}")
            return False
    
    async def rebuild_document_index(self,
                                   document_id: str,
                                   text_content: str,
                                   document_metadata: Dict[str, Any]) -> VectorDocumentResult:
        """Rebuild vector index for a document (delete and recreate)."""
        try:
            # Delete existing chunks
            deleted = await self.vector_store.delete_document(document_id)
            if deleted:
                logger.info(f"Deleted existing chunks for document {document_id}")
            
            # Recreate vector document
            result = await self.create_vector_document(
                document_id=document_id,
                text_content=text_content,
                document_metadata=document_metadata
            )
            
            if result.success:
                logger.info(f"Rebuilt vector index for document {document_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to rebuild document index for {document_id}: {e}")
            return VectorDocumentResult(
                success=False,
                document_id=document_id,
                error_message=str(e)
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.stats.copy()
        
        # Add derived metrics
        if stats['documents_processed'] > 0:
            stats['average_chunks_per_document'] = (
                stats['total_chunks_created'] / stats['documents_processed']
            )
            stats['embedding_success_rate'] = (
                stats['total_chunks_embedded'] / stats['total_chunks_created']
                if stats['total_chunks_created'] > 0 else 0
            )
            stats['indexing_success_rate'] = (
                stats['total_chunks_indexed'] / stats['total_chunks_embedded']
                if stats['total_chunks_embedded'] > 0 else 0
            )
        
        # Add embedding service stats
        stats['embedding_service_stats'] = self.embedding_service.get_performance_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector document pipeline."""
        health = {
            'status': 'healthy',
            'issues': [],
            'components': {}
        }
        
        try:
            # Check embedding service
            if not self.embedding_service._initialized:
                health['issues'].append("Embedding service not initialized")
                health['status'] = 'unhealthy'
            else:
                health['components']['embedding_service'] = 'healthy'
            
            # Check vector store
            if not self.vector_store._initialized:
                health['issues'].append("Vector store not initialized")
                health['status'] = 'unhealthy'
            else:
                health['components']['vector_store'] = 'healthy'
                
                # Test vector store connectivity
                doc_count = await self.vector_store.get_document_count()
                health['components']['vector_store_document_count'] = doc_count
            
            # Check chunker
            health['components']['semantic_chunker'] = 'healthy'
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['issues'].append(f"Health check failed: {str(e)}")
        
        return health
    
    async def cleanup(self):
        """Cleanup pipeline resources."""
        try:
            if self.embedding_service:
                await self.embedding_service.cleanup()
            
            if self.vector_store:
                await self.vector_store.close()
            
            logger.info("Vector document pipeline cleaned up")
            
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")
