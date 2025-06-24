"""
Complete Historical Research Pipeline Integration
Combines all components into a unified system.
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

# Import all our new components
from ..acquisition.document_download import DocumentDownloadPipeline
from ..acquisition.document_storage import DocumentStorageSystem, DocumentMetadata
from ..acquisition.text_extraction import PyMuPDFTextExtractor
from ..acquisition.ocr_extraction import OCRTextExtractor
from ..acquisition.processing_metadata import DocumentProcessingTracker, DocumentProcessingMetadata
from .vector_config import HistoricalResearchVectorStore, HistoricalResearchVectorConfig
from .embedding_service import HistoricalResearchEmbeddingService
from .semantic_chunking import SemanticChunker
from .vector_pipeline import VectorDocumentPipeline

logger = logging.getLogger(__name__)

class HistoricalResearchPipeline:
    """Complete pipeline for historical research document processing."""
    
    def __init__(self, settings, base_path: str = "data/historical_research"):
        self.settings = settings
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.storage_system = DocumentStorageSystem(str(self.base_path / "documents"))
        self.download_pipeline = DocumentDownloadPipeline(str(self.base_path / "downloads"))
        self.processing_tracker = DocumentProcessingTracker(self.storage_system)
        
        # Text extraction components
        self.text_extractor = PyMuPDFTextExtractor(self.storage_system)
        self.ocr_extractor = OCRTextExtractor()
          # Vector components
        self.vector_config = HistoricalResearchVectorConfig(
            storage_path=str(self.base_path / "vector_store")
        )
        self.embedding_service = HistoricalResearchEmbeddingService(
            cache_dir=str(self.base_path / "embeddings_cache"),
            settings=self.settings
        )
        self.vector_store = HistoricalResearchVectorStore(self.vector_config)
        self.semantic_chunker = SemanticChunker()
        self.vector_pipeline = None  # Initialized after other components
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the complete research pipeline."""
        try:
            logger.info("Initializing historical research pipeline...")
            
            # Initialize embedding service
            if not await self.embedding_service.initialize():
                raise Exception("Failed to initialize embedding service")
            
            # Initialize vector store
            if not await self.vector_store.initialize(self.settings):
                raise Exception("Failed to initialize vector store")
            
            # Initialize vector pipeline
            self.vector_pipeline = VectorDocumentPipeline(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store,
                chunker=self.semantic_chunker
            )
            
            self._initialized = True
            logger.info("Historical research pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize research pipeline: {e}")
            return False
    
    async def process_document_from_url(self, 
                                      url: str, 
                                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Complete pipeline: download, extract, and index a document."""
        if not self._initialized:
            raise Exception("Pipeline not initialized")
        
        try:
            # Step 1: Download document
            download_result = self.download_pipeline.download_document(url)
            if not download_result.success:
                return {
                    'success': False,
                    'stage': 'download',
                    'error': download_result.error_message
                }
            
            document_id = download_result.document_id
            file_path = download_result.file_path
            
            # Step 2: Store document with metadata
            doc_metadata = DocumentMetadata(
                document_id=document_id,
                original_url=url,
                **metadata
            )
            
            stored = self.storage_system.store_document(file_path, doc_metadata)
            if not stored:
                return {
                    'success': False,
                    'stage': 'storage',
                    'error': 'Failed to store document'
                }
            
            # Step 3: Extract text
            text_result = self.text_extractor.extract_text_multiple_methods(
                file_path, document_id
            )
            
            # Step 4: Try OCR if text extraction failed or quality is poor
            if not text_result.success or (text_result.quality_score and text_result.quality_score < 0.3):
                ocr_result = self.ocr_extractor.extract_text_with_ocr(file_path, document_id)
                if ocr_result.success:
                    # Compare and choose best extraction
                    comparison = self.ocr_extractor.compare_extraction_quality(
                        text_result.text_content or "", ocr_result.text_content or ""
                    )
                    if comparison['recommended_method'] == 'ocr':
                        text_result.text_content = ocr_result.text_content
                        text_result.extraction_method = 'ocr'
                        text_result.metadata['ocr_confidence'] = ocr_result.confidence_score
            
            if not text_result.text_content:
                return {
                    'success': False,
                    'stage': 'text_extraction',
                    'error': 'Failed to extract text content'
                }
            
            # Step 5: Record processing metadata
            processing_metadata = DocumentProcessingMetadata(
                document_id=document_id,
                extraction_method=text_result.extraction_method,
                extraction_success=text_result.success,
                quality_score=text_result.quality_score,
                text_length=len(text_result.text_content),
                word_count=text_result.word_count
            )
            self.processing_tracker.record_processing_result(processing_metadata)
            
            # Step 6: Create vector documents
            vector_result = await self.vector_pipeline.create_vector_document(
                document_id=document_id,
                text_content=text_result.text_content,
                document_metadata=metadata
            )
            
            # Step 7: Update document metadata to mark as processed
            doc_metadata.text_extracted = True
            doc_metadata.vector_indexed = vector_result.success
            doc_metadata.extraction_quality = text_result.quality_score
            
            return {
                'success': True,
                'document_id': document_id,
                'download_result': download_result,
                'text_extraction_result': text_result,
                'vector_result': vector_result,
                'processing_metadata': processing_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to process document from {url}: {e}")
            return {
                'success': False,
                'stage': 'pipeline',
                'error': str(e)
            }
    
    async def search_historical_documents(self,
                                        query: str,
                                        limit: int = 10,
                                        filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for relevant historical documents."""
        if not self._initialized:
            raise Exception("Pipeline not initialized")
        
        try:
            results = await self.vector_store.search_documents(
                query=query,
                limit=limit,
                metadata_filter=filters
            )
            
            # Enhance results with additional metadata
            enhanced_results = []
            for result in results:
                # Get document metadata
                doc_metadata = self.storage_system.get_document(result['document_id'])
                
                enhanced_result = {
                    **result,
                    'document_metadata': doc_metadata.__dict__ if doc_metadata else {},
                    'processing_metadata': self.processing_tracker.get_processing_metadata(
                        result['document_id']
                    ).__dict__ if self.processing_tracker.get_processing_metadata(result['document_id']) else {}
                }
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Storage statistics
            storage_stats = self.storage_system.get_storage_stats()
            
            # Processing statistics
            processing_stats = self.processing_tracker.get_statistics()
            
            # Vector store statistics
            vector_stats = await self.vector_store.get_collection_stats()
            
            # Pipeline statistics
            pipeline_stats = self.vector_pipeline.get_processing_stats() if self.vector_pipeline else {}
            
            # Embedding service statistics
            embedding_stats = self.embedding_service.get_performance_stats()
            
            return {
                'initialized': self._initialized,
                'storage': storage_stats,
                'processing': processing_stats.__dict__ if hasattr(processing_stats, '__dict__') else processing_stats,
                'vector_store': vector_stats,
                'pipeline': pipeline_stats,
                'embedding': embedding_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Cleanup all pipeline resources."""
        try:
            if self.vector_pipeline:
                await self.vector_pipeline.cleanup()
            
            if self.vector_store:
                await self.vector_store.close()
            
            if self.embedding_service:
                await self.embedding_service.cleanup()
            
            # Cleanup temporary files
            self.storage_system.cleanup_temp_files()
            
            self._initialized = False
            logger.info("Historical research pipeline cleaned up")
            
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")

# Integration with existing PyGent Factory system
async def create_historical_research_pipeline(settings) -> HistoricalResearchPipeline:
    """Factory function to create and initialize the research pipeline."""
    pipeline = HistoricalResearchPipeline(settings)
    
    if await pipeline.initialize():
        return pipeline
    else:
        raise Exception("Failed to initialize historical research pipeline")
