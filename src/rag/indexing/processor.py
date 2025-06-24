"""
Document Processor Manager

Manages document processing workflows for the RAG indexing system.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .base import DocumentProcessor
from .chunker import TextChunker, SemanticChunker, HierarchicalChunker, DocumentChunk

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingJob:
    """Document processing job"""
    job_id: str
    document: Dict[str, Any]
    processor_type: str
    status: ProcessingStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[List[DocumentChunk]] = None


class DocumentProcessorManager:
    """
    Manages document processing workflows and coordination.
    
    Provides centralized management of document processors, job queuing,
    and processing pipeline coordination.
    """
    
    def __init__(self):
        self.processors: Dict[str, DocumentProcessor] = {}
        self.chunkers = {
            'text': TextChunker(),
            'semantic': SemanticChunker(),
            'hierarchical': HierarchicalChunker()
        }
        self.processing_jobs: Dict[str, ProcessingJob] = {}
        self.job_queue = asyncio.Queue()
        self.is_processing = False
        self.max_concurrent_jobs = 5
        
    def register_processor(self, name: str, processor: DocumentProcessor) -> None:
        """Register a document processor"""
        self.processors[name] = processor
        logger.info(f"Registered document processor: {name}")
    
    def register_chunker(self, name: str, chunker) -> None:
        """Register a document chunker"""
        self.chunkers[name] = chunker
        logger.info(f"Registered document chunker: {name}")
    
    async def process_document(self, document: Dict[str, Any], 
                             processor_type: str = 'default',
                             chunker_type: str = 'text') -> List[DocumentChunk]:
        """
        Process a single document.
        
        Args:
            document: Document to process
            processor_type: Type of processor to use
            chunker_type: Type of chunker to use
            
        Returns:
            List of processed document chunks
        """
        try:
            # Get chunker
            chunker = self.chunkers.get(chunker_type)
            if not chunker:
                raise ValueError(f"Unknown chunker type: {chunker_type}")
            
            # Process document with chunker
            chunks = chunker.chunk_document(document)
            
            # Apply additional processing if processor is available
            if processor_type in self.processors:
                processor = self.processors[processor_type]
                # Apply processor to each chunk
                processed_chunks = []
                for chunk in chunks:
                    # Convert chunk to document format for processor
                    chunk_doc = {
                        'id': chunk.metadata.chunk_id,
                        'content': chunk.content,
                        'metadata': chunk.metadata.metadata
                    }
                    
                    # Process chunk (assuming processor returns processed document)
                    processed_doc = await processor.process_document(chunk_doc)
                    
                    # Update chunk with processed content
                    chunk.content = processed_doc.get('content', chunk.content)
                    processed_chunks.append(chunk)
                
                chunks = processed_chunks
            
            logger.debug(f"Processed document {document.get('id', 'unknown')} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return []
    
    async def process_documents_batch(self, documents: List[Dict[str, Any]],
                                    processor_type: str = 'default',
                                    chunker_type: str = 'text') -> List[List[DocumentChunk]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of documents to process
            processor_type: Type of processor to use
            chunker_type: Type of chunker to use
            
        Returns:
            List of chunk lists for each document
        """
        try:
            # Process documents concurrently
            tasks = []
            for document in documents:
                task = self.process_document(document, processor_type, chunker_type)
                tasks.append(task)
            
            # Wait for all processing to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return successful results
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Document {i} processing failed: {result}")
                    successful_results.append([])
                else:
                    successful_results.append(result)
            
            logger.info(f"Batch processed {len(documents)} documents")
            return successful_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []
    
    async def submit_processing_job(self, document: Dict[str, Any],
                                  processor_type: str = 'default',
                                  chunker_type: str = 'text') -> str:
        """
        Submit a document processing job to the queue.
        
        Args:
            document: Document to process
            processor_type: Type of processor to use
            chunker_type: Type of chunker to use
            
        Returns:
            Job ID for tracking
        """
        import time
        import uuid
        
        job_id = str(uuid.uuid4())
        
        job = ProcessingJob(
            job_id=job_id,
            document=document,
            processor_type=f"{processor_type}:{chunker_type}",
            status=ProcessingStatus.PENDING,
            created_at=time.time()
        )
        
        self.processing_jobs[job_id] = job
        await self.job_queue.put(job)
        
        logger.debug(f"Submitted processing job: {job_id}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get the status of a processing job"""
        return self.processing_jobs.get(job_id)
    
    async def start_processing_worker(self) -> None:
        """Start the background processing worker"""
        if self.is_processing:
            return
        
        self.is_processing = True
        logger.info("Started document processing worker")
        
        try:
            while self.is_processing:
                try:
                    # Get job from queue with timeout
                    job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                    
                    # Process the job
                    await self._process_job(job)
                    
                except asyncio.TimeoutError:
                    # No jobs in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Processing worker error: {e}")
                    
        except Exception as e:
            logger.error(f"Processing worker failed: {e}")
        finally:
            self.is_processing = False
            logger.info("Stopped document processing worker")
    
    async def stop_processing_worker(self) -> None:
        """Stop the background processing worker"""
        self.is_processing = False
        logger.info("Stopping document processing worker")
    
    async def _process_job(self, job: ProcessingJob) -> None:
        """Process a single job"""
        import time
        
        try:
            # Update job status
            job.status = ProcessingStatus.PROCESSING
            job.started_at = time.time()
            
            # Parse processor and chunker types
            processor_type, chunker_type = job.processor_type.split(':', 1)
            
            # Process document
            chunks = await self.process_document(
                job.document, processor_type, chunker_type
            )
            
            # Update job with results
            job.result = chunks
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = time.time()
            
            logger.debug(f"Completed processing job: {job.job_id}")
            
        except Exception as e:
            # Update job with error
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            
            logger.error(f"Processing job {job.job_id} failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total_jobs = len(self.processing_jobs)
        completed_jobs = sum(1 for job in self.processing_jobs.values() 
                           if job.status == ProcessingStatus.COMPLETED)
        failed_jobs = sum(1 for job in self.processing_jobs.values() 
                        if job.status == ProcessingStatus.FAILED)
        pending_jobs = sum(1 for job in self.processing_jobs.values() 
                         if job.status == ProcessingStatus.PENDING)
        
        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "pending_jobs": pending_jobs,
            "queue_size": self.job_queue.qsize(),
            "is_processing": self.is_processing,
            "registered_processors": list(self.processors.keys()),
            "registered_chunkers": list(self.chunkers.keys())
        }
