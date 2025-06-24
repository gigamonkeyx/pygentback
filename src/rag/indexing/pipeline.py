"""
Indexing Pipeline Manager

Manages end-to-end document indexing pipelines for the RAG system.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .base import IndexingPipeline
from .processor import DocumentProcessorManager, ProcessingStatus
from .chunker import DocumentChunk

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for indexing pipeline"""
    name: str
    processor_type: str = 'default'
    chunker_type: str = 'text'
    batch_size: int = 10
    max_retries: int = 3
    enable_embeddings: bool = True
    enable_indexing: bool = True
    output_format: str = 'chunks'


@dataclass
class PipelineRun:
    """Pipeline execution run"""
    run_id: str
    pipeline_name: str
    config: PipelineConfig
    status: PipelineStatus
    total_documents: int
    processed_documents: int
    failed_documents: int
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    results: List[Any] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []


class IndexingPipelineManager:
    """
    Manages indexing pipelines for document processing workflows.
    
    Provides pipeline configuration, execution, monitoring, and result management.
    """
    
    def __init__(self):
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.pipeline_runs: Dict[str, PipelineRun] = {}
        self.processor_manager = DocumentProcessorManager()
        self.embedding_functions: Dict[str, Callable] = {}
        self.index_functions: Dict[str, Callable] = {}
        
    def register_pipeline(self, config: PipelineConfig) -> None:
        """Register a pipeline configuration"""
        self.pipelines[config.name] = config
        logger.info(f"Registered pipeline: {config.name}")
    
    def register_embedding_function(self, name: str, func: Callable) -> None:
        """Register an embedding function"""
        self.embedding_functions[name] = func
        logger.info(f"Registered embedding function: {name}")
    
    def register_index_function(self, name: str, func: Callable) -> None:
        """Register an index function"""
        self.index_functions[name] = func
        logger.info(f"Registered index function: {name}")
    
    async def execute_pipeline(self, pipeline_name: str, 
                             documents: List[Dict[str, Any]],
                             config_overrides: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a pipeline on a set of documents.
        
        Args:
            pipeline_name: Name of the pipeline to execute
            documents: List of documents to process
            config_overrides: Optional configuration overrides
            
        Returns:
            Run ID for tracking execution
        """
        import time
        import uuid
        
        try:
            # Get pipeline configuration
            if pipeline_name not in self.pipelines:
                raise ValueError(f"Unknown pipeline: {pipeline_name}")
            
            config = self.pipelines[pipeline_name]
            
            # Apply configuration overrides
            if config_overrides:
                config = PipelineConfig(
                    name=config.name,
                    processor_type=config_overrides.get('processor_type', config.processor_type),
                    chunker_type=config_overrides.get('chunker_type', config.chunker_type),
                    batch_size=config_overrides.get('batch_size', config.batch_size),
                    max_retries=config_overrides.get('max_retries', config.max_retries),
                    enable_embeddings=config_overrides.get('enable_embeddings', config.enable_embeddings),
                    enable_indexing=config_overrides.get('enable_indexing', config.enable_indexing),
                    output_format=config_overrides.get('output_format', config.output_format)
                )
            
            # Create pipeline run
            run_id = str(uuid.uuid4())
            pipeline_run = PipelineRun(
                run_id=run_id,
                pipeline_name=pipeline_name,
                config=config,
                status=PipelineStatus.RUNNING,
                total_documents=len(documents),
                processed_documents=0,
                failed_documents=0,
                created_at=time.time(),
                started_at=time.time()
            )
            
            self.pipeline_runs[run_id] = pipeline_run
            
            # Execute pipeline asynchronously
            asyncio.create_task(self._execute_pipeline_async(pipeline_run, documents))
            
            logger.info(f"Started pipeline execution: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start pipeline execution: {e}")
            raise
    
    async def _execute_pipeline_async(self, pipeline_run: PipelineRun, 
                                    documents: List[Dict[str, Any]]) -> None:
        """Execute pipeline asynchronously"""
        import time
        
        try:
            config = pipeline_run.config
            
            # Process documents in batches
            all_chunks = []
            batch_size = config.batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                try:
                    # Process batch
                    batch_results = await self.processor_manager.process_documents_batch(
                        batch, config.processor_type, config.chunker_type
                    )
                    
                    # Flatten results
                    for doc_chunks in batch_results:
                        all_chunks.extend(doc_chunks)
                    
                    # Update progress
                    pipeline_run.processed_documents += len(batch)
                    
                    logger.debug(f"Processed batch {i//batch_size + 1} for run {pipeline_run.run_id}")
                    
                except Exception as e:
                    logger.error(f"Batch processing failed for run {pipeline_run.run_id}: {e}")
                    pipeline_run.failed_documents += len(batch)
            
            # Generate embeddings if enabled
            if config.enable_embeddings and all_chunks:
                all_chunks = await self._generate_embeddings(all_chunks)
            
            # Index chunks if enabled
            if config.enable_indexing and all_chunks:
                await self._index_chunks(all_chunks)
            
            # Store results
            pipeline_run.results = all_chunks
            pipeline_run.status = PipelineStatus.COMPLETED
            pipeline_run.completed_at = time.time()
            
            logger.info(f"Pipeline run {pipeline_run.run_id} completed successfully")
            
        except Exception as e:
            pipeline_run.status = PipelineStatus.FAILED
            pipeline_run.error = str(e)
            pipeline_run.completed_at = time.time()
            
            logger.error(f"Pipeline run {pipeline_run.run_id} failed: {e}")
    
    async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for chunks"""
        try:
            # Use default embedding function if available
            if 'default' in self.embedding_functions:
                embedding_func = self.embedding_functions['default']
                
                for chunk in chunks:
                    try:
                        # Generate embedding for chunk content
                        embedding = await embedding_func(chunk.content)
                        chunk.embedding = embedding
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for chunk {chunk.metadata.chunk_id}: {e}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return chunks
    
    async def _index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks using registered index functions"""
        try:
            # Use default index function if available
            if 'default' in self.index_functions:
                index_func = self.index_functions['default']
                
                # Convert chunks to indexable format
                indexable_docs = []
                for chunk in chunks:
                    doc = {
                        'id': chunk.metadata.chunk_id,
                        'content': chunk.content,
                        'embedding': chunk.embedding,
                        'metadata': chunk.metadata.__dict__
                    }
                    indexable_docs.append(doc)
                
                # Index documents
                await index_func(indexable_docs)
                
                logger.debug(f"Indexed {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
    
    def get_pipeline_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run by ID"""
        return self.pipeline_runs.get(run_id)
    
    def list_pipeline_runs(self, pipeline_name: Optional[str] = None) -> List[PipelineRun]:
        """List pipeline runs, optionally filtered by pipeline name"""
        runs = list(self.pipeline_runs.values())
        
        if pipeline_name:
            runs = [run for run in runs if run.pipeline_name == pipeline_name]
        
        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x.created_at, reverse=True)
        
        return runs
    
    def get_pipeline_stats(self, pipeline_name: str) -> Dict[str, Any]:
        """Get statistics for a specific pipeline"""
        runs = [run for run in self.pipeline_runs.values() 
                if run.pipeline_name == pipeline_name]
        
        if not runs:
            return {"pipeline_name": pipeline_name, "total_runs": 0}
        
        total_runs = len(runs)
        completed_runs = sum(1 for run in runs if run.status == PipelineStatus.COMPLETED)
        failed_runs = sum(1 for run in runs if run.status == PipelineStatus.FAILED)
        total_documents = sum(run.total_documents for run in runs)
        processed_documents = sum(run.processed_documents for run in runs)
        
        return {
            "pipeline_name": pipeline_name,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "success_rate": completed_runs / total_runs if total_runs > 0 else 0,
            "total_documents": total_documents,
            "processed_documents": processed_documents,
            "processing_rate": processed_documents / total_documents if total_documents > 0 else 0
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall pipeline manager statistics"""
        return {
            "registered_pipelines": len(self.pipelines),
            "total_runs": len(self.pipeline_runs),
            "embedding_functions": len(self.embedding_functions),
            "index_functions": len(self.index_functions),
            "processor_stats": self.processor_manager.get_stats()
        }
