"""
RAG Indexing Base Classes

This module defines the core interfaces and data structures for the RAG indexing system.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid


logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type enumeration"""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    CODE = "code"
    JSON = "json"
    CSV = "csv"
    DOCX = "docx"
    UNKNOWN = "unknown"


class ChunkingStrategy(Enum):
    """Chunking strategy options"""
    FIXED_SIZE = "fixed_size"           # Fixed character/token count
    SENTENCE = "sentence"               # Sentence-based chunking
    PARAGRAPH = "paragraph"             # Paragraph-based chunking
    SEMANTIC = "semantic"               # Semantic similarity-based
    HIERARCHICAL = "hierarchical"      # Multi-level hierarchical
    ADAPTIVE = "adaptive"               # Adaptive based on content


class ProcessingStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DocumentMetadata:
    """Comprehensive document metadata"""
    # Basic identification
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    
    # Content information
    language: str = "en"
    encoding: str = "utf-8"
    file_size: int = 0
    content_hash: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None
    
    # Processing information
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_error: Optional[str] = None
    chunk_count: int = 0
    
    # Custom metadata
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    # Authority and quality indicators
    authority_score: float = 0.5
    quality_score: float = 0.5
    verified_source: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "document_type": self.document_type.value,
            "language": self.language,
            "encoding": self.encoding,
            "file_size": self.file_size,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "processing_status": self.processing_status.value,
            "processing_error": self.processing_error,
            "chunk_count": self.chunk_count,
            "tags": self.tags,
            "categories": self.categories,
            "custom_fields": self.custom_fields,
            "authority_score": self.authority_score,
            "quality_score": self.quality_score,
            "verified_source": self.verified_source
        }


@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Chunk positioning
    start_char: int = 0
    end_char: int = 0
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    
    # Chunk metadata
    chunk_type: str = "text"  # text, header, code, table, etc.
    heading: Optional[str] = None
    section: Optional[str] = None
    
    # Processing information
    token_count: int = 0
    char_count: int = 0
    embedding: Optional[List[float]] = None
    
    # Relationships
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    
    # Metadata inheritance from document
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.char_count:
            self.char_count = len(self.content)
        
        if not self.token_count:
            # Simple token estimation (words * 1.3)
            self.token_count = int(len(self.content.split()) * 1.3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "heading": self.heading,
            "section": self.section,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
            "metadata": self.metadata
        }


@dataclass
class ProcessingResult:
    """Result of document processing operation"""
    document_id: str
    status: ProcessingStatus
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    
    # Statistics
    total_chunks: int = 0
    total_tokens: int = 0
    total_chars: int = 0
    
    def __post_init__(self):
        """Calculate statistics from chunks"""
        if self.chunks:
            self.total_chunks = len(self.chunks)
            self.total_tokens = sum(chunk.token_count for chunk in self.chunks)
            self.total_chars = sum(chunk.char_count for chunk in self.chunks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "document_id": self.document_id,
            "status": self.status.value,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
            "statistics": {
                "total_chunks": self.total_chunks,
                "total_tokens": self.total_tokens,
                "total_chars": self.total_chars
            }
        }


@runtime_checkable
class DocumentProcessor(Protocol):
    """Protocol defining the interface for document processors"""
    
    async def process_document(self, content: str, metadata: DocumentMetadata) -> ProcessingResult:
        """Process a document and return chunks"""
        ...
    
    def supports_type(self, document_type: DocumentType) -> bool:
        """Check if processor supports the document type"""
        ...


@runtime_checkable
class ChunkingEngine(Protocol):
    """Protocol defining the interface for chunking engines"""
    
    async def chunk_text(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Chunk text into smaller pieces"""
        ...
    
    def get_chunk_size_estimate(self, text: str) -> int:
        """Estimate number of chunks for given text"""
        ...


@runtime_checkable
class IndexingPipeline(Protocol):
    """Protocol defining the interface for indexing pipelines"""
    
    async def index_document(self, content: str, metadata: DocumentMetadata) -> ProcessingResult:
        """Process and index a document"""
        ...
    
    async def index_batch(self, documents: List[tuple]) -> List[ProcessingResult]:
        """Process and index multiple documents"""
        ...


class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors"""
    
    def __init__(self, name: str, supported_types: List[DocumentType]):
        self.name = name
        self.supported_types = supported_types
        self.stats = {
            "documents_processed": 0,
            "total_chunks_created": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }
    
    @abstractmethod
    async def process_document(self, content: str, metadata: DocumentMetadata) -> ProcessingResult:
        """Process a document and return chunks"""
        pass
    
    def supports_type(self, document_type: DocumentType) -> bool:
        """Check if processor supports the document type"""
        return document_type in self.supported_types
    
    def _update_stats(self, result: ProcessingResult) -> None:
        """Update processing statistics"""
        self.stats["documents_processed"] += 1
        self.stats["total_chunks_created"] += result.total_chunks
        self.stats["total_processing_time"] += result.processing_time_ms
        
        if result.status == ProcessingStatus.FAILED:
            self.stats["errors"] += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "name": self.name,
            "supported_types": [t.value for t in self.supported_types],
            "stats": self.stats.copy()
        }


class BaseChunkingEngine(ABC):
    """Abstract base class for chunking engines"""
    
    def __init__(self, name: str, strategy: ChunkingStrategy):
        self.name = name
        self.strategy = strategy
        self.stats = {
            "texts_chunked": 0,
            "total_chunks_created": 0,
            "avg_chunk_size": 0.0,
            "avg_chunks_per_document": 0.0
        }
    
    @abstractmethod
    async def chunk_text(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Chunk text into smaller pieces"""
        pass
    
    def get_chunk_size_estimate(self, text: str) -> int:
        """Estimate number of chunks for given text"""
        # Default implementation - override in subclasses
        return max(1, len(text) // 1000)  # Rough estimate
    
    def _update_stats(self, text: str, chunks: List[DocumentChunk]) -> None:
        """Update chunking statistics"""
        self.stats["texts_chunked"] += 1
        self.stats["total_chunks_created"] += len(chunks)
        
        # Update averages
        total_texts = self.stats["texts_chunked"]
        
        # Average chunk size
        if chunks:
            avg_chunk_size = sum(chunk.char_count for chunk in chunks) / len(chunks)
            current_avg = self.stats["avg_chunk_size"]
            self.stats["avg_chunk_size"] = (
                (current_avg * (total_texts - 1) + avg_chunk_size) / total_texts
            )
        
        # Average chunks per document
        current_avg_chunks = self.stats["avg_chunks_per_document"]
        self.stats["avg_chunks_per_document"] = (
            (current_avg_chunks * (total_texts - 1) + len(chunks)) / total_texts
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "stats": self.stats.copy()
        }


class IndexingError(Exception):
    """Base exception for indexing errors"""
    pass


class DocumentProcessingError(IndexingError):
    """Exception raised for document processing errors"""
    pass


class ChunkingError(IndexingError):
    """Exception raised for chunking errors"""
    pass


class EmbeddingGenerationError(IndexingError):
    """Exception raised for embedding generation errors"""
    pass
