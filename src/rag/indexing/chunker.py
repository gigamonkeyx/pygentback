"""
Document Chunking Implementations

Provides various chunking strategies for document processing in the RAG system.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .base import BaseChunkingEngine, ChunkingStrategy

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    chunk_id: str
    document_id: str
    chunk_index: int
    start_position: int
    end_position: int
    chunk_type: str
    word_count: int
    char_count: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentChunk:
    """A chunk of a document"""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class TextChunker(BaseChunkingEngine):
    """
    Simple text-based chunking strategy.
    
    Splits documents into fixed-size chunks with optional overlap.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        super().__init__("text_chunker", ChunkingStrategy.FIXED_SIZE)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk a document into fixed-size text chunks.
        
        Args:
            document: Document with 'content' and 'id' fields
            
        Returns:
            List of document chunks
        """
        try:
            content = document.get('content', '')
            doc_id = document.get('id', 'unknown')
            
            if not content:
                return []
            
            chunks = []
            start = 0
            chunk_index = 0
            
            while start < len(content):
                # Calculate end position
                end = min(start + self.chunk_size, len(content))
                
                # Extract chunk content
                chunk_content = content[start:end]
                
                # Create metadata
                metadata = ChunkMetadata(
                    chunk_id=f"{doc_id}_chunk_{chunk_index}",
                    document_id=doc_id,
                    chunk_index=chunk_index,
                    start_position=start,
                    end_position=end,
                    chunk_type="text",
                    word_count=len(chunk_content.split()),
                    char_count=len(chunk_content),
                    metadata=document.get('metadata', {})
                )
                
                # Create chunk
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata=metadata
                )
                chunks.append(chunk)
                
                # Move to next chunk with overlap
                start = end - self.overlap
                chunk_index += 1
                
                # Prevent infinite loop
                if start >= end:
                    break
            
            logger.debug(f"Text chunker created {len(chunks)} chunks for document {doc_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            return []


class SemanticChunker(BaseChunkingEngine):
    """
    Semantic-based chunking strategy.
    
    Splits documents based on semantic boundaries like sentences and paragraphs.
    """
    
    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 100):
        super().__init__("semantic_chunker", ChunkingStrategy.SEMANTIC)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
    def chunk_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk a document based on semantic boundaries.
        
        Args:
            document: Document with 'content' and 'id' fields
            
        Returns:
            List of document chunks
        """
        try:
            content = document.get('content', '')
            doc_id = document.get('id', 'unknown')
            
            if not content:
                return []
            
            # Split into paragraphs first
            paragraphs = self._split_paragraphs(content)
            
            chunks = []
            current_chunk = ""
            chunk_index = 0
            start_position = 0
            
            for paragraph in paragraphs:
                # If adding this paragraph would exceed max size, finalize current chunk
                if (len(current_chunk) + len(paragraph) > self.max_chunk_size and 
                    len(current_chunk) >= self.min_chunk_size):
                    
                    # Create chunk from current content
                    chunk = self._create_chunk(
                        current_chunk, doc_id, chunk_index, 
                        start_position, start_position + len(current_chunk),
                        document.get('metadata', {})
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk
                    start_position += len(current_chunk)
                    current_chunk = paragraph
                    chunk_index += 1
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # Add final chunk if it has content
            if current_chunk and len(current_chunk) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    current_chunk, doc_id, chunk_index,
                    start_position, start_position + len(current_chunk),
                    document.get('metadata', {})
                )
                chunks.append(chunk)
            
            logger.debug(f"Semantic chunker created {len(chunks)} chunks for document {doc_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            return []
    
    def _split_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs"""
        # Split on double newlines, filter empty paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        return paragraphs
    
    def _create_chunk(self, content: str, doc_id: str, chunk_index: int, 
                     start_pos: int, end_pos: int, doc_metadata: Dict[str, Any]) -> DocumentChunk:
        """Create a document chunk with metadata"""
        metadata = ChunkMetadata(
            chunk_id=f"{doc_id}_semantic_{chunk_index}",
            document_id=doc_id,
            chunk_index=chunk_index,
            start_position=start_pos,
            end_position=end_pos,
            chunk_type="semantic",
            word_count=len(content.split()),
            char_count=len(content),
            metadata=doc_metadata
        )
        
        return DocumentChunk(content=content, metadata=metadata)


class HierarchicalChunker(BaseChunkingEngine):
    """
    Hierarchical chunking strategy.
    
    Creates chunks at multiple levels (sections, subsections, paragraphs).
    """
    
    def __init__(self, levels: List[int] = None):
        super().__init__("hierarchical_chunker", ChunkingStrategy.HIERARCHICAL)
        self.levels = levels or [2000, 1000, 500]  # Large, medium, small chunks
        
    def chunk_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk a document hierarchically at multiple levels.
        
        Args:
            document: Document with 'content' and 'id' fields
            
        Returns:
            List of document chunks at multiple hierarchical levels
        """
        try:
            content = document.get('content', '')
            doc_id = document.get('id', 'unknown')
            
            if not content:
                return []
            
            all_chunks = []
            
            # Create chunks at each hierarchical level
            for level_index, chunk_size in enumerate(self.levels):
                level_chunks = self._create_level_chunks(
                    content, doc_id, level_index, chunk_size,
                    document.get('metadata', {})
                )
                all_chunks.extend(level_chunks)
            
            logger.debug(f"Hierarchical chunker created {len(all_chunks)} chunks for document {doc_id}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Hierarchical chunking failed: {e}")
            return []
    
    def _create_level_chunks(self, content: str, doc_id: str, level: int, 
                           chunk_size: int, doc_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create chunks for a specific hierarchical level"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_content = content[start:end]
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence ending within last 100 characters
                sentence_end = chunk_content.rfind('.', max(0, len(chunk_content) - 100))
                if sentence_end > len(chunk_content) * 0.8:  # Only if we don't lose too much
                    chunk_content = chunk_content[:sentence_end + 1]
                    end = start + len(chunk_content)
            
            metadata = ChunkMetadata(
                chunk_id=f"{doc_id}_level{level}_{chunk_index}",
                document_id=doc_id,
                chunk_index=chunk_index,
                start_position=start,
                end_position=end,
                chunk_type=f"hierarchical_level_{level}",
                word_count=len(chunk_content.split()),
                char_count=len(chunk_content),
                metadata={**doc_metadata, "hierarchical_level": level}
            )
            
            chunk = DocumentChunk(content=chunk_content, metadata=metadata)
            chunks.append(chunk)
            
            start = end
            chunk_index += 1
        
        return chunks
