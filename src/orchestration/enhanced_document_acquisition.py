# src/orchestration/enhanced_document_acquisition.py

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import pymupdf

from ..storage.vector.manager import VectorStoreManager
from ..utils.embedding import EmbeddingService
from ..storage.vector.base import VectorDocument


logger = logging.getLogger(__name__)


class EnhancedDocumentAcquisition:
    """Enhanced document acquisition with vector storage integration."""
    
    def __init__(self, vector_manager: VectorStoreManager, embedding_service: EmbeddingService, storage_path: str):
        self.vector_manager = vector_manager
        self.embedding_service = embedding_service
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize HTTP session with retry logic
        self.session = self._create_robust_session()
        
        # Document processing stats
        self.processing_stats = {
            'documents_acquired': 0,
            'extraction_successes': 0,
            'extraction_failures': 0,
            'vector_storage_successes': 0,
            'vector_storage_failures': 0
        }
    
    def _create_robust_session(self) -> requests.Session:
        """Create HTTP session with retry and error handling."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={'GET', 'HEAD'}
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        
        session.headers.update({
            'User-Agent': 'PyGent Historical Research System 1.0 (Academic Use)'
        })
        
        return session
    
    async def acquire_and_process_document(self, url: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Complete document acquisition and processing pipeline."""
        try:
            # Step 1: Download document
            download_result = await self._download_document(url, metadata)
            if 'error' in download_result:
                return download_result
            
            # Step 2: Extract text with multiple methods
            extraction_result = await self._extract_comprehensive_text(
                download_result['file_path'], 
                metadata
            )
            if 'error' in extraction_result:
                return extraction_result
            
            # Step 3: Create vector embeddings and store
            vector_result = await self._create_and_store_vectors(
                extraction_result['text_data'],
                metadata,
                download_result['document_id']
            )
            
            # Step 4: Store processing metadata
            processing_metadata = {
                'document_id': download_result['document_id'],
                'source_url': url,
                'download_metadata': download_result['download_metadata'],
                'extraction_metadata': extraction_result['extraction_metadata'],
                'vector_metadata': vector_result,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'original_metadata': metadata
            }
            
            await self._store_processing_metadata(processing_metadata)
            
            # Update stats
            self.processing_stats['documents_acquired'] += 1
            if 'error' not in extraction_result:
                self.processing_stats['extraction_successes'] += 1
            if 'error' not in vector_result:
                self.processing_stats['vector_storage_successes'] += 1
            
            return {
                'success': True,
                'document_id': download_result['document_id'],
                'file_path': download_result['file_path'],
                'text_quality': extraction_result['text_quality'],
                'vector_chunks': vector_result.get('chunks_stored', 0),
                'processing_metadata': processing_metadata
            }
            
        except Exception as e:
            logger.error(f"Document acquisition failed for {url}: {e}")
            return {'error': f'Acquisition failed: {str(e)}'}
    
    async def _download_document(self, url: str, metadata: Dict) -> Dict[str, Any]:
        """Download document with validation."""
        try:
            # Generate document ID
            doc_id = hashlib.md5(url.encode()).hexdigest()
            file_path = self.storage_path / f"{doc_id}.pdf"
            
            # Skip if already exists
            if file_path.exists():
                logger.info(f"Document already exists: {doc_id}")
                return {
                    'document_id': doc_id,
                    'file_path': str(file_path),
                    'download_metadata': {'status': 'already_exists'}
                }
            
            # Download with streaming
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                logger.warning(f"Unexpected content type: {content_type}")
            
            # Stream download
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
            
            # Validate PDF
            if not content.startswith(b'%PDF-'):
                return {'error': 'Downloaded content is not a valid PDF'}
            
            # Save to disk
            with open(file_path, 'wb') as f:
                f.write(content)
            
            download_metadata = {
                'url': url,
                'download_timestamp': datetime.utcnow().isoformat(),
                'content_length': len(content),
                'content_type': content_type,
                'status_code': response.status_code,
                'headers': dict(response.headers)
            }
            
            return {
                'document_id': doc_id,
                'file_path': str(file_path),
                'download_metadata': download_metadata
            }
            
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return {'error': f'Download failed: {str(e)}'}
    
    async def _extract_comprehensive_text(self, file_path: str, metadata: Dict) -> Dict[str, Any]:
        """Extract text using multiple methods with quality assessment."""
        try:
            doc = pymupdf.open(file_path)
            text_data = {}
            extraction_metadata = {'methods_used': []}
            
            # Method 1: Standard text extraction
            standard_text = []
            structured_data = []
            
            logger.info(f"Starting text extraction for {len(doc)} pages...")
            for page_num, page in enumerate(doc):
                logger.info(f"Extracting text from page {page_num + 1}/{len(doc)}...")
                # Get plain text
                page_text = page.get_text('text')
                standard_text.append(page_text)
                
                # Get structured text data
                text_dict = page.get_text('dict')
                for block in text_dict.get('blocks', []):
                    if block.get('type') == 0:  # Text block
                        for line in block.get('lines', []):
                            for span in line.get('spans', []):
                                structured_data.append({
                                    'text': span.get('text', ''),
                                    'font': span.get('font', ''),
                                    'size': span.get('size', 12),
                                    'flags': span.get('flags', 0),
                                    'bbox': span.get('bbox', []),
                                    'page': page_num
                                })
            
            extraction_metadata['methods_used'].append('standard')
            
            # Method 2: OCR if text quality is poor
            combined_text = '\n'.join(standard_text)
            text_quality = self._assess_text_quality(combined_text)
            
            if text_quality['quality'] == 'low':
                logger.info("Low text quality detected, attempting OCR")
                try:
                    ocr_text = []
                    for page_num, page in enumerate(doc):
                        textpage_ocr = page.get_textpage_ocr(language='eng', dpi=300, full=True)
                        ocr_page_text = page.get_text(textpage=textpage_ocr)
                        ocr_text.append(ocr_page_text)
                    
                    # Use OCR if significantly better
                    ocr_combined = '\n'.join(ocr_text)
                    ocr_quality = self._assess_text_quality(ocr_combined)
                    
                    if ocr_quality['score'] > text_quality['score'] * 1.2:
                        standard_text = ocr_text
                        combined_text = ocr_combined
                        text_quality = ocr_quality
                        extraction_metadata['methods_used'].append('ocr')
                        
                except Exception as e:
                    logger.warning(f"OCR failed: {e}")
            
            doc.close()
            
            text_data = {
                'page_texts': standard_text,
                'combined_text': combined_text,
                'structured_data': structured_data,
                'total_pages': len(standard_text),
                'total_characters': len(combined_text),
                'quality_assessment': text_quality
            }
            
            extraction_metadata.update({
                'extraction_timestamp': datetime.utcnow().isoformat(),
                'total_pages': len(standard_text),
                'text_quality': text_quality,
                'file_path': file_path
            })
            
            return {
                'text_data': text_data,
                'text_quality': text_quality,
                'extraction_metadata': extraction_metadata
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            self.processing_stats['extraction_failures'] += 1
            return {'error': f'Text extraction failed: {str(e)}'}
    
    def _assess_text_quality(self, text: str) -> Dict[str, Any]:
        """Assess extracted text quality."""
        if not text:
            return {'quality': 'empty', 'score': 0}
        
        char_count = len(text)
        word_count = len(text.split())
        alpha_ratio = sum(c.isalpha() for c in text) / char_count if char_count > 0 else 0
        
        # Common OCR artifacts
        artifacts = ['□', '■', '○', '●', '?', '…', '§', '†']
        artifact_ratio = sum(text.count(a) for a in artifacts) / char_count if char_count > 0 else 0
        
        # Calculate quality score
        score = alpha_ratio - artifact_ratio * 2  # Penalize artifacts more
        
        if score > 0.8:
            quality = 'high'
        elif score > 0.5:
            quality = 'medium'
        else:
            quality = 'low'
        
        return {
            'quality': quality,
            'score': score,
            'char_count': char_count,
            'word_count': word_count,
            'alpha_ratio': alpha_ratio,
            'artifact_ratio': artifact_ratio
        }
    
    async def _create_and_store_vectors(self, text_data: Dict, metadata: Dict, doc_id: str) -> Dict[str, Any]:
        """Create vector embeddings and store in vector database."""
        try:
            # Create semantic chunks
            chunks = await self._create_semantic_chunks(text_data, metadata)
            
            # Generate embeddings
            vector_documents = []
            for i, chunk in enumerate(chunks):
                embedding_result = await self.embedding_service.generate_embedding(chunk['text'])
                
                vector_doc = VectorDocument(
                    id=f"{doc_id}_chunk_{i}",
                    content=chunk['text'],
                    embedding=embedding_result.embedding,
                    metadata={
                        **metadata,
                        'document_id': doc_id,
                        'chunk_index': i,
                        'chunk_type': chunk['type'],
                        'page_number': chunk.get('page', 0),
                        'total_chunks': len(chunks),
                        'extraction_method': chunk.get('method', 'standard'),
                        'created_at': datetime.utcnow().isoformat()
                    },
                    collection="historical_documents",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                vector_documents.append(vector_doc)
            
            # Store in vector database
            store = await self.vector_manager.get_store()
            doc_ids = await store.add_documents(vector_documents)
            
            self.processing_stats['vector_storage_successes'] += 1
            
            return {
                'success': True,
                'chunks_stored': len(doc_ids),
                'document_ids': doc_ids,
                'embedding_model': embedding_result.model
            }
            
        except Exception as e:
            logger.error(f"Vector storage failed for {doc_id}: {e}")
            self.processing_stats['vector_storage_failures'] += 1
            return {'error': f'Vector storage failed: {str(e)}'}
    
    async def _create_semantic_chunks(self, text_data: Dict, metadata: Dict) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks from text data."""
        chunks = []
        
        # Add metadata chunk if available
        if metadata.get('title') or metadata.get('abstract'):
            chunks.append({
                'text': f"Title: {metadata.get('title', 'Unknown')}\n\nAbstract: {metadata.get('abstract', 'N/A')}",
                'type': 'metadata',
                'page': 0,
                'method': 'metadata'
            })
        
        # Process structured data if available
        if text_data.get('structured_data'):
            structured_chunks = self._chunk_structured_text(text_data['structured_data'])
            chunks.extend(structured_chunks)
        else:
            # Fallback to plain text chunking
            plain_chunks = self._chunk_plain_text(text_data['combined_text'])
            chunks.extend(plain_chunks)
        
        return chunks
    
    def _chunk_structured_text(self, structured_data: List[Dict]) -> List[Dict[str, Any]]:
        """Create chunks from structured text data."""
        chunks = []
        current_chunk = []
        current_size = 0
        max_chunk_size = 1000  # characters
        current_page = 0
        
        for item in structured_data:
            text = item.get('text', '').strip()
            if not text:
                continue
            
            # Check for natural break points
            is_heading = item.get('flags', 0) & 16  # Bold flag
            is_large_font = item.get('size', 12) > 14
            page_changed = item.get('page', 0) != current_page
            
            if (is_heading or is_large_font or page_changed) and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'type': 'paragraph',
                    'page': current_page,
                    'method': 'structured'
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(text)
            current_size += len(text)
            current_page = item.get('page', current_page)
            
            # Check size limit
            if current_size > max_chunk_size:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'type': 'paragraph',
                    'page': current_page,
                    'method': 'structured'
                })
                current_chunk = []
                current_size = 0
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'type': 'paragraph',
                'page': current_page,
                'method': 'structured'
            })
        
        return chunks
    
    def _chunk_plain_text(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks from plain text."""
        chunks = []
        max_chunk_size = 1000
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if current_size + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append({
                    'text': '\n\n'.join(current_chunk),
                    'type': 'paragraph',
                    'page': 0,  # Page info not available in plain text
                    'method': 'plain_text'
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(paragraph)
            current_size += len(paragraph)
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'type': 'paragraph',
                'page': 0,
                'method': 'plain_text'
            })
        
        return chunks
    
    async def _store_processing_metadata(self, metadata: Dict[str, Any]) -> None:
        """Store processing metadata for tracking."""
        metadata_file = self.storage_path / f"{metadata['document_id']}_processing.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats
