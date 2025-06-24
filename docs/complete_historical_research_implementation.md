# Complete Historical Research System Implementation Plan

## Overview
This document provides a comprehensive implementation plan that integrates all the research components to create a robust, academic-quality historical research system. This plan leverages PyGent Factory's existing infrastructure while adding specialized historical research capabilities.

## System Architecture Integration

### Current PyGent Factory Infrastructure
✅ **Vector Store System** - FAISS, ChromaDB, PostgreSQL implementations  
✅ **Embedding Service** - OpenAI + SentenceTransformers with caching  
✅ **Database Management** - SQLAlchemy with async support  
✅ **API Framework** - FastAPI with comprehensive routing  
✅ **Configuration System** - Centralized settings management  

### New Components to Implement
🚧 **Document Acquisition Pipeline** - PDF download and validation  
🚧 **Historical Text Extraction** - PyMuPDF with OCR integration  
🚧 **Vector-Enhanced Research Agent** - Context building and search  
🚧 **Anti-hallucination Framework** - Source verification and validation  
🚧 **Academic PDF Generation** - LaTeX/Pandoc with citations  

## Phase 1: Foundation (Week 1)

### 1.1 Enhanced Document Acquisition System

#### Document Downloader with Vector Integration
```python
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
            
            for page_num, page in enumerate(doc):
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
                logger.info(f"Low text quality detected, attempting OCR")
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
        return {
            **self.processing_stats,
            'success_rate': self.processing_stats['documents_acquired'] / max(1, self.processing_stats['documents_acquired']) * 100,
            'extraction_success_rate': self.processing_stats['extraction_successes'] / max(1, self.processing_stats['documents_acquired']) * 100,
            'vector_storage_success_rate': self.processing_stats['vector_storage_successes'] / max(1, self.processing_stats['documents_acquired']) * 100
        }
```

### 1.2 Vector-Enhanced Historical Research Agent

```python
# src/orchestration/vector_enhanced_historical_agent.py

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..storage.vector.manager import VectorStoreManager
from ..storage.vector.base import VectorQuery
from ..utils.embedding import EmbeddingService
from .historical_research_agent import HistoricalResearchAgent


logger = logging.getLogger(__name__)


class VectorEnhancedHistoricalAgent:
    """Enhanced historical research agent with vector search capabilities."""
    
    def __init__(self, 
                 vector_manager: VectorStoreManager, 
                 embedding_service: EmbeddingService,
                 traditional_agent: HistoricalResearchAgent):
        self.vector_manager = vector_manager
        self.embedding_service = embedding_service
        self.traditional_agent = traditional_agent
        self.collection_name = "historical_documents"
    
    async def comprehensive_research(self, query: str, scope: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute comprehensive research using both vector and traditional methods."""
        
        # Build comprehensive research context
        research_context = await self._build_comprehensive_context(query, scope or {})
        
        # Execute vector search
        vector_results = await self._execute_vector_research(query, scope)
        
        # Execute traditional search
        traditional_results = await self.traditional_agent.search_and_validate_sources(query)
        
        # Cross-validate and synthesize results
        synthesis = await self._synthesize_research_results(
            research_context, vector_results, traditional_results, query
        )
        
        # Generate comprehensive output
        return {
            'query': query,
            'scope': scope,
            'research_context': research_context,
            'vector_results': vector_results,
            'traditional_results': traditional_results,
            'synthesis': synthesis,
            'confidence_assessment': synthesis['confidence'],
            'source_validation': synthesis['source_validation'],
            'research_timestamp': datetime.utcnow().isoformat()
        }
    
    async def _build_comprehensive_context(self, query: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive research context using multiple query strategies."""
        
        # Multiple query strategies for thorough coverage
        query_strategies = [
            f"historical events {query}",
            f"primary sources {query}",
            f"scholarly analysis {query}",
            f"causes effects {query}",
            f"timeline chronology {query}",
            f"contemporary accounts {query}"
        ]
        
        all_context_results = {}
        for strategy in query_strategies:
            try:
                context_results = await self._execute_contextual_search(strategy, scope)
                all_context_results[strategy] = context_results
            except Exception as e:
                logger.warning(f"Context search failed for strategy '{strategy}': {e}")
                all_context_results[strategy] = []
        
        # Synthesize context from all strategies
        context_synthesis = await self._synthesize_context(all_context_results, query, scope)
        
        return context_synthesis
    
    async def _execute_contextual_search(self, query: str, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute contextual search with scope filters."""
        try:
            # Generate query embedding
            embedding_result = await self.embedding_service.generate_embedding(query)
            
            # Build metadata filters from scope
            metadata_filter = self._build_metadata_filter(scope)
            
            # Execute vector search
            vector_query = VectorQuery(
                query_vector=embedding_result.embedding,
                collection=self.collection_name,
                limit=20,  # Get good coverage
                similarity_threshold=0.65,
                metadata_filter=metadata_filter
            )
            
            store = await self.vector_manager.get_store()
            search_results = await store.search_similar(vector_query)
            
            # Convert to analysis format
            analysis_results = []
            for result in search_results:
                analysis_results.append({
                    'content': result.document.content,
                    'metadata': result.document.metadata,
                    'similarity_score': result.similarity_score,
                    'document_id': result.document.metadata.get('document_id'),
                    'source': result.document.metadata.get('title', 'Unknown'),
                    'source_url': result.document.metadata.get('source_url')
                })
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Contextual search failed for query '{query}': {e}")
            return []
    
    def _build_metadata_filter(self, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata filter from research scope."""
        metadata_filter = {}
        
        if scope.get('time_period'):
            metadata_filter['time_period'] = scope['time_period']
        
        if scope.get('geographical_focus'):
            metadata_filter['geography'] = scope['geographical_focus']
        
        if scope.get('source_types'):
            metadata_filter['source_type'] = {'$in': scope['source_types']}
        
        if scope.get('language'):
            metadata_filter['language'] = scope['language']
        
        return metadata_filter
    
    async def _synthesize_context(self, search_results: Dict, query: str, scope: Dict) -> Dict[str, Any]:
        """Synthesize search results into comprehensive research context."""
        
        primary_sources = []
        secondary_sources = []
        key_themes = {}
        temporal_info = []
        geographical_info = []
        source_reliability = {}
        
        # Process all search results
        for strategy, results in search_results.items():
            for result in results:
                metadata = result['metadata']
                content = result['content']
                
                # Categorize by source type and strategy
                source_type = metadata.get('source_type', 'unknown')
                
                if 'primary sources' in strategy or source_type == 'primary':
                    primary_sources.append({
                        'content': content[:300] + "..." if len(content) > 300 else content,
                        'source': result['source'],
                        'source_url': result.get('source_url'),
                        'confidence': result['similarity_score'],
                        'document_id': result['document_id']
                    })
                else:
                    secondary_sources.append({
                        'content': content[:300] + "..." if len(content) > 300 else content,
                        'source': result['source'],
                        'source_url': result.get('source_url'),
                        'confidence': result['similarity_score'],
                        'document_id': result['document_id']
                    })
                
                # Extract temporal information
                if metadata.get('date_range'):
                    temporal_info.append(metadata['date_range'])
                
                # Extract geographical information
                if metadata.get('geography'):
                    geographical_info.append(metadata['geography'])
                
                # Track source reliability
                source_id = result['document_id']
                if source_id not in source_reliability:
                    source_reliability[source_id] = {
                        'source': result['source'],
                        'confidence_scores': [],
                        'appears_in_strategies': []
                    }
                
                source_reliability[source_id]['confidence_scores'].append(result['similarity_score'])
                source_reliability[source_id]['appears_in_strategies'].append(strategy)
                
                # Extract themes (simple keyword analysis)
                words = content.lower().split()
                for word in words:
                    if len(word) > 5:  # Focus on meaningful words
                        key_themes[word] = key_themes.get(word, 0) + result['similarity_score']
        
        # Sort and filter results
        primary_sources = sorted(primary_sources, key=lambda x: x['confidence'], reverse=True)[:5]
        secondary_sources = sorted(secondary_sources, key=lambda x: x['confidence'], reverse=True)[:5]
        top_themes = sorted(key_themes.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Calculate source reliability scores
        for source_id, data in source_reliability.items():
            data['avg_confidence'] = sum(data['confidence_scores']) / len(data['confidence_scores'])
            data['strategy_coverage'] = len(set(data['appears_in_strategies']))
        
        return {
            'query': query,
            'scope': scope,
            'primary_sources': primary_sources,
            'secondary_sources': secondary_sources,
            'key_themes': dict(top_themes),
            'temporal_context': list(set(temporal_info)),
            'geographical_context': list(set(geographical_info)),
            'source_reliability': source_reliability,
            'context_strength': self._assess_context_strength(primary_sources, secondary_sources),
            'research_gaps': self._identify_potential_gaps(search_results, scope)
        }
    
    async def _execute_vector_research(self, query: str, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Execute focused vector research query."""
        try:
            embedding_result = await self.embedding_service.generate_embedding(query)
            metadata_filter = self._build_metadata_filter(scope or {})
            
            vector_query = VectorQuery(
                query_vector=embedding_result.embedding,
                collection=self.collection_name,
                limit=30,
                similarity_threshold=0.7,
                metadata_filter=metadata_filter
            )
            
            store = await self.vector_manager.get_store()
            search_results = await store.search_similar(vector_query)
            
            # Group and analyze results
            grouped_results = self._group_results_by_document(search_results)
            
            return {
                'total_results': len(search_results),
                'unique_documents': len(grouped_results),
                'grouped_results': grouped_results,
                'confidence_distribution': self._analyze_confidence_distribution(search_results),
                'source_analysis': self._analyze_sources(search_results)
            }
            
        except Exception as e:
            logger.error(f"Vector research failed: {e}")
            return {'error': str(e)}
    
    def _group_results_by_document(self, search_results: List) -> Dict[str, Any]:
        """Group search results by source document."""
        grouped = {}
        
        for result in search_results:
            doc_id = result.document.metadata.get('document_id', 'unknown')
            
            if doc_id not in grouped:
                grouped[doc_id] = {
                    'document_id': doc_id,
                    'title': result.document.metadata.get('title', 'Unknown'),
                    'source_url': result.document.metadata.get('source_url'),
                    'chunks': [],
                    'max_relevance': 0,
                    'avg_relevance': 0,
                    'total_chunks': 0
                }
            
            grouped[doc_id]['chunks'].append({
                'content': result.document.content,
                'similarity': result.similarity_score,
                'chunk_index': result.document.metadata.get('chunk_index', 0),
                'page': result.document.metadata.get('page_number', 0)
            })
            
            grouped[doc_id]['max_relevance'] = max(
                grouped[doc_id]['max_relevance'], 
                result.similarity_score
            )
            grouped[doc_id]['total_chunks'] += 1
        
        # Calculate average relevance for each document
        for doc_data in grouped.values():
            doc_data['avg_relevance'] = sum(c['similarity'] for c in doc_data['chunks']) / len(doc_data['chunks'])
        
        # Sort by relevance
        return dict(sorted(grouped.items(), key=lambda x: x[1]['max_relevance'], reverse=True))
    
    async def _synthesize_research_results(self, context, vector_results, traditional_results, query) -> Dict[str, Any]:
        """Synthesize results from all research methods."""
        
        synthesis = {
            'combined_evidence': [],
            'source_cross_validation': {},
            'confidence_assessment': {},
            'source_validation': {},
            'research_completeness': {},
            'identified_gaps': [],
            'recommendations': []
        }
        
        # Combine evidence from all sources
        vector_sources = set()
        traditional_sources = set()
        
        # Process vector results
        if 'grouped_results' in vector_results:
            for doc_id, doc_data in vector_results['grouped_results'].items():
                source_url = doc_data.get('source_url', '')
                if source_url:
                    vector_sources.add(source_url)
                
                synthesis['combined_evidence'].append({
                    'source': doc_data['title'],
                    'source_url': source_url,
                    'relevance': doc_data['max_relevance'],
                    'method': 'vector_search',
                    'chunks_found': doc_data['total_chunks']
                })
        
        # Process traditional results
        if traditional_results.get('validated_sources'):
            for source in traditional_results['validated_sources']:
                source_url = source.get('url', '')
                if source_url:
                    traditional_sources.add(source_url)
                
                synthesis['combined_evidence'].append({
                    'source': source.get('title', 'Unknown'),
                    'source_url': source_url,
                    'relevance': source.get('credibility_score', 0),
                    'method': 'traditional_search',
                    'validation_score': source.get('credibility_score', 0)
                })
        
        # Cross-validation analysis
        synthesis['source_cross_validation'] = {
            'overlapping_sources': len(vector_sources & traditional_sources),
            'unique_vector_sources': len(vector_sources - traditional_sources),
            'unique_traditional_sources': len(traditional_sources - vector_sources),
            'total_unique_sources': len(vector_sources | traditional_sources),
            'cross_validation_ratio': len(vector_sources & traditional_sources) / max(1, len(vector_sources | traditional_sources))
        }
        
        # Confidence assessment
        synthesis['confidence_assessment'] = {
            'vector_confidence': self._calculate_vector_confidence(vector_results),
            'traditional_confidence': self._calculate_traditional_confidence(traditional_results),
            'cross_validation_confidence': synthesis['source_cross_validation']['cross_validation_ratio'],
            'overall_confidence': self._calculate_overall_confidence(vector_results, traditional_results, synthesis['source_cross_validation'])
        }
        
        # Source validation
        synthesis['source_validation'] = {
            'total_sources_found': synthesis['source_cross_validation']['total_unique_sources'],
            'validated_sources': synthesis['source_cross_validation']['overlapping_sources'],
            'validation_rate': synthesis['source_cross_validation']['cross_validation_ratio'],
            'high_confidence_sources': len([s for s in synthesis['combined_evidence'] if s['relevance'] > 0.8])
        }
        
        return synthesis
    
    def _calculate_overall_confidence(self, vector_results, traditional_results, cross_validation) -> float:
        """Calculate overall research confidence score."""
        vector_conf = self._calculate_vector_confidence(vector_results)
        traditional_conf = self._calculate_traditional_confidence(traditional_results)
        cross_val_conf = cross_validation['cross_validation_ratio']
        
        # Weighted average with cross-validation bonus
        base_confidence = (vector_conf * 0.4 + traditional_conf * 0.4 + cross_val_conf * 0.2)
        
        # Bonus for high cross-validation
        if cross_val_conf > 0.3:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_vector_confidence(self, vector_results) -> float:
        """Calculate confidence from vector search results."""
        if 'error' in vector_results:
            return 0.0
        
        total_results = vector_results.get('total_results', 0)
        if total_results == 0:
            return 0.0
        
        # Base confidence on number of results and their quality
        result_coverage = min(1.0, total_results / 20)  # Normalize to 20 as good coverage
        
        confidence_dist = vector_results.get('confidence_distribution', {})
        avg_confidence = confidence_dist.get('average', 0)
        
        return (result_coverage * 0.3 + avg_confidence * 0.7)
    
    def _calculate_traditional_confidence(self, traditional_results) -> float:
        """Calculate confidence from traditional search results."""
        if not traditional_results or 'error' in traditional_results:
            return 0.0
        
        validated_sources = traditional_results.get('validated_sources', [])
        if not validated_sources:
            return 0.0
        
        avg_credibility = sum(s.get('credibility_score', 0) for s in validated_sources) / len(validated_sources)
        source_coverage = min(1.0, len(validated_sources) / 10)  # Normalize to 10 as good coverage
        
        return (avg_credibility * 0.7 + source_coverage * 0.3)
```

## Implementation Timeline

### Week 1: Foundation
- ✅ Document acquisition pipeline with vector integration
- ✅ Enhanced text extraction with OCR fallback  
- ✅ Vector storage and embedding integration
- ✅ Basic vector search capabilities

### Week 2: Research Integration  
- 🚧 Vector-enhanced research agent
- 🚧 Multi-strategy context building
- 🚧 Cross-validation between vector and traditional search
- 🚧 Confidence scoring and source validation

### Week 3: Quality & Validation
- 🚧 Anti-hallucination framework integration
- 🚧 Academic PDF generation with LaTeX
- 🚧 Comprehensive testing with real historical data
- 🚧 Performance optimization and monitoring

This comprehensive implementation leverages all of PyGent Factory's existing infrastructure while adding the specialized capabilities needed for rigorous historical research. The modular design allows for incremental development and testing.
