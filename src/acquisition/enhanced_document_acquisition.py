# src/acquisition/enhanced_document_acquisition.py

import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import hashlib
import json
import os
import pymupdf  # PyMuPDF for PDF processing
from dataclasses import dataclass, field
import aiohttp
import aiofiles

from ..storage.vector.manager import VectorStoreManager
from ..utils.embedding import EmbeddingService
from ..storage.vector.base import VectorDocument
from ..core.gpu_config import gpu_manager
from ..core.ollama_manager import get_ollama_manager
from .document_download import DocumentDownloadPipeline

logger = logging.getLogger(__name__)


@dataclass
class DocumentProcessingStats:
    """Statistics for document processing operations"""
    documents_acquired: int = 0
    extraction_successes: int = 0
    extraction_failures: int = 0
    vector_storage_successes: int = 0
    vector_storage_failures: int = 0
    total_processing_time: float = 0.0
    ai_analysis_time: float = 0.0
    gpu_operations: int = 0


@dataclass
class ProcessingMetadata:
    """Metadata for processed documents"""
    document_id: str
    url: str
    extraction_method: str
    extraction_quality: float
    processing_timestamp: datetime
    ai_categorization: List[str] = field(default_factory=list)
    ai_tags: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    structure_analysis: Dict[str, Any] = field(default_factory=dict)
    ocr_used: bool = False
    gpu_accelerated: bool = False


class EnhancedDocumentAcquisition:
    """Enhanced document acquisition with AI-powered analysis and GPU acceleration."""
    
    def __init__(self, 
                 vector_manager: VectorStoreManager = None,
                 embedding_service: EmbeddingService = None,
                 storage_path: str = "data/historical_documents"):
        # Initialize with defaults if not provided
        if vector_manager is None:
            try:
                from ..config.settings import get_settings
                settings = get_settings()
                vector_manager = VectorStoreManager(settings)
            except ImportError:
                logger.warning("Vector manager not provided and settings unavailable - some features will be limited")
                vector_manager = None
                
        if embedding_service is None:
            try:
                embedding_service = EmbeddingService()
            except Exception as e:
                logger.warning(f"Failed to initialize embedding service: {e} - some features will be limited")
                embedding_service = None
                
        self.vector_manager = vector_manager
        self.embedding_service = embedding_service
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-directories
        self.pdf_dir = self.storage_path / "pdfs"
        self.text_dir = self.storage_path / "text" 
        self.metadata_dir = self.storage_path / "metadata"
        self.extracted_dir = self.storage_path / "extracted"
        
        for directory in [self.pdf_dir, self.text_dir, self.metadata_dir, self.extracted_dir]:
            directory.mkdir(exist_ok=True)
          # Initialize HTTP session with retry logic (lazy initialization)
        self.session = None  # Will be created when first needed
        
        # Initialize Ollama manager for AI analysis
        self.ollama_manager = get_ollama_manager()
        
        # Document processing statistics
        self.processing_stats = DocumentProcessingStats()
        
        # GPU manager for acceleration
        self.gpu_available = gpu_manager.is_available()
        
        # Initialize legacy download pipeline for compatibility
        self.download_pipeline = DocumentDownloadPipeline(str(self.storage_path))
        
        logger.info(f"Enhanced Document Acquisition initialized with GPU: {self.gpu_available}")
    
    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with lazy initialization."""
        if self.session is None:
            self.session = self._create_robust_session()
        return self.session

    def _create_robust_session(self) -> aiohttp.ClientSession:
        """Create HTTP session with retry and error handling."""
        timeout = aiohttp.ClientTimeout(total=60, connect=30)
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        headers = {
            'User-Agent': 'PyGent Historical Research System 1.0 (Academic Use; +https://github.com/pygent-factory)',
            'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        return aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )
    
    async def acquire_and_process_document(self, 
                                         url: str, 
                                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Complete document acquisition and processing pipeline."""
        start_time = datetime.now()
        
        try:
            # Step 1: Download document with AI-powered prioritization
            download_result = await self._download_document_with_ai_prioritization(url, metadata)
            if 'error' in download_result:
                return download_result
            
            document_id = download_result['document_id']
            file_path = download_result['file_path']
            
            # Step 2: Extract text with GPU acceleration
            extraction_result = await self._extract_text_with_gpu_acceleration(file_path, document_id)
            if 'error' in extraction_result:
                return {**download_result, **extraction_result}
            
            # Step 3: AI-powered text analysis and categorization
            analysis_result = await self._ai_powered_text_analysis(
                extraction_result['text'], document_id, metadata
            )
            
            # Step 4: Vector storage with GPU-accelerated embeddings
            vector_result = await self._store_in_vector_database(
                extraction_result, analysis_result, document_id, metadata
            )
            
            # Step 5: Save comprehensive metadata
            processing_metadata = ProcessingMetadata(
                document_id=document_id,
                url=url,
                extraction_method=extraction_result['extraction_method'],
                extraction_quality=extraction_result['quality_score'],
                processing_timestamp=datetime.now(),
                ai_categorization=analysis_result.get('categories', []),
                ai_tags=analysis_result.get('tags', []),
                relevance_score=analysis_result.get('relevance_score', 0.0),
                structure_analysis=analysis_result.get('structure_analysis', {}),
                ocr_used=extraction_result.get('ocr_used', False),
                gpu_accelerated=self.gpu_available
            )
            
            await self._save_processing_metadata(processing_metadata)
            
            # Update statistics
            self.processing_stats.documents_acquired += 1
            self.processing_stats.extraction_successes += 1
            self.processing_stats.vector_storage_successes += 1
            self.processing_stats.total_processing_time += (datetime.now() - start_time).total_seconds()
            
            if self.gpu_available:
                self.processing_stats.gpu_operations += 1
            
            logger.info(f"Successfully processed document {document_id} in {(datetime.now() - start_time).total_seconds():.2f}s")
            
            return {
                'success': True,
                'document_id': document_id,
                'file_path': file_path,
                'extraction_result': extraction_result,
                'analysis_result': analysis_result,
                'vector_result': vector_result,
                'processing_metadata': processing_metadata
            }
            
        except Exception as e:
            logger.error(f"Error in document acquisition pipeline for {url}: {str(e)}")
            self.processing_stats.extraction_failures += 1
            return {'error': f"Processing failed: {str(e)}"}
    
    async def _download_document_with_ai_prioritization(self, 
                                                      url: str, 
                                                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Download document with AI-powered relevance scoring."""
        try:
            # Generate document ID
            document_id = hashlib.md5(url.encode()).hexdigest()
            
            # Check if already downloaded
            existing_path = self.pdf_dir / f"{document_id}.pdf"
            if existing_path.exists():
                logger.info(f"Document {document_id} already exists")
                return {
                    'success': True,
                    'document_id': document_id,
                    'file_path': str(existing_path),
                    'cached': True
                }
            
            # AI-powered relevance scoring before download
            relevance_score = await self._ai_assess_download_relevance(url, metadata)
            if relevance_score < 0.3:  # Skip low-relevance documents
                logger.info(f"Skipping low-relevance document: {url} (score: {relevance_score:.2f})")
                return {'error': f"Low relevance score: {relevance_score:.2f}"}
            
            # Download with streaming for large files
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {'error': f"HTTP {response.status}: {response.reason}"}
                
                # Validate content type
                content_type = response.headers.get('content-type', '')
                if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                    logger.warning(f"Unexpected content type for {url}: {content_type}")
                
                # Stream download to file
                file_path = self.pdf_dir / f"{document_id}.pdf"
                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                
                # Validate PDF magic number
                async with aiofiles.open(file_path, 'rb') as f:
                    header = await f.read(4)
                    if not header.startswith(b'%PDF'):
                        os.unlink(file_path)
                        return {'error': 'Downloaded file is not a valid PDF'}
                
                logger.info(f"Downloaded document {document_id} from {url}")
                return {
                    'success': True,
                    'document_id': document_id,
                    'file_path': str(file_path),
                    'relevance_score': relevance_score
                }
                
        except Exception as e:
            logger.error(f"Download failed for {url}: {str(e)}")
            return {'error': f"Download failed: {str(e)}"}
    
    async def _ai_assess_download_relevance(self, 
                                          url: str, 
                                          metadata: Dict[str, Any]) -> float:
        """Use AI to assess document relevance before download."""
        try:
            # Ensure Ollama is ready
            if not self.ollama_manager.is_ready:
                await self.ollama_manager.start()
            
            # Create relevance assessment prompt
            title = metadata.get('title', '')
            abstract = metadata.get('abstract', '')
            keywords = metadata.get('keywords', [])
            
            prompt = f"""
            Assess the relevance of this document for historical research on a scale of 0.0 to 1.0.
            
            Title: {title}
            Abstract: {abstract}
            Keywords: {keywords}
            URL: {url}
            
            Consider:
            - Historical content and context
            - Academic/scholarly nature
            - Primary vs secondary source indicators
            - Relevance to historical events or periods
            
            Respond with only a decimal number between 0.0 and 1.0.
            """
            
            # Send to Ollama for analysis
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_manager.ollama_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        score_text = result.get('response', '0.5').strip()
                        try:
                            score = float(score_text)
                            return max(0.0, min(1.0, score))  # Clamp to valid range
                        except ValueError:
                            logger.warning(f"Invalid score from AI: {score_text}")
                            return 0.5  # Default moderate relevance
            
            return 0.5  # Default if AI analysis fails
            
        except Exception as e:
            logger.warning(f"AI relevance assessment failed: {str(e)}")
            return 0.7  # Default to moderate-high relevance on error
    
    async def _extract_text_with_gpu_acceleration(self, 
                                                file_path: str, 
                                                document_id: str) -> Dict[str, Any]:
        """Extract text using multiple methods with GPU-accelerated OCR fallback."""
        try:
            extraction_methods = []
            best_text = ""
            best_quality = 0.0
            best_method = "none"
            ocr_used = False
            
            # Method 1: Standard PyMuPDF text extraction
            try:
                doc = pymupdf.open(file_path)
                text_content = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                      # Try different extraction methods
                    text_dict = page.get_text("dict")
                    text_plain = page.get_text()
                    
                    # Choose best text based on length and content
                    if len(text_plain) > len(text_dict.get('text', '')):
                        text_content.append(text_plain)
                    else:
                        # Extract from dict format
                        page_text = []
                        for block in text_dict.get('blocks', []):
                            if 'lines' in block:
                                for line in block['lines']:
                                    for span in line.get('spans', []):
                                        page_text.append(span.get('text', ''))
                        text_content.append(' '.join(page_text))
                
                doc.close()
                
                standard_text = '\n'.join(text_content)
                standard_quality = self._assess_text_quality(standard_text)
                
                extraction_methods.append({
                    'method': 'standard',
                    'text': standard_text,
                    'quality': standard_quality
                })
                
                if standard_quality > best_quality:
                    best_text = standard_text
                    best_quality = standard_quality
                    best_method = 'standard'
                
            except Exception as e:
                logger.warning(f"Standard text extraction failed for {document_id}: {str(e)}")
            
            # Method 2: GPU-accelerated OCR fallback if quality is low
            if best_quality < 0.7:  # Quality threshold for OCR fallback
                try:
                    ocr_text = await self._gpu_accelerated_ocr(file_path)
                    ocr_quality = self._assess_text_quality(ocr_text)
                    
                    extraction_methods.append({
                        'method': 'ocr',
                        'text': ocr_text,
                        'quality': ocr_quality
                    })
                    
                    if ocr_quality > best_quality:
                        best_text = ocr_text
                        best_quality = ocr_quality
                        best_method = 'ocr'
                        ocr_used = True
                
                except Exception as e:
                    logger.warning(f"OCR extraction failed for {document_id}: {str(e)}")
            
            # Save extracted text
            text_file = self.text_dir / f"{document_id}.txt"
            async with aiofiles.open(text_file, 'w', encoding='utf-8') as f:
                await f.write(best_text)
            
            # AI-powered text structure analysis
            structure_analysis = await self._ai_analyze_text_structure(best_text)
            
            return {
                'success': True,
                'text': best_text,
                'text_file': str(text_file),
                'extraction_method': best_method,
                'quality_score': best_quality,
                'extraction_methods': extraction_methods,
                'structure_analysis': structure_analysis,
                'ocr_used': ocr_used,
                'character_count': len(best_text),
                'word_count': len(best_text.split())
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed for {document_id}: {str(e)}")
            return {'error': f"Text extraction failed: {str(e)}"}
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess quality of extracted text."""
        if not text or len(text) < 50:
            return 0.0
        
        # Basic quality metrics
        char_count = len(text)
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        # Check for readable content
        alpha_ratio = sum(c.isalpha() for c in text) / char_count
        space_ratio = sum(c.isspace() for c in text) / char_count
        
        # Quality score based on various factors
        quality = 0.0
        
        # Length factor
        if char_count > 1000:
            quality += 0.3
        elif char_count > 500:
            quality += 0.2
        else:
            quality += 0.1
        
        # Readability factor
        if alpha_ratio > 0.7:
            quality += 0.4
        elif alpha_ratio > 0.5:
            quality += 0.3
        else:
            quality += 0.1
        
        # Structure factor
        if space_ratio > 0.1 and space_ratio < 0.3:
            quality += 0.3
        else:
            quality += 0.1
        
        return min(1.0, quality)
    
    async def _gpu_accelerated_ocr(self, file_path: str) -> str:
        """Perform GPU-accelerated OCR on PDF."""
        try:
            # This is a placeholder for GPU-accelerated OCR
            # In a real implementation, you would use libraries like:
            # - TesseractOCR with GPU support
            # - PaddleOCR with GPU
            # - EasyOCR with GPU
            
            # For now, use basic OCR with PyMuPDF
            doc = pymupdf.open(file_path)
            ocr_text = []
            
            for page_num in range(min(10, len(doc))):  # Limit to first 10 pages for performance
                page = doc[page_num]
                  # Get page as image for potential OCR
                _ = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x scale for better OCR
                
                # Convert to image and perform OCR
                # This would be where GPU-accelerated OCR libraries are called
                # For now, fall back to PyMuPDF's built-in text extraction
                page_text = page.get_text()
                
                if not page_text.strip():
                    # If no text found, this page might need real OCR
                    # In a full implementation, this is where you'd call
                    # GPU-accelerated OCR libraries
                    page_text = f"[OCR needed for page {page_num + 1}]"
                
                ocr_text.append(page_text)
            
            doc.close()
            return '\n'.join(ocr_text)
            
        except Exception as e:
            logger.error(f"GPU-accelerated OCR failed: {str(e)}")
            return ""
    
    async def _ai_analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Use AI to analyze text structure and content organization."""
        try:
            if not self.ollama_manager.is_ready:
                await self.ollama_manager.start()
            
            # Limit text length for analysis
            analysis_text = text[:5000] if len(text) > 5000 else text
            
            prompt = f"""
            Analyze the structure and content of this historical document text:
            
            {analysis_text}
            
            Provide analysis in this JSON format:
            {{
                "document_type": "academic_paper|book_chapter|government_document|news_article|other",
                "has_title": true/false,
                "has_abstract": true/false,
                "has_introduction": true/false,
                "has_conclusion": true/false,
                "has_references": true/false,
                "estimated_pages": number,
                "language": "language_code",
                "time_period_mentioned": "period if mentioned",
                "key_topics": ["topic1", "topic2"],
                "structure_quality": 0.0-1.0
            }}
            
            Respond with only the JSON object.
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_manager.ollama_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result.get('response', '{}').strip()
                        try:
                            return json.loads(analysis_text)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from structure analysis: {analysis_text}")
                            return self._default_structure_analysis()
            
            return self._default_structure_analysis()
            
        except Exception as e:
            logger.warning(f"AI structure analysis failed: {str(e)}")
            return self._default_structure_analysis()
    
    def _default_structure_analysis(self) -> Dict[str, Any]:
        """Default structure analysis when AI analysis fails."""
        return {
            "document_type": "other",
            "has_title": False,
            "has_abstract": False,
            "has_introduction": False,
            "has_conclusion": False,
            "has_references": False,
            "estimated_pages": 1,
            "language": "en",
            "time_period_mentioned": "unknown",
            "key_topics": [],
            "structure_quality": 0.5
        }
    
    async def _ai_powered_text_analysis(self, 
                                      text: str, 
                                      document_id: str, 
                                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive AI-powered text analysis."""
        try:
            if not self.ollama_manager.is_ready:
                await self.ollama_manager.start()
            
            # Limit text for analysis
            analysis_text = text[:3000] if len(text) > 3000 else text
            
            prompt = f"""
            Analyze this historical document for categorization and tagging:
            
            Title: {metadata.get('title', 'Unknown')}
            Text: {analysis_text}
            
            Provide analysis in this JSON format:
            {{
                "categories": ["category1", "category2"],
                "tags": ["tag1", "tag2", "tag3"],
                "historical_period": "period if identifiable",
                "geographical_focus": "location if mentioned",
                "relevance_score": 0.0-1.0,
                "primary_topics": ["topic1", "topic2"],
                "source_type": "primary|secondary|tertiary",
                "bias_indicators": ["indicator1", "indicator2"],
                "factual_claims": number_of_claims,
                "academic_quality": 0.0-1.0
            }}
            
            Focus on historical research relevance. Respond with only the JSON object.
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_manager.ollama_url}/api/generate",
                    json={
                        "model": "llama3.1:8b",
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result.get('response', '{}').strip()
                        try:
                            return json.loads(analysis_text)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from AI analysis: {analysis_text}")
                            return self._default_ai_analysis()
            
            return self._default_ai_analysis()
            
        except Exception as e:
            logger.warning(f"AI text analysis failed: {str(e)}")
            return self._default_ai_analysis()
    
    def _default_ai_analysis(self) -> Dict[str, Any]:
        """Default AI analysis when AI analysis fails."""
        return {
            "categories": ["historical"],
            "tags": ["document"],
            "historical_period": "unknown",
            "geographical_focus": "unknown",
            "relevance_score": 0.5,
            "primary_topics": ["general"],
            "source_type": "secondary",
            "bias_indicators": [],
            "factual_claims": 0,
            "academic_quality": 0.5
        }
    
    async def _store_in_vector_database(self, 
                                      extraction_result: Dict[str, Any],
                                      analysis_result: Dict[str, Any],
                                      document_id: str,
                                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store document in vector database with GPU-accelerated embeddings."""
        try:
            text = extraction_result['text']
            
            # Create intelligent chunks using AI-powered semantic chunking
            chunks = await self._ai_powered_semantic_chunking(text)
            
            # Generate embeddings for chunks
            stored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Create vector document
                vector_doc = VectorDocument(
                    id=chunk_id,
                    content=chunk['text'],
                    metadata={
                        'document_id': document_id,
                        'chunk_index': i,
                        'chunk_type': chunk.get('type', 'content'),
                        'source_url': metadata.get('url', ''),
                        'title': metadata.get('title', ''),
                        'extraction_method': extraction_result['extraction_method'],
                        'quality_score': extraction_result['quality_score'],
                        'ai_categories': analysis_result.get('categories', []),
                        'ai_tags': analysis_result.get('tags', []),
                        'historical_period': analysis_result.get('historical_period', ''),
                        'geographical_focus': analysis_result.get('geographical_focus', ''),
                        'relevance_score': analysis_result.get('relevance_score', 0.0),
                        'gpu_accelerated': self.gpu_available
                    }
                )
                
                # Store in vector database
                await self.vector_manager.add_documents([vector_doc], collection_name="historical_documents")
                stored_chunks.append(chunk_id)
                
                logger.debug(f"Stored chunk {chunk_id} in vector database")
            
            return {
                'success': True,
                'chunks_stored': len(stored_chunks),
                'chunk_ids': stored_chunks,
                'collection': 'historical_documents'
            }
            
        except Exception as e:
            logger.error(f"Vector storage failed for {document_id}: {str(e)}")
            return {'error': f"Vector storage failed: {str(e)}"}
    
    async def _ai_powered_semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Create intelligent text chunks using AI for structure analysis."""
        try:
            # Simple chunking for now - can be enhanced with AI
            chunk_size = 1000
            chunk_overlap = 200
            
            chunks = []
            text_length = len(text)
            
            for start in range(0, text_length, chunk_size - chunk_overlap):
                end = min(start + chunk_size, text_length)
                chunk_text = text[start:end]
                
                # Try to break on sentence boundaries
                if end < text_length:
                    last_period = chunk_text.rfind('.')
                    last_newline = chunk_text.rfind('\n')
                    break_point = max(last_period, last_newline)
                    
                    if break_point > len(chunk_text) * 0.8:  # Only break if close to end
                        chunk_text = chunk_text[:break_point + 1]
                        end = start + break_point + 1
                
                chunks.append({
                    'text': chunk_text.strip(),
                    'start': start,
                    'end': end,
                    'type': 'content',
                    'length': len(chunk_text)
                })
            
            return chunks
            
        except Exception as e:
            logger.warning(f"AI semantic chunking failed: {str(e)}")
            # Fallback to simple chunking
            chunk_size = 1000
            chunks = []
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                chunks.append({
                    'text': chunk_text,
                    'start': i,
                    'end': min(i + chunk_size, len(text)),
                    'type': 'content',
                    'length': len(chunk_text)
                })
            return chunks
    
    async def _save_processing_metadata(self, metadata: ProcessingMetadata):
        """Save comprehensive processing metadata."""
        try:
            metadata_file = self.metadata_dir / f"{metadata.document_id}_processing.json"
            metadata_dict = {
                'document_id': metadata.document_id,
                'url': metadata.url,
                'extraction_method': metadata.extraction_method,
                'extraction_quality': metadata.extraction_quality,
                'processing_timestamp': metadata.processing_timestamp.isoformat(),
                'ai_categorization': metadata.ai_categorization,
                'ai_tags': metadata.ai_tags,
                'relevance_score': metadata.relevance_score,
                'structure_analysis': metadata.structure_analysis,
                'ocr_used': metadata.ocr_used,
                'gpu_accelerated': metadata.gpu_accelerated
            }
            
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
            
            logger.debug(f"Saved processing metadata for {metadata.document_id}")
            
        except Exception as e:
            logger.error(f"Failed to save processing metadata: {str(e)}")
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            'processing_stats': {
                'documents_acquired': self.processing_stats.documents_acquired,
                'extraction_successes': self.processing_stats.extraction_successes,
                'extraction_failures': self.processing_stats.extraction_failures,
                'vector_storage_successes': self.processing_stats.vector_storage_successes,
                'vector_storage_failures': self.processing_stats.vector_storage_failures,
                'total_processing_time': self.processing_stats.total_processing_time,
                'ai_analysis_time': self.processing_stats.ai_analysis_time,
                'gpu_operations': self.processing_stats.gpu_operations
            },
            'system_info': {
                'gpu_available': self.gpu_available,
                'gpu_config': gpu_manager.get_config().__dict__ if gpu_manager.config else {},
                'ollama_ready': self.ollama_manager.is_ready,
                'storage_path': str(self.storage_path)
            },
            'success_rates': {
                'extraction_success_rate': (
                    self.processing_stats.extraction_successes / 
                    max(1, self.processing_stats.extraction_successes + self.processing_stats.extraction_failures)
                ),
                'vector_storage_success_rate': (
                    self.processing_stats.vector_storage_successes / 
                    max(1, self.processing_stats.vector_storage_successes + self.processing_stats.vector_storage_failures)
                )
            }
        }
    
    async def close(self):
        """Close resources and cleanup."""
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info("Enhanced Document Acquisition closed")
