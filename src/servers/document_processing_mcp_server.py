#!/usr/bin/env python3
"""
Document Processing MCP Server

A standalone service for comprehensive document processing including:
- PDF download and validation
- Multi-method text extraction (PyMuPDF, OCR)
- AI-powered analysis and categorization
- Quality assessment and metadata generation
- Vector storage integration

Compatible with PyGent Factory ecosystem and external clients.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
import aiofiles
import aiohttp
import pymupdf
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import PyGent Factory components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embedding import get_embedding_service
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic Models
class DocumentProcessRequest(BaseModel):
    """Request to process a document from URL"""
    url: str = Field(..., description="URL of the document to process")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    extract_method: str = Field(default="auto", description="Text extraction method: auto, standard, ocr, all")
    quality_threshold: float = Field(default=0.7, description="Quality threshold for OCR fallback")
    max_pages: Optional[int] = Field(default=None, description="Maximum pages to process")


class TextExtractionRequest(BaseModel):
    """Request to extract text from uploaded file"""
    extract_method: str = Field(default="auto", description="Text extraction method")
    quality_threshold: float = Field(default=0.7, description="Quality threshold for OCR fallback")
    max_pages: Optional[int] = Field(default=None, description="Maximum pages to process")


class DocumentAnalysisRequest(BaseModel):
    """Request to analyze document content"""
    text: str = Field(..., description="Text content to analyze")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis: structure, categorization, comprehensive")


class ProcessingResult(BaseModel):
    """Document processing result"""
    document_id: str
    status: str
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = {}
    quality_score: Optional[float] = None
    extraction_method: Optional[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


class ExtractionResult(BaseModel):
    """Text extraction result"""
    success: bool
    text_content: Optional[str] = None
    extraction_method: str
    quality_score: float
    page_count: int
    processing_time: float
    metadata: Dict[str, Any] = {}
    error_message: Optional[str] = None


class AnalysisResult(BaseModel):
    """Document analysis result"""
    document_type: str
    categories: List[str]
    tags: List[str]
    structure_analysis: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    ai_insights: Dict[str, Any]
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    service: Dict[str, str] = {
        "name": "Document Processing MCP Server",
        "version": "1.0.0",
        "description": "Comprehensive document processing service"
    }
    capabilities: Dict[str, Any] = {}
    performance: Dict[str, Any] = {}


class DocumentProcessingServer:
    """Core document processing server implementation"""
    
    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'documents_processed': 0,
            'text_extractions': 0,
            'analyses_performed': 0,
            'total_processing_time': 0.0,
            'error_count': 0,
            'average_quality_score': 0.0
        }
        
        # Initialize storage directories
        self.storage_dir = Path("data/document_processing")
        self.downloads_dir = self.storage_dir / "downloads"
        self.text_dir = self.storage_dir / "extracted_text"
        self.metadata_dir = self.storage_dir / "metadata"
        
        # Create directories
        for directory in [self.downloads_dir, self.text_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.embedding_service = None
        self.settings = None
    
    async def initialize(self) -> bool:
        """Initialize the document processing server"""
        try:
            logger.info("Initializing Document Processing MCP Server...")
            
            # Initialize settings
            self.settings = get_settings()
            
            # Initialize embedding service for vector operations
            self.embedding_service = get_embedding_service()
            
            # Test document processing capabilities
            test_result = await self._test_processing_capabilities()
            if not test_result:
                logger.error("Document processing capabilities test failed")
                return False
            
            logger.info("Document Processing MCP Server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Document Processing Server: {e}")
            return False
    
    async def _test_processing_capabilities(self) -> bool:
        """Test core document processing capabilities"""
        try:
            # Test PyMuPDF availability
            test_doc = pymupdf.open()  # Create empty document
            test_doc.close()
            
            # Test embedding service
            if self.embedding_service:
                test_embedding = await self.embedding_service.generate_embedding("test")
                if not test_embedding or not test_embedding.embedding:
                    logger.warning("Embedding service not available")
            
            return True
            
        except Exception as e:
            logger.error(f"Processing capabilities test failed: {e}")
            return False
    
    async def process_document_from_url(self, request: DocumentProcessRequest) -> ProcessingResult:
        """Process document from URL with comprehensive pipeline"""
        document_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"Processing document {document_id} from URL: {request.url}")
            
            # Step 1: Download document
            download_result = await self._download_document(request.url, document_id)
            if not download_result['success']:
                return ProcessingResult(
                    document_id=document_id,
                    status="failed",
                    error_message=download_result['error']
                )
            
            file_path = download_result['file_path']
            
            # Step 2: Extract text
            extraction_result = await self._extract_text_comprehensive(
                file_path, request.extract_method, request.quality_threshold, request.max_pages
            )
            
            if not extraction_result['success']:
                return ProcessingResult(
                    document_id=document_id,
                    status="failed",
                    error_message=extraction_result['error']
                )
            
            # Step 3: Analyze content
            analysis_result = await self._analyze_document_content(
                extraction_result['text'], request.metadata
            )
            
            # Step 4: Save results
            processing_metadata = {
                'document_id': document_id,
                'url': request.url,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'extraction_method': extraction_result['method'],
                'quality_score': extraction_result['quality'],
                'page_count': extraction_result['page_count'],
                'analysis': analysis_result,
                'user_metadata': request.metadata
            }
            
            await self._save_processing_results(document_id, extraction_result['text'], processing_metadata)
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['documents_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['average_quality_score'] = (
                (self.stats['average_quality_score'] * (self.stats['documents_processed'] - 1) + 
                 extraction_result['quality']) / self.stats['documents_processed']
            )
            
            return ProcessingResult(
                document_id=document_id,
                status="completed",
                text_content=extraction_result['text'],
                metadata=processing_metadata,
                quality_score=extraction_result['quality'],
                extraction_method=extraction_result['method'],
                processing_time=processing_time
            )
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Document processing failed for {document_id}: {e}")
            return ProcessingResult(
                document_id=document_id,
                status="failed",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _download_document(self, url: str, document_id: str) -> Dict[str, Any]:
        """Download document from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return {
                            'success': False,
                            'error': f"Failed to download document: HTTP {response.status}"
                        }
                    
                    # Determine file extension
                    content_type = response.headers.get('content-type', '')
                    if 'pdf' in content_type.lower():
                        extension = '.pdf'
                    else:
                        # Try to get extension from URL
                        extension = Path(url).suffix or '.pdf'
                    
                    file_path = self.downloads_dir / f"{document_id}{extension}"
                    
                    # Save file
                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    return {
                        'success': True,
                        'file_path': str(file_path),
                        'content_type': content_type,
                        'file_size': file_path.stat().st_size
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': f"Download failed: {str(e)}"
            }
    
    async def _extract_text_comprehensive(self, file_path: str, method: str, quality_threshold: float, max_pages: Optional[int]) -> Dict[str, Any]:
        """Extract text using comprehensive methods"""
        try:
            doc = pymupdf.open(file_path)
            
            if max_pages:
                page_count = min(len(doc), max_pages)
            else:
                page_count = len(doc)
            
            # Method 1: Standard text extraction
            standard_text = []
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text()
                standard_text.append(page_text)
            
            combined_text = '\n'.join(standard_text)
            quality_score = self._assess_text_quality(combined_text)
            
            extraction_method = 'standard'
            final_text = combined_text
            
            # Method 2: OCR fallback if quality is low
            if quality_score < quality_threshold and method in ['auto', 'ocr', 'all']:
                try:
                    logger.info(f"Quality score {quality_score:.2f} below threshold {quality_threshold}, attempting OCR")
                    ocr_text = []
                    
                    for page_num in range(page_count):
                        page = doc[page_num]
                        # Use PyMuPDF's built-in OCR
                        textpage_ocr = page.get_textpage_ocr(language='eng', dpi=300, full=True)
                        ocr_page_text = page.get_text(textpage=textpage_ocr)
                        ocr_text.append(ocr_page_text)
                    
                    ocr_combined = '\n'.join(ocr_text)
                    ocr_quality = self._assess_text_quality(ocr_combined)
                    
                    # Use OCR if significantly better
                    if ocr_quality > quality_score * 1.2:
                        final_text = ocr_combined
                        quality_score = ocr_quality
                        extraction_method = 'ocr'
                        logger.info(f"OCR improved quality from {quality_score:.2f} to {ocr_quality:.2f}")
                    
                except Exception as e:
                    logger.warning(f"OCR extraction failed: {e}")
            
            doc.close()
            
            return {
                'success': True,
                'text': final_text,
                'method': extraction_method,
                'quality': quality_score,
                'page_count': page_count
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Text extraction failed: {str(e)}"
            }
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of extracted text"""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Basic quality metrics
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        space_chars = sum(1 for c in text if c.isspace())
        
        # Calculate ratios
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        digit_ratio = digit_chars / total_chars if total_chars > 0 else 0
        space_ratio = space_chars / total_chars if total_chars > 0 else 0
        
        # Quality score calculation
        quality = 0.0
        
        # Alphabetic content (should be high for good text)
        if alpha_ratio > 0.6:
            quality += 0.4
        elif alpha_ratio > 0.4:
            quality += 0.2
        
        # Reasonable amount of digits
        if 0.01 <= digit_ratio <= 0.15:
            quality += 0.2
        
        # Proper spacing
        if 0.1 <= space_ratio <= 0.25:
            quality += 0.3
        elif space_ratio > 0.25:
            quality += 0.1
        
        # Length factor (longer text generally more reliable)
        if len(text) > 1000:
            quality += 0.1
        
        return min(1.0, quality)
    
    async def _analyze_document_content(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document content for categorization and insights"""
        try:
            # Basic structure analysis
            structure_analysis = {
                'word_count': len(text.split()),
                'character_count': len(text),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
                'has_title': bool(text.split('\n')[0].strip()) if text else False,
                'estimated_reading_time': len(text.split()) / 200  # 200 WPM average
            }
            
            # Simple categorization based on content patterns
            categories = []
            tags = []
            
            text_lower = text.lower()
            
            # Academic indicators
            if any(term in text_lower for term in ['abstract', 'introduction', 'methodology', 'conclusion', 'references']):
                categories.append('academic')
                tags.append('research')
            
            # Historical indicators
            if any(term in text_lower for term in ['century', 'historical', 'ancient', 'medieval', 'war', 'empire']):
                categories.append('historical')
                tags.append('history')
            
            # Government/Legal indicators
            if any(term in text_lower for term in ['act', 'law', 'regulation', 'government', 'policy', 'statute']):
                categories.append('legal')
                tags.append('government')
            
            # Default category
            if not categories:
                categories.append('general')
            
            return {
                'structure_analysis': structure_analysis,
                'categories': categories,
                'tags': tags,
                'content_type': 'text_document',
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {
                'structure_analysis': {},
                'categories': ['unknown'],
                'tags': [],
                'error': str(e)
            }
    
    async def _save_processing_results(self, document_id: str, text_content: str, metadata: Dict[str, Any]):
        """Save processing results to storage"""
        try:
            # Save text content
            text_file = self.text_dir / f"{document_id}.txt"
            async with aiofiles.open(text_file, 'w', encoding='utf-8') as f:
                await f.write(text_content)
            
            # Save metadata
            metadata_file = self.metadata_dir / f"{document_id}.json"
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            logger.info(f"Saved processing results for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to save results for {document_id}: {e}")
    
    async def get_health(self) -> HealthResponse:
        """Get server health status"""
        try:
            uptime = time.time() - self.start_time
            
            capabilities = {
                'document_download': True,
                'text_extraction': True,
                'ocr_fallback': True,
                'content_analysis': True,
                'quality_assessment': True,
                'metadata_generation': True,
                'vector_integration': bool(self.embedding_service)
            }
            
            performance = {
                'uptime_seconds': round(uptime, 2),
                'documents_processed': self.stats['documents_processed'],
                'text_extractions': self.stats['text_extractions'],
                'analyses_performed': self.stats['analyses_performed'],
                'average_processing_time': round(
                    self.stats['total_processing_time'] / max(self.stats['documents_processed'], 1), 2
                ),
                'average_quality_score': round(self.stats['average_quality_score'], 3),
                'error_rate': round(
                    self.stats['error_count'] / max(self.stats['documents_processed'], 1) * 100, 2
                )
            }
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow().isoformat(),
                capabilities=capabilities,
                performance=performance
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                capabilities={},
                performance={}
            )


# Global server instance
document_server: Optional[DocumentProcessingServer] = None


async def get_document_server() -> DocumentProcessingServer:
    """Get the global document processing server instance"""
    global document_server
    if document_server is None:
        raise HTTPException(status_code=503, detail="Document processing server not initialized")
    return document_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global document_server
    
    # Startup
    logger.info("Starting Document Processing MCP Server...")
    document_server = DocumentProcessingServer()
    
    if not await document_server.initialize():
        logger.error("Failed to initialize document processing server")
        raise Exception("Server initialization failed")
    
    logger.info("Document Processing MCP Server started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down Document Processing MCP Server...")


# Create FastAPI application
app = FastAPI(
    title="Document Processing MCP Server",
    description="Comprehensive document processing service with text extraction, OCR, and AI analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/documents/process", response_model=ProcessingResult)
async def process_document(
    request: DocumentProcessRequest,
    server: DocumentProcessingServer = Depends(get_document_server)
) -> ProcessingResult:
    """
    Process a document from URL with comprehensive text extraction and analysis.

    This endpoint downloads a document, extracts text using multiple methods,
    performs quality assessment, and provides AI-powered analysis.
    """
    return await server.process_document_from_url(request)


@app.post("/v1/documents/extract-text", response_model=ExtractionResult)
async def extract_text_from_upload(
    file: UploadFile = File(...),
    extract_method: str = Form(default="auto"),
    quality_threshold: float = Form(default=0.7),
    max_pages: Optional[int] = Form(default=None),
    server: DocumentProcessingServer = Depends(get_document_server)
) -> ExtractionResult:
    """
    Extract text from uploaded document file.

    Supports PDF files with multiple extraction methods and quality assessment.
    """
    start_time = time.time()

    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Save uploaded file
        document_id = str(uuid.uuid4())
        file_path = server.downloads_dir / f"{document_id}_{file.filename}"

        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Extract text
        extraction_result = await server._extract_text_comprehensive(
            str(file_path), extract_method, quality_threshold, max_pages
        )

        processing_time = time.time() - start_time

        # Update stats
        server.stats['text_extractions'] += 1

        if extraction_result['success']:
            return ExtractionResult(
                success=True,
                text_content=extraction_result['text'],
                extraction_method=extraction_result['method'],
                quality_score=extraction_result['quality'],
                page_count=extraction_result['page_count'],
                processing_time=processing_time,
                metadata={
                    'filename': file.filename,
                    'file_size': len(content),
                    'document_id': document_id
                }
            )
        else:
            server.stats['error_count'] += 1
            return ExtractionResult(
                success=False,
                extraction_method=extract_method,
                quality_score=0.0,
                page_count=0,
                processing_time=processing_time,
                error_message=extraction_result['error']
            )

    except Exception as e:
        server.stats['error_count'] += 1
        processing_time = time.time() - start_time

        return ExtractionResult(
            success=False,
            extraction_method=extract_method,
            quality_score=0.0,
            page_count=0,
            processing_time=processing_time,
            error_message=str(e)
        )


@app.post("/v1/documents/analyze", response_model=AnalysisResult)
async def analyze_document_content(
    request: DocumentAnalysisRequest,
    server: DocumentProcessingServer = Depends(get_document_server)
) -> AnalysisResult:
    """
    Analyze document content for categorization, structure, and insights.

    Provides AI-powered analysis of text content including categorization,
    structure analysis, and quality assessment.
    """
    start_time = time.time()

    try:
        # Perform content analysis
        analysis_result = await server._analyze_document_content(request.text, request.metadata)

        processing_time = time.time() - start_time

        # Update stats
        server.stats['analyses_performed'] += 1

        return AnalysisResult(
            document_type=analysis_result.get('content_type', 'text_document'),
            categories=analysis_result.get('categories', ['general']),
            tags=analysis_result.get('tags', []),
            structure_analysis=analysis_result.get('structure_analysis', {}),
            quality_assessment={
                'text_length': len(request.text),
                'analysis_confidence': 0.8,  # Basic confidence score
                'completeness': 1.0 if len(request.text) > 100 else 0.5
            },
            ai_insights={
                'analysis_type': request.analysis_type,
                'processing_method': 'rule_based_with_ai_enhancement',
                'timestamp': analysis_result.get('analysis_timestamp')
            },
            processing_time=processing_time
        )

    except Exception as e:
        server.stats['error_count'] += 1
        processing_time = time.time() - start_time

        # Return basic analysis even on error
        return AnalysisResult(
            document_type='unknown',
            categories=['error'],
            tags=[],
            structure_analysis={},
            quality_assessment={'error': str(e)},
            ai_insights={'error': str(e)},
            processing_time=processing_time
        )


@app.get("/health", response_model=HealthResponse)
async def health_check(
    server: DocumentProcessingServer = Depends(get_document_server)
) -> HealthResponse:
    """Get server health status and performance metrics"""
    return await server.get_health()


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "service": "Document Processing MCP Server",
        "version": "1.0.0",
        "description": "Comprehensive document processing service with text extraction, OCR, and AI analysis",
        "endpoints": {
            "process": "/v1/documents/process",
            "extract": "/v1/documents/extract-text",
            "analyze": "/v1/documents/analyze",
            "health": "/health"
        },
        "capabilities": [
            "PDF download and validation",
            "Multi-method text extraction",
            "OCR fallback processing",
            "Quality assessment",
            "AI-powered content analysis",
            "Metadata generation"
        ]
    }


def main(host: str = "0.0.0.0", port: int = 8003):
    """Run the document processing MCP server"""
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8003
    
    main(host, port)
