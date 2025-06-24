# Document Acquisition & Text Extraction Research

## Overview
This document summarizes research into best practices for document acquisition, PDF downloading, text extraction, and metadata management for historical research systems. The focus is on implementing robust, production-ready capabilities for PyGent Factory's academic research pipeline.

## 1. Web Scraping & Document Download

### HTTP Client Best Practices (Requests Library)

#### Core Capabilities
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Session with connection pooling and retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.1,
    status_forcelist=[502, 503, 504],
    allowed_methods={'GET', 'POST'}
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount('https://', adapter)
session.mount('http://', adapter)
```

#### Robust Document Download
```python
def download_document(url, timeout=30, chunk_size=8192):
    """Download document with proper error handling and streaming."""
    try:
        headers = {
            'User-Agent': 'Academic Research Bot 1.0 (Historical Document Collection)'
        }
        
        response = session.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Stream download for large files
        content = b''
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                content += chunk
                
        return {
            'content': content,
            'headers': dict(response.headers),
            'url': response.url,
            'status_code': response.status_code,
            'encoding': response.encoding
        }
    except requests.exceptions.RequestException as e:
        return {'error': str(e), 'url': url}
```

#### Key Features for Academic Use
- **Session Management**: Persistent connections with connection pooling
- **Retry Logic**: Automatic retry with exponential backoff for temporary failures
- **Streaming Downloads**: Memory-efficient handling of large PDF files
- **Header Management**: Proper User-Agent strings for academic/research use
- **Timeout Handling**: Prevent hanging on slow connections
- **Error Recovery**: Graceful handling of network failures

### Implementation Classes for PyGent Factory
```python
class DocumentDownloader:
    """Robust document downloader with retry and validation."""
    
    def __init__(self, max_retries=3, timeout=30):
        self.session = self._create_session(max_retries)
        self.timeout = timeout
        self.downloaded_files = {}
        
    def _create_session(self, max_retries):
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={'GET', 'HEAD'}
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        return session
        
    def download_pdf(self, url, save_path=None):
        """Download PDF with validation and metadata extraction."""
        # Implementation details...
        pass
        
    def validate_pdf(self, content):
        """Validate downloaded content is a valid PDF."""
        return content.startswith(b'%PDF-')
        
    def extract_metadata(self, headers, url):
        """Extract download metadata for provenance tracking."""
        return {
            'url': url,
            'download_time': datetime.utcnow().isoformat(),
            'content_type': headers.get('content-type'),
            'content_length': headers.get('content-length'),
            'last_modified': headers.get('last-modified'),
            'server': headers.get('server')
        }
```

## 2. PDF Text Extraction with PyMuPDF

### Core Text Extraction Capabilities

#### Basic Text Extraction Methods
```python
import pymupdf

def extract_text_comprehensive(pdf_path):
    """Extract text using multiple methods for maximum coverage."""
    doc = pymupdf.open(pdf_path)
    results = {}
    
    for page_num, page in enumerate(doc):
        page_data = {
            'plain_text': page.get_text('text'),
            'blocks': page.get_text('blocks'),
            'dict': page.get_text('dict'),
            'html': page.get_text('html'),
            'xml': page.get_text('xml')
        }
        results[page_num] = page_data
        
    doc.close()
    return results
```

#### OCR Integration for Scanned Documents
```python
def extract_with_ocr(pdf_path, language='eng', dpi=300):
    """Extract text with OCR for scanned documents."""
    doc = pymupdf.open(pdf_path)
    results = {}
    
    for page_num, page in enumerate(doc):
        # Try normal text extraction first
        text = page.get_text('text').strip()
        
        if not text or len(text) < 50:  # Likely scanned page
            try:
                # Use OCR for better results
                textpage_ocr = page.get_textpage_ocr(
                    language=language,
                    dpi=dpi,
                    full=True  # OCR entire page
                )
                text = page.get_text(textpage=textpage_ocr)
            except Exception as e:
                print(f"OCR failed for page {page_num}: {e}")
                text = page.get_text('text')  # Fallback
                
        results[page_num] = {
            'text': text,
            'method': 'ocr' if not page.get_text('text').strip() else 'standard'
        }
        
    doc.close()
    return results
```

#### Advanced Text Processing
```python
def extract_structured_text(pdf_path):
    """Extract text with structural information."""
    doc = pymupdf.open(pdf_path)
    structured_data = []
    
    for page in doc:
        # Extract with detailed structure
        text_dict = page.get_text('dict', flags=pymupdf.TEXTFLAGS_DICT)
        
        for block in text_dict['blocks']:
            if block['type'] == 0:  # Text block
                for line in block.get('lines', []):
                    for span in line['spans']:
                        structured_data.append({
                            'text': span['text'],
                            'font': span['font'],
                            'size': span['size'],
                            'flags': span['flags'],
                            'bbox': span['bbox'],
                            'page': page.number
                        })
                        
    doc.close()
    return structured_data
```

### Performance Optimization

#### Memory Management
```python
def process_large_pdf(pdf_path, chunk_size=10):
    """Process large PDFs in chunks to manage memory."""
    doc = pymupdf.open(pdf_path)
    total_pages = doc.page_count
    
    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        
        # Process chunk
        chunk_text = []
        for page_num in range(start, end):
            page = doc[page_num]
            text = page.get_text()
            chunk_text.append(text)
            
        # Yield chunk for processing
        yield {
            'pages': range(start, end),
            'text': chunk_text
        }
        
    doc.close()
```

#### Text Quality Assessment
```python
def assess_text_quality(text):
    """Assess extracted text quality for OCR decision."""
    if not text:
        return {'quality': 'empty', 'score': 0}
        
    # Calculate quality metrics
    char_count = len(text)
    word_count = len(text.split())
    alpha_ratio = sum(c.isalpha() for c in text) / char_count if char_count > 0 else 0
    
    # Check for common OCR artifacts
    artifacts = ['□', '■', '○', '●', '?', '…']
    artifact_ratio = sum(text.count(a) for a in artifacts) / char_count if char_count > 0 else 0
    
    score = alpha_ratio - artifact_ratio
    
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
```

## 3. Document Storage & Metadata Management

### Document Storage Schema
```python
class DocumentStore:
    """Manage document storage with metadata."""
    
    def __init__(self, storage_path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
    def store_document(self, url, content, metadata):
        """Store document with full metadata."""
        doc_id = self._generate_doc_id(url)
        doc_path = self.storage_path / f"{doc_id}.pdf"
        meta_path = self.storage_path / f"{doc_id}_meta.json"
        
        # Store document
        with open(doc_path, 'wb') as f:
            f.write(content)
            
        # Store metadata
        full_metadata = {
            'document_id': doc_id,
            'source_url': url,
            'file_path': str(doc_path),
            'file_size': len(content),
            'stored_at': datetime.utcnow().isoformat(),
            'checksum': hashlib.sha256(content).hexdigest(),
            **metadata
        }
        
        with open(meta_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
            
        return doc_id, doc_path
        
    def _generate_doc_id(self, url):
        """Generate stable document ID from URL."""
        return hashlib.md5(url.encode()).hexdigest()
```

### Text Extraction Integration
```python
class TextExtractionPipeline:
    """Complete text extraction pipeline."""
    
    def __init__(self, doc_store, enable_ocr=True):
        self.doc_store = doc_store
        self.enable_ocr = enable_ocr
        
    def process_document(self, doc_id):
        """Process document through complete extraction pipeline."""
        # Load document metadata
        metadata = self.doc_store.get_metadata(doc_id)
        pdf_path = metadata['file_path']
        
        # Extract text with quality assessment
        extraction_results = self._extract_text_smart(pdf_path)
        
        # Store extracted text
        text_path = pdf_path.replace('.pdf', '_text.json')
        with open(text_path, 'w') as f:
            json.dump(extraction_results, f, indent=2)
            
        return extraction_results
        
    def _extract_text_smart(self, pdf_path):
        """Smart text extraction with OCR fallback."""
        # Try standard extraction first
        standard_text = self._extract_standard(pdf_path)
        
        # Assess quality and decide on OCR
        needs_ocr = []
        for page_num, page_data in standard_text.items():
            quality = assess_text_quality(page_data['text'])
            if quality['quality'] == 'low' and self.enable_ocr:
                needs_ocr.append(page_num)
                
        # Apply OCR to problematic pages
        if needs_ocr:
            ocr_results = self._extract_ocr_pages(pdf_path, needs_ocr)
            for page_num in needs_ocr:
                standard_text[page_num] = ocr_results[page_num]
                
        return standard_text
```

## 4. Integration with PyGent Factory

### Proposed Architecture
```python
class HistoricalDocumentProcessor:
    """Complete document processing for historical research."""
    
    def __init__(self, config):
        self.downloader = DocumentDownloader(
            max_retries=config.get('max_retries', 3),
            timeout=config.get('timeout', 30)
        )
        self.doc_store = DocumentStore(config['storage_path'])
        self.text_pipeline = TextExtractionPipeline(
            self.doc_store,
            enable_ocr=config.get('enable_ocr', True)
        )
        
    def acquire_document(self, url, source_info=None):
        """Complete document acquisition workflow."""
        try:
            # Download document
            download_result = self.downloader.download_pdf(url)
            if 'error' in download_result:
                return {'error': download_result['error']}
                
            # Validate PDF
            if not self.downloader.validate_pdf(download_result['content']):
                return {'error': 'Invalid PDF format'}
                
            # Store document with metadata
            metadata = {
                'source_info': source_info or {},
                'download_metadata': self.downloader.extract_metadata(
                    download_result['headers'], 
                    url
                )
            }
            
            doc_id, doc_path = self.doc_store.store_document(
                url, 
                download_result['content'], 
                metadata
            )
            
            # Extract text
            extraction_results = self.text_pipeline.process_document(doc_id)
            
            return {
                'success': True,
                'document_id': doc_id,
                'document_path': str(doc_path),
                'extraction_results': extraction_results
            }
            
        except Exception as e:
            return {'error': f'Document acquisition failed: {str(e)}'}
```

## 5. Best Practices Summary

### Download Best Practices
1. **Use Session objects** for connection pooling and performance
2. **Implement retry logic** with exponential backoff
3. **Stream large files** to manage memory usage
4. **Validate downloaded content** before processing
5. **Extract comprehensive metadata** for provenance tracking
6. **Use appropriate User-Agent strings** for academic research

### Text Extraction Best Practices
1. **Try multiple extraction methods** (text, dict, html, xml)
2. **Assess text quality** before deciding on OCR
3. **Use OCR selectively** for poor-quality or scanned pages
4. **Process large documents in chunks** to manage memory
5. **Preserve document structure** when possible
6. **Store extraction metadata** for reproducibility

### Storage Best Practices
1. **Generate stable document IDs** from source URLs
2. **Store original documents** alongside extracted text
3. **Maintain comprehensive metadata** for each document
4. **Use checksums** for integrity verification
5. **Implement backup strategies** for important documents
6. **Track extraction methods** used for each document

## 6. Error Handling & Resilience

### Common Issues & Solutions
- **Network timeouts**: Implement retry with backoff
- **Large file handling**: Use streaming downloads
- **OCR failures**: Graceful fallback to standard extraction
- **Invalid PDFs**: Validate before processing
- **Memory issues**: Process documents in chunks
- **Encoding problems**: Proper character encoding handling

### Monitoring & Logging
```python
import logging

class DocumentProcessingMonitor:
    """Monitor document processing pipeline."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.stats = {
            'documents_processed': 0,
            'downloads_failed': 0,
            'extractions_failed': 0,
            'ocr_used': 0
        }
        
    def log_download(self, url, success, error=None):
        if success:
            self.logger.info(f"Downloaded: {url}")
            self.stats['documents_processed'] += 1
        else:
            self.logger.error(f"Download failed: {url} - {error}")
            self.stats['downloads_failed'] += 1
            
    def log_extraction(self, doc_id, method, success, error=None):
        if success:
            self.logger.info(f"Extracted text from {doc_id} using {method}")
            if method == 'ocr':
                self.stats['ocr_used'] += 1
        else:
            self.logger.error(f"Extraction failed: {doc_id} - {error}")
            self.stats['extractions_failed'] += 1
```

## 7. Next Steps for Implementation

1. **Implement DocumentDownloader class** with retry and validation
2. **Create TextExtractionPipeline** with OCR integration
3. **Build DocumentStore** with metadata management
4. **Integrate with existing PyGent Factory architecture**
5. **Add comprehensive error handling and logging**
6. **Create unit tests** for all components
7. **Performance testing** with large document sets
8. **Documentation** for API usage and configuration

This research provides the foundation for implementing robust document acquisition and text extraction capabilities in PyGent Factory's historical research pipeline.
