"""
Advanced text extraction using PyMuPDF with multiple extraction methods and quality assessment.
"""
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
import json
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    import fitz

logger = logging.getLogger(__name__)

@dataclass
class TextExtractionResult:
    """Result of text extraction operation."""
    success: bool
    document_id: str
    text_content: Optional[str] = None
    page_count: Optional[int] = None
    extraction_method: str = "unknown"
    quality_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None

class PyMuPDFTextExtractor:
    """Advanced text extraction using PyMuPDF with quality assessment."""
    
    def __init__(self, storage_system):
        self.storage_system = storage_system
        
    def extract_text_multiple_methods(self, pdf_path: str, document_id: str) -> TextExtractionResult:
        """Extract text using multiple methods and select the best result."""
        try:
            doc = fitz.open(pdf_path)
            
            # Try different extraction methods
            methods = {
                'text': self._extract_plain_text,
                'dict': self._extract_dict_method,
                'html': self._extract_html_method,
                'xml': self._extract_xml_method
            }
            
            results = {}
            for method_name, method_func in methods.items():
                try:
                    result = method_func(doc)
                    quality = self._assess_text_quality(result)
                    results[method_name] = {
                        'text': result,
                        'quality': quality,
                        'word_count': len(result.split()) if result else 0,
                        'char_count': len(result) if result else 0
                    }
                except Exception as e:
                    logger.warning(f"Extraction method {method_name} failed: {e}")
                    results[method_name] = {
                        'text': "",
                        'quality': 0.0,
                        'word_count': 0,
                        'char_count': 0
                    }
            
            # Select best method based on quality score
            best_method = max(results.keys(), key=lambda k: results[k]['quality'])
            best_result = results[best_method]
            
            # Create structured text data
            structured_data = self._create_structured_data(doc, best_result['text'])
            
            # Save extracted text
            self._save_extracted_text(document_id, best_result['text'], structured_data)
            
            doc.close()
            
            return TextExtractionResult(
                success=True,
                document_id=document_id,
                text_content=best_result['text'],
                page_count=len(doc),
                extraction_method=best_method,
                quality_score=best_result['quality'],
                metadata={
                    'all_methods_results': results,
                    'structured_data': structured_data
                },
                word_count=best_result['word_count'],
                char_count=best_result['char_count']
            )
            
        except Exception as e:
            logger.error(f"Text extraction failed for {document_id}: {e}")
            return TextExtractionResult(
                success=False,
                document_id=document_id,
                error_message=str(e)
            )
    
    def _extract_plain_text(self, doc) -> str:
        """Extract plain text from all pages."""
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(f"=== Page {page_num + 1} ===\n{text}")
        return "\n\n".join(text_parts)
    
    def _extract_dict_method(self, doc) -> str:
        """Extract text using dictionary method for better structure preservation."""
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            page_text = []
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            page_text.append(line_text)
            
            if page_text:
                text_parts.append(f"=== Page {page_num + 1} ===\n" + "\n".join(page_text))
        
        return "\n\n".join(text_parts)
    
    def _extract_html_method(self, doc) -> str:
        """Extract text using HTML method and strip HTML tags."""
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            html_text = page.get_text("html")
            
            # Strip HTML tags but preserve structure
            clean_text = self._strip_html_tags(html_text)
            if clean_text.strip():
                text_parts.append(f"=== Page {page_num + 1} ===\n{clean_text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_xml_method(self, doc) -> str:
        """Extract text using XML method and parse structure."""
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            xml_text = page.get_text("xml")
            
            # Parse XML and extract text content
            clean_text = self._parse_xml_text(xml_text)
            if clean_text.strip():
                text_parts.append(f"=== Page {page_num + 1} ===\n{clean_text}")
        
        return "\n\n".join(text_parts)
    
    def _strip_html_tags(self, html_text: str) -> str:
        """Strip HTML tags while preserving text structure."""
        # Remove HTML tags but keep line breaks
        import re
        
        # Replace block elements with newlines
        html_text = re.sub(r'<(p|div|br)[^>]*>', '\n', html_text, flags=re.IGNORECASE)
        html_text = re.sub(r'</(p|div)[^>]*>', '\n', html_text, flags=re.IGNORECASE)
        
        # Remove all other HTML tags
        html_text = re.sub(r'<[^>]+>', '', html_text)
        
        # Clean up whitespace
        html_text = re.sub(r'\n\s*\n', '\n\n', html_text)
        html_text = re.sub(r'[ \t]+', ' ', html_text)
        
        return html_text.strip()
    
    def _parse_xml_text(self, xml_text: str) -> str:
        """Parse XML and extract readable text."""
        import xml.etree.ElementTree as ET
        
        try:
            # Simple XML text extraction
            root = ET.fromstring(xml_text)
            text_parts = []
            
            def extract_text_from_element(element):
                if element.text:
                    text_parts.append(element.text)
                for child in element:
                    extract_text_from_element(child)
                if element.tail:
                    text_parts.append(element.tail)
            
            extract_text_from_element(root)
            return " ".join(text_parts)
            
        except Exception:
            # Fallback: simple regex extraction
            import re
            text = re.sub(r'<[^>]+>', ' ', xml_text)
            return re.sub(r'\s+', ' ', text).strip()
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of extracted text."""
        if not text or not text.strip():
            return 0.0
        
        score = 0.0
        
        # Length factor (longer text generally better, up to a point)
        length_score = min(len(text) / 10000, 1.0) * 0.3
        score += length_score
        
        # Word density (words per character)
        words = text.split()
        if len(text) > 0:
            word_density = len(words) / len(text)
            word_density_score = min(word_density * 10, 1.0) * 0.2
            score += word_density_score
        
        # Character variety (more diverse characters = better)
        unique_chars = len(set(text.lower()))
        char_variety_score = min(unique_chars / 50, 1.0) * 0.2
        score += char_variety_score
        
        # Sentence structure (presence of punctuation)
        sentence_chars = sum(1 for c in text if c in '.!?;:')
        sentence_score = min(sentence_chars / len(words) if words else 0, 0.1) * 10 * 0.1
        score += sentence_score
        
        # Readable character ratio (alphanumeric + common punctuation)
        readable_chars = sum(1 for c in text if c.isalnum() or c in ' .,!?;:-"\'()\n\t')
        readable_ratio = readable_chars / len(text) if text else 0
        score += readable_ratio * 0.2
        
        return min(score, 1.0)
    
    def _create_structured_data(self, doc, text: str) -> Dict[str, Any]:
        """Create structured data from document with page-level information."""
        try:
            structured_data = {
                'total_pages': len(doc),
                'pages': [],
                'metadata': {
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', ''),
                    'creator': doc.metadata.get('creator', ''),
                    'producer': doc.metadata.get('producer', ''),
                    'creation_date': doc.metadata.get('creationDate', ''),
                    'modification_date': doc.metadata.get('modDate', '')
                }
            }
            
            # Extract page-level information
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                page_info = {
                    'page_number': page_num + 1,
                    'text_length': len(page_text),
                    'word_count': len(page_text.split()),
                    'has_images': len(page.get_images()) > 0,
                    'image_count': len(page.get_images()),
                    'has_links': len(page.get_links()) > 0,
                    'link_count': len(page.get_links())
                }
                
                structured_data['pages'].append(page_info)
            
            return structured_data
            
        except Exception as e:
            logger.warning(f"Failed to create structured data: {e}")
            return {'error': str(e)}
    
    def _save_extracted_text(self, document_id: str, text: str, structured_data: Dict[str, Any]):
        """Save extracted text and structured data."""
        text_dir = self.storage_system.directories['text']
        
        # Save plain text
        text_file = text_dir / f"{document_id}.txt"
        try:
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Failed to save text for {document_id}: {e}")
        
        # Save structured data
        structured_file = text_dir / f"{document_id}_structured.json"
        try:
            with open(structured_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save structured data for {document_id}: {e}")
    
    def get_extracted_text(self, document_id: str) -> Optional[str]:
        """Retrieve previously extracted text."""
        text_file = self.storage_system.directories['text'] / f"{document_id}.txt"
        
        if not text_file.exists():
            return None
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read extracted text for {document_id}: {e}")
            return None
    
    def get_structured_data(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve structured data for document."""
        structured_file = self.storage_system.directories['text'] / f"{document_id}_structured.json"
        
        if not structured_file.exists():
            return None
        
        try:
            with open(structured_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read structured data for {document_id}: {e}")
            return None
