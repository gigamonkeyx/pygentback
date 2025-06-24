"""
OCR integration for fallback text extraction when standard methods fail.
"""
try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF for image extraction
    HAS_OCR = True
except ImportError:
    pytesseract = None
    Image = None
    fitz = None
    HAS_OCR = False

import io
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Result of OCR text extraction."""
    success: bool
    text_content: Optional[str] = None
    confidence_score: Optional[float] = None
    language_detected: Optional[str] = None
    page_count: Optional[int] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

class OCRTextExtractor:
    """OCR-based text extraction for low-quality or image-based PDFs."""
    
    def __init__(self, 
                 tesseract_path: Optional[str] = None,
                 default_language: str = 'eng'):
        self.default_language = default_language
        
        if not HAS_OCR:
            logger.warning("OCR dependencies not available. Install pytesseract and Pillow for OCR support.")
            return
        
        # Configure Tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Test OCR availability
        try:
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            logger.info("OCR system initialized successfully")
        except Exception as e:
            logger.error(f"OCR initialization failed: {e}")
            self.ocr_available = False
    
    def extract_text_with_ocr(self, pdf_path: str, document_id: str) -> OCRResult:
        """Extract text using OCR on PDF pages."""
        if not HAS_OCR or not self.ocr_available:
            return OCRResult(
                success=False,
                error_message="OCR not available"
            )
        
        import time
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            all_text = []
            total_confidence = 0
            processed_pages = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Process with OCR
                image = Image.open(io.BytesIO(img_data))
                
                # Extract text with confidence
                ocr_data = pytesseract.image_to_data(
                    image, 
                    lang=self.default_language,
                    output_type=pytesseract.Output.DICT
                )
                
                # Filter and combine text with confidence checking
                page_text = []
                page_confidences = []
                
                for i, text in enumerate(ocr_data['text']):
                    confidence = int(ocr_data['conf'][i])
                    if confidence > 30 and text.strip():  # Filter low-confidence text
                        page_text.append(text)
                        page_confidences.append(confidence)
                
                if page_text:
                    page_content = f"=== Page {page_num + 1} ===\n" + " ".join(page_text)
                    all_text.append(page_content)
                    
                    # Calculate average confidence for this page
                    if page_confidences:
                        total_confidence += sum(page_confidences) / len(page_confidences)
                        processed_pages += 1
            
            doc.close()
            
            # Calculate overall confidence
            overall_confidence = total_confidence / processed_pages if processed_pages > 0 else 0
            
            # Detect language
            detected_language = self._detect_language("\n\n".join(all_text))
            
            processing_time = time.time() - start_time
            
            result_text = "\n\n".join(all_text)
            
            return OCRResult(
                success=True,
                text_content=result_text,
                confidence_score=overall_confidence,
                language_detected=detected_language,
                page_count=len(doc),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {document_id}: {e}")
            return OCRResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time if 'start_time' in locals() else None
            )
    
    def compare_extraction_quality(self, 
                                 standard_text: str, 
                                 ocr_text: str) -> Dict[str, Any]:
        """Compare quality between standard extraction and OCR."""
        comparison = {
            'standard_length': len(standard_text),
            'ocr_length': len(ocr_text),
            'standard_words': len(standard_text.split()),
            'ocr_words': len(ocr_text.split()),
            'recommended_method': 'standard'
        }
        
        # Calculate basic quality metrics
        standard_density = len(standard_text.split()) / len(standard_text) if standard_text else 0
        ocr_density = len(ocr_text.split()) / len(ocr_text) if ocr_text else 0
        
        comparison['standard_word_density'] = standard_density
        comparison['ocr_word_density'] = ocr_density
        
        # Simple heuristic: prefer OCR if standard extraction is very poor
        if standard_density < 0.05 and ocr_density > 0.08:
            comparison['recommended_method'] = 'ocr'
            comparison['reason'] = 'Standard extraction appears to be low quality'
        elif len(standard_text.strip()) < 100 and len(ocr_text.strip()) > 500:
            comparison['recommended_method'] = 'ocr'
            comparison['reason'] = 'Standard extraction produced minimal text'
        else:
            comparison['reason'] = 'Standard extraction appears adequate'
        
        return comparison
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns."""
        if not text:
            return 'unknown'
        
        # Simple heuristic based on common English patterns
        english_indicators = [
            'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that'
        ]
        
        text_lower = text.lower()
        english_matches = sum(1 for word in english_indicators if word in text_lower)
        
        if english_matches >= 3:
            return 'english'
        else:
            return 'unknown'
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract all images from PDF for potential OCR processing."""
        if not HAS_OCR:
            return []
        
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    images.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'xref': xref,
                        'extension': image_ext,
                        'size': len(image_bytes),
                        'width': base_image.get('width'),
                        'height': base_image.get('height')
                    })
            
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {e}")
            return []
    
    def is_text_scannable(self, pdf_path: str) -> bool:
        """Determine if PDF likely contains scannable text (vs pure images)."""
        if not HAS_OCR:
            return False
        
        try:
            doc = fitz.open(pdf_path)
            
            # Sample first few pages
            sample_pages = min(3, len(doc))
            total_text_length = 0
            total_images = 0
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                text = page.get_text()
                images = page.get_images()
                
                total_text_length += len(text.strip())
                total_images += len(images)
            
            doc.close()
            
            # Heuristic: if very little text but many images, likely scannable
            avg_text_per_page = total_text_length / sample_pages
            avg_images_per_page = total_images / sample_pages
            
            # Consider scannable if < 200 chars per page but > 1 image per page
            return avg_text_per_page < 200 and avg_images_per_page > 0.5
            
        except Exception as e:
            logger.error(f"Failed to analyze text scannability for {pdf_path}: {e}")
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of languages supported by Tesseract."""
        if not HAS_OCR or not self.ocr_available:
            return []
        
        try:
            return pytesseract.get_languages()
        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return ['eng']  # Return default
