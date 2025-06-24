"""
Document download pipeline with PDF validation and streaming support.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass

from .http_session import AcademicHTTPSession

logger = logging.getLogger(__name__)

@dataclass
class DownloadResult:
    """Result of document download operation."""
    success: bool
    file_path: Optional[str] = None
    document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    file_size: Optional[int] = None

class DocumentDownloadPipeline:
    """Pipeline for downloading and validating academic documents."""
    
    # PDF magic numbers for validation
    PDF_MAGIC_NUMBERS = [
        b'%PDF-',  # Standard PDF header
        b'\x25\x50\x44\x46\x2d'  # PDF header in hex
    ]
    
    def __init__(self, storage_root: str = "data/documents"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.pdf_dir = self.storage_root / "pdfs"
        self.text_dir = self.storage_root / "text"
        self.metadata_dir = self.storage_root / "metadata"
        
        for directory in [self.pdf_dir, self.text_dir, self.metadata_dir]:
            directory.mkdir(exist_ok=True)
    
    def validate_pdf_content(self, content: bytes) -> bool:
        """Validate that content is a valid PDF file."""
        if not content:
            return False
        
        # Check PDF magic number
        for magic in self.PDF_MAGIC_NUMBERS:
            if content.startswith(magic):
                return True
        
        return False
    
    def get_safe_filename(self, document_id: str, extension: str = ".pdf") -> str:
        """Generate safe filename from document ID."""
        # Ensure filename is safe for filesystem
        safe_id = "".join(c for c in document_id if c.isalnum() or c in '-_')
        return f"{safe_id}{extension}"
    
    def download_document(self, url: str, 
                         expected_type: str = "application/pdf") -> DownloadResult:
        """Download document with validation and storage."""
        
        with AcademicHTTPSession() as session:
            # Get file info first
            file_info = session.get_file_info(url)
            if not file_info:
                return DownloadResult(
                    success=False,
                    error_message="Failed to get file information"
                )
            
            document_id = file_info['document_id']
            
            # Check if already downloaded
            existing_file = self.pdf_dir / self.get_safe_filename(document_id)
            if existing_file.exists():
                logger.info(f"Document {document_id} already exists, skipping download")
                return DownloadResult(
                    success=True,
                    file_path=str(existing_file),
                    document_id=document_id,
                    metadata=file_info,
                    file_size=existing_file.stat().st_size
                )
            
            # Download the file
            logger.info(f"Downloading document from {url}")
            response = session.download_file(url, stream=True)
            
            if not response:
                return DownloadResult(
                    success=False,
                    error_message="Failed to download file"
                )
            
            # Download to temporary file first
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    total_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                            total_size += len(chunk)
                    
                    temp_path = temp_file.name
                
                # Read and validate content
                with open(temp_path, 'rb') as f:
                    content = f.read(1024)  # Read first 1KB for validation
                
                # Validate PDF if expected
                if expected_type == "application/pdf":
                    if not self.validate_pdf_content(content):
                        os.unlink(temp_path)
                        return DownloadResult(
                            success=False,
                            error_message="Downloaded content is not a valid PDF"
                        )
                
                # Move to final location
                final_path = self.pdf_dir / self.get_safe_filename(document_id)
                shutil.move(temp_path, final_path)
                
                # Update metadata with file size
                file_info['file_size'] = total_size
                file_info['local_path'] = str(final_path)
                
                # Save metadata
                self._save_metadata(document_id, file_info)
                
                logger.info(f"Successfully downloaded document {document_id} ({total_size} bytes)")
                
                return DownloadResult(
                    success=True,
                    file_path=str(final_path),
                    document_id=document_id,
                    metadata=file_info,
                    file_size=total_size
                )
                
            except Exception as e:
                # Cleanup temp file if it exists
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                logger.error(f"Error downloading document from {url}: {e}")
                return DownloadResult(
                    success=False,
                    error_message=f"Download error: {str(e)}"
                )
    
    def _save_metadata(self, document_id: str, metadata: Dict[str, Any]):
        """Save document metadata to JSON file."""
        import json
        
        metadata_file = self.metadata_dir / f"{document_id}.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata for {document_id}: {e}")
    
    def get_document_path(self, document_id: str) -> Optional[str]:
        """Get local path for downloaded document."""
        file_path = self.pdf_dir / self.get_safe_filename(document_id)
        return str(file_path) if file_path.exists() else None
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get stored metadata for document."""
        import json
        
        metadata_file = self.metadata_dir / f"{document_id}.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata for {document_id}: {e}")
            return None
    
    def list_downloaded_documents(self) -> Dict[str, Dict[str, Any]]:
        """List all downloaded documents with their metadata."""
        documents = {}
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            document_id = metadata_file.stem
            metadata = self.get_document_metadata(document_id)
            if metadata:
                documents[document_id] = metadata
        
        return documents
