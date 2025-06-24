"""
Document storage system with organized directory structure and integrity verification.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Comprehensive document metadata structure."""
    document_id: str
    original_url: str
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    source_type: str = "unknown"  # pdf, text, html, etc.
    content_type: str = "application/pdf"
    file_size: Optional[int] = None
    checksum_md5: Optional[str] = None
    checksum_sha256: Optional[str] = None
    download_timestamp: str = ""
    last_accessed: str = ""
    tags: List[str] = None
    categories: List[str] = None
    local_path: Optional[str] = None
    text_extracted: bool = False
    vector_indexed: bool = False
    validation_status: str = "pending"  # pending, valid, invalid, corrupted
    extraction_quality: Optional[float] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.categories is None:
            self.categories = []
        if not self.download_timestamp:
            self.download_timestamp = datetime.utcnow().isoformat()

class DocumentStorageSystem:
    """Organized storage system for historical research documents."""
    
    def __init__(self, base_path: str = "data/research_documents"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create organized directory structure
        self.directories = {
            'documents': self.base_path / "documents",
            'text': self.base_path / "extracted_text", 
            'metadata': self.base_path / "metadata",
            'cache': self.base_path / "cache",
            'temp': self.base_path / "temp",
            'index': self.base_path / "index"
        }
        
        for directory in self.directories.values():
            directory.mkdir(exist_ok=True)
        
        # Create subdirectories for different document types
        self.doc_subdirs = {
            'pdf': self.directories['documents'] / "pdfs",
            'text': self.directories['documents'] / "text_files",
            'html': self.directories['documents'] / "html_files",
            'images': self.directories['documents'] / "images"
        }
        
        for subdir in self.doc_subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        # Initialize document index
        self.index_file = self.directories['index'] / "document_index.json"
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Ensure document index file exists."""
        if not self.index_file.exists():
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
    
    def _calculate_checksums(self, file_path: str) -> Dict[str, str]:
        """Calculate MD5 and SHA256 checksums for file integrity."""
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        
        return {
            'md5': md5_hash.hexdigest(),
            'sha256': sha256_hash.hexdigest()
        }
    
    def store_document(self, 
                      file_path: str, 
                      metadata: DocumentMetadata,
                      verify_integrity: bool = True) -> bool:
        """Store document with metadata and integrity verification."""
        try:
            # Determine target directory based on content type
            if metadata.content_type.startswith('application/pdf'):
                target_dir = self.doc_subdirs['pdf']
                extension = '.pdf'
            elif metadata.content_type.startswith('text/'):
                target_dir = self.doc_subdirs['text']
                extension = '.txt'
            elif metadata.content_type.startswith('text/html'):
                target_dir = self.doc_subdirs['html']
                extension = '.html'
            else:
                target_dir = self.directories['documents']
                extension = os.path.splitext(file_path)[1] or '.bin'
            
            # Generate safe filename
            safe_filename = self._generate_safe_filename(metadata.document_id, extension)
            target_path = target_dir / safe_filename
            
            # Copy file to storage location
            import shutil
            shutil.copy2(file_path, target_path)
            
            # Calculate checksums for integrity
            if verify_integrity:
                checksums = self._calculate_checksums(str(target_path))
                metadata.checksum_md5 = checksums['md5']
                metadata.checksum_sha256 = checksums['sha256']
            
            # Update metadata
            metadata.local_path = str(target_path)
            metadata.file_size = target_path.stat().st_size
            metadata.validation_status = "valid"
            
            # Store metadata
            self._store_metadata(metadata)
            
            # Update index
            self._update_index(metadata)
            
            logger.info(f"Document {metadata.document_id} stored successfully at {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document {metadata.document_id}: {e}")
            return False
    
    def _generate_safe_filename(self, document_id: str, extension: str) -> str:
        """Generate filesystem-safe filename."""
        safe_id = "".join(c for c in document_id if c.isalnum() or c in '-_')
        return f"{safe_id}{extension}"
    
    def _store_metadata(self, metadata: DocumentMetadata):
        """Store document metadata as JSON file."""
        metadata_file = self.directories['metadata'] / f"{metadata.document_id}.json"
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to store metadata for {metadata.document_id}: {e}")
            raise
    
    def _update_index(self, metadata: DocumentMetadata):
        """Update the central document index."""
        try:
            # Load existing index
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            # Update with new document
            index[metadata.document_id] = {
                'title': metadata.title,
                'author': metadata.author,
                'source_type': metadata.source_type,
                'local_path': metadata.local_path,
                'download_timestamp': metadata.download_timestamp,
                'tags': metadata.tags,
                'categories': metadata.categories,
                'text_extracted': metadata.text_extracted,
                'vector_indexed': metadata.vector_indexed
            }
            
            # Save updated index
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to update index for {metadata.document_id}: {e}")
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata by ID."""
        metadata_file = self.directories['metadata'] / f"{document_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return DocumentMetadata(**data)
            
        except Exception as e:
            logger.error(f"Failed to load metadata for {document_id}: {e}")
            return None
    
    def verify_document_integrity(self, document_id: str) -> bool:
        """Verify document integrity using stored checksums."""
        metadata = self.get_document(document_id)
        if not metadata or not metadata.local_path:
            return False
        
        if not os.path.exists(metadata.local_path):
            logger.error(f"Document file missing: {metadata.local_path}")
            return False
        
        if not metadata.checksum_sha256:
            logger.warning(f"No checksum available for {document_id}")
            return True  # Assume valid if no checksum
        
        try:
            current_checksums = self._calculate_checksums(metadata.local_path)
            return current_checksums['sha256'] == metadata.checksum_sha256
        except Exception as e:
            logger.error(f"Failed to verify integrity for {document_id}: {e}")
            return False
    
    def list_documents(self, 
                      category: Optional[str] = None,
                      tag: Optional[str] = None,
                      source_type: Optional[str] = None) -> List[DocumentMetadata]:
        """List documents with optional filtering."""
        documents = []
        
        for metadata_file in self.directories['metadata'].glob("*.json"):
            document_id = metadata_file.stem
            metadata = self.get_document(document_id)
            
            if not metadata:
                continue
            
            # Apply filters
            if category and category not in metadata.categories:
                continue
            if tag and tag not in metadata.tags:
                continue
            if source_type and metadata.source_type != source_type:
                continue
            
            documents.append(metadata)
        
        return documents
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage system statistics."""
        all_docs = self.list_documents()
        
        total_size = sum(doc.file_size or 0 for doc in all_docs)
        type_counts = {}
        
        for doc in all_docs:
            source_type = doc.source_type
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
        
        return {
            'total_documents': len(all_docs),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'documents_by_type': type_counts,
            'text_extracted_count': sum(1 for doc in all_docs if doc.text_extracted),
            'vector_indexed_count': sum(1 for doc in all_docs if doc.vector_indexed)
        }
    
    def cleanup_temp_files(self):
        """Clean up temporary files older than 24 hours."""
        import time
        current_time = time.time()
        
        for temp_file in self.directories['temp'].glob("*"):
            try:
                if current_time - temp_file.stat().st_mtime > 86400:  # 24 hours
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
