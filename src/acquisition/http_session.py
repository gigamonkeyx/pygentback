"""
Enhanced HTTP session management for academic research with retry logic and proper headers.
"""
import requests
import time
import hashlib
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AcademicHTTPSession:
    """HTTP session optimized for academic research with robust error handling."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 backoff_factor: float = 1.0,
                 timeout: int = 30):
        self.session = requests.Session()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        
        # Set academic research user agent
        self.session.headers.update({
            'User-Agent': 'PyGent Academic Research System/1.0 (Historical Document Analysis; +https://github.com/pygent-factory)',
            'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0  # We handle retries manually
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def get_with_retry(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Get URL with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url, 
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    return None
                
                wait_time = self.backoff_factor * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {url}, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
        
        return None
    
    def download_file(self, url: str, stream: bool = True) -> Optional[requests.Response]:
        """Download file with streaming support for large files."""
        try:
            response = self.session.get(
                url,
                stream=stream,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file from {url}: {e}")
            return None
    
    def head_request(self, url: str) -> Optional[requests.Response]:
        """Perform HEAD request to check file existence and get metadata."""
        try:
            response = self.session.head(url, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.debug(f"HEAD request failed for {url}: {e}")
            return None
    
    def generate_document_id(self, url: str) -> str:
        """Generate unique document ID from URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def get_file_info(self, url: str) -> Dict[str, Any]:
        """Get file information without downloading the full content."""
        head_response = self.head_request(url)
        if not head_response:
            return {}
        
        return {
            'content_type': head_response.headers.get('content-type', ''),
            'content_length': head_response.headers.get('content-length'),
            'last_modified': head_response.headers.get('last-modified'),
            'etag': head_response.headers.get('etag'),
            'server': head_response.headers.get('server', ''),
            'document_id': self.generate_document_id(url)
        }
    
    def close(self):
        """Close the session and cleanup resources."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
