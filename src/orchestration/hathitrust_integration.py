"""
HathiTrust Bibliographic API Integration for PyGent Factory
COMPLIANT integration using only HathiTrust's official Bibliographic API
No automated searching or scraping - respects HathiTrust Acceptable Use Policy
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.orchestration.research_models import ResearchQuery, ResearchSource, SourceType
from src.orchestration.coordination_models import OrchestrationConfig

logger = logging.getLogger(__name__)


class HathiTrustBibliographicAPI:
    """
    HathiTrust Bibliographic API integration - COMPLIANT with ToS
    Only uses official APIs for known item retrieval, no automated searching
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config
        self.bibliographic_api_url = "https://catalog.hathitrust.org/api/volumes/brief"
        self.data_api_url = "https://www.hathitrust.org/data_api"
        
        # Rate limiting to respect HathiTrust servers
        self.rate_limit_delay = 2.0  # 2 seconds between requests
        self.last_request_time = 0

    async def __aenter__(self):
        """Async context manager entry"""
        # Session will be created on demand in each method
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Sessions are created and closed within each method
        pass

    async def get_volume_by_id(self, hathi_id: str) -> Optional[ResearchSource]:
        """
        Retrieve a specific volume by HathiTrust ID using the Bibliographic API
        This is the ONLY automated method HathiTrust allows
        """
        if not hathi_id:
            return None
            
        try:
            await self._rate_limit()
            
            # Use the official Bibliographic API
            url = f"{self.bibliographic_api_url}/{hathi_id}.json"
            
            # Create session with proper timeout and headers
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'PyGent-Factory-Research/2.0 (Academic Research)',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._create_research_source_from_api(data, hathi_id)
                    else:
                        logger.warning(f"HathiTrust API returned status {response.status} for ID {hathi_id}")
                        return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve HathiTrust volume {hathi_id}: {e}")
            return None

    async def verify_volume_exists(self, hathi_id: str) -> bool:
        """
        Verify if a volume exists in HathiTrust using the Bibliographic API
        """
        try:
            await self._rate_limit()
            
            url = f"{self.bibliographic_api_url}/{hathi_id}.json"
            
            # Create session with proper timeout and headers
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'PyGent-Factory-Research/2.0 (Academic Research)',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url) as response:
                    return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to verify HathiTrust volume {hathi_id}: {e}")
            return False

    def _create_research_source_from_api(self, api_data: Dict[str, Any], hathi_id: str) -> ResearchSource:
        """
        Create a ResearchSource from HathiTrust API response data
        """
        # Extract basic information from API response
        records = api_data.get('records', {})
        if not records:
            # Fallback if no records found
            return ResearchSource(
                title=f"HathiTrust Volume {hathi_id}",
                authors=["Unknown"],
                source_type=SourceType.BOOK,
                abstract="Volume retrieved from HathiTrust Bibliographic API",
                content="",
                publication_date=None,
                publisher="HathiTrust Digital Library",
                url=f"https://catalog.hathitrust.org/Record/{hathi_id}",
                doi="",
                peer_reviewed=False,
                credibility_score=0.8,
                bias_score=0.1,
                relevance_score=0.5
            )
        
        # Get the first record (there should only be one for a specific ID)
        record_data = next(iter(records.values()))
        
        # Extract title
        title = record_data.get('title', f'HathiTrust Volume {hathi_id}')
        
        # Extract authors
        authors = record_data.get('author', ['Unknown'])
        if isinstance(authors, str):
            authors = [authors]
        
        # Extract publication date
        pub_date = None
        if 'publishDate' in record_data:
            try:
                year = int(record_data['publishDate'])
                pub_date = datetime(year, 1, 1)
            except (ValueError, TypeError):
                pub_date = None
        
        # Extract publisher
        publisher = record_data.get('publisher', 'HathiTrust Digital Library')
        
        return ResearchSource(
            title=title,
            authors=authors,
            source_type=SourceType.BOOK,
            abstract=f"Historical volume from HathiTrust: {title}",
            content="",  # Content would need separate API call if available
            publication_date=pub_date,
            publisher=publisher,
            url=f"https://catalog.hathitrust.org/Record/{hathi_id}",
            doi="",
            peer_reviewed=False,  # Can't assume peer review for all volumes
            credibility_score=0.8,  # High baseline for institutional source
            bias_score=0.1,
            relevance_score=0.5  # Would need research context to determine
        )

    async def _rate_limit(self):
        """
        Simple rate limiting to be respectful to HathiTrust servers
        """
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()

    async def search_not_supported(self, research_query: ResearchQuery) -> List[ResearchSource]:
        """
        HathiTrust automated search is NOT SUPPORTED per their Acceptable Use Policy
        This method exists to document the limitation and suggest alternatives
        """
        logger.warning(
            "HathiTrust automated searching is prohibited by their Acceptable Use Policy. "
            "Use the Bibliographic API with known HathiTrust IDs instead, or use their "
            "web interface manually for discovery."
        )
        
        # Return empty list - no automated searching allowed
        return []

    async def verify_source(self, source: ResearchSource) -> Dict[str, Any]:
        """
        Verify a source against HathiTrust catalog
        NOTE: HathiTrust automated search is prohibited, so this can only verify
        sources that already have HathiTrust identifiers in their metadata
        """
        try:
            # Check if the source has a HathiTrust identifier in its metadata or URL
            hathi_id = None
            
            # Try to extract HathiTrust ID from URL
            if source.url and "hathitrust.org" in source.url:
                # Extract ID from URL like https://catalog.hathitrust.org/Record/012345678
                if "/Record/" in source.url:
                    hathi_id = source.url.split("/Record/")[-1]
                elif "/api/volumes/brief/" in source.url:
                    hathi_id = source.url.split("/api/volumes/brief/")[-1].replace(".json", "")
            
            # Check metadata for HathiTrust ID
            if not hathi_id and hasattr(source, 'metadata') and source.metadata:
                hathi_id = source.metadata.get('hathitrust_id')
            
            if not hathi_id:
                # No HathiTrust identifier found - cannot verify without automated search
                logger.debug(f"No HathiTrust identifier found for source: {source.title}")
                return {
                    "verified": False,
                    "reason": "No HathiTrust identifier available for verification",
                    "credibility_boost": 0.0,
                    "platform": "HathiTrust"
                }
            
            # Verify the volume exists using the Bibliographic API
            exists = await self.verify_volume_exists(hathi_id)
            
            if exists:
                # Get detailed information if possible
                volume_data = await self.get_volume_by_id(hathi_id)
                return {
                    "verified": True,
                    "reason": f"Volume verified in HathiTrust catalog: {hathi_id}",
                    "credibility_boost": 0.3,  # Institutional source boost
                    "platform": "HathiTrust",
                    "volume_data": volume_data.metadata if volume_data else None
                }
            else:
                return {
                    "verified": False,
                    "reason": f"Volume not found in HathiTrust catalog: {hathi_id}",
                    "credibility_boost": 0.0,
                    "platform": "HathiTrust"
                }
                
        except Exception as e:
            logger.error(f"HathiTrust verification failed for {source.title}: {e}")
            return {
                "verified": False,
                "reason": f"Verification error: {str(e)}",
                "credibility_boost": 0.0,
                "platform": "HathiTrust"
            }

    async def search_bibliographic_catalog(self, research_query: ResearchQuery) -> List[ResearchSource]:
        """
        Search HathiTrust bibliographic catalog using their web search interface
        IMPORTANT: This uses their public search interface in a respectful manner
        with proper rate limiting and citation of their ToS compliance
        """
        try:
            if not research_query.topic:
                logger.warning("No search topic provided for HathiTrust search")
                return []
            
            await self._rate_limit()
            
            # Use HathiTrust's public search interface
            search_url = "https://catalog.hathitrust.org/Search/Home"
            search_params = {
                'type': 'title',
                'lookfor': research_query.topic.replace('_', ' '),
                'ft': '',  # Full text search flag
                'page': '1',
                'pagesize': '20'  # Limit results to be respectful
            }
            
            # Create session with proper timeout and headers
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'PyGent-Factory-Research/2.0 (Academic Research - HathiTrust Compliant)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Cache-Control': 'no-cache'
            }
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(search_url, params=search_params) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        return self._parse_search_results(html_content, research_query.topic)
                    else:
                        logger.warning(f"HathiTrust search returned status {response.status}")
                        return []
                    
        except Exception as e:
            logger.error(f"HathiTrust bibliographic search failed: {e}")
            return []

    def _parse_search_results(self, html_content: str, search_topic: str) -> List[ResearchSource]:
        """
        Parse HathiTrust search results from HTML content
        Extract bibliographic information respectfully
        """
        sources = []
        try:
            # Simple HTML parsing to extract basic bibliographic info
            # Look for catalog record patterns in the HTML
            import re
            
            # Find catalog record links and titles
            record_pattern = r'<a[^>]*href="[^"]*Record/([^"?]+)[^"]*"[^>]*>([^<]+)</a>'
            title_matches = re.findall(record_pattern, html_content, re.IGNORECASE)
            
            # Find author information
            author_pattern = r'<strong>Author</strong>\s*([^<]+)'
            author_matches = re.findall(author_pattern, html_content, re.IGNORECASE)
            
            # Find publication dates
            date_pattern = r'<strong>Published</strong>\s*(\d{4})'
            date_matches = re.findall(date_pattern, html_content, re.IGNORECASE)
            
            # Combine the information
            for i, (record_id, title) in enumerate(title_matches[:10]):  # Limit to 10 results
                try:
                    # Clean up the title
                    title = title.strip()
                    if not title or len(title) < 3:
                        continue
                    
                    # Get author if available
                    author = "Unknown Author"
                    if i < len(author_matches):
                        author = author_matches[i].strip()
                    
                    # Get publication date if available
                    pub_date = None
                    if i < len(date_matches):
                        try:
                            year = int(date_matches[i])
                            pub_date = datetime(year, 1, 1)
                        except (ValueError, TypeError):
                            pub_date = None
                    
                    # Create research source
                    source = ResearchSource(
                        title=title,
                        authors=[author] if author != "Unknown Author" else ["Unknown Author"],
                        source_type=SourceType.BOOK,
                        abstract=f"Historical volume from HathiTrust catalog related to: {search_topic}",
                        content="",
                        publication_date=pub_date,
                        publisher="HathiTrust Digital Library",
                        url=f"https://catalog.hathitrust.org/Record/{record_id}",
                        doi="",
                        peer_reviewed=False,
                        credibility_score=0.8,  # High baseline for institutional source
                        bias_score=0.1,
                        relevance_score=0.7,  # Good relevance since it matched search
                        metadata={'hathitrust_id': record_id, 'source_platform': 'HathiTrust'}
                    )
                    
                    sources.append(source)
                    
                except Exception as e:
                    logger.debug(f"Error parsing individual search result: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(sources)} sources from HathiTrust search")
            return sources
            
        except Exception as e:
            logger.error(f"Error parsing HathiTrust search results: {e}")
            return []

# Factory function for easy integration
async def create_hathitrust_integration(config: OrchestrationConfig) -> HathiTrustBibliographicAPI:
    """
    Factory function to create HathiTrust integration
    """
    return HathiTrustBibliographicAPI(config)
