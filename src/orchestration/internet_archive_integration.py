"""
Internet Archive API Integration for PyGent Factory
Uses Internet Archive's legitimate APIs for historical research
Focuses on books, texts, and historical documents
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.orchestration.research_models import ResearchQuery, ResearchSource, SourceType
from src.orchestration.coordination_models import OrchestrationConfig

logger = logging.getLogger(__name__)


class InternetArchiveAPI:
    """
    Internet Archive API integration for historical research
    Uses their legitimate search and metadata APIs
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config
        self.search_api_url = "https://archive.org/advancedsearch.php"
        self.metadata_api_url = "https://archive.org/metadata"
        self.details_url = "https://archive.org/details"
        
        # Rate limiting to be respectful
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
        
        # Session for HTTP requests
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'PyGent-Factory-Research/2.0 (Academic Research)',
                'Accept': 'application/json'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def search_historical_documents(self, research_query: ResearchQuery) -> List[ResearchSource]:
        """
        Search Internet Archive for historical documents related to the research query
        """
        try:
            if not self.session:
                await self.__aenter__()
            
            # Build search query
            search_params = self._build_search_params(research_query)
            
            await self._rate_limit()
            
            # Execute search
            async with self.session.get(self.search_api_url, params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_search_results(data)
                else:
                    logger.warning(f"Internet Archive API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Internet Archive search failed: {e}")
            return []

    def _build_search_params(self, research_query: ResearchQuery) -> Dict[str, Any]:
        """
        Build search parameters for Internet Archive API
        """
        # Base query from topic
        query_terms = []
        if research_query.topic:
            # Clean up topic for search
            clean_topic = research_query.topic.replace('_', ' ')
            query_terms.append(f'"{clean_topic}"')
          # Add related terms if available
        if hasattr(research_query, 'keywords') and research_query.keywords:
            query_terms.extend(research_query.keywords[:3])  # Limit to avoid too complex queries
          # Combine query terms
        q = ' OR '.join(query_terms) if query_terms else research_query.topic or 'history'
        
        return {
            'q': q,
            'fl': 'identifier,title,creator,date,description,subject,mediatype,format,downloads',
            'sort': 'downloads desc',  # Popular items first
            'rows': 20,  # Reasonable number of results
            'page': 1,
            'output': 'json',
            # Remove the problematic filter for now to get basic functionality working
        }

    async def _process_search_results(self, data: Dict[str, Any]) -> List[ResearchSource]:
        """
        Process Internet Archive search results into ResearchSource objects
        """
        sources = []
        docs = data.get('response', {}).get('docs', [])
        
        for doc in docs:
            try:
                source = await self._create_research_source(doc)
                if source:
                    sources.append(source)
            except Exception as e:
                logger.warning(f"Failed to process IA document: {e}")
                continue
        
        logger.info(f"Internet Archive returned {len(sources)} sources")
        return sources

    async def _create_research_source(self, doc: Dict[str, Any]) -> Optional[ResearchSource]:
        """
        Create ResearchSource from Internet Archive document data
        """
        identifier = doc.get('identifier')
        if not identifier:
            return None
        
        # Extract basic metadata
        title = doc.get('title', 'Unknown Title')
        if isinstance(title, list):
            title = title[0] if title else 'Unknown Title'
        
        # Extract creators/authors
        creators = doc.get('creator', [])
        if isinstance(creators, str):
            creators = [creators]
        elif not isinstance(creators, list):
            creators = ['Unknown Author']
        
        # Extract publication date
        pub_date = None
        date_str = doc.get('date')
        if date_str:
            try:
                if isinstance(date_str, list):
                    date_str = date_str[0]
                # Try to parse year from date string
                year = int(str(date_str)[:4])
                if 1000 <= year <= datetime.now().year:
                    pub_date = datetime(year, 1, 1)
            except (ValueError, TypeError):
                pub_date = None
        
        # Extract description
        description = doc.get('description', '')
        if isinstance(description, list):
            description = ' '.join(description) if description else ''
        
        # Extract subjects for additional context
        subjects = doc.get('subject', [])
        if isinstance(subjects, str):
            subjects = [subjects]
        
        # Determine source type
        mediatype = doc.get('mediatype', 'texts')
        source_type = SourceType.BOOK if mediatype == 'texts' else SourceType.ACADEMIC_PAPER
        
        # Calculate basic scores
        downloads = doc.get('downloads', 0)
        popularity_score = min(downloads / 1000.0, 1.0) if downloads else 0.1
        
        return ResearchSource(
            title=title,
            authors=creators,
            source_type=source_type,
            abstract=description[:500] + '...' if len(description) > 500 else description,
            content="",  # Would need separate API call for full text
            publication_date=pub_date,
            publisher="Internet Archive",
            url=f"{self.details_url}/{identifier}",
            doi="",
            peer_reviewed=False,  # Can't assume peer review
            credibility_score=min(0.7 + popularity_score * 0.2, 0.9),  # Base 0.7, up to 0.9
            bias_score=0.2,  # Moderate baseline
            relevance_score=0.6,  # Would need semantic analysis to improve
            metadata={
                'identifier': identifier,
                'subjects': subjects,
                'downloads': downloads,
                'mediatype': mediatype,
                'platform': 'Internet Archive'
            }
        )

    async def get_document_by_id(self, identifier: str) -> Optional[ResearchSource]:
        """
        Retrieve a specific document by Internet Archive identifier
        """
        try:
            await self._rate_limit()
            
            if not self.session:
                await self.__aenter__()
            
            # Get metadata for the document
            url = f"{self.metadata_api_url}/{identifier}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._create_research_source_from_metadata(data, identifier)
                else:
                    logger.warning(f"Internet Archive metadata API returned status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to retrieve IA document {identifier}: {e}")
            return None

    async def _create_research_source_from_metadata(self, metadata: Dict[str, Any], identifier: str) -> Optional[ResearchSource]:
        """
        Create ResearchSource from detailed metadata response
        """
        # Extract metadata from the response
        meta = metadata.get('metadata', {})
        if not meta:
            return None
        
        # Similar extraction logic as search results but with richer metadata
        title = meta.get('title', 'Unknown Title')
        if isinstance(title, list):
            title = title[0] if title else 'Unknown Title'
        
        creators = meta.get('creator', [])
        if isinstance(creators, str):
            creators = [creators]
        elif not isinstance(creators, list):
            creators = ['Unknown Author']
        
        # Extract publication date
        pub_date = None
        date_str = meta.get('date') or meta.get('year')
        if date_str:
            try:
                if isinstance(date_str, list):
                    date_str = date_str[0]
                year = int(str(date_str)[:4])
                if 1000 <= year <= datetime.now().year:
                    pub_date = datetime(year, 1, 1)
            except (ValueError, TypeError):
                pub_date = None
        
        description = meta.get('description', '')
        if isinstance(description, list):
            description = ' '.join(description) if description else ''
        
        subjects = meta.get('subject', [])
        if isinstance(subjects, str):
            subjects = [subjects]
        
        return ResearchSource(
            title=title,
            authors=creators,
            source_type=SourceType.BOOK,
            abstract=description[:500] + '...' if len(description) > 500 else description,
            content="",
            publication_date=pub_date,
            publisher="Internet Archive",
            url=f"{self.details_url}/{identifier}",
            doi="",
            peer_reviewed=False,
            credibility_score=0.8,  # High baseline for IA
            bias_score=0.2,
            relevance_score=0.7,
            metadata={
                'identifier': identifier,
                'subjects': subjects,
                'platform': 'Internet Archive'
            }
        )

    async def _rate_limit(self):
        """
        Simple rate limiting to be respectful to Internet Archive servers
        """
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()


# Factory function for easy integration
async def create_internet_archive_integration(config: OrchestrationConfig) -> InternetArchiveAPI:
    """
    Factory function to create Internet Archive integration
    """
    return InternetArchiveAPI(config)
