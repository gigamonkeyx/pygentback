"""
Robust Research Portal Manager

Implements best practices for academic research API integration with:
- Proper rate limiting for ArXiv, Semantic Scholar, and Google Scholar
- Robust error handling with exponential backoff
- Connection pooling and session management
- Circuit breaker pattern for reliability
- Comprehensive logging and monitoring
"""

import asyncio
import logging
import time
import requests
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class ResearchPortal(Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    GOOGLE_SCHOLAR = "google_scholar"


@dataclass
class PortalConfig:
    """Configuration for a research portal"""
    name: str
    base_url: str
    rate_limit_per_second: float
    max_retries: int = 3
    timeout_seconds: int = 30
    requires_proxy: bool = False
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a portal"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    timeout_seconds: int = 300  # 5 minutes


class RobustResearchPortalManager:
    """
    Robust manager for academic research portals with comprehensive error handling
    """
    
    def __init__(self):
        self.portal_configs = self._initialize_portal_configs()
        self.circuit_breakers = {portal: CircuitBreakerState() for portal in ResearchPortal}
        self.last_request_times = {portal: 0.0 for portal in ResearchPortal}
        self.session_pools = {}
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'circuit_breaker_trips': 0
        }
        
        # Initialize session pools
        self._initialize_session_pools()
        
        # Skip Google Scholar setup to avoid CAPTCHA issues during initialization
        # Google Scholar will be bypassed in favor of ArXiv + Semantic Scholar + CrossRef
        self.google_scholar_proxy = None
        logger.info("Google Scholar disabled to avoid CAPTCHA issues - using ArXiv + Semantic Scholar + CrossRef")
    
    def _initialize_portal_configs(self) -> Dict[ResearchPortal, PortalConfig]:
        """Initialize configuration for each research portal"""
        return {
            ResearchPortal.ARXIV: PortalConfig(
                name="ArXiv",
                base_url="http://export.arxiv.org/api/query",
                rate_limit_per_second=3.0,  # 3 requests per second max
                max_retries=3,
                timeout_seconds=30,
                headers={
                    'User-Agent': 'PyGent-Factory-Research/1.0 (mailto:research@pygent.ai)'
                }
            ),
            ResearchPortal.SEMANTIC_SCHOLAR: PortalConfig(
                name="Semantic Scholar",
                base_url="https://api.semanticscholar.org/graph/v1",
                rate_limit_per_second=1.0,  # Free tier: 1 request per second
                max_retries=3,
                timeout_seconds=30,
                headers={
                    'User-Agent': 'PyGent-Factory-Research/1.0 (mailto:research@pygent.ai)'
                }
            ),
            ResearchPortal.GOOGLE_SCHOLAR: PortalConfig(
                name="Google Scholar",
                base_url="https://scholar.google.com",
                rate_limit_per_second=0.1,  # Very conservative: 1 request per 10 seconds
                max_retries=2,
                timeout_seconds=45,
                requires_proxy=True,  # Requires proxy for production use
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
        }
    
    def _initialize_session_pools(self):
        """Initialize persistent HTTP session pools for each portal"""
        for portal in ResearchPortal:
            config = self.portal_configs[portal]
            
            # Create session with retry strategy
            session = requests.Session()
            
            # Configure retry adapter
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            retry_strategy = Retry(
                total=config.max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Set headers
            session.headers.update(config.headers)
            
            # Set timeout
            session.timeout = (5, config.timeout_seconds)  # connect, read
            
            self.session_pools[portal] = session
    
    def _setup_google_scholar_proxy(self):
        """Setup Google Scholar with proper Chrome configuration and proxy"""
        try:
            from scholarly import scholarly, ProxyGenerator
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            
            logger.info("Setting up Google Scholar with Chrome WebDriver...")
            
            # Set up Chrome options for scholarly
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Run in background
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            try:
                # Use webdriver-manager to automatically manage ChromeDriver
                chrome_driver_path = ChromeDriverManager().install()
                logger.info(f"Chrome driver installed at: {chrome_driver_path}")
                
                # Test if we can create a webdriver instance
                service = Service(chrome_driver_path)
                test_driver = webdriver.Chrome(service=service, options=chrome_options)
                test_driver.quit()
                logger.info("Chrome WebDriver test successful")
                
                # Configure scholarly to work with our Chrome setup
                # Note: scholarly will use its own webdriver management
                
                # Now set up proxy configuration
                pg = ProxyGenerator()
                
                # Try different proxy configurations
                proxy_success = False
                
                # Option 1: Try FreeProxies (updated API)
                try:
                    pg.FreeProxies()
                    scholarly.use_proxy(pg)
                    proxy_success = True
                    logger.info("Google Scholar configured with FreeProxies")
                except Exception as e:
                    logger.warning(f"FreeProxies failed: {e}")
                
                # Option 2: If FreeProxies failed, try direct connection
                if not proxy_success:
                    try:
                        # Use direct connection without proxy
                        scholarly.use_proxy(None)
                        proxy_success = True
                        logger.info("Google Scholar configured for direct connection")
                    except Exception as e:
                        logger.warning(f"Direct connection setup failed: {e}")
                        # Try without any proxy configuration
                        proxy_success = True  # Continue anyway
                        logger.info("Google Scholar using default configuration")
                
                if proxy_success:
                    self.google_scholar_proxy = pg if 'pg' in locals() else None
                    
                    # Test the configuration with a simple query
                    try:
                        test_search = scholarly.search_pubs("artificial intelligence")
                        test_result = next(test_search)
                        logger.info(f"Google Scholar test successful - found: {test_result.get('bib', {}).get('title', 'Unknown')}")
                    except Exception as e:
                        logger.warning(f"Google Scholar test failed: {e}")
                        # Don't fail completely, just log the warning
                        
            except Exception as e:
                logger.error(f"Chrome WebDriver setup failed: {e}")
                raise
                
        except ImportError as e:
            logger.error(f"Required libraries not available for Google Scholar: {e}")
            logger.info("Install with: pip install selenium webdriver-manager")
        except Exception as e:
            logger.error(f"Google Scholar setup failed: {e}")
            raise
    
    async def search_portal(self, portal: ResearchPortal, query: str, 
                          max_results: int = 20, **kwargs) -> Dict[str, Any]:
        """
        Search a specific research portal with robust error handling
        """
        # Check circuit breaker
        if not self._check_circuit_breaker(portal):
            return {
                'success': False,
                'error': f'{portal.value} circuit breaker is OPEN',
                'papers': []
            }
        
        # Apply rate limiting
        await self._apply_rate_limiting(portal)
        
        try:
            self.stats['total_requests'] += 1
            
            # Route to specific portal handler
            if portal == ResearchPortal.ARXIV:
                result = await self._search_arxiv(query, max_results, **kwargs)
            elif portal == ResearchPortal.SEMANTIC_SCHOLAR:
                result = await self._search_semantic_scholar(query, max_results, **kwargs)
            elif portal == ResearchPortal.GOOGLE_SCHOLAR:
                # Google Scholar disabled due to CAPTCHA issues
                logger.info("Skipping Google Scholar (disabled due to CAPTCHA issues)")
                return {
                    'success': False,
                    'error': 'Google Scholar disabled (CAPTCHA issues)',
                    'papers': []
                }
            else:
                raise ValueError(f"Unknown portal: {portal}")
            
            # Success - reset circuit breaker
            self._reset_circuit_breaker(portal)
            self.stats['successful_requests'] += 1
            
            return result
            
        except Exception as e:
            # Handle failure
            self._handle_portal_failure(portal, e)
            self.stats['failed_requests'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'papers': []
            }
    
    async def _apply_rate_limiting(self, portal: ResearchPortal):
        """Apply rate limiting for a specific portal"""
        config = self.portal_configs[portal]
        current_time = time.time()
        last_request = self.last_request_times[portal]
        
        if last_request > 0:
            elapsed = current_time - last_request
            min_interval = 1.0 / config.rate_limit_per_second
            
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                # Add jitter to prevent thundering herd
                jitter = random.uniform(0, 0.1)
                total_wait = wait_time + jitter
                
                logger.info(f"Rate limiting {portal.value}: waiting {total_wait:.2f}s")
                await asyncio.sleep(total_wait)
                self.stats['rate_limited_requests'] += 1
        
        self.last_request_times[portal] = time.time()
    
    def _check_circuit_breaker(self, portal: ResearchPortal) -> bool:
        """Check if circuit breaker allows requests"""
        breaker = self.circuit_breakers[portal]
        
        if breaker.state == "CLOSED":
            return True
        elif breaker.state == "OPEN":
            # Check if timeout has passed
            if breaker.last_failure_time:
                elapsed = (datetime.now() - breaker.last_failure_time).total_seconds()
                if elapsed > breaker.timeout_seconds:
                    breaker.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker for {portal.value} moving to HALF_OPEN")
                    return True
            return False
        elif breaker.state == "HALF_OPEN":
            return True
        
        return False
    
    def _handle_portal_failure(self, portal: ResearchPortal, error: Exception):
        """Handle portal failure and update circuit breaker"""
        breaker = self.circuit_breakers[portal]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        logger.error(f"Portal {portal.value} failure #{breaker.failure_count}: {error}")
        
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = "OPEN"
            self.stats['circuit_breaker_trips'] += 1
            logger.warning(f"Circuit breaker for {portal.value} is now OPEN")
    
    def _reset_circuit_breaker(self, portal: ResearchPortal):
        """Reset circuit breaker on successful request"""
        breaker = self.circuit_breakers[portal]
        if breaker.failure_count > 0:
            breaker.failure_count = 0
            breaker.state = "CLOSED"
            logger.info(f"Circuit breaker for {portal.value} reset to CLOSED")

    async def _search_arxiv(self, query: str, max_results: int, **kwargs) -> Dict[str, Any]:
        """Search ArXiv with fixed query formatting"""
        try:
            config = self.portal_configs[ResearchPortal.ARXIV]
            session = self.session_pools[ResearchPortal.ARXIV]

            # Clean and format query for ArXiv - FIX THE 404 ERRORS
            clean_query = self._clean_arxiv_query(query)
            
            # Construct ArXiv query parameters
            params = {
                'search_query': clean_query,
                'start': 0,
                'max_results': min(max_results, 50),  # Reasonable limit
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            logger.info(f"ArXiv search with cleaned query: {clean_query}")

            # Make request with timeout
            response = session.get(config.base_url, params=params, timeout=config.timeout_seconds)
            
            # If 404, try alternative query format
            if response.status_code == 400 or response.status_code == 404:
                params['search_query'] = f'ti:"{query}" OR abs:"{query}"'
                logger.info(f"Retrying ArXiv with alternative query: {params['search_query']}")
                response = session.get(config.base_url, params=params, timeout=config.timeout_seconds)
            
            response.raise_for_status()

            # Parse XML response with improved parser
            papers = self._parse_arxiv_xml_fixed(response.text)

            logger.info(f"ArXiv search returned {len(papers)} papers for query: {query}")

            return {
                'success': True,
                'papers': papers,
                'total_found': len(papers),
                'source': 'arxiv'
            }

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            # Return empty result instead of raising
            return {
                'success': False,
                'error': str(e),
                'papers': [],
                'source': 'arxiv'
            }

    def _clean_arxiv_query(self, query: str) -> str:
        """Clean and format query for ArXiv API to prevent 404s"""
        # Remove special characters that cause issues
        clean_query = re.sub(r'[^\w\s\-\+]', ' ', query)
        clean_query = ' '.join(clean_query.split())  # Normalize whitespace
        
        # Use title and abstract search for better results
        if len(clean_query.split()) > 1:
            return f'ti:"{clean_query}" OR abs:"{clean_query}"'
        else:
            return f'all:{clean_query}'

    def _parse_arxiv_xml_fixed(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse ArXiv XML response with better error handling"""
        papers = []
        
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                try:
                    # Extract paper information
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip() if title_elem is not None else "Unknown Title"
                    
                    summary_elem = entry.find('atom:summary', ns)
                    abstract = summary_elem.text.strip() if summary_elem is not None else ""
                    
                    # Get authors
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    
                    # Get URL
                    url = ""
                    id_elem = entry.find('atom:id', ns)
                    if id_elem is not None:
                        url = id_elem.text.strip()
                    
                    # Get publication date
                    published_elem = entry.find('atom:published', ns)
                    published = published_elem.text[:4] if published_elem is not None else "Unknown"
                    
                    paper = {
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'url': url,
                        'year': published,
                        'source': 'arxiv',
                        'journal': 'arXiv preprint'
                    }
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse ArXiv entry: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to parse ArXiv XML: {e}")
        
        return papers

    async def _search_semantic_scholar(self, query: str, max_results: int, **kwargs) -> Dict[str, Any]:
        """Search Semantic Scholar with robust error handling"""
        try:
            config = self.portal_configs[ResearchPortal.SEMANTIC_SCHOLAR]
            session = self.session_pools[ResearchPortal.SEMANTIC_SCHOLAR]

            # Construct Semantic Scholar query
            url = f"{config.base_url}/paper/search"
            params = {
                'query': query,
                'limit': min(max_results, 100),  # API limit
                'fields': 'paperId,title,authors,abstract,year,journal,citationCount,url,venue'
            }

            # Add API key if available
            headers = {}
            if config.api_key:
                headers['x-api-key'] = config.api_key
                # Higher rate limit with API key
                config.rate_limit_per_second = 100.0

            # Make request
            response = session.get(url, params=params, headers=headers, timeout=config.timeout_seconds)
            response.raise_for_status()

            # Parse JSON response
            data = response.json()
            papers = self._parse_semantic_scholar_json(data)

            logger.info(f"Semantic Scholar search returned {len(papers)} papers for query: {query}")

            return {
                'success': True,
                'papers': papers,
                'total_found': data.get('total', len(papers)),
                'source': 'semantic_scholar'
            }

        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            raise

    async def _search_google_scholar(self, query: str, max_results: int, **kwargs) -> Dict[str, Any]:
        """Search Google Scholar with robust error handling and proxy support"""
        try:
            # Import scholarly library
            try:
                from scholarly import scholarly
                from scholarly._proxy_generator import MaxTriesExceededException, DOSException
            except ImportError:
                raise ImportError("Scholarly library not available - install with: pip install scholarly")

            config = self.portal_configs[ResearchPortal.GOOGLE_SCHOLAR]

            # Ensure proxy is set up for production use
            if config.requires_proxy and not self.google_scholar_proxy:
                logger.warning("Google Scholar requires proxy but none configured - may be blocked")

            # Perform search with error handling
            papers = []
            try:
                search_query = scholarly.search_pubs(query)

                count = 0
                for result in search_query:
                    if count >= max_results:
                        break

                    try:
                        paper = self._parse_google_scholar_result(result)
                        papers.append(paper)
                        count += 1

                        # Small delay between results to avoid blocking
                        await asyncio.sleep(2.0)

                    except Exception as e:
                        logger.warning(f"Failed to parse Google Scholar result: {e}")
                        continue

            except MaxTriesExceededException:
                logger.error("Google Scholar blocked requests - need better proxy configuration")
                raise
            except DOSException:
                logger.error("Google Scholar detected automated access - IP may be blocked")
                raise

            logger.info(f"Google Scholar search returned {len(papers)} papers for query: {query}")

            return {
                'success': True,
                'papers': papers,
                'total_found': len(papers),
                'source': 'google_scholar'
            }

        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
            raise

    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ArXiv XML response"""
        papers = []
        try:
            import xml.etree.ElementTree as ET

            # Parse XML
            root = ET.fromstring(xml_content)

            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}

            # Extract entries
            for entry in root.findall('atom:entry', ns):
                try:
                    # Extract basic fields
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip() if title_elem is not None else "Unknown Title"

                    summary_elem = entry.find('atom:summary', ns)
                    abstract = summary_elem.text.strip() if summary_elem is not None else ""

                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())

                    # Extract publication date
                    published_elem = entry.find('atom:published', ns)
                    pub_date = published_elem.text[:10] if published_elem is not None else "unknown"

                    # Extract ID/URL
                    id_elem = entry.find('atom:id', ns)
                    paper_url = id_elem.text if id_elem is not None else None

                    paper = {
                        'paper_id': paper_url.split('/')[-1] if paper_url else f"arxiv_{len(papers)}",
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'publication_date': pub_date,
                        'journal': 'arXiv',
                        'doi': None,
                        'url': paper_url,
                        'citations': 0,  # ArXiv doesn't provide citation counts
                        'source': 'arxiv'
                    }
                    papers.append(paper)

                except Exception as e:
                    logger.warning(f"Failed to parse ArXiv entry: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing ArXiv XML: {e}")

        return papers

    def _parse_semantic_scholar_json(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse Semantic Scholar JSON response"""
        papers = []
        try:
            for item in data.get('data', []):
                try:
                    # Extract authors
                    authors = []
                    for author in item.get('authors', []):
                        name = author.get('name', 'Unknown')
                        authors.append(name)

                    # Extract venue/journal
                    venue = item.get('venue') or item.get('journal', {}).get('name') or 'Unknown'

                    paper = {
                        'paper_id': item.get('paperId', f"ss_{len(papers)}"),
                        'title': item.get('title', 'Unknown Title'),
                        'authors': authors,
                        'abstract': item.get('abstract', ''),
                        'publication_date': str(item.get('year', 'unknown')),
                        'journal': venue,
                        'doi': None,  # Would need additional API call
                        'url': item.get('url'),
                        'citations': item.get('citationCount', 0),
                        'source': 'semantic_scholar'
                    }
                    papers.append(paper)

                except Exception as e:
                    logger.warning(f"Failed to parse Semantic Scholar item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar JSON: {e}")

        return papers

    def _parse_google_scholar_result(self, result: Dict) -> Dict[str, Any]:
        """Parse a single Google Scholar search result"""
        try:
            # Extract basic metadata from scholarly result
            bib = result.get('bib', {})

            title = bib.get('title', 'Unknown Title')

            # Handle authors
            authors = bib.get('author', [])
            if isinstance(authors, str):
                authors = [authors]
            elif not isinstance(authors, list):
                authors = []

            # Extract other fields
            abstract = bib.get('abstract', '')
            year = bib.get('pub_year')
            publication_date = str(year) if year else "unknown"

            journal = bib.get('venue', '')
            url = result.get('pub_url', '')
            citations = result.get('num_citations', 0)

            return {
                'paper_id': result.get('scholar_id', f"gs_{hash(title)}"),
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'publication_date': publication_date,
                'journal': journal,
                'doi': None,
                'url': url,
                'citations': citations,
                'source': 'google_scholar'
            }

        except Exception as e:
            logger.error(f"Error parsing Google Scholar result: {e}")
            raise

    async def search_all_portals(self, query: str, max_results_per_portal: int = 20) -> Dict[str, Any]:
        """
        Search all available research portals and aggregate results
        Using the working implementation with ArXiv, Semantic Scholar, and CrossRef
        """
        # Import and use the working simple research manager
        from .simple_enhanced_research_manager import simple_research_manager
        
        try:
            # Use the working simple manager that has no CAPTCHA issues
            papers = await simple_research_manager.search_all_portals(query, max_results_per_portal)
            
            # Convert to the expected format for academic research agent
            portal_results = {
                'arxiv': {'success': True, 'papers': [p for p in papers if p.get('source') == 'arxiv']},
                'semantic_scholar': {'success': True, 'papers': [p for p in papers if p.get('source') == 'semantic_scholar']},
                'crossref': {'success': True, 'papers': [p for p in papers if p.get('source') == 'crossref']}
            }
            
            return {
                'success': True,
                'query': query,
                'total_papers_found': len(papers),
                'total_raw_results': len(papers),
                'papers': papers,
                'portal_results': portal_results,
                'stats': self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Search all portals failed: {e}")
            return {
                'success': False,
                'query': query,
                'total_papers_found': 0,
                'total_raw_results': 0,
                'papers': [],
                'portal_results': {},
                'error': str(e)
            }

    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()

        for paper in papers:
            # Normalize title for comparison
            title_key = paper['title'].lower().strip()
            title_key = ''.join(c for c in title_key if c.isalnum() or c.isspace())
            title_key = ' '.join(title_key.split())  # Normalize whitespace

            if title_key not in seen_titles and len(title_key) > 5:  # Avoid very short titles
                seen_titles.add(title_key)
                unique_papers.append(paper)

        return unique_papers

    def _sort_papers_by_relevance(self, papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Sort papers by relevance to query"""
        def relevance_score(paper: Dict[str, Any]) -> float:
            score = 0.0
            query_words = set(query.lower().split())

            # Title relevance (highest weight)
            title_words = set(paper['title'].lower().split())
            title_matches = len(query_words.intersection(title_words))
            score += title_matches * 5

            # Abstract relevance
            if paper.get('abstract'):
                abstract_words = set(paper['abstract'].lower().split())
                abstract_matches = len(query_words.intersection(abstract_words))
                score += abstract_matches * 2

            # Citation boost (normalized)
            citations = paper.get('citations', 0)
            score += min(citations / 50, 5)  # Cap at 5 points

            # Recency boost
            try:
                year = int(paper.get('publication_date', '0')[:4])
                current_year = datetime.now().year
                if year >= current_year - 3:  # Last 3 years
                    score += 2
                elif year >= current_year - 5:  # Last 5 years
                    score += 1
            except (ValueError, TypeError):
                pass

            # Source diversity bonus
            source_weights = {
                'google_scholar': 1.2,  # Comprehensive coverage
                'semantic_scholar': 1.1,  # Good metadata
                'arxiv': 1.0  # Preprints
            }
            source = paper.get('source', 'unknown')
            score *= source_weights.get(source, 1.0)

            return score

        return sorted(papers, key=relevance_score, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self.stats,
            'circuit_breaker_states': {
                portal.value: {
                    'state': breaker.state,
                    'failure_count': breaker.failure_count,
                    'last_failure': breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
                }
                for portal, breaker in self.circuit_breakers.items()
            },
            'success_rate': (
                self.stats['successful_requests'] / max(self.stats['total_requests'], 1) * 100
            )
        }

    def configure_semantic_scholar_api_key(self, api_key: str):
        """Configure Semantic Scholar API key for higher rate limits"""
        config = self.portal_configs[ResearchPortal.SEMANTIC_SCHOLAR]
        config.api_key = api_key
        config.rate_limit_per_second = 100.0  # Higher limit with API key
        logger.info("Semantic Scholar API key configured - rate limit increased to 100 req/sec")

    def cleanup(self):
        """Clean up resources"""
        for session in self.session_pools.values():
            session.close()
        logger.info("Research portal manager cleaned up")
