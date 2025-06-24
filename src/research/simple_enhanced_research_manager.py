"""
Simple Enhanced Research Manager - Focused Fix

This is a simplified version that focuses on fixing the core issues:
1. ArXiv query formatting
2. Semantic Scholar rate limiting
3. Fallback mechanisms
"""

import asyncio
import logging
import time
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Any
import re
import sys
import os

logger = logging.getLogger(__name__)

# Try to import MCP server functions for Google Scholar
try:
    # Add MCP server path
    mcp_server_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mcp_servers', 'google-scholar')
    if os.path.exists(mcp_server_path):
        sys.path.insert(0, mcp_server_path)
        # Check if the module can be imported
        import importlib.util
        spec = importlib.util.find_spec("google_scholar_web_search")
        MCP_GOOGLE_SCHOLAR_AVAILABLE = spec is not None
        if MCP_GOOGLE_SCHOLAR_AVAILABLE:
            logger.info("Google Scholar MCP server functions available")
        else:
            logger.info("Google Scholar MCP server module not importable")
    else:
        MCP_GOOGLE_SCHOLAR_AVAILABLE = False
        logger.info("Google Scholar MCP server not found, will skip Google Scholar search")
except ImportError as e:
    MCP_GOOGLE_SCHOLAR_AVAILABLE = False
    logger.warning(f"Could not import Google Scholar MCP functions: {e}")


class SimpleEnhancedResearchManager:
    """Simple enhanced research manager with core fixes"""
    
    def __init__(self):
        self.last_request_times = {
            'arxiv': 0,
            'semantic_scholar': 0,
            'crossref': 0,  # Add CrossRef as third source
            'google_scholar': 0  # Add Google Scholar timing
        }
        self.failure_counts = {
            'arxiv': 0,
            'semantic_scholar': 0,
            'crossref': 0,  # Add CrossRef failure tracking
            'google_scholar': 0  # Add Google Scholar failure tracking
        }
        self.max_failures = 3
        
        logger.info("Simple Enhanced Research Manager initialized")
    
    async def search_all_portals(self, query: str, max_results_per_portal: int = 10) -> List[Dict[str, Any]]:
        """Search available portals with fallback handling"""
        all_results = []
        
        # Try ArXiv first
        try:
            if self.failure_counts['arxiv'] < self.max_failures:
                arxiv_results = await self._search_arxiv_fixed(query, max_results_per_portal)
                if arxiv_results.get('success', False):
                    all_results.extend(arxiv_results.get('papers', []))
                    self.failure_counts['arxiv'] = 0  # Reset on success
                else:
                    self.failure_counts['arxiv'] += 1
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            self.failure_counts['arxiv'] += 1
        
        # Try Semantic Scholar
        try:
            if self.failure_counts['semantic_scholar'] < self.max_failures:
                semantic_results = await self._search_semantic_scholar_fixed(query, max_results_per_portal)
                if semantic_results.get('success', False):
                    all_results.extend(semantic_results.get('papers', []))
                    self.failure_counts['semantic_scholar'] = 0  # Reset on success
                else:
                    self.failure_counts['semantic_scholar'] += 1
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            self.failure_counts['semantic_scholar'] += 1
          # Try CrossRef as third source (no CAPTCHAs!)
        try:
            if self.failure_counts['crossref'] < self.max_failures:
                crossref_results = await self._search_crossref_fixed(query, max_results_per_portal)
                if crossref_results.get('success', False):
                    all_results.extend(crossref_results.get('papers', []))
                    self.failure_counts['crossref'] = 0  # Reset on success
                else:
                    self.failure_counts['crossref'] += 1
        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            self.failure_counts['crossref'] += 1
          # Try Google Scholar via MCP server (if available)
        if MCP_GOOGLE_SCHOLAR_AVAILABLE:
            try:
                if self.failure_counts['google_scholar'] < self.max_failures:
                    # Add timeout to prevent hanging
                    gs_results = await asyncio.wait_for(
                        self._search_google_scholar_mcp(query, max_results_per_portal),
                        timeout=15.0  # 15 second timeout
                    )
                    if gs_results.get('success', False):
                        all_results.extend(gs_results.get('papers', []))
                        self.failure_counts['google_scholar'] = 0  # Reset on success
                    else:
                        self.failure_counts['google_scholar'] += 1
            except asyncio.TimeoutError:
                logger.warning("Google Scholar MCP search timed out, skipping")
                self.failure_counts['google_scholar'] += 1
            except Exception as e:
                logger.error(f"Google Scholar MCP search failed: {e}")
                self.failure_counts['google_scholar'] += 1
        else:
            logger.info("Google Scholar MCP server not available, skipping")
          # If no results, log warning but don't provide fake data
        if not all_results:
            logger.warning("No results from any portal - all searches failed or returned empty")
        
        logger.info(f"Search completed: {len(all_results)} total papers")
        return all_results
    
    async def _enforce_rate_limit(self, portal: str, min_interval: float):
        """Enforce rate limiting"""
        current_time = time.time()
        last_request = self.last_request_times.get(portal, 0)
        
        time_since_last = current_time - last_request
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.info(f"Rate limiting {portal}: waiting {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        
        self.last_request_times[portal] = time.time()
    
    async def _search_arxiv_fixed(self, query: str, max_results: int) -> Dict[str, Any]:
        """Fixed ArXiv search with proper query formatting"""
        try:
            # Rate limiting: 3 requests per second max
            await self._enforce_rate_limit('arxiv', 0.34)
            
            # Clean and format query for ArXiv
            clean_query = self._clean_arxiv_query(query)
            
            # Construct ArXiv query parameters
            params = {
                'search_query': clean_query,
                'start': 0,
                'max_results': min(max_results, 20),  # Conservative limit
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            logger.info(f"ArXiv search with query: {clean_query}")
            
            # Make request with proper headers
            headers = {
                'User-Agent': 'PyGent-Factory-Research/1.0 (mailto:research@example.com)'
            }
            
            response = requests.get(
                'http://export.arxiv.org/api/query',
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 400:
                # Try even simpler query format
                params['search_query'] = f'all:{query.replace(" ", "+")}'
                response = requests.get(
                    'http://export.arxiv.org/api/query',
                    params=params,
                    headers=headers,
                    timeout=30
                )
            
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_arxiv_xml_fixed(response.text)
            
            logger.info(f"ArXiv search returned {len(papers)} papers")
            
            return {
                'success': True,
                'papers': papers,
                'total_found': len(papers),
                'source': 'arxiv'
            }
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return {'success': False, 'error': str(e), 'papers': [], 'source': 'arxiv'}
    
    def _clean_arxiv_query(self, query: str) -> str:
        """Clean and format query for ArXiv API"""
        # Remove special characters that cause issues
        clean_query = re.sub(r'[^\w\s\-]', ' ', query)
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
    
    async def _search_semantic_scholar_fixed(self, query: str, max_results: int) -> Dict[str, Any]:
        """Fixed Semantic Scholar search with better rate limiting"""
        try:            # Very conservative rate limiting: 1 request per 15 seconds
            await self._enforce_rate_limit('semantic_scholar', 15.0)
            
            # Construct Semantic Scholar query
            params = {
                'query': query,
                'limit': min(max_results, 10),  # Very conservative limit
                'fields': 'paperId,title,authors,abstract,year,journal,citationCount,url,venue'
            }
            
            headers = {
                'User-Agent': 'PyGent-Factory-Research/1.0 (mailto:research@example.com)'
            }
            
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            
            logger.info(f"Semantic Scholar search with query: {query}")
            
            # Make request
            response = requests.get(url, params=params, headers=headers, timeout=45)
            
            if response.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit, skipping")
                return {'success': False, 'error': 'Rate limited', 'papers': [], 'source': 'semantic_scholar'}
            
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for paper_data in data.get('data', []):
                try:
                    authors = [author.get('name', 'Unknown') for author in paper_data.get('authors', [])]
                    
                    paper = {
                        'title': paper_data.get('title', 'Unknown Title'),
                        'abstract': paper_data.get('abstract', ''),
                        'authors': authors,
                        'url': paper_data.get('url', ''),
                        'year': str(paper_data.get('year', 'Unknown')),
                        'source': 'semantic_scholar',
                        'journal': paper_data.get('venue', paper_data.get('journal', {}).get('name', 'Unknown Journal')),
                        'citation_count': paper_data.get('citationCount', 0)
                    }
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse Semantic Scholar entry: {e}")
                    continue
            
            logger.info(f"Semantic Scholar search returned {len(papers)} papers")
            
            return {
                'success': True,
                'papers': papers,
                'total_found': len(papers),
                'source': 'semantic_scholar'
            }
            
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return {'success': False, 'error': str(e), 'papers': [], 'source': 'semantic_scholar'}
    
    async def _search_crossref_fixed(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search CrossRef API - no CAPTCHAs, free, comprehensive academic database"""
        try:
            # Conservative rate limiting: 1 request per 2 seconds
            await self._enforce_rate_limit('crossref', 2.0)
            
            # Construct CrossRef query
            params = {
                'query': query,
                'rows': min(max_results, 20),  # CrossRef allows up to 1000
                'mailto': 'research@pygent-factory.com',  # Polite pool for better performance
                'sort': 'relevance'
            }
            
            headers = {
                'User-Agent': 'PyGent-Factory-Research/1.0 (mailto:research@pygent-factory.com)'
            }
            
            url = "https://api.crossref.org/works"
            
            logger.info(f"CrossRef search with query: {query}")
            
            # Make request
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('message', {}).get('items', []):
                try:
                    # Extract title
                    title = ' '.join(item.get('title', []))
                    if not title:
                        continue
                    
                    # Extract authors
                    authors = []
                    for author in item.get('author', []):
                        given = author.get('given', '')
                        family = author.get('family', '')
                        if family:
                            full_name = f"{given} {family}".strip()
                            authors.append(full_name)
                    
                    # Extract publication year
                    year = 'Unknown'
                    if 'published-print' in item:
                        date_parts = item['published-print'].get('date-parts', [])
                        if date_parts and date_parts[0]:
                            year = str(date_parts[0][0])
                    elif 'published-online' in item:
                        date_parts = item['published-online'].get('date-parts', [])
                        if date_parts and date_parts[0]:
                            year = str(date_parts[0][0])
                    
                    # Extract journal/venue
                    journal = 'Unknown Journal'
                    if 'container-title' in item and item['container-title']:
                        journal = item['container-title'][0]
                    
                    # Build URL from DOI
                    url = ""
                    if 'DOI' in item:
                        url = f"https://doi.org/{item['DOI']}"
                    
                    paper = {
                        'title': title,
                        'abstract': item.get('abstract', ''),  # CrossRef often doesn't have abstracts
                        'authors': authors,
                        'url': url,
                        'year': year,
                        'source': 'crossref',
                        'journal': journal,
                        'doi': item.get('DOI', ''),
                        'citation_count': item.get('is-referenced-by-count', 0)
                    }
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse CrossRef entry: {e}")
                    continue
            
            logger.info(f"CrossRef search returned {len(papers)} papers")
            
            return {
                'success': True,
                'papers': papers,
                'total_found': len(papers),
                'source': 'crossref'
            }
            
        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            return {'success': False, 'error': str(e), 'papers': [], 'source': 'crossref'}
    
    def _clean_crossref_query(self, query: str) -> str:
        """Clean and format query for CrossRef API"""
        # Remove special characters that cause issues
        clean_query = re.sub(r'[^\w\s\-]', ' ', query)
        clean_query = ' '.join(clean_query.split())  # Normalize whitespace
        
        return clean_query
    
    async def _search_google_scholar_mcp(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search Google Scholar using the MCP server (if available)"""
        if not MCP_GOOGLE_SCHOLAR_AVAILABLE:
            return {'success': False, 'error': 'MCP server not available', 'papers': [], 'source': 'google_scholar_mcp'}
        
        try:
            # More reasonable rate limiting for Google Scholar: 1 request per 5 seconds
            await self._enforce_rate_limit('google_scholar', 5.0)
            
            logger.info(f"Google Scholar MCP search with query: {query}")
            
            # Import the search function dynamically to avoid import errors
            try:
                from google_scholar_web_search import google_scholar_search
            except ImportError:
                logger.error("Google Scholar MCP module not available")
                return {'success': False, 'error': 'MCP module not available', 'papers': [], 'source': 'google_scholar_mcp'}
            
            # Use the MCP server's search function
            raw_results = await asyncio.to_thread(
                google_scholar_search, 
                query, 
                min(max_results, 5)  # Conservative limit
            )
            
            # Convert MCP results to our standard format
            papers = []
            for result in raw_results:
                paper = {
                    'title': result.get('title', 'Unknown Title'),
                    'abstract': result.get('snippet', ''),
                    'authors': result.get('authors', ['Unknown']),
                    'url': result.get('link', ''),
                    'year': str(result.get('year', 'Unknown')),
                    'source': 'google_scholar_mcp',
                    'journal': result.get('publication', 'Unknown Journal'),
                    'citation_count': result.get('citations', 0)
                }
                papers.append(paper)
            
            logger.info(f"Google Scholar MCP returned {len(papers)} papers")
            
            return {
                'success': True,
                'papers': papers,
                'total_found': len(papers),
                'source': 'google_scholar_mcp'
            }
            
        except Exception as e:
            logger.error(f"Google Scholar MCP search failed: {e}")
            return {'success': False, 'error': str(e), 'papers': [], 'source': 'google_scholar_mcp'}
            
            # Convert MCP results to our standard format
            papers = []
            for result in raw_results:
                try:
                    # Extract authors from the combined authors string
                    authors_text = result.get('Authors', '')
                    authors = []
                    if authors_text and authors_text != 'No authors available':
                        # Try to parse author names from the string
                        # Google Scholar format is often "Author1, Author2 - Journal, Year"
                        if ' - ' in authors_text:
                            authors_part = authors_text.split(' - ')[0]
                            authors = [name.strip() for name in authors_part.split(',')]
                        else:
                            authors = [authors_text]
                    
                    # Extract year from authors string if possible
                    year = 'Unknown'
                    if ' - ' in authors_text:
                        info_part = authors_text.split(' - ', 1)[1]
                        # Look for 4-digit year
                        import re
                        year_match = re.search(r'\b(19|20)\d{2}\b', info_part)
                        if year_match:
                            year = year_match.group()
                    
                    paper = {
                        'title': result.get('Title', 'Unknown Title'),
                        'abstract': result.get('Abstract', ''),
                        'authors': authors,
                        'url': result.get('URL', ''),
                        'year': year,
                        'source': 'google_scholar_mcp',
                        'journal': 'Google Scholar',
                        'citation_count': 0  # MCP server doesn't provide citation counts
                    }
                    
                    # Only add if we have a valid title
                    if paper['title'] and paper['title'] != 'No title available':
                        papers.append(paper)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse Google Scholar MCP result: {e}")
                    continue
            
            logger.info(f"Google Scholar MCP search returned {len(papers)} papers")
            
            return {
                'success': True,
                'papers': papers,
                'total_found': len(papers),
                'source': 'google_scholar_mcp'
            }
            
        except Exception as e:
            logger.error(f"Google Scholar MCP search failed: {e}")
            return {'success': False, 'error': str(e), 'papers': [], 'source': 'google_scholar_mcp'}

# Create a singleton instance
simple_research_manager = SimpleEnhancedResearchManager()
