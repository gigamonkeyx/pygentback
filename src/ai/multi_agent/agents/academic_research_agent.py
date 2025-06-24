"""
Academic Research Agent

Enhanced research agent with real academic database integration capabilities.
Supports multiple academic sources including arXiv, Google Scholar, PubMed, and Semantic Scholar.
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import the robust research portal manager
from src.research.robust_research_portal_manager import RobustResearchPortalManager, ResearchPortal

logger = logging.getLogger(__name__)


class AcademicDatabase(Enum):
    """Supported academic databases"""
    ARXIV = "arxiv"
    GOOGLE_SCHOLAR = "google_scholar"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"


@dataclass
class AcademicPaper:
    """Academic paper metadata"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    publication_date: str
    journal: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    citations: int
    database_source: AcademicDatabase
    keywords: List[str]
    full_text_available: bool = False
    
    def is_recent(self, years: int = 5) -> bool:
        """Check if paper is recent (within specified years)"""
        try:
            pub_year = int(self.publication_date[:4])
            current_year = datetime.now().year
            return (current_year - pub_year) <= years
        except:
            return False


class AcademicResearchAgent:
    """
    Enhanced research agent with real academic database access.
    
    Capabilities:
    - Multi-database academic search
    - Literature review automation
    - Citation analysis
    - Research gap identification
    - Primary source discovery
    """
    
    def __init__(self, agent_id: str = "academic_research_agent"):
        self.agent_id = agent_id
        self.agent_type = "academic_research"
        self.status = "initialized"
        self.capabilities = [
            "multi_database_search",
            "literature_review",
            "citation_analysis",
            "research_gap_identification",
            "primary_source_discovery"
        ]
        
        # Academic database configurations with proper rate limiting
        self.database_configs = {
            AcademicDatabase.ARXIV: {
                'base_url': 'http://export.arxiv.org/api/query',
                'rate_limit': 1,  # 1 request per second (conservative)
                'max_results': 50,
                'retry_attempts': 3,
                'retry_delay': 2.0
            },
            AcademicDatabase.SEMANTIC_SCHOLAR: {
                'base_url': 'https://api.semanticscholar.org/graph/v1',
                'rate_limit': 0.2,  # 1 request per 5 seconds (very conservative for free tier)
                'max_results': 20,  # Reduced to minimize API calls
                'retry_attempts': 3,
                'retry_delay': 10.0,  # Longer delay for 429 errors
                'headers': {
                    'User-Agent': 'PyGent-Factory-Research-Agent/1.0 (academic-research)'
                }
            },
            AcademicDatabase.CROSSREF: {
                'base_url': 'https://api.crossref.org/works',
                'rate_limit': 1,  # 1 request per second
                'max_results': 50,
                'retry_attempts': 3,
                'retry_delay': 1.0,
                'headers': {
                    'User-Agent': 'PyGent-Factory-Research-Agent/1.0 (mailto:research@pygent.ai)'
                }
            },
            AcademicDatabase.GOOGLE_SCHOLAR: {
                'rate_limit': 0.1,  # 1 request per 10 seconds (very conservative)
                'max_results': 20,
                'retry_attempts': 2,
                'retry_delay': 15.0,  # Longer delay for Google Scholar
                'delay_between_requests': 10.0
            }
        }

        # Rate limiting state
        self.last_request_time = {db: 0.0 for db in AcademicDatabase}
        self.request_counts = {db: 0 for db in AcademicDatabase}
        
        # Research state
        self.active_searches: Dict[str, Dict[str, Any]] = {}
        self.paper_cache: Dict[str, AcademicPaper] = {}
        self.search_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.config = {
            'max_papers_per_search': 50,
            'min_citation_threshold': 5,
            'preferred_publication_years': 10,
            'search_timeout_seconds': 30,
            'enable_full_text_search': False,
            'preferred_languages': ['en'],
            'quality_filters': {
                'min_citations': 1,
                'peer_reviewed_only': True,
                'exclude_preprints': False
            }
        }
        
        # Statistics
        self.stats = {
            'searches_performed': 0,
            'papers_retrieved': 0,
            'databases_queried': 0,
            'literature_reviews_completed': 0,
            'avg_search_time_ms': 0.0,
            'successful_searches': 0
        }

        # Initialize robust research portal manager
        self.portal_manager = RobustResearchPortalManager()

        logger.info(f"AcademicResearchAgent {agent_id} initialized with robust portal manager")
    
    async def start(self) -> bool:
        """Start the academic research agent"""
        try:
            self.status = "active"
            
            # Test database connections
            await self._test_database_connections()
            
            logger.info(f"AcademicResearchAgent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start AcademicResearchAgent {self.agent_id}: {e}")
            self.status = "error"
            return False
    
    async def stop(self) -> bool:
        """Stop the academic research agent"""
        try:
            # Cleanup portal manager resources
            if hasattr(self, 'portal_manager'):
                self.portal_manager.cleanup()

            self.status = "stopped"
            logger.info(f"AcademicResearchAgent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop AcademicResearchAgent {self.agent_id}: {e}")
            return False

    def configure_semantic_scholar_api_key(self, api_key: str):
        """Configure Semantic Scholar API key for higher rate limits"""
        if hasattr(self, 'portal_manager'):
            self.portal_manager.configure_semantic_scholar_api_key(api_key)
            logger.info("Semantic Scholar API key configured for enhanced access")
        else:
            logger.warning("Portal manager not initialized - cannot configure API key")

    def get_portal_stats(self) -> Dict[str, Any]:
        """Get statistics from the robust portal manager"""
        if hasattr(self, 'portal_manager'):
            return self.portal_manager.get_stats()
        else:
            return {}
    
    async def conduct_literature_review(self, topic: str, research_question: str, 
                                      **kwargs) -> Dict[str, Any]:
        """
        Conduct comprehensive literature review using multiple academic databases.
        
        Args:
            topic: Research topic
            research_question: Specific research question
            **kwargs: Additional parameters (databases, years, etc.)
            
        Returns:
            Comprehensive literature review results
        """
        start_time = datetime.utcnow()
        search_id = f"lit_review_{int(start_time.timestamp())}"
        
        try:
            # Initialize literature review
            self.active_searches[search_id] = {
                'topic': topic,
                'research_question': research_question,
                'start_time': start_time,
                'phase': 'initialization',
                'databases': kwargs.get('databases', [AcademicDatabase.ARXIV, AcademicDatabase.SEMANTIC_SCHOLAR, AcademicDatabase.GOOGLE_SCHOLAR]),
                'years': kwargs.get('years', 10),
                'max_papers': kwargs.get('max_papers', 50)
            }
            
            self.stats['searches_performed'] += 1
            
            # Phase 1: Multi-database search
            search_results = await self._multi_database_search(search_id, topic, research_question)
            self.active_searches[search_id]['phase'] = 'analysis'
            
            # Phase 2: Paper analysis and filtering
            analyzed_papers = await self._analyze_papers(search_id, search_results)
            self.active_searches[search_id]['phase'] = 'synthesis'
            
            # Phase 3: Literature synthesis
            literature_synthesis = await self._synthesize_literature(search_id, analyzed_papers)
            self.active_searches[search_id]['phase'] = 'gap_analysis'
            
            # Phase 4: Research gap identification
            research_gaps = await self._identify_research_gaps(search_id, analyzed_papers, literature_synthesis)
            self.active_searches[search_id]['phase'] = 'reporting'
            
            # Phase 5: Generate literature review report
            final_report = await self._generate_literature_report(
                search_id, analyzed_papers, literature_synthesis, research_gaps
            )
            
            # Calculate search time
            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_search_stats(True, search_time, len(analyzed_papers))
            
            # Clean up active search
            del self.active_searches[search_id]
            
            logger.info(f"Literature review completed for '{topic}' in {search_time:.2f}ms")
            return final_report
            
        except Exception as e:
            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_search_stats(False, search_time, 0)
            
            logger.error(f"Literature review failed for '{topic}': {e}")
            
            # Clean up active search
            if search_id in self.active_searches:
                del self.active_searches[search_id]
            
            return {
                'success': False,
                'error': str(e),
                'search_id': search_id,
                'topic': topic,
                'search_time_ms': search_time
            }
    
    async def search_academic_papers(self, query: str, databases: List[AcademicDatabase] = None,
                                   **kwargs) -> List[AcademicPaper]:
        """Search for academic papers across multiple databases using robust portal manager"""
        try:
            max_results = kwargs.get('max_results', 50)
            max_per_portal = max_results // 3  # Distribute across 3 main portals

            # Use the robust portal manager for comprehensive search
            result = await self.portal_manager.search_all_portals(query, max_per_portal)

            if result['success']:
                # Convert portal manager results to AcademicPaper objects
                papers = []
                for paper_data in result['papers']:
                    paper = self._convert_portal_result_to_academic_paper(paper_data)
                    papers.append(paper)

                # Update statistics
                self.stats['databases_queried'] += len([r for r in result['portal_results'].values() if r['success']])
                self.stats['papers_retrieved'] += len(papers)

                logger.info(f"Robust search returned {len(papers)} papers for query: {query}")
                return papers[:max_results]
            else:
                logger.error(f"Robust search failed for query: {query}")
                return []

        except Exception as e:
            logger.error(f"Academic paper search failed: {e}")
            return []
    
    def _convert_portal_result_to_academic_paper(self, paper_data: Dict[str, Any]) -> AcademicPaper:
        """Convert portal manager result to AcademicPaper object"""
        # Map source to AcademicDatabase enum
        source_mapping = {
            'arxiv': AcademicDatabase.ARXIV,
            'semantic_scholar': AcademicDatabase.SEMANTIC_SCHOLAR,
            'google_scholar': AcademicDatabase.GOOGLE_SCHOLAR
        }

        source = paper_data.get('source', 'unknown')
        database_source = source_mapping.get(source, AcademicDatabase.ARXIV)

        return AcademicPaper(
            paper_id=paper_data.get('paper_id', ''),
            title=paper_data.get('title', 'Unknown Title'),
            authors=paper_data.get('authors', []),
            abstract=paper_data.get('abstract', ''),
            publication_date=paper_data.get('publication_date', 'unknown'),
            journal=paper_data.get('journal'),
            doi=paper_data.get('doi'),
            url=paper_data.get('url'),
            citations=paper_data.get('citations', 0),
            database_source=database_source,
            keywords=paper_data.get('keywords', [])
        )

    async def _multi_database_search(self, search_id: str, topic: str,
                                   research_question: str) -> List[AcademicPaper]:
        """Perform search across multiple academic databases using robust portal manager"""
        search_config = self.active_searches[search_id]

        # Generate search queries
        queries = self._generate_search_queries(topic, research_question)

        all_papers = []
        max_per_query = search_config['max_papers'] // len(queries)

        for query in queries:
            try:
                # Use robust portal manager for each query
                result = await self.portal_manager.search_all_portals(query, max_per_query)

                if result['success']:
                    # Convert results to AcademicPaper objects
                    for paper_data in result['papers']:
                        paper = self._convert_portal_result_to_academic_paper(paper_data)
                        all_papers.append(paper)

                    logger.info(f"Query '{query}' returned {len(result['papers'])} papers")
                else:
                    logger.warning(f"Search failed for query '{query}'")

            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")

        return self._deduplicate_papers(all_papers)
    
    async def _search_database(self, database: AcademicDatabase, query: str,
                             **kwargs) -> List[AcademicPaper]:
        """Search a specific academic database"""
        if database == AcademicDatabase.ARXIV:
            return await self._search_arxiv(query, **kwargs)
        elif database == AcademicDatabase.SEMANTIC_SCHOLAR:
            return await self._search_semantic_scholar(query, **kwargs)
        elif database == AcademicDatabase.CROSSREF:
            return await self._search_crossref(query, **kwargs)
        elif database == AcademicDatabase.GOOGLE_SCHOLAR:
            return await self._search_google_scholar(query, **kwargs)
        else:
            logger.warning(f"Database {database.value} not implemented")
            return []
    
    async def _search_arxiv(self, query: str, **kwargs) -> List[AcademicPaper]:
        """Search arXiv database"""
        try:
            config = self.database_configs[AcademicDatabase.ARXIV]
            max_results = min(kwargs.get('max_results', 20), config['max_results'])
            
            # Construct arXiv API query
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(config['base_url'], params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_arxiv_response(content)
                    else:
                        logger.error(f"arXiv search failed with status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []
    
    async def _search_semantic_scholar(self, query: str, **kwargs) -> List[AcademicPaper]:
        """Search Semantic Scholar database"""
        try:
            config = self.database_configs[AcademicDatabase.SEMANTIC_SCHOLAR]
            max_results = min(kwargs.get('max_results', 20), config['max_results'])
            
            # Construct Semantic Scholar API query
            url = f"{config['base_url']}/paper/search"
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'paperId,title,authors,abstract,year,journal,citationCount,url'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_semantic_scholar_response(data)
                    else:
                        logger.error(f"Semantic Scholar search failed with status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []
    
    async def _search_crossref(self, query: str, **kwargs) -> List[AcademicPaper]:
        """Search CrossRef database"""
        try:
            config = self.database_configs[AcademicDatabase.CROSSREF]
            max_results = min(kwargs.get('max_results', 20), config['max_results'])
            
            # Construct CrossRef API query
            params = {
                'query': query,
                'rows': max_results,
                'sort': 'relevance',
                'order': 'desc'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(config['base_url'], params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_crossref_response(data)
                    else:
                        logger.error(f"CrossRef search failed with status {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"CrossRef search error: {e}")
            return []

    async def _search_google_scholar(self, query: str, **kwargs) -> List[AcademicPaper]:
        """Search Google Scholar using the scholarly library"""
        try:
            config = self.database_configs[AcademicDatabase.GOOGLE_SCHOLAR]
            max_results = min(kwargs.get('max_results', 10), config['max_results'])

            # Import scholarly library
            try:
                import scholarly
            except ImportError:
                logger.error("Scholarly library not available - install with: pip install scholarly")
                return []

            # Rate limiting check
            import time
            current_time = time.time()
            last_request = self.last_request_time[AcademicDatabase.GOOGLE_SCHOLAR]

            if last_request > 0:
                elapsed = current_time - last_request
                min_delay = config['delay_between_requests']
                if elapsed < min_delay:
                    wait_time = min_delay - elapsed
                    logger.info(f"Rate limiting: waiting {wait_time:.1f}s for Google Scholar")
                    await asyncio.sleep(wait_time)

            # Update last request time
            self.last_request_time[AcademicDatabase.GOOGLE_SCHOLAR] = time.time()

            # Perform search
            papers = []
            search_query = scholarly.search_pubs(query)

            count = 0
            for result in search_query:
                if count >= max_results:
                    break

                try:
                    # Extract paper metadata
                    paper = self._parse_google_scholar_result(result)
                    papers.append(paper)
                    count += 1

                    # Small delay between results
                    await asyncio.sleep(1.0)

                except Exception as e:
                    logger.warning(f"Failed to parse Google Scholar result: {e}")
                    continue

            logger.info(f"Google Scholar search returned {len(papers)} papers for query: {query}")
            return papers

        except Exception as e:
            logger.error(f"Google Scholar search error: {e}")
            return []

    def _parse_google_scholar_result(self, result: Dict) -> AcademicPaper:
        """Parse a single Google Scholar search result"""
        # Extract basic metadata
        bib = result.get('bib', {})
        title = bib.get('title', 'Unknown Title')
        authors = bib.get('author', [])
        if isinstance(authors, str):
            authors = [authors]

        abstract = bib.get('abstract', '')
        year = bib.get('pub_year')
        if year:
            try:
                year = int(year)
                publication_date = str(year)
            except (ValueError, TypeError):
                publication_date = "unknown"
        else:
            publication_date = "unknown"

        journal = bib.get('venue', '')
        url = result.get('pub_url', '')
        citations = result.get('num_citations', 0)

        return AcademicPaper(
            paper_id=result.get('scholar_id', f"gs_{hash(title)}"),
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=publication_date,
            journal=journal,
            doi=None,  # Google Scholar doesn't always provide DOI
            url=url,
            citations=citations,
            database_source=AcademicDatabase.GOOGLE_SCHOLAR,
            keywords=[]
        )

    def _parse_arxiv_response(self, xml_content: str) -> List[AcademicPaper]:
        """Parse arXiv XML response"""
        papers = []
        try:
            # Simple XML parsing for arXiv (would use proper XML parser in production)
            # This is a simplified implementation
            import re

            # Extract entries from XML
            entries = re.findall(r'<entry>(.*?)</entry>', xml_content, re.DOTALL)

            for entry in entries:
                # Extract basic fields
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                authors_matches = re.findall(r'<name>(.*?)</name>', entry)
                published_match = re.search(r'<published>(.*?)</published>', entry)
                id_match = re.search(r'<id>(.*?)</id>', entry)

                if title_match and summary_match:
                    paper = AcademicPaper(
                        paper_id=id_match.group(1) if id_match else f"arxiv_{len(papers)}",
                        title=title_match.group(1).strip(),
                        authors=authors_matches,
                        abstract=summary_match.group(1).strip(),
                        publication_date=published_match.group(1)[:10] if published_match else "unknown",
                        journal="arXiv",
                        doi=None,
                        url=id_match.group(1) if id_match else None,
                        citations=0,  # arXiv doesn't provide citation counts
                        database_source=AcademicDatabase.ARXIV,
                        keywords=[]
                    )
                    papers.append(paper)

        except Exception as e:
            logger.error(f"Error parsing arXiv response: {e}")

        return papers

    def _parse_semantic_scholar_response(self, data: Dict) -> List[AcademicPaper]:
        """Parse Semantic Scholar JSON response"""
        papers = []
        try:
            for item in data.get('data', []):
                authors = [author.get('name', 'Unknown') for author in item.get('authors', [])]

                paper = AcademicPaper(
                    paper_id=item.get('paperId', f"ss_{len(papers)}"),
                    title=item.get('title', 'Unknown Title'),
                    authors=authors,
                    abstract=item.get('abstract', ''),
                    publication_date=str(item.get('year', 'unknown')),
                    journal=item.get('journal', {}).get('name') if item.get('journal') else None,
                    doi=None,
                    url=item.get('url'),
                    citations=item.get('citationCount', 0),
                    database_source=AcademicDatabase.SEMANTIC_SCHOLAR,
                    keywords=[]
                )
                papers.append(paper)

        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar response: {e}")

        return papers

    def _parse_crossref_response(self, data: Dict) -> List[AcademicPaper]:
        """Parse CrossRef JSON response"""
        papers = []
        try:
            for item in data.get('message', {}).get('items', []):
                # Extract authors
                authors = []
                for author in item.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    full_name = f"{given} {family}".strip()
                    if full_name:
                        authors.append(full_name)

                # Extract publication date
                pub_date = "unknown"
                if 'published-print' in item:
                    date_parts = item['published-print'].get('date-parts', [[]])[0]
                    if date_parts:
                        pub_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}" if len(date_parts) >= 3 else str(date_parts[0])

                paper = AcademicPaper(
                    paper_id=item.get('DOI', f"crossref_{len(papers)}"),
                    title=' '.join(item.get('title', ['Unknown Title'])),
                    authors=authors,
                    abstract=item.get('abstract', ''),
                    publication_date=pub_date,
                    journal=' '.join(item.get('container-title', [])),
                    doi=item.get('DOI'),
                    url=item.get('URL'),
                    citations=item.get('is-referenced-by-count', 0),
                    database_source=AcademicDatabase.CROSSREF,
                    keywords=[]
                )
                papers.append(paper)

        except Exception as e:
            logger.error(f"Error parsing CrossRef response: {e}")

        return papers

    def _generate_search_queries(self, topic: str, research_question: str) -> List[str]:
        """Generate multiple search queries for comprehensive coverage"""
        queries = [
            topic,
            research_question,
            f"{topic} history",
            f"{topic} global perspective",
            f"{topic} primary sources",
            f"{topic} cross-cultural",
            f"{topic} comparative study"
        ]

        # Remove duplicates and empty queries
        return list(set([q.strip() for q in queries if q.strip()]))

    def _deduplicate_papers(self, papers: List[AcademicPaper]) -> List[AcademicPaper]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()

        for paper in papers:
            # Simple deduplication based on title
            title_key = paper.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)

        return unique_papers

    def _sort_papers_by_relevance(self, papers: List[AcademicPaper], query: str) -> List[AcademicPaper]:
        """Sort papers by relevance to query"""
        def relevance_score(paper: AcademicPaper) -> float:
            score = 0.0
            query_words = query.lower().split()

            # Title relevance
            title_words = paper.title.lower().split()
            title_matches = sum(1 for word in query_words if word in title_words)
            score += title_matches * 3

            # Abstract relevance
            if paper.abstract:
                abstract_words = paper.abstract.lower().split()
                abstract_matches = sum(1 for word in query_words if word in abstract_words)
                score += abstract_matches

            # Citation boost
            score += min(paper.citations / 100, 2)  # Cap citation boost at 2 points

            # Recency boost
            if paper.is_recent(5):
                score += 1

            return score

        return sorted(papers, key=relevance_score, reverse=True)

    async def _analyze_papers(self, search_id: str, papers: List[AcademicPaper]) -> List[AcademicPaper]:
        """Analyze and filter papers based on quality criteria"""
        analyzed_papers = []

        for paper in papers:
            # Apply quality filters
            if self._meets_quality_criteria(paper):
                # Cache paper
                self.paper_cache[paper.paper_id] = paper
                analyzed_papers.append(paper)

        self.stats['papers_retrieved'] += len(analyzed_papers)
        return analyzed_papers

    def _meets_quality_criteria(self, paper: AcademicPaper) -> bool:
        """Check if paper meets quality criteria"""
        criteria = self.config['quality_filters']

        # Citation threshold
        if paper.citations < criteria['min_citations']:
            return False

        # Language filter
        if self.config['preferred_languages'] and 'en' not in self.config['preferred_languages']:
            # Would implement language detection here
            pass

        return True

    async def _synthesize_literature(self, search_id: str, papers: List[AcademicPaper]) -> Dict[str, Any]:
        """Synthesize literature findings"""
        synthesis = {
            'total_papers': len(papers),
            'key_themes': self._extract_themes_from_papers(papers),
            'temporal_distribution': self._analyze_temporal_distribution(papers),
            'citation_analysis': self._analyze_citations(papers),
            'author_analysis': self._analyze_authors(papers),
            'journal_distribution': self._analyze_journals(papers),
            'database_coverage': self._analyze_database_coverage(papers)
        }

        return synthesis

    def _extract_themes_from_papers(self, papers: List[AcademicPaper]) -> List[str]:
        """Extract key themes from paper titles and abstracts"""
        # Simple keyword extraction (would use NLP in production)
        all_text = ' '.join([paper.title + ' ' + paper.abstract for paper in papers])
        words = all_text.lower().split()

        # Count word frequency (excluding common words)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Return top themes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]

    def _analyze_temporal_distribution(self, papers: List[AcademicPaper]) -> Dict[str, int]:
        """Analyze temporal distribution of papers"""
        year_counts = {}
        for paper in papers:
            try:
                year = paper.publication_date[:4]
                year_counts[year] = year_counts.get(year, 0) + 1
            except:
                year_counts['unknown'] = year_counts.get('unknown', 0) + 1

        return year_counts

    def _analyze_citations(self, papers: List[AcademicPaper]) -> Dict[str, Any]:
        """Analyze citation patterns"""
        citations = [paper.citations for paper in papers]
        return {
            'total_citations': sum(citations),
            'avg_citations': sum(citations) / len(citations) if citations else 0,
            'max_citations': max(citations) if citations else 0,
            'highly_cited_papers': len([c for c in citations if c > 100])
        }

    def _analyze_authors(self, papers: List[AcademicPaper]) -> Dict[str, int]:
        """Analyze author frequency"""
        author_counts = {}
        for paper in papers:
            for author in paper.authors:
                author_counts[author] = author_counts.get(author, 0) + 1

        # Return top authors
        sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_authors[:10])

    def _analyze_journals(self, papers: List[AcademicPaper]) -> Dict[str, int]:
        """Analyze journal distribution"""
        journal_counts = {}
        for paper in papers:
            if paper.journal:
                journal_counts[paper.journal] = journal_counts.get(paper.journal, 0) + 1

        return journal_counts

    def _analyze_database_coverage(self, papers: List[AcademicPaper]) -> Dict[str, int]:
        """Analyze coverage across databases"""
        db_counts = {}
        for paper in papers:
            db_name = paper.database_source.value
            db_counts[db_name] = db_counts.get(db_name, 0) + 1

        return db_counts

    async def _identify_research_gaps(self, search_id: str, papers: List[AcademicPaper],
                                    synthesis: Dict[str, Any]) -> List[str]:
        """Identify research gaps based on literature analysis"""
        gaps = []

        # Temporal gaps
        temporal_dist = synthesis['temporal_distribution']
        recent_papers = sum(count for year, count in temporal_dist.items()
                          if year.isdigit() and int(year) >= 2020)
        if recent_papers < len(papers) * 0.3:
            gaps.append("Limited recent research (post-2020) on this topic")

        # Geographic/cultural gaps
        themes = synthesis['key_themes']
        global_keywords = ['global', 'international', 'cross-cultural', 'comparative', 'non-western']
        global_coverage = sum(1 for theme in themes if any(keyword in theme.lower() for keyword in global_keywords))
        if global_coverage < 2:
            gaps.append("Limited global or cross-cultural perspectives in current literature")

        # Methodological gaps
        method_keywords = ['primary', 'sources', 'archival', 'ethnographic', 'qualitative']
        method_coverage = sum(1 for theme in themes if any(keyword in theme.lower() for keyword in method_keywords))
        if method_coverage < 1:
            gaps.append("Limited use of primary sources or qualitative methodologies")

        # Citation gaps
        citation_analysis = synthesis['citation_analysis']
        if citation_analysis['highly_cited_papers'] < len(papers) * 0.1:
            gaps.append("Few highly-cited foundational works, indicating emerging field")

        return gaps

    async def _generate_literature_report(self, search_id: str, papers: List[AcademicPaper],
                                        synthesis: Dict[str, Any], gaps: List[str]) -> Dict[str, Any]:
        """Generate comprehensive literature review report"""
        search_config = self.active_searches[search_id]

        # Select top papers for detailed analysis
        top_papers = sorted(papers, key=lambda p: p.citations, reverse=True)[:10]

        report = {
            'search_id': search_id,
            'success': True,
            'topic': search_config['topic'],
            'research_question': search_config['research_question'],
            'literature_summary': {
                'total_papers_found': len(papers),
                'databases_searched': len(search_config['databases']),
                'search_queries_used': len(self._generate_search_queries(search_config['topic'], search_config['research_question'])),
                'key_themes': synthesis['key_themes'],
                'temporal_coverage': synthesis['temporal_distribution'],
                'top_authors': synthesis['author_analysis'],
                'journal_distribution': synthesis['journal_distribution']
            },
            'key_findings': self._generate_key_findings(papers, synthesis),
            'research_gaps': gaps,
            'recommended_sources': [
                {
                    'title': paper.title,
                    'authors': ', '.join(paper.authors),
                    'year': paper.publication_date[:4],
                    'journal': paper.journal,
                    'citations': paper.citations,
                    'url': paper.url,
                    'relevance_reason': self._explain_relevance(paper, search_config['topic'])
                }
                for paper in top_papers
            ],
            'methodology_recommendations': self._generate_methodology_recommendations(synthesis, gaps),
            'future_research_directions': self._suggest_future_research(synthesis, gaps),
            'confidence_score': self._calculate_literature_confidence(papers, synthesis),
            'metadata': {
                'search_agent': self.agent_id,
                'completion_time': datetime.utcnow().isoformat(),
                'databases_coverage': synthesis['database_coverage'],
                'citation_analysis': synthesis['citation_analysis']
            }
        }

        self.stats['literature_reviews_completed'] += 1
        return report

    def _generate_key_findings(self, papers: List[AcademicPaper], synthesis: Dict[str, Any]) -> List[str]:
        """Generate key findings from literature analysis"""
        findings = []

        # Theme-based findings
        themes = synthesis['key_themes'][:5]
        if themes:
            findings.append(f"Major research themes include: {', '.join(themes)}")

        # Temporal findings
        temporal_dist = synthesis['temporal_distribution']
        if temporal_dist:
            peak_year = max(temporal_dist.items(), key=lambda x: x[1] if x[0].isdigit() else 0)
            findings.append(f"Research activity peaked around {peak_year[0]} with {peak_year[1]} publications")

        # Citation findings
        citation_analysis = synthesis['citation_analysis']
        if citation_analysis['highly_cited_papers'] > 0:
            findings.append(f"Field includes {citation_analysis['highly_cited_papers']} highly-cited works (>100 citations)")

        # Author findings
        top_authors = list(synthesis['author_analysis'].keys())[:3]
        if top_authors:
            findings.append(f"Leading researchers include: {', '.join(top_authors)}")

        return findings

    def _explain_relevance(self, paper: AcademicPaper, topic: str) -> str:
        """Explain why a paper is relevant to the research topic"""
        reasons = []

        topic_words = topic.lower().split()
        title_words = paper.title.lower().split()

        # Direct topic match
        matches = [word for word in topic_words if word in title_words]
        if matches:
            reasons.append(f"Direct topic relevance: {', '.join(matches)}")

        # High citations
        if paper.citations > 100:
            reasons.append(f"Highly cited ({paper.citations} citations)")

        # Recent work
        if paper.is_recent(3):
            reasons.append("Recent research")

        return "; ".join(reasons) if reasons else "Thematically relevant"

    def _generate_methodology_recommendations(self, synthesis: Dict[str, Any], gaps: List[str]) -> List[str]:
        """Generate methodology recommendations based on gaps"""
        recommendations = []

        for gap in gaps:
            if "global" in gap.lower() or "cross-cultural" in gap.lower():
                recommendations.append("Conduct comparative studies across multiple cultural contexts")
            elif "primary sources" in gap.lower():
                recommendations.append("Incorporate archival research and primary source analysis")
            elif "recent research" in gap.lower():
                recommendations.append("Focus on contemporary developments and recent case studies")
            elif "qualitative" in gap.lower():
                recommendations.append("Employ ethnographic or qualitative research methods")

        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "Consider mixed-methods approach combining quantitative and qualitative analysis",
                "Ensure global perspective by including non-Western sources and viewpoints",
                "Incorporate primary source analysis where possible"
            ])

        return recommendations

    def _suggest_future_research(self, synthesis: Dict[str, Any], gaps: List[str]) -> List[str]:
        """Suggest future research directions"""
        suggestions = []

        # Based on gaps
        for gap in gaps:
            if "global" in gap.lower():
                suggestions.append("Expand research to include non-Western perspectives and case studies")
            elif "recent" in gap.lower():
                suggestions.append("Investigate contemporary developments and their historical connections")
            elif "primary sources" in gap.lower():
                suggestions.append("Develop comprehensive primary source databases for the field")

        # Based on themes
        themes = synthesis['key_themes'][:3]
        for theme in themes:
            suggestions.append(f"Deeper investigation into {theme} and its implications")

        return suggestions

    def _calculate_literature_confidence(self, papers: List[AcademicPaper], synthesis: Dict[str, Any]) -> float:
        """Calculate confidence score for literature review"""
        score = 0.0

        # Paper quantity
        paper_count = len(papers)
        if paper_count >= 20:
            score += 0.3
        elif paper_count >= 10:
            score += 0.2
        elif paper_count >= 5:
            score += 0.1

        # Citation quality
        citation_analysis = synthesis['citation_analysis']
        if citation_analysis['highly_cited_papers'] > 0:
            score += 0.2

        # Database coverage
        db_coverage = len(synthesis['database_coverage'])
        if db_coverage >= 3:
            score += 0.2
        elif db_coverage >= 2:
            score += 0.1

        # Temporal coverage
        temporal_dist = synthesis['temporal_distribution']
        years_covered = len([year for year in temporal_dist.keys() if year.isdigit()])
        if years_covered >= 10:
            score += 0.2
        elif years_covered >= 5:
            score += 0.1

        # Recent coverage
        recent_papers = sum(count for year, count in temporal_dist.items()
                          if year.isdigit() and int(year) >= 2020)
        if recent_papers >= paper_count * 0.3:
            score += 0.1

        return min(score, 1.0)

    async def _test_database_connections(self):
        """Test connections to academic databases"""
        for database in [AcademicDatabase.ARXIV, AcademicDatabase.SEMANTIC_SCHOLAR]:
            try:
                # Test with a simple query
                test_papers = await self._search_database(database, "test", max_results=1)
                logger.info(f"Database {database.value} connection: OK")
            except Exception as e:
                logger.warning(f"Database {database.value} connection failed: {e}")

    def _update_search_stats(self, success: bool, search_time_ms: float, papers_count: int):
        """Update search statistics"""
        if success:
            self.stats['successful_searches'] += 1

        # Update average search time
        current_avg = self.stats['avg_search_time_ms']
        count = self.stats['searches_performed']
        self.stats['avg_search_time_ms'] = ((current_avg * (count - 1)) + search_time_ms) / count

    def get_status(self) -> Dict[str, Any]:
        """Get academic research agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status,
            'capabilities': self.capabilities,
            'active_searches': len(self.active_searches),
            'papers_cached': len(self.paper_cache),
            'search_history': len(self.search_history),
            'statistics': self.stats.copy(),
            'supported_databases': [db.value for db in AcademicDatabase]
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'is_healthy': self.status == "active",
            'searches_performed': self.stats['searches_performed'],
            'success_rate': (
                self.stats['successful_searches'] / max(1, self.stats['searches_performed'])
            ),
            'papers_retrieved': self.stats['papers_retrieved'],
            'literature_reviews_completed': self.stats['literature_reviews_completed'],
            'avg_search_time_ms': self.stats['avg_search_time_ms'],
            'last_check': datetime.utcnow().isoformat()
        }
