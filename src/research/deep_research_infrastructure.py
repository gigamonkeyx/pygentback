"""
Deep Research Infrastructure

Comprehensive academic research system with multiple data sources,
global search capabilities, and country-based grouping.
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import urllib.parse

logger = logging.getLogger(__name__)


class ResearchDatabase(Enum):
    """Available research databases"""
    OPENALEX = "openalex"           # 250M+ papers, FREE, no auth
    CORE = "core"                   # 100M+ open access papers, FREE
    SEMANTIC_SCHOLAR = "semantic_scholar"  # FREE with limits
    CROSSREF = "crossref"           # FREE DOI database
    ARXIV = "arxiv"                 # FREE preprints
    UNPAYWALL = "unpaywall"         # FREE open access detection
    PUBMED = "pubmed"               # FREE biomedical
    DOAJ = "doaj"                   # FREE open access journals
    # Premium databases (require API keys)
    SCOPUS = "scopus"               # Elsevier subscription
    WEB_OF_SCIENCE = "web_of_science"  # Clarivate subscription


@dataclass
class ResearchPaper:
    """Comprehensive research paper metadata"""
    paper_id: str
    title: str
    authors: List[Dict[str, Any]]
    abstract: str
    publication_date: str
    journal: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    pdf_url: Optional[str]
    citations: int
    database_source: ResearchDatabase
    keywords: List[str]
    countries: List[str] = field(default_factory=list)
    institutions: List[Dict[str, Any]] = field(default_factory=list)
    open_access: bool = False
    language: str = "en"
    subject_areas: List[str] = field(default_factory=list)
    funding: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_primary_country(self) -> Optional[str]:
        """Get primary country based on first author affiliation"""
        if self.countries:
            return self.countries[0]
        return None


@dataclass
class GlobalSearchResult:
    """Results from global academic search"""
    query: str
    total_papers: int
    papers: List[ResearchPaper]
    country_distribution: Dict[str, int]
    database_coverage: Dict[str, int]
    temporal_distribution: Dict[str, int]
    subject_distribution: Dict[str, int]
    open_access_ratio: float
    search_metadata: Dict[str, Any]


class DeepResearchInfrastructure:
    """
    Comprehensive academic research infrastructure with global search
    and country-based analysis capabilities.
    """
    
    def __init__(self, infrastructure_id: str = "deep_research_infra"):
        self.infrastructure_id = infrastructure_id
        self.status = "initialized"
        
        # Database configurations
        self.database_configs = {
            ResearchDatabase.OPENALEX: {
                'base_url': 'https://api.openalex.org',
                'rate_limit': 10,  # requests per second
                'daily_limit': 100000,
                'requires_auth': False,
                'description': '250M+ scholarly works, completely free',
                'strengths': ['comprehensive coverage', 'institutional data', 'country info']
            },
            ResearchDatabase.CORE: {
                'base_url': 'https://api.core.ac.uk/v3',
                'rate_limit': 5,
                'daily_limit': 10000,
                'requires_auth': True,  # API key recommended
                'description': '100M+ open access papers',
                'strengths': ['open access focus', 'full text', 'repository data']
            },
            ResearchDatabase.SEMANTIC_SCHOLAR: {
                'base_url': 'https://api.semanticscholar.org/graph/v1',
                'rate_limit': 1,  # Conservative for free tier
                'daily_limit': 1000,
                'requires_auth': False,
                'description': 'AI-powered paper analysis',
                'strengths': ['citation analysis', 'paper recommendations', 'AI summaries']
            },
            ResearchDatabase.CROSSREF: {
                'base_url': 'https://api.crossref.org',
                'rate_limit': 50,
                'daily_limit': 50000,
                'requires_auth': False,
                'description': 'Comprehensive DOI database',
                'strengths': ['citation data', 'publisher info', 'funding data']
            },
            ResearchDatabase.ARXIV: {
                'base_url': 'http://export.arxiv.org/api',
                'rate_limit': 1,
                'daily_limit': 1000,
                'requires_auth': False,
                'description': 'Preprints and early research',
                'strengths': ['cutting edge research', 'STEM focus', 'preprint access']
            },
            ResearchDatabase.PUBMED: {
                'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                'rate_limit': 3,
                'daily_limit': 10000,
                'requires_auth': False,
                'description': 'Biomedical and life sciences',
                'strengths': ['medical research', 'clinical studies', 'health data']
            }
        }
        
        # Country mapping for global analysis
        self.country_mappings = {
            'US': 'United States', 'GB': 'United Kingdom', 'DE': 'Germany',
            'FR': 'France', 'JP': 'Japan', 'CN': 'China', 'CA': 'Canada',
            'AU': 'Australia', 'IT': 'Italy', 'ES': 'Spain', 'NL': 'Netherlands',
            'SE': 'Sweden', 'CH': 'Switzerland', 'KR': 'South Korea',
            'BR': 'Brazil', 'IN': 'India', 'RU': 'Russia', 'MX': 'Mexico',
            'ZA': 'South Africa', 'EG': 'Egypt', 'NG': 'Nigeria', 'KE': 'Kenya'
        }
        
        # Research state
        self.active_searches: Dict[str, Dict[str, Any]] = {}
        self.paper_cache: Dict[str, ResearchPaper] = {}
        self.country_cache: Dict[str, List[ResearchPaper]] = {}
        
        # Statistics
        self.stats = {
            'searches_performed': 0,
            'papers_retrieved': 0,
            'databases_queried': 0,
            'countries_analyzed': 0,
            'global_searches': 0
        }
        
        logger.info(f"DeepResearchInfrastructure {infrastructure_id} initialized")
    
    async def start(self) -> bool:
        """Start the deep research infrastructure"""
        try:
            self.status = "starting"
            
            # Test database connections
            await self._test_database_connections()
            
            self.status = "active"
            logger.info(f"DeepResearchInfrastructure {self.infrastructure_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start DeepResearchInfrastructure: {e}")
            self.status = "error"
            return False
    
    async def global_academic_search(self, query: str, **kwargs) -> GlobalSearchResult:
        """
        Perform comprehensive global academic search across multiple databases.
        
        Args:
            query: Search query
            **kwargs: Additional parameters
                - databases: List of databases to search
                - max_papers: Maximum papers to retrieve
                - years: Tuple of (start_year, end_year)
                - countries: List of countries to focus on
                - open_access_only: Boolean
                - subject_areas: List of subject areas
                
        Returns:
            GlobalSearchResult with comprehensive analysis
        """
        search_id = f"global_search_{int(datetime.utcnow().timestamp())}"
        
        try:
            # Parse search parameters
            databases = kwargs.get('databases', [
                ResearchDatabase.OPENALEX,
                ResearchDatabase.CORE,
                ResearchDatabase.SEMANTIC_SCHOLAR,
                ResearchDatabase.CROSSREF
            ])
            max_papers = kwargs.get('max_papers', 100)
            years = kwargs.get('years', None)
            target_countries = kwargs.get('countries', [])
            open_access_only = kwargs.get('open_access_only', False)
            subject_areas = kwargs.get('subject_areas', [])
            
            self.stats['global_searches'] += 1
            
            # Initialize search
            self.active_searches[search_id] = {
                'query': query,
                'start_time': datetime.utcnow(),
                'databases': databases,
                'status': 'searching'
            }
            
            # Search across databases
            all_papers = []
            database_results = {}
            
            for database in databases:
                try:
                    papers = await self._search_database_comprehensive(
                        database, query, max_papers // len(databases), **kwargs
                    )
                    all_papers.extend(papers)
                    database_results[database.value] = len(papers)
                    self.stats['databases_queried'] += 1
                    
                    # Rate limiting
                    await asyncio.sleep(1.0 / self.database_configs[database]['rate_limit'])
                    
                except Exception as e:
                    logger.warning(f"Search failed for {database.value}: {e}")
                    database_results[database.value] = 0
            
            # Remove duplicates
            unique_papers = self._deduplicate_papers_advanced(all_papers)
            
            # Apply filters
            filtered_papers = self._apply_search_filters(
                unique_papers, years, target_countries, open_access_only, subject_areas
            )
            
            # Limit results
            final_papers = filtered_papers[:max_papers]
            
            # Analyze results
            analysis = await self._analyze_global_results(final_papers, query)
            
            # Create result
            result = GlobalSearchResult(
                query=query,
                total_papers=len(final_papers),
                papers=final_papers,
                country_distribution=analysis['country_distribution'],
                database_coverage=database_results,
                temporal_distribution=analysis['temporal_distribution'],
                subject_distribution=analysis['subject_distribution'],
                open_access_ratio=analysis['open_access_ratio'],
                search_metadata={
                    'search_id': search_id,
                    'databases_searched': len(databases),
                    'total_raw_results': len(all_papers),
                    'deduplication_ratio': len(unique_papers) / max(len(all_papers), 1),
                    'filter_ratio': len(final_papers) / max(len(unique_papers), 1),
                    'search_time_seconds': (datetime.utcnow() - self.active_searches[search_id]['start_time']).total_seconds()
                }
            )
            
            # Update statistics
            self.stats['papers_retrieved'] += len(final_papers)
            self.stats['searches_performed'] += 1
            
            # Clean up
            del self.active_searches[search_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Global search failed: {e}")
            if search_id in self.active_searches:
                del self.active_searches[search_id]
            raise
    
    async def search_by_country(self, query: str, countries: List[str], **kwargs) -> Dict[str, GlobalSearchResult]:
        """
        Search for research papers grouped by country.
        
        Args:
            query: Search query
            countries: List of country codes or names
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary mapping country to search results
        """
        country_results = {}
        
        for country in countries:
            try:
                # Search with country filter
                result = await self.global_academic_search(
                    query, 
                    countries=[country],
                    **kwargs
                )
                
                country_results[country] = result
                self.stats['countries_analyzed'] += 1
                
                # Delay between country searches
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Country search failed for {country}: {e}")
                country_results[country] = None
        
        return country_results
    
    async def comparative_country_analysis(self, query: str, countries: List[str], **kwargs) -> Dict[str, Any]:
        """
        Perform comparative analysis of research across countries.
        
        Args:
            query: Search query
            countries: List of countries to compare
            **kwargs: Additional parameters
            
        Returns:
            Comprehensive comparative analysis
        """
        # Get results for each country
        country_results = await self.search_by_country(query, countries, **kwargs)
        
        # Perform comparative analysis
        comparison = {
            'query': query,
            'countries_analyzed': len([c for c in country_results.values() if c is not None]),
            'total_papers': sum(r.total_papers for r in country_results.values() if r is not None),
            'country_rankings': {},
            'research_strengths': {},
            'collaboration_patterns': {},
            'temporal_trends': {},
            'subject_focus': {},
            'open_access_adoption': {}
        }
        
        # Analyze each country
        for country, result in country_results.items():
            if result is None:
                continue
                
            comparison['country_rankings'][country] = {
                'total_papers': result.total_papers,
                'citations_avg': self._calculate_avg_citations(result.papers),
                'open_access_ratio': result.open_access_ratio,
                'international_collaboration': self._calculate_collaboration_score(result.papers)
            }
            
            comparison['research_strengths'][country] = result.subject_distribution
            comparison['temporal_trends'][country] = result.temporal_distribution
            comparison['open_access_adoption'][country] = result.open_access_ratio
        
        return comparison

    async def _search_database_comprehensive(self, database: ResearchDatabase, query: str,
                                           max_results: int, **kwargs) -> List[ResearchPaper]:
        """Search a specific database with comprehensive metadata extraction"""

        if database == ResearchDatabase.OPENALEX:
            return await self._search_openalex(query, max_results, **kwargs)
        elif database == ResearchDatabase.CORE:
            return await self._search_core(query, max_results, **kwargs)
        elif database == ResearchDatabase.SEMANTIC_SCHOLAR:
            return await self._search_semantic_scholar_enhanced(query, max_results, **kwargs)
        elif database == ResearchDatabase.CROSSREF:
            return await self._search_crossref_enhanced(query, max_results, **kwargs)
        elif database == ResearchDatabase.ARXIV:
            return await self._search_arxiv_enhanced(query, max_results, **kwargs)
        elif database == ResearchDatabase.PUBMED:
            return await self._search_pubmed(query, max_results, **kwargs)
        else:
            logger.warning(f"Database {database.value} not implemented")
            return []

    async def _search_openalex(self, query: str, max_results: int, **kwargs) -> List[ResearchPaper]:
        """Search OpenAlex - the most comprehensive free academic database"""
        try:
            config = self.database_configs[ResearchDatabase.OPENALEX]

            # Build OpenAlex query
            params = {
                'search': query,
                'per-page': min(max_results, 200),  # OpenAlex max per page
                'mailto': 'research@pygent.ai',  # Polite pool
                'select': 'id,title,display_name,publication_year,publication_date,doi,open_access,authorships,institutions_distinct_count,countries_distinct_count,cited_by_count,concepts,primary_location,locations,abstract_inverted_index'
            }

            # Add filters
            filters = []
            if kwargs.get('years'):
                start_year, end_year = kwargs['years']
                filters.append(f'publication_year:{start_year}-{end_year}')

            if kwargs.get('open_access_only'):
                filters.append('is_oa:true')

            if kwargs.get('countries'):
                country_filter = '|'.join(kwargs['countries'])
                filters.append(f'institutions.country_code:{country_filter}')

            if filters:
                params['filter'] = ','.join(filters)

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{config['base_url']}/works", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_openalex_response(data)
                    else:
                        logger.error(f"OpenAlex search failed with status {response.status}")
                        return []

        except Exception as e:
            logger.error(f"OpenAlex search error: {e}")
            return []

    async def _search_core(self, query: str, max_results: int, **kwargs) -> List[ResearchPaper]:
        """Search CORE - open access focused database"""
        try:
            config = self.database_configs[ResearchDatabase.CORE]

            # CORE API v3 search
            params = {
                'q': query,
                'limit': min(max_results, 100),
                'offset': 0
            }

            headers = {}
            # Add API key if available
            api_key = kwargs.get('core_api_key')
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{config['base_url']}/search/works",
                                     params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_core_response(data)
                    else:
                        logger.error(f"CORE search failed with status {response.status}")
                        return []

        except Exception as e:
            logger.error(f"CORE search error: {e}")
            return []

    async def _search_semantic_scholar_enhanced(self, query: str, max_results: int, **kwargs) -> List[ResearchPaper]:
        """Enhanced Semantic Scholar search with comprehensive metadata"""
        try:
            config = self.database_configs[ResearchDatabase.SEMANTIC_SCHOLAR]

            params = {
                'query': query,
                'limit': min(max_results, 100),
                'fields': 'paperId,title,authors,year,abstract,venue,journal,doi,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,externalIds'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{config['base_url']}/paper/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_semantic_scholar_enhanced_response(data)
                    else:
                        logger.error(f"Semantic Scholar search failed with status {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []

    async def _search_crossref_enhanced(self, query: str, max_results: int, **kwargs) -> List[ResearchPaper]:
        """Enhanced CrossRef search with funding and institutional data"""
        try:
            config = self.database_configs[ResearchDatabase.CROSSREF]

            params = {
                'query': query,
                'rows': min(max_results, 1000),
                'sort': 'relevance',
                'order': 'desc',
                'mailto': 'research@pygent.ai'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{config['base_url']}/works", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_crossref_enhanced_response(data)
                    else:
                        logger.error(f"CrossRef search failed with status {response.status}")
                        return []

        except Exception as e:
            logger.error(f"CrossRef search error: {e}")
            return []

    async def _search_arxiv_enhanced(self, query: str, max_results: int, **kwargs) -> List[ResearchPaper]:
        """Enhanced arXiv search with category and subject analysis"""
        try:
            config = self.database_configs[ResearchDatabase.ARXIV]

            # Build arXiv query
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': min(max_results, 1000),
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{config['base_url']}/query", params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_arxiv_enhanced_response(content)
                    else:
                        logger.error(f"arXiv search failed with status {response.status}")
                        return []

        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []

    async def _search_pubmed(self, query: str, max_results: int, **kwargs) -> List[ResearchPaper]:
        """Search PubMed for biomedical research"""
        try:
            config = self.database_configs[ResearchDatabase.PUBMED]

            # PubMed E-utilities search
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': min(max_results, 1000),
                'retmode': 'json',
                'sort': 'relevance'
            }

            async with aiohttp.ClientSession() as session:
                # First, search for PMIDs
                async with session.get(f"{config['base_url']}/esearch.fcgi", params=search_params) as response:
                    if response.status == 200:
                        search_data = await response.json()
                        pmids = search_data.get('esearchresult', {}).get('idlist', [])

                        if pmids:
                            # Fetch detailed information
                            fetch_params = {
                                'db': 'pubmed',
                                'id': ','.join(pmids[:max_results]),
                                'retmode': 'json',
                                'rettype': 'abstract'
                            }

                            async with session.get(f"{config['base_url']}/efetch.fcgi", params=fetch_params) as fetch_response:
                                if fetch_response.status == 200:
                                    fetch_data = await fetch_response.json()
                                    return self._parse_pubmed_response(fetch_data)

                        return []
                    else:
                        logger.error(f"PubMed search failed with status {response.status}")
                        return []

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    def _parse_openalex_response(self, data: Dict) -> List[ResearchPaper]:
        """Parse OpenAlex API response with comprehensive metadata"""
        papers = []

        try:
            for item in data.get('results', []):
                # Extract authors with affiliations
                authors = []
                countries = []
                institutions = []

                for authorship in item.get('authorships', []):
                    author_info = {
                        'name': authorship.get('author', {}).get('display_name', 'Unknown'),
                        'id': authorship.get('author', {}).get('id'),
                        'orcid': authorship.get('author', {}).get('orcid')
                    }
                    authors.append(author_info)

                    # Extract institutional and country info
                    for institution in authorship.get('institutions', []):
                        inst_info = {
                            'name': institution.get('display_name'),
                            'country': institution.get('country_code'),
                            'type': institution.get('type')
                        }
                        institutions.append(inst_info)

                        if institution.get('country_code'):
                            countries.append(institution['country_code'])

                # Extract subject areas from concepts
                subject_areas = []
                for concept in item.get('concepts', []):
                    if concept.get('score', 0) > 0.3:  # Only high-confidence concepts
                        subject_areas.append(concept.get('display_name'))

                # Create paper object
                paper = ResearchPaper(
                    paper_id=item.get('id', ''),
                    title=item.get('title', 'Unknown Title'),
                    authors=authors,
                    abstract=self._reconstruct_abstract(item.get('abstract_inverted_index', {})),
                    publication_date=item.get('publication_date', ''),
                    journal=item.get('primary_location', {}).get('source', {}).get('display_name'),
                    doi=item.get('doi'),
                    url=item.get('id'),
                    pdf_url=item.get('open_access', {}).get('oa_url'),
                    citations=item.get('cited_by_count', 0),
                    database_source=ResearchDatabase.OPENALEX,
                    keywords=[],
                    countries=list(set(countries)),
                    institutions=institutions,
                    open_access=item.get('open_access', {}).get('is_oa', False),
                    subject_areas=subject_areas
                )

                papers.append(paper)

        except Exception as e:
            logger.error(f"Error parsing OpenAlex response: {e}")

        return papers
