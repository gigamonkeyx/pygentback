"""
Resource Mapper

Comprehensive discovery and mapping of all available academic research resources,
APIs, libraries, and their actual capabilities before building the system.
"""

import logging
import asyncio
import importlib
import inspect
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ResourceCapability:
    """Detailed capability information for a resource"""
    name: str
    type: str  # 'library', 'api', 'database', 'service'
    status: str  # 'available', 'unavailable', 'requires_auth', 'rate_limited'
    methods: List[str] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    authentication_required: bool = False
    rate_limits: Dict[str, Any] = field(default_factory=dict)
    cost: str = "unknown"  # 'free', 'paid', 'freemium', 'institutional'
    documentation_url: Optional[str] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    last_tested: Optional[str] = None


class ResourceMapper:
    """
    Comprehensive mapper of all available academic research resources.
    Tests and catalogs actual capabilities before system implementation.
    """
    
    def __init__(self):
        self.resources = {}
        self.test_results = {}
        
        # Define resources to test
        self.target_resources = {
            'python_libraries': [
                'scholarly',
                'internetarchive', 
                'requests',
                'beautifulsoup4',
                'arxiv',
                'crossref-commons',
                'pymed',
                'habanero'  # CrossRef client
            ],
            'academic_apis': [
                {
                    'name': 'OpenAlex',
                    'base_url': 'https://api.openalex.org',
                    'test_endpoint': '/works?search=history',
                    'auth_required': False,
                    'documentation': 'https://docs.openalex.org/'
                },
                {
                    'name': 'CrossRef',
                    'base_url': 'https://api.crossref.org',
                    'test_endpoint': '/works?query=history',
                    'auth_required': False,
                    'documentation': 'https://github.com/CrossRef/rest-api-doc'
                },
                {
                    'name': 'CORE',
                    'base_url': 'https://api.core.ac.uk/v3',
                    'test_endpoint': '/search/works?q=history',
                    'auth_required': True,
                    'documentation': 'https://core.ac.uk/docs/'
                },
                {
                    'name': 'Semantic Scholar',
                    'base_url': 'https://api.semanticscholar.org/graph/v1',
                    'test_endpoint': '/paper/search?query=history',
                    'auth_required': False,
                    'documentation': 'https://api.semanticscholar.org/'
                },
                {
                    'name': 'arXiv',
                    'base_url': 'http://export.arxiv.org/api',
                    'test_endpoint': '/query?search_query=all:history&max_results=1',
                    'auth_required': False,
                    'documentation': 'https://arxiv.org/help/api'
                },
                {
                    'name': 'PubMed',
                    'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                    'test_endpoint': '/esearch.fcgi?db=pubmed&term=history&retmode=json',
                    'auth_required': False,
                    'documentation': 'https://www.ncbi.nlm.nih.gov/books/NBK25501/'
                },
                {
                    'name': 'HathiTrust',
                    'base_url': 'https://babel.hathitrust.org/cgi',
                    'test_endpoint': '/ssd?id=test',
                    'auth_required': True,
                    'documentation': 'https://www.hathitrust.org/htrc'
                },
                {
                    'name': 'Internet Archive',
                    'base_url': 'https://archive.org',
                    'test_endpoint': '/advancedsearch.php?q=history&output=json',
                    'auth_required': False,
                    'documentation': 'https://archive.org/help/aboutsearch.htm'
                },
                {
                    'name': 'Europeana',
                    'base_url': 'https://api.europeana.eu/record/v2',
                    'test_endpoint': '/search.json?wskey=api2demo&query=history',
                    'auth_required': True,
                    'documentation': 'https://pro.europeana.eu/page/apis'
                },
                {
                    'name': 'DOAJ',
                    'base_url': 'https://doaj.org/api/v2',
                    'test_endpoint': '/search/articles/history',
                    'auth_required': False,
                    'documentation': 'https://doaj.org/api/v2/docs'
                }
            ],
            'mcp_servers': [
                'scholarly-mcp',
                'academic-search-mcp',
                'arxiv-mcp',
                'pubmed-mcp'
            ]
        }
    
    async def map_all_resources(self) -> Dict[str, ResourceCapability]:
        """Comprehensively map all available resources"""
        
        print("🗺️  COMPREHENSIVE RESOURCE MAPPING")
        print("=" * 80)
        print("📝 Discovering and testing all available academic research resources...")
        print()
        
        # Test Python libraries
        print("🐍 TESTING PYTHON LIBRARIES")
        print("-" * 50)
        await self._test_python_libraries()
        
        # Test Academic APIs
        print("\n🌐 TESTING ACADEMIC APIs")
        print("-" * 50)
        await self._test_academic_apis()
        
        # Test MCP servers
        print("\n🔧 TESTING MCP SERVERS")
        print("-" * 50)
        await self._test_mcp_servers()
        
        # Generate comprehensive report
        print("\n📊 GENERATING COMPREHENSIVE RESOURCE MAP")
        print("-" * 50)
        self._generate_resource_map()
        
        return self.resources
    
    async def _test_python_libraries(self):
        """Test availability and capabilities of Python libraries"""
        
        for library in self.target_resources['python_libraries']:
            print(f"   Testing {library}...")
            
            try:
                # Try to import the library
                module = importlib.import_module(library)
                
                # Get available methods/functions
                methods = [name for name in dir(module) if not name.startswith('_')]
                
                # Special testing for key libraries
                test_results = {}
                
                if library == 'scholarly':
                    test_results = await self._test_scholarly_library(module)
                elif library == 'internetarchive':
                    test_results = await self._test_internetarchive_library(module)
                elif library == 'arxiv':
                    test_results = await self._test_arxiv_library(module)
                elif library == 'requests':
                    test_results = {'basic_functionality': 'available'}
                
                self.resources[library] = ResourceCapability(
                    name=library,
                    type='library',
                    status='available',
                    methods=methods,
                    cost='free',
                    test_results=test_results,
                    last_tested=datetime.utcnow().isoformat()
                )
                
                print(f"      ✅ {library}: Available ({len(methods)} methods)")
                
            except ImportError as e:
                self.resources[library] = ResourceCapability(
                    name=library,
                    type='library',
                    status='unavailable',
                    error_messages=[f"Import failed: {str(e)}"],
                    last_tested=datetime.utcnow().isoformat()
                )
                print(f"      ❌ {library}: Not available - {e}")
            
            except Exception as e:
                self.resources[library] = ResourceCapability(
                    name=library,
                    type='library',
                    status='error',
                    error_messages=[f"Test failed: {str(e)}"],
                    last_tested=datetime.utcnow().isoformat()
                )
                print(f"      ⚠️  {library}: Error - {e}")
    
    async def _test_scholarly_library(self, module) -> Dict[str, Any]:
        """Detailed testing of scholarly library"""
        
        results = {}
        
        try:
            # Check for scholarly object
            if hasattr(module, 'scholarly'):
                scholarly_obj = module.scholarly
                scholarly_methods = [name for name in dir(scholarly_obj) if not name.startswith('_')]
                results['scholarly_methods'] = scholarly_methods
                
                # Test search methods
                search_methods = [m for m in scholarly_methods if 'search' in m.lower()]
                results['search_methods'] = search_methods
                
                # Test specific methods
                if 'search_pubs' in scholarly_methods:
                    results['search_pubs_available'] = True
                    try:
                        # Try a simple search (don't actually execute to avoid rate limits)
                        search_obj = scholarly_obj.search_pubs("test")
                        results['search_pubs_functional'] = True
                    except Exception as e:
                        results['search_pubs_error'] = str(e)
                
                if 'search_author' in scholarly_methods:
                    results['search_author_available'] = True
                
            else:
                results['error'] = 'No scholarly object found in module'
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _test_internetarchive_library(self, module) -> Dict[str, Any]:
        """Test Internet Archive library"""
        
        results = {}
        
        try:
            # Check for search functionality
            if hasattr(module, 'search'):
                results['search_available'] = True
            
            if hasattr(module, 'get_item'):
                results['get_item_available'] = True
            
            if hasattr(module, 'download'):
                results['download_available'] = True
            
            # Check for session
            if hasattr(module, 'get_session'):
                results['session_available'] = True
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _test_arxiv_library(self, module) -> Dict[str, Any]:
        """Test arXiv library"""
        
        results = {}
        
        try:
            # Check for Search class
            if hasattr(module, 'Search'):
                results['Search_class_available'] = True
            
            if hasattr(module, 'Client'):
                results['Client_class_available'] = True
            
            # Check for sort criteria
            if hasattr(module, 'SortCriterion'):
                results['SortCriterion_available'] = True
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _test_academic_apis(self):
        """Test academic APIs for availability and functionality"""
        
        for api_config in self.target_resources['academic_apis']:
            name = api_config['name']
            print(f"   Testing {name} API...")
            
            try:
                # Test basic connectivity
                test_url = api_config['base_url'] + api_config['test_endpoint']
                
                # Make a simple request with timeout
                response = requests.get(test_url, timeout=10)
                
                if response.status_code == 200:
                    status = 'available'
                    test_results = {
                        'status_code': response.status_code,
                        'response_size': len(response.content),
                        'content_type': response.headers.get('content-type', 'unknown')
                    }
                    print(f"      ✅ {name}: Available (HTTP {response.status_code})")
                    
                elif response.status_code == 401:
                    status = 'requires_auth'
                    test_results = {'status_code': response.status_code, 'message': 'Authentication required'}
                    print(f"      🔐 {name}: Requires authentication")
                    
                elif response.status_code == 429:
                    status = 'rate_limited'
                    test_results = {'status_code': response.status_code, 'message': 'Rate limited'}
                    print(f"      ⏱️  {name}: Rate limited")
                    
                else:
                    status = 'error'
                    test_results = {'status_code': response.status_code}
                    print(f"      ⚠️  {name}: HTTP {response.status_code}")
                
                self.resources[name] = ResourceCapability(
                    name=name,
                    type='api',
                    status=status,
                    endpoints=[api_config['test_endpoint']],
                    authentication_required=api_config['auth_required'],
                    cost='free',
                    documentation_url=api_config['documentation'],
                    test_results=test_results,
                    last_tested=datetime.utcnow().isoformat()
                )
                
            except requests.exceptions.Timeout:
                self.resources[name] = ResourceCapability(
                    name=name,
                    type='api',
                    status='timeout',
                    error_messages=['Request timeout'],
                    last_tested=datetime.utcnow().isoformat()
                )
                print(f"      ⏰ {name}: Timeout")
                
            except requests.exceptions.ConnectionError:
                self.resources[name] = ResourceCapability(
                    name=name,
                    type='api',
                    status='unavailable',
                    error_messages=['Connection failed'],
                    last_tested=datetime.utcnow().isoformat()
                )
                print(f"      ❌ {name}: Connection failed")
                
            except Exception as e:
                self.resources[name] = ResourceCapability(
                    name=name,
                    type='api',
                    status='error',
                    error_messages=[str(e)],
                    last_tested=datetime.utcnow().isoformat()
                )
                print(f"      ❌ {name}: Error - {e}")
    
    async def _test_mcp_servers(self):
        """Test MCP server availability"""
        
        for mcp_server in self.target_resources['mcp_servers']:
            print(f"   Testing {mcp_server}...")
            
            # For now, just check if they're installable
            # In a full implementation, we'd test actual MCP functionality
            
            self.resources[mcp_server] = ResourceCapability(
                name=mcp_server,
                type='mcp_server',
                status='unknown',
                test_results={'note': 'MCP server testing not yet implemented'},
                last_tested=datetime.utcnow().isoformat()
            )
            print(f"      ❓ {mcp_server}: Status unknown (MCP testing not implemented)")
    
    def _generate_resource_map(self):
        """Generate comprehensive resource map report"""
        
        print("📋 COMPREHENSIVE RESOURCE MAP")
        print("=" * 80)
        
        # Categorize resources
        available_libraries = [r for r in self.resources.values() if r.type == 'library' and r.status == 'available']
        available_apis = [r for r in self.resources.values() if r.type == 'api' and r.status == 'available']
        auth_required_apis = [r for r in self.resources.values() if r.type == 'api' and r.status == 'requires_auth']
        unavailable_resources = [r for r in self.resources.values() if r.status in ['unavailable', 'error', 'timeout']]
        
        print(f"✅ AVAILABLE PYTHON LIBRARIES ({len(available_libraries)}):")
        for lib in available_libraries:
            methods_count = len(lib.methods) if lib.methods else 0
            print(f"   • {lib.name}: {methods_count} methods available")
            
            # Show key test results
            if lib.test_results:
                for key, value in lib.test_results.items():
                    if isinstance(value, list) and len(value) > 0:
                        print(f"     - {key}: {len(value)} items")
                    elif isinstance(value, bool) and value:
                        print(f"     - {key}: ✅")
        
        print(f"\n✅ AVAILABLE APIs ({len(available_apis)}):")
        for api in available_apis:
            print(f"   • {api.name}: {api.test_results.get('status_code', 'unknown')} response")
            if api.documentation_url:
                print(f"     - Documentation: {api.documentation_url}")
        
        print(f"\n🔐 AUTHENTICATION REQUIRED ({len(auth_required_apis)}):")
        for api in auth_required_apis:
            print(f"   • {api.name}: Requires API key or authentication")
            if api.documentation_url:
                print(f"     - Documentation: {api.documentation_url}")
        
        print(f"\n❌ UNAVAILABLE RESOURCES ({len(unavailable_resources)}):")
        for resource in unavailable_resources:
            print(f"   • {resource.name} ({resource.type}): {resource.status}")
            if resource.error_messages:
                print(f"     - Error: {resource.error_messages[0]}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        
        if any(lib.name == 'scholarly' for lib in available_libraries):
            print("✅ Google Scholar access available via scholarly library")
        else:
            print("❌ Install scholarly library: pip install scholarly")
        
        if any(api.name == 'OpenAlex' for api in available_apis):
            print("✅ OpenAlex API available (250M+ papers, completely free)")
        
        if any(api.name == 'CrossRef' for api in available_apis):
            print("✅ CrossRef API available (comprehensive DOI database)")
        
        if any(api.name == 'Semantic Scholar' for api in available_apis):
            print("✅ Semantic Scholar API available")
        
        # Installation recommendations
        missing_libraries = [lib for lib in self.target_resources['python_libraries'] 
                           if lib not in [r.name for r in available_libraries]]
        
        if missing_libraries:
            print(f"\n📦 INSTALL MISSING LIBRARIES:")
            print(f"   pip install {' '.join(missing_libraries)}")
        
        print(f"\n🎯 RECOMMENDED ZERO-COST RESEARCH STACK:")
        print("   Primary: Google Scholar (via scholarly library)")
        print("   Secondary: OpenAlex API (250M+ papers)")
        print("   Tertiary: CrossRef API (metadata)")
        print("   Archives: Internet Archive API")
        print("   Preprints: arXiv API")
        
        return self.resources


async def main():
    """Run comprehensive resource mapping"""
    
    mapper = ResourceMapper()
    resources = await mapper.map_all_resources()
    
    print(f"\n🎉 RESOURCE MAPPING COMPLETED!")
    print(f"   Total resources tested: {len(resources)}")
    print(f"   Available resources: {len([r for r in resources.values() if r.status == 'available'])}")
    print(f"   Resources requiring auth: {len([r for r in resources.values() if r.status == 'requires_auth'])}")
    print(f"   Unavailable resources: {len([r for r in resources.values() if r.status in ['unavailable', 'error']])}")

if __name__ == "__main__":
    asyncio.run(main())
