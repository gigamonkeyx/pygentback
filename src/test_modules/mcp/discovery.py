"""
MCP Server Discovery System

This module provides automated discovery, cataloging, and installation of MCP servers
from various sources including public registries, GitHub repositories, and npm packages.
"""

import asyncio
import logging
import json
import aiohttp
import subprocess
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from enum import Enum

try:
    from ...mcp.server.config import MCPServerType, MCPServerConfig
except ImportError:
    # Fallback for testing
    from enum import Enum
    from dataclasses import dataclass
    from typing import Dict, Any

    class MCPServerType(Enum):
        FILESYSTEM = "filesystem"
        DATABASE = "database"
        API = "api"
        TOOL = "tool"
        CUSTOM = "custom"

    @dataclass
    class MCPServerConfig:
        name: str
        type: MCPServerType
        config: Dict[str, Any]

# Add new server types for expanded MCP ecosystem
class ExtendedMCPServerType(Enum):
    """Extended MCP server types for comprehensive testing"""
    # Core types (from existing MCPServerType)
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    WEB_SEARCH = "web_search"
    VERSION_CONTROL = "version_control"

    # NLP and Language Processing
    NLP = "nlp"
    TRANSFORMERS = "transformers"
    SPACY = "spacy"
    NLTK = "nltk"
    OPENAI = "openai"
    LANGCHAIN = "langchain"

    # Graphics and Media
    GRAPHICS = "graphics"
    IMAGE_PROCESSING = "image_processing"
    COMPUTER_VISION = "computer_vision"
    VISUALIZATION = "visualization"
    VIDEO_PROCESSING = "video_processing"

    # Web and UI
    WEB_UI = "web_ui"
    REACT = "react"
    HTML_CSS = "html_css"
    STREAMLIT = "streamlit"
    FASTAPI = "fastapi"

    # Development and DevOps
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    JENKINS = "jenkins"

    # Custom and General
    CUSTOM = "custom"


logger = logging.getLogger(__name__)


@dataclass
class MCPServerInfo:
    """Information about a discovered MCP server"""
    name: str
    description: str
    server_type: MCPServerType
    source_url: str
    install_command: List[str]
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    version: str = "latest"
    author: str = ""
    license: str = ""
    documentation_url: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    config_template: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    verified: bool = False
    install_size_mb: Optional[float] = None
    last_updated: Optional[datetime] = None


class MCPServerDiscovery:
    """
    Automated MCP server discovery system.
    
    Discovers MCP servers from multiple sources and maintains a catalog
    of available servers with their capabilities and installation information.
    """
    
    def __init__(self, cache_dir: str = "./data/mcp_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.discovered_servers: Dict[str, MCPServerInfo] = {}
        self.discovery_sources = {
            "npm": self._discover_npm_servers,
            "github": self._discover_github_servers,
            "official": self._discover_official_servers,
            "community": self._discover_community_servers
        }
        
        # Known MCP server categories and their typical tools
        self.server_categories = {
            "nlp": {
                "keywords": ["nlp", "natural language", "text processing", "spacy", "transformers"],
                "expected_tools": ["tokenize", "analyze", "classify", "extract"]
            },
            "graphics": {
                "keywords": ["image", "graphics", "opencv", "pillow", "matplotlib", "visualization"],
                "expected_tools": ["process_image", "generate_chart", "convert_format"]
            },
            "database": {
                "keywords": ["database", "sql", "postgres", "mongo", "redis", "sqlite"],
                "expected_tools": ["query", "insert", "update", "delete", "schema"]
            },
            "web_ui": {
                "keywords": ["web", "ui", "react", "html", "css", "streamlit", "fastapi"],
                "expected_tools": ["create_component", "generate_page", "style", "deploy"]
            },
            "development": {
                "keywords": ["git", "github", "docker", "kubernetes", "ci", "cd", "devops"],
                "expected_tools": ["commit", "deploy", "build", "test", "monitor"]
            },
            "coding": {
                "keywords": ["code", "programming", "vscode", "jupyter", "copilot", "eslint", "prettier", "ide"],
                "expected_tools": ["edit_code", "debug", "format", "lint", "complete", "refactor"]
            },
            "academic_research": {
                "keywords": ["academic", "research", "arxiv", "pubmed", "scholar", "zotero", "latex", "citation"],
                "expected_tools": ["search_papers", "format_citation", "compile_latex", "manage_references"]
            }
        }
    
    async def discover_all_servers(self, sources: Optional[List[str]] = None) -> Dict[str, MCPServerInfo]:
        """
        Discover MCP servers from all or specified sources.
        
        Args:
            sources: List of source names to search, or None for all sources
            
        Returns:
            Dict mapping server names to server info
        """
        if sources is None:
            sources = list(self.discovery_sources.keys())
        
        logger.info(f"Starting MCP server discovery from sources: {sources}")
        
        # Run discovery from all sources concurrently
        discovery_tasks = []
        for source in sources:
            if source in self.discovery_sources:
                discovery_tasks.append(self.discovery_sources[source]())
        
        # Wait for all discovery tasks to complete
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        # Process results
        total_discovered = 0
        for i, result in enumerate(results):
            source = sources[i]
            if isinstance(result, Exception):
                logger.error(f"Discovery failed for source {source}: {result}")
            else:
                servers_found = len(result) if result else 0
                total_discovered += servers_found
                logger.info(f"Discovered {servers_found} servers from {source}")
                
                # Merge discovered servers
                if result:
                    self.discovered_servers.update(result)
        
        logger.info(f"Total servers discovered: {total_discovered}")
        
        # Save discovery cache
        await self._save_discovery_cache()
        
        return self.discovered_servers
    
    async def _discover_npm_servers(self) -> Dict[str, MCPServerInfo]:
        """Discover MCP servers from npm registry"""
        servers = {}
        
        try:
            # Search npm for MCP-related packages
            search_terms = [
                "mcp-server",
                "model-context-protocol",
                "@modelcontextprotocol/server"
            ]
            
            for term in search_terms:
                npm_results = await self._search_npm_packages(term)
                
                for package in npm_results:
                    server_info = await self._analyze_npm_package(package)
                    if server_info:
                        servers[server_info.name] = server_info
        
        except Exception as e:
            logger.error(f"NPM discovery failed: {e}")
        
        return servers
    
    async def _discover_github_servers(self) -> Dict[str, MCPServerInfo]:
        """Discover MCP servers from GitHub repositories"""
        servers = {}
        
        try:
            # Search GitHub for MCP server repositories
            search_queries = [
                "mcp-server language:python",
                "mcp-server language:javascript",
                "model-context-protocol server",
                "mcp tool server"
            ]
            
            for query in search_queries:
                github_results = await self._search_github_repositories(query)
                
                for repo in github_results:
                    server_info = await self._analyze_github_repository(repo)
                    if server_info:
                        servers[server_info.name] = server_info
        
        except Exception as e:
            logger.error(f"GitHub discovery failed: {e}")
        
        return servers
    
    async def _discover_official_servers(self) -> Dict[str, MCPServerInfo]:
        """Discover official MCP servers"""
        servers = {}
        
        # Official MCP servers from the Model Context Protocol organization
        official_servers = [
            {
                "name": "filesystem",
                "description": "File system operations and management",
                "server_type": MCPServerType.FILESYSTEM,
                "install_command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
                "capabilities": ["file-read", "file-write", "directory-list"],
                "tools": ["read_file", "write_file", "list_directory", "create_directory"],
                "category": "development",
                "source_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem"
            },
            {
                "name": "brave-search",
                "description": "Web search using Brave Search API",
                "server_type": MCPServerType.WEB_SEARCH,
                "install_command": ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
                "capabilities": ["web-search", "search-results"],
                "tools": ["brave_web_search"],
                "category": "nlp",
                "source_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search"
            },
            {
                "name": "postgres",
                "description": "PostgreSQL database operations",
                "server_type": MCPServerType.DATABASE,
                "install_command": ["npx", "-y", "@modelcontextprotocol/server-postgres"],
                "capabilities": ["sql-execution", "schema-management"],
                "tools": ["query", "list_tables", "describe_table"],
                "category": "database",
                "source_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/postgres"
            },
            {
                "name": "github",
                "description": "GitHub repository operations",
                "server_type": MCPServerType.VERSION_CONTROL,
                "install_command": ["npx", "-y", "@modelcontextprotocol/server-github"],
                "capabilities": ["repository-operations", "commit-management"],
                "tools": ["create_repository", "list_files", "get_file", "create_issue"],
                "category": "development",
                "source_url": "https://github.com/modelcontextprotocol/servers/tree/main/src/github"
            }
        ]
        
        for server_data in official_servers:
            server_info = MCPServerInfo(
                name=server_data["name"],
                description=server_data["description"],
                server_type=server_data["server_type"],
                source_url=server_data["source_url"],
                install_command=server_data["install_command"],
                capabilities=server_data["capabilities"],
                tools=server_data["tools"],
                category=server_data["category"],
                author="Model Context Protocol",
                verified=True
            )
            servers[server_info.name] = server_info
        
        return servers
    
    async def _discover_community_servers(self) -> Dict[str, MCPServerInfo]:
        """Discover community-contributed MCP servers"""
        servers = {}
        
        # Community server definitions (would be loaded from external sources)
        community_servers = [
            {
                "name": "spacy-nlp",
                "description": "Advanced NLP processing with spaCy",
                "server_type": MCPServerType.NLP,
                "install_command": ["pip", "install", "mcp-spacy-server"],
                "capabilities": ["tokenization", "ner", "pos-tagging", "sentiment"],
                "tools": ["tokenize", "extract_entities", "analyze_sentiment"],
                "category": "nlp"
            },
            {
                "name": "opencv-vision",
                "description": "Computer vision and image processing",
                "server_type": MCPServerType.GRAPHICS,
                "install_command": ["pip", "install", "mcp-opencv-server"],
                "capabilities": ["image-processing", "object-detection", "feature-extraction"],
                "tools": ["process_image", "detect_objects", "extract_features"],
                "category": "graphics"
            },
            {
                "name": "streamlit-ui",
                "description": "Interactive web UI generation",
                "server_type": MCPServerType.WEB_UI,
                "install_command": ["pip", "install", "mcp-streamlit-server"],
                "capabilities": ["ui-generation", "dashboard-creation", "interactive-widgets"],
                "tools": ["create_app", "add_widget", "generate_dashboard"],
                "category": "web_ui"
            },

            # Coding and Development Servers
            {
                "name": "vscode-server",
                "description": "VS Code integration for code editing and debugging",
                "server_type": MCPServerType.DEVELOPMENT,
                "install_command": ["npm", "install", "-g", "mcp-vscode-server"],
                "capabilities": ["code-editing", "debugging", "intellisense", "git-integration"],
                "tools": ["edit_file", "debug_code", "get_suggestions", "git_operations"],
                "category": "coding"
            },
            {
                "name": "jupyter-server",
                "description": "Jupyter notebook integration for interactive coding",
                "server_type": MCPServerType.DEVELOPMENT,
                "install_command": ["pip", "install", "mcp-jupyter-server"],
                "capabilities": ["notebook-execution", "code-cells", "data-visualization", "kernel-management"],
                "tools": ["create_notebook", "execute_cell", "manage_kernel", "export_notebook"],
                "category": "coding"
            },
            {
                "name": "github-copilot",
                "description": "AI-powered code completion and generation",
                "server_type": MCPServerType.AI_ASSISTANT,
                "install_command": ["npm", "install", "-g", "mcp-copilot-server"],
                "capabilities": ["code-completion", "code-generation", "code-explanation", "refactoring"],
                "tools": ["complete_code", "generate_function", "explain_code", "refactor_code"],
                "category": "coding"
            },
            {
                "name": "eslint-server",
                "description": "JavaScript/TypeScript linting and code quality",
                "server_type": MCPServerType.DEVELOPMENT,
                "install_command": ["npm", "install", "-g", "mcp-eslint-server"],
                "capabilities": ["linting", "code-quality", "style-checking", "auto-fixing"],
                "tools": ["lint_code", "fix_issues", "check_style", "analyze_quality"],
                "category": "coding"
            },
            {
                "name": "prettier-server",
                "description": "Code formatting for multiple languages",
                "server_type": MCPServerType.DEVELOPMENT,
                "install_command": ["npm", "install", "-g", "mcp-prettier-server"],
                "capabilities": ["code-formatting", "style-enforcement", "multi-language"],
                "tools": ["format_code", "check_formatting", "apply_style"],
                "category": "coding"
            },

            # Academic Research Servers
            {
                "name": "arxiv-server",
                "description": "ArXiv paper search and retrieval",
                "server_type": MCPServerType.WEB_SEARCH,
                "install_command": ["pip", "install", "mcp-arxiv-server"],
                "capabilities": ["paper-search", "metadata-extraction", "pdf-download", "citation-formatting"],
                "tools": ["search_papers", "get_metadata", "download_pdf", "format_citation"],
                "category": "academic_research"
            },
            {
                "name": "pubmed-server",
                "description": "PubMed medical literature search",
                "server_type": MCPServerType.WEB_SEARCH,
                "install_command": ["pip", "install", "mcp-pubmed-server"],
                "capabilities": ["medical-search", "abstract-retrieval", "mesh-terms", "citation-analysis"],
                "tools": ["search_pubmed", "get_abstract", "extract_mesh", "analyze_citations"],
                "category": "academic_research"
            },
            {
                "name": "scholar-server",
                "description": "Google Scholar academic search",
                "server_type": MCPServerType.WEB_SEARCH,
                "install_command": ["pip", "install", "mcp-scholar-server"],
                "capabilities": ["academic-search", "citation-metrics", "author-profiles", "h-index"],
                "tools": ["search_scholar", "get_citations", "author_profile", "calculate_metrics"],
                "category": "academic_research"
            },
            {
                "name": "zotero-server",
                "description": "Zotero reference management integration",
                "server_type": MCPServerType.DATABASE,
                "install_command": ["pip", "install", "mcp-zotero-server"],
                "capabilities": ["reference-management", "bibliography-generation", "pdf-annotation", "group-libraries"],
                "tools": ["add_reference", "generate_bibliography", "annotate_pdf", "sync_library"],
                "category": "academic_research"
            },
            {
                "name": "mendeley-server",
                "description": "Mendeley reference manager integration",
                "server_type": MCPServerType.DATABASE,
                "install_command": ["pip", "install", "mcp-mendeley-server"],
                "capabilities": ["reference-sync", "pdf-management", "collaboration", "citation-styles"],
                "tools": ["sync_references", "manage_pdfs", "collaborate", "format_citations"],
                "category": "academic_research"
            },
            {
                "name": "latex-server",
                "description": "LaTeX document compilation and formatting",
                "server_type": MCPServerType.DOCUMENT_PROCESSING,
                "install_command": ["pip", "install", "mcp-latex-server"],
                "capabilities": ["latex-compilation", "bibliography-formatting", "figure-management", "template-generation"],
                "tools": ["compile_latex", "format_bibliography", "manage_figures", "generate_template"],
                "category": "academic_research"
            },
            {
                "name": "overleaf-server",
                "description": "Overleaf collaborative LaTeX editing",
                "server_type": MCPServerType.COLLABORATION,
                "install_command": ["npm", "install", "-g", "mcp-overleaf-server"],
                "capabilities": ["collaborative-editing", "version-control", "template-library", "real-time-sync"],
                "tools": ["create_project", "collaborate_edit", "manage_versions", "sync_changes"],
                "category": "academic_research"
            },
            {
                "name": "statistical-analysis",
                "description": "Statistical analysis tools for research data",
                "server_type": MCPServerType.DATA_ANALYSIS,
                "install_command": ["pip", "install", "mcp-stats-server"],
                "capabilities": ["descriptive-stats", "hypothesis-testing", "regression-analysis", "data-visualization"],
                "tools": ["descriptive_stats", "hypothesis_test", "regression_analysis", "create_plots"],
                "category": "academic_research"
            }
        ]
        
        for server_data in community_servers:
            server_info = MCPServerInfo(
                name=server_data["name"],
                description=server_data["description"],
                server_type=server_data["server_type"],
                source_url="",  # Would be filled from actual source
                install_command=server_data["install_command"],
                capabilities=server_data["capabilities"],
                tools=server_data["tools"],
                category=server_data["category"],
                author="Community"
            )
            servers[server_info.name] = server_info
        
        return servers
    
    async def _search_npm_packages(self, search_term: str) -> List[Dict[str, Any]]:
        """Search npm registry for packages"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://registry.npmjs.org/-/v1/search?text={search_term}&size=50"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("objects", [])
        except Exception as e:
            logger.error(f"NPM search failed for '{search_term}': {e}")
        
        return []
    
    async def _search_github_repositories(self, query: str) -> List[Dict[str, Any]]:
        """Search GitHub for repositories"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=50"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("items", [])
        except Exception as e:
            logger.error(f"GitHub search failed for '{query}': {e}")
        
        return []
    
    async def _analyze_npm_package(self, package_data: Dict[str, Any]) -> Optional[MCPServerInfo]:
        """Analyze npm package to extract MCP server information"""
        try:
            package = package_data.get("package", {})
            name = package.get("name", "")
            
            # Skip if not an MCP server
            if "mcp" not in name.lower() and "server" not in name.lower():
                return None
            
            description = package.get("description", "")
            version = package.get("version", "latest")
            
            # Determine server type and category from package info
            server_type = self._infer_server_type(name, description)
            category = self._infer_category(name, description)
            
            return MCPServerInfo(
                name=name,
                description=description,
                server_type=server_type,
                source_url=package.get("links", {}).get("repository", ""),
                install_command=["npm", "install", "-g", name],
                version=version,
                author=package.get("author", {}).get("name", "") if isinstance(package.get("author"), dict) else str(package.get("author", "")),
                category=category,
                tags=package.get("keywords", [])
            )
        
        except Exception as e:
            logger.error(f"Failed to analyze npm package: {e}")
            return None
    
    async def _analyze_github_repository(self, repo_data: Dict[str, Any]) -> Optional[MCPServerInfo]:
        """Analyze GitHub repository to extract MCP server information"""
        try:
            name = repo_data.get("name", "")
            description = repo_data.get("description", "")
            
            # Skip if not an MCP server
            if "mcp" not in name.lower() and "server" not in description.lower():
                return None
            
            # Determine server type and category
            server_type = self._infer_server_type(name, description)
            category = self._infer_category(name, description)
            
            # Determine install command based on repository language
            language = repo_data.get("language", "").lower()
            if language == "python":
                install_command = ["pip", "install", f"git+{repo_data.get('clone_url', '')}"]
            elif language == "javascript":
                install_command = ["npm", "install", f"git+{repo_data.get('clone_url', '')}"]
            else:
                install_command = ["git", "clone", repo_data.get("clone_url", "")]
            
            return MCPServerInfo(
                name=name,
                description=description,
                server_type=server_type,
                source_url=repo_data.get("html_url", ""),
                install_command=install_command,
                author=repo_data.get("owner", {}).get("login", ""),
                category=category,
                tags=repo_data.get("topics", []),
                last_updated=datetime.fromisoformat(repo_data.get("updated_at", "").replace("Z", "+00:00")) if repo_data.get("updated_at") else None
            )
        
        except Exception as e:
            logger.error(f"Failed to analyze GitHub repository: {e}")
            return None
    
    def _infer_server_type(self, name: str, description: str) -> MCPServerType:
        """Infer server type from name and description"""
        text = (name + " " + description).lower()
        
        if any(keyword in text for keyword in ["file", "filesystem", "directory"]):
            return MCPServerType.FILESYSTEM
        elif any(keyword in text for keyword in ["database", "sql", "postgres", "mongo"]):
            return MCPServerType.DATABASE
        elif any(keyword in text for keyword in ["web", "search", "brave", "google"]):
            return MCPServerType.WEB_SEARCH
        elif any(keyword in text for keyword in ["git", "github", "version"]):
            return MCPServerType.VERSION_CONTROL
        elif any(keyword in text for keyword in ["nlp", "text", "language", "spacy"]):
            return MCPServerType.NLP
        elif any(keyword in text for keyword in ["image", "graphics", "opencv", "vision"]):
            return MCPServerType.GRAPHICS
        elif any(keyword in text for keyword in ["ui", "web", "react", "streamlit"]):
            return MCPServerType.WEB_UI
        else:
            return MCPServerType.CUSTOM
    
    def _infer_category(self, name: str, description: str) -> str:
        """Infer category from name and description"""
        text = (name + " " + description).lower()
        
        for category, info in self.server_categories.items():
            if any(keyword in text for keyword in info["keywords"]):
                return category
        
        return "general"
    
    async def _save_discovery_cache(self) -> None:
        """Save discovered servers to cache"""
        try:
            cache_file = self.cache_dir / "discovered_servers.json"
            cache_data = {
                name: {
                    "name": server.name,
                    "description": server.description,
                    "server_type": server.server_type.value,
                    "source_url": server.source_url,
                    "install_command": server.install_command,
                    "capabilities": server.capabilities,
                    "tools": server.tools,
                    "version": server.version,
                    "author": server.author,
                    "license": server.license,
                    "documentation_url": server.documentation_url,
                    "category": server.category,
                    "tags": server.tags,
                    "requirements": server.requirements,
                    "config_template": server.config_template,
                    "discovered_at": server.discovered_at.isoformat(),
                    "verified": server.verified,
                    "install_size_mb": server.install_size_mb,
                    "last_updated": server.last_updated.isoformat() if server.last_updated else None
                }
                for name, server in self.discovered_servers.items()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Saved {len(cache_data)} servers to discovery cache")
        
        except Exception as e:
            logger.error(f"Failed to save discovery cache: {e}")
    
    async def load_discovery_cache(self) -> None:
        """Load discovered servers from cache"""
        try:
            cache_file = self.cache_dir / "discovered_servers.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for name, server_data in cache_data.items():
                    server_info = MCPServerInfo(
                        name=server_data["name"],
                        description=server_data["description"],
                        server_type=MCPServerType(server_data["server_type"]),
                        source_url=server_data["source_url"],
                        install_command=server_data["install_command"],
                        capabilities=server_data["capabilities"],
                        tools=server_data["tools"],
                        version=server_data["version"],
                        author=server_data["author"],
                        license=server_data["license"],
                        documentation_url=server_data["documentation_url"],
                        category=server_data["category"],
                        tags=server_data["tags"],
                        requirements=server_data["requirements"],
                        config_template=server_data["config_template"],
                        discovered_at=datetime.fromisoformat(server_data["discovered_at"]),
                        verified=server_data["verified"],
                        install_size_mb=server_data["install_size_mb"],
                        last_updated=datetime.fromisoformat(server_data["last_updated"]) if server_data["last_updated"] else None
                    )
                    self.discovered_servers[name] = server_info
                
                logger.info(f"Loaded {len(cache_data)} servers from discovery cache")
        
        except Exception as e:
            logger.error(f"Failed to load discovery cache: {e}")
    
    def get_servers_by_category(self, category: str) -> List[MCPServerInfo]:
        """Get servers by category"""
        return [server for server in self.discovered_servers.values() if server.category == category]
    
    def get_servers_by_type(self, server_type: MCPServerType) -> List[MCPServerInfo]:
        """Get servers by type"""
        return [server for server in self.discovered_servers.values() if server.server_type == server_type]
    
    def search_servers(self, query: str) -> List[MCPServerInfo]:
        """Search servers by name, description, or capabilities"""
        query = query.lower()
        results = []
        
        for server in self.discovered_servers.values():
            if (query in server.name.lower() or 
                query in server.description.lower() or
                any(query in cap.lower() for cap in server.capabilities) or
                any(query in tool.lower() for tool in server.tools) or
                any(query in tag.lower() for tag in server.tags)):
                results.append(server)
        
        return results
