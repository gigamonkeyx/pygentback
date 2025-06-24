"""
MCP Analyzer

Analyzes MCP servers to understand their capabilities, tools, and performance characteristics.
Builds comprehensive profiles for intelligent server selection and orchestration.
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of MCP server capabilities"""
    FILE_OPERATIONS = "file_operations"
    DATABASE_ACCESS = "database_access"
    WEB_SCRAPING = "web_scraping"
    API_INTEGRATION = "api_integration"
    DATA_PROCESSING = "data_processing"
    CODE_EXECUTION = "code_execution"
    NATURAL_LANGUAGE = "natural_language"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"
    SYSTEM_INTEGRATION = "system_integration"
    SECURITY_TOOLS = "security_tools"
    DEVELOPMENT_TOOLS = "development_tools"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    AUTOMATION = "automation"


class PerformanceLevel(Enum):
    """Performance levels for capabilities"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass
class MCPCapability:
    """Represents a specific capability of an MCP server"""
    capability_type: CapabilityType
    tools: List[str] = field(default_factory=list)
    performance_level: PerformanceLevel = PerformanceLevel.UNKNOWN
    reliability_score: float = 0.0
    latency_ms: float = 0.0
    success_rate: float = 0.0
    complexity_support: str = "basic"  # basic, intermediate, advanced
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServerProfile:
    """Comprehensive profile of an MCP server"""
    server_name: str
    server_version: str = "unknown"
    description: str = ""
    capabilities: List[MCPCapability] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    
    # Performance metrics
    overall_performance: PerformanceLevel = PerformanceLevel.UNKNOWN
    average_latency_ms: float = 0.0
    uptime_percentage: float = 0.0
    error_rate: float = 0.0
    
    # Resource information
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0
    
    # Compatibility and requirements
    supported_protocols: List[str] = field(default_factory=list)
    required_dependencies: List[str] = field(default_factory=list)
    operating_systems: List[str] = field(default_factory=list)
    
    # Analysis metadata
    last_analyzed: datetime = field(default_factory=datetime.utcnow)
    analysis_confidence: float = 0.0
    tags: Set[str] = field(default_factory=set)
    
    def get_capability_by_type(self, capability_type: CapabilityType) -> Optional[MCPCapability]:
        """Get capability by type"""
        for capability in self.capabilities:
            if capability.capability_type == capability_type:
                return capability
        return None
    
    def has_capability(self, capability_type: CapabilityType) -> bool:
        """Check if server has specific capability"""
        return self.get_capability_by_type(capability_type) is not None
    
    def get_tools_for_capability(self, capability_type: CapabilityType) -> List[str]:
        """Get tools that support a specific capability"""
        capability = self.get_capability_by_type(capability_type)
        return capability.tools if capability else []


class MCPAnalyzer:
    """
    Analyzes MCP servers to build comprehensive capability profiles.
    
    Uses various analysis techniques including tool inspection, performance testing,
    and capability inference to understand server characteristics.
    """
    
    def __init__(self):
        # Analysis patterns for capability detection
        self.capability_patterns = self._initialize_capability_patterns()
        
        # Tool categorization
        self.tool_categories = self._initialize_tool_categories()
        
        # Performance benchmarks
        self.performance_benchmarks = self._initialize_performance_benchmarks()
        
        # Analysis cache
        self.analysis_cache: Dict[str, MCPServerProfile] = {}
        self.cache_ttl = timedelta(hours=24)
    
    def _initialize_capability_patterns(self) -> Dict[CapabilityType, List[str]]:
        """Initialize patterns for capability detection"""
        return {
            CapabilityType.FILE_OPERATIONS: [
                r"file.*read", r"file.*write", r"directory", r"path", r"folder",
                r"upload", r"download", r"copy", r"move", r"delete"
            ],
            CapabilityType.DATABASE_ACCESS: [
                r"database", r"sql", r"query", r"table", r"record", r"schema",
                r"mysql", r"postgres", r"sqlite", r"mongodb", r"redis"
            ],
            CapabilityType.WEB_SCRAPING: [
                r"scrape", r"crawl", r"extract", r"html", r"web.*page",
                r"selenium", r"beautifulsoup", r"requests", r"http.*get"
            ],
            CapabilityType.API_INTEGRATION: [
                r"api", r"rest", r"graphql", r"webhook", r"endpoint",
                r"http.*request", r"json", r"xml", r"soap"
            ],
            CapabilityType.DATA_PROCESSING: [
                r"process.*data", r"transform", r"parse", r"convert",
                r"csv", r"json", r"xml", r"excel", r"pandas"
            ],
            CapabilityType.CODE_EXECUTION: [
                r"execute", r"run.*code", r"python", r"javascript", r"shell",
                r"script", r"compile", r"interpret"
            ],
            CapabilityType.NATURAL_LANGUAGE: [
                r"nlp", r"text.*process", r"language", r"sentiment",
                r"translate", r"summarize", r"tokenize", r"embedding"
            ],
            CapabilityType.IMAGE_PROCESSING: [
                r"image", r"photo", r"picture", r"visual", r"opencv",
                r"resize", r"crop", r"filter", r"recognition"
            ],
            CapabilityType.AUDIO_PROCESSING: [
                r"audio", r"sound", r"music", r"speech", r"voice",
                r"transcribe", r"synthesize", r"frequency"
            ],
            CapabilityType.SYSTEM_INTEGRATION: [
                r"system", r"os", r"process", r"service", r"daemon",
                r"registry", r"environment", r"configuration"
            ],
            CapabilityType.SECURITY_TOOLS: [
                r"security", r"encrypt", r"decrypt", r"hash", r"certificate",
                r"authentication", r"authorization", r"vulnerability"
            ],
            CapabilityType.DEVELOPMENT_TOOLS: [
                r"git", r"version.*control", r"build", r"deploy", r"test",
                r"debug", r"lint", r"format", r"documentation"
            ],
            CapabilityType.COMMUNICATION: [
                r"email", r"sms", r"chat", r"message", r"notification",
                r"slack", r"discord", r"telegram", r"webhook"
            ],
            CapabilityType.ANALYTICS: [
                r"analytics", r"metrics", r"statistics", r"report",
                r"dashboard", r"visualization", r"chart", r"graph"
            ],
            CapabilityType.AUTOMATION: [
                r"automate", r"schedule", r"workflow", r"trigger",
                r"cron", r"batch", r"pipeline", r"orchestrate"
            ]
        }
    
    def _initialize_tool_categories(self) -> Dict[str, CapabilityType]:
        """Initialize tool to capability mapping"""
        return {
            # File operations
            "read_file": CapabilityType.FILE_OPERATIONS,
            "write_file": CapabilityType.FILE_OPERATIONS,
            "list_directory": CapabilityType.FILE_OPERATIONS,
            "create_directory": CapabilityType.FILE_OPERATIONS,
            
            # Database
            "execute_query": CapabilityType.DATABASE_ACCESS,
            "connect_database": CapabilityType.DATABASE_ACCESS,
            "create_table": CapabilityType.DATABASE_ACCESS,
            
            # Web scraping
            "scrape_webpage": CapabilityType.WEB_SCRAPING,
            "extract_links": CapabilityType.WEB_SCRAPING,
            "get_page_content": CapabilityType.WEB_SCRAPING,
            
            # API integration
            "make_api_request": CapabilityType.API_INTEGRATION,
            "call_rest_api": CapabilityType.API_INTEGRATION,
            "send_webhook": CapabilityType.API_INTEGRATION,
            
            # Data processing
            "process_csv": CapabilityType.DATA_PROCESSING,
            "parse_json": CapabilityType.DATA_PROCESSING,
            "transform_data": CapabilityType.DATA_PROCESSING,
            
            # Code execution
            "execute_python": CapabilityType.CODE_EXECUTION,
            "run_script": CapabilityType.CODE_EXECUTION,
            "compile_code": CapabilityType.CODE_EXECUTION,
            
            # Natural language
            "analyze_sentiment": CapabilityType.NATURAL_LANGUAGE,
            "translate_text": CapabilityType.NATURAL_LANGUAGE,
            "summarize_text": CapabilityType.NATURAL_LANGUAGE,
            
            # Image processing
            "resize_image": CapabilityType.IMAGE_PROCESSING,
            "detect_objects": CapabilityType.IMAGE_PROCESSING,
            "apply_filter": CapabilityType.IMAGE_PROCESSING,
            
            # Communication
            "send_email": CapabilityType.COMMUNICATION,
            "send_sms": CapabilityType.COMMUNICATION,
            "post_message": CapabilityType.COMMUNICATION
        }
    
    def _initialize_performance_benchmarks(self) -> Dict[CapabilityType, Dict[str, float]]:
        """Initialize performance benchmarks for capabilities"""
        return {
            CapabilityType.FILE_OPERATIONS: {
                "excellent_latency": 50.0,
                "good_latency": 200.0,
                "average_latency": 500.0,
                "min_success_rate": 0.95
            },
            CapabilityType.DATABASE_ACCESS: {
                "excellent_latency": 100.0,
                "good_latency": 500.0,
                "average_latency": 1000.0,
                "min_success_rate": 0.98
            },
            CapabilityType.WEB_SCRAPING: {
                "excellent_latency": 1000.0,
                "good_latency": 3000.0,
                "average_latency": 5000.0,
                "min_success_rate": 0.90
            },
            CapabilityType.API_INTEGRATION: {
                "excellent_latency": 200.0,
                "good_latency": 1000.0,
                "average_latency": 2000.0,
                "min_success_rate": 0.95
            },
            CapabilityType.DATA_PROCESSING: {
                "excellent_latency": 500.0,
                "good_latency": 2000.0,
                "average_latency": 5000.0,
                "min_success_rate": 0.98
            }
        }
    
    async def analyze_server(self, server_name: str, server_info: Dict[str, Any]) -> MCPServerProfile:
        """
        Analyze an MCP server and create comprehensive profile.
        
        Args:
            server_name: Name of the MCP server
            server_info: Server information and metadata
            
        Returns:
            Comprehensive server profile
        """
        try:
            # Check cache first
            if server_name in self.analysis_cache:
                cached_profile = self.analysis_cache[server_name]
                if datetime.utcnow() - cached_profile.last_analyzed < self.cache_ttl:
                    logger.debug(f"Using cached analysis for server: {server_name}")
                    return cached_profile
            
            logger.info(f"Analyzing MCP server: {server_name}")
            
            # Create base profile
            profile = MCPServerProfile(
                server_name=server_name,
                server_version=server_info.get('version', 'unknown'),
                description=server_info.get('description', ''),
                last_analyzed=datetime.utcnow()
            )
            
            # Extract available tools
            tools = server_info.get('tools', [])
            if isinstance(tools, dict):
                profile.available_tools = list(tools.keys())
            elif isinstance(tools, list):
                profile.available_tools = tools
            
            # Analyze capabilities
            profile.capabilities = await self._analyze_capabilities(profile.available_tools, server_info)
            
            # Analyze performance
            await self._analyze_performance(profile, server_info)
            
            # Extract metadata
            self._extract_metadata(profile, server_info)
            
            # Calculate analysis confidence
            profile.analysis_confidence = self._calculate_analysis_confidence(profile)
            
            # Generate tags
            profile.tags = self._generate_tags(profile)
            
            # Cache the result
            self.analysis_cache[server_name] = profile
            
            logger.info(f"Analysis completed for {server_name}: "
                       f"{len(profile.capabilities)} capabilities, "
                       f"confidence: {profile.analysis_confidence:.2f}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to analyze server {server_name}: {e}")
            # Return minimal profile
            return MCPServerProfile(
                server_name=server_name,
                description=f"Analysis failed: {e}",
                analysis_confidence=0.0
            )
    
    async def _analyze_capabilities(self, tools: List[str], server_info: Dict[str, Any]) -> List[MCPCapability]:
        """Analyze server capabilities based on available tools"""
        capabilities = {}
        
        # Analyze each tool
        for tool_name in tools:
            # Direct mapping from tool categories
            if tool_name in self.tool_categories:
                capability_type = self.tool_categories[tool_name]
                if capability_type not in capabilities:
                    capabilities[capability_type] = MCPCapability(capability_type=capability_type)
                capabilities[capability_type].tools.append(tool_name)
            
            # Pattern-based detection
            for capability_type, patterns in self.capability_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, tool_name.lower()):
                        if capability_type not in capabilities:
                            capabilities[capability_type] = MCPCapability(capability_type=capability_type)
                        if tool_name not in capabilities[capability_type].tools:
                            capabilities[capability_type].tools.append(tool_name)
        
        # Analyze server description for additional capabilities
        description = server_info.get('description', '').lower()
        for capability_type, patterns in self.capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, description):
                    if capability_type not in capabilities:
                        capabilities[capability_type] = MCPCapability(capability_type=capability_type)
        
        # Set complexity support based on number of tools
        for capability in capabilities.values():
            tool_count = len(capability.tools)
            if tool_count >= 5:
                capability.complexity_support = "advanced"
            elif tool_count >= 2:
                capability.complexity_support = "intermediate"
            else:
                capability.complexity_support = "basic"
        
        return list(capabilities.values())
    
    async def _analyze_performance(self, profile: MCPServerProfile, server_info: Dict[str, Any]):
        """Analyze server performance characteristics"""
        # Extract performance metrics from server info
        performance_data = server_info.get('performance', {})
        
        profile.average_latency_ms = performance_data.get('average_latency_ms', 0.0)
        profile.uptime_percentage = performance_data.get('uptime_percentage', 0.0)
        profile.error_rate = performance_data.get('error_rate', 0.0)
        
        # Resource usage
        resources = server_info.get('resources', {})
        profile.memory_usage_mb = resources.get('memory_mb', 0.0)
        profile.cpu_usage_percent = resources.get('cpu_percent', 0.0)
        profile.disk_usage_mb = resources.get('disk_mb', 0.0)
        
        # Determine overall performance level
        if profile.average_latency_ms > 0:
            if profile.average_latency_ms < 200 and profile.error_rate < 0.01:
                profile.overall_performance = PerformanceLevel.EXCELLENT
            elif profile.average_latency_ms < 1000 and profile.error_rate < 0.05:
                profile.overall_performance = PerformanceLevel.GOOD
            elif profile.average_latency_ms < 3000 and profile.error_rate < 0.10:
                profile.overall_performance = PerformanceLevel.AVERAGE
            else:
                profile.overall_performance = PerformanceLevel.POOR
        
        # Analyze capability-specific performance
        for capability in profile.capabilities:
            await self._analyze_capability_performance(capability, performance_data)
    
    async def _analyze_capability_performance(self, capability: MCPCapability, performance_data: Dict[str, Any]):
        """Analyze performance for a specific capability"""
        capability_name = capability.capability_type.value
        capability_perf = performance_data.get(capability_name, {})
        
        capability.latency_ms = capability_perf.get('latency_ms', 0.0)
        capability.success_rate = capability_perf.get('success_rate', 0.0)
        capability.reliability_score = capability_perf.get('reliability_score', 0.0)
        
        # Determine performance level based on benchmarks
        benchmarks = self.performance_benchmarks.get(capability.capability_type, {})
        
        if capability.latency_ms > 0 and benchmarks:
            if (capability.latency_ms <= benchmarks.get('excellent_latency', 100) and
                capability.success_rate >= benchmarks.get('min_success_rate', 0.95)):
                capability.performance_level = PerformanceLevel.EXCELLENT
            elif (capability.latency_ms <= benchmarks.get('good_latency', 500) and
                  capability.success_rate >= benchmarks.get('min_success_rate', 0.90)):
                capability.performance_level = PerformanceLevel.GOOD
            elif (capability.latency_ms <= benchmarks.get('average_latency', 1000) and
                  capability.success_rate >= 0.80):
                capability.performance_level = PerformanceLevel.AVERAGE
            else:
                capability.performance_level = PerformanceLevel.POOR
    
    def _extract_metadata(self, profile: MCPServerProfile, server_info: Dict[str, Any]):
        """Extract additional metadata from server info"""
        # Supported protocols
        protocols = server_info.get('protocols', [])
        if isinstance(protocols, list):
            profile.supported_protocols = protocols
        
        # Dependencies
        dependencies = server_info.get('dependencies', [])
        if isinstance(dependencies, list):
            profile.required_dependencies = dependencies
        
        # Operating systems
        os_support = server_info.get('operating_systems', [])
        if isinstance(os_support, list):
            profile.operating_systems = os_support
        
        # Default values if not specified
        if not profile.supported_protocols:
            profile.supported_protocols = ['mcp']
        
        if not profile.operating_systems:
            profile.operating_systems = ['linux', 'windows', 'macos']
    
    def _calculate_analysis_confidence(self, profile: MCPServerProfile) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.0
        
        # Base confidence from available information
        if profile.available_tools:
            confidence += 0.3
        
        if profile.description:
            confidence += 0.2
        
        if profile.capabilities:
            confidence += 0.3
        
        # Performance data availability
        if profile.average_latency_ms > 0:
            confidence += 0.1
        
        if profile.uptime_percentage > 0:
            confidence += 0.1
        
        # Capability detail level
        detailed_capabilities = sum(1 for cap in profile.capabilities if cap.tools)
        if detailed_capabilities > 0:
            confidence += min(0.2, detailed_capabilities * 0.05)
        
        return min(1.0, confidence)
    
    def _generate_tags(self, profile: MCPServerProfile) -> Set[str]:
        """Generate tags for the server profile"""
        tags = set()
        
        # Capability-based tags
        for capability in profile.capabilities:
            tags.add(capability.capability_type.value)
            
            # Performance tags
            if capability.performance_level != PerformanceLevel.UNKNOWN:
                tags.add(f"{capability.capability_type.value}_{capability.performance_level.value}")
            
            # Complexity tags
            tags.add(f"{capability.capability_type.value}_{capability.complexity_support}")
        
        # Performance tags
        if profile.overall_performance != PerformanceLevel.UNKNOWN:
            tags.add(f"performance_{profile.overall_performance.value}")
        
        # Resource tags
        if profile.memory_usage_mb > 1000:
            tags.add("high_memory")
        elif profile.memory_usage_mb > 0:
            tags.add("low_memory")
        
        if profile.cpu_usage_percent > 50:
            tags.add("high_cpu")
        elif profile.cpu_usage_percent > 0:
            tags.add("low_cpu")
        
        # Tool count tags
        tool_count = len(profile.available_tools)
        if tool_count > 20:
            tags.add("many_tools")
        elif tool_count > 10:
            tags.add("moderate_tools")
        elif tool_count > 0:
            tags.add("few_tools")
        
        return tags
    
    def get_servers_by_capability(self, capability_type: CapabilityType) -> List[MCPServerProfile]:
        """Get all servers that have a specific capability"""
        matching_servers = []
        
        for profile in self.analysis_cache.values():
            if profile.has_capability(capability_type):
                matching_servers.append(profile)
        
        # Sort by performance and confidence
        matching_servers.sort(
            key=lambda p: (
                p.get_capability_by_type(capability_type).performance_level.value,
                p.analysis_confidence
            ),
            reverse=True
        )
        
        return matching_servers
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about analyzed servers"""
        if not self.analysis_cache:
            return {"error": "No servers analyzed"}
        
        profiles = list(self.analysis_cache.values())
        
        # Capability distribution
        capability_counts = {}
        for profile in profiles:
            for capability in profile.capabilities:
                cap_type = capability.capability_type.value
                capability_counts[cap_type] = capability_counts.get(cap_type, 0) + 1
        
        # Performance distribution
        performance_counts = {}
        for profile in profiles:
            perf_level = profile.overall_performance.value
            performance_counts[perf_level] = performance_counts.get(perf_level, 0) + 1
        
        return {
            "total_servers": len(profiles),
            "average_confidence": sum(p.analysis_confidence for p in profiles) / len(profiles),
            "capability_distribution": capability_counts,
            "performance_distribution": performance_counts,
            "total_capabilities": sum(len(p.capabilities) for p in profiles),
            "total_tools": sum(len(p.available_tools) for p in profiles),
            "cache_size": len(self.analysis_cache)
        }
