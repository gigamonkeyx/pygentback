#!/usr/bin/env python3
"""
Automated Feature Registry System

This system prevents the "forgotten features" problem by:
1. Automatically discovering and cataloging all project features
2. Tracking feature implementation status and documentation
3. Generating living documentation that stays current
4. Alerting when features become orphaned or undocumented
5. Providing API endpoints for feature discovery and management

This is designed to be a core part of the development workflow.
"""

import json
import logging
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import aiofiles


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features that can be discovered"""
    API_ENDPOINT = "api_endpoint"
    MCP_SERVER = "mcp_server"
    CLOUDFLARE_WORKER = "cloudflare_worker"
    UI_COMPONENT = "ui_component"
    DATABASE_MODEL = "database_model"
    CONFIGURATION = "configuration"
    UTILITY_SCRIPT = "utility_script"
    DOCUMENTATION = "documentation"
    TEST_SUITE = "test_suite"


class FeatureStatus(Enum):
    """Feature implementation and documentation status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ORPHANED = "orphaned"
    DOCUMENTED = "documented"
    UNDOCUMENTED = "undocumented"


@dataclass
class Feature:
    """Represents a discovered feature in the system"""
    name: str
    type: FeatureType
    status: FeatureStatus
    file_path: str
    description: str
    dependencies: List[str]
    last_modified: datetime
    discovered_at: datetime
    documentation_path: Optional[str] = None
    api_route: Optional[str] = None
    config_reference: Optional[str] = None
    tests_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert enums to strings
        data['type'] = self.type.value
        data['status'] = self.status.value
        # Convert datetime to ISO format
        data['last_modified'] = self.last_modified.isoformat()
        data['discovered_at'] = self.discovered_at.isoformat()
        return data


class FeatureRegistry:
    """
    Main feature registry that discovers, tracks, and manages all project features
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.registry_file = self.project_root / "feature_registry.json"
        self.features: Dict[str, Feature] = {}
        
    async def discover_all_features(self) -> Dict[str, List[Feature]]:
        """
        Comprehensive feature discovery across the entire project
        """
        logger.info("Starting comprehensive feature discovery...")
        
        features_by_type = defaultdict(list)
        
        # Discover different types of features
        api_features = await self._discover_api_endpoints()
        mcp_features = await self._discover_mcp_servers()
        cloudflare_features = await self._discover_cloudflare_workers()
        ui_features = await self._discover_ui_components()
        db_features = await self._discover_database_models()
        config_features = await self._discover_configurations()
        script_features = await self._discover_utility_scripts()
        doc_features = await self._discover_documentation()
        test_features = await self._discover_test_suites()
        
        # Categorize features
        for feature_list, feature_type in [
            (api_features, FeatureType.API_ENDPOINT),
            (mcp_features, FeatureType.MCP_SERVER),
            (cloudflare_features, FeatureType.CLOUDFLARE_WORKER),
            (ui_features, FeatureType.UI_COMPONENT),
            (db_features, FeatureType.DATABASE_MODEL),
            (config_features, FeatureType.CONFIGURATION),
            (script_features, FeatureType.UTILITY_SCRIPT),
            (doc_features, FeatureType.DOCUMENTATION),
            (test_features, FeatureType.TEST_SUITE)
        ]:
            for feature in feature_list:
                features_by_type[feature_type].append(feature)
                self.features[feature.name] = feature
        
        logger.info(f"Discovered {len(self.features)} features across {len(features_by_type)} categories")
        return dict(features_by_type)
    
    async def _discover_api_endpoints(self) -> List[Feature]:
        """Discover FastAPI endpoints"""
        features = []
        
        # Look for FastAPI route definitions
        api_files = [
            "src/api/main.py",
            "src/api/routes/",
        ]
        
        for api_path in api_files:
            full_path = self.project_root / api_path
            if full_path.exists():
                if full_path.is_file():
                    endpoints = await self._extract_fastapi_routes(full_path)
                    features.extend(endpoints)
                elif full_path.is_dir():
                    for py_file in full_path.rglob("*.py"):
                        endpoints = await self._extract_fastapi_routes(py_file)
                        features.extend(endpoints)
        
        return features
    
    async def _extract_fastapi_routes(self, file_path: Path) -> List[Feature]:
        """Extract FastAPI route definitions from a Python file"""
        features = []
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Simple pattern matching for FastAPI routes
            import re
            route_patterns = [
                r'@app\.(get|post|put|delete|patch)\("([^"]+)"',
                r'@router\.(get|post|put|delete|patch)\("([^"]+)"',
            ]
            
            for pattern in route_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for method, route in matches:
                    feature_name = f"{method.upper()} {route}"
                    
                    feature = Feature(
                        name=feature_name,
                        type=FeatureType.API_ENDPOINT,
                        status=FeatureStatus.ACTIVE,
                        file_path=str(file_path),
                        description=f"FastAPI {method.upper()} endpoint: {route}",
                        dependencies=[],
                        last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                        discovered_at=datetime.utcnow(),
                        api_route=route                    )
                    features.append(feature)
        
        except Exception as e:
            logger.warning(f"Error extracting routes from {file_path}: {e}")
        
        return features
    
    async def _discover_mcp_servers(self) -> List[Feature]:
        """Discover MCP servers (both local and configuration-based)"""
        features = []
        
        # Check mcp-servers directory for local servers
        mcp_servers_dir = self.project_root / "mcp-servers"
        if mcp_servers_dir.exists():
            try:
                # Use iterdir instead of rglob to avoid circular symlinks
                for server_dir in mcp_servers_dir.iterdir():
                    if server_dir.is_dir() and not server_dir.name.startswith('.'):
                        # Look for src subdirectory structure
                        src_dir = server_dir / "src"
                        if src_dir.exists():
                            for potential_server in src_dir.iterdir():
                                if potential_server.is_dir() and (potential_server / "pyproject.toml").exists():
                                    features.append(Feature(
                                        name=f"MCP Server: {potential_server.name}",
                                        type=FeatureType.MCP_SERVER,
                                        status=FeatureStatus.ACTIVE,
                                        file_path=str(potential_server),
                                        description=f"Local MCP server implementation: {potential_server.name}",
                                        dependencies=[],
                                        last_modified=datetime.fromtimestamp(potential_server.stat().st_mtime),
                                        discovered_at=datetime.utcnow(),
                                        config_reference="mcp_server_configs.json"
                                    ))
                        elif (server_dir / "pyproject.toml").exists():
                            # Server directly in directory
                            features.append(Feature(
                                name=f"MCP Server: {server_dir.name}",
                                type=FeatureType.MCP_SERVER,
                                status=FeatureStatus.ACTIVE,
                                file_path=str(server_dir),
                                description=f"Local MCP server implementation: {server_dir.name}",
                                dependencies=[],
                                last_modified=datetime.fromtimestamp(server_dir.stat().st_mtime),
                                discovered_at=datetime.utcnow(),
                                config_reference="mcp_server_configs.json"
                            ))
            except (OSError, PermissionError) as e:
                logger.warning(f"Error scanning mcp-servers directory: {e}")
        
        # Check configuration file for registered servers
        config_file = self.project_root / "mcp_server_configs.json"
        if config_file.exists():
            try:
                async with aiofiles.open(config_file, 'r') as f:
                    config_content = await f.read()
                config_data = json.loads(config_content)
                
                for server_name, server_config in config_data.get("servers", {}).items():
                    features.append(Feature(
                        name=f"MCP Server Config: {server_name}",
                        type=FeatureType.MCP_SERVER,
                        status=FeatureStatus.ACTIVE,
                        file_path=str(config_file),
                        description=f"Configured MCP server: {server_name}",
                        dependencies=[],
                        last_modified=datetime.fromtimestamp(config_file.stat().st_mtime),
                        discovered_at=datetime.utcnow(),
                        config_reference=str(config_file)
                    ))
            except Exception as e:
                logger.warning(f"Error reading MCP config: {e}")
        
        return features
    
    async def _discover_cloudflare_workers(self) -> List[Feature]:
        """Discover Cloudflare Worker MCP servers"""
        features = []
        
        cf_dir = self.project_root / "mcp-server-cloudflare" / "apps"
        if cf_dir.exists():
            for app_dir in cf_dir.iterdir():
                if app_dir.is_dir():
                    # Check for wrangler config
                    wrangler_files = list(app_dir.glob("wrangler.*"))
                    if wrangler_files:
                        features.append(Feature(
                            name=f"Cloudflare Worker: {app_dir.name}",
                            type=FeatureType.CLOUDFLARE_WORKER,
                            status=FeatureStatus.ACTIVE,
                            file_path=str(app_dir),
                            description=f"Cloudflare Worker MCP server: {app_dir.name}",
                            dependencies=["cloudflare", "wrangler"],
                            last_modified=datetime.fromtimestamp(app_dir.stat().st_mtime),
                            discovered_at=datetime.utcnow(),
                            config_reference=str(wrangler_files[0])
                        ))
        
        return features
    
    async def _discover_ui_components(self) -> List[Feature]:
        """Discover React UI components"""
        features = []
        
        # Look for React components
        frontend_dir = self.project_root / "frontend" / "src"
        if frontend_dir.exists():
            for tsx_file in frontend_dir.rglob("*.tsx"):
                if tsx_file.stem.startswith(tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')):
                    # Likely a React component (starts with capital letter)
                    features.append(Feature(
                        name=f"React Component: {tsx_file.stem}",
                        type=FeatureType.UI_COMPONENT,
                        status=FeatureStatus.ACTIVE,
                        file_path=str(tsx_file),
                        description=f"React TypeScript component: {tsx_file.stem}",
                        dependencies=["react", "typescript"],
                        last_modified=datetime.fromtimestamp(tsx_file.stat().st_mtime),
                        discovered_at=datetime.utcnow()
                    ))
        
        return features
    
    async def _discover_database_models(self) -> List[Feature]:
        """Discover database models and schemas"""
        features = []
        
        # Look for SQLAlchemy models
        models_patterns = [
            "src/database/models",
            "src/models",
            "src/api/models",
        ]
        
        for pattern in models_patterns:
            models_dir = self.project_root / pattern
            if models_dir.exists():
                for py_file in models_dir.rglob("*.py"):
                    if py_file.name != "__init__.py":
                        features.append(Feature(
                            name=f"Database Model: {py_file.stem}",
                            type=FeatureType.DATABASE_MODEL,
                            status=FeatureStatus.ACTIVE,
                            file_path=str(py_file),
                            description=f"Database model definition: {py_file.stem}",
                            dependencies=["sqlalchemy"],
                            last_modified=datetime.fromtimestamp(py_file.stat().st_mtime),
                            discovered_at=datetime.utcnow()
                        ))
        
        return features
    
    async def _discover_configurations(self) -> List[Feature]:
        """Discover configuration files"""
        features = []
        
        config_patterns = [
            "*.json", "*.yaml", "*.yml", "*.toml", "*.env*",
            "*.config.js", "*.config.ts", "docker-compose.yml",
            "Dockerfile", "requirements.txt", "package.json"
        ]
        
        for pattern in config_patterns:
            for config_file in self.project_root.glob(pattern):
                if config_file.is_file():
                    features.append(Feature(
                        name=f"Configuration: {config_file.name}",
                        type=FeatureType.CONFIGURATION,
                        status=FeatureStatus.ACTIVE,
                        file_path=str(config_file),
                        description=f"Configuration file: {config_file.name}",
                        dependencies=[],
                        last_modified=datetime.fromtimestamp(config_file.stat().st_mtime),
                        discovered_at=datetime.utcnow()
                    ))
        
        return features
    
    async def _discover_utility_scripts(self) -> List[Feature]:
        """Discover utility and analysis scripts"""
        features = []
        
        # Look for Python scripts in root directory
        for py_file in self.project_root.glob("*.py"):
            if py_file.is_file():
                features.append(Feature(
                    name=f"Utility Script: {py_file.stem}",
                    type=FeatureType.UTILITY_SCRIPT,
                    status=FeatureStatus.ACTIVE,
                    file_path=str(py_file),
                    description=f"Utility/analysis script: {py_file.stem}",
                    dependencies=[],
                    last_modified=datetime.fromtimestamp(py_file.stat().st_mtime),
                    discovered_at=datetime.utcnow()
                ))
        
        return features
    
    async def _discover_documentation(self) -> List[Feature]:
        """Discover documentation files"""
        features = []
        
        doc_patterns = ["*.md", "*.rst", "*.txt"]
        doc_dirs = ["docs", "documentation", "."]
        
        for doc_dir_name in doc_dirs:
            doc_dir = self.project_root / doc_dir_name
            if doc_dir.exists():
                for pattern in doc_patterns:
                    for doc_file in doc_dir.glob(pattern):
                        if doc_file.is_file():
                            features.append(Feature(
                                name=f"Documentation: {doc_file.stem}",
                                type=FeatureType.DOCUMENTATION,
                                status=FeatureStatus.ACTIVE,
                                file_path=str(doc_file),
                                description=f"Documentation file: {doc_file.name}",
                                dependencies=[],
                                last_modified=datetime.fromtimestamp(doc_file.stat().st_mtime),
                                discovered_at=datetime.utcnow()
                            ))
        
        return features
    
    async def _discover_test_suites(self) -> List[Feature]:
        """Discover test files and suites"""
        features = []
        
        test_patterns = ["test_*.py", "*_test.py", "*.test.js", "*.test.ts"]
        test_dirs = ["tests", "test", "frontend/src/__tests__"]
        
        for test_dir_name in test_dirs:
            test_dir = self.project_root / test_dir_name
            if test_dir.exists():
                for pattern in test_patterns:
                    for test_file in test_dir.rglob(pattern):
                        if test_file.is_file():
                            features.append(Feature(
                                name=f"Test Suite: {test_file.stem}",
                                type=FeatureType.TEST_SUITE,
                                status=FeatureStatus.ACTIVE,
                                file_path=str(test_file),
                                description=f"Test suite: {test_file.name}",
                                dependencies=[],
                                last_modified=datetime.fromtimestamp(test_file.stat().st_mtime),
                                discovered_at=datetime.utcnow()
                            ))
        
        # Also check root directory for test files
        for pattern in test_patterns:
            for test_file in self.project_root.glob(pattern):
                if test_file.is_file():
                    features.append(Feature(
                        name=f"Test Suite: {test_file.stem}",
                        type=FeatureType.TEST_SUITE,
                        status=FeatureStatus.ACTIVE,
                        file_path=str(test_file),
                        description=f"Test suite: {test_file.name}",
                        dependencies=[],
                        last_modified=datetime.fromtimestamp(test_file.stat().st_mtime),
                        discovered_at=datetime.utcnow()
                    ))
        
        return features
    
    async def analyze_feature_health(self) -> Dict[str, Any]:
        """
        Analyze the health of features and identify issues
        """
        analysis = {
            "total_features": len(self.features),
            "by_type": defaultdict(int),
            "by_status": defaultdict(int),
            "orphaned_features": [],
            "undocumented_features": [],
            "stale_features": [],
            "missing_tests": [],
            "recommendations": []
        }
        
        stale_threshold = datetime.utcnow() - timedelta(days=90)
        
        for feature in self.features.values():
            analysis["by_type"][feature.type.value] += 1
            analysis["by_status"][feature.status.value] += 1
            
            # Check for orphaned features (no dependencies or references)
            if not feature.dependencies and not feature.tests_path:
                analysis["orphaned_features"].append(feature.name)
            
            # Check for undocumented features
            if not feature.documentation_path:
                analysis["undocumented_features"].append(feature.name)
            
            # Check for stale features
            if feature.last_modified < stale_threshold:
                analysis["stale_features"].append(feature.name)
            
            # Check for missing tests
            if not feature.tests_path and feature.type in [
                FeatureType.API_ENDPOINT, 
                FeatureType.MCP_SERVER,
                FeatureType.UI_COMPONENT
            ]:
                analysis["missing_tests"].append(feature.name)
        
        # Generate recommendations
        if analysis["undocumented_features"]:
            analysis["recommendations"].append(
                f"Document {len(analysis['undocumented_features'])} undocumented features"
            )
        
        if analysis["missing_tests"]:
            analysis["recommendations"].append(
                f"Add tests for {len(analysis['missing_tests'])} features"
            )
        
        if analysis["stale_features"]:
            analysis["recommendations"].append(
                f"Review {len(analysis['stale_features'])} stale features for deprecation"
            )
        
        return dict(analysis)
    
    async def save_registry(self) -> None:
        """Save the feature registry to disk"""
        registry_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_features": len(self.features),
            "features": {name: feature.to_dict() for name, feature in self.features.items()}
        }
        
        async with aiofiles.open(self.registry_file, 'w') as f:
            await f.write(json.dumps(registry_data, indent=2))
        
        logger.info(f"Feature registry saved to {self.registry_file}")
    
    async def load_registry(self) -> None:
        """Load the feature registry from disk"""
        if not self.registry_file.exists():
            return
        
        try:
            async with aiofiles.open(self.registry_file, 'r') as f:
                content = await f.read()
            registry_data = json.loads(content)
            
            for name, feature_data in registry_data.get("features", {}).items():
                # Convert back from dictionary
                feature_data["type"] = FeatureType(feature_data["type"])
                feature_data["status"] = FeatureStatus(feature_data["status"])
                feature_data["last_modified"] = datetime.fromisoformat(feature_data["last_modified"])
                feature_data["discovered_at"] = datetime.fromisoformat(feature_data["discovered_at"])
                
                self.features[name] = Feature(**feature_data)
            
            logger.info(f"Loaded {len(self.features)} features from registry")
        
        except Exception as e:
            logger.warning(f"Error loading feature registry: {e}")
    
    async def generate_documentation(self) -> str:
        """Generate comprehensive documentation from the feature registry"""
        doc_content = [
            "# PyGent Factory - Complete Feature Registry",
            "",
            f"**Generated**: {datetime.utcnow().isoformat()}",
            f"**Total Features**: {len(self.features)}",
            "",
            "This document provides a comprehensive overview of all features in the PyGent Factory project.",
            "It is automatically generated and should not be manually edited.",
            "",
        ]
        
        # Group features by type
        features_by_type = defaultdict(list)
        for feature in self.features.values():
            features_by_type[feature.type].append(feature)
        
        for feature_type, features in features_by_type.items():
            doc_content.extend([
                f"## {feature_type.value.replace('_', ' ').title()}",
                "",
                f"**Count**: {len(features)}",
                ""
            ])
            
            for feature in sorted(features, key=lambda x: x.name):
                doc_content.extend([
                    f"### {feature.name}",
                    "",
                    f"- **Status**: {feature.status.value}",
                    f"- **File**: `{feature.file_path}`",
                    f"- **Description**: {feature.description}",
                    f"- **Last Modified**: {feature.last_modified.strftime('%Y-%m-%d %H:%M:%S')}",
                ])
                
                if feature.api_route:
                    doc_content.append(f"- **API Route**: `{feature.api_route}`")
                
                if feature.config_reference:
                    doc_content.append(f"- **Config**: `{feature.config_reference}`")
                
                if feature.dependencies:
                    doc_content.append(f"- **Dependencies**: {', '.join(feature.dependencies)}")
                
                doc_content.append("")
        
        return "\n".join(doc_content)


async def main():
    """Main function to run feature discovery and analysis"""
    logger.info("Starting PyGent Factory Feature Registry System")
    
    registry = FeatureRegistry(".")
    
    # Load existing registry
    await registry.load_registry()
    
    # Discover all features
    features_by_type = await registry.discover_all_features()
    
    # Analyze feature health
    health_analysis = await registry.analyze_feature_health()
    
    # Save updated registry
    await registry.save_registry()
    
    # Generate documentation
    documentation = await registry.generate_documentation()
    
    # Save documentation
    doc_path = Path("docs/COMPLETE_FEATURE_REGISTRY.md")
    doc_path.parent.mkdir(exist_ok=True)
    async with aiofiles.open(doc_path, 'w') as f:
        await f.write(documentation)
    
    # Save health analysis
    health_path = Path("feature_health_analysis.json")
    async with aiofiles.open(health_path, 'w') as f:
        await f.write(json.dumps(health_analysis, indent=2))
    
    # Print summary
    print(f"""
=== PyGent Factory Feature Registry Complete ===

Discovered Features: {len(registry.features)}
Feature Categories: {len(features_by_type)}

Health Analysis:
- Undocumented: {len(health_analysis['undocumented_features'])}
- Missing Tests: {len(health_analysis['missing_tests'])}
- Stale Features: {len(health_analysis['stale_features'])}
- Orphaned: {len(health_analysis['orphaned_features'])}

Recommendations:
{chr(10).join(f"  - {rec}" for rec in health_analysis['recommendations'])}

Files Generated:
- Feature Registry: feature_registry.json
- Documentation: docs/COMPLETE_FEATURE_REGISTRY.md
- Health Analysis: feature_health_analysis.json
""")
    
    return registry


if __name__ == "__main__":
    asyncio.run(main())
