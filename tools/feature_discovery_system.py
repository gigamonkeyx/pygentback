#!/usr/bin/env python3
"""
Feature Discovery and Documentation Automation System

This script addresses the critical issue of features being forgotten and needing rediscovery.
It implements automated feature detection, documentation generation, and knowledge preservation.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureDiscoverySystem:
    """
    Automated system to discover, document, and track all features in the codebase
    to prevent the "forgotten feature" problem.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.features_db = {}
        self.discovered_features = []
        self.config_files = []
        self.documentation_gaps = []
    
    def discover_all_features(self) -> Dict[str, Any]:
        """
        Comprehensive feature discovery across the entire project.
        
        Returns:
            Dict containing all discovered features and their metadata
        """
        logger.info("🔍 Starting comprehensive feature discovery...")
        
        discovery_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "project_root": str(self.project_root),
            "categories": {
                "mcp_servers": self._discover_mcp_servers(),
                "api_endpoints": self._discover_api_endpoints(), 
                "ui_components": self._discover_ui_components(),
                "services": self._discover_services(),
                "databases": self._discover_databases(),
                "integrations": self._discover_integrations(),
                "configuration": self._discover_configurations(),
                "deployment": self._discover_deployment_features(),
                "scripts": self._discover_automation_scripts(),
                "documentation": self._analyze_documentation_coverage()
            },
            "summary": {},
            "recommendations": []
        }
        
        # Generate summary
        discovery_result["summary"] = self._generate_discovery_summary(discovery_result)
        
        # Generate recommendations
        discovery_result["recommendations"] = self._generate_recommendations(discovery_result)
        
        return discovery_result
    
    def _discover_mcp_servers(self) -> Dict[str, Any]:
        """Discover all MCP servers in the project"""
        mcp_servers = {
            "local_servers": [],
            "cloudflare_workers": [],
            "configurations": [],
            "total_count": 0
        }
        
        # Local MCP servers
        mcp_dirs = [
            self.project_root / "mcp-servers",
            self.project_root / "src" / "mcp"
        ]
        
        for mcp_dir in mcp_dirs:
            if mcp_dir.exists():
                mcp_servers["local_servers"].extend(self._scan_local_mcp_servers(mcp_dir))
        
        # Cloudflare Workers MCP servers
        cf_dir = self.project_root / "mcp-server-cloudflare"
        if cf_dir.exists():
            mcp_servers["cloudflare_workers"] = self._scan_cloudflare_mcp_servers(cf_dir)
        
        # MCP configurations
        config_files = [
            self.project_root / "mcp_server_configs.json",
            self.project_root / ".claude_desktop",
            self.project_root / "mcp" / "config.json"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                mcp_servers["configurations"].append({
                    "file": str(config_file),
                    "type": "mcp_config",
                    "discovered_at": datetime.utcnow().isoformat()
                })
        
        mcp_servers["total_count"] = (
            len(mcp_servers["local_servers"]) + 
            len(mcp_servers["cloudflare_workers"])
        )
        
        return mcp_servers    def _scan_local_mcp_servers(self, mcp_dir: Path) -> List[Dict[str, Any]]:
        """Scan for local MCP servers"""
        servers = []
        
        try:
            # Look for Python servers (avoid circular symlinks)
            python_servers = []
            for root, dirs, files in os.walk(mcp_dir):
                # Skip node_modules and other problematic directories
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__']]
                
                for file in files:
                    if file.endswith('.py') and ('server' in file.lower() or 'mcp' in file.lower()):
                        python_servers.append(Path(root) / file)
            
            for server_file in python_servers:
                servers.append({
                    "name": server_file.stem,
                    "file": str(server_file),
                    "language": "python",
                    "type": "local_mcp_server",
                    "discovered_at": datetime.utcnow().isoformat()
                })
        except Exception as e:
            logger.warning(f"Error scanning Python servers in {mcp_dir}: {e}")
        
        try:
            # Look for Node.js servers (avoid circular symlinks)
            for root, dirs, files in os.walk(mcp_dir):
                # Skip problematic directories
                dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__']]
                
                if 'package.json' in files:
                    pkg_file = Path(root) / 'package.json'
                    try:
                        with open(pkg_file) as f:
                            pkg_data = json.load(f)
                            if "mcp" in pkg_data.get("name", "").lower():
                                servers.append({
                                    "name": pkg_data.get("name", pkg_file.parent.name),
                                    "file": str(pkg_file.parent),
                                    "language": "javascript",
                                    "type": "local_mcp_server",
                                    "package_info": pkg_data,
                                    "discovered_at": datetime.utcnow().isoformat()
                                })
                    except Exception as e:
                        logger.warning(f"Failed to parse {pkg_file}: {e}")
        except Exception as e:
            logger.warning(f"Error scanning Node.js servers in {mcp_dir}: {e}")
        
        return servers
    
    def _scan_cloudflare_mcp_servers(self, cf_dir: Path) -> List[Dict[str, Any]]:
        """Scan for Cloudflare Workers MCP servers"""
        servers = []
        apps_dir = cf_dir / "apps"
        
        if apps_dir.exists():
            for app_dir in apps_dir.iterdir():
                if app_dir.is_dir():
                    wrangler_configs = list(app_dir.glob("wrangler.*"))
                    if wrangler_configs:
                        servers.append({
                            "name": app_dir.name,
                            "directory": str(app_dir),
                            "type": "cloudflare_worker",
                            "language": "typescript",
                            "config_files": [str(c) for c in wrangler_configs],
                            "discovered_at": datetime.utcnow().isoformat()
                        })
        
        return servers
    
    def _discover_api_endpoints(self) -> Dict[str, Any]:
        """Discover all API endpoints"""
        endpoints = {
            "fastapi_routes": [],
            "total_count": 0
        }
        
        # Look for FastAPI route files
        api_files = list(self.project_root.rglob("*.py"))
        
        for api_file in api_files:
            if "api" in str(api_file) or "route" in str(api_file):
                try:
                    with open(api_file) as f:
                        content = f.read()
                        
                    # Look for FastAPI decorators
                    route_patterns = [
                        r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                        r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
                    ]
                    
                    for pattern in route_patterns:
                        matches = re.findall(pattern, content)
                        for method, path in matches:
                            endpoints["fastapi_routes"].append({
                                "method": method.upper(),
                                "path": path,
                                "file": str(api_file),
                                "discovered_at": datetime.utcnow().isoformat()
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to parse API file {api_file}: {e}")
        
        endpoints["total_count"] = len(endpoints["fastapi_routes"])
        return endpoints
    
    def _discover_ui_components(self) -> Dict[str, Any]:
        """Discover UI components and pages"""
        components = {
            "react_components": [],
            "pages": [],
            "total_count": 0
        }
        
        # Look for React components
        ui_files = list(self.project_root.rglob("*.tsx")) + list(self.project_root.rglob("*.jsx"))
        
        for ui_file in ui_files:
            try:
                with open(ui_file) as f:
                    content = f.read()
                
                # Check if it's a React component
                if "export" in content and ("function" in content or "const" in content):
                    component_type = "page" if "Page" in ui_file.stem else "component"
                    
                    components["react_components"].append({
                        "name": ui_file.stem,
                        "file": str(ui_file),
                        "type": component_type,
                        "discovered_at": datetime.utcnow().isoformat()
                    })
                    
                    if component_type == "page":
                        components["pages"].append(ui_file.stem)
                        
            except Exception as e:
                logger.warning(f"Failed to parse UI file {ui_file}: {e}")
        
        components["total_count"] = len(components["react_components"])
        return components
    
    def _discover_services(self) -> Dict[str, Any]:
        """Discover backend services"""
        services = {
            "python_services": [],
            "total_count": 0
        }
        
        # Look for service files
        service_dirs = [
            self.project_root / "src" / "services",
            self.project_root / "src" / "api",
            self.project_root / "services"
        ]
        
        for service_dir in service_dirs:
            if service_dir.exists():
                service_files = list(service_dir.rglob("*.py"))
                for service_file in service_files:
                    services["python_services"].append({
                        "name": service_file.stem,
                        "file": str(service_file),
                        "discovered_at": datetime.utcnow().isoformat()
                    })
        
        services["total_count"] = len(services["python_services"])
        return services
    
    def _discover_databases(self) -> Dict[str, Any]:
        """Discover database configurations and schemas"""
        databases = {
            "configurations": [],
            "models": [],
            "migrations": [],
            "total_count": 0
        }
        
        # Look for database configs
        db_files = [
            "docker-compose.yml",
            "database.py",
            "models.py",
            "schema.sql"
        ]
        
        for db_file in db_files:
            file_paths = list(self.project_root.rglob(db_file))
            for file_path in file_paths:
                databases["configurations"].append({
                    "name": db_file,
                    "file": str(file_path),
                    "discovered_at": datetime.utcnow().isoformat()
                })
        
        databases["total_count"] = len(databases["configurations"])
        return databases
    
    def _discover_integrations(self) -> Dict[str, Any]:
        """Discover external integrations"""
        integrations = {
            "cloudflare": [],
            "ai_services": [],
            "external_apis": [],
            "total_count": 0
        }
        
        # Look for integration files
        integration_patterns = [
            ("cloudflare", ["cloudflare", "wrangler", "workers"]),
            ("ai_services", ["ollama", "openai", "anthropic", "transformers"]),
            ("external_apis", ["api", "client", "webhook"])
        ]
        
        for category, keywords in integration_patterns:
            for keyword in keywords:
                files = list(self.project_root.rglob(f"*{keyword}*"))
                for file_path in files:
                    if file_path.is_file():
                        integrations[category].append({
                            "name": file_path.name,
                            "file": str(file_path),
                            "keyword": keyword,
                            "discovered_at": datetime.utcnow().isoformat()
                        })
        
        total = sum(len(integrations[cat]) for cat in ["cloudflare", "ai_services", "external_apis"])
        integrations["total_count"] = total
        return integrations
    
    def _discover_configurations(self) -> Dict[str, Any]:
        """Discover configuration files"""
        configs = {
            "config_files": [],
            "total_count": 0
        }
        
        config_patterns = [
            "*.json", "*.yaml", "*.yml", "*.toml", "*.ini", 
            "*.env", "*.config", "Dockerfile", "docker-compose*"
        ]
        
        for pattern in config_patterns:
            files = list(self.project_root.rglob(pattern))
            for file_path in files:
                if file_path.is_file() and "node_modules" not in str(file_path):
                    configs["config_files"].append({
                        "name": file_path.name,
                        "file": str(file_path),
                        "type": file_path.suffix or "no_extension",
                        "discovered_at": datetime.utcnow().isoformat()
                    })
        
        configs["total_count"] = len(configs["config_files"])
        return configs
    
    def _discover_deployment_features(self) -> Dict[str, Any]:
        """Discover deployment and infrastructure features"""
        deployment = {
            "docker": [],
            "scripts": [],
            "ci_cd": [],
            "total_count": 0
        }
        
        # Docker files
        docker_files = list(self.project_root.rglob("Dockerfile*")) + list(self.project_root.rglob("docker-compose*"))
        for docker_file in docker_files:
            deployment["docker"].append({
                "name": docker_file.name,
                "file": str(docker_file),
                "discovered_at": datetime.utcnow().isoformat()
            })
        
        # Deployment scripts
        script_files = list(self.project_root.rglob("*.sh")) + list(self.project_root.rglob("*.ps1"))
        for script_file in script_files:
            if any(keyword in script_file.name.lower() for keyword in ["deploy", "start", "build", "setup"]):
                deployment["scripts"].append({
                    "name": script_file.name,
                    "file": str(script_file),
                    "discovered_at": datetime.utcnow().isoformat()
                })
        
        # CI/CD files
        ci_dirs = [".github", ".gitlab-ci", "ci", "cd"]
        for ci_dir in ci_dirs:
            ci_path = self.project_root / ci_dir
            if ci_path.exists():
                ci_files = list(ci_path.rglob("*"))
                for ci_file in ci_files:
                    if ci_file.is_file():
                        deployment["ci_cd"].append({
                            "name": ci_file.name,
                            "file": str(ci_file),
                            "discovered_at": datetime.utcnow().isoformat()
                        })
        
        total = len(deployment["docker"]) + len(deployment["scripts"]) + len(deployment["ci_cd"])
        deployment["total_count"] = total
        return deployment
    
    def _discover_automation_scripts(self) -> Dict[str, Any]:
        """Discover automation and utility scripts"""
        scripts = {
            "python_scripts": [],
            "shell_scripts": [],
            "analysis_tools": [],
            "total_count": 0
        }
        
        # Python scripts
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            if any(keyword in py_file.name.lower() for keyword in 
                  ["test", "analyze", "check", "validate", "setup", "install", "migrate"]):
                scripts["python_scripts"].append({
                    "name": py_file.name,
                    "file": str(py_file),
                    "discovered_at": datetime.utcnow().isoformat()
                })
        
        # Analysis tools (like our current analysis scripts)
        analysis_files = list(self.project_root.glob("analyze_*.py"))
        for analysis_file in analysis_files:
            scripts["analysis_tools"].append({
                "name": analysis_file.name,
                "file": str(analysis_file),
                "purpose": "system_analysis",
                "discovered_at": datetime.utcnow().isoformat()
            })
        
        total = len(scripts["python_scripts"]) + len(scripts["shell_scripts"]) + len(scripts["analysis_tools"])
        scripts["total_count"] = total
        return scripts
    
    def _analyze_documentation_coverage(self) -> Dict[str, Any]:
        """Analyze documentation coverage and gaps"""
        docs = {
            "documentation_files": [],
            "readme_files": [],
            "coverage_gaps": [],
            "total_count": 0
        }
        
        # Find documentation files
        doc_patterns = ["*.md", "*.rst", "*.txt"]
        for pattern in doc_patterns:
            doc_files = list(self.project_root.rglob(pattern))
            for doc_file in doc_files:
                if "node_modules" not in str(doc_file):
                    file_type = "readme" if "readme" in doc_file.name.lower() else "documentation"
                    docs["documentation_files"].append({
                        "name": doc_file.name,
                        "file": str(doc_file),
                        "type": file_type,
                        "discovered_at": datetime.utcnow().isoformat()
                    })
                    
                    if file_type == "readme":
                        docs["readme_files"].append(str(doc_file))
        
        docs["total_count"] = len(docs["documentation_files"])
        return docs
    
    def _generate_discovery_summary(self, discovery_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of discovery results"""
        categories = discovery_result["categories"]
        
        return {
            "total_features_discovered": sum(
                cat.get("total_count", 0) for cat in categories.values()
            ),
            "mcp_servers_total": categories["mcp_servers"]["total_count"],
            "api_endpoints_total": categories["api_endpoints"]["total_count"],
            "ui_components_total": categories["ui_components"]["total_count"],
            "configuration_files": categories["configuration"]["total_count"],
            "cloudflare_workers": len(categories["mcp_servers"]["cloudflare_workers"]),
            "documentation_files": categories["documentation"]["total_count"]
        }
    
    def _generate_recommendations(self, discovery_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations to prevent feature loss"""
        recommendations = []
        
        # Check for undocumented Cloudflare workers
        cf_workers = discovery_result["categories"]["mcp_servers"]["cloudflare_workers"]
        if cf_workers:
            recommendations.append({
                "type": "critical",
                "category": "documentation",
                "title": "Document Cloudflare MCP Workers",
                "description": f"Found {len(cf_workers)} Cloudflare Workers that need proper documentation",
                "action": "Create feature inventory documentation",
                "priority": "high"
            })
        
        # Check for missing feature registry
        recommendations.append({
            "type": "infrastructure",
            "category": "knowledge_management", 
            "title": "Implement Automated Feature Registry",
            "description": "Create automated system to track and document all features",
            "action": "Build feature discovery automation",
            "priority": "high"
        })
        
        # Check documentation coverage
        total_features = discovery_result["summary"]["total_features_discovered"]
        doc_files = discovery_result["summary"]["documentation_files"]
        
        if total_features > doc_files * 10:  # Rough heuristic
            recommendations.append({
                "type": "documentation",
                "category": "coverage",
                "title": "Improve Documentation Coverage",
                "description": f"Only {doc_files} docs for {total_features} features detected",
                "action": "Generate comprehensive feature documentation",
                "priority": "medium"
            })
        
        return recommendations
    
    def generate_feature_inventory(self, discovery_result: Dict[str, Any]) -> str:
        """Generate comprehensive feature inventory document"""
        
        inventory_md = f"""# PyGent Factory - Complete Feature Inventory

**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Total Features Discovered**: {discovery_result['summary']['total_features_discovered']}

> **Purpose**: This document serves as the definitive inventory of ALL features in PyGent Factory 
> to prevent the "forgotten feature" problem where built functionality is lost or rediscovered.

---

## 🔍 DISCOVERY SUMMARY

"""
        
        # Add summary table
        summary = discovery_result["summary"]
        inventory_md += f"""
| Category | Count | Details |
|----------|-------|---------|
| MCP Servers | {summary['mcp_servers_total']} | Including {summary['cloudflare_workers']} Cloudflare Workers |
| API Endpoints | {summary['api_endpoints_total']} | FastAPI routes discovered |
| UI Components | {summary['ui_components_total']} | React components and pages |
| Configuration Files | {summary['configuration_files']} | All config files found |
| Documentation Files | {summary['documentation_files']} | Existing documentation |

"""
        
        # Add detailed sections for each category
        categories = discovery_result["categories"]
        
        # MCP Servers section
        inventory_md += "\n## 🤖 MCP SERVERS\n\n"
        
        if categories["mcp_servers"]["cloudflare_workers"]:
            inventory_md += "### Cloudflare Workers (CRITICAL - Previously Forgotten!)\n\n"
            for worker in categories["mcp_servers"]["cloudflare_workers"]:
                inventory_md += f"- **{worker['name']}** (`{worker['directory']}`)\n"
                inventory_md += f"  - Type: {worker['type']}\n"
                inventory_md += f"  - Language: {worker['language']}\n\n"
        
        if categories["mcp_servers"]["local_servers"]:
            inventory_md += "### Local MCP Servers\n\n"
            for server in categories["mcp_servers"]["local_servers"]:
                inventory_md += f"- **{server['name']}** (`{server['file']}`)\n"
                inventory_md += f"  - Language: {server['language']}\n\n"
        
        # API Endpoints section
        if categories["api_endpoints"]["fastapi_routes"]:
            inventory_md += "\n## 🛣️ API ENDPOINTS\n\n"
            
            # Group by method
            methods = {}
            for route in categories["api_endpoints"]["fastapi_routes"]:
                method = route["method"]
                if method not in methods:
                    methods[method] = []
                methods[method].append(route)
            
            for method, routes in methods.items():
                inventory_md += f"### {method} Routes\n\n"
                for route in routes:
                    inventory_md += f"- `{method} {route['path']}` - {route['file']}\n"
                inventory_md += "\n"
        
        # Add recommendations section
        if discovery_result["recommendations"]:
            inventory_md += "\n## ⚠️ CRITICAL RECOMMENDATIONS\n\n"
            for rec in discovery_result["recommendations"]:
                inventory_md += f"### {rec['title']} ({rec['priority'].upper()})\n\n"
                inventory_md += f"**Category**: {rec['category']}\n"
                inventory_md += f"**Description**: {rec['description']}\n"
                inventory_md += f"**Action Required**: {rec['action']}\n\n"
        
        # Add prevention strategies
        inventory_md += """
---

## 🛡️ PREVENTION STRATEGIES

### Immediate Actions
1. **Feature Registry Database**: Implement automated feature tracking in database
2. **Documentation Automation**: Auto-generate docs from code discovery
3. **Regular Audits**: Schedule monthly feature discovery runs
4. **Integration Alerts**: Notify when new features are added without documentation

### Long-term Solutions
1. **Development Workflow**: Require feature documentation before merge
2. **Automated Testing**: Include feature registry tests in CI/CD
3. **Knowledge Base**: Searchable feature database with tags and relationships
4. **Team Training**: Educate on documentation-first development

### Technology Implementation
- **Feature Discovery Script**: This current script should run automatically
- **Database Integration**: Store feature metadata in PostgreSQL
- **Search Integration**: Make features searchable in UI
- **Version Tracking**: Track feature evolution over time

"""
        
        return inventory_md
    
    def save_feature_inventory(self, discovery_result: Dict[str, Any], output_file: str = None) -> str:
        """Save the feature inventory to a file"""
        if output_file is None:
            output_file = f"docs/FEATURE_INVENTORY_{datetime.utcnow().strftime('%Y%m%d')}.md"
        
        inventory_content = self.generate_feature_inventory(discovery_result)
        
        output_path = self.project_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(inventory_content)
        
        logger.info(f"✅ Feature inventory saved to: {output_path}")
        return str(output_path)


def main():
    """Main function to run feature discovery"""
    print("🔍 PyGent Factory Feature Discovery System")
    print("=" * 60)
    print("Solving the 'forgotten feature' problem...")
    
    # Initialize discovery system
    discovery_system = FeatureDiscoverySystem()
    
    # Run comprehensive discovery
    results = discovery_system.discover_all_features()
    
    # Print summary
    print(f"\n📊 DISCOVERY COMPLETE")
    print(f"Total features found: {results['summary']['total_features_discovered']}")
    print(f"MCP servers: {results['summary']['mcp_servers_total']}")
    print(f"API endpoints: {results['summary']['api_endpoints_total']}")
    print(f"UI components: {results['summary']['ui_components_total']}")
    
    # Highlight critical findings
    cf_workers = results['categories']['mcp_servers']['cloudflare_workers']
    if cf_workers:
        print(f"\n🚨 CRITICAL: Found {len(cf_workers)} Cloudflare Workers that were 'forgotten'!")
        for worker in cf_workers:
            print(f"   • {worker['name']} in {worker['directory']}")
    
    # Save comprehensive inventory
    inventory_file = discovery_system.save_feature_inventory(results)
    print(f"\n✅ Feature inventory saved: {inventory_file}")
    
    # Show recommendations
    if results['recommendations']:
        print(f"\n⚠️ RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"   • {rec['title']} ({rec['priority']})")
    
    return results


if __name__ == "__main__":
    main()
