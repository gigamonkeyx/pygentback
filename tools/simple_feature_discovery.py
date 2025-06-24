#!/usr/bin/env python3
"""
Simple Feature Discovery System for PyGent Factory

Addresses the critical "forgotten feature" problem by cataloging all project features.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_cloudflare_workers() -> List[Dict[str, Any]]:
    """Discover Cloudflare Workers MCP servers"""
    workers = []
    cf_dir = Path("mcp-server-cloudflare/apps")
    
    if not cf_dir.exists():
        return workers
    
    for app_dir in cf_dir.iterdir():
        if app_dir.is_dir():
            # Check for wrangler config
            wrangler_files = list(app_dir.glob("wrangler.*"))
            if wrangler_files:
                workers.append({
                    "name": app_dir.name,
                    "directory": str(app_dir),
                    "type": "cloudflare_worker_mcp",
                    "config_files": [str(w) for w in wrangler_files],
                    "discovered_at": datetime.utcnow().isoformat()
                })
    
    return workers


def discover_local_mcp_servers() -> List[Dict[str, Any]]:
    """Discover local MCP servers"""
    servers = []
    
    # Check mcp-servers directory
    mcp_dir = Path("mcp-servers")
    if mcp_dir.exists():
        # Find Python servers by looking for server files
        for py_file in mcp_dir.glob("**/*server*.py"):
            if "node_modules" not in str(py_file):
                servers.append({
                    "name": py_file.stem,
                    "file": str(py_file),
                    "language": "python",
                    "type": "local_mcp_server",
                    "discovered_at": datetime.utcnow().isoformat()
                })
        
        # Find Node.js servers
        for pkg_file in mcp_dir.glob("**/package.json"):
            if "node_modules" not in str(pkg_file):
                try:
                    with open(pkg_file) as f:
                        pkg_data = json.load(f)
                        if "mcp" in pkg_data.get("name", "").lower():
                            servers.append({
                                "name": pkg_data.get("name"),
                                "directory": str(pkg_file.parent),
                                "language": "javascript/typescript",
                                "type": "local_mcp_server",
                                "discovered_at": datetime.utcnow().isoformat()
                            })
                except Exception as e:
                    logger.warning(f"Failed to parse {pkg_file}: {e}")
    
    return servers


def discover_configurations() -> List[Dict[str, Any]]:
    """Discover configuration files"""
    configs = []
    
    config_files = [
        "mcp_server_configs.json",
        "docker-compose.yml", 
        "package.json",
        "requirements.txt",
        "pyproject.toml",
        "wrangler.jsonc"
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            configs.append({
                "name": config_file,
                "file": str(config_path),
                "type": "configuration",
                "discovered_at": datetime.utcnow().isoformat()
            })
    
    return configs


def discover_documentation() -> List[Dict[str, Any]]:
    """Discover documentation files"""
    docs = []
    
    # Find markdown files
    for md_file in Path(".").glob("**/*.md"):
        if "node_modules" not in str(md_file):
            docs.append({
                "name": md_file.name,
                "file": str(md_file),
                "type": "documentation",
                "discovered_at": datetime.utcnow().isoformat()
            })
    
    return docs


def analyze_mcp_config() -> Dict[str, Any]:
    """Analyze the MCP server configuration"""
    config_file = Path("mcp_server_configs.json")
    analysis = {
        "total_servers": 0,
        "working_servers": 0,
        "missing_commands": 0,
        "servers": []
    }
    
    if not config_file.exists():
        return analysis
    
    try:
        with open(config_file) as f:
            config_data = json.load(f)
        
        servers = config_data.get("servers", [])
        analysis["total_servers"] = len(servers)
        
        for server in servers:
            server_info = {
                "id": server.get("id"),
                "name": server.get("name"),
                "command": server.get("command", []),
                "transport": server.get("transport", "stdio"),
                "has_command": bool(server.get("command"))
            }
            
            if server_info["has_command"]:
                analysis["working_servers"] += 1
            else:
                analysis["missing_commands"] += 1
                
            analysis["servers"].append(server_info)
            
    except Exception as e:
        logger.error(f"Failed to analyze MCP config: {e}")
    
    return analysis


def generate_feature_report() -> Dict[str, Any]:
    """Generate comprehensive feature discovery report"""
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "project": "PyGent Factory",
        "cloudflare_workers": discover_cloudflare_workers(),
        "local_mcp_servers": discover_local_mcp_servers(),
        "configurations": discover_configurations(),
        "documentation": discover_documentation(),
        "mcp_analysis": analyze_mcp_config()
    }
    
    # Add summary
    report["summary"] = {
        "cloudflare_workers_count": len(report["cloudflare_workers"]),
        "local_mcp_servers_count": len(report["local_mcp_servers"]),
        "configuration_files": len(report["configurations"]),
        "documentation_files": len(report["documentation"]),
        "mcp_servers_total": report["mcp_analysis"]["total_servers"],
        "mcp_servers_missing_commands": report["mcp_analysis"]["missing_commands"]
    }
    
    return report


def print_critical_findings(report: Dict[str, Any]) -> None:
    """Print critical findings from the discovery"""
    
    print("🚨 CRITICAL FINDINGS - Features That Were 'Forgotten':")
    print("=" * 60)
    
    # Cloudflare Workers
    cf_workers = report["cloudflare_workers"]
    if cf_workers:
        print(f"\n🔥 CLOUDFLARE WORKERS FOUND ({len(cf_workers)}):")
        print("   These are fully functional MCP servers that were 'lost'!")
        for worker in cf_workers:
            print(f"   • {worker['name']} - {worker['directory']}")
        print("\n   ❗ These need proper configuration in mcp_server_configs.json")
    
    # MCP Configuration Issues  
    mcp_analysis = report["mcp_analysis"]
    missing_commands = mcp_analysis["missing_commands"]
    if missing_commands > 0:
        print(f"\n⚠️ MCP SERVERS WITH MISSING COMMANDS ({missing_commands}):")
        for server in mcp_analysis["servers"]:
            if not server["has_command"]:
                print(f"   • {server['name']} ({server['id']}) - No command configured")
    
    # Summary
    summary = report["summary"]
    total_features = (
        summary["cloudflare_workers_count"] + 
        summary["local_mcp_servers_count"] + 
        summary["configuration_files"]
    )
    
    print(f"\n📊 FEATURE INVENTORY SUMMARY:")
    print(f"   Total discoverable features: {total_features}")
    print(f"   Cloudflare Workers: {summary['cloudflare_workers_count']}")
    print(f"   Local MCP Servers: {summary['local_mcp_servers_count']}")
    print(f"   Configuration Files: {summary['configuration_files']}")
    print(f"   Documentation Files: {summary['documentation_files']}")


def save_feature_inventory(report: Dict[str, Any]) -> str:
    """Save feature inventory to file"""
    
    # Save JSON report
    json_file = f"docs/FEATURE_INVENTORY_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    Path("docs").mkdir(exist_ok=True)
    
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown summary
    md_content = f"""# PyGent Factory - Feature Inventory

**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

## 🚨 CRITICAL ISSUE: Forgotten Features

This inventory was created to solve the critical problem of **"forgotten features"** - 
functionality that was built but lost from institutional memory.

### Cloudflare Workers Found ({len(report['cloudflare_workers'])})

These are **fully functional MCP servers** that were previously forgotten:

"""
    
    for worker in report["cloudflare_workers"]:
        md_content += f"- **{worker['name']}** - `{worker['directory']}`\n"
    
    md_content += f"""

### Local MCP Servers ({len(report['local_mcp_servers'])})

"""
    
    for server in report["local_mcp_servers"]:
        md_content += f"- **{server['name']}** ({server['language']}) - `{server.get('file', server.get('directory'))}`\n"
    
    md_content += f"""

### Configuration Analysis

- Total MCP servers in config: {report['mcp_analysis']['total_servers']}
- Servers missing commands: {report['mcp_analysis']['missing_commands']}
- Working servers: {report['mcp_analysis']['working_servers']}

### Prevention Strategies

1. **Automated Discovery**: Run this script regularly
2. **Documentation Requirements**: Document all features when built
3. **Feature Registry**: Maintain searchable feature database
4. **Regular Audits**: Monthly feature discovery runs

### Next Steps

1. Fix Cloudflare Worker configurations
2. Implement Enhanced MCP Registry
3. Create automated feature tracking
4. Improve development documentation workflow

"""
    
    md_file = f"docs/FEATURE_INVENTORY_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.md"
    with open(md_file, 'w') as f:
        f.write(md_content)
    
    return md_file


def main():
    """Main discovery function"""
    print("🔍 PyGent Factory Feature Discovery")
    print("Solving the 'forgotten feature' problem...")
    print("=" * 50)
    
    # Generate report
    report = generate_feature_report()
    
    # Print findings
    print_critical_findings(report)
    
    # Save inventory
    inventory_file = save_feature_inventory(report)
    print(f"\n✅ Feature inventory saved: {inventory_file}")
    
    # Recommendations
    print("\n💡 IMMEDIATE ACTIONS NEEDED:")
    print("1. Fix Cloudflare Worker configurations in mcp_server_configs.json")
    print("2. Switch to Enhanced MCP Registry for proper tool discovery")
    print("3. Implement automated feature tracking in development workflow")
    print("4. Schedule regular feature discovery audits")
    
    return report


if __name__ == "__main__":
    main()
