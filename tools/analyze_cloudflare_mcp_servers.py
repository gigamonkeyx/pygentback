#!/usr/bin/env python3
"""
Cloudflare MCP Server Analysis and Configuration Generator

This script analyzes the Cloudflare Workers MCP servers that we built,
determines their capabilities, and generates proper configuration entries
for mcp_server_configs.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_cloudflare_mcp_servers() -> Dict[str, Any]:
    """
    Analyze the Cloudflare MCP servers in the mcp-server-cloudflare directory.
    
    Returns:
        Dict containing analysis results and generated configurations
    """
    cloudflare_dir = Path("mcp-server-cloudflare")
    if not cloudflare_dir.exists():
        return {"error": "mcp-server-cloudflare directory not found"}
    
    apps_dir = cloudflare_dir / "apps"
    if not apps_dir.exists():
        return {"error": "apps directory not found in mcp-server-cloudflare"}
    
    analysis = {
        "total_apps": 0,
        "mcp_servers": [],
        "configurations": [],
        "deployment_info": {},
        "missing_configs": []
    }
    
    # List all apps
    app_dirs = [d for d in apps_dir.iterdir() if d.is_dir()]
    analysis["total_apps"] = len(app_dirs)
    
    # Target servers from config that are missing commands
    target_servers = {
        "cloudflare-browser": "browser-rendering",
        "cloudflare-docs": None,  # Need to find
        "cloudflare-radar": "radar", 
        "cloudflare-bindings": "workers-bindings"
    }
    
    for config_id, expected_dir in target_servers.items():
        if expected_dir:
            app_dir = apps_dir / expected_dir
            if app_dir.exists():
                server_info = analyze_single_server(app_dir, config_id)
                analysis["mcp_servers"].append(server_info)
                
                # Generate configuration
                config = generate_server_config(server_info)
                analysis["configurations"].append(config)
            else:
                analysis["missing_configs"].append({
                    "config_id": config_id,
                    "expected_dir": expected_dir,
                    "status": "directory_not_found"
                })
        else:
            # Try to find by searching
            found_dir = find_docs_server(apps_dir)
            if found_dir:
                server_info = analyze_single_server(found_dir, config_id)
                analysis["mcp_servers"].append(server_info)
                config = generate_server_config(server_info)
                analysis["configurations"].append(config)
            else:
                analysis["missing_configs"].append({
                    "config_id": config_id,
                    "status": "not_found"
                })
    
    return analysis


def analyze_single_server(app_dir: Path, config_id: str) -> Dict[str, Any]:
    """Analyze a single Cloudflare MCP server app"""
    server_info = {
        "config_id": config_id,
        "app_name": app_dir.name,
        "app_dir": str(app_dir),
        "package_json": None,
        "wrangler_config": None,
        "main_file": None,
        "tools": [],
        "capabilities": [],
        "deployment_url": None,
        "server_type": "cloudflare_worker"
    }
    
    # Read package.json
    package_json_path = app_dir / "package.json"
    if package_json_path.exists():
        try:
            with open(package_json_path) as f:
                server_info["package_json"] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read package.json for {app_dir.name}: {e}")
    
    # Read wrangler config
    wrangler_paths = [
        app_dir / "wrangler.jsonc",
        app_dir / "wrangler.json",
        app_dir / "wrangler.toml"
    ]
    
    for wrangler_path in wrangler_paths:
        if wrangler_path.exists():
            try:
                with open(wrangler_path) as f:
                    content = f.read()
                    # Simple JSONC parsing (remove comments)
                    lines = content.split('\n')
                    clean_lines = []
                    for line in lines:
                        if '//' in line:
                            line = line[:line.index('//')]
                        clean_lines.append(line)
                    clean_content = '\n'.join(clean_lines)
                    
                    if wrangler_path.suffix in ['.json', '.jsonc']:
                        server_info["wrangler_config"] = json.loads(clean_content)
                    break
            except Exception as e:
                logger.warning(f"Failed to read wrangler config for {app_dir.name}: {e}")
    
    # Find main file
    src_dir = app_dir / "src"
    if src_dir.exists():
        possible_mains = [
            f"{app_dir.name}.app.ts",
            "index.ts",
            "main.ts",
            "app.ts"
        ]
        
        for main_file in possible_mains:
            main_path = src_dir / main_file
            if main_path.exists():
                server_info["main_file"] = str(main_path)
                break
    
    # Analyze tools (look for .tools.ts files)
    if src_dir.exists():
        tools_dir = src_dir / "tools"
        if tools_dir.exists():
            tool_files = list(tools_dir.glob("*.tools.ts"))
            server_info["tools"] = [f.stem.replace('.tools', '') for f in tool_files]
    
    # Determine capabilities based on app name and tools
    server_info["capabilities"] = determine_capabilities(app_dir.name, server_info["tools"])
    
    # Generate deployment URL
    if server_info["wrangler_config"]:
        name = server_info["wrangler_config"].get("name")
        if name:
            server_info["deployment_url"] = f"https://{name}.your-subdomain.workers.dev"
    
    return server_info


def find_docs_server(apps_dir: Path) -> Optional[Path]:
    """Find the documentation server by searching for docs-related directories"""
    docs_candidates = [
        "docs-autorag",
        "docs-vectorize", 
        "autorag",
        "docs"
    ]
    
    for candidate in docs_candidates:
        candidate_dir = apps_dir / candidate
        if candidate_dir.exists():
            return candidate_dir
    
    return None


def determine_capabilities(app_name: str, tools: List[str]) -> List[str]:
    """Determine server capabilities based on app name and tools"""
    capabilities = []
    
    if "browser" in app_name:
        capabilities.extend([
            "browser-rendering", "html-content", "web-scraping", "screenshot-capture"
        ])
    
    if "radar" in app_name:
        capabilities.extend([
            "security-analytics", "threat-intelligence", "url-scanning", "dns-analytics"
        ])
    
    if "docs" in app_name or "autorag" in app_name:
        capabilities.extend([
            "documentation-search", "rag-queries", "knowledge-base", "vectorize-search"
        ])
    
    if "workers" in app_name or "bindings" in app_name:
        capabilities.extend([
            "workers-deployment", "bindings-management", "serverless-functions", "edge-computing"
        ])
    
    # Add tool-based capabilities
    for tool in tools:
        if "account" in tool:
            capabilities.append("account-management")
        if "browser" in tool:
            capabilities.append("browser-automation")
        if "radar" in tool:
            capabilities.append("security-monitoring")
        if "url" in tool:
            capabilities.append("url-analysis")
    
    return list(set(capabilities))  # Remove duplicates


def generate_server_config(server_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate MCP server configuration for a Cloudflare Worker"""
    
    # For Cloudflare Workers, we need to determine if they're deployed or local
    # Since these are workers, they should be accessed via HTTP/SSE, not stdio
    
    config = {
        "id": server_info["config_id"],
        "name": get_friendly_name(server_info["config_id"]),
        "command": [],  # Will be empty for HTTP/SSE transport
        "capabilities": server_info["capabilities"],
        "transport": "sse",  # Server-Sent Events for Cloudflare Workers
        "config": {
            "category": get_category(server_info["app_name"]),
            "author": "Cloudflare",
            "verified": True,
            "description": get_description(server_info["config_id"]),
            "priority": 3,
            "deployment_type": "cloudflare_worker",
            "tools": server_info["tools"]
        },
        "auto_start": False,  # HTTP/SSE servers don't need process start
        "restart_on_failure": False,
        "max_restarts": 0,
        "timeout": 30,
        "custom_config": {
            "base_url": server_info.get("deployment_url", ""),
            "auth_required": True,
            "transport_type": "sse",
            "worker_name": (server_info.get("wrangler_config") or {}).get("name", ""),
            "app_directory": server_info["app_dir"]
        }
    }
    
    return config


def get_friendly_name(config_id: str) -> str:
    """Get friendly name from config ID"""
    names = {
        "cloudflare-browser": "Cloudflare Browser Rendering",
        "cloudflare-docs": "Cloudflare Documentation",
        "cloudflare-radar": "Cloudflare Radar",
        "cloudflare-bindings": "Cloudflare Workers Bindings"
    }
    return names.get(config_id, config_id.replace("-", " ").title())


def get_category(app_name: str) -> str:
    """Get category based on app name"""
    if "browser" in app_name:
        return "web"
    elif "radar" in app_name:
        return "security"
    elif "docs" in app_name:
        return "documentation"
    elif "workers" in app_name or "bindings" in app_name:
        return "deployment"
    else:
        return "utilities"


def get_description(config_id: str) -> str:
    """Get description based on config ID"""
    descriptions = {
        "cloudflare-browser": "Browser rendering and web scraping capabilities via Cloudflare Workers",
        "cloudflare-docs": "Documentation search and RAG queries via Cloudflare Vectorize",
        "cloudflare-radar": "Security analytics and threat intelligence via Cloudflare Radar",
        "cloudflare-bindings": "Cloudflare Workers deployment and bindings management"
    }
    return descriptions.get(config_id, f"Cloudflare MCP server: {config_id}")


def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Print formatted analysis report"""
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    print("🔍 Cloudflare MCP Server Analysis Report")
    print("=" * 60)
    
    print(f"\n📊 Summary:")
    print(f"   Total apps found: {analysis['total_apps']}")
    print(f"   MCP servers analyzed: {len(analysis['mcp_servers'])}")
    print(f"   Configurations generated: {len(analysis['configurations'])}")
    print(f"   Missing/problematic: {len(analysis['missing_configs'])}")
    
    print(f"\n🔧 MCP Servers Found:")
    for server in analysis["mcp_servers"]:
        print(f"   • {server['config_id']} ({server['app_name']})")
        print(f"     Directory: {server['app_dir']}")
        print(f"     Main file: {server.get('main_file', 'Not found')}")
        print(f"     Tools: {', '.join(server['tools']) if server['tools'] else 'None detected'}")
        print(f"     Capabilities: {', '.join(server['capabilities'])}")
        
        if server.get('deployment_url'):
            print(f"     Deployment URL: {server['deployment_url']}")
        print()
    
    if analysis['missing_configs']:
        print(f"❌ Missing/Problematic Servers:")
        for missing in analysis['missing_configs']:
            print(f"   • {missing['config_id']}: {missing['status']}")
    
    print(f"\n⚙️ Generated Configurations:")
    for config in analysis['configurations']:
        print(f"   • {config['id']}: {config['name']}")
        print(f"     Transport: {config['transport']}")
        print(f"     Category: {config['config']['category']}")
        print(f"     Auto-start: {config['auto_start']}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Deploy Cloudflare Workers if not already deployed")
    print(f"   2. Update mcp_server_configs.json with proper URLs")
    print(f"   3. Configure authentication for worker access")
    print(f"   4. Test SSE connections to deployed workers")


def update_mcp_config_file(analysis: Dict[str, Any]) -> None:
    """Update the mcp_server_configs.json file with proper configurations"""
    config_file = Path("mcp_server_configs.json")
    if not config_file.exists():
        logger.error("mcp_server_configs.json not found")
        return
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Update each missing server configuration
        servers = config_data.get("servers", [])
        
        for generated_config in analysis["configurations"]:
            # Find existing server entry
            for i, server in enumerate(servers):
                if server.get("id") == generated_config["id"]:
                    # Update the existing entry
                    servers[i] = generated_config
                    logger.info(f"Updated configuration for {generated_config['id']}")
                    break
        
        # Save updated configuration
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info("mcp_server_configs.json updated successfully")
        
    except Exception as e:
        logger.error(f"Failed to update mcp_server_configs.json: {e}")


if __name__ == "__main__":
    print("Analyzing Cloudflare MCP servers...")
    analysis = analyze_cloudflare_mcp_servers()
    print_analysis_report(analysis)
    
    if analysis.get("configurations"):
        print(f"\n🔄 Would you like to update mcp_server_configs.json? (y/n)")
        # For automation, we'll just show what would be updated
        print("📝 Configuration updates that would be applied:")
        for config in analysis["configurations"]:
            print(f"   {config['id']}: Set transport to '{config['transport']}', add deployment info")
