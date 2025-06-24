#!/usr/bin/env python3
"""
MCP Server Language Detection Analysis

This script analyzes the mcp_server_configs.json to detect and categorize
MCP servers by their implementation language and runtime.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_server_language(command: List[str], server_name: str) -> Dict[str, Any]:
    """
    Detect the language/runtime of an MCP server based on its command.
    
    Args:
        command: The command array used to start the server
        server_name: Name of the server for context
        
    Returns:
        Dict with language detection results
    """
    if not command:
        return {
            "language": "unknown",
            "runtime": "unknown",
            "executable": None,
            "confidence": "low"
        }
    
    executable = command[0].lower()
    
    # Python detection
    if "python" in executable or executable.endswith("python.exe"):
        # Check if it's a module invocation
        if len(command) > 1 and command[1] == "-m":
            module_name = command[2] if len(command) > 2 else "unknown"
            return {
                "language": "python",
                "runtime": "python_module",
                "executable": command[0],
                "module": module_name,
                "confidence": "high",
                "pattern": "python -m module_name"
            }
        # Direct script execution
        elif len(command) > 1 and command[1].endswith(".py"):
            return {
                "language": "python",
                "runtime": "python_script",
                "executable": command[0],
                "script": command[1],
                "confidence": "high",
                "pattern": "python script.py"
            }
        else:
            return {
                "language": "python",
                "runtime": "python_unknown",
                "executable": command[0],
                "confidence": "medium"
            }
    
    # Node.js detection
    elif executable == "node" or executable.endswith("node.exe"):
        if len(command) > 1:
            script_path = command[1]
            if script_path.endswith(".js"):
                # Check if it's TypeScript compiled
                if "dist" in script_path or "build" in script_path:
                    return {
                        "language": "typescript",
                        "runtime": "node_compiled_ts",
                        "executable": command[0],
                        "script": script_path,
                        "confidence": "high",
                        "pattern": "node dist/index.js (compiled TypeScript)"
                    }
                else:
                    return {
                        "language": "javascript",
                        "runtime": "node_js",
                        "executable": command[0],
                        "script": script_path,
                        "confidence": "high",
                        "pattern": "node script.js"
                    }
            elif script_path.endswith(".ts"):
                return {
                    "language": "typescript",
                    "runtime": "node_ts_direct",
                    "executable": command[0],
                    "script": script_path,
                    "confidence": "high",
                    "pattern": "node script.ts (with ts-node)"
                }
        return {
            "language": "javascript",
            "runtime": "node_unknown",
            "executable": command[0],
            "confidence": "medium"
        }
    
    # NPX detection (Node.js package runner)
    elif "npx" in executable:
        package = command[1] if len(command) > 1 else "unknown"
        return {
            "language": "javascript",
            "runtime": "npx_package",
            "executable": command[0],
            "package": package,
            "confidence": "high",
            "pattern": "npx package-name"
        }
    
    # Other executables
    else:
        return {
            "language": "unknown",
            "runtime": "native_executable",
            "executable": command[0],
            "confidence": "low",
            "pattern": f"direct executable: {executable}"
        }


def analyze_mcp_servers() -> Dict[str, Any]:
    """
    Analyze all MCP servers in the configuration file.
    
    Returns:
        Dict containing analysis results
    """
    config_file = Path("mcp_server_configs.json")
    if not config_file.exists():
        logger.error("mcp_server_configs.json not found")
        return {"error": "Configuration file not found"}
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        servers = config_data.get("servers", [])
        analysis = {
            "total_servers": len(servers),
            "language_breakdown": {},
            "runtime_breakdown": {},
            "servers": [],
            "summary": {}
        }
        
        for server in servers:
            server_id = server.get("id", "unknown")
            server_name = server.get("name", "unknown")
            command = server.get("command", [])
            
            detection = detect_server_language(command, server_name)
            
            server_analysis = {
                "id": server_id,
                "name": server_name,
                "command": command,
                **detection
            }
            
            analysis["servers"].append(server_analysis)
            
            # Update breakdowns
            language = detection["language"]
            runtime = detection["runtime"]
            
            analysis["language_breakdown"][language] = analysis["language_breakdown"].get(language, 0) + 1
            analysis["runtime_breakdown"][runtime] = analysis["runtime_breakdown"].get(runtime, 0) + 1
        
        # Create summary
        analysis["summary"] = {
            "languages_detected": list(analysis["language_breakdown"].keys()),
            "most_common_language": max(analysis["language_breakdown"], key=analysis["language_breakdown"].get) if analysis["language_breakdown"] else "none",
            "runtimes_detected": list(analysis["runtime_breakdown"].keys()),
            "python_servers": [s for s in analysis["servers"] if s["language"] == "python"],
            "javascript_servers": [s for s in analysis["servers"] if s["language"] == "javascript"],
            "typescript_servers": [s for s in analysis["servers"] if s["language"] == "typescript"],
            "unknown_servers": [s for s in analysis["servers"] if s["language"] == "unknown"]
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze MCP servers: {e}")
        return {"error": str(e)}


def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """Print a formatted analysis report"""
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    print("🔍 MCP Server Language Analysis Report")
    print("=" * 50)
    
    print(f"\n📊 Summary:")
    print(f"   Total servers: {analysis['total_servers']}")
    print(f"   Languages detected: {', '.join(analysis['summary']['languages_detected'])}")
    print(f"   Most common language: {analysis['summary']['most_common_language']}")
    
    print(f"\n📈 Language Breakdown:")
    for language, count in analysis['language_breakdown'].items():
        percentage = (count / analysis['total_servers']) * 100
        print(f"   {language}: {count} servers ({percentage:.1f}%)")
    
    print(f"\n🔧 Runtime Breakdown:")
    for runtime, count in analysis['runtime_breakdown'].items():
        percentage = (count / analysis['total_servers']) * 100
        print(f"   {runtime}: {count} servers ({percentage:.1f}%)")
    
    print(f"\n🐍 Python Servers ({len(analysis['summary']['python_servers'])}):")
    for server in analysis['summary']['python_servers']:
        print(f"   • {server['name']} ({server['id']})")
        print(f"     Runtime: {server['runtime']}")
        print(f"     Pattern: {server.get('pattern', 'N/A')}")
        if 'module' in server:
            print(f"     Module: {server['module']}")
        elif 'script' in server:
            print(f"     Script: {server['script']}")
    
    print(f"\n📜 JavaScript/TypeScript Servers ({len(analysis['summary']['javascript_servers']) + len(analysis['summary']['typescript_servers'])}):")
    for server in analysis['summary']['javascript_servers'] + analysis['summary']['typescript_servers']:
        print(f"   • {server['name']} ({server['id']})")
        print(f"     Language: {server['language']}")
        print(f"     Runtime: {server['runtime']}")
        print(f"     Pattern: {server.get('pattern', 'N/A')}")
        if 'script' in server:
            print(f"     Script: {server['script']}")
        elif 'package' in server:
            print(f"     Package: {server['package']}")
    
    if analysis['summary']['unknown_servers']:
        print(f"\n❓ Unknown/Other Servers ({len(analysis['summary']['unknown_servers'])}):")
        for server in analysis['summary']['unknown_servers']:
            print(f"   • {server['name']} ({server['id']})")
            print(f"     Command: {' '.join(server['command'])}")
    
    print(f"\n💡 Insights:")
    python_count = len(analysis['summary']['python_servers'])
    js_ts_count = len(analysis['summary']['javascript_servers']) + len(analysis['summary']['typescript_servers'])
    
    if python_count > js_ts_count:
        print(f"   • Python is the dominant language ({python_count} servers)")
    elif js_ts_count > python_count:
        print(f"   • JavaScript/TypeScript is the dominant language ({js_ts_count} servers)")
    else:
        print(f"   • Even split between Python and JavaScript/TypeScript")
    
    has_modules = any('module' in s for s in analysis['summary']['python_servers'])
    has_scripts = any('script' in s for s in analysis['summary']['python_servers'])
    
    if has_modules and has_scripts:
        print(f"   • Python servers use both module (-m) and script patterns")
    elif has_modules:
        print(f"   • Python servers primarily use module (-m) pattern")
    elif has_scripts:
        print(f"   • Python servers primarily use direct script execution")
    
    compiled_ts = [s for s in analysis['summary']['typescript_servers'] if 'compiled' in s.get('pattern', '')]
    if compiled_ts:
        print(f"   • TypeScript servers are pre-compiled to JavaScript ({len(compiled_ts)} servers)")


if __name__ == "__main__":
    print("Analyzing MCP server languages and runtimes...")
    analysis = analyze_mcp_servers()
    print_analysis_report(analysis)
