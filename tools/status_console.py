#!/usr/bin/env python3
"""
PyGent Factory Status Console

A lightweight headless status monitor that connects to a running PyGent Factory backend
and displays real-time system health, agent statistics, and MCP server status.

Usage:
    python status_console.py [--backend-url http://localhost:8000] [--refresh-interval 5]
"""

import asyncio
import aiohttp
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
import sys
import os

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    CLEAR_SCREEN = '\033[2J\033[H'

def print_header():
    """Print the console header"""
    print(f"{Colors.CLEAR_SCREEN}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}             PYGENT FACTORY - STATUS CONSOLE{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.WHITE}Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print()

def format_status(status: bool) -> str:
    """Format status with colors"""
    if status:
        return f"{Colors.GREEN}●{Colors.RESET} ONLINE"
    else:
        return f"{Colors.RED}●{Colors.RESET} OFFLINE"

def format_number(num: int, label: str) -> str:
    """Format numbers with colors"""
    if num > 0:
        return f"{Colors.GREEN}{num}{Colors.RESET} {label}"
    else:
        return f"{Colors.YELLOW}{num}{Colors.RESET} {label}"

class StatusConsole:
    def __init__(self, backend_url: str = "http://localhost:8000", refresh_interval: int = 5):
        self.backend_url = backend_url.rstrip('/')
        self.refresh_interval = refresh_interval
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        """Start the status console"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        
        print(f"{Colors.BOLD}Starting PyGent Factory Status Console...{Colors.RESET}")
        print(f"Backend URL: {Colors.CYAN}{self.backend_url}{Colors.RESET}")
        print(f"Refresh Interval: {Colors.CYAN}{self.refresh_interval}s{Colors.RESET}")
        print(f"Press {Colors.BOLD}Ctrl+C{Colors.RESET} to exit\n")
        
        try:
            while True:
                await self.update_display()
                await asyncio.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Shutting down status console...{Colors.RESET}")
        finally:
            if self.session:
                await self.session.close()
    
    async def update_display(self):
        """Update the status display"""
        try:
            # Fetch all status data
            backend_status = await self.check_backend_health()
            agent_stats = await self.get_agent_stats()
            mcp_status = await self.get_mcp_status()
            system_info = await self.get_system_info()
            
            # Display the status
            print_header()
            self.display_backend_status(backend_status)
            self.display_agent_stats(agent_stats)
            self.display_mcp_status(mcp_status)
            self.display_system_info(system_info)
            
        except Exception as e:
            print_header()
            print(f"{Colors.RED}ERROR: Failed to fetch status - {str(e)}{Colors.RESET}")
            print(f"{Colors.YELLOW}Make sure the PyGent Factory backend is running at {self.backend_url}{Colors.RESET}")
    
    async def check_backend_health(self) -> Dict[str, Any]:
        """Check backend health"""
        try:
            async with self.session.get(f"{self.backend_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return {"status": True, "data": data}
                else:
                    return {"status": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": False, "error": str(e)}
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        try:
            async with self.session.get(f"{self.backend_url}/api/v1/agents") as response:
                if response.status == 200:
                    agents = await response.json()
                    
                    # Process agent stats
                    total = len(agents)
                    active = sum(1 for agent in agents if agent.get('status') == 'active')
                    by_type = {}
                    
                    for agent in agents:
                        agent_type = agent.get('type', 'unknown')
                        by_type[agent_type] = by_type.get(agent_type, 0) + 1
                    
                    return {
                        "status": True,
                        "total": total,
                        "active": active,
                        "inactive": total - active,
                        "by_type": by_type
                    }
                else:
                    return {"status": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": False, "error": str(e)}
    
    async def get_mcp_status(self) -> Dict[str, Any]:
        """Get MCP server status"""
        try:
            async with self.session.get(f"{self.backend_url}/api/v1/mcp/servers") as response:
                if response.status == 200:
                    servers = await response.json()
                    
                    total = len(servers)
                    online = sum(1 for server in servers if server.get('status') == 'running')
                    
                    return {
                        "status": True,
                        "total": total,
                        "online": online,
                        "offline": total - online,
                        "servers": servers[:5]  # Show first 5 servers
                    }
                else:
                    return {"status": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": False, "error": str(e)}
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            async with self.session.get(f"{self.backend_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": True,
                        "uptime": data.get("uptime", "unknown"),
                        "version": data.get("version", "unknown"),
                        "environment": data.get("environment", "unknown")
                    }
                else:
                    return {"status": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": False, "error": str(e)}
    
    def display_backend_status(self, status: Dict[str, Any]):
        """Display backend status"""
        print(f"{Colors.BOLD}🏢 BACKEND STATUS{Colors.RESET}")
        print("-" * 40)
        
        if status["status"]:
            print(f"Backend:     {format_status(True)}")
            if "data" in status:
                data = status["data"]
                print(f"Database:    {format_status(data.get('database', {}).get('status') == 'healthy')}")
                print(f"Vector Store: {format_status(data.get('vector_store', {}).get('status') == 'healthy')}")
                print(f"Memory:      {format_status(data.get('memory', {}).get('status') == 'healthy')}")
        else:
            print(f"Backend:     {format_status(False)} ({status.get('error', 'Unknown error')})")
        
        print()
    
    def display_agent_stats(self, stats: Dict[str, Any]):
        """Display agent statistics"""
        print(f"{Colors.BOLD}🤖 AGENT STATISTICS{Colors.RESET}")
        print("-" * 40)
        
        if stats["status"]:
            print(f"Total Agents:    {format_number(stats['total'], 'agents')}")
            print(f"Active Agents:   {format_number(stats['active'], 'active')}")
            print(f"Inactive Agents: {format_number(stats['inactive'], 'inactive')}")
            
            if stats["by_type"]:
                print(f"\nBy Type:")
                for agent_type, count in stats["by_type"].items():
                    print(f"  {agent_type.capitalize()}: {Colors.CYAN}{count}{Colors.RESET}")
        else:
            print(f"{Colors.RED}Failed to fetch agent stats: {stats.get('error', 'Unknown error')}{Colors.RESET}")
        
        print()
    
    def display_mcp_status(self, status: Dict[str, Any]):
        """Display MCP server status"""
        print(f"{Colors.BOLD}🔌 MCP SERVERS{Colors.RESET}")
        print("-" * 40)
        
        if status["status"]:
            print(f"Total Servers:  {format_number(status['total'], 'servers')}")
            print(f"Online Servers: {format_number(status['online'], 'online')}")
            print(f"Offline Servers: {format_number(status['offline'], 'offline')}")
            
            if status.get("servers"):
                print(f"\nServer Status:")
                for server in status["servers"]:
                    name = server.get("name", "Unknown")[:25]
                    server_status = server.get("status", "unknown")
                    status_icon = "●" if server_status == "running" else "○"
                    color = Colors.GREEN if server_status == "running" else Colors.RED
                    print(f"  {color}{status_icon}{Colors.RESET} {name}")
        else:
            print(f"{Colors.RED}Failed to fetch MCP status: {status.get('error', 'Unknown error')}{Colors.RESET}")
        
        print()
    
    def display_system_info(self, info: Dict[str, Any]):
        """Display system information"""
        print(f"{Colors.BOLD}📊 SYSTEM INFO{Colors.RESET}")
        print("-" * 40)
        
        if info["status"]:
            print(f"Uptime:      {Colors.CYAN}{info.get('uptime', 'Unknown')}{Colors.RESET}")
            print(f"Version:     {Colors.CYAN}{info.get('version', 'Unknown')}{Colors.RESET}")
            print(f"Environment: {Colors.CYAN}{info.get('environment', 'Unknown')}{Colors.RESET}")
        else:
            print(f"{Colors.RED}Failed to fetch system info: {info.get('error', 'Unknown error')}{Colors.RESET}")
        
        print()
        print(f"{Colors.MAGENTA}Press Ctrl+C to exit{Colors.RESET}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PyGent Factory Status Console")
    parser.add_argument("--backend-url", default="http://localhost:8000", 
                       help="Backend URL (default: http://localhost:8000)")
    parser.add_argument("--refresh-interval", type=int, default=5,
                       help="Refresh interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    console = StatusConsole(args.backend_url, args.refresh_interval)
    await console.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
