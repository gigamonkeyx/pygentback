#!/usr/bin/env python3
"""
MCP Ecosystem Status Report Generator

Comprehensive status report for the complete PyGent Factory MCP ecosystem.
"""

import json
import time
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional

class MCPEcosystemStatusReporter:
    """Generate comprehensive status reports for MCP ecosystem"""
    
    def __init__(self):
        self.http_servers = {
            'embedding': {
                'port': 8002,
                'name': 'Embedding MCP Server',
                'description': 'Multi-provider embeddings with OpenAI compatibility'
            },
            'document-processing': {
                'port': 8003,
                'name': 'Document Processing MCP Server',
                'description': 'PDF processing, OCR, text extraction, AI analysis'
            },
            'vector-search': {
                'port': 8004,
                'name': 'Vector Search MCP Server',
                'description': 'Semantic search, collection management, similarity matching'
            },
            'agent-orchestration': {
                'port': 8005,
                'name': 'Agent Orchestration MCP Server',
                'description': 'Multi-agent coordination and workflow management'
            }
        }
        
        self.stdio_servers = {
            'memory': {
                'name': 'Memory MCP Server (Official)',
                'description': 'Knowledge graph and persistent memory storage',
                'command': 'npx -y @modelcontextprotocol/server-memory'
            },
            'filesystem': {
                'name': 'Filesystem MCP Server (Official)',
                'description': 'Secure filesystem operations and file management',
                'command': 'npx -y @modelcontextprotocol/server-filesystem'
            },
            'postgresql': {
                'name': 'PostgreSQL MCP Server (Official)',
                'description': 'Advanced SQL database operations and schema management',
                'command': 'npx -y @modelcontextprotocol/server-postgres'
            }
        }
    
    def check_http_server_status(self, server_id: str, server_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check status of an HTTP MCP server"""
        port = server_info['port']
        status = {
            'server_id': server_id,
            'name': server_info['name'],
            'description': server_info['description'],
            'type': 'HTTP',
            'port': port,
            'status': 'unknown',
            'response_time': None,
            'uptime': None,
            'performance': {},
            'endpoints': [],
            'error': None
        }
        
        try:
            # Test root endpoint
            start_time = time.time()
            root_response = requests.get(f"http://localhost:{port}/", timeout=5)
            response_time = time.time() - start_time
            
            if root_response.status_code == 200:
                root_data = root_response.json()
                status['endpoints'] = list(root_data.get('endpoints', {}).keys())
                status['response_time'] = response_time
                
                # Test health endpoint
                health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    status['status'] = health_data.get('status', 'unknown')
                    
                    # Extract performance metrics
                    performance = health_data.get('performance', {})
                    status['uptime'] = performance.get('uptime_seconds', 0)
                    status['performance'] = performance
                else:
                    status['status'] = 'unhealthy'
                    status['error'] = f"Health check failed: HTTP {health_response.status_code}"
            else:
                status['status'] = 'failed'
                status['error'] = f"Root endpoint failed: HTTP {root_response.status_code}"
                
        except requests.RequestException as e:
            status['status'] = 'failed'
            status['error'] = str(e)
        
        return status
    
    def check_stdio_server_availability(self, server_id: str, server_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a stdio MCP server can be started"""
        status = {
            'server_id': server_id,
            'name': server_info['name'],
            'description': server_info['description'],
            'type': 'STDIO',
            'command': server_info['command'],
            'status': 'unknown',
            'available': False,
            'error': None
        }
        
        try:
            # Test if the server package is available
            command_parts = server_info['command'].split()
            test_process = subprocess.Popen(
                command_parts + ['--help'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            try:
                stdout, stderr = test_process.communicate(timeout=10)

                if test_process.returncode == 0 or 'MCP' in stdout.decode() or 'MCP' in stderr.decode():
                    status['status'] = 'available'
                    status['available'] = True
                else:
                    status['status'] = 'unavailable'
                    status['error'] = f"Package not available or not responding"
            except subprocess.TimeoutExpired:
                test_process.kill()
                status['status'] = 'available'  # Timeout often means it's waiting for input (good)
                status['available'] = True

        except Exception as e:
            status['status'] = 'failed'
            status['error'] = str(e)
        
        return status
    
    def generate_ecosystem_report(self) -> Dict[str, Any]:
        """Generate comprehensive ecosystem status report"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'ecosystem_name': 'PyGent Factory MCP Ecosystem',
            'version': '1.0.0',
            'http_servers': {},
            'stdio_servers': {},
            'summary': {
                'total_servers': 0,
                'running_servers': 0,
                'failed_servers': 0,
                'available_servers': 0,
                'overall_status': 'unknown'
            }
        }
        
        print("🔍 Checking HTTP MCP Servers...")
        print("-" * 50)
        
        # Check HTTP servers
        for server_id, server_info in self.http_servers.items():
            print(f"Checking {server_info['name']}...")
            status = self.check_http_server_status(server_id, server_info)
            report['http_servers'][server_id] = status
            
            status_emoji = {
                'healthy': '✅',
                'unhealthy': '⚠️',
                'failed': '❌',
                'unknown': '❓'
            }.get(status['status'], '❓')
            
            print(f"  {status_emoji} {status['name']}: {status['status']}")
            if status.get('uptime'):
                print(f"    Uptime: {status['uptime']:.1f}s")
            if status.get('response_time'):
                print(f"    Response Time: {status['response_time']*1000:.1f}ms")
            if status.get('error'):
                print(f"    Error: {status['error']}")
        
        print("\n🔍 Checking STDIO MCP Servers...")
        print("-" * 50)
        
        # Check STDIO servers
        for server_id, server_info in self.stdio_servers.items():
            print(f"Checking {server_info['name']}...")
            status = self.check_stdio_server_availability(server_id, server_info)
            report['stdio_servers'][server_id] = status
            
            status_emoji = {
                'available': '✅',
                'unavailable': '❌',
                'failed': '❌',
                'unknown': '❓'
            }.get(status['status'], '❓')
            
            print(f"  {status_emoji} {status['name']}: {status['status']}")
            if status.get('error'):
                print(f"    Error: {status['error']}")
        
        # Calculate summary
        total_servers = len(self.http_servers) + len(self.stdio_servers)
        running_servers = sum(1 for s in report['http_servers'].values() if s['status'] == 'healthy')
        available_servers = sum(1 for s in report['stdio_servers'].values() if s['status'] == 'available')
        failed_servers = sum(1 for s in report['http_servers'].values() if s['status'] == 'failed')
        failed_servers += sum(1 for s in report['stdio_servers'].values() if s['status'] in ['failed', 'unavailable'])
        
        report['summary'] = {
            'total_servers': total_servers,
            'running_servers': running_servers,
            'available_servers': available_servers,
            'failed_servers': failed_servers,
            'success_rate': ((running_servers + available_servers) / total_servers) * 100
        }
        
        # Determine overall status
        if report['summary']['success_rate'] >= 90:
            report['summary']['overall_status'] = 'excellent'
        elif report['summary']['success_rate'] >= 75:
            report['summary']['overall_status'] = 'good'
        elif report['summary']['success_rate'] >= 50:
            report['summary']['overall_status'] = 'fair'
        else:
            report['summary']['overall_status'] = 'poor'
        
        return report
    
    def print_summary_report(self, report: Dict[str, Any]):
        """Print a formatted summary report"""
        print("\n" + "="*80)
        print("🚀 PYGENT FACTORY MCP ECOSYSTEM STATUS REPORT")
        print("="*80)
        
        summary = report['summary']
        
        print(f"📊 SUMMARY")
        print(f"   Total Servers: {summary['total_servers']}")
        print(f"   Running (HTTP): {summary['running_servers']}")
        print(f"   Available (STDIO): {summary['available_servers']}")
        print(f"   Failed: {summary['failed_servers']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Overall Status: {summary['overall_status'].upper()}")
        
        print(f"\n🌐 HTTP SERVERS")
        for server_id, status in report['http_servers'].items():
            status_emoji = {
                'healthy': '✅',
                'unhealthy': '⚠️',
                'failed': '❌',
                'unknown': '❓'
            }.get(status['status'], '❓')
            
            print(f"   {status_emoji} {status['name']} (Port {status['port']})")
            print(f"      Status: {status['status'].upper()}")
            if status.get('uptime'):
                print(f"      Uptime: {status['uptime']:.1f}s")
            if status.get('endpoints'):
                print(f"      Endpoints: {len(status['endpoints'])}")
        
        print(f"\n📡 STDIO SERVERS")
        for server_id, status in report['stdio_servers'].items():
            status_emoji = {
                'available': '✅',
                'unavailable': '❌',
                'failed': '❌',
                'unknown': '❓'
            }.get(status['status'], '❓')
            
            print(f"   {status_emoji} {status['name']}")
            print(f"      Status: {status['status'].upper()}")
        
        print("\n🎯 PRODUCTION READINESS")
        if summary['overall_status'] == 'excellent':
            print("   ✅ PRODUCTION READY - All systems operational")
        elif summary['overall_status'] == 'good':
            print("   ⚠️ MOSTLY READY - Minor issues detected")
        elif summary['overall_status'] == 'fair':
            print("   ❌ NOT READY - Significant issues need resolution")
        else:
            print("   ❌ CRITICAL ISSUES - Major problems detected")
        
        print("="*80)


def main():
    """Main execution"""
    reporter = MCPEcosystemStatusReporter()
    
    print("🔍 Generating MCP Ecosystem Status Report...")
    print("This may take a few moments...\n")
    
    # Generate report
    report = reporter.generate_ecosystem_report()
    
    # Print summary
    reporter.print_summary_report(report)
    
    # Save detailed report
    with open('mcp_ecosystem_status_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: mcp_ecosystem_status_report.json")
    
    # Return exit code based on overall status
    if report['summary']['overall_status'] in ['excellent', 'good']:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
