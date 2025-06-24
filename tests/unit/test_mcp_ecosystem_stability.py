#!/usr/bin/env python3
"""
MCP Ecosystem Stability Test Suite

Comprehensive testing of the complete MCP ecosystem including:
- Server startup sequence validation
- Health monitoring and failover testing
- Load testing and stability under stress
- Resource usage monitoring
- Recovery testing after failures
"""

import asyncio
import json
import time
import requests
import psutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

class MCPEcosystemTester:
    """Comprehensive MCP ecosystem testing"""
    
    def __init__(self):
        self.results = []
        self.servers = {
            'postgresql': {'port': None, 'transport': 'stdio'},
            'github': {'port': None, 'transport': 'stdio'},
            'memory': {'port': None, 'transport': 'stdio'},
            'filesystem': {'port': None, 'transport': 'stdio'},
            'embedding': {'port': 8002, 'transport': 'http'},
            'document-processing': {'port': 8003, 'transport': 'http'},
            'vector-search': {'port': 8004, 'transport': 'http'},
            'agent-orchestration': {'port': 8005, 'transport': 'http'}
        }
        self.start_time = time.time()
    
    def log_result(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'duration_ms': round(duration * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name} ({duration*1000:.1f}ms)")
        if details:
            print(f"    {details}")
    
    def test_server_startup_sequence(self) -> bool:
        """Test that servers start in correct dependency order"""
        start = time.time()
        try:
            # Start the MCP server manager
            print("🚀 Starting MCP Server Manager...")
            
            # Import and run the manager
            import sys
            sys.path.append('src')
            from mcp_server_manager import MCPServerManager
            
            manager = MCPServerManager()
            
            # Check that startup order is correct
            expected_order = [
                'postgresql-official',
                'memory-official', 
                'github-official',
                'filesystem-python',
                'embedding-mcp-server',
                'document-processing-mcp-server',
                'vector-search-mcp-server',
                'agent-orchestration-mcp-server'
            ]
            
            actual_order = manager.startup_order
            
            # Check if critical servers are in correct order
            critical_servers_correct = True
            for i, server_id in enumerate(expected_order[:4]):  # First 4 are critical
                if server_id in actual_order:
                    actual_index = actual_order.index(server_id)
                    if actual_index > i + 2:  # Allow some flexibility
                        critical_servers_correct = False
                        break
            
            duration = time.time() - start
            
            if critical_servers_correct:
                details = f"Startup order validated: {len(actual_order)} servers configured"
                self.log_result("Startup Sequence Order", True, details, duration)
                return True
            else:
                details = f"Startup order incorrect. Expected: {expected_order[:4]}, Got: {actual_order[:4]}"
                self.log_result("Startup Sequence Order", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Startup Sequence Order", False, f"Error: {str(e)}", duration)
            return False
    
    def test_port_conflict_detection(self) -> bool:
        """Test port conflict detection"""
        start = time.time()
        try:
            import socket
            
            # Test port availability checking
            test_port = 8999
            
            # Occupy the port
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind(('localhost', test_port))
            server_socket.listen(1)
            
            # Test port conflict detection
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
                test_socket.settimeout(1)
                result = test_socket.connect_ex(('localhost', test_port))
                port_occupied = (result == 0)
            
            server_socket.close()
            
            duration = time.time() - start
            
            if port_occupied:
                self.log_result("Port Conflict Detection", True, f"Successfully detected port {test_port} conflict", duration)
                return True
            else:
                self.log_result("Port Conflict Detection", False, f"Failed to detect port {test_port} conflict", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Port Conflict Detection", False, f"Error: {str(e)}", duration)
            return False
    
    def test_http_server_health_endpoints(self) -> bool:
        """Test health endpoints of all HTTP servers"""
        start = time.time()
        try:
            http_servers = {k: v for k, v in self.servers.items() if v['transport'] == 'http'}
            healthy_servers = 0
            total_servers = len(http_servers)
            
            for server_name, server_info in http_servers.items():
                port = server_info['port']
                health_url = f"http://localhost:{port}/health"
                
                try:
                    response = requests.get(health_url, timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        if health_data.get('status') == 'healthy':
                            healthy_servers += 1
                            print(f"    ✅ {server_name} (port {port}): healthy")
                        else:
                            print(f"    ⚠️ {server_name} (port {port}): unhealthy status")
                    else:
                        print(f"    ❌ {server_name} (port {port}): HTTP {response.status_code}")
                except requests.RequestException as e:
                    print(f"    ❌ {server_name} (port {port}): {str(e)}")
            
            duration = time.time() - start
            
            if healthy_servers == total_servers:
                details = f"All {healthy_servers}/{total_servers} HTTP servers healthy"
                self.log_result("HTTP Health Endpoints", True, details, duration)
                return True
            else:
                details = f"Only {healthy_servers}/{total_servers} HTTP servers healthy"
                self.log_result("HTTP Health Endpoints", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("HTTP Health Endpoints", False, f"Error: {str(e)}", duration)
            return False
    
    def test_resource_usage_monitoring(self) -> bool:
        """Test resource usage monitoring"""
        start = time.time()
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check for reasonable resource usage
            cpu_ok = cpu_percent < 80  # Less than 80% CPU
            memory_ok = memory.percent < 85  # Less than 85% memory
            disk_ok = disk.percent < 90  # Less than 90% disk
            
            # Count running Python processes (our servers)
            python_processes = 0
            total_memory_mb = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes += 1
                        total_memory_mb += proc.info['memory_info'].rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            duration = time.time() - start
            
            resource_ok = cpu_ok and memory_ok and disk_ok
            
            details = f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk.percent:.1f}%, Python processes: {python_processes}, Total memory: {total_memory_mb:.1f}MB"
            
            if resource_ok:
                self.log_result("Resource Usage Monitoring", True, details, duration)
                return True
            else:
                self.log_result("Resource Usage Monitoring", False, f"High resource usage - {details}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Resource Usage Monitoring", False, f"Error: {str(e)}", duration)
            return False
    
    def test_concurrent_load_handling(self) -> bool:
        """Test concurrent load handling across all HTTP servers"""
        start = time.time()
        try:
            import concurrent.futures
            import threading
            
            def test_server_load(server_name: str, port: int) -> bool:
                """Test load on a single server"""
                try:
                    # Test health endpoint under load
                    health_url = f"http://localhost:{port}/health"
                    successful_requests = 0
                    
                    for _ in range(10):  # 10 requests per server
                        try:
                            response = requests.get(health_url, timeout=2)
                            if response.status_code == 200:
                                successful_requests += 1
                        except:
                            pass
                    
                    return successful_requests >= 8  # At least 80% success rate
                    
                except Exception:
                    return False
            
            # Test all HTTP servers concurrently
            http_servers = {k: v for k, v in self.servers.items() if v['transport'] == 'http'}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(http_servers)) as executor:
                futures = {
                    executor.submit(test_server_load, name, info['port']): name 
                    for name, info in http_servers.items()
                }
                
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    server_name = futures[future]
                    results[server_name] = future.result()
            
            duration = time.time() - start
            
            successful_servers = sum(1 for success in results.values() if success)
            total_servers = len(results)
            
            if successful_servers == total_servers:
                details = f"All {successful_servers}/{total_servers} servers handled concurrent load"
                self.log_result("Concurrent Load Handling", True, details, duration)
                return True
            else:
                failed_servers = [name for name, success in results.items() if not success]
                details = f"Only {successful_servers}/{total_servers} servers handled load. Failed: {failed_servers}"
                self.log_result("Concurrent Load Handling", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Concurrent Load Handling", False, f"Error: {str(e)}", duration)
            return False
    
    def test_service_integration(self) -> bool:
        """Test integration between services"""
        start = time.time()
        try:
            integration_tests = []
            
            # Test 1: Embedding + Vector Search integration
            try:
                # Generate embedding
                embedding_response = requests.post(
                    "http://localhost:8002/v1/embeddings",
                    json={"input": "test integration", "model": "sentence-transformers"},
                    timeout=10
                )
                
                if embedding_response.status_code == 200:
                    embedding_data = embedding_response.json()
                    embedding = embedding_data['data'][0]['embedding']
                    
                    # Create collection in vector search
                    collection_response = requests.post(
                        "http://localhost:8004/v1/collections",
                        json={"name": "integration_test", "dimension": len(embedding)},
                        timeout=10
                    )
                    
                    if collection_response.status_code == 200:
                        # Add document with embedding
                        doc_response = requests.post(
                            "http://localhost:8004/v1/collections/integration_test/documents",
                            json={
                                "collection": "integration_test",
                                "documents": [{
                                    "id": "test_doc",
                                    "content": "test integration",
                                    "embedding": embedding
                                }]
                            },
                            timeout=10
                        )
                        
                        if doc_response.status_code == 200:
                            integration_tests.append(("Embedding + Vector Search", True))
                        else:
                            integration_tests.append(("Embedding + Vector Search", False))
                    else:
                        integration_tests.append(("Embedding + Vector Search", False))
                else:
                    integration_tests.append(("Embedding + Vector Search", False))
                    
            except Exception as e:
                integration_tests.append(("Embedding + Vector Search", False))
            
            # Test 2: Agent Orchestration + Task Management
            try:
                # Create agent
                agent_response = requests.post(
                    "http://localhost:8005/v1/agents",
                    json={
                        "agent_type": "general",
                        "name": "Integration Test Agent",
                        "capabilities": ["test_capability"]
                    },
                    timeout=10
                )
                
                if agent_response.status_code == 200:
                    # Submit task
                    task_response = requests.post(
                        "http://localhost:8005/v1/tasks",
                        json={
                            "task_type": "test",
                            "description": "Integration test task",
                            "priority": "normal"
                        },
                        timeout=10
                    )
                    
                    if task_response.status_code == 200:
                        integration_tests.append(("Agent Orchestration + Tasks", True))
                    else:
                        integration_tests.append(("Agent Orchestration + Tasks", False))
                else:
                    integration_tests.append(("Agent Orchestration + Tasks", False))
                    
            except Exception as e:
                integration_tests.append(("Agent Orchestration + Tasks", False))
            
            duration = time.time() - start
            
            successful_integrations = sum(1 for _, success in integration_tests if success)
            total_integrations = len(integration_tests)
            
            if successful_integrations == total_integrations:
                details = f"All {successful_integrations}/{total_integrations} service integrations working"
                self.log_result("Service Integration", True, details, duration)
                return True
            else:
                failed_integrations = [name for name, success in integration_tests if not success]
                details = f"Only {successful_integrations}/{total_integrations} integrations working. Failed: {failed_integrations}"
                self.log_result("Service Integration", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Service Integration", False, f"Error: {str(e)}", duration)
            return False
    
    def test_ecosystem_stability_over_time(self) -> bool:
        """Test ecosystem stability over a period of time"""
        start = time.time()
        try:
            print("    🕐 Running 60-second stability test...")
            
            stability_checks = []
            test_duration = 60  # seconds
            check_interval = 10  # seconds
            
            end_time = time.time() + test_duration
            
            while time.time() < end_time:
                check_start = time.time()
                
                # Check all HTTP servers
                all_healthy = True
                for server_name, server_info in self.servers.items():
                    if server_info['transport'] == 'http':
                        port = server_info['port']
                        try:
                            response = requests.get(f"http://localhost:{port}/health", timeout=3)
                            if response.status_code != 200:
                                all_healthy = False
                                break
                        except:
                            all_healthy = False
                            break
                
                stability_checks.append(all_healthy)
                
                # Wait for next check
                elapsed = time.time() - check_start
                sleep_time = max(0, check_interval - elapsed)
                time.sleep(sleep_time)
            
            duration = time.time() - start
            
            stable_checks = sum(1 for check in stability_checks if check)
            total_checks = len(stability_checks)
            stability_rate = (stable_checks / total_checks) * 100 if total_checks > 0 else 0
            
            if stability_rate >= 90:  # 90% stability required
                details = f"Stability rate: {stability_rate:.1f}% ({stable_checks}/{total_checks} checks passed)"
                self.log_result("Ecosystem Stability Over Time", True, details, duration)
                return True
            else:
                details = f"Poor stability rate: {stability_rate:.1f}% ({stable_checks}/{total_checks} checks passed)"
                self.log_result("Ecosystem Stability Over Time", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Ecosystem Stability Over Time", False, f"Error: {str(e)}", duration)
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all ecosystem stability tests"""
        print("🏗️ MCP Ecosystem Stability Test Suite")
        print("=" * 60)
        
        tests = [
            ("Startup Sequence Order", self.test_server_startup_sequence),
            ("Port Conflict Detection", self.test_port_conflict_detection),
            ("HTTP Health Endpoints", self.test_http_server_health_endpoints),
            ("Resource Usage Monitoring", self.test_resource_usage_monitoring),
            ("Concurrent Load Handling", self.test_concurrent_load_handling),
            ("Service Integration", self.test_service_integration),
            ("Ecosystem Stability Over Time", self.test_ecosystem_stability_over_time)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()  # Add spacing
        
        total = len(tests)
        
        print("=" * 60)
        print(f"📊 Ecosystem Stability Results: {passed}/{total} tests passed")
        print(f"🕐 Total test duration: {time.time() - self.start_time:.1f} seconds")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results,
            'total_duration': time.time() - self.start_time
        }


def main():
    """Main test execution"""
    tester = MCPEcosystemTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('mcp_ecosystem_stability_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: mcp_ecosystem_stability_results.json")
    
    # Return appropriate exit code
    return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(main())
