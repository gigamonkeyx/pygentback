#!/usr/bin/env python3
"""
MCP Registry Integration Test Suite

Tests the integration of the embedding server with the MCP registry system.
"""

import json
import time
import requests
from typing import Dict, List, Any
from pathlib import Path

class MCPRegistryTester:
    """Test MCP registry integration"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.results = []
        self.config_file = Path("mcp_server_configs.json")
    
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': time.time()
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
    
    def test_config_file_exists(self) -> bool:
        """Test that MCP config file exists"""
        if self.config_file.exists():
            self.log_result("Config File Exists", True, f"Found at {self.config_file}")
            return True
        else:
            self.log_result("Config File Exists", False, f"Not found at {self.config_file}")
            return False
    
    def test_embedding_server_in_config(self) -> bool:
        """Test that embedding server is configured in MCP registry"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            servers = config.get('servers', [])
            embedding_servers = [s for s in servers if 'embedding' in s.get('id', '').lower()]
            
            if embedding_servers:
                server_config = embedding_servers[0]
                details = f"Found server: {server_config.get('id', 'unknown')}"
                self.log_result("Embedding Server in Config", True, details)
                return True
            else:
                self.log_result("Embedding Server in Config", False, "No embedding server found in config")
                return False
                
        except Exception as e:
            self.log_result("Embedding Server in Config", False, f"Error reading config: {str(e)}")
            return False
    
    def test_server_configuration_validity(self) -> bool:
        """Test that server configuration is valid"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            servers = config.get('servers', [])
            embedding_servers = [s for s in servers if 'embedding' in s.get('id', '').lower()]
            
            if not embedding_servers:
                self.log_result("Config Validity", False, "No embedding server found")
                return False
            
            server_config = embedding_servers[0]
            
            # Check required fields
            required_fields = ['id', 'name', 'command', 'capabilities']
            missing_fields = [field for field in required_fields if field not in server_config]
            
            if missing_fields:
                self.log_result("Config Validity", False, f"Missing fields: {missing_fields}")
                return False
            
            # Check capabilities
            capabilities = server_config.get('capabilities', [])
            expected_capabilities = ['embeddings', 'vector-operations']
            has_expected = any(cap in capabilities for cap in expected_capabilities)
            
            if has_expected:
                self.log_result("Config Validity", True, f"Valid config with capabilities: {capabilities}")
                return True
            else:
                self.log_result("Config Validity", False, f"Missing expected capabilities: {expected_capabilities}")
                return False
                
        except Exception as e:
            self.log_result("Config Validity", False, f"Error validating config: {str(e)}")
            return False
    
    def test_server_accessibility(self) -> bool:
        """Test that server is accessible at configured endpoint"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            servers = config.get('servers', [])
            embedding_servers = [s for s in servers if 'embedding' in s.get('id', '').lower()]
            
            if not embedding_servers:
                self.log_result("Server Accessibility", False, "No embedding server config found")
                return False
            
            server_config = embedding_servers[0]
            host = server_config.get('host', 'localhost')
            port = server_config.get('port', 8001)
            
            # Test health endpoint
            test_url = f"http://{host}:{port}/health"
            response = requests.get(test_url, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get('status', 'unknown')
                self.log_result("Server Accessibility", True, f"Server accessible at {test_url}, status: {status}")
                return True
            else:
                self.log_result("Server Accessibility", False, f"HTTP {response.status_code} from {test_url}")
                return False
                
        except Exception as e:
            self.log_result("Server Accessibility", False, f"Error accessing server: {str(e)}")
            return False
    
    def test_endpoint_configuration(self) -> bool:
        """Test that configured endpoints match actual server endpoints"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            servers = config.get('servers', [])
            embedding_servers = [s for s in servers if 'embedding' in s.get('id', '').lower()]
            
            if not embedding_servers:
                self.log_result("Endpoint Configuration", False, "No embedding server config found")
                return False
            
            server_config = embedding_servers[0]
            configured_endpoints = server_config.get('config', {}).get('endpoints', {})
            
            # Test actual server endpoints
            host = server_config.get('host', 'localhost')
            port = server_config.get('port', 8001)
            base_url = f"http://{host}:{port}"
            
            # Test root endpoint to get actual endpoints
            response = requests.get(f"{base_url}/", timeout=10)
            if response.status_code == 200:
                actual_endpoints = response.json().get('endpoints', {})
                
                # Compare configured vs actual
                matches = []
                mismatches = []
                
                for endpoint_name, endpoint_path in configured_endpoints.items():
                    if endpoint_name in actual_endpoints:
                        if actual_endpoints[endpoint_name] == endpoint_path:
                            matches.append(endpoint_name)
                        else:
                            mismatches.append(f"{endpoint_name}: config={endpoint_path}, actual={actual_endpoints[endpoint_name]}")
                    else:
                        mismatches.append(f"{endpoint_name}: not found in actual endpoints")
                
                if mismatches:
                    self.log_result("Endpoint Configuration", False, f"Mismatches: {mismatches}")
                    return False
                else:
                    self.log_result("Endpoint Configuration", True, f"All endpoints match: {matches}")
                    return True
            else:
                self.log_result("Endpoint Configuration", False, f"Cannot access server root endpoint")
                return False
                
        except Exception as e:
            self.log_result("Endpoint Configuration", False, f"Error checking endpoints: {str(e)}")
            return False
    
    def test_auto_start_configuration(self) -> bool:
        """Test auto-start and restart configuration"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            servers = config.get('servers', [])
            embedding_servers = [s for s in servers if 'embedding' in s.get('id', '').lower()]
            
            if not embedding_servers:
                self.log_result("Auto-Start Config", False, "No embedding server config found")
                return False
            
            server_config = embedding_servers[0]
            
            # Check auto-start settings
            auto_start = server_config.get('auto_start', False)
            restart_on_failure = server_config.get('restart_on_failure', False)
            max_restarts = server_config.get('max_restarts', 0)
            
            config_details = f"auto_start: {auto_start}, restart_on_failure: {restart_on_failure}, max_restarts: {max_restarts}"
            
            # For production, we want auto_start enabled
            if auto_start:
                self.log_result("Auto-Start Config", True, config_details)
                return True
            else:
                self.log_result("Auto-Start Config", False, f"Auto-start disabled. {config_details}")
                return False
                
        except Exception as e:
            self.log_result("Auto-Start Config", False, f"Error checking auto-start config: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all MCP registry integration tests"""
        print("🔗 MCP Registry Integration Test Suite")
        print("=" * 40)
        
        tests = [
            ("Config File Exists", self.test_config_file_exists),
            ("Embedding Server in Config", self.test_embedding_server_in_config),
            ("Config Validity", self.test_server_configuration_validity),
            ("Server Accessibility", self.test_server_accessibility),
            ("Endpoint Configuration", self.test_endpoint_configuration),
            ("Auto-Start Config", self.test_auto_start_configuration)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()  # Add spacing
        
        total = len(tests)
        
        print("=" * 40)
        print(f"📊 Registry Integration Results: {passed}/{total} tests passed")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results
        }


def main():
    """Main test execution"""
    tester = MCPRegistryTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('mcp_registry_integration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: mcp_registry_integration_results.json")
    return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(main())
