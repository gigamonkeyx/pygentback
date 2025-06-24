#!/usr/bin/env python3
"""
Check Proxy Configuration

Check for proxy settings that could be causing 2-second delays.
"""

import os
import subprocess
import requests
import time
from typing import Dict, Any

class ProxyChecker:
    """Check proxy configurations that might cause delays"""
    
    def __init__(self):
        self.results = {}
    
    def check_environment_proxies(self) -> Dict[str, Any]:
        """Check environment variable proxy settings"""
        print("🔍 Checking Environment Proxy Variables...")
        
        proxy_vars = [
            'HTTP_PROXY', 'HTTPS_PROXY', 'FTP_PROXY', 'SOCKS_PROXY',
            'http_proxy', 'https_proxy', 'ftp_proxy', 'socks_proxy',
            'NO_PROXY', 'no_proxy'
        ]
        
        env_proxies = {}
        for var in proxy_vars:
            value = os.environ.get(var)
            if value:
                env_proxies[var] = value
                print(f"   ⚠️ {var}: {value}")
            else:
                print(f"   ✅ {var}: Not set")
        
        return env_proxies
    
    def check_requests_proxy_behavior(self) -> Dict[str, Any]:
        """Check how requests library handles proxies"""
        print("\n🌐 Testing Requests Library Proxy Behavior...")
        
        # Test with no proxy
        print("   Testing without proxy...")
        start_time = time.time()
        try:
            response = requests.get("http://localhost:8003/health", timeout=5, proxies={})
            no_proxy_time = time.time() - start_time
            print(f"   No proxy: {no_proxy_time*1000:.1f}ms")
        except Exception as e:
            no_proxy_time = None
            print(f"   No proxy: ERROR ({str(e)})")
        
        # Test with system proxy (default)
        print("   Testing with system proxy...")
        start_time = time.time()
        try:
            response = requests.get("http://localhost:8003/health", timeout=5)
            system_proxy_time = time.time() - start_time
            print(f"   System proxy: {system_proxy_time*1000:.1f}ms")
        except Exception as e:
            system_proxy_time = None
            print(f"   System proxy: ERROR ({str(e)})")
        
        return {
            'no_proxy_time': no_proxy_time,
            'system_proxy_time': system_proxy_time
        }
    
    def check_windows_proxy_settings(self) -> Dict[str, Any]:
        """Check Windows proxy settings"""
        print("\n🪟 Checking Windows Proxy Settings...")
        
        try:
            # Check Windows proxy settings via registry
            result = subprocess.run([
                "reg", "query", 
                "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings",
                "/v", "ProxyEnable"
            ], capture_output=True, text=True, timeout=10)
            
            proxy_enabled = "0x1" in result.stdout
            print(f"   Proxy Enabled: {proxy_enabled}")
            
            if proxy_enabled:
                # Get proxy server
                result = subprocess.run([
                    "reg", "query", 
                    "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings",
                    "/v", "ProxyServer"
                ], capture_output=True, text=True, timeout=10)
                
                proxy_server = None
                for line in result.stdout.split('\n'):
                    if 'ProxyServer' in line and 'REG_SZ' in line:
                        proxy_server = line.split('REG_SZ')[-1].strip()
                        break
                
                print(f"   Proxy Server: {proxy_server}")
                
                return {
                    'proxy_enabled': proxy_enabled,
                    'proxy_server': proxy_server
                }
            else:
                return {'proxy_enabled': False}
                
        except Exception as e:
            print(f"   Error checking Windows proxy: {str(e)}")
            return {'error': str(e)}
    
    def check_localhost_bypass(self) -> Dict[str, Any]:
        """Check if localhost is bypassed in proxy settings"""
        print("\n🏠 Checking Localhost Proxy Bypass...")
        
        try:
            # Check proxy bypass list
            result = subprocess.run([
                "reg", "query", 
                "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings",
                "/v", "ProxyOverride"
            ], capture_output=True, text=True, timeout=10)
            
            bypass_list = None
            for line in result.stdout.split('\n'):
                if 'ProxyOverride' in line and 'REG_SZ' in line:
                    bypass_list = line.split('REG_SZ')[-1].strip()
                    break
            
            print(f"   Proxy Bypass List: {bypass_list}")
            
            localhost_bypassed = False
            if bypass_list:
                bypass_entries = [entry.strip() for entry in bypass_list.split(';')]
                localhost_bypassed = any(
                    'localhost' in entry.lower() or '127.0.0.1' in entry or '<local>' in entry.lower()
                    for entry in bypass_entries
                )
            
            print(f"   Localhost Bypassed: {localhost_bypassed}")
            
            return {
                'bypass_list': bypass_list,
                'localhost_bypassed': localhost_bypassed
            }
            
        except Exception as e:
            print(f"   Error checking proxy bypass: {str(e)}")
            return {'error': str(e)}
    
    def test_proxy_timeout_behavior(self) -> Dict[str, Any]:
        """Test if proxy is causing timeout behavior"""
        print("\n⏱️ Testing Proxy Timeout Behavior...")
        
        # Test different proxy configurations
        test_configs = [
            {'name': 'No proxy', 'proxies': {}},
            {'name': 'Explicit no proxy', 'proxies': {'http': None, 'https': None}},
            {'name': 'System default', 'proxies': None}
        ]
        
        results = {}
        
        for config in test_configs:
            print(f"   Testing {config['name']}...")
            times = []
            
            for i in range(3):
                try:
                    start_time = time.time()
                    if config['proxies'] is not None:
                        response = requests.get(
                            "http://localhost:8003/health", 
                            timeout=5, 
                            proxies=config['proxies']
                        )
                    else:
                        response = requests.get("http://localhost:8003/health", timeout=5)
                    
                    req_time = time.time() - start_time
                    times.append(req_time)
                    print(f"     Request {i+1}: {req_time*1000:.1f}ms")
                    
                except Exception as e:
                    print(f"     Request {i+1}: ERROR ({str(e)})")
            
            if times:
                avg_time = sum(times) / len(times)
                results[config['name']] = {
                    'avg_time': avg_time,
                    'times': times,
                    'is_slow': avg_time > 1.0
                }
                print(f"     Average: {avg_time*1000:.1f}ms")
            else:
                results[config['name']] = {'error': 'No successful requests'}
        
        return results
    
    def run_proxy_check(self) -> Dict[str, Any]:
        """Run complete proxy check"""
        print("🔍 Proxy Configuration Check")
        print("=" * 50)
        
        self.results['env_proxies'] = self.check_environment_proxies()
        self.results['requests_behavior'] = self.check_requests_proxy_behavior()
        self.results['windows_proxy'] = self.check_windows_proxy_settings()
        self.results['localhost_bypass'] = self.check_localhost_bypass()
        self.results['timeout_behavior'] = self.test_proxy_timeout_behavior()
        
        print("\n" + "=" * 50)
        print("📊 PROXY ANALYSIS SUMMARY")
        print("=" * 50)
        
        # Analyze results
        issues_found = []
        
        # Check for environment proxy variables
        if self.results['env_proxies']:
            issues_found.append(f"Environment proxy variables set: {list(self.results['env_proxies'].keys())}")
        
        # Check Windows proxy settings
        windows_proxy = self.results.get('windows_proxy', {})
        if windows_proxy.get('proxy_enabled'):
            issues_found.append(f"Windows proxy enabled: {windows_proxy.get('proxy_server')}")
        
        # Check localhost bypass
        localhost_bypass = self.results.get('localhost_bypass', {})
        if not localhost_bypass.get('localhost_bypassed', True):
            issues_found.append("Localhost NOT bypassed in proxy settings")
        
        # Check timeout behavior
        timeout_behavior = self.results.get('timeout_behavior', {})
        slow_configs = [name for name, data in timeout_behavior.items() if data.get('is_slow')]
        if slow_configs:
            issues_found.append(f"Slow proxy configurations: {slow_configs}")
        
        if issues_found:
            print("🚨 PROXY ISSUES FOUND:")
            for issue in issues_found:
                print(f"   • {issue}")
            
            print("\n💡 RECOMMENDED FIXES:")
            if self.results['env_proxies']:
                print("   • Unset proxy environment variables for localhost testing")
            if windows_proxy.get('proxy_enabled') and not localhost_bypass.get('localhost_bypassed'):
                print("   • Add localhost to Windows proxy bypass list")
            if slow_configs:
                print("   • Use explicit no-proxy configuration for localhost requests")
                
        else:
            print("✅ NO PROXY ISSUES DETECTED")
            print("   The 2-second delay is likely caused by something else.")
        
        return self.results


def main():
    """Main execution"""
    checker = ProxyChecker()
    results = checker.run_proxy_check()
    
    # Save results
    import json
    with open('proxy_check_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: proxy_check_results.json")
    
    # Return exit code based on issues found
    has_issues = bool(
        results.get('env_proxies') or 
        results.get('windows_proxy', {}).get('proxy_enabled') or
        not results.get('localhost_bypass', {}).get('localhost_bypassed', True)
    )
    
    return 1 if has_issues else 0


if __name__ == "__main__":
    exit(main())
