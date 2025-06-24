#!/usr/bin/env python3
"""
Diagnose System-Level Delays

Investigate system-level causes of the persistent 2-second delays.
"""

import time
import socket
import requests
import subprocess
import platform
from datetime import datetime
from typing import Dict, List, Any

class SystemDelayDiagnostic:
    """Diagnose system-level causes of delays"""
    
    def __init__(self):
        self.results = []
        self.system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture(),
            'processor': platform.processor()
        }
    
    def log_result(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log diagnostic result"""
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
    
    def test_localhost_resolution(self) -> bool:
        """Test localhost DNS resolution speed"""
        start = time.time()
        try:
            print("🔍 Testing Localhost DNS Resolution...")
            
            resolution_times = []
            
            for i in range(5):
                resolve_start = time.time()
                try:
                    socket.gethostbyname('localhost')
                    resolve_time = time.time() - resolve_start
                    resolution_times.append(resolve_time)
                    print(f"   Resolution {i+1}: {resolve_time*1000:.1f}ms")
                except Exception as e:
                    print(f"   Resolution {i+1}: ERROR ({str(e)})")
            
            duration = time.time() - start
            
            if resolution_times:
                avg_time = sum(resolution_times) / len(resolution_times)
                max_time = max(resolution_times)
                
                # Check if DNS resolution is causing delays
                if avg_time > 0.1:  # Over 100ms is suspicious
                    details = f"Slow DNS resolution: avg {avg_time*1000:.1f}ms, max {max_time*1000:.1f}ms"
                    self.log_result("Localhost DNS Resolution", False, details, duration)
                    return False
                else:
                    details = f"DNS resolution normal: avg {avg_time*1000:.1f}ms"
                    self.log_result("Localhost DNS Resolution", True, details, duration)
                    return True
            else:
                details = "No successful DNS resolutions"
                self.log_result("Localhost DNS Resolution", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Localhost DNS Resolution", False, f"Error: {str(e)}", duration)
            return False
    
    def test_tcp_connection_speed(self) -> bool:
        """Test TCP connection establishment speed"""
        start = time.time()
        try:
            print("\n🔗 Testing TCP Connection Speed...")
            
            ports = [8002, 8003, 8004, 8005]
            connection_times = []
            
            for port in ports:
                connect_start = time.time()
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex(('localhost', port))
                    connect_time = time.time() - connect_start
                    sock.close()
                    
                    if result == 0:  # Connection successful
                        connection_times.append(connect_time)
                        print(f"   Port {port}: {connect_time*1000:.1f}ms")
                    else:
                        print(f"   Port {port}: Connection failed")
                        
                except Exception as e:
                    print(f"   Port {port}: ERROR ({str(e)})")
            
            duration = time.time() - start
            
            if connection_times:
                avg_time = sum(connection_times) / len(connection_times)
                max_time = max(connection_times)
                
                # Check if TCP connection is causing delays
                if avg_time > 0.05:  # Over 50ms is suspicious for localhost
                    details = f"Slow TCP connections: avg {avg_time*1000:.1f}ms, max {max_time*1000:.1f}ms"
                    self.log_result("TCP Connection Speed", False, details, duration)
                    return False
                else:
                    details = f"TCP connections normal: avg {avg_time*1000:.1f}ms"
                    self.log_result("TCP Connection Speed", True, details, duration)
                    return True
            else:
                details = "No successful TCP connections"
                self.log_result("TCP Connection Speed", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("TCP Connection Speed", False, f"Error: {str(e)}", duration)
            return False
    
    def test_http_client_behavior(self) -> bool:
        """Test HTTP client behavior and timeouts"""
        start = time.time()
        try:
            print("\n🌐 Testing HTTP Client Behavior...")
            
            # Test different HTTP client configurations
            test_configs = [
                {"timeout": 1, "name": "Short timeout"},
                {"timeout": 5, "name": "Normal timeout"},
                {"timeout": 10, "name": "Long timeout"}
            ]
            
            response_times = []
            
            for config in test_configs:
                print(f"   Testing {config['name']} ({config['timeout']}s)...")
                
                for i in range(3):
                    try:
                        req_start = time.time()
                        response = requests.get(
                            "http://localhost:8002/health",
                            timeout=config['timeout']
                        )
                        req_time = time.time() - req_start
                        
                        if response.status_code == 200:
                            response_times.append(req_time)
                            print(f"     Request {i+1}: {req_time*1000:.1f}ms")
                        else:
                            print(f"     Request {i+1}: HTTP {response.status_code}")
                            
                    except requests.exceptions.Timeout:
                        print(f"     Request {i+1}: TIMEOUT")
                    except Exception as e:
                        print(f"     Request {i+1}: ERROR ({str(e)})")
            
            duration = time.time() - start
            
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                consistent_delay = all(abs(t - 2.0) < 0.1 for t in response_times)
                
                if consistent_delay:
                    details = f"Consistent 2-second delays detected: avg {avg_time*1000:.1f}ms"
                    self.log_result("HTTP Client Behavior", False, details, duration)
                    return False
                else:
                    details = f"Variable response times: avg {avg_time*1000:.1f}ms"
                    self.log_result("HTTP Client Behavior", True, details, duration)
                    return True
            else:
                details = "No successful HTTP requests"
                self.log_result("HTTP Client Behavior", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("HTTP Client Behavior", False, f"Error: {str(e)}", duration)
            return False
    
    def test_python_import_delays(self) -> bool:
        """Test if Python imports are causing delays"""
        start = time.time()
        try:
            print("\n📦 Testing Python Import Delays...")
            
            import_tests = [
                ("import requests", "requests"),
                ("import fastapi", "fastapi"),
                ("import uvicorn", "uvicorn"),
                ("from sentence_transformers import SentenceTransformer", "sentence_transformers"),
                ("import asyncio", "asyncio"),
                ("import time", "time")
            ]
            
            slow_imports = []
            
            for import_stmt, module_name in import_tests:
                import_start = time.time()
                try:
                    exec(import_stmt)
                    import_time = time.time() - import_start
                    
                    print(f"   {module_name}: {import_time*1000:.1f}ms")
                    
                    if import_time > 0.5:  # Over 500ms is slow
                        slow_imports.append((module_name, import_time))
                        
                except Exception as e:
                    print(f"   {module_name}: ERROR ({str(e)})")
            
            duration = time.time() - start
            
            if slow_imports:
                slow_modules = [f"{name} ({time*1000:.1f}ms)" for name, time in slow_imports]
                details = f"Slow imports detected: {', '.join(slow_modules)}"
                self.log_result("Python Import Delays", False, details, duration)
                return False
            else:
                details = f"All imports fast (under 500ms)"
                self.log_result("Python Import Delays", True, details, duration)
                return True
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Python Import Delays", False, f"Error: {str(e)}", duration)
            return False
    
    def test_windows_specific_issues(self) -> bool:
        """Test Windows-specific issues that might cause delays"""
        start = time.time()
        try:
            print("\n🪟 Testing Windows-Specific Issues...")
            
            issues_found = []
            
            # Test Windows Defender real-time scanning
            try:
                # Check if we're on Windows
                if platform.system() == "Windows":
                    print("   Checking Windows Defender status...")
                    
                    # Try to check Windows Defender status
                    result = subprocess.run(
                        ["powershell", "-Command", "Get-MpPreference | Select-Object -Property DisableRealtimeMonitoring"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if "False" in result.stdout:
                        issues_found.append("Windows Defender real-time scanning enabled (may cause delays)")
                        print("   ⚠️ Windows Defender real-time scanning is enabled")
                    else:
                        print("   ✅ Windows Defender real-time scanning appears disabled")
                        
                else:
                    print("   Not running on Windows, skipping Windows-specific tests")
                    
            except Exception as e:
                print(f"   Could not check Windows Defender: {str(e)}")
            
            # Test Windows firewall
            try:
                if platform.system() == "Windows":
                    print("   Checking Windows Firewall...")
                    
                    result = subprocess.run(
                        ["netsh", "advfirewall", "show", "allprofiles", "state"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if "ON" in result.stdout:
                        issues_found.append("Windows Firewall is enabled (may cause connection delays)")
                        print("   ⚠️ Windows Firewall is enabled")
                    else:
                        print("   ✅ Windows Firewall appears disabled")
                        
            except Exception as e:
                print(f"   Could not check Windows Firewall: {str(e)}")
            
            duration = time.time() - start
            
            if issues_found:
                details = f"Windows issues found: {'; '.join(issues_found)}"
                self.log_result("Windows-Specific Issues", False, details, duration)
                return False
            else:
                details = "No Windows-specific issues detected"
                self.log_result("Windows-Specific Issues", True, details, duration)
                return True
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Windows-Specific Issues", False, f"Error: {str(e)}", duration)
            return False
    
    def test_process_startup_delays(self) -> bool:
        """Test if process startup is causing delays"""
        start = time.time()
        try:
            print("\n🚀 Testing Process Startup Delays...")
            
            # Test simple Python script execution time
            startup_times = []
            
            for i in range(3):
                startup_start = time.time()
                try:
                    result = subprocess.run(
                        ["python", "-c", "print('hello')"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    startup_time = time.time() - startup_start
                    
                    if result.returncode == 0:
                        startup_times.append(startup_time)
                        print(f"   Python startup {i+1}: {startup_time*1000:.1f}ms")
                    else:
                        print(f"   Python startup {i+1}: FAILED")
                        
                except Exception as e:
                    print(f"   Python startup {i+1}: ERROR ({str(e)})")
            
            duration = time.time() - start
            
            if startup_times:
                avg_time = sum(startup_times) / len(startup_times)
                
                if avg_time > 1.0:  # Over 1 second is slow
                    details = f"Slow Python startup: avg {avg_time*1000:.1f}ms"
                    self.log_result("Process Startup Delays", False, details, duration)
                    return False
                else:
                    details = f"Python startup normal: avg {avg_time*1000:.1f}ms"
                    self.log_result("Process Startup Delays", True, details, duration)
                    return True
            else:
                details = "No successful Python startups"
                self.log_result("Process Startup Delays", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Process Startup Delays", False, f"Error: {str(e)}", duration)
            return False
    
    def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run all system-level diagnostics"""
        print("🔧 System-Level Delay Diagnostics")
        print("=" * 50)
        
        print(f"System Info:")
        for key, value in self.system_info.items():
            print(f"   {key}: {value}")
        print()
        
        tests = [
            ("Localhost DNS Resolution", self.test_localhost_resolution),
            ("TCP Connection Speed", self.test_tcp_connection_speed),
            ("HTTP Client Behavior", self.test_http_client_behavior),
            ("Python Import Delays", self.test_python_import_delays),
            ("Windows-Specific Issues", self.test_windows_specific_issues),
            ("Process Startup Delays", self.test_process_startup_delays)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()
        
        total = len(tests)
        
        print("=" * 50)
        print(f"📊 System Diagnostics Results: {passed}/{total} tests passed")
        
        # Analyze results for root cause
        failed_tests = [result for result in self.results if not result['success']]
        
        if failed_tests:
            print(f"\n🚨 POTENTIAL ROOT CAUSES:")
            for result in failed_tests:
                print(f"   • {result['test']}: {result['details']}")
        else:
            print(f"\n✅ NO SYSTEM-LEVEL ISSUES DETECTED")
            print(f"   The 2-second delay is likely in application code.")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results,
            'system_info': self.system_info,
            'failed_tests': failed_tests
        }


def main():
    """Main execution"""
    diagnostic = SystemDelayDiagnostic()
    results = diagnostic.run_system_diagnostics()
    
    # Save results
    with open('system_delay_diagnostic_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: system_delay_diagnostic_results.json")
    
    return 0 if results['success_rate'] >= 70 else 1


if __name__ == "__main__":
    exit(main())
