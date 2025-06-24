#!/usr/bin/env python3
"""
Comprehensive frontend connectivity diagnosis for timpayne.net/pygent
"""

import asyncio
import aiohttp
import socket
import subprocess
import json
from datetime import datetime

class FrontendConnectivityDiagnosis:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {},
            "recommendations": []
        }
    
    async def test_dns_resolution(self):
        """Test DNS resolution for all endpoints"""
        print("🔍 Testing DNS Resolution...")
        
        domains = [
            "timpayne.net",
            "api.timpayne.net", 
            "ws.timpayne.net",
            "2c34f6aa-7978-4a1a-8410-50af0047925e.cfargotunnel.com"
        ]
        
        dns_results = {}
        
        for domain in domains:
            try:
                ip = socket.gethostbyname(domain)
                dns_results[domain] = {"status": "✅ RESOLVED", "ip": ip}
                print(f"  ✅ {domain} -> {ip}")
            except socket.gaierror as e:
                dns_results[domain] = {"status": "❌ FAILED", "error": str(e)}
                print(f"  ❌ {domain} -> DNS resolution failed: {e}")
        
        self.results["tests"]["dns_resolution"] = dns_results
        return dns_results
    
    async def test_local_backend(self):
        """Test local backend connectivity"""
        print("🔍 Testing Local Backend...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/api/v1/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"  ✅ Local backend healthy: {data}")
                        self.results["tests"]["local_backend"] = {"status": "✅ HEALTHY", "data": data}
                        return True
                    else:
                        print(f"  ❌ Local backend unhealthy: {response.status}")
                        self.results["tests"]["local_backend"] = {"status": "❌ UNHEALTHY", "status_code": response.status}
                        return False
        except Exception as e:
            print(f"  ❌ Local backend connection failed: {e}")
            self.results["tests"]["local_backend"] = {"status": "❌ FAILED", "error": str(e)}
            return False
    
    async def test_tunnel_status(self):
        """Test Cloudflare tunnel status"""
        print("🔍 Testing Cloudflare Tunnel Status...")
        
        try:
            # Run cloudflared tunnel list command
            result = subprocess.run(
                ["cloudflared", "tunnel", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                output = result.stdout
                if "pygent-factory-v2" in output and "CONNECTIONS" in output:
                    print("  ✅ Tunnel is running with connections")
                    self.results["tests"]["tunnel_status"] = {"status": "✅ RUNNING", "output": output}
                    return True
                else:
                    print("  ❌ Tunnel not found or no connections")
                    self.results["tests"]["tunnel_status"] = {"status": "❌ NO_CONNECTIONS", "output": output}
                    return False
            else:
                print(f"  ❌ Tunnel command failed: {result.stderr}")
                self.results["tests"]["tunnel_status"] = {"status": "❌ COMMAND_FAILED", "error": result.stderr}
                return False
                
        except Exception as e:
            print(f"  ❌ Tunnel status check failed: {e}")
            self.results["tests"]["tunnel_status"] = {"status": "❌ FAILED", "error": str(e)}
            return False
    
    async def test_frontend_loading(self):
        """Test if the frontend loads properly"""
        print("🔍 Testing Frontend Loading...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://timpayne.net/pygent") as response:
                    if response.status == 200:
                        content = await response.text()
                        if "PyGent Factory" in content or "React" in content:
                            print("  ✅ Frontend loads successfully")
                            self.results["tests"]["frontend_loading"] = {"status": "✅ LOADED"}
                            return True
                        else:
                            print("  ⚠️  Frontend loads but content unexpected")
                            self.results["tests"]["frontend_loading"] = {"status": "⚠️ UNEXPECTED_CONTENT"}
                            return False
                    else:
                        print(f"  ❌ Frontend failed to load: {response.status}")
                        self.results["tests"]["frontend_loading"] = {"status": "❌ FAILED", "status_code": response.status}
                        return False
        except Exception as e:
            print(f"  ❌ Frontend loading test failed: {e}")
            self.results["tests"]["frontend_loading"] = {"status": "❌ FAILED", "error": str(e)}
            return False
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        print("📋 Generating Recommendations...")
        
        recommendations = []
        
        # Check DNS issues
        dns_results = self.results["tests"].get("dns_resolution", {})
        if any("FAILED" in result.get("status", "") for result in dns_results.values()):
            recommendations.append({
                "priority": "HIGH",
                "issue": "DNS Resolution Failed",
                "action": "Configure DNS records in Cloudflare dashboard",
                "details": "Add CNAME records for api.timpayne.net and ws.timpayne.net pointing to tunnel domain"
            })
        
        # Check tunnel issues
        tunnel_status = self.results["tests"].get("tunnel_status", {})
        if "FAILED" in tunnel_status.get("status", ""):
            recommendations.append({
                "priority": "HIGH", 
                "issue": "Tunnel Not Running",
                "action": "Start Cloudflare tunnel",
                "details": "Run: cloudflared tunnel run pygent-factory-v2"
            })
        
        # Check backend issues
        backend_status = self.results["tests"].get("local_backend", {})
        if "FAILED" in backend_status.get("status", ""):
            recommendations.append({
                "priority": "HIGH",
                "issue": "Local Backend Not Running", 
                "action": "Start PyGent Factory backend",
                "details": "Run: python main.py server --host 0.0.0.0 --port 8000"
            })
        
        self.results["recommendations"] = recommendations
        
        for rec in recommendations:
            print(f"  🔧 {rec['priority']}: {rec['issue']}")
            print(f"     Action: {rec['action']}")
            print(f"     Details: {rec['details']}")
    
    async def run_diagnosis(self):
        """Run complete diagnosis"""
        print("🚀 Frontend Connectivity Diagnosis for timpayne.net/pygent")
        print("=" * 60)
        
        # Run all tests
        await self.test_dns_resolution()
        print()
        
        await self.test_local_backend()
        print()
        
        await self.test_tunnel_status()
        print()
        
        await self.test_frontend_loading()
        print()
        
        # Generate recommendations
        self.generate_recommendations()
        print()
        
        # Summary
        print("📊 Diagnosis Summary:")
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for test in self.results["tests"].values() 
                          if "✅" in test.get("status", ""))
        
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Recommendations: {len(self.results['recommendations'])}")
        
        # Save results
        with open("frontend_connectivity_diagnosis.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Full results saved to: frontend_connectivity_diagnosis.json")
        
        return self.results

async def main():
    diagnosis = FrontendConnectivityDiagnosis()
    await diagnosis.run_diagnosis()

if __name__ == "__main__":
    asyncio.run(main())
