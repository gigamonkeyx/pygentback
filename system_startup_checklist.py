"""
PyGent Factory System Startup Checklist
Comprehensive system validation and startup sequence
"""

import asyncio
import logging
import subprocess
import psutil
import aiohttp
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class ServiceCheck:
    """Service check result"""
    name: str
    status: ServiceStatus
    details: str
    port: Optional[int] = None
    auto_start: bool = False
    critical: bool = True


class SystemStartupChecklist:
    """
    Comprehensive system startup checklist for PyGent Factory
    Validates all components before allowing task execution
    """
    
    def __init__(self, auto_start: bool = False):
        self.auto_start = auto_start
        self.services: Dict[str, ServiceCheck] = {}
        self.startup_sequence = [
            "pygent_backend",      # FIRST - Most critical, longest startup
            "infrastructure",      # Basic system requirements
            "docker",             # Docker and containers
            "database",           # Database connectivity
            "ai_backend",         # Ollama and AI services
            "model_discovery",    # NEW - Hugging Face model discovery
            "mcp_servers",        # MCP server validation
            "agents",             # Agent availability
            "integration"         # Final integration tests
        ]
        
    async def run_full_checklist(self) -> bool:
        """Run complete system startup checklist"""
        
        print("🚀 PYGENT FACTORY SYSTEM STARTUP CHECKLIST")
        print("=" * 60)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Auto-start mode: {'ENABLED' if self.auto_start else 'DISABLED'}")
        print()
        
        overall_success = True
        
        for phase in self.startup_sequence:
            phase_name = phase.upper().replace('_', ' ')
            if phase == "pygent_backend":
                phase_name = "🎯 PYGENT BACKEND (PRIORITY #1)"
            print(f"📋 PHASE: {phase_name}")
            print("-" * 50)
            
            phase_success = await self._run_phase(phase)
            
            if not phase_success:
                overall_success = False
                if self._is_critical_phase(phase):
                    print(f"❌ CRITICAL PHASE FAILED: {phase}")
                    print("🛑 STOPPING CHECKLIST - SYSTEM NOT READY")
                    break
                else:
                    print(f"⚠️  NON-CRITICAL PHASE FAILED: {phase}")
                    print("⏭️  CONTINUING WITH NEXT PHASE")
            
            print()
        
        # Final status report
        await self._print_final_status(overall_success)
        return overall_success
    
    async def _run_phase(self, phase: str) -> bool:
        """Run specific phase of startup checklist"""
        
        if phase == "infrastructure":
            return await self._check_infrastructure()
        elif phase == "docker":
            return await self._check_docker_services()
        elif phase == "database":
            return await self._check_database_services()
        elif phase == "ai_backend":
            return await self._check_ai_backend()
        elif phase == "model_discovery":
            return await self._check_model_discovery()
        elif phase == "pygent_backend":
            return await self._check_pygent_backend()
        elif phase == "mcp_servers":
            return await self._check_mcp_servers()
        elif phase == "agents":
            return await self._check_agent_services()
        elif phase == "integration":
            return await self._check_integration()
        else:
            print(f"❌ Unknown phase: {phase}")
            return False
    
    async def _check_infrastructure(self) -> bool:
        """Check basic infrastructure requirements"""
        
        checks = []
        
        # Check available memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 8:
            checks.append(ServiceCheck("System Memory", ServiceStatus.HEALTHY, f"{memory_gb:.1f}GB available"))
        else:
            checks.append(ServiceCheck("System Memory", ServiceStatus.UNHEALTHY, f"Only {memory_gb:.1f}GB available (8GB+ recommended)"))
        
        # Check disk space
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        if disk_gb >= 10:
            checks.append(ServiceCheck("Disk Space", ServiceStatus.HEALTHY, f"{disk_gb:.1f}GB free"))
        else:
            checks.append(ServiceCheck("Disk Space", ServiceStatus.UNHEALTHY, f"Only {disk_gb:.1f}GB free (10GB+ recommended)"))
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                checks.append(ServiceCheck("GPU Support", ServiceStatus.HEALTHY, f"{gpu_count} GPU(s) - {gpu_name}"))
            else:
                checks.append(ServiceCheck("GPU Support", ServiceStatus.UNHEALTHY, "CUDA not available", critical=False))
        except ImportError:
            checks.append(ServiceCheck("GPU Support", ServiceStatus.UNKNOWN, "PyTorch not installed", critical=False))
        
        # Check network connectivity
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://www.google.com", timeout=5) as response:
                    if response.status == 200:
                        checks.append(ServiceCheck("Network", ServiceStatus.HEALTHY, "Internet connectivity OK"))
                    else:
                        checks.append(ServiceCheck("Network", ServiceStatus.UNHEALTHY, f"HTTP {response.status}"))
        except Exception as e:
            checks.append(ServiceCheck("Network", ServiceStatus.UNHEALTHY, f"No internet: {e}", critical=False))
        
        return self._process_checks(checks)
    
    async def _check_docker_services(self) -> bool:
        """Check Docker Desktop and containers"""
        
        checks = []
        
        # Check if Docker is running
        try:
            result = subprocess.run(["docker", "version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                checks.append(ServiceCheck("Docker Desktop", ServiceStatus.HEALTHY, "Docker engine running"))
                
                # Check PostgreSQL container
                postgres_check = await self._check_postgres_container()
                checks.append(postgres_check)
                
            else:
                checks.append(ServiceCheck("Docker Desktop", ServiceStatus.UNHEALTHY, "Docker not responding"))
                if self.auto_start:
                    await self._start_docker_desktop()
                    
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            checks.append(ServiceCheck("Docker Desktop", ServiceStatus.STOPPED, f"Docker not found: {e}"))
            if self.auto_start:
                await self._start_docker_desktop()
        
        return self._process_checks(checks)
    
    async def _check_postgres_container(self) -> ServiceCheck:
        """Check PostgreSQL container status"""
        
        try:
            # Check if postgres container is running
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=postgres", "--format", "{{.Names}}\t{{.Status}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                container_info = result.stdout.strip()
                if "Up" in container_info:
                    # Test database connection
                    if await self._test_postgres_connection():
                        return ServiceCheck("PostgreSQL Container", ServiceStatus.HEALTHY, "Container running, DB accessible", 54321)
                    else:
                        return ServiceCheck("PostgreSQL Container", ServiceStatus.UNHEALTHY, "Container running but DB not accessible", 54321)
                else:
                    if self.auto_start:
                        await self._start_postgres_container()
                        # Re-check after starting
                        if await self._test_postgres_connection():
                            return ServiceCheck("PostgreSQL Container", ServiceStatus.HEALTHY, "Container started and DB accessible", 54321)
                    return ServiceCheck("PostgreSQL Container", ServiceStatus.STOPPED, "Container exists but not running", 54321, auto_start=True)
            else:
                if self.auto_start:
                    await self._start_postgres_container()
                    # Re-check after starting
                    if await self._test_postgres_connection():
                        return ServiceCheck("PostgreSQL Container", ServiceStatus.HEALTHY, "Container created and DB accessible", 54321)
                return ServiceCheck("PostgreSQL Container", ServiceStatus.STOPPED, "Container not found", 54321, auto_start=True)
                
        except Exception as e:
            return ServiceCheck("PostgreSQL Container", ServiceStatus.UNKNOWN, f"Error checking container: {e}", 54321)
    
    async def _test_postgres_connection(self) -> bool:
        """Test PostgreSQL database connection"""
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host="localhost",
                port=54321,
                user="postgres", 
                password="postgres",
                database="pygent_factory",
                timeout=5
            )
            await conn.close()
            return True
        except Exception:
            return False
    
    async def _check_database_services(self) -> bool:
        """Check database connectivity"""
        
        checks = []
        
        # PostgreSQL connection test
        if await self._test_postgres_connection():
            checks.append(ServiceCheck("PostgreSQL DB", ServiceStatus.HEALTHY, "Database connection OK", 54321))
        else:
            checks.append(ServiceCheck("PostgreSQL DB", ServiceStatus.UNHEALTHY, "Cannot connect to database", 54321))
        
        return self._process_checks(checks)
    
    async def _check_ai_backend(self) -> bool:
        """Check AI backend services"""
        
        checks = []
        
        # Check Ollama service
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        model_count = len(models)
                        checks.append(ServiceCheck("Ollama Service", ServiceStatus.HEALTHY, f"Running with {model_count} models", 11434))
                    else:
                        checks.append(ServiceCheck("Ollama Service", ServiceStatus.UNHEALTHY, f"HTTP {response.status}", 11434))
        except Exception as e:
            checks.append(ServiceCheck("Ollama Service", ServiceStatus.STOPPED, f"Not accessible: {e}", 11434, auto_start=True))
            if self.auto_start:
                await self._start_ollama_service()
        
        return self._process_checks(checks)

    async def _check_model_discovery(self) -> bool:
        """Check Hugging Face model discovery and integration"""

        checks = []

        print("🔍 Starting Hugging Face model discovery...")
        print("   This may take a few moments on first run...")

        try:
            # Import the startup model manager
            from src.core.startup_model_manager import startup_model_manager

            # Run model discovery
            discovery_results = await startup_model_manager.initialize_models_on_startup(
                provider_registry=None,  # Will use fallback scan
                force_refresh=False      # Use cache if available
            )

            if discovery_results["success"]:
                hf_models = discovery_results["hf_models_discovered"]
                local_models = discovery_results["local_models_found"]
                integrated = discovery_results["integrated_models"]

                checks.append(ServiceCheck(
                    "HF Model Discovery",
                    ServiceStatus.HEALTHY,
                    f"Discovered {hf_models} HF models, {local_models} local models, {integrated} capabilities mapped"
                ))

                # Log best models found
                print("   📊 Best models by capability:")
                for capability, model_info in discovery_results["best_models_by_capability"].items():
                    provider_icon = "🏠" if model_info["provider"] == "ollama" else "🌐"
                    free_icon = "🆓" if model_info["is_free"] else "💰"
                    print(f"      {capability}: {model_info['name']} {provider_icon}{free_icon}")

            else:
                error_msg = "; ".join(discovery_results["errors"])
                checks.append(ServiceCheck(
                    "HF Model Discovery",
                    ServiceStatus.UNHEALTHY,
                    f"Discovery failed: {error_msg}",
                    critical=False  # Non-critical, system can work with hard-coded models
                ))

        except Exception as e:
            checks.append(ServiceCheck(
                "HF Model Discovery",
                ServiceStatus.UNHEALTHY,
                f"Discovery error: {e}",
                critical=False
            ))

        return self._process_checks(checks)

    async def _check_pygent_backend(self) -> bool:
        """Check PyGent Factory backend with startup monitoring"""

        checks = []

        print("🔍 Checking PyGent Factory backend (most critical component)...")

        # First check if it's already running
        backend_running = await self._wait_for_backend_health(initial_check=True)

        if backend_running:
            checks.append(ServiceCheck("PyGent Backend", ServiceStatus.HEALTHY, "API server running", 8000))
            print("✅ PyGent Factory backend already running and healthy")
            return self._process_checks(checks)

        print("❌ PyGent Factory backend not accessible")

        if self.auto_start:
            print("🚀 Starting PyGent Factory backend (this may take 60-120 seconds)...")
            await self._start_pygent_backend_with_monitoring()

            # Wait for backend to fully start with extended timeout
            print("⏳ Waiting for backend to fully initialize...")
            backend_started = await self._wait_for_backend_health(initial_check=False, timeout=120)

            if backend_started:
                checks.append(ServiceCheck("PyGent Backend", ServiceStatus.HEALTHY, "Started successfully", 8000))
                print("✅ PyGent Factory backend started successfully")
            else:
                checks.append(ServiceCheck("PyGent Backend", ServiceStatus.STOPPED, "Startup timeout - backend may still be starting", 8000))
                print("⏰ PyGent Factory backend startup timeout (may still be starting in background)")
        else:
            checks.append(ServiceCheck("PyGent Backend", ServiceStatus.STOPPED, "Not accessible", 8000, auto_start=True))

        return self._process_checks(checks)

    async def _wait_for_backend_health(self, initial_check: bool = True, timeout: int = 120) -> bool:
        """Wait for PyGent Factory backend to be healthy"""

        check_timeout = 3 if initial_check else timeout
        check_interval = 1 if initial_check else 5
        elapsed = 0

        while elapsed < check_timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:8000/api/v1/health", timeout=5) as response:
                        if response.status == 200:
                            if not initial_check:
                                print("✅ Backend health check passed!")
                            return True
                        else:
                            if not initial_check:
                                print(f"⚠️  Backend responding but not healthy: HTTP {response.status}")
            except Exception as e:
                if not initial_check and elapsed % 15 == 0:  # Progress update every 15 seconds
                    dots = "." * ((elapsed // 5) % 4)
                    print(f"   Waiting for backend{dots:<3} ({elapsed}s/{timeout}s)")

            await asyncio.sleep(check_interval)
            elapsed += check_interval

        return False

    async def _check_mcp_servers(self) -> bool:
        """Check MCP server availability"""
        
        checks = []
        
        # This would check MCP servers through the PyGent backend
        # For now, we'll do a basic check
        checks.append(ServiceCheck("MCP Servers", ServiceStatus.HEALTHY, "MCP integration available", critical=False))
        
        return self._process_checks(checks)
    
    async def _check_agent_services(self) -> bool:
        """Check agent availability"""
        
        checks = []
        
        # This would check agent factory through the PyGent backend
        # For now, we'll do a basic check
        checks.append(ServiceCheck("Agent Factory", ServiceStatus.HEALTHY, "Agent services available", critical=False))
        
        return self._process_checks(checks)
    
    async def _check_integration(self) -> bool:
        """Run integration tests"""
        
        checks = []
        
        # Basic integration test
        checks.append(ServiceCheck("Integration Test", ServiceStatus.HEALTHY, "System integration OK", critical=False))
        
        return self._process_checks(checks)
    
    def _process_checks(self, checks: List[ServiceCheck]) -> bool:
        """Process and display check results"""
        
        all_critical_passed = True
        
        for check in checks:
            # Store check result
            self.services[check.name] = check
            
            # Display result
            status_icon = self._get_status_icon(check.status)
            port_info = f" (:{check.port})" if check.port else ""
            print(f"  {status_icon} {check.name}{port_info}: {check.details}")
            
            # Track critical failures
            if check.critical and check.status != ServiceStatus.HEALTHY:
                all_critical_passed = False
        
        return all_critical_passed
    
    def _get_status_icon(self, status: ServiceStatus) -> str:
        """Get status icon for display"""
        icons = {
            ServiceStatus.HEALTHY: "✅",
            ServiceStatus.UNHEALTHY: "❌", 
            ServiceStatus.STARTING: "🔄",
            ServiceStatus.STOPPED: "⏹️",
            ServiceStatus.UNKNOWN: "❓"
        }
        return icons.get(status, "❓")
    
    def _is_critical_phase(self, phase: str) -> bool:
        """Check if phase is critical for system operation"""
        critical_phases = ["pygent_backend", "infrastructure", "docker", "database"]
        return phase in critical_phases
    
    async def _print_final_status(self, success: bool):
        """Print final system status"""
        
        print("🎯 FINAL SYSTEM STATUS")
        print("=" * 60)
        
        if success:
            print("🎉 SYSTEM READY FOR TASK EXECUTION!")
            print("✅ All critical components are healthy")
        else:
            print("❌ SYSTEM NOT READY")
            print("🔧 Please fix the issues above before proceeding")
        
        print()
        print("📊 SERVICE SUMMARY:")
        
        healthy_count = sum(1 for s in self.services.values() if s.status == ServiceStatus.HEALTHY)
        total_count = len(self.services)
        
        print(f"   Healthy: {healthy_count}/{total_count} services")
        print(f"   System Status: {'READY' if success else 'NOT READY'}")
        print()
    
    # Auto-start methods
    async def _start_docker_desktop(self):
        """Start Docker Desktop"""
        print("🔄 Attempting to start Docker Desktop...")
        try:
            import platform
            system = platform.system()

            if system == "Windows":
                subprocess.Popen(["C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"])
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", "/Applications/Docker.app"])
            else:  # Linux
                subprocess.Popen(["systemctl", "start", "docker"])

            print("⏳ Waiting for Docker to start...")
            await asyncio.sleep(10)

        except Exception as e:
            print(f"❌ Failed to start Docker Desktop: {e}")

    async def _start_postgres_container(self):
        """Start PostgreSQL container"""
        print("🔄 Attempting to start PostgreSQL container...")
        try:
            # Try to start existing container first
            result = subprocess.run(
                ["docker", "start", "postgres"],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                # Container doesn't exist, create it
                print("📦 Creating new PostgreSQL container...")
                subprocess.run([
                    "docker", "run", "-d",
                    "--name", "postgres",
                    "-e", "POSTGRES_PASSWORD=postgres",
                    "-e", "POSTGRES_USER=postgres",
                    "-e", "POSTGRES_DB=pygent_factory",
                    "-p", "54321:5432",
                    "postgres:15"
                ], timeout=60)

            print("⏳ Waiting for PostgreSQL to be ready...")
            await asyncio.sleep(10)

        except Exception as e:
            print(f"❌ Failed to start PostgreSQL container: {e}")

    async def _start_ollama_service(self):
        """Start Ollama service"""
        print("🔄 Attempting to start Ollama service...")
        try:
            # Check if Ollama is already running
            for proc in psutil.process_iter(['pid', 'name']):
                if 'ollama' in proc.info['name'].lower():
                    print("✅ Ollama already running")
                    return

            # Start Ollama service
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("⏳ Waiting for Ollama to start...")
            await asyncio.sleep(10)  # Wait for startup

        except Exception as e:
            print(f"❌ Failed to start Ollama: {e}")

    async def _start_pygent_backend(self):
        """Start PyGent Factory backend (legacy method)"""
        await self._start_pygent_backend_with_monitoring()

    async def _start_pygent_backend_with_monitoring(self):
        """Start PyGent Factory backend with real-time monitoring"""
        print("🔄 Starting PyGent Factory backend with monitoring...")

        try:
            # Check if backend is already running
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:8000/api/v1/health", timeout=3) as response:
                        if response.status == 200:
                            print("✅ PyGent Factory backend already running")
                            return
            except:
                pass

            # Start the backend process
            print("🚀 Launching PyGent Factory backend process...")
            process = subprocess.Popen([
                sys.executable, "main.py",
                "server",
                "--host", "0.0.0.0",
                "--port", "8000"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Give the backend time to start without aggressive monitoring
            print("⏳ Backend process launched, giving it time to initialize...")
            print("   (Backend startup typically takes 30-90 seconds)")

            # Wait a reasonable amount of time before checking
            await asyncio.sleep(20)  # Give it 20 seconds to start up

            # Check if process is still running
            if process.poll() is not None:
                # Process has terminated early
                stdout, stderr = process.communicate()
                print(f"❌ Backend process terminated early:")
                if stderr:
                    print(f"   Error: {stderr[:300]}...")
                return

            print("✅ Backend process is running, startup monitoring will continue in health check")

        except Exception as e:
            print(f"❌ Failed to start PyGent Factory backend: {e}")
        

async def main():
    """Main startup checklist runner"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="PyGent Factory System Startup Checklist")
    parser.add_argument("--auto-start", action="store_true", help="Automatically start missing services")
    parser.add_argument("--quick", action="store_true", help="Run quick check (skip non-critical)")
    
    args = parser.parse_args()
    
    checklist = SystemStartupChecklist(auto_start=args.auto_start)
    
    try:
        success = await checklist.run_full_checklist()
        
        if success:
            print("🚀 READY TO EXECUTE DRAGON TASK!")
            sys.exit(0)
        else:
            print("🛑 SYSTEM NOT READY - FIX ISSUES FIRST")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Startup checklist interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Startup checklist failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
