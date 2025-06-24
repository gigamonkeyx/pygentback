#!/usr/bin/env python3
"""
PyGent Factory Service Startup Script

This script starts all PyGent Factory services in the correct order
and validates they are running properly following Context7 MCP best practices.
"""

import asyncio
import subprocess
import time
import sys
import signal
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class ServiceManager:
    """Manages PyGent Factory service lifecycle"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.service_configs = {
            "backend": {
                "name": "Backend API",
                "command": ["python", "main.py", "server", "--host", "0.0.0.0", "--port", "8000"],
                "cwd": Path(__file__).parent.parent,
                "port": 8000,
                "health_url": "http://localhost:8000/api/v1/health",
                "startup_delay": 5
            },
            "frontend": {
                "name": "Frontend UI",
                "command": ["npm", "run", "dev"],
                "cwd": Path(__file__).parent.parent / "ui",
                "port": 5173,
                "health_url": "http://localhost:5173",
                "startup_delay": 10
            },
            "documentation": {
                "name": "Documentation Server",
                "command": ["python", "-m", "http.server", "3001"],
                "cwd": Path(__file__).parent.parent / "docs-simple",
                "port": 3001,
                "health_url": "http://localhost:3001",
                "startup_delay": 2
            }
        }
        self.startup_order = ["backend", "documentation", "frontend"]
        self.shutdown_handlers_registered = False
    
    def register_shutdown_handlers(self):
        """Register signal handlers for graceful shutdown"""
        if self.shutdown_handlers_registered:
            return
            
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down services...")
            asyncio.create_task(self.stop_all_services())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self.shutdown_handlers_registered = True
    
    async def start_service(self, service_id: str) -> bool:
        """Start a single service"""
        config = self.service_configs[service_id]
        
        print(f"🚀 Starting {config['name']}...")
        
        try:
            # Check if port is already in use
            if self.is_port_in_use(config['port']):
                print(f"⚠️ Port {config['port']} already in use, assuming {config['name']} is running")
                return True
            
            # Start the process
            process = subprocess.Popen(
                config['command'],
                cwd=config['cwd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[service_id] = process
            
            # Wait for startup
            print(f"⏳ Waiting {config['startup_delay']}s for {config['name']} to start...")
            await asyncio.sleep(config['startup_delay'])
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ {config['name']} failed to start:")
                print(f"   Exit code: {process.returncode}")
                if stdout:
                    print(f"   STDOUT: {stdout[:500]}")
                if stderr:
                    print(f"   STDERR: {stderr[:500]}")
                return False
            
            print(f"✅ {config['name']} started successfully (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {config['name']}: {e}")
            return False
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return False
            except OSError:
                return True
    
    async def stop_service(self, service_id: str) -> bool:
        """Stop a single service"""
        if service_id not in self.processes:
            return True
        
        config = self.service_configs[service_id]
        process = self.processes[service_id]
        
        print(f"🛑 Stopping {config['name']}...")
        
        try:
            # Try graceful shutdown first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                print(f"✅ {config['name']} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                print(f"⚠️ Force killing {config['name']}...")
                process.kill()
                process.wait()
                print(f"✅ {config['name']} force stopped")
            
            del self.processes[service_id]
            return True
            
        except Exception as e:
            print(f"❌ Error stopping {config['name']}: {e}")
            return False
    
    async def start_all_services(self) -> bool:
        """Start all services in the correct order"""
        print("🚀 Starting PyGent Factory services...\n")
        
        self.register_shutdown_handlers()
        
        success_count = 0
        for service_id in self.startup_order:
            success = await self.start_service(service_id)
            if success:
                success_count += 1
            else:
                print(f"❌ Failed to start {service_id}, stopping startup process")
                break
        
        if success_count == len(self.startup_order):
            print(f"\n🎉 All {success_count} services started successfully!")
            
            # Print service status
            print("\n📊 Service Status:")
            for service_id in self.startup_order:
                config = self.service_configs[service_id]
                process = self.processes.get(service_id)
                if process and process.poll() is None:
                    print(f"   ✅ {config['name']}: Running (PID: {process.pid}, Port: {config['port']})")
                else:
                    print(f"   ❌ {config['name']}: Not running")
            
            return True
        else:
            print(f"\n💥 Only {success_count}/{len(self.startup_order)} services started successfully")
            await self.stop_all_services()
            return False
    
    async def stop_all_services(self) -> None:
        """Stop all services in reverse order"""
        print("\n🛑 Stopping all services...")
        
        # Stop in reverse order
        for service_id in reversed(self.startup_order):
            if service_id in self.processes:
                await self.stop_service(service_id)
        
        print("✅ All services stopped")
    
    async def validate_services(self) -> bool:
        """Validate that all services are running correctly"""
        print("\n🔍 Validating services...")
        
        try:
            # Import and run validation
            from validate_services import ServiceValidator
            
            validator = ServiceValidator()
            results = await validator.run_validation()
            
            return results["overall"]["status"] == "healthy"
            
        except Exception as e:
            print(f"❌ Service validation failed: {e}")
            return False
    
    async def run_services(self) -> None:
        """Start services and keep them running"""
        try:
            # Start all services
            if not await self.start_all_services():
                sys.exit(1)
            
            # Wait a bit for services to fully initialize
            print("\n⏳ Waiting for services to fully initialize...")
            await asyncio.sleep(5)
            
            # Validate services
            if await self.validate_services():
                print("\n🎉 All services validated successfully!")
                print("\n🔗 Service URLs:")
                print("   Backend API: http://localhost:8000")
                print("   Frontend UI: http://localhost:5173")
                print("   Documentation: http://localhost:3001")
                print("   API Health: http://localhost:8000/api/v1/health")
                print("   WebSocket: ws://localhost:8000/ws")
                
                print("\n⌨️ Press Ctrl+C to stop all services")
                
                # Keep services running
                while True:
                    await asyncio.sleep(1)
                    
                    # Check if any process has died
                    for service_id, process in list(self.processes.items()):
                        if process.poll() is not None:
                            config = self.service_configs[service_id]
                            print(f"❌ {config['name']} has stopped unexpectedly!")
                            await self.stop_all_services()
                            sys.exit(1)
            else:
                print("\n💥 Service validation failed!")
                await self.stop_all_services()
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
            await self.stop_all_services()
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            await self.stop_all_services()
            sys.exit(1)


async def main():
    """Main function"""
    manager = ServiceManager()
    await manager.run_services()


if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print("🏭 PyGent Factory Service Manager")
    print(f"📁 Working directory: {project_root}")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
