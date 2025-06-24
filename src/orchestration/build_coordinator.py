"""
Build Coordinator

Manages VitePress build processes with intelligent conflict resolution
and performance optimization.
"""

import asyncio
import logging
import subprocess
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .documentation_models import (
    DocumentationTask, DocumentationConfig, BuildResult,
    DocumentationTaskStatus, DocumentationEventType
)

# Handle integration events gracefully
try:
    from ..integration.events import EventBus, Event
    INTEGRATION_EVENTS_AVAILABLE = True
except ImportError:
    EventBus = None
    Event = None
    INTEGRATION_EVENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BuildCoordinator:
    """
    Coordinates VitePress build processes with intelligent optimization
    and conflict resolution.
    """
    
    def __init__(self, config: DocumentationConfig, event_bus):
        self.config = config
        self.event_bus = event_bus
        if self.event_bus is None:
            raise RuntimeError("EventBus is required for BuildCoordinator")
        self.is_running = False
        self.active_builds: Dict[str, subprocess.Popen] = {}
        self.dev_server_process: Optional[subprocess.Popen] = None
        self.build_metrics = {
            "total_builds": 0,
            "successful_builds": 0,
            "failed_builds": 0,
            "average_build_time": 0.0
        }
    
    async def start(self):
        """Start the build coordinator"""
        self.is_running = True
        logger.info("Build Coordinator started")
    
    async def stop(self):
        """Stop the build coordinator"""
        # Terminate active builds
        for build_id, process in self.active_builds.items():
            if process.poll() is None:
                process.terminate()
                logger.info(f"Terminated build: {build_id}")
        
        # Stop dev server
        if self.dev_server_process and self.dev_server_process.poll() is None:
            self.dev_server_process.terminate()
            logger.info("Terminated development server")
        
        self.is_running = False
        logger.info("Build Coordinator stopped")
    
    async def build_vitepress(self, task: DocumentationTask) -> BuildResult:
        """Build VitePress documentation"""
        build_id = f"build_{int(time.time())}"
        start_time = time.time()
        
        try:
            # Emit build started event
            if INTEGRATION_EVENTS_AVAILABLE and Event:
                await self.event_bus.publish_event(
                    event_type="build_started",
                    source="build_coordinator",
                    data={"build_id": build_id, "task_id": task.task_id}
                )
            else:
                await self.event_bus.publish_event(
                    event_type="build_started",
                    source="build_coordinator",
                    data={"build_id": build_id, "task_id": task.task_id}
                )
            
            # Prepare build environment
            await self._prepare_build_environment()
            
            # Determine build command
            is_production = task.parameters.get("production", False)
            build_cmd = self._get_build_command(is_production)
            
            # Execute build
            process = await self._execute_build_command(build_cmd, build_id)
            
            # Wait for completion
            stdout, stderr = await self._wait_for_build_completion(process, build_id)
            
            # Analyze build results
            build_result = await self._analyze_build_results(
                process.returncode, stdout, stderr, start_time
            )
            
            # Update metrics
            self._update_build_metrics(build_result)
            
            # Emit completion event
            await self.event_bus.publish_event(
                event_type="build_completed",
                source="build_coordinator",
                data={
                    "build_id": build_id,
                    "success": build_result.success,
                    "build_time": build_result.build_time,
                    "files_generated": build_result.files_generated
                }
            )
            
            return build_result
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            build_result = BuildResult(
                success=False,
                build_time=time.time() - start_time,
                output_size=0,
                files_generated=0,
                errors=[str(e)]
            )
            self._update_build_metrics(build_result)
            raise
        finally:
            # Cleanup
            if build_id in self.active_builds:
                del self.active_builds[build_id]
    
    async def start_dev_server(self, task: DocumentationTask) -> Dict[str, Any]:
        """Start VitePress development server"""
        try:
            if self.dev_server_process and self.dev_server_process.poll() is None:
                logger.info("Development server already running")
                return {"server_running": True, "port": self.config.vitepress_port}
            
            # Prepare environment
            await self._prepare_build_environment()
            
            # Start development server
            cmd = ["npx", "vitepress", "dev", "--port", str(self.config.vitepress_port)]
            
            self.dev_server_process = subprocess.Popen(
                cmd,
                cwd=self.config.docs_source_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            await self._wait_for_dev_server_startup()
            
            logger.info(f"Development server started on port {self.config.vitepress_port}")
            
            return {
                "server_running": True,
                "port": self.config.vitepress_port,
                "pid": self.dev_server_process.pid,
                "url": f"http://localhost:{self.config.vitepress_port}"
            }
            
        except Exception as e:
            logger.error(f"Failed to start development server: {e}")
            raise
    
    async def _prepare_build_environment(self):
        """Prepare the build environment"""
        # Ensure build directory exists
        self.config.docs_build_path.mkdir(parents=True, exist_ok=True)

        # Check Node.js and npm availability
        try:
            subprocess.run(["node", "--version"], check=True, capture_output=True)
            logger.info("Node.js is available")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Node.js not available: {e}")
            raise RuntimeError("Node.js not available - required for VitePress builds")

        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
            logger.info("npm is available")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"npm not available: {e}")
            # Try alternative package managers
            try:
                subprocess.run(["yarn", "--version"], check=True, capture_output=True)
                logger.info("yarn is available as alternative")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error("No package manager (npm/yarn) available")
                raise RuntimeError("No package manager available - npm or yarn required for VitePress builds")

        # Verify VitePress installation
        try:
            subprocess.run(
                ["npx", "vitepress", "--version"],
                check=True,
                capture_output=True,
                cwd=self.config.docs_source_path
            )
            logger.info("VitePress is available")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"VitePress not available via npx: {e}")
            # Check if VitePress is installed locally
            vitepress_path = self.config.docs_source_path / "node_modules" / ".bin" / "vitepress"
            if vitepress_path.exists():
                logger.info("VitePress found in local node_modules")
            else:
                raise RuntimeError("VitePress not available - run 'npm install' in docs directory")
    
    def _get_build_command(self, is_production: bool = False) -> List[str]:
        """Get the appropriate build command"""
        # Try local VitePress installation first
        local_vitepress = self.config.docs_source_path / "node_modules" / ".bin" / "vitepress.cmd"
        if local_vitepress.exists():
            if is_production:
                return [str(local_vitepress), "build", "--minify"]
            else:
                return [str(local_vitepress), "build"]

        # Fallback to npx
        if is_production:
            return ["npx", "vitepress", "build", "--minify"]
        else:
            return ["npx", "vitepress", "build"]
    
    async def _execute_build_command(self, cmd: List[str], build_id: str) -> subprocess.Popen:
        """Execute the build command"""
        logger.info(f"Executing build command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            cwd=self.config.docs_source_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.active_builds[build_id] = process
        return process
    
    async def _wait_for_build_completion(self, process: subprocess.Popen, build_id: str) -> tuple:
        """Wait for build completion with timeout"""
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.create_task(self._communicate_with_process(process)),
                timeout=self.config.build_timeout
            )
            return stdout, stderr
        except asyncio.TimeoutError:
            process.terminate()
            raise RuntimeError(f"Build timeout after {self.config.build_timeout} seconds")
    
    async def _communicate_with_process(self, process: subprocess.Popen) -> tuple:
        """Communicate with process asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process.communicate)
    
    async def _analyze_build_results(self, return_code: int, stdout: str, stderr: str, start_time: float) -> BuildResult:
        """Analyze build results and generate report"""
        build_time = time.time() - start_time
        success = return_code == 0
        
        # Count generated files
        files_generated = 0
        output_size = 0
        
        if self.config.docs_build_path.exists():
            for file_path in self.config.docs_build_path.rglob("*"):
                if file_path.is_file():
                    files_generated += 1
                    output_size += file_path.stat().st_size
        
        # Parse warnings and errors from output
        warnings = []
        errors = []
        
        for line in stderr.split('\n'):
            if 'warning' in line.lower():
                warnings.append(line.strip())
            elif 'error' in line.lower():
                errors.append(line.strip())
        
        return BuildResult(
            success=success,
            build_time=build_time,
            output_size=output_size,
            files_generated=files_generated,
            warnings=warnings,
            errors=errors,
            build_log=f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )
    
    def _update_build_metrics(self, build_result: BuildResult):
        """Update build metrics"""
        self.build_metrics["total_builds"] += 1
        
        if build_result.success:
            self.build_metrics["successful_builds"] += 1
        else:
            self.build_metrics["failed_builds"] += 1
        
        # Update average build time
        total_builds = self.build_metrics["total_builds"]
        current_avg = self.build_metrics["average_build_time"]
        new_avg = ((current_avg * (total_builds - 1)) + build_result.build_time) / total_builds
        self.build_metrics["average_build_time"] = new_avg
    
    async def _wait_for_dev_server_startup(self, max_wait: int = 30):
        """Wait for development server to start"""
        import aiohttp
        
        for _ in range(max_wait):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{self.config.vitepress_port}") as response:
                        if response.status == 200:
                            return
            except:
                pass
            
            await asyncio.sleep(1)
        
        raise RuntimeError("Development server failed to start within timeout")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get build coordinator status"""
        return {
            "running": self.is_running,
            "active_builds": len(self.active_builds),
            "dev_server_running": self.dev_server_process is not None and self.dev_server_process.poll() is None,
            "metrics": self.build_metrics.copy()
        }
    
    async def stop_dev_server(self):
        """Stop the development server"""
        if self.dev_server_process and self.dev_server_process.poll() is None:
            self.dev_server_process.terminate()
            await asyncio.sleep(2)
            
            if self.dev_server_process.poll() is None:
                self.dev_server_process.kill()
            
            self.dev_server_process = None
            logger.info("Development server stopped")
    
    async def get_build_logs(self, build_id: str) -> Optional[str]:
        """Get logs for a specific build"""
        # In a real implementation, you would store and retrieve build logs
        # For now, return a placeholder
        return f"Build logs for {build_id} would be stored and retrieved here"
