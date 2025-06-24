"""
MCP Server Installer

Handles installation, management, and lifecycle of MCP servers for testing.
Provides automated installation from various sources and dependency management.
"""

import os
import shutil
import subprocess
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class InstallationConfig:
    """Configuration for MCP server installation"""
    server_id: str
    name: str
    source_type: str  # git, npm, pip, local
    source_url: str
    version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    install_args: List[str] = field(default_factory=list)
    post_install_commands: List[str] = field(default_factory=list)


@dataclass
class InstallationResult:
    """Result of MCP server installation"""
    server_id: str
    success: bool
    install_path: Optional[str] = None
    version: Optional[str] = None
    error_message: Optional[str] = None
    installation_time: float = 0.0
    installed_at: datetime = field(default_factory=datetime.utcnow)
    dependencies_installed: List[str] = field(default_factory=list)


class MCPServerInstaller:
    """
    MCP Server Installation and Management System.
    
    Handles automated installation of MCP servers from various sources
    including Git repositories, npm packages, pip packages, and local files.
    """
    
    def __init__(self, installation_directory: str):
        self.installation_dir = Path(installation_directory)
        self.installation_dir.mkdir(parents=True, exist_ok=True)
        
        # Installation tracking
        self.installed_servers: Dict[str, InstallationResult] = {}
        self.installation_lock = asyncio.Lock()
        
        # Installation metadata
        self.metadata_file = self.installation_dir / "installations.json"
        self._load_installation_metadata()
    
    def _load_installation_metadata(self):
        """Load installation metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for server_id, result_data in data.items():
                        result = InstallationResult(**result_data)
                        result.installed_at = datetime.fromisoformat(result_data['installed_at'])
                        self.installed_servers[server_id] = result
        except Exception as e:
            logger.warning(f"Failed to load installation metadata: {e}")
    
    def _save_installation_metadata(self):
        """Save installation metadata to disk"""
        try:
            data = {}
            for server_id, result in self.installed_servers.items():
                data[server_id] = {
                    'server_id': result.server_id,
                    'success': result.success,
                    'install_path': result.install_path,
                    'version': result.version,
                    'error_message': result.error_message,
                    'installation_time': result.installation_time,
                    'installed_at': result.installed_at.isoformat(),
                    'dependencies_installed': result.dependencies_installed
                }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save installation metadata: {e}")
    
    async def install_server(self, config: InstallationConfig) -> InstallationResult:
        """
        Install an MCP server based on configuration.
        
        Args:
            config: Installation configuration
            
        Returns:
            InstallationResult with installation details
        """
        async with self.installation_lock:
            start_time = datetime.utcnow()
            
            try:
                logger.info(f"Installing MCP server {config.server_id} from {config.source_url}")
                
                # Check if already installed
                if config.server_id in self.installed_servers:
                    existing = self.installed_servers[config.server_id]
                    if existing.success:
                        logger.info(f"Server {config.server_id} already installed")
                        return existing
                
                # Create server-specific directory
                server_dir = self.installation_dir / config.server_id
                server_dir.mkdir(exist_ok=True)
                
                # Install based on source type
                if config.source_type == "git":
                    result = await self._install_from_git(config, server_dir)
                elif config.source_type == "npm":
                    result = await self._install_from_npm(config, server_dir)
                elif config.source_type == "pip":
                    result = await self._install_from_pip(config, server_dir)
                elif config.source_type == "local":
                    result = await self._install_from_local(config, server_dir)
                else:
                    raise ValueError(f"Unsupported source type: {config.source_type}")
                
                # Calculate installation time
                result.installation_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Store result
                self.installed_servers[config.server_id] = result
                self._save_installation_metadata()
                
                if result.success:
                    logger.info(f"Successfully installed {config.server_id} in {result.installation_time:.2f}s")
                else:
                    logger.error(f"Failed to install {config.server_id}: {result.error_message}")
                
                return result
                
            except Exception as e:
                error_result = InstallationResult(
                    server_id=config.server_id,
                    success=False,
                    error_message=str(e),
                    installation_time=(datetime.utcnow() - start_time).total_seconds()
                )
                
                self.installed_servers[config.server_id] = error_result
                self._save_installation_metadata()
                
                logger.error(f"Installation failed for {config.server_id}: {e}")
                return error_result
    
    async def _install_from_git(self, config: InstallationConfig, server_dir: Path) -> InstallationResult:
        """Install MCP server from Git repository"""
        try:
            # Clone repository
            clone_cmd = ["git", "clone", config.source_url, str(server_dir)]
            if config.version:
                clone_cmd.extend(["--branch", config.version])
            
            process = await asyncio.create_subprocess_exec(
                *clone_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Git clone failed: {stderr.decode()}")
            
            # Install dependencies if package.json exists
            package_json = server_dir / "package.json"
            if package_json.exists():
                npm_cmd = ["npm", "install"]
                process = await asyncio.create_subprocess_exec(
                    *npm_cmd,
                    cwd=server_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
            
            # Install Python dependencies if requirements.txt exists
            requirements_txt = server_dir / "requirements.txt"
            if requirements_txt.exists():
                pip_cmd = ["pip", "install", "-r", "requirements.txt"]
                process = await asyncio.create_subprocess_exec(
                    *pip_cmd,
                    cwd=server_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
            
            return InstallationResult(
                server_id=config.server_id,
                success=True,
                install_path=str(server_dir),
                version=config.version or "latest"
            )
            
        except Exception as e:
            return InstallationResult(
                server_id=config.server_id,
                success=False,
                error_message=str(e)
            )
    
    async def _install_from_npm(self, config: InstallationConfig, server_dir: Path) -> InstallationResult:
        """Install MCP server from npm package"""
        try:
            # Install npm package
            npm_cmd = ["npm", "install", config.source_url]
            if config.version:
                npm_cmd[-1] += f"@{config.version}"
            
            process = await asyncio.create_subprocess_exec(
                *npm_cmd,
                cwd=server_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"npm install failed: {stderr.decode()}")
            
            return InstallationResult(
                server_id=config.server_id,
                success=True,
                install_path=str(server_dir),
                version=config.version or "latest"
            )
            
        except Exception as e:
            return InstallationResult(
                server_id=config.server_id,
                success=False,
                error_message=str(e)
            )
    
    async def _install_from_pip(self, config: InstallationConfig, server_dir: Path) -> InstallationResult:
        """Install MCP server from pip package"""
        try:
            # Install pip package
            pip_cmd = ["pip", "install", config.source_url]
            if config.version:
                pip_cmd[-1] += f"=={config.version}"
            
            # Install to specific directory
            pip_cmd.extend(["--target", str(server_dir)])
            
            process = await asyncio.create_subprocess_exec(
                *pip_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"pip install failed: {stderr.decode()}")
            
            return InstallationResult(
                server_id=config.server_id,
                success=True,
                install_path=str(server_dir),
                version=config.version or "latest"
            )
            
        except Exception as e:
            return InstallationResult(
                server_id=config.server_id,
                success=False,
                error_message=str(e)
            )
    
    async def _install_from_local(self, config: InstallationConfig, server_dir: Path) -> InstallationResult:
        """Install MCP server from local path"""
        try:
            source_path = Path(config.source_url)
            if not source_path.exists():
                raise FileNotFoundError(f"Source path does not exist: {config.source_url}")
            
            # Copy files to server directory
            if source_path.is_file():
                shutil.copy2(source_path, server_dir)
            else:
                shutil.copytree(source_path, server_dir, dirs_exist_ok=True)
            
            return InstallationResult(
                server_id=config.server_id,
                success=True,
                install_path=str(server_dir),
                version="local"
            )
            
        except Exception as e:
            return InstallationResult(
                server_id=config.server_id,
                success=False,
                error_message=str(e)
            )
    
    async def uninstall_server(self, server_id: str) -> bool:
        """Uninstall an MCP server"""
        try:
            if server_id not in self.installed_servers:
                logger.warning(f"Server {server_id} not found in installations")
                return False
            
            result = self.installed_servers[server_id]
            if result.install_path and Path(result.install_path).exists():
                shutil.rmtree(result.install_path)
            
            del self.installed_servers[server_id]
            self._save_installation_metadata()
            
            logger.info(f"Successfully uninstalled server {server_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to uninstall server {server_id}: {e}")
            return False
    
    def is_installed(self, server_id: str) -> bool:
        """Check if a server is installed"""
        return (server_id in self.installed_servers and 
                self.installed_servers[server_id].success)
    
    def get_installation_info(self, server_id: str) -> Optional[InstallationResult]:
        """Get installation information for a server"""
        return self.installed_servers.get(server_id)
    
    def list_installed_servers(self) -> List[str]:
        """List all installed server IDs"""
        return [
            server_id for server_id, result in self.installed_servers.items()
            if result.success
        ]
