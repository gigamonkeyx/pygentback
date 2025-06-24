"""
Sync Manager

Manages synchronization of built documentation files from the backend
to the frontend serving location with intelligent optimization.
"""

import asyncio
import logging
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .documentation_models import (
    DocumentationTask, DocumentationConfig, SyncResult,
    DocumentationEventType
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

class SyncManager:
    """
    Manages synchronization of documentation files between backend and frontend.
    
    Provides intelligent file copying, manifest generation, and cache management
    for optimal documentation serving performance.
    """
    
    def __init__(self, config: DocumentationConfig, event_bus):
        self.config = config
        self.event_bus = event_bus
        if self.event_bus is None:
            raise RuntimeError("EventBus is required for SyncManager")
        self.is_running = False
        self.sync_metrics = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "total_files_synced": 0,
            "average_sync_time": 0.0
        }
        self.last_sync_manifest = {}
    
    async def start(self):
        """Start the sync manager"""
        self.is_running = True
        logger.info("Sync Manager started")
    
    async def stop(self):
        """Stop the sync manager"""
        self.is_running = False
        logger.info("Sync Manager stopped")
    
    async def sync_to_frontend(self, task: DocumentationTask) -> SyncResult:
        """Sync built documentation to frontend serving location"""
        start_time = time.time()
        sync_id = f"sync_{int(start_time)}"
        
        try:
            # Emit sync started event
            await self.event_bus.publish_event(
                event_type="sync_started",
                source="sync_manager",
                data={"sync_id": sync_id, "task_id": task.task_id}
            )
            
            # Validate source directory
            if not self.config.docs_build_path.exists():
                raise FileNotFoundError(f"Build directory not found: {self.config.docs_build_path}")
            
            # Prepare target directory
            await self._prepare_target_directory()
            
            # Determine sync strategy
            is_incremental = task.parameters.get("incremental", False)
            changed_file = task.parameters.get("changed_file")
            
            if is_incremental and changed_file:
                files_copied = await self._incremental_sync(changed_file)
            else:
                files_copied = await self._full_sync()
            
            # Generate route manifest
            manifest_created = await self._generate_route_manifest()
            
            # Update cache headers
            await self._update_cache_headers()
            
            # Calculate sync metrics
            sync_time = time.time() - start_time
            
            sync_result = SyncResult(
                success=True,
                files_copied=files_copied,
                sync_time=sync_time,
                target_path=self.config.frontend_docs_path,
                manifest_created=manifest_created
            )
            
            # Update metrics
            self._update_sync_metrics(sync_result)
            
            # Emit completion event
            await self.event_bus.publish_event(
                event_type="sync_completed",
                source="sync_manager",
                data={
                    "sync_id": sync_id,
                    "files_copied": files_copied,
                    "sync_time": sync_time,
                    "success": True
                }
            )
            
            logger.info(f"Sync completed: {files_copied} files in {sync_time:.2f}s")
            return sync_result
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            sync_result = SyncResult(
                success=False,
                files_copied=0,
                sync_time=time.time() - start_time,
                target_path=self.config.frontend_docs_path,
                errors=[str(e)]
            )
            self._update_sync_metrics(sync_result)
            raise
    
    async def _prepare_target_directory(self):
        """Prepare the target directory for sync"""
        # Ensure parent directory exists
        self.config.frontend_docs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean target directory if it exists
        if self.config.frontend_docs_path.exists():
            shutil.rmtree(self.config.frontend_docs_path)
        
        # Create fresh target directory
        self.config.frontend_docs_path.mkdir(parents=True, exist_ok=True)
    
    async def _full_sync(self) -> int:
        """Perform full synchronization of all files"""
        files_copied = 0
        
        # Copy all files from build directory to target
        for source_file in self.config.docs_build_path.rglob("*"):
            if source_file.is_file():
                # Calculate relative path
                relative_path = source_file.relative_to(self.config.docs_build_path)
                target_file = self.config.frontend_docs_path / relative_path
                
                # Ensure target directory exists
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_file, target_file)
                files_copied += 1
        
        return files_copied
    
    async def _incremental_sync(self, changed_file: str) -> int:
        """Perform incremental sync for changed files"""
        files_copied = 0
        changed_path = Path(changed_file)
        
        # If the changed file is in the docs source, rebuild and sync specific files
        if self.config.docs_source_path in changed_path.parents:
            # For now, perform full sync for incremental updates
            # In a more sophisticated implementation, you would:
            # 1. Determine which built files correspond to the changed source
            # 2. Only copy those specific files
            # 3. Update the manifest accordingly
            files_copied = await self._full_sync()
        
        return files_copied
    
    async def _generate_route_manifest(self) -> bool:
        """Generate route manifest for frontend integration"""
        try:
            manifest = {
                "timestamp": datetime.utcnow().isoformat(),
                "baseUrl": "/docs/",
                "routes": [],
                "assets": [],
                "metadata": {
                    "generator": "PyGent Factory Documentation Orchestrator",
                    "version": "1.0.0"
                }
            }
            
            # Discover HTML files for routes
            for html_file in self.config.frontend_docs_path.rglob("*.html"):
                relative_path = html_file.relative_to(self.config.frontend_docs_path)
                route_path = "/" + str(relative_path).replace("\\", "/")
                
                # Convert index.html to directory routes
                if relative_path.name == "index.html":
                    if relative_path.parent == Path("."):
                        route_path = "/"
                    else:
                        route_path = "/" + str(relative_path.parent).replace("\\", "/") + "/"
                
                manifest["routes"].append(route_path)
            
            # Discover asset files
            assets_dir = self.config.frontend_docs_path / "assets"
            if assets_dir.exists():
                for asset_file in assets_dir.rglob("*"):
                    if asset_file.is_file():
                        relative_path = asset_file.relative_to(self.config.frontend_docs_path)
                        asset_path = "/" + str(relative_path).replace("\\", "/")
                        manifest["assets"].append(asset_path)
            
            # Write manifest file
            manifest_path = self.config.frontend_docs_path / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.last_sync_manifest = manifest
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate route manifest: {e}")
            return False
    
    async def _update_cache_headers(self):
        """Update cache headers for optimal performance"""
        try:
            # Create .htaccess file for Apache servers
            htaccess_content = """
# Documentation cache headers
<IfModule mod_expires.c>
    ExpiresActive On
    
    # HTML files - short cache for content updates
    ExpiresByType text/html "access plus 1 hour"
    
    # CSS and JS files - longer cache with versioning
    ExpiresByType text/css "access plus 1 month"
    ExpiresByType application/javascript "access plus 1 month"
    
    # Images - long cache
    ExpiresByType image/png "access plus 6 months"
    ExpiresByType image/jpg "access plus 6 months"
    ExpiresByType image/jpeg "access plus 6 months"
    ExpiresByType image/gif "access plus 6 months"
    ExpiresByType image/svg+xml "access plus 6 months"
</IfModule>

# Gzip compression
<IfModule mod_deflate.c>
    AddOutputFilterByType DEFLATE text/plain
    AddOutputFilterByType DEFLATE text/html
    AddOutputFilterByType DEFLATE text/xml
    AddOutputFilterByType DEFLATE text/css
    AddOutputFilterByType DEFLATE application/xml
    AddOutputFilterByType DEFLATE application/xhtml+xml
    AddOutputFilterByType DEFLATE application/rss+xml
    AddOutputFilterByType DEFLATE application/javascript
    AddOutputFilterByType DEFLATE application/x-javascript
</IfModule>
"""
            
            htaccess_path = self.config.frontend_docs_path / ".htaccess"
            with open(htaccess_path, 'w') as f:
                f.write(htaccess_content.strip())
            
        except Exception as e:
            logger.warning(f"Could not create cache headers: {e}")
    
    def _update_sync_metrics(self, sync_result: SyncResult):
        """Update sync metrics"""
        self.sync_metrics["total_syncs"] += 1
        
        if sync_result.success:
            self.sync_metrics["successful_syncs"] += 1
            self.sync_metrics["total_files_synced"] += sync_result.files_copied
        else:
            self.sync_metrics["failed_syncs"] += 1
        
        # Update average sync time
        total_syncs = self.sync_metrics["total_syncs"]
        current_avg = self.sync_metrics["average_sync_time"]
        new_avg = ((current_avg * (total_syncs - 1)) + sync_result.sync_time) / total_syncs
        self.sync_metrics["average_sync_time"] = new_avg
    
    async def verify_sync(self) -> Dict[str, Any]:
        """Verify that sync was successful"""
        verification_result = {
            "target_exists": self.config.frontend_docs_path.exists(),
            "manifest_exists": False,
            "index_exists": False,
            "assets_exist": False,
            "file_count": 0,
            "total_size": 0
        }
        
        if verification_result["target_exists"]:
            # Check for manifest
            manifest_path = self.config.frontend_docs_path / "manifest.json"
            verification_result["manifest_exists"] = manifest_path.exists()
            
            # Check for index
            index_path = self.config.frontend_docs_path / "index.html"
            verification_result["index_exists"] = index_path.exists()
            
            # Check for assets
            assets_path = self.config.frontend_docs_path / "assets"
            verification_result["assets_exist"] = assets_path.exists()
            
            # Count files and calculate size
            for file_path in self.config.frontend_docs_path.rglob("*"):
                if file_path.is_file():
                    verification_result["file_count"] += 1
                    verification_result["total_size"] += file_path.stat().st_size
        
        return verification_result
    
    async def get_status(self) -> Dict[str, Any]:
        """Get sync manager status"""
        return {
            "running": self.is_running,
            "metrics": self.sync_metrics.copy(),
            "last_sync_manifest": self.last_sync_manifest.copy(),
            "target_path": str(self.config.frontend_docs_path)
        }
    
    async def cleanup_old_syncs(self, keep_count: int = 5):
        """Clean up old sync artifacts"""
        # In a real implementation, you would:
        # 1. Keep track of sync history
        # 2. Clean up old backup files
        # 3. Remove outdated cache files
        # 4. Optimize storage usage
        pass
    
    async def get_sync_history(self) -> List[Dict[str, Any]]:
        """Get history of sync operations"""
        # In a real implementation, you would store and return sync history
        return []
