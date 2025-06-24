"""
Conflict Resolver

Intelligent resolution of configuration conflicts, particularly
Tailwind CSS and PostCSS conflicts that affect VitePress builds.
"""

import asyncio
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .documentation_models import (
    DocumentationTask, DocumentationConfig, ConflictResolutionResult,
    ConflictType, ConfigurationConflict, ResolutionAction
)
# Handle integration events gracefully
try:
    from ..integration.events import EventBus, Event, EventType
    INTEGRATION_EVENTS_AVAILABLE = True
except ImportError:
    EventBus = None
    Event = None
    EventType = None
    INTEGRATION_EVENTS_AVAILABLE = False


logger = logging.getLogger(__name__)

class ConflictResolver:
    """
    Intelligent conflict resolution system for documentation builds.
    
    Automatically detects and resolves configuration conflicts that
    prevent successful VitePress builds.
    """
    
    def __init__(self, config: DocumentationConfig, event_bus):
        self.config = config
        self.event_bus = event_bus
        if self.event_bus is None:
            raise RuntimeError("EventBus is required for ConflictResolver")
        self.is_running = False
        self.detected_conflicts: List[ConfigurationConflict] = []
        self.resolution_history: List[ResolutionAction] = []
        
        # Conflict detection patterns
        self.conflict_patterns = {
            ConflictType.TAILWIND_CSS: [
                "tailwindcss",
                "@tailwindcss/postcss",
                "tailwind.config.js",
                "tailwind.config.ts"
            ],
            ConflictType.POSTCSS: [
                "postcss.config.js",
                "postcss.config.ts", 
                "postcss",
                "autoprefixer"
            ],
            ConflictType.VITE_CONFIG: [
                "vite.config.js",
                "vite.config.ts"
            ]
        }
    
    async def start(self):
        """Start the conflict resolver"""
        self.is_running = True
        logger.info("Conflict Resolver started")
    
    async def stop(self):
        """Stop the conflict resolver"""
        self.is_running = False
        logger.info("Conflict Resolver stopped")
    
    async def resolve_conflicts(self, task: DocumentationTask) -> ConflictResolutionResult:
        """Main conflict resolution method"""
        try:
            # Detect conflicts
            conflicts = await self.detect_conflicts()
            
            if not conflicts:
                return ConflictResolutionResult(
                    conflicts_found=[],
                    conflicts_resolved=[],
                    resolution_actions=[],
                    success=True
                )
            
            # Create backup if configured
            backup_path = None
            if self.config.backup_on_conflict:
                backup_path = await self._create_backup()
            
            # Resolve each conflict
            resolved_conflicts = []
            resolution_actions = []
            
            for conflict in conflicts:
                if conflict.auto_resolvable and self.config.auto_resolve_conflicts:
                    actions = await self._resolve_conflict(conflict)
                    if actions:
                        resolved_conflicts.append(conflict.description)
                        resolution_actions.extend([action.description for action in actions])
            
            # Verify resolution
            remaining_conflicts = await self.detect_conflicts()
            success = len(remaining_conflicts) == 0
            
            result = ConflictResolutionResult(
                conflicts_found=[c.description for c in conflicts],
                conflicts_resolved=resolved_conflicts,
                resolution_actions=resolution_actions,
                backup_created=backup_path is not None,
                backup_path=backup_path,
                success=success
            )
            
            # Emit resolution event
            await self.event_bus.publish_event(
                event_type="conflict_resolution_completed",
                source="conflict_resolver",
                data={
                    "component": "conflict_resolver",
                    "conflicts_resolved": len(resolved_conflicts),
                    "success": success
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            return ConflictResolutionResult(
                conflicts_found=[],
                conflicts_resolved=[],
                resolution_actions=[],
                success=False,
                error_message=str(e)
            )
    
    async def detect_conflicts(self) -> List[ConfigurationConflict]:
        """Detect configuration conflicts"""
        conflicts = []
        
        # Check for Tailwind CSS conflicts
        tailwind_conflicts = await self._detect_tailwind_conflicts()
        conflicts.extend(tailwind_conflicts)
        
        # Check for PostCSS conflicts
        postcss_conflicts = await self._detect_postcss_conflicts()
        conflicts.extend(postcss_conflicts)
        
        # Check for Vite config conflicts
        vite_conflicts = await self._detect_vite_conflicts()
        conflicts.extend(vite_conflicts)
        
        # Check for dependency conflicts
        dependency_conflicts = await self._detect_dependency_conflicts()
        conflicts.extend(dependency_conflicts)
        
        self.detected_conflicts = conflicts
        return conflicts
    
    async def _detect_tailwind_conflicts(self) -> List[ConfigurationConflict]:
        """Detect Tailwind CSS related conflicts"""
        conflicts = []
        
        # Check for Tailwind CSS in docs dependencies
        package_json_path = self.config.docs_source_path / "package.json"
        if package_json_path.exists():
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
                
                dependencies = package_data.get("dependencies", {})
                dev_dependencies = package_data.get("devDependencies", {})
                
                if "tailwindcss" in dependencies or "tailwindcss" in dev_dependencies:
                    conflicts.append(ConfigurationConflict(
                        conflict_id="tailwind_in_docs",
                        conflict_type=ConflictType.TAILWIND_CSS,
                        description="Tailwind CSS found in documentation dependencies",
                        affected_files=[package_json_path],
                        severity="high",
                        auto_resolvable=True,
                        resolution_strategy="remove_tailwind_dependencies"
                    ))
                
                if "@tailwindcss/postcss" in dependencies or "@tailwindcss/postcss" in dev_dependencies:
                    conflicts.append(ConfigurationConflict(
                        conflict_id="tailwind_postcss_in_docs",
                        conflict_type=ConflictType.TAILWIND_CSS,
                        description="Tailwind PostCSS plugin found in documentation dependencies",
                        affected_files=[package_json_path],
                        severity="high",
                        auto_resolvable=True,
                        resolution_strategy="remove_tailwind_postcss"
                    ))
        
        # Check for Tailwind config files
        tailwind_configs = [
            self.config.docs_source_path / "tailwind.config.js",
            self.config.docs_source_path / "tailwind.config.ts"
        ]
        
        for config_file in tailwind_configs:
            if config_file.exists():
                conflicts.append(ConfigurationConflict(
                    conflict_id=f"tailwind_config_{config_file.name}",
                    conflict_type=ConflictType.TAILWIND_CSS,
                    description=f"Tailwind config file found: {config_file.name}",
                    affected_files=[config_file],
                    severity="medium",
                    auto_resolvable=True,
                    resolution_strategy="remove_tailwind_config"
                ))
        
        return conflicts
    
    async def _detect_postcss_conflicts(self) -> List[ConfigurationConflict]:
        """Detect PostCSS related conflicts"""
        conflicts = []
        
        # Check for PostCSS config files
        postcss_configs = [
            self.config.docs_source_path / "postcss.config.js",
            self.config.docs_source_path / "postcss.config.ts"
        ]
        
        for config_file in postcss_configs:
            if config_file.exists():
                # Read config to check for Tailwind CSS
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                        if "tailwindcss" in content:
                            conflicts.append(ConfigurationConflict(
                                conflict_id=f"postcss_tailwind_{config_file.name}",
                                conflict_type=ConflictType.POSTCSS,
                                description=f"PostCSS config with Tailwind CSS: {config_file.name}",
                                affected_files=[config_file],
                                severity="high",
                                auto_resolvable=True,
                                resolution_strategy="remove_postcss_config"
                            ))
                except Exception as e:
                    logger.warning(f"Could not read PostCSS config {config_file}: {e}")
        
        return conflicts
    
    async def _detect_vite_conflicts(self) -> List[ConfigurationConflict]:
        """Detect Vite configuration conflicts"""
        conflicts = []
        
        # Check VitePress config for conflicting settings
        vitepress_config = self.config.docs_source_path / ".vitepress" / "config.ts"
        if vitepress_config.exists():
            try:
                with open(vitepress_config, 'r') as f:
                    content = f.read()
                    
                    # Check for PostCSS configuration that might conflict
                    if "postcss" in content and "tailwindcss" in content:
                        conflicts.append(ConfigurationConflict(
                            conflict_id="vitepress_postcss_conflict",
                            conflict_type=ConflictType.VITE_CONFIG,
                            description="VitePress config contains conflicting PostCSS settings",
                            affected_files=[vitepress_config],
                            severity="medium",
                            auto_resolvable=True,
                            resolution_strategy="fix_vitepress_postcss"
                        ))
            except Exception as e:
                logger.warning(f"Could not read VitePress config: {e}")
        
        return conflicts
    
    async def _detect_dependency_conflicts(self) -> List[ConfigurationConflict]:
        """Detect dependency conflicts"""
        conflicts = []
        
        # Check for conflicting dependencies in package.json
        package_json_path = self.config.docs_source_path / "package.json"
        if package_json_path.exists():
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
                
                dependencies = {**package_data.get("dependencies", {}), 
                              **package_data.get("devDependencies", {})}
                
                # Check for conflicting CSS processing dependencies
                css_processors = ["tailwindcss", "postcss", "autoprefixer", "@tailwindcss/postcss"]
                found_processors = [dep for dep in css_processors if dep in dependencies]
                
                if len(found_processors) > 1:
                    conflicts.append(ConfigurationConflict(
                        conflict_id="multiple_css_processors",
                        conflict_type=ConflictType.DEPENDENCY,
                        description=f"Multiple CSS processors found: {', '.join(found_processors)}",
                        affected_files=[package_json_path],
                        severity="medium",
                        auto_resolvable=True,
                        resolution_strategy="remove_conflicting_css_deps"
                    ))
        
        return conflicts
    
    async def _resolve_conflict(self, conflict: ConfigurationConflict) -> List[ResolutionAction]:
        """Resolve a specific conflict"""
        actions = []
        
        try:
            if conflict.resolution_strategy == "remove_tailwind_dependencies":
                actions.extend(await self._remove_tailwind_dependencies(conflict))
            elif conflict.resolution_strategy == "remove_tailwind_postcss":
                actions.extend(await self._remove_tailwind_postcss(conflict))
            elif conflict.resolution_strategy == "remove_tailwind_config":
                actions.extend(await self._remove_tailwind_config(conflict))
            elif conflict.resolution_strategy == "remove_postcss_config":
                actions.extend(await self._remove_postcss_config(conflict))
            elif conflict.resolution_strategy == "fix_vitepress_postcss":
                actions.extend(await self._fix_vitepress_postcss(conflict))
            elif conflict.resolution_strategy == "remove_conflicting_css_deps":
                actions.extend(await self._remove_conflicting_css_deps(conflict))
            
            self.resolution_history.extend(actions)
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict {conflict.conflict_id}: {e}")
        
        return actions
    
    async def _remove_tailwind_dependencies(self, conflict: ConfigurationConflict) -> List[ResolutionAction]:
        """Remove Tailwind CSS dependencies from package.json"""
        actions = []
        package_json_path = conflict.affected_files[0]
        
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
        
        # Remove from dependencies
        if "tailwindcss" in package_data.get("dependencies", {}):
            del package_data["dependencies"]["tailwindcss"]
            actions.append(ResolutionAction(
                action_id=f"remove_tailwind_dep_{int(datetime.utcnow().timestamp())}",
                conflict_id=conflict.conflict_id,
                action_type="modify",
                target_file=package_json_path,
                description="Removed tailwindcss from dependencies",
                success=True
            ))
        
        # Remove from devDependencies
        if "tailwindcss" in package_data.get("devDependencies", {}):
            del package_data["devDependencies"]["tailwindcss"]
            actions.append(ResolutionAction(
                action_id=f"remove_tailwind_devdep_{int(datetime.utcnow().timestamp())}",
                conflict_id=conflict.conflict_id,
                action_type="modify",
                target_file=package_json_path,
                description="Removed tailwindcss from devDependencies",
                success=True
            ))
        
        # Write back to file
        with open(package_json_path, 'w') as f:
            json.dump(package_data, f, indent=2)
        
        return actions

    async def _remove_tailwind_postcss(self, conflict: ConfigurationConflict) -> List[ResolutionAction]:
        """Remove Tailwind PostCSS plugin from package.json"""
        actions = []
        package_json_path = conflict.affected_files[0]

        with open(package_json_path, 'r') as f:
            package_data = json.load(f)

        # Remove from dependencies and devDependencies
        for dep_type in ["dependencies", "devDependencies"]:
            if "@tailwindcss/postcss" in package_data.get(dep_type, {}):
                del package_data[dep_type]["@tailwindcss/postcss"]
                actions.append(ResolutionAction(
                    action_id=f"remove_tailwind_postcss_{dep_type}_{int(datetime.utcnow().timestamp())}",
                    conflict_id=conflict.conflict_id,
                    action_type="modify",
                    target_file=package_json_path,
                    description=f"Removed @tailwindcss/postcss from {dep_type}",
                    success=True
                ))

        # Write back to file
        with open(package_json_path, 'w') as f:
            json.dump(package_data, f, indent=2)

        return actions

    async def _remove_tailwind_config(self, conflict: ConfigurationConflict) -> List[ResolutionAction]:
        """Remove Tailwind config files"""
        actions = []

        for config_file in conflict.affected_files:
            if config_file.exists():
                config_file.unlink()
                actions.append(ResolutionAction(
                    action_id=f"remove_config_{config_file.name}_{int(datetime.utcnow().timestamp())}",
                    conflict_id=conflict.conflict_id,
                    action_type="delete",
                    target_file=config_file,
                    description=f"Removed Tailwind config file: {config_file.name}",
                    success=True
                ))

        return actions

    async def _remove_postcss_config(self, conflict: ConfigurationConflict) -> List[ResolutionAction]:
        """Remove PostCSS config files"""
        actions = []

        for config_file in conflict.affected_files:
            if config_file.exists():
                config_file.unlink()
                actions.append(ResolutionAction(
                    action_id=f"remove_postcss_{config_file.name}_{int(datetime.utcnow().timestamp())}",
                    conflict_id=conflict.conflict_id,
                    action_type="delete",
                    target_file=config_file,
                    description=f"Removed PostCSS config file: {config_file.name}",
                    success=True
                ))

        return actions

    async def _fix_vitepress_postcss(self, conflict: ConfigurationConflict) -> List[ResolutionAction]:
        """Fix VitePress PostCSS configuration"""
        actions = []
        vitepress_config = conflict.affected_files[0]

        try:
            with open(vitepress_config, 'r') as f:
                content = f.read()

            # Replace problematic PostCSS configuration
            fixed_content = content.replace(
                'postcss: {',
                'css: {\n      postcss: false // Disable PostCSS to avoid conflicts\n    },\n    // Original postcss: {'
            )

            with open(vitepress_config, 'w') as f:
                f.write(fixed_content)

            actions.append(ResolutionAction(
                action_id=f"fix_vitepress_postcss_{int(datetime.utcnow().timestamp())}",
                conflict_id=conflict.conflict_id,
                action_type="modify",
                target_file=vitepress_config,
                description="Fixed VitePress PostCSS configuration",
                success=True
            ))

        except Exception as e:
            logger.error(f"Failed to fix VitePress PostCSS config: {e}")

        return actions

    async def _remove_conflicting_css_deps(self, conflict: ConfigurationConflict) -> List[ResolutionAction]:
        """Remove conflicting CSS dependencies"""
        actions = []
        package_json_path = conflict.affected_files[0]

        with open(package_json_path, 'r') as f:
            package_data = json.load(f)

        # Remove conflicting CSS processors (keep only essential ones)
        conflicting_deps = ["tailwindcss", "@tailwindcss/postcss"]

        for dep_type in ["dependencies", "devDependencies"]:
            deps = package_data.get(dep_type, {})
            for dep in conflicting_deps:
                if dep in deps:
                    del deps[dep]
                    actions.append(ResolutionAction(
                        action_id=f"remove_css_dep_{dep}_{int(datetime.utcnow().timestamp())}",
                        conflict_id=conflict.conflict_id,
                        action_type="modify",
                        target_file=package_json_path,
                        description=f"Removed conflicting CSS dependency: {dep}",
                        success=True
                    ))

        # Write back to file
        with open(package_json_path, 'w') as f:
            json.dump(package_data, f, indent=2)

        return actions

    async def _create_backup(self) -> Path:
        """Create backup of documentation source"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config.docs_source_path.parent / f"docs_backup_{timestamp}"
        
        shutil.copytree(self.config.docs_source_path, backup_path)
        logger.info(f"Created backup at: {backup_path}")
        
        return backup_path
    
    async def get_status(self) -> Dict[str, Any]:
        """Get conflict resolver status"""
        return {
            "running": self.is_running,
            "conflicts_detected": len(self.detected_conflicts),
            "resolutions_performed": len(self.resolution_history),
            "auto_resolve_enabled": self.config.auto_resolve_conflicts
        }
