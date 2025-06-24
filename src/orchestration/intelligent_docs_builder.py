"""
Intelligent Documentation Build Orchestrator

Coordinates the entire documentation build process with smart triggers,
Mermaid caching, and optimized build strategies.
"""

import asyncio
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import subprocess
import json

from .mermaid_cache_manager import MermaidCacheManager
from .build_trigger_detector import BuildTriggerDetector

logger = logging.getLogger(__name__)


class IntelligentDocsBuilder:
    """
    Orchestrates intelligent documentation builds with:
    - Smart trigger detection
    - Mermaid diagram caching
    - Incremental builds
    - Fallback strategies
    """
    
    def __init__(self,
                 docs_path: Path = Path("src/docs"),
                 output_path: Path = Path("src/docs/.vitepress/dist"),
                 cache_path: Path = Path("src/docs/public/diagrams")):
        self.docs_path = Path(docs_path)
        self.output_path = Path(output_path)
        self.cache_path = Path(cache_path)
        
        # Initialize components
        self.mermaid_manager = MermaidCacheManager(
            docs_path=self.docs_path,
            cache_path=self.cache_path
        )
        self.trigger_detector = BuildTriggerDetector(
            docs_path=self.docs_path
        )
        
        # Build state
        self.build_history_file = self.docs_path / ".build_history.json"
        self.build_history = self._load_build_history()
    
    def _load_build_history(self) -> List[Dict]:
        """Load build history from disk"""
        if self.build_history_file.exists():
            try:
                with open(self.build_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load build history: {e}")
        return []
    
    def _save_build_history(self):
        """Save build history to disk"""
        try:
            # Keep only last 50 builds
            if len(self.build_history) > 50:
                self.build_history = self.build_history[-50:]
            
            with open(self.build_history_file, 'w') as f:
                json.dump(self.build_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save build history: {e}")
    
    async def check_build_triggers(self) -> Dict[str, Any]:
        """Check if documentation build should be triggered"""
        return await self.trigger_detector.check_all_triggers()
    
    async def prepare_mermaid_diagrams(self, force: bool = False) -> Dict[str, bool]:
        """Prepare Mermaid diagrams (regenerate if needed)"""
        logger.info("Preparing Mermaid diagrams")
        
        # Get cache status
        cache_status = self.mermaid_manager.get_cache_status()
        logger.info(f"Mermaid cache status: {cache_status}")
        
        # Regenerate diagrams if needed
        if force or cache_status['outdated_diagrams'] > 0 or cache_status['missing_diagrams'] > 0:
            logger.info("Regenerating Mermaid diagrams")
            results = await self.mermaid_manager.regenerate_diagrams(force=force)
            return results
        else:
            logger.info("All Mermaid diagrams are up to date")
            return {}
    
    async def run_vitepress_build(self, production: bool = True) -> Dict[str, Any]:
        """Run VitePress build with optimized settings"""
        
        build_result = {
            'success': False,
            'duration_seconds': 0,
            'output_size_mb': 0,
            'files_generated': 0,
            'error': None
        }
        
        start_time = datetime.utcnow()
        
        try:
            # Ensure we're in the docs directory
            original_cwd = Path.cwd()
            os.chdir(self.docs_path)
            
            # Build command
            if production:
                cmd = ["npx", "vitepress", "build"]
            else:
                cmd = ["npx", "vitepress", "build", "--mode", "development"]
            
            logger.info(f"Running VitePress build: {' '.join(cmd)}")
            
            # Run build
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.docs_path
            )
            
            stdout, stderr = await process.communicate()
            
            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            build_result['duration_seconds'] = duration
            
            if process.returncode == 0:
                build_result['success'] = True
                logger.info(f"VitePress build completed successfully in {duration:.1f}s")
                
                # Calculate output statistics
                if self.output_path.exists():
                    total_size = 0
                    file_count = 0
                    for file_path in self.output_path.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                            file_count += 1
                    
                    build_result['output_size_mb'] = total_size / (1024 * 1024)
                    build_result['files_generated'] = file_count
                
            else:
                build_result['error'] = stderr.decode()
                logger.error(f"VitePress build failed: {build_result['error']}")
            
        except Exception as e:
            build_result['error'] = str(e)
            logger.error(f"VitePress build error: {e}")
        
        finally:
            # Restore original working directory
            try:
                os.chdir(original_cwd)
            except:
                pass
        
        return build_result
    
    async def execute_intelligent_build(self, 
                                      force: bool = False,
                                      production: bool = True) -> Dict[str, Any]:
        """
        Execute intelligent documentation build with all optimizations.
        
        Process:
        1. Check build triggers
        2. Prepare Mermaid diagrams
        3. Run optimized VitePress build
        4. Record build history
        """
        
        build_session = {
            'build_id': f"build_{int(datetime.utcnow().timestamp())}",
            'timestamp': datetime.utcnow().isoformat(),
            'force': force,
            'production': production,
            'triggers': {},
            'mermaid_results': {},
            'build_results': {},
            'success': False,
            'total_duration_seconds': 0,
            'error': None
        }
        
        session_start = datetime.utcnow()
        
        try:
            logger.info(f"Starting intelligent documentation build: {build_session['build_id']}")
            
            # Step 1: Check build triggers (unless forced)
            if not force:
                logger.info("Checking build triggers")
                trigger_status = await self.check_build_triggers()
                build_session['triggers'] = trigger_status
                
                if not trigger_status['should_build']:
                    logger.info("No build triggers detected, skipping build")
                    build_session['success'] = True
                    build_session['skipped'] = True
                    return build_session
                
                logger.info(f"Build triggered by: {', '.join(trigger_status['summary']['trigger_reasons'])}")
            else:
                logger.info("Forced build requested")
                build_session['triggers'] = {'forced': True}
            
            # Step 2: Prepare Mermaid diagrams
            logger.info("Preparing Mermaid diagrams")
            mermaid_results = await self.prepare_mermaid_diagrams(force=force)
            build_session['mermaid_results'] = mermaid_results
            
            if mermaid_results:
                successful_diagrams = sum(1 for success in mermaid_results.values() if success)
                logger.info(f"Mermaid preparation: {successful_diagrams}/{len(mermaid_results)} diagrams successful")
            
            # Step 3: Run VitePress build
            logger.info("Running VitePress build")
            build_results = await self.run_vitepress_build(production=production)
            build_session['build_results'] = build_results
            
            if build_results['success']:
                logger.info(f"Build completed successfully in {build_results['duration_seconds']:.1f}s")
                logger.info(f"Generated {build_results['files_generated']} files ({build_results['output_size_mb']:.1f} MB)")
                build_session['success'] = True
            else:
                logger.error(f"Build failed: {build_results.get('error', 'Unknown error')}")
                build_session['error'] = build_results.get('error')
            
        except Exception as e:
            logger.error(f"Build session error: {e}")
            build_session['error'] = str(e)
        
        finally:
            # Calculate total duration
            total_duration = (datetime.utcnow() - session_start).total_seconds()
            build_session['total_duration_seconds'] = total_duration
            
            # Record in build history
            self.build_history.append(build_session)
            self._save_build_history()
            
            logger.info(f"Build session complete: {build_session['build_id']} "
                       f"(success={build_session['success']}, duration={total_duration:.1f}s)")
        
        return build_session
    
    async def get_build_status(self) -> Dict[str, Any]:
        """Get comprehensive build system status"""
        
        # Get trigger status
        trigger_status = await self.check_build_triggers()
        
        # Get Mermaid cache status
        cache_status = self.mermaid_manager.get_cache_status()
        
        # Get recent build history
        recent_builds = self.build_history[-10:] if self.build_history else []
        
        # Calculate success rate
        if recent_builds:
            successful_builds = sum(1 for build in recent_builds if build.get('success', False))
            success_rate = successful_builds / len(recent_builds)
        else:
            success_rate = 0.0
        
        return {
            'system_status': 'ready',
            'triggers': trigger_status,
            'mermaid_cache': cache_status,
            'build_history': {
                'total_builds': len(self.build_history),
                'recent_builds': len(recent_builds),
                'success_rate': success_rate,
                'last_build': recent_builds[-1] if recent_builds else None
            },
            'recommendations': self._generate_recommendations(trigger_status, cache_status)
        }
    
    def _generate_recommendations(self, trigger_status: Dict, cache_status: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if trigger_status['should_build']:
            recommendations.append("Documentation build is recommended due to detected changes")
        
        if cache_status['missing_diagrams'] > 0:
            recommendations.append(f"Regenerate {cache_status['missing_diagrams']} missing Mermaid diagrams")
        
        if cache_status['outdated_diagrams'] > 0:
            recommendations.append(f"Update {cache_status['outdated_diagrams']} outdated Mermaid diagrams")
        
        if not recommendations:
            recommendations.append("Documentation system is up to date")
        
        return recommendations
    
    async def force_rebuild(self) -> Dict[str, Any]:
        """Force a complete rebuild of all documentation"""
        logger.info("Forcing complete documentation rebuild")
        return await self.execute_intelligent_build(force=True, production=True)
