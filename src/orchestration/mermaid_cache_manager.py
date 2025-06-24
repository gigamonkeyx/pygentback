"""
Intelligent Mermaid Diagram Cache Management System

Handles pre-generation, caching, and smart regeneration of Mermaid diagrams
to eliminate VitePress build performance bottlenecks.
"""

import os
import hashlib
import json
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MermaidCacheManager:
    """
    Manages Mermaid diagram caching and regeneration with intelligent triggers.
    
    Features:
    - Pre-generates Mermaid diagrams to static SVG files
    - Caches generated SVGs with modification tracking
    - Only regenerates when source .mmd files change
    - Integrates with documentation build triggers
    """
    
    def __init__(self,
                 docs_path: Path = Path("src/docs"),
                 cache_path: Path = Path("src/docs/public/diagrams"),
                 mermaid_cli_path: Optional[str] = None):
        self.docs_path = Path(docs_path)
        self.cache_path = Path(cache_path)

        # Set up Mermaid CLI path - use local installation if available
        if mermaid_cli_path:
            self.mermaid_cli_path = mermaid_cli_path
        else:
            # Try to use local mmdc installation first
            import platform
            if platform.system() == "Windows":
                local_mmdc = self.docs_path / "node_modules" / ".bin" / "mmdc.cmd"
            else:
                local_mmdc = self.docs_path / "node_modules" / ".bin" / "mmdc"

            if local_mmdc.exists():
                self.mermaid_cli_path = str(local_mmdc)
                logger.info(f"Using local Mermaid CLI: {self.mermaid_cli_path}")
            else:
                # Fallback to npx
                self.mermaid_cli_path = "npx @mermaid-js/mermaid-cli"
                logger.info("Using npx for Mermaid CLI")
        
        # Cache metadata
        self.cache_metadata_file = self.cache_path / "cache_metadata.json"
        self.cache_metadata: Dict[str, Dict] = {}
        
        # Ensure cache directory exists
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache metadata
        self._load_cache_metadata()
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    self.cache_metadata = json.load(f)
                logger.info(f"Loaded cache metadata for {len(self.cache_metadata)} diagrams")
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.cache_metadata = {}
        else:
            self.cache_metadata = {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2, default=str)
            logger.debug("Saved cache metadata")
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get SHA256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            return ""
    
    def _extract_mermaid_from_markdown(self, md_file: Path) -> List[Tuple[str, str]]:
        """
        Extract Mermaid diagrams from markdown files.
        Returns list of (diagram_id, mermaid_content) tuples.
        """
        diagrams = []
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for ArchitectureDiagram components with Mermaid content
            import re
            
            # Pattern to match ArchitectureDiagram components with multi-line content
            pattern = r'<ArchitectureDiagram[^>]*content="([^"]*(?:\n[^"]*)*)"[^>]*/?>'
            matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            
            for i, mermaid_content in enumerate(matches):
                # Clean up the content
                mermaid_content = mermaid_content.strip()
                if mermaid_content.startswith('`'):
                    mermaid_content = mermaid_content[1:]
                if mermaid_content.endswith('`'):
                    mermaid_content = mermaid_content[:-1]
                
                # Generate diagram ID
                diagram_id = f"{md_file.stem}_diagram_{i+1}"
                diagrams.append((diagram_id, mermaid_content))
            
            logger.debug(f"Extracted {len(diagrams)} diagrams from {md_file}")
            
        except Exception as e:
            logger.error(f"Failed to extract Mermaid from {md_file}: {e}")
        
        return diagrams
    
    def find_all_mermaid_sources(self) -> Dict[str, Dict]:
        """
        Find all Mermaid diagram sources in the documentation.
        Returns dict mapping diagram_id to source info.
        """
        sources = {}
        
        # Find standalone .mmd files
        for mmd_file in self.docs_path.rglob("*.mmd"):
            diagram_id = mmd_file.stem
            sources[diagram_id] = {
                'type': 'standalone',
                'source_file': mmd_file,
                'content': mmd_file.read_text(encoding='utf-8'),
                'hash': self._get_file_hash(mmd_file),
                'modified': datetime.fromtimestamp(mmd_file.stat().st_mtime)
            }
        
        # Find embedded diagrams in markdown files
        for md_file in self.docs_path.rglob("*.md"):
            diagrams = self._extract_mermaid_from_markdown(md_file)
            for diagram_id, content in diagrams:
                sources[diagram_id] = {
                    'type': 'embedded',
                    'source_file': md_file,
                    'content': content,
                    'hash': hashlib.sha256(content.encode()).hexdigest(),
                    'modified': datetime.fromtimestamp(md_file.stat().st_mtime)
                }
        
        logger.info(f"Found {len(sources)} Mermaid diagram sources")
        return sources
    
    def needs_regeneration(self, diagram_id: str, source_info: Dict) -> bool:
        """Check if a diagram needs regeneration"""
        
        # Check if SVG file exists
        svg_file = self.cache_path / f"{diagram_id}.svg"
        if not svg_file.exists():
            logger.debug(f"SVG missing for {diagram_id}, needs regeneration")
            return True
        
        # Check cache metadata
        if diagram_id not in self.cache_metadata:
            logger.debug(f"No cache metadata for {diagram_id}, needs regeneration")
            return True
        
        cached_info = self.cache_metadata[diagram_id]
        
        # Check if source hash changed
        if cached_info.get('hash') != source_info['hash']:
            logger.debug(f"Hash changed for {diagram_id}, needs regeneration")
            return True
        
        # Check if source file is newer than cached SVG
        source_modified = source_info['modified']
        svg_modified = datetime.fromtimestamp(svg_file.stat().st_mtime)
        
        if source_modified > svg_modified:
            logger.debug(f"Source newer than cache for {diagram_id}, needs regeneration")
            return True
        
        logger.debug(f"Diagram {diagram_id} is up to date")
        return False
    
    async def generate_diagram(self, diagram_id: str, mermaid_content: str) -> bool:
        """Generate SVG from Mermaid content"""

        try:
            # Create temporary .mmd file
            temp_mmd = self.cache_path / f"{diagram_id}_temp.mmd"
            svg_file = self.cache_path / f"{diagram_id}.svg"

            # Write Mermaid content to temp file
            with open(temp_mmd, 'w', encoding='utf-8') as f:
                f.write(mermaid_content)

            # Build command based on CLI path
            if self.mermaid_cli_path.endswith("mmdc") or "mmdc" in self.mermaid_cli_path:
                # Direct mmdc command
                cmd = [
                    self.mermaid_cli_path,
                    "-i", str(temp_mmd),
                    "-o", str(svg_file),
                    "--backgroundColor", "transparent"
                ]
            else:
                # npx command
                cmd = [
                    "npx", "@mermaid-js/mermaid-cli",
                    "-i", str(temp_mmd),
                    "-o", str(svg_file),
                    "--backgroundColor", "transparent"
                ]

            logger.info(f"Generating diagram: {diagram_id}")
            logger.debug(f"Command: {' '.join(cmd)}")
            logger.debug(f"Working directory: {self.docs_path}")

            # Execute with proper working directory
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.docs_path)  # Set working directory to docs path
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"Successfully generated {diagram_id}.svg")

                # Clean up temp file
                temp_mmd.unlink(missing_ok=True)

                return True
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Failed to generate {diagram_id}: {error_msg}")
                logger.debug(f"stdout: {stdout.decode() if stdout else 'None'}")
                return False

        except Exception as e:
            logger.error(f"Error generating diagram {diagram_id}: {e}")
            return False
    
    async def regenerate_diagrams(self, force: bool = False) -> Dict[str, bool]:
        """
        Regenerate Mermaid diagrams that need updating.
        Returns dict mapping diagram_id to success status.
        """
        
        logger.info("Starting diagram regeneration process")
        
        # Find all diagram sources
        sources = self.find_all_mermaid_sources()
        
        # Determine which diagrams need regeneration
        to_regenerate = []
        for diagram_id, source_info in sources.items():
            if force or self.needs_regeneration(diagram_id, source_info):
                to_regenerate.append((diagram_id, source_info))
        
        if not to_regenerate:
            logger.info("All diagrams are up to date")
            return {}
        
        logger.info(f"Regenerating {len(to_regenerate)} diagrams")
        
        # Generate diagrams
        results = {}
        for diagram_id, source_info in to_regenerate:
            success = await self.generate_diagram(diagram_id, source_info['content'])
            results[diagram_id] = success
            
            if success:
                # Update cache metadata
                self.cache_metadata[diagram_id] = {
                    'hash': source_info['hash'],
                    'generated_at': datetime.utcnow().isoformat(),
                    'source_file': str(source_info['source_file']),
                    'type': source_info['type']
                }
        
        # Save updated metadata
        self._save_cache_metadata()
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Regeneration complete: {successful}/{len(results)} successful")
        
        return results
    
    def get_cache_status(self) -> Dict[str, any]:
        """Get current cache status"""
        sources = self.find_all_mermaid_sources()
        
        status = {
            'total_diagrams': len(sources),
            'cached_diagrams': 0,
            'outdated_diagrams': 0,
            'missing_diagrams': 0,
            'cache_size_mb': 0,
            'last_updated': None
        }
        
        for diagram_id, source_info in sources.items():
            svg_file = self.cache_path / f"{diagram_id}.svg"
            
            if svg_file.exists():
                status['cached_diagrams'] += 1
                status['cache_size_mb'] += svg_file.stat().st_size / (1024 * 1024)
                
                if self.needs_regeneration(diagram_id, source_info):
                    status['outdated_diagrams'] += 1
            else:
                status['missing_diagrams'] += 1
        
        # Get last update time
        if self.cache_metadata:
            last_times = [
                datetime.fromisoformat(info['generated_at'])
                for info in self.cache_metadata.values()
                if 'generated_at' in info
            ]
            if last_times:
                status['last_updated'] = max(last_times).isoformat()
        
        return status
