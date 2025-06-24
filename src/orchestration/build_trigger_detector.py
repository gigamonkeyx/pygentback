"""
Intelligent Documentation Build Trigger Detection System

Detects when documentation should be rebuilt based on various trigger conditions:
- Git commits to main/master branch
- Documentation file changes
- Version tag creation
- Manual triggers
"""

import os
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class BuildTriggerDetector:
    """
    Detects conditions that should trigger documentation rebuilds.
    
    Monitors:
    - Git repository changes
    - File system modifications
    - Version tags
    - Manual trigger requests
    """
    
    def __init__(self, 
                 repo_path: Path = Path("."),
                 docs_path: Path = Path("src/docs"),
                 trigger_state_file: Path = Path("src/docs/.build_triggers.json")):
        self.repo_path = Path(repo_path)
        self.docs_path = Path(docs_path)
        self.trigger_state_file = Path(trigger_state_file)
        
        # Monitored file patterns
        self.monitored_patterns = [
            "*.md",      # Markdown files
            "*.vue",     # Vue components
            "*.mmd",     # Mermaid diagrams
            "*.ts",      # TypeScript config
            "*.js",      # JavaScript config
            "*.json",    # Config files
        ]
        
        # Git branches to monitor
        self.monitored_branches = ["main", "master", "develop"]
        
        # Load previous state
        self.previous_state = self._load_trigger_state()
    
    def _load_trigger_state(self) -> Dict:
        """Load previous trigger state from disk"""
        if self.trigger_state_file.exists():
            try:
                with open(self.trigger_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load trigger state: {e}")
        
        return {
            'last_build_time': None,
            'last_commit_hash': None,
            'last_tag': None,
            'file_hashes': {},
            'manual_triggers': []
        }
    
    def _save_trigger_state(self, state: Dict):
        """Save trigger state to disk"""
        try:
            self.trigger_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trigger_state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.debug("Saved trigger state")
        except Exception as e:
            logger.error(f"Failed to save trigger state: {e}")
    
    def _run_git_command(self, args: List[str]) -> Optional[str]:
        """Run git command and return output"""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.warning(f"Git command failed: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Git command error: {e}")
            return None
    
    def get_current_commit_hash(self) -> Optional[str]:
        """Get current commit hash"""
        return self._run_git_command(["rev-parse", "HEAD"])
    
    def get_current_branch(self) -> Optional[str]:
        """Get current branch name"""
        return self._run_git_command(["branch", "--show-current"])
    
    def get_latest_tag(self) -> Optional[str]:
        """Get latest git tag"""
        return self._run_git_command(["describe", "--tags", "--abbrev=0"])
    
    def get_commits_since(self, since_hash: str) -> List[Dict]:
        """Get commits since a specific hash"""
        if not since_hash:
            return []
        
        output = self._run_git_command([
            "log", 
            f"{since_hash}..HEAD",
            "--pretty=format:%H|%s|%an|%ad",
            "--date=iso"
        ])
        
        if not output:
            return []
        
        commits = []
        for line in output.split('\n'):
            if '|' in line:
                parts = line.split('|', 3)
                if len(parts) == 4:
                    commits.append({
                        'hash': parts[0],
                        'message': parts[1],
                        'author': parts[2],
                        'date': parts[3]
                    })
        
        return commits
    
    def check_git_triggers(self) -> Dict[str, any]:
        """Check for git-based trigger conditions"""
        triggers = {
            'new_commits': False,
            'new_tag': False,
            'branch_change': False,
            'commits': [],
            'details': {}
        }
        
        # Check current state
        current_commit = self.get_current_commit_hash()
        current_branch = self.get_current_branch()
        current_tag = self.get_latest_tag()
        
        previous_commit = self.previous_state.get('last_commit_hash')
        previous_tag = self.previous_state.get('last_tag')
        
        # Check for new commits
        if current_commit != previous_commit:
            triggers['new_commits'] = True
            triggers['commits'] = self.get_commits_since(previous_commit)
            triggers['details']['commit_change'] = {
                'from': previous_commit,
                'to': current_commit,
                'count': len(triggers['commits'])
            }
        
        # Check for new tags
        if current_tag != previous_tag:
            triggers['new_tag'] = True
            triggers['details']['tag_change'] = {
                'from': previous_tag,
                'to': current_tag
            }
        
        # Check if we're on a monitored branch
        if current_branch in self.monitored_branches:
            triggers['details']['monitored_branch'] = current_branch
        
        return triggers
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
    def check_file_triggers(self) -> Dict[str, any]:
        """Check for file-based trigger conditions"""
        triggers = {
            'files_changed': False,
            'changed_files': [],
            'new_files': [],
            'deleted_files': [],
            'details': {}
        }
        
        current_hashes = {}
        previous_hashes = self.previous_state.get('file_hashes', {})
        
        # Scan all monitored files
        for pattern in self.monitored_patterns:
            for file_path in self.docs_path.rglob(pattern):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(self.docs_path))
                    current_hashes[rel_path] = self.get_file_hash(file_path)
        
        # Check for changes
        for file_path, current_hash in current_hashes.items():
            previous_hash = previous_hashes.get(file_path)
            
            if previous_hash is None:
                # New file
                triggers['new_files'].append(file_path)
                triggers['files_changed'] = True
            elif previous_hash != current_hash:
                # Changed file
                triggers['changed_files'].append(file_path)
                triggers['files_changed'] = True
        
        # Check for deleted files
        for file_path in previous_hashes:
            if file_path not in current_hashes:
                triggers['deleted_files'].append(file_path)
                triggers['files_changed'] = True
        
        triggers['details'] = {
            'total_files': len(current_hashes),
            'new_count': len(triggers['new_files']),
            'changed_count': len(triggers['changed_files']),
            'deleted_count': len(triggers['deleted_files'])
        }
        
        return triggers, current_hashes
    
    def check_manual_triggers(self) -> Dict[str, any]:
        """Check for manual trigger requests"""
        triggers = {
            'manual_requested': False,
            'trigger_requests': [],
            'details': {}
        }
        
        # Check for trigger files or environment variables
        trigger_file = self.docs_path / ".force_rebuild"
        if trigger_file.exists():
            triggers['manual_requested'] = True
            triggers['trigger_requests'].append({
                'type': 'file',
                'source': str(trigger_file),
                'timestamp': datetime.fromtimestamp(trigger_file.stat().st_mtime).isoformat()
            })
            
            # Remove trigger file after detection
            try:
                trigger_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove trigger file: {e}")
        
        # Check environment variable
        if os.getenv('FORCE_DOCS_REBUILD'):
            triggers['manual_requested'] = True
            triggers['trigger_requests'].append({
                'type': 'environment',
                'source': 'FORCE_DOCS_REBUILD',
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return triggers
    
    def check_time_triggers(self) -> Dict[str, any]:
        """Check for time-based trigger conditions"""
        triggers = {
            'time_triggered': False,
            'details': {}
        }
        
        last_build = self.previous_state.get('last_build_time')
        if last_build:
            last_build_time = datetime.fromisoformat(last_build)
            time_since_build = datetime.utcnow() - last_build_time
            
            # Trigger if it's been more than 24 hours since last build
            if time_since_build > timedelta(hours=24):
                triggers['time_triggered'] = True
                triggers['details'] = {
                    'last_build': last_build,
                    'hours_since': time_since_build.total_seconds() / 3600
                }
        else:
            # No previous build recorded
            triggers['time_triggered'] = True
            triggers['details'] = {'reason': 'no_previous_build'}
        
        return triggers
    
    async def check_all_triggers(self) -> Dict[str, any]:
        """Check all trigger conditions and return comprehensive status"""
        
        logger.info("Checking documentation build triggers")
        
        # Check all trigger types
        git_triggers = self.check_git_triggers()
        file_triggers, current_hashes = self.check_file_triggers()
        manual_triggers = self.check_manual_triggers()
        time_triggers = self.check_time_triggers()
        
        # Determine if build should be triggered
        should_build = (
            git_triggers['new_commits'] or
            git_triggers['new_tag'] or
            file_triggers['files_changed'] or
            manual_triggers['manual_requested'] or
            time_triggers['time_triggered']
        )
        
        # Compile comprehensive trigger status
        trigger_status = {
            'should_build': should_build,
            'timestamp': datetime.utcnow().isoformat(),
            'triggers': {
                'git': git_triggers,
                'files': file_triggers,
                'manual': manual_triggers,
                'time': time_triggers
            },
            'summary': {
                'total_triggers': sum([
                    git_triggers['new_commits'],
                    git_triggers['new_tag'],
                    file_triggers['files_changed'],
                    manual_triggers['manual_requested'],
                    time_triggers['time_triggered']
                ]),
                'trigger_reasons': []
            }
        }
        
        # Build summary of trigger reasons
        if git_triggers['new_commits']:
            trigger_status['summary']['trigger_reasons'].append(
                f"New commits: {len(git_triggers['commits'])}"
            )
        if git_triggers['new_tag']:
            trigger_status['summary']['trigger_reasons'].append("New version tag")
        if file_triggers['files_changed']:
            trigger_status['summary']['trigger_reasons'].append(
                f"File changes: {file_triggers['details']['changed_count']} changed, "
                f"{file_triggers['details']['new_count']} new"
            )
        if manual_triggers['manual_requested']:
            trigger_status['summary']['trigger_reasons'].append("Manual trigger")
        if time_triggers['time_triggered']:
            trigger_status['summary']['trigger_reasons'].append("Time-based trigger")
        
        # Update state if build should be triggered
        if should_build:
            new_state = {
                'last_build_time': datetime.utcnow().isoformat(),
                'last_commit_hash': self.get_current_commit_hash(),
                'last_tag': self.get_latest_tag(),
                'file_hashes': current_hashes,
                'manual_triggers': []
            }
            self._save_trigger_state(new_state)
            self.previous_state = new_state
        
        logger.info(f"Trigger check complete: should_build={should_build}")
        return trigger_status
