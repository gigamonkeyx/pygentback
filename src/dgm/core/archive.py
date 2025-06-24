"""
DGM Archive - Stores and manages improvement history and rollback capabilities
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ..models import DGMArchiveEntry

logger = logging.getLogger(__name__)

class DGMArchive:
    """Archive for DGM improvement history and rollback"""
    
    def __init__(self, archive_path: str):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # Archive structure
        self.entries_dir = self.archive_path / "entries"
        self.entries_dir.mkdir(exist_ok=True)
        
        self.backups_dir = self.archive_path / "backups"
        self.backups_dir.mkdir(exist_ok=True)
        
        self.index_file = self.archive_path / "index.json"
        
        # In-memory index for fast lookups
        self._index: Dict[str, DGMArchiveEntry] = {}
        self._load_index()
    
    async def store_entry(self, entry: DGMArchiveEntry) -> bool:
        """Store an archive entry"""
        try:
            # Create backup if needed
            if entry.applied:
                backup_created = await self._create_backup(entry)
                if backup_created:
                    entry.rollback_info = {
                        "backup_path": str(self.backups_dir / f"{entry.id}_backup.json"),
                        "backup_timestamp": datetime.utcnow().isoformat()
                    }
            
            # Store entry to disk
            entry_file = self.entries_dir / f"{entry.id}.json"
            with open(entry_file, 'w') as f:
                json.dump(entry.model_dump(), f, indent=2, default=str)
            
            # Update index
            self._index[entry.id] = entry
            await self._save_index()
            
            logger.info(f"Stored archive entry {entry.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store archive entry {entry.id}: {e}")
            return False
    
    async def get_entry(self, entry_id: str) -> Optional[DGMArchiveEntry]:
        """Retrieve an archive entry by ID"""
        if entry_id in self._index:
            return self._index[entry_id]
        
        # Try loading from disk if not in memory
        entry_file = self.entries_dir / f"{entry_id}.json"
        if entry_file.exists():
            try:
                with open(entry_file, 'r') as f:
                    data = json.load(f)
                entry = DGMArchiveEntry(**data)
                self._index[entry_id] = entry
                return entry
            except Exception as e:
                logger.error(f"Failed to load archive entry {entry_id}: {e}")
        
        return None
    
    async def get_entries_by_agent(self, agent_id: str) -> List[DGMArchiveEntry]:
        """Get all entries for a specific agent"""
        entries = []
        for entry in self._index.values():
            if entry.agent_id == agent_id:
                entries.append(entry)
        
        # Sort by timestamp (newest first)
        entries.sort(key=lambda x: x.improvement_candidate.created_at, reverse=True)
        return entries
    
    async def get_successful_improvements(self, agent_id: str) -> List[DGMArchiveEntry]:
        """Get all successful improvements for an agent"""
        entries = await self.get_entries_by_agent(agent_id)
        return [entry for entry in entries if entry.applied and 
                entry.validation_result and entry.validation_result.success]
    
    async def rollback_improvement(self, entry_id: str) -> bool:
        """Rollback a previously applied improvement"""
        entry = await self.get_entry(entry_id)
        if not entry:
            logger.error(f"Archive entry {entry_id} not found")
            return False
        
        if not entry.applied:
            logger.warning(f"Entry {entry_id} was not applied, nothing to rollback")
            return True
        
        if not entry.rollback_info:
            logger.error(f"No rollback info available for entry {entry_id}")
            return False
        
        try:
            # Load backup information
            backup_path = entry.rollback_info.get("backup_path")
            if not backup_path or not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Restore original files
            for filename, original_content in backup_data.get("original_files", {}).items():
                try:
                    with open(filename, 'w') as f:
                        f.write(original_content)
                    logger.info(f"Restored file: {filename}")
                except Exception as e:
                    logger.error(f"Failed to restore file {filename}: {e}")
                    return False
            
            # Update entry status
            entry.applied = False
            entry.rollback_info["rollback_timestamp"] = datetime.utcnow().isoformat()
            
            # Save updated entry
            await self.store_entry(entry)
            
            logger.info(f"Successfully rolled back improvement {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback improvement {entry_id}: {e}")
            return False
    
    async def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """Clean up old archive entries"""
        cutoff_date = datetime.utcnow().timestamp() - (max_age_days * 24 * 3600)
        removed_count = 0
        
        entries_to_remove = []
        for entry_id, entry in self._index.items():
            entry_age = entry.improvement_candidate.created_at.timestamp()
            if entry_age < cutoff_date and not entry.applied:
                entries_to_remove.append(entry_id)
        
        for entry_id in entries_to_remove:
            try:
                # Remove entry file
                entry_file = self.entries_dir / f"{entry_id}.json"
                if entry_file.exists():
                    entry_file.unlink()
                
                # Remove from index
                del self._index[entry_id]
                removed_count += 1
                
                logger.info(f"Cleaned up old archive entry {entry_id}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup entry {entry_id}: {e}")
        
        if removed_count > 0:
            await self._save_index()
        
        return removed_count
    
    async def get_statistics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get archive statistics"""
        entries = list(self._index.values())
        
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        
        stats = {
            "total_entries": len(entries),
            "applied_improvements": sum(1 for e in entries if e.applied),
            "successful_validations": sum(1 for e in entries if e.validation_result and e.validation_result.success),
            "failed_validations": sum(1 for e in entries if e.validation_result and not e.validation_result.success),
            "pending_validations": sum(1 for e in entries if not e.validation_result),
            "average_improvement_score": 0.0,
            "archive_size_mb": self._calculate_archive_size()
        }
        
        # Calculate average improvement score
        improvement_scores = []
        for entry in entries:
            if entry.validation_result and entry.validation_result.success:
                improvement_scores.append(entry.validation_result.improvement_score)
        
        if improvement_scores:
            stats["average_improvement_score"] = sum(improvement_scores) / len(improvement_scores)
        
        return stats
    
    def _load_index(self):
        """Load the archive index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                
                for entry_id, entry_data in index_data.items():
                    try:
                        entry = DGMArchiveEntry(**entry_data)
                        self._index[entry_id] = entry
                    except Exception as e:
                        logger.error(f"Failed to load index entry {entry_id}: {e}")
                        
                logger.info(f"Loaded archive index with {len(self._index)} entries")
                
            except Exception as e:
                logger.error(f"Failed to load archive index: {e}")
                self._index = {}
    
    async def _save_index(self):
        """Save the archive index to disk"""
        try:
            index_data = {}
            for entry_id, entry in self._index.items():
                index_data[entry_id] = entry.model_dump()
            
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save archive index: {e}")
    
    async def _create_backup(self, entry: DGMArchiveEntry) -> bool:
        """Create backup of current files before applying improvement"""
        try:
            backup_data = {
                "entry_id": entry.id,
                "agent_id": entry.agent_id,
                "backup_timestamp": datetime.utcnow().isoformat(),
                "original_files": {}
            }
            
            # Read current content of files that will be modified
            for filename in entry.improvement_candidate.code_changes.keys():
                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        backup_data["original_files"][filename] = f.read()
            
            # Save backup
            backup_file = self.backups_dir / f"{entry.id}_backup.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Created backup for entry {entry.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup for entry {entry.id}: {e}")
            return False
    
    def _calculate_archive_size(self) -> float:
        """Calculate total archive size in MB"""
        total_size = 0
        
        for root, dirs, files in os.walk(self.archive_path):
            for file in files:
                filepath = os.path.join(root, file)
                total_size += os.path.getsize(filepath)
        
        return total_size / (1024 * 1024)  # Convert to MB
