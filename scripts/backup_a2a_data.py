#!/usr/bin/env python3
"""
A2A Database Backup Strategy Script

Creates comprehensive backups of A2A-related data and provides rollback procedures.
This script handles both SQLite and PostgreSQL databases.

Features:
1. Full database backup
2. A2A-specific table backup
3. Schema backup
4. Rollback procedures
5. Backup validation
"""

import asyncio
import aiosqlite
import json
import os
import sys
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class A2ABackupManager:
    """Manages A2A database backups and rollback procedures"""
    
    def __init__(self, database_url: str, backup_dir: str = "./backups"):
        self.database_url = database_url
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.connection: Optional[aiosqlite.Connection] = None
        
    async def connect(self) -> bool:
        """Connect to database"""
        try:
            # Extract database path from URL
            if ":///" in self.database_url:
                db_path = self.database_url.split("///")[1]
            else:
                db_path = self.database_url
            
            self.connection = await aiosqlite.connect(db_path)
            self.connection.row_factory = aiosqlite.Row
            print("✅ Connected to database successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            await self.connection.close()
            print("✅ Disconnected from database")
    
    def get_backup_filename(self, backup_type: str) -> str:
        """Generate backup filename with timestamp"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"a2a_{backup_type}_backup_{timestamp}"
    
    async def backup_full_database(self) -> Dict[str, Any]:
        """Create full database backup"""
        print("\n🔄 Creating full database backup...")
        
        try:
            # For SQLite, copy the entire database file
            if "sqlite" in self.database_url.lower():
                db_path = self.database_url.split("///")[1] if ":///" in self.database_url else self.database_url
                
                if not os.path.exists(db_path):
                    print(f"⚠️ Database file not found: {db_path}")
                    return {"success": False, "error": "Database file not found"}
                
                backup_filename = self.get_backup_filename("full") + ".db"
                backup_path = self.backup_dir / backup_filename
                
                # Copy database file
                shutil.copy2(db_path, backup_path)
                
                # Verify backup
                backup_size = os.path.getsize(backup_path)
                original_size = os.path.getsize(db_path)
                
                if backup_size == original_size:
                    print(f"✅ Full database backup created: {backup_path}")
                    print(f"   📊 Backup size: {backup_size:,} bytes")
                    
                    return {
                        "success": True,
                        "backup_path": str(backup_path),
                        "backup_size": backup_size,
                        "original_size": original_size,
                        "backup_type": "full_database"
                    }
                else:
                    print(f"❌ Backup verification failed: size mismatch")
                    return {"success": False, "error": "Backup verification failed"}
            
            else:
                print("⚠️ PostgreSQL backup not implemented in this version")
                return {"success": False, "error": "PostgreSQL backup not implemented"}
                
        except Exception as e:
            print(f"❌ Full database backup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def backup_a2a_tables(self) -> Dict[str, Any]:
        """Backup A2A-specific tables"""
        print("\n🔄 Creating A2A tables backup...")
        
        try:
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "tables": {}
            }
            
            # List of A2A tables to backup
            a2a_tables = ["a2a_agents", "a2a_tasks"]
            
            for table_name in a2a_tables:
                # Check if table exists
                cursor = await self.connection.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table_name,))
                table_exists = await cursor.fetchone()
                
                if table_exists:
                    print(f"   📋 Backing up table: {table_name}")
                    
                    # Get table schema
                    cursor = await self.connection.execute(f"PRAGMA table_info({table_name})")
                    schema = await cursor.fetchall()
                    
                    # Get table data
                    cursor = await self.connection.execute(f"SELECT * FROM {table_name}")
                    rows = await cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    table_data = []
                    for row in rows:
                        row_dict = {}
                        for key in row.keys():
                            row_dict[key] = row[key]
                        table_data.append(row_dict)
                    
                    backup_data["tables"][table_name] = {
                        "schema": [dict(col) for col in schema],
                        "data": table_data,
                        "row_count": len(table_data)
                    }
                    
                    print(f"   ✅ Backed up {len(table_data)} rows from {table_name}")
                else:
                    print(f"   ⚠️ Table {table_name} does not exist")
                    backup_data["tables"][table_name] = {
                        "exists": False,
                        "message": "Table does not exist"
                    }
            
            # Save backup to JSON file
            backup_filename = self.get_backup_filename("tables") + ".json"
            backup_path = self.backup_dir / backup_filename
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            print(f"✅ A2A tables backup created: {backup_path}")
            
            return {
                "success": True,
                "backup_path": str(backup_path),
                "tables_backed_up": len([t for t in backup_data["tables"] if backup_data["tables"][t].get("exists", True)]),
                "backup_type": "a2a_tables"
            }
            
        except Exception as e:
            print(f"❌ A2A tables backup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def backup_schema(self) -> Dict[str, Any]:
        """Backup database schema"""
        print("\n🔄 Creating schema backup...")
        
        try:
            schema_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "database_type": "sqlite",
                "tables": {}
            }
            
            # Get all tables
            cursor = await self.connection.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = await cursor.fetchall()
            
            for table in tables:
                table_name = table['name']
                print(f"   📋 Backing up schema for: {table_name}")
                
                # Get table schema
                cursor = await self.connection.execute(f"PRAGMA table_info({table_name})")
                columns = await cursor.fetchall()
                
                # Get CREATE TABLE statement
                cursor = await self.connection.execute("""
                    SELECT sql FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table_name,))
                create_sql = await cursor.fetchone()
                
                schema_data["tables"][table_name] = {
                    "columns": [dict(col) for col in columns],
                    "create_sql": create_sql['sql'] if create_sql else None
                }
            
            # Save schema backup
            backup_filename = self.get_backup_filename("schema") + ".json"
            backup_path = self.backup_dir / backup_filename
            
            with open(backup_path, 'w') as f:
                json.dump(schema_data, f, indent=2)
            
            print(f"✅ Schema backup created: {backup_path}")
            print(f"   📊 Tables backed up: {len(schema_data['tables'])}")
            
            return {
                "success": True,
                "backup_path": str(backup_path),
                "tables_count": len(schema_data['tables']),
                "backup_type": "schema"
            }
            
        except Exception as e:
            print(f"❌ Schema backup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_rollback_script(self, backup_info: List[Dict[str, Any]]) -> str:
        """Create rollback script for backups"""
        print("\n📝 Creating rollback script...")
        
        rollback_script = f"""#!/usr/bin/env python3
\"\"\"
A2A Database Rollback Script
Generated: {datetime.utcnow().isoformat()}

This script can restore the database from backups created during A2A integration.
\"\"\"

import os
import shutil
import json
from pathlib import Path

def rollback_full_database(backup_path: str, target_path: str):
    \"\"\"Rollback full database from backup\"\"\"
    print(f"Rolling back database from {{backup_path}}")

    if not os.path.exists(backup_path):
        print(f"ERROR: Backup file not found: {{backup_path}}")
        return False

    try:
        # Stop any running services first
        print("WARNING: Make sure to stop all database connections before rollback!")

        # Create backup of current state
        current_backup = target_path + ".pre_rollback_" + str(int(time.time()))
        if os.path.exists(target_path):
            shutil.copy2(target_path, current_backup)
            print(f"Current database backed up to: {{current_backup}}")

        # Restore from backup
        shutil.copy2(backup_path, target_path)
        print(f"Database restored from backup")
        return True

    except Exception as e:
        print(f"ERROR: Rollback failed: {{e}}")
        return False

def main():
    \"\"\"Main rollback function\"\"\"
    print("A2A Database Rollback Utility")
    print("=" * 50)
    
    # Available backups:
"""
        
        for backup in backup_info:
            if backup.get("success"):
                rollback_script += f"""    # {backup['backup_type']}: {backup['backup_path']}
"""
        
        rollback_script += f"""
    # Example usage:
    # rollback_full_database("./backups/a2a_full_backup_20231222_120000.db", "./data/pygent_factory.db")
    
    print("Available backups:")
"""
        
        for i, backup in enumerate(backup_info):
            if backup.get("success"):
                rollback_script += f"""    print("  {i+1}. {backup['backup_type']}: {backup['backup_path']}")
"""
        
        rollback_script += """
    print("\\nTo perform rollback, uncomment and modify the appropriate rollback function call above.")
    print("WARNING: Always stop all database connections before rollback!")

if __name__ == "__main__":
    main()
"""
        
        # Save rollback script
        rollback_filename = self.get_backup_filename("rollback") + ".py"
        rollback_path = self.backup_dir / rollback_filename
        
        with open(rollback_path, 'w', encoding='utf-8') as f:
            f.write(rollback_script)
        
        # Make executable
        os.chmod(rollback_path, 0o755)
        
        print(f"✅ Rollback script created: {rollback_path}")
        return str(rollback_path)
    
    async def run_backup_strategy(self) -> Dict[str, Any]:
        """Run complete backup strategy"""
        print("🚀 Starting A2A Database Backup Strategy...")
        print("=" * 60)
        
        if not await self.connect():
            raise Exception("Failed to connect to database")
        
        backup_results = []
        
        try:
            # 1. Full database backup
            full_backup = await self.backup_full_database()
            backup_results.append(full_backup)
            
            # 2. A2A tables backup
            tables_backup = await self.backup_a2a_tables()
            backup_results.append(tables_backup)
            
            # 3. Schema backup
            schema_backup = await self.backup_schema()
            backup_results.append(schema_backup)
            
            # 4. Create rollback script
            rollback_script = await self.create_rollback_script(backup_results)
            
            # Summary
            successful_backups = [b for b in backup_results if b.get("success")]
            
            return {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "backups_created": len(successful_backups),
                "total_backups": len(backup_results),
                "backup_results": backup_results,
                "rollback_script": rollback_script,
                "backup_directory": str(self.backup_dir)
            }
            
        finally:
            await self.disconnect()

async def main():
    """Main function"""
    # Get database URL from environment or use default
    database_url = os.getenv(
        "DATABASE_URL", 
        "sqlite+aiosqlite:///./data/pygent_factory.db"
    )
    
    print(f"🔗 Using database: {database_url.split('///')[-1] if '///' in database_url else database_url}")
    
    try:
        backup_manager = A2ABackupManager(database_url)
        results = await backup_manager.run_backup_strategy()
        
        # Save backup report
        report_file = "a2a_backup_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("📊 A2A BACKUP STRATEGY COMPLETE")
        print("=" * 60)
        print(f"📄 Report saved to: {report_file}")
        print(f"📊 Successful backups: {results['backups_created']}/{results['total_backups']}")
        print(f"📁 Backup directory: {results['backup_directory']}")
        print(f"🔄 Rollback script: {results['rollback_script']}")
        
        if results['backups_created'] == results['total_backups']:
            print("\n✅ All backups completed successfully!")
        else:
            print(f"\n⚠️ {results['total_backups'] - results['backups_created']} backups failed")
        
        return results['success']
        
    except Exception as e:
        print(f"\n❌ Backup strategy failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
