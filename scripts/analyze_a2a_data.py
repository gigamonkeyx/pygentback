#!/usr/bin/env python3
"""
A2A Data Analysis Script

Analyzes existing A2A data in separate tables to prepare for migration
to unified database schema.

This script:
1. Connects to the database
2. Queries a2a_tasks and a2a_agents tables
3. Generates data mapping report
4. Identifies conflicts and inconsistencies
5. Provides migration recommendations
"""

import asyncio
import aiosqlite
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

@dataclass
class A2ADataAnalysis:
    """Analysis results for A2A data migration"""
    timestamp: str
    a2a_agents_count: int
    a2a_tasks_count: int
    agents_with_a2a_fields: int
    tasks_with_a2a_fields: int
    potential_conflicts: List[Dict[str, Any]]
    data_quality_issues: List[Dict[str, Any]]
    migration_recommendations: List[str]

class A2ADataAnalyzer:
    """Analyzes A2A data for migration planning"""

    def __init__(self, database_url: str):
        self.database_url = database_url
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
    
    async def check_table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            cursor = await self.connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?
            """, (table_name,))
            result = await cursor.fetchone()
            return result is not None
        except Exception as e:
            print(f"❌ Error checking table {table_name}: {e}")
            return False
    
    async def analyze_a2a_agents_table(self) -> Dict[str, Any]:
        """Analyze a2a_agents table"""
        print("\n🔍 Analyzing a2a_agents table...")
        
        if not await self.check_table_exists('a2a_agents'):
            print("⚠️ a2a_agents table does not exist")
            return {"exists": False, "count": 0, "sample_data": []}
        
        try:
            # Get count
            cursor = await self.connection.execute("SELECT COUNT(*) FROM a2a_agents")
            count_result = await cursor.fetchone()
            count = count_result[0] if count_result else 0
            print(f"   📊 Total a2a_agents records: {count}")

            # Get sample data
            sample_data = []
            if count > 0:
                cursor = await self.connection.execute("""
                    SELECT id, name, url, agent_card, registered_at, last_seen
                    FROM a2a_agents
                    LIMIT 5
                """)
                rows = await cursor.fetchall()

                for row in rows:
                    sample_data.append({
                        "id": str(row['id']),
                        "name": row['name'],
                        "url": row['url'],
                        "agent_card_size": len(str(row['agent_card'])) if row['agent_card'] else 0,
                        "registered_at": row['registered_at'] if row['registered_at'] else None,
                        "last_seen": row['last_seen'] if row['last_seen'] else None
                    })

                print(f"   📋 Sample data collected: {len(sample_data)} records")
            
            return {
                "exists": True,
                "count": count,
                "sample_data": sample_data
            }
            
        except Exception as e:
            print(f"❌ Error analyzing a2a_agents table: {e}")
            return {"exists": True, "count": 0, "sample_data": [], "error": str(e)}
    
    async def analyze_a2a_tasks_table(self) -> Dict[str, Any]:
        """Analyze a2a_tasks table"""
        print("\n🔍 Analyzing a2a_tasks table...")
        
        if not await self.check_table_exists('a2a_tasks'):
            print("⚠️ a2a_tasks table does not exist")
            return {"exists": False, "count": 0, "sample_data": []}
        
        try:
            # Get count
            cursor = await self.connection.execute("SELECT COUNT(*) FROM a2a_tasks")
            count_result = await cursor.fetchone()
            count = count_result[0] if count_result else 0
            print(f"   📊 Total a2a_tasks records: {count}")

            # Get sample data
            sample_data = []
            if count > 0:
                cursor = await self.connection.execute("""
                    SELECT id, session_id, status, agent_url, created_at, updated_at, task_data
                    FROM a2a_tasks
                    LIMIT 5
                """)
                rows = await cursor.fetchall()

                for row in rows:
                    sample_data.append({
                        "id": str(row['id']),
                        "session_id": str(row['session_id']) if row['session_id'] else None,
                        "status": row['status'],
                        "agent_url": row['agent_url'],
                        "created_at": row['created_at'] if row['created_at'] else None,
                        "updated_at": row['updated_at'] if row['updated_at'] else None,
                        "task_data_size": len(str(row['task_data'])) if row['task_data'] else 0
                    })

                print(f"   📋 Sample data collected: {len(sample_data)} records")
            
            return {
                "exists": True,
                "count": count,
                "sample_data": sample_data
            }
            
        except Exception as e:
            print(f"❌ Error analyzing a2a_tasks table: {e}")
            return {"exists": True, "count": 0, "sample_data": [], "error": str(e)}
    
    async def analyze_main_tables_a2a_fields(self) -> Dict[str, Any]:
        """Analyze existing A2A fields in main tables"""
        print("\n🔍 Analyzing A2A fields in main tables...")
        
        results = {}
        
        # Check agents table
        if await self.check_table_exists('agents'):
            try:
                # Check for A2A fields
                cursor = await self.connection.execute("""
                    SELECT COUNT(*) FROM agents
                    WHERE a2a_url IS NOT NULL OR a2a_agent_card IS NOT NULL
                """)
                agents_with_a2a_result = await cursor.fetchone()
                agents_with_a2a = agents_with_a2a_result[0] if agents_with_a2a_result else 0

                cursor = await self.connection.execute("SELECT COUNT(*) FROM agents")
                total_agents_result = await cursor.fetchone()
                total_agents = total_agents_result[0] if total_agents_result else 0
                
                results['agents'] = {
                    "total_count": total_agents,
                    "with_a2a_fields": agents_with_a2a,
                    "percentage": (agents_with_a2a / total_agents * 100) if total_agents > 0 else 0
                }
                
                print(f"   📊 Agents with A2A fields: {agents_with_a2a}/{total_agents} ({results['agents']['percentage']:.1f}%)")
                
            except Exception as e:
                print(f"❌ Error analyzing agents table: {e}")
                results['agents'] = {"error": str(e)}
        else:
            print("⚠️ agents table does not exist")
            results['agents'] = {"exists": False}
        
        # Check tasks table
        if await self.check_table_exists('tasks'):
            try:
                # Check for A2A fields
                cursor = await self.connection.execute("""
                    SELECT COUNT(*) FROM tasks
                    WHERE a2a_context_id IS NOT NULL OR a2a_message_history IS NOT NULL
                """)
                tasks_with_a2a_result = await cursor.fetchone()
                tasks_with_a2a = tasks_with_a2a_result[0] if tasks_with_a2a_result else 0

                cursor = await self.connection.execute("SELECT COUNT(*) FROM tasks")
                total_tasks_result = await cursor.fetchone()
                total_tasks = total_tasks_result[0] if total_tasks_result else 0
                
                results['tasks'] = {
                    "total_count": total_tasks,
                    "with_a2a_fields": tasks_with_a2a,
                    "percentage": (tasks_with_a2a / total_tasks * 100) if total_tasks > 0 else 0
                }
                
                print(f"   📊 Tasks with A2A fields: {tasks_with_a2a}/{total_tasks} ({results['tasks']['percentage']:.1f}%)")
                
            except Exception as e:
                print(f"❌ Error analyzing tasks table: {e}")
                results['tasks'] = {"error": str(e)}
        else:
            print("⚠️ tasks table does not exist")
            results['tasks'] = {"exists": False}
        
        return results
    
    async def identify_conflicts(self, a2a_agents_data: Dict, a2a_tasks_data: Dict, main_tables_data: Dict) -> List[Dict[str, Any]]:
        """Identify potential data conflicts"""
        print("\n🔍 Identifying potential conflicts...")
        
        conflicts = []
        
        # Check for agent name conflicts
        if a2a_agents_data.get("exists") and main_tables_data.get("agents", {}).get("total_count", 0) > 0:
            try:
                # Check for agents with same names
                cursor = await self.connection.execute("""
                    SELECT a.name, COUNT(*) as count
                    FROM a2a_agents a
                    WHERE EXISTS (
                        SELECT 1 FROM agents ag WHERE ag.name = a.name
                    )
                    GROUP BY a.name
                    HAVING COUNT(*) > 0
                """)
                conflict_check = await cursor.fetchall()
                
                if conflict_check:
                    conflicts.append({
                        "type": "agent_name_conflict",
                        "description": "Agents with same names in both a2a_agents and agents tables",
                        "count": len(conflict_check),
                        "details": [{"name": row['name'], "count": row['count']} for row in conflict_check]
                    })
                    print(f"   ⚠️ Found {len(conflict_check)} agent name conflicts")
                
            except Exception as e:
                print(f"❌ Error checking agent conflicts: {e}")
        
        # Check for task ID conflicts
        if a2a_tasks_data.get("exists") and main_tables_data.get("tasks", {}).get("total_count", 0) > 0:
            try:
                # Check for tasks with same IDs
                cursor = await self.connection.execute("""
                    SELECT at.id
                    FROM a2a_tasks at
                    WHERE EXISTS (
                        SELECT 1 FROM tasks t WHERE t.id = at.id
                    )
                    LIMIT 10
                """)
                conflict_check = await cursor.fetchall()
                
                if conflict_check:
                    conflicts.append({
                        "type": "task_id_conflict",
                        "description": "Tasks with same IDs in both a2a_tasks and tasks tables",
                        "count": len(conflict_check),
                        "sample_ids": [str(row['id']) for row in conflict_check]
                    })
                    print(f"   ⚠️ Found {len(conflict_check)} task ID conflicts")
                
            except Exception as e:
                print(f"❌ Error checking task conflicts: {e}")
        
        if not conflicts:
            print("   ✅ No conflicts detected")
        
        return conflicts
    
    async def identify_data_quality_issues(self, a2a_agents_data: Dict, a2a_tasks_data: Dict) -> List[Dict[str, Any]]:
        """Identify data quality issues"""
        print("\n🔍 Identifying data quality issues...")
        
        issues = []
        
        # Check a2a_agents data quality
        if a2a_agents_data.get("exists") and a2a_agents_data.get("count", 0) > 0:
            try:
                # Check for missing required fields
                cursor = await self.connection.execute("""
                    SELECT COUNT(*) FROM a2a_agents WHERE name IS NULL OR name = ''
                """)
                missing_names_result = await cursor.fetchone()
                missing_names = missing_names_result[0] if missing_names_result else 0

                if missing_names > 0:
                    issues.append({
                        "type": "missing_agent_names",
                        "description": "A2A agents with missing or empty names",
                        "count": missing_names
                    })

                # Check for invalid URLs
                cursor = await self.connection.execute("""
                    SELECT COUNT(*) FROM a2a_agents
                    WHERE url IS NULL OR url = '' OR url NOT LIKE 'http%'
                """)
                invalid_urls_result = await cursor.fetchone()
                invalid_urls = invalid_urls_result[0] if invalid_urls_result else 0
                
                if invalid_urls > 0:
                    issues.append({
                        "type": "invalid_agent_urls",
                        "description": "A2A agents with invalid or missing URLs",
                        "count": invalid_urls
                    })
                
            except Exception as e:
                print(f"❌ Error checking a2a_agents data quality: {e}")
        
        # Check a2a_tasks data quality
        if a2a_tasks_data.get("exists") and a2a_tasks_data.get("count", 0) > 0:
            try:
                # Check for missing status
                cursor = await self.connection.execute("""
                    SELECT COUNT(*) FROM a2a_tasks WHERE status IS NULL OR status = ''
                """)
                missing_status_result = await cursor.fetchone()
                missing_status = missing_status_result[0] if missing_status_result else 0

                if missing_status > 0:
                    issues.append({
                        "type": "missing_task_status",
                        "description": "A2A tasks with missing or empty status",
                        "count": missing_status
                    })

                # Check for orphaned tasks (no corresponding agent)
                cursor = await self.connection.execute("""
                    SELECT COUNT(*) FROM a2a_tasks at
                    WHERE at.agent_url IS NOT NULL
                    AND NOT EXISTS (
                        SELECT 1 FROM a2a_agents aa WHERE aa.url = at.agent_url
                    )
                """)
                orphaned_tasks_result = await cursor.fetchone()
                orphaned_tasks = orphaned_tasks_result[0] if orphaned_tasks_result else 0
                
                if orphaned_tasks > 0:
                    issues.append({
                        "type": "orphaned_tasks",
                        "description": "A2A tasks referencing non-existent agents",
                        "count": orphaned_tasks
                    })
                
            except Exception as e:
                print(f"❌ Error checking a2a_tasks data quality: {e}")
        
        if issues:
            print(f"   ⚠️ Found {len(issues)} data quality issues")
        else:
            print("   ✅ No data quality issues detected")
        
        return issues
    
    def generate_migration_recommendations(self, 
                                         a2a_agents_data: Dict, 
                                         a2a_tasks_data: Dict, 
                                         main_tables_data: Dict,
                                         conflicts: List[Dict],
                                         issues: List[Dict]) -> List[str]:
        """Generate migration recommendations"""
        print("\n💡 Generating migration recommendations...")
        
        recommendations = []
        
        # Basic migration strategy
        if a2a_agents_data.get("count", 0) > 0:
            recommendations.append(f"Migrate {a2a_agents_data['count']} agents from a2a_agents to agents.a2a_url and agents.a2a_agent_card fields")
        
        if a2a_tasks_data.get("count", 0) > 0:
            recommendations.append(f"Migrate {a2a_tasks_data['count']} tasks from a2a_tasks to tasks.a2a_context_id and tasks.a2a_message_history fields")
        
        # Conflict resolution
        if conflicts:
            recommendations.append("Resolve data conflicts before migration:")
            for conflict in conflicts:
                if conflict["type"] == "agent_name_conflict":
                    recommendations.append(f"  - Handle {conflict['count']} agent name conflicts by using unique identifiers")
                elif conflict["type"] == "task_id_conflict":
                    recommendations.append(f"  - Handle {conflict['count']} task ID conflicts by generating new UUIDs")
        
        # Data quality fixes
        if issues:
            recommendations.append("Fix data quality issues before migration:")
            for issue in issues:
                recommendations.append(f"  - {issue['description']}: {issue['count']} records")
        
        # Migration strategy
        if a2a_agents_data.get("count", 0) > 0 or a2a_tasks_data.get("count", 0) > 0:
            recommendations.extend([
                "Use transaction-based migration to ensure data consistency",
                "Create backup of existing data before migration",
                "Implement rollback procedures in case of migration failure",
                "Test migration on development environment first",
                "Plan for zero-downtime deployment if needed"
            ])
        else:
            recommendations.append("No A2A data found - migration not required, can proceed with schema cleanup")
        
        return recommendations
    
    async def run_analysis(self) -> A2ADataAnalysis:
        """Run complete A2A data analysis"""
        print("🚀 Starting A2A Data Analysis...")
        print("=" * 60)
        
        if not await self.connect():
            raise Exception("Failed to connect to database")
        
        try:
            # Analyze separate A2A tables
            a2a_agents_data = await self.analyze_a2a_agents_table()
            a2a_tasks_data = await self.analyze_a2a_tasks_table()
            
            # Analyze main tables with A2A fields
            main_tables_data = await self.analyze_main_tables_a2a_fields()
            
            # Identify conflicts and issues
            conflicts = await self.identify_conflicts(a2a_agents_data, a2a_tasks_data, main_tables_data)
            issues = await self.identify_data_quality_issues(a2a_agents_data, a2a_tasks_data)
            
            # Generate recommendations
            recommendations = self.generate_migration_recommendations(
                a2a_agents_data, a2a_tasks_data, main_tables_data, conflicts, issues
            )
            
            # Create analysis result
            analysis = A2ADataAnalysis(
                timestamp=datetime.utcnow().isoformat(),
                a2a_agents_count=a2a_agents_data.get("count", 0),
                a2a_tasks_count=a2a_tasks_data.get("count", 0),
                agents_with_a2a_fields=main_tables_data.get("agents", {}).get("with_a2a_fields", 0),
                tasks_with_a2a_fields=main_tables_data.get("tasks", {}).get("with_a2a_fields", 0),
                potential_conflicts=conflicts,
                data_quality_issues=issues,
                migration_recommendations=recommendations
            )
            
            return analysis
            
        finally:
            await self.disconnect()

async def main():
    """Main function"""
    # Get database URL from environment or use default
    database_url = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./data/pygent_factory.db"
    )
    
    print(f"🔗 Using database: {database_url.split('@')[1] if '@' in database_url else database_url}")
    
    try:
        analyzer = A2ADataAnalyzer(database_url)
        analysis = await analyzer.run_analysis()
        
        # Save analysis report
        report_file = "a2a_data_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(analysis), f, indent=2)
        
        print("\n" + "=" * 60)
        print("📊 A2A DATA ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"📄 Report saved to: {report_file}")
        print(f"📊 A2A Agents: {analysis.a2a_agents_count}")
        print(f"📊 A2A Tasks: {analysis.a2a_tasks_count}")
        print(f"📊 Agents with A2A fields: {analysis.agents_with_a2a_fields}")
        print(f"📊 Tasks with A2A fields: {analysis.tasks_with_a2a_fields}")
        print(f"⚠️ Conflicts: {len(analysis.potential_conflicts)}")
        print(f"⚠️ Data quality issues: {len(analysis.data_quality_issues)}")
        print(f"💡 Recommendations: {len(analysis.migration_recommendations)}")
        
        if analysis.migration_recommendations:
            print("\n💡 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(analysis.migration_recommendations[:5], 1):
                print(f"   {i}. {rec}")
            if len(analysis.migration_recommendations) > 5:
                print(f"   ... and {len(analysis.migration_recommendations) - 5} more (see report file)")
        
        print("\n✅ Analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
