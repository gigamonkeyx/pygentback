#!/usr/bin/env python3
"""
Simple A2A Integration Validation

Validates the A2A integration by checking files and code structure.
"""

import os
import sys
from pathlib import Path

def validate_file_exists(file_path, description):
    """Validate that a file exists"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def validate_file_content(file_path, required_content, description):
    """Validate that a file contains required content"""
    try:
        if not Path(file_path).exists():
            print(f"❌ {description}: {file_path} - FILE NOT FOUND")
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing_content = []
        for item in required_content:
            if item not in content:
                missing_content.append(item)
        
        if missing_content:
            print(f"❌ {description}: Missing content - {missing_content}")
            return False
        else:
            print(f"✅ {description}: All required content present")
            return True
            
    except Exception as e:
        print(f"❌ {description}: Error reading file - {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 A2A Integration Validation")
    print("="*50)
    
    all_tests_passed = True
    
    # Test 1: Check migration script exists and has required content
    print("\n📋 TEST 1: Database Migration Script")
    migration_file = "src/database/migrations/versions/0002_a2a_integration.py"
    migration_required = [
        "def upgrade():",
        "def downgrade():",
        "idx_agents_a2a_url",
        "idx_tasks_a2a_context",
        "A2A Protocol Integration"
    ]
    
    if not validate_file_exists(migration_file, "Migration script"):
        all_tests_passed = False
    elif not validate_file_content(migration_file, migration_required, "Migration content"):
        all_tests_passed = False
    
    # Test 2: Check init-db.sql has been updated
    print("\n📋 TEST 2: Database Initialization Updates")
    init_db_file = "init-db.sql"
    
    if validate_file_exists(init_db_file, "Database initialization script"):
        try:
            with open(init_db_file, 'r', encoding='utf-8') as f:
                init_content = f.read()
            
            # Check that separate A2A tables are removed
            if "CREATE TABLE IF NOT EXISTS a2a_tasks" in init_content:
                print("❌ Separate A2A tables still present in init-db.sql")
                all_tests_passed = False
            else:
                print("✅ Separate A2A tables removed from init-db.sql")
                
            # Check for A2A integration note
            if "A2A Protocol Integration" in init_content:
                print("✅ A2A integration documentation present")
            else:
                print("❌ A2A integration documentation missing")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ Error reading init-db.sql: {e}")
            all_tests_passed = False
    else:
        all_tests_passed = False
    
    # Test 3: Check database models have A2A fields
    print("\n📋 TEST 3: Database Models A2A Integration")
    models_file = "src/database/models.py"
    models_required = [
        "a2a_url = Column(String(512))",
        "a2a_agent_card = Column(JSONB)",
        "a2a_context_id = Column(String(255))",
        "a2a_message_history = Column(JSONB, default=list)",
        "def is_a2a_enabled(self)",
        "def is_a2a_task(self)"
    ]
    
    if not validate_file_exists(models_file, "Database models"):
        all_tests_passed = False
    elif not validate_file_content(models_file, models_required, "A2A model fields"):
        all_tests_passed = False
    
    # Test 4: Check AgentFactory A2A integration
    print("\n📋 TEST 4: AgentFactory A2A Integration")
    factory_file = "src/core/agent_factory.py"
    factory_required = [
        "a2a_manager: Optional['A2AManager'] = None",
        "self.a2a_manager = a2a_manager",
        "async def discover_a2a_agents(self)",
        "async def send_a2a_message(self",
        "async def _register_with_a2a(self",
        "\"a2a_protocol\": self.a2a_manager is not None"
    ]
    
    if not validate_file_exists(factory_file, "Agent factory"):
        all_tests_passed = False
    elif not validate_file_content(factory_file, factory_required, "AgentFactory A2A integration"):
        all_tests_passed = False
    
    # Test 5: Check A2A Manager database integration
    print("\n📋 TEST 5: A2A Manager Database Integration")
    manager_file = "src/a2a_protocol/manager.py"
    manager_required = [
        "from ..database.models import Agent",
        "async with self.database_manager.get_session() as session:",
        "existing_agent.a2a_url = wrapper.agent_url",
        "existing_agent.a2a_agent_card = agent_card_data"
    ]
    
    if not validate_file_exists(manager_file, "A2A manager"):
        all_tests_passed = False
    elif not validate_file_content(manager_file, manager_required, "A2A manager database integration"):
        all_tests_passed = False
    
    # Test 6: Check that separate A2A tables are not created
    print("\n📋 TEST 6: Separate A2A Tables Removal")
    if validate_file_exists(manager_file, "A2A manager for table check"):
        try:
            with open(manager_file, 'r', encoding='utf-8') as f:
                manager_content = f.read()
            
            # Check that separate table creation is removed
            if "CREATE TABLE IF NOT EXISTS a2a_agents" in manager_content:
                print("❌ A2A manager still creates separate a2a_agents table")
                all_tests_passed = False
            else:
                print("✅ A2A manager no longer creates separate a2a_agents table")
                
            if "CREATE TABLE IF NOT EXISTS a2a_tasks" in manager_content:
                print("❌ A2A manager still creates separate a2a_tasks table")
                all_tests_passed = False
            else:
                print("✅ A2A manager no longer creates separate a2a_tasks table")
                
        except Exception as e:
            print(f"❌ Error checking A2A manager: {e}")
            all_tests_passed = False
    
    # Print final summary
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    if all_tests_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ A2A database integration appears to be correctly implemented")
        print("✅ Database migration script is ready")
        print("✅ AgentFactory has A2A integration")
        print("✅ A2A Manager uses main database models")
        print("✅ Separate A2A tables have been removed")
        print("\n🚀 READY TO PROCEED WITH PHASE 3: ORCHESTRATION MANAGER INTEGRATION")
        return True
    else:
        print("⚠️  SOME VALIDATION TESTS FAILED")
        print("🔧 Please review and fix the issues above before proceeding")
        print("❌ NOT READY for Phase 3")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
