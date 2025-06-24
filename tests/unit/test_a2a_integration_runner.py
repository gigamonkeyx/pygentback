#!/usr/bin/env python3
"""
A2A Integration Test Runner

Simplified test runner that can execute without full database setup
to validate the A2A integration components.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class A2AIntegrationValidator:
    """Validates A2A integration components"""
    
    def __init__(self):
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all A2A integration validation tests"""
        logger.info("🚀 Starting A2A Integration Validation Tests")
        
        # Test 1: Import and Module Structure Validation
        await self.test_import_validation()
        
        # Test 2: Database Migration Script Validation
        await self.test_migration_script_validation()
        
        # Test 3: AgentFactory A2A Integration Validation
        await self.test_agent_factory_integration()
        
        # Test 4: A2A Manager Integration Validation
        await self.test_a2a_manager_integration()
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    async def test_import_validation(self):
        """Test 1: Validate all A2A components can be imported"""
        logger.info("🧪 TEST 1: Import and Module Structure Validation")
        
        test_result = {
            "test_name": "Import Validation",
            "passed": False,
            "details": {},
            "errors": []
        }
        
        try:
            # Test A2A protocol imports
            logger.info("  Testing A2A protocol imports...")
            try:
                from src.a2a_protocol.manager import A2AManager, a2a_manager
                from src.a2a_protocol.protocol import A2AProtocol, a2a_protocol
                test_result["details"]["a2a_protocol"] = "✅ Success"
            except Exception as e:
                test_result["details"]["a2a_protocol"] = f"❌ Failed: {e}"
                test_result["errors"].append(f"A2A protocol import: {e}")
            
            # Test database models imports
            logger.info("  Testing database models imports...")
            try:
                from src.database.models import Agent, Task
                test_result["details"]["database_models"] = "✅ Success"
            except Exception as e:
                test_result["details"]["database_models"] = f"❌ Failed: {e}"
                test_result["errors"].append(f"Database models import: {e}")
            
            # Test agent factory imports
            logger.info("  Testing agent factory imports...")
            try:
                from src.core.agent_factory import AgentFactory
                test_result["details"]["agent_factory"] = "✅ Success"
            except Exception as e:
                test_result["details"]["agent_factory"] = f"❌ Failed: {e}"
                test_result["errors"].append(f"Agent factory import: {e}")
            
            # Test migration script imports
            logger.info("  Testing migration script imports...")
            try:
                from src.database.migrations.versions.a2a_integration_0002 import upgrade, downgrade
                test_result["details"]["migration_script"] = "✅ Success"
            except Exception as e:
                test_result["details"]["migration_script"] = f"❌ Failed: {e}"
                test_result["errors"].append(f"Migration script import: {e}")
            
            # Determine overall success
            test_result["passed"] = len(test_result["errors"]) == 0
            
        except Exception as e:
            test_result["errors"].append(f"General import test failure: {e}")
        
        self.test_results.append(test_result)
        logger.info(f"✅ Import Validation: {'PASSED' if test_result['passed'] else 'FAILED'}")
    
    async def test_migration_script_validation(self):
        """Test 2: Validate migration script structure"""
        logger.info("🧪 TEST 2: Database Migration Script Validation")
        
        test_result = {
            "test_name": "Migration Script Validation",
            "passed": False,
            "details": {},
            "errors": []
        }
        
        try:
            # Check migration file exists
            migration_file = Path("src/database/migrations/versions/0002_a2a_integration.py")
            if migration_file.exists():
                test_result["details"]["file_exists"] = "✅ Migration file exists"
                
                # Read and validate migration content
                with open(migration_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for required components
                required_components = [
                    "def upgrade():",
                    "def downgrade():",
                    "idx_agents_a2a_url",
                    "idx_tasks_a2a_context",
                    "A2A Protocol Integration"
                ]
                
                missing_components = []
                for component in required_components:
                    if component not in content:
                        missing_components.append(component)
                
                if missing_components:
                    test_result["details"]["missing_components"] = f"❌ Missing: {missing_components}"
                    test_result["errors"].append(f"Missing components: {missing_components}")
                else:
                    test_result["details"]["components_check"] = "✅ All required components present"
                
                test_result["details"]["file_size"] = f"📄 {len(content):,} characters"
                
            else:
                test_result["details"]["file_exists"] = "❌ Migration file not found"
                test_result["errors"].append("Migration file not found")
            
            # Check init-db.sql updates
            init_db_file = Path("init-db.sql")
            if init_db_file.exists():
                with open(init_db_file, 'r', encoding='utf-8') as f:
                    init_content = f.read()
                
                # Check that separate A2A tables are removed
                if "CREATE TABLE IF NOT EXISTS a2a_tasks" in init_content:
                    test_result["details"]["init_db_cleanup"] = "❌ Separate A2A tables still present"
                    test_result["errors"].append("Separate A2A tables not removed from init-db.sql")
                else:
                    test_result["details"]["init_db_cleanup"] = "✅ Separate A2A tables removed"
            
            test_result["passed"] = len(test_result["errors"]) == 0
            
        except Exception as e:
            test_result["errors"].append(f"Migration validation error: {e}")
        
        self.test_results.append(test_result)
        logger.info(f"✅ Migration Script Validation: {'PASSED' if test_result['passed'] else 'FAILED'}")
    
    async def test_agent_factory_integration(self):
        """Test 3: Validate AgentFactory A2A integration"""
        logger.info("🧪 TEST 3: AgentFactory A2A Integration Validation")
        
        test_result = {
            "test_name": "AgentFactory A2A Integration",
            "passed": False,
            "details": {},
            "errors": []
        }
        
        try:
            from src.core.agent_factory import AgentFactory
            
            # Check constructor accepts a2a_manager parameter
            import inspect
            constructor_sig = inspect.signature(AgentFactory.__init__)
            if 'a2a_manager' in constructor_sig.parameters:
                test_result["details"]["constructor_parameter"] = "✅ a2a_manager parameter present"
            else:
                test_result["details"]["constructor_parameter"] = "❌ a2a_manager parameter missing"
                test_result["errors"].append("AgentFactory constructor missing a2a_manager parameter")
            
            # Check for A2A methods
            a2a_methods = [
                "discover_a2a_agents",
                "get_a2a_agent_capabilities", 
                "send_a2a_message",
                "coordinate_multi_agent_task",
                "_register_with_a2a",
                "_get_a2a_config"
            ]
            
            missing_methods = []
            for method in a2a_methods:
                if not hasattr(AgentFactory, method):
                    missing_methods.append(method)
            
            if missing_methods:
                test_result["details"]["a2a_methods"] = f"❌ Missing methods: {missing_methods}"
                test_result["errors"].append(f"Missing A2A methods: {missing_methods}")
            else:
                test_result["details"]["a2a_methods"] = "✅ All A2A methods present"
            
            # Check system readiness includes A2A
            try:
                # Create a test instance (without initialization)
                factory = AgentFactory()
                # Check if get_system_readiness method exists
                if hasattr(factory, 'get_system_readiness'):
                    test_result["details"]["system_readiness"] = "✅ System readiness method present"
                else:
                    test_result["details"]["system_readiness"] = "❌ System readiness method missing"
            except Exception as e:
                test_result["details"]["system_readiness"] = f"❌ Error testing: {e}"
            
            test_result["passed"] = len(test_result["errors"]) == 0
            
        except Exception as e:
            test_result["errors"].append(f"AgentFactory integration test error: {e}")
        
        self.test_results.append(test_result)
        logger.info(f"✅ AgentFactory A2A Integration: {'PASSED' if test_result['passed'] else 'FAILED'}")
    
    async def test_a2a_manager_integration(self):
        """Test 4: Validate A2A Manager database integration"""
        logger.info("🧪 TEST 4: A2A Manager Integration Validation")
        
        test_result = {
            "test_name": "A2A Manager Integration",
            "passed": False,
            "details": {},
            "errors": []
        }
        
        try:
            from src.a2a_protocol.manager import A2AManager
            
            # Check A2A manager file for database integration
            manager_file = Path("src/a2a_protocol/manager.py")
            if manager_file.exists():
                with open(manager_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for main model usage
                if "from ..database.models import Agent" in content:
                    test_result["details"]["main_model_import"] = "✅ Uses main Agent model"
                else:
                    test_result["details"]["main_model_import"] = "❌ Missing main model import"
                    test_result["errors"].append("A2A manager doesn't import main Agent model")
                
                # Check that separate table creation is removed
                if "CREATE TABLE IF NOT EXISTS a2a_agents" in content:
                    test_result["details"]["separate_tables"] = "❌ Still creates separate A2A tables"
                    test_result["errors"].append("A2A manager still creates separate tables")
                else:
                    test_result["details"]["separate_tables"] = "✅ No separate table creation"
                
                # Check for SQLAlchemy session usage
                if "get_session" in content:
                    test_result["details"]["sqlalchemy_integration"] = "✅ Uses SQLAlchemy sessions"
                else:
                    test_result["details"]["sqlalchemy_integration"] = "❌ Missing SQLAlchemy integration"
                    test_result["errors"].append("A2A manager missing SQLAlchemy session usage")
            
            test_result["passed"] = len(test_result["errors"]) == 0
            
        except Exception as e:
            test_result["errors"].append(f"A2A manager integration test error: {e}")
        
        self.test_results.append(test_result)
        logger.info(f"✅ A2A Manager Integration: {'PASSED' if test_result['passed'] else 'FAILED'}")
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("🧪 A2A INTEGRATION VALIDATION SUMMARY")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get("passed", False))
        
        for i, result in enumerate(self.test_results, 1):
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            print(f"\nTest {i}: {result.get('test_name', 'Unknown')} - {status}")
            
            if result.get("errors"):
                print(f"  ❌ Errors:")
                for error in result["errors"]:
                    print(f"    • {error}")
            
            if result.get("details"):
                print(f"  📋 Details:")
                for key, value in result["details"].items():
                    print(f"    • {key}: {value}")
        
        print(f"\n📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL VALIDATION TESTS PASSED!")
            print("✅ A2A integration appears to be correctly implemented")
            print("🚀 Ready to proceed with Phase 3: Orchestration Manager Integration")
        else:
            print("⚠️  Some validation tests failed")
            print("🔧 Please review and fix the issues before proceeding")
        
        print("="*80)

async def main():
    """Main test runner"""
    validator = A2AIntegrationValidator()
    results = await validator.run_all_tests()
    
    # Return exit code based on results
    all_passed = all(r.get("passed", False) for r in results)
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
