#!/usr/bin/env python3
"""Test orchestrator component imports systematically"""

import sys
import traceback

def test_component_import(component_name):
    """Test import of a specific component"""
    try:
        print(f"Testing {component_name}...")
        
        if component_name == "documentation_models":
            from src.orchestration.documentation_models import DocumentationConfig
            config = DocumentationConfig()
            print(f"✅ {component_name}: Config created with path {config.docs_source_path}")
            
        elif component_name == "build_coordinator":
            from src.orchestration.build_coordinator import BuildCoordinator
            print(f"✅ {component_name}: Class imported successfully")
            
        elif component_name == "conflict_resolver":
            from src.orchestration.conflict_resolver import ConflictResolver
            print(f"✅ {component_name}: Class imported successfully")
            
        elif component_name == "sync_manager":
            from src.orchestration.sync_manager import SyncManager
            print(f"✅ {component_name}: Class imported successfully")
            
        elif component_name == "health_monitor":
            from src.orchestration.health_monitor import HealthMonitor
            print(f"✅ {component_name}: Class imported successfully")
            
        elif component_name == "documentation_orchestrator":
            from src.orchestration.documentation_orchestrator import DocumentationOrchestrator
            print(f"✅ {component_name}: Class imported successfully")
            
        return True
        
    except Exception as e:
        print(f"❌ {component_name}: Import failed")
        print(f"   Error: {e}")
        if "No module named" in str(e):
            print(f"   Issue: Module not found")
        elif "cannot import name" in str(e):
            print(f"   Issue: Import name error")
        else:
            print(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    """Test all components"""
    print("🔍 SYSTEMATIC COMPONENT IMPORT TESTING")
    print("=" * 50)
    
    components = [
        "documentation_models",
        "build_coordinator", 
        "conflict_resolver",
        "sync_manager",
        "health_monitor",
        "documentation_orchestrator"
    ]
    
    results = {}
    for component in components:
        success = test_component_import(component)
        results[component] = success
        print()
    
    print("=" * 50)
    print("📊 IMPORT TEST SUMMARY:")
    print("=" * 50)
    
    passed = sum(results.values())
    failed = len(results) - passed
    
    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {component}")
    
    print(f"\nRESULT: {passed}/{len(results)} components imported successfully")
    
    if failed == 0:
        print("🎉 ALL IMPORTS SUCCESSFUL - READY FOR INTEGRATION TESTING")
    else:
        print("🚨 SOME IMPORTS FAILED - NEED TO FIX BEFORE PROCEEDING")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
