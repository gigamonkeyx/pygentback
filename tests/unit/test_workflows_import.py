#!/usr/bin/env python3
"""
Test script to check if workflows router can be imported
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_workflows_import():
    """Test importing the workflows router"""
    print("🧪 Testing workflows router import...")
    
    try:
        print("   Importing workflows router...")
        from src.api.routes.workflows import router as workflows_router
        print("   ✅ Workflows router imported successfully")
        
        print("   Checking router endpoints...")
        routes = [route.path for route in workflows_router.routes]
        print(f"   ✅ Found {len(routes)} routes:")
        for route in routes:
            print(f"      - {route}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_import():
    """Test importing the orchestrator"""
    print("\n🧪 Testing orchestrator import...")
    
    try:
        print("   Importing orchestrator...")
        from src.workflows.research_analysis_orchestrator import ResearchAnalysisOrchestrator
        print("   ✅ Orchestrator imported successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies_import():
    """Test importing dependencies"""
    print("\n🧪 Testing dependencies import...")
    
    try:
        print("   Importing dependencies...")
        from src.api.dependencies import get_agent_factory
        print("   ✅ Dependencies imported successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 Workflows Import Test")
    print("=" * 30)
    
    tests = [
        ("Dependencies", test_dependencies_import),
        ("Orchestrator", test_orchestrator_import),
        ("Workflows Router", test_workflows_import)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"\n❌ {test_name} test failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All imports working correctly!")
        print("The issue might be in the server configuration or route registration.")
    else:
        print("❌ Import issues found - these need to be fixed first.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
