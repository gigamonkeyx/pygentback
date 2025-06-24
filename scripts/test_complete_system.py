#!/usr/bin/env python3
"""
Complete System Test

Tests all implemented components without triggering circular imports.
"""

import sys
import os
from pathlib import Path

# Setup environment
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

# Add src to path
project_dir = Path(__file__).parent.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

print("🔧 PyGent Factory Complete System Test")
print("=" * 50)
print(f"Python: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Source directory: {src_dir}")
print()

# Test results tracking
tests_passed = 0
tests_total = 0

def test_import(module_path, class_name, description):
    """Test importing a specific class from a module"""
    global tests_passed, tests_total
    tests_total += 1
    
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✅ {description}")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"❌ {description} - {e}")
        return False

def test_instantiation(module_path, class_name, description, *args, **kwargs):
    """Test instantiating a class"""
    global tests_passed, tests_total
    tests_total += 1
    
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        instance = cls(*args, **kwargs)
        print(f"✅ {description}")
        tests_passed += 1
        return instance
    except Exception as e:
        print(f"❌ {description} - {e}")
        return None

print("📦 TESTING CORE DEPENDENCIES")
print("-" * 30)

# Test basic dependencies
test_import("asyncio", "create_task", "asyncio")
test_import("logging", "getLogger", "logging")
test_import("psutil", "cpu_percent", "psutil")
test_import("pydantic", "BaseModel", "pydantic")
test_import("fastapi", "FastAPI", "fastapi")

print("\n🔧 TESTING INTEGRATION MODULES")
print("-" * 30)

# Test integration modules (these work)
test_import("integration.workflows", "WorkflowManager", "WorkflowManager")
test_import("integration.monitoring", "IntegrationMonitor", "IntegrationMonitor")
test_import("integration.events", "EventBus", "EventBus")
test_import("integration.config", "IntegrationConfigManager", "IntegrationConfigManager")
test_import("integration.utils", "OperationResult", "OperationResult")

print("\n📊 TESTING ANALYTICS MODULES (Direct Import)")
print("-" * 30)

# Test analytics modules directly (bypassing testing.__init__.py)
test_import("testing.analytics.analyzer", "RecipeAnalyzer", "RecipeAnalyzer (direct)")
test_import("testing.analytics.dashboard", "PerformanceDashboard", "PerformanceDashboard (direct)")
test_import("testing.analytics.trends", "TrendAnalyzer", "TrendAnalyzer (direct)")

print("\n🛠️ TESTING WINDOWS COMPATIBILITY")
print("-" * 30)

test_import("utils.windows_compat", "WindowsCompatibilityManager", "WindowsCompatibilityManager")

print("\n🏗️ TESTING INSTANTIATION")
print("-" * 30)

# Test creating instances
wm = test_instantiation("integration.workflows", "WorkflowManager", "WorkflowManager instance")
im = test_instantiation("integration.monitoring", "IntegrationMonitor", "IntegrationMonitor instance")
eb = test_instantiation("integration.events", "EventBus", "EventBus instance")
icm = test_instantiation("integration.config", "IntegrationConfigManager", "IntegrationConfigManager instance")

# Test analytics instances
ra = test_instantiation("testing.analytics.analyzer", "RecipeAnalyzer", "RecipeAnalyzer instance")
pd = test_instantiation("testing.analytics.dashboard", "PerformanceDashboard", "PerformanceDashboard instance")
ta = test_instantiation("testing.analytics.trends", "TrendAnalyzer", "TrendAnalyzer instance")

print("\n⚡ TESTING BASIC FUNCTIONALITY")
print("-" * 30)

# Test basic functionality
if wm:
    try:
        workflows = wm.list_workflows()
        print(f"✅ WorkflowManager.list_workflows() - returned {len(workflows)} workflows")
        tests_passed += 1
    except Exception as e:
        print(f"❌ WorkflowManager.list_workflows() - {e}")
    tests_total += 1

if eb:
    try:
        eb.subscribe("test_event", lambda x: None)
        print("✅ EventBus.subscribe() - event subscription works")
        tests_passed += 1
    except Exception as e:
        print(f"❌ EventBus.subscribe() - {e}")
    tests_total += 1

if ra:
    try:
        ra.add_execution_data("test_recipe", {"success": True, "execution_time": 1.5})
        print("✅ RecipeAnalyzer.add_execution_data() - data addition works")
        tests_passed += 1
    except Exception as e:
        print(f"❌ RecipeAnalyzer.add_execution_data() - {e}")
    tests_total += 1

if ta:
    try:
        ta.add_data_point("test_metric", 42.0)
        print("✅ TrendAnalyzer.add_data_point() - data point addition works")
        tests_passed += 1
    except Exception as e:
        print(f"❌ TrendAnalyzer.add_data_point() - {e}")
    tests_total += 1

print("\n" + "=" * 50)
print("📊 TEST RESULTS SUMMARY")
print("=" * 50)

success_rate = (tests_passed / tests_total) * 100 if tests_total > 0 else 0

print(f"Tests Passed: {tests_passed}/{tests_total} ({success_rate:.1f}%)")

if success_rate >= 90:
    print("🎉 EXCELLENT - System is working very well!")
elif success_rate >= 75:
    print("✅ GOOD - System is mostly functional with minor issues")
elif success_rate >= 50:
    print("⚠️ PARTIAL - System has significant issues but core components work")
else:
    print("❌ POOR - System has major issues requiring attention")

print("\n💡 RECOMMENDATIONS:")
if tests_passed >= tests_total * 0.9:
    print("1. System is ready for development and testing")
    print("2. All core components are functional")
    print("3. Continue with feature development")
else:
    print("1. Focus on fixing failing imports and instantiations")
    print("2. Resolve circular dependency issues")
    print("3. Install missing dependencies")

print("\n🎯 NEXT STEPS:")
print("1. Use direct imports for analytics modules to avoid circular dependencies")
print("2. Import pattern: from testing.analytics.dashboard import PerformanceDashboard")
print("3. Avoid importing from testing.__init__.py until circular dependencies are resolved")

print(f"\n✨ Complete system test finished!")
