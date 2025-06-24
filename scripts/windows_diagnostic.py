#!/usr/bin/env python3
"""
Windows Diagnostic Script for PyGent Factory

Comprehensive diagnostic tool to identify and resolve Windows-specific
import hanging and compatibility issues.
"""

import sys
import os
import time
import threading
import signal
import traceback
from pathlib import Path

def timeout_handler():
    """Handle timeout for hanging operations"""
    print("⚠️ TIMEOUT: Operation took too long, likely hanging")
    os._exit(1)

def test_with_timeout(test_func, timeout_seconds=10, test_name="Unknown"):
    """Run a test function with timeout protection"""
    print(f"\n🔍 Testing: {test_name}")
    
    result = {"completed": False, "success": False, "error": None}
    
    def worker():
        try:
            test_func()
            result["success"] = True
            result["completed"] = True
        except Exception as e:
            result["error"] = str(e)
            result["completed"] = True
    
    # Start test in separate thread
    thread = threading.Thread(target=worker, daemon=True)
    start_time = time.time()
    thread.start()
    thread.join(timeout_seconds)
    
    elapsed = time.time() - start_time
    
    if not result["completed"]:
        print(f"❌ {test_name} - HANGING (timeout after {timeout_seconds}s)")
        return False
    elif result["success"]:
        print(f"✅ {test_name} - SUCCESS ({elapsed:.2f}s)")
        return True
    else:
        print(f"❌ {test_name} - ERROR: {result['error']} ({elapsed:.2f}s)")
        return False

def test_basic_python():
    """Test basic Python functionality"""
    import sys
    import os
    import time
    assert sys.version_info >= (3, 8)
    assert os.path.exists(".")
    time.sleep(0.1)  # Small delay to test timing

def test_standard_imports():
    """Test standard library imports"""
    import asyncio
    import logging
    import json
    import datetime
    import pathlib
    import dataclasses
    import typing
    import collections

def test_third_party_imports():
    """Test third-party library imports"""
    import psutil
    import pydantic
    import fastapi
    import uvicorn

def test_psutil_operations():
    """Test psutil operations that commonly hang"""
    import psutil
    
    # Test with very short intervals
    cpu = psutil.cpu_percent(interval=0.01)
    memory = psutil.virtual_memory()
    
    # Test Windows-compatible disk usage
    if os.name == 'nt':
        disk = psutil.disk_usage('C:\\')
    else:
        disk = psutil.disk_usage('/')
    
    # Test process operations
    pids = psutil.pids()[:5]  # Only first 5 to avoid hanging
    
    print(f"   CPU: {cpu}%, Memory: {memory.percent}%, Disk: {disk.percent}%")

def test_src_structure():
    """Test source code structure"""
    src_path = Path("src")
    assert src_path.exists(), "src directory not found"
    
    # Check key directories
    key_dirs = ["integration", "testing", "utils"]
    for dir_name in key_dirs:
        dir_path = src_path / dir_name
        assert dir_path.exists(), f"{dir_name} directory not found"
        
        # Check for __init__.py
        init_file = dir_path / "__init__.py"
        if init_file.exists():
            print(f"   Found: {dir_name}/__init__.py")

def test_simple_src_import():
    """Test simple source imports"""
    sys.path.insert(0, str(Path("src").absolute()))
    
    # Test very basic import
    import utils
    print(f"   utils module: {utils}")

def test_integration_imports():
    """Test integration module imports"""
    sys.path.insert(0, str(Path("src").absolute()))
    
    # Test individual integration modules
    from integration import workflows
    from integration import monitoring
    from integration import events
    
    print(f"   workflows: {workflows}")
    print(f"   monitoring: {monitoring}")
    print(f"   events: {events}")

def test_analytics_imports():
    """Test analytics module imports"""
    sys.path.insert(0, str(Path("src").absolute()))
    
    # Test analytics modules directly
    from testing.analytics import dashboard
    from testing.analytics import trends
    from testing.analytics import analyzer
    
    print(f"   dashboard: {dashboard}")
    print(f"   trends: {trends}")
    print(f"   analyzer: {analyzer}")

def test_class_instantiation():
    """Test class instantiation"""
    sys.path.insert(0, str(Path("src").absolute()))
    
    from integration.workflows import WorkflowManager
    from testing.analytics.dashboard import PerformanceDashboard
    
    # Test instantiation
    wm = WorkflowManager()
    pd = PerformanceDashboard()
    
    print(f"   WorkflowManager: {type(wm)}")
    print(f"   PerformanceDashboard: {type(pd)}")

def main():
    """Main diagnostic function"""
    print("🔧 PyGent Factory Windows Diagnostic Tool")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python: {sys.version}")
    print(f"💻 Platform: {sys.platform}")
    
    # Set environment variables
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # Run diagnostic tests
    tests = [
        (test_basic_python, "Basic Python functionality"),
        (test_standard_imports, "Standard library imports"),
        (test_third_party_imports, "Third-party library imports"),
        (test_psutil_operations, "psutil operations"),
        (test_src_structure, "Source code structure"),
        (test_simple_src_import, "Simple src import"),
        (test_integration_imports, "Integration module imports"),
        (test_analytics_imports, "Analytics module imports"),
        (test_class_instantiation, "Class instantiation"),
    ]
    
    results = []
    for test_func, test_name in tests:
        try:
            success = test_with_timeout(test_func, timeout_seconds=15, test_name=test_name)
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} - EXCEPTION: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System is working correctly!")
    elif passed >= total * 0.7:
        print("⚠️ MOSTLY WORKING - Some issues detected but core functionality works")
    else:
        print("❌ SIGNIFICANT ISSUES - Multiple failures detected")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if passed < total:
        print("1. Use the Windows batch launcher: .\\run_windows.bat")
        print("2. Import modules directly instead of through package __init__.py")
        print("3. Set environment variables before running Python")
        print("4. Use shorter timeouts for psutil operations")
    else:
        print("1. System appears to be working correctly")
        print("2. Continue with normal development workflow")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Diagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Diagnostic failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
