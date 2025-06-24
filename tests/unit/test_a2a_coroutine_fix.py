#!/usr/bin/env python3
"""
Test A2A Coroutine Fix

Quick test to verify the A2A discovery coroutine warning is fixed.
"""

import sys
import os
import warnings
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_a2a_discovery_no_warnings():
    """Test that A2A discovery doesn't create coroutine warnings"""
    print("🔍 Testing A2A Discovery Coroutine Fix")
    print("=" * 50)
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # Import and create A2A discovery
            from a2a_protocol.discovery import A2AAgentDiscovery
            
            print("✅ A2AAgentDiscovery imported successfully")
            
            # Create discovery instance (this should not create warnings)
            discovery = A2AAgentDiscovery()
            print("✅ A2AAgentDiscovery instance created")
            
            # Check for RuntimeWarnings about unawaited coroutines
            coroutine_warnings = [warning for warning in w 
                                if issubclass(warning.category, RuntimeWarning) 
                                and "coroutine" in str(warning.message)]
            
            if coroutine_warnings:
                print(f"❌ Found {len(coroutine_warnings)} coroutine warnings:")
                for warning in coroutine_warnings:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ No coroutine warnings detected")
                return True
                
        except Exception as e:
            print(f"❌ Error during test: {e}")
            return False

def test_a2a_components_no_warnings():
    """Test that A2A components don't create warnings"""
    print("\n🔍 Testing A2A Components Initialization")
    print("=" * 50)
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            # Import A2A components
            from a2a_protocol.transport import A2ATransportLayer
            from a2a_protocol.task_manager import A2ATaskManager
            from a2a_protocol.security import A2ASecurityManager
            
            print("✅ A2A components imported successfully")
            
            # Create instances
            transport = A2ATransportLayer()
            task_manager = A2ATaskManager()
            security = A2ASecurityManager()
            
            print("✅ A2A component instances created")
            
            # Check for RuntimeWarnings about unawaited coroutines
            coroutine_warnings = [warning for warning in w 
                                if issubclass(warning.category, RuntimeWarning) 
                                and "coroutine" in str(warning.message)]
            
            if coroutine_warnings:
                print(f"❌ Found {len(coroutine_warnings)} coroutine warnings:")
                for warning in coroutine_warnings:
                    print(f"   - {warning.message}")
                return False
            else:
                print("✅ No coroutine warnings detected")
                return True
                
        except Exception as e:
            print(f"❌ Error during test: {e}")
            return False

def main():
    """Run all tests"""
    print("🚀 Testing A2A Coroutine Warning Fixes")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 2
    
    # Test 1: A2A Discovery
    if test_a2a_discovery_no_warnings():
        tests_passed += 1
    
    # Test 2: A2A Components
    if test_a2a_components_no_warnings():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{tests_total} passed")
    success_rate = (tests_passed / tests_total) * 100 if tests_total > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if tests_passed == tests_total:
        print("🎉 ALL COROUTINE WARNING FIXES WORKING!")
        return True
    else:
        print("❌ Some coroutine warnings still present.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
