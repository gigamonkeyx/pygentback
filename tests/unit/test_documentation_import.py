#!/usr/bin/env python3
"""
Test Documentation Orchestrator Import

This script tests the import of our documentation orchestrator components
to validate Phase 1 of our systematic testing approach.
"""

import sys
import traceback

def test_import(module_name, description):
    """Test import of a specific module"""
    try:
        print(f"Testing {description}...")
        exec(f"from {module_name} import *")
        print(f"✅ SUCCESS: {description}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {description}")
        print(f"   Error: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all import tests"""
    print("🔍 PHASE 1: DEPENDENCY & IMPORT VALIDATION")
    print("=" * 50)
    
    tests = [
        ("src.orchestration.documentation_models", "Documentation Models"),
        ("src.orchestration.build_coordinator", "Build Coordinator"),
        ("src.orchestration.conflict_resolver", "Conflict Resolver"),
        ("src.orchestration.sync_manager", "Sync Manager"),
        ("src.orchestration.health_monitor", "Health Monitor"),
        ("src.orchestration.documentation_orchestrator", "Documentation Orchestrator"),
    ]
    
    results = []
    for module, description in tests:
        success = test_import(module, description)
        results.append((module, description, success))
        print()
    
    print("=" * 50)
    print("📊 IMPORT TEST RESULTS:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for module, description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nSUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL IMPORTS SUCCESSFUL - READY FOR PHASE 2")
        return True
    else:
        print("🚨 IMPORT FAILURES DETECTED - MUST FIX BEFORE PROCEEDING")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
