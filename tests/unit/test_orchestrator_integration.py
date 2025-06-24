#!/usr/bin/env python3
"""
Test Documentation Orchestrator Integration

Phase 2: Core Component Testing - Test actual integration with OrchestrationManager
"""

import sys
import asyncio
import traceback

async def test_orchestration_manager_integration():
    """Test DocumentationOrchestrator integration with OrchestrationManager"""
    try:
        print("🔍 PHASE 2: CORE COMPONENT TESTING")
        print("=" * 50)
        
        print("Testing OrchestrationManager import...")
        from src.orchestration.orchestration_manager import OrchestrationManager
        print("✅ OrchestrationManager imported successfully")
        
        print("Testing DocumentationOrchestrator import...")
        from src.orchestration.documentation_orchestrator import DocumentationOrchestrator
        print("✅ DocumentationOrchestrator imported successfully")
        
        print("Creating OrchestrationManager instance...")
        manager = OrchestrationManager()
        print("✅ OrchestrationManager created successfully")
        
        print("Checking if DocumentationOrchestrator is already integrated...")
        if hasattr(manager, 'documentation_orchestrator'):
            print("✅ DocumentationOrchestrator already integrated in OrchestrationManager")
            doc_orchestrator = manager.documentation_orchestrator
            print(f"✅ DocumentationOrchestrator instance: {type(doc_orchestrator)}")
        else:
            print("❌ DocumentationOrchestrator not found in OrchestrationManager")
            return False
        
        print("Testing DocumentationOrchestrator configuration...")
        config = doc_orchestrator.config
        print(f"✅ Config loaded: docs_source_path = {config.docs_source_path}")
        print(f"✅ Config loaded: frontend_docs_path = {config.frontend_docs_path}")
        
        print("Testing component references...")
        print(f"✅ task_dispatcher: {type(doc_orchestrator.task_dispatcher)}")
        print(f"✅ metrics_collector: {type(doc_orchestrator.metrics_collector)}")
        print(f"✅ event_bus: {type(doc_orchestrator.event_bus)}")
        print(f"✅ pygent_integration: {type(doc_orchestrator.pygent_integration)}")
        
        print("Testing sub-components...")
        print(f"✅ build_coordinator: {type(doc_orchestrator.build_coordinator)}")
        print(f"✅ sync_manager: {type(doc_orchestrator.sync_manager)}")
        print(f"✅ conflict_resolver: {type(doc_orchestrator.conflict_resolver)}")
        print(f"✅ health_monitor: {type(doc_orchestrator.health_monitor)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_conflict_detection():
    """Test ConflictResolver.detect_conflicts() on real VitePress setup"""
    try:
        print("\n🔍 TESTING CONFLICT DETECTION")
        print("=" * 50)
        
        from src.orchestration.conflict_resolver import ConflictResolver
        from src.orchestration.documentation_models import DocumentationConfig
        
        print("Creating ConflictResolver...")
        config = DocumentationConfig()
        resolver = ConflictResolver(config, None)  # No event bus for testing
        print("✅ ConflictResolver created")
        
        print(f"Testing conflict detection on: {config.docs_source_path}")
        conflicts = await resolver.detect_conflicts()
        print(f"✅ Conflict detection completed")
        print(f"📊 Found {len(conflicts)} conflicts:")
        
        for conflict in conflicts:
            print(f"   - {conflict.conflict_type.value}: {conflict.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conflict detection test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_build_coordinator():
    """Test BuildCoordinator basic functionality"""
    try:
        print("\n🔍 TESTING BUILD COORDINATOR")
        print("=" * 50)
        
        from src.orchestration.build_coordinator import BuildCoordinator
        from src.orchestration.documentation_models import DocumentationConfig
        
        print("Creating BuildCoordinator...")
        config = DocumentationConfig()
        coordinator = BuildCoordinator(config, None)  # No event bus for testing
        print("✅ BuildCoordinator created")
        
        print("Testing status method...")
        status = await coordinator.get_status()
        print(f"✅ Status retrieved: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ BuildCoordinator test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all Phase 2 tests"""
    print("🚀 STARTING PHASE 2: CORE COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        ("OrchestrationManager Integration", test_orchestration_manager_integration),
        ("Conflict Detection", test_conflict_detection),
        ("Build Coordinator", test_build_coordinator),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        success = await test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("📊 PHASE 2 TEST RESULTS:")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nSUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 PHASE 2 COMPLETE - READY FOR PHASE 3 (ORIGINAL PROBLEM RESOLUTION)")
        return True
    else:
        print("🚨 PHASE 2 FAILURES DETECTED - MUST FIX BEFORE PROCEEDING")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
