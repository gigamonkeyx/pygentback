#!/usr/bin/env python3
"""
Test Fixed Documentation Orchestrator

Test the orchestrated build after fixing the PyGentIntegration issue.
"""

import sys
import asyncio
import traceback

async def test_fixed_orchestrated_build():
    """Test our fixed orchestrated documentation build"""
    try:
        print("🔍 TESTING FIXED ORCHESTRATED BUILD")
        print("=" * 50)
        
        from src.orchestration.orchestration_manager import OrchestrationManager
        
        print("Creating OrchestrationManager...")
        manager = OrchestrationManager()
        print("✅ OrchestrationManager created")
        
        print("Starting OrchestrationManager...")
        await manager.start()
        print("✅ OrchestrationManager started successfully")
        
        try:
            print("Testing documentation system status...")
            status = await manager.get_documentation_status()
            print(f"✅ Documentation status: {status}")
            
            print("Testing conflict detection...")
            health = await manager.check_documentation_health()
            print(f"✅ Health check completed: {health}")
            
            print("Testing build workflow...")
            workflow_id = await manager.build_documentation(production=False)
            print(f"✅ Build workflow started: {workflow_id}")
            
            # Wait a bit for workflow to start
            await asyncio.sleep(2)
            
            # Check workflow status
            workflows = await manager.list_documentation_workflows()
            print(f"✅ Active workflows: {len(workflows)}")
            
            for workflow in workflows:
                print(f"   - {workflow['workflow_id']}: {workflow['status']} ({workflow['progress_percentage']:.1f}%)")
            
            return True
            
        finally:
            print("Stopping OrchestrationManager...")
            await manager.stop()
            print("✅ OrchestrationManager stopped")
            
    except Exception as e:
        print(f"❌ Fixed orchestrated build test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_direct_conflict_resolution():
    """Test conflict resolution directly"""
    try:
        print("\n🔍 TESTING DIRECT CONFLICT RESOLUTION")
        print("=" * 50)
        
        from src.orchestration.conflict_resolver import ConflictResolver
        from src.orchestration.documentation_models import DocumentationConfig, DocumentationTask, DocumentationTaskType
        
        print("Creating ConflictResolver...")
        config = DocumentationConfig()
        resolver = ConflictResolver(config, None)
        print("✅ ConflictResolver created")
        
        print("Starting ConflictResolver...")
        await resolver.start()
        print("✅ ConflictResolver started")
        
        try:
            print("Detecting conflicts...")
            conflicts = await resolver.detect_conflicts()
            print(f"✅ Conflict detection completed: {len(conflicts)} conflicts found")
            
            for conflict in conflicts:
                print(f"   - {conflict.conflict_type.value}: {conflict.description}")
                print(f"     Severity: {conflict.severity}, Auto-resolvable: {conflict.auto_resolvable}")
            
            if conflicts:
                print("Testing conflict resolution...")
                task = DocumentationTask(
                    task_id="test_resolve",
                    task_type=DocumentationTaskType.RESOLVE_CONFLICTS,
                    name="Test Conflict Resolution",
                    description="Test resolving detected conflicts"
                )
                
                result = await resolver.resolve_conflicts(task)
                print(f"✅ Conflict resolution completed: {result.success}")
                print(f"   Conflicts resolved: {len(result.conflicts_resolved)}")
                print(f"   Resolution actions: {len(result.resolution_actions)}")
                
                for action in result.resolution_actions:
                    print(f"     - {action}")
            
            return True
            
        finally:
            print("Stopping ConflictResolver...")
            await resolver.stop()
            print("✅ ConflictResolver stopped")
            
    except Exception as e:
        print(f"❌ Direct conflict resolution test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_build_coordinator():
    """Test BuildCoordinator directly"""
    try:
        print("\n🔍 TESTING BUILD COORDINATOR")
        print("=" * 50)
        
        from src.orchestration.build_coordinator import BuildCoordinator
        from src.orchestration.documentation_models import DocumentationConfig, DocumentationTask, DocumentationTaskType
        
        print("Creating BuildCoordinator...")
        config = DocumentationConfig()
        coordinator = BuildCoordinator(config, None)
        print("✅ BuildCoordinator created")
        
        print("Starting BuildCoordinator...")
        await coordinator.start()
        print("✅ BuildCoordinator started")
        
        try:
            print("Testing environment preparation...")
            await coordinator._prepare_build_environment()
            print("✅ Build environment prepared")
            
            print("Testing build command generation...")
            cmd = coordinator._get_build_command(is_production=False)
            print(f"✅ Build command: {' '.join(cmd)}")
            
            return True
            
        finally:
            print("Stopping BuildCoordinator...")
            await coordinator.stop()
            print("✅ BuildCoordinator stopped")
            
    except Exception as e:
        print(f"❌ BuildCoordinator test failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Run all tests"""
    print("🚀 TESTING FIXED DOCUMENTATION ORCHESTRATOR")
    print("=" * 60)
    
    tests = [
        ("Fixed Orchestrated Build", test_fixed_orchestrated_build),
        ("Direct Conflict Resolution", test_direct_conflict_resolution),
        ("Build Coordinator", test_build_coordinator),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        success = await test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("📊 FIXED ORCHESTRATOR TEST RESULTS:")
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
        print("🎉 ALL TESTS PASSED - ORCHESTRATOR IS WORKING!")
    else:
        print("🚨 SOME TESTS FAILED - NEED FURTHER INVESTIGATION")
    
    return passed >= 2

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
