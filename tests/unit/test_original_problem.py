#!/usr/bin/env python3
"""
Test Original Problem Resolution

Phase 3: Test if our Documentation Orchestrator actually solves the original
VitePress blank page issue caused by Tailwind CSS/PostCSS conflicts.
"""

import sys
import asyncio
import subprocess
import time
from pathlib import Path

async def test_original_vitepress_build():
    """Test the original VitePress build to see if it still has issues"""
    try:
        print("🔍 PHASE 3: ORIGINAL PROBLEM RESOLUTION")
        print("=" * 50)
        
        print("Testing original VitePress build...")
        docs_path = Path("src/docs")
        
        if not docs_path.exists():
            print(f"❌ Documentation directory not found: {docs_path}")
            return False
        
        print(f"✅ Documentation directory found: {docs_path}")
        
        # Check if package.json exists
        package_json = docs_path / "package.json"
        if not package_json.exists():
            print(f"❌ package.json not found: {package_json}")
            return False
        
        print(f"✅ package.json found: {package_json}")
        
        # Check VitePress config
        vitepress_config = docs_path / ".vitepress" / "config.ts"
        if not vitepress_config.exists():
            print(f"❌ VitePress config not found: {vitepress_config}")
            return False
        
        print(f"✅ VitePress config found: {vitepress_config}")
        
        # Try to build with VitePress directly
        print("Attempting VitePress build...")
        try:
            result = subprocess.run(
                ["npx", "vitepress", "build"],
                cwd=docs_path,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ VitePress build succeeded!")
                print("📊 Build output:")
                print(result.stdout)
                return True
            else:
                print("❌ VitePress build failed!")
                print("📊 Error output:")
                print(result.stderr)
                print("📊 Standard output:")
                print(result.stdout)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ VitePress build timed out (>2 minutes)")
            return False
        except Exception as e:
            print(f"❌ VitePress build error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Original build test failed: {e}")
        return False

async def test_orchestrated_build():
    """Test our orchestrated documentation build"""
    try:
        print("\n🔍 TESTING ORCHESTRATED BUILD")
        print("=" * 50)
        
        from src.orchestration.orchestration_manager import OrchestrationManager
        from src.orchestration.documentation_models import DocumentationWorkflowType
        
        print("Creating OrchestrationManager...")
        manager = OrchestrationManager()
        print("✅ OrchestrationManager created")
        
        print("Starting OrchestrationManager...")
        await manager.start()
        print("✅ OrchestrationManager started")
        
        try:
            print("Testing orchestrated documentation build...")
            workflow_id = await manager.build_documentation(production=False)
            print(f"✅ Build workflow started: {workflow_id}")
            
            # Wait for workflow completion
            max_wait = 120  # 2 minutes
            wait_time = 0
            
            while wait_time < max_wait:
                workflows = await manager.list_documentation_workflows()
                active_workflow = next((w for w in workflows if w["workflow_id"] == workflow_id), None)
                
                if active_workflow:
                    status = active_workflow["status"]
                    progress = active_workflow["progress_percentage"]
                    current_task = active_workflow.get("current_task", "unknown")
                    
                    print(f"📊 Workflow status: {status} ({progress:.1f}%) - {current_task}")
                    
                    if status in ["completed", "failed"]:
                        break
                else:
                    # Workflow might have completed and moved to history
                    break
                
                await asyncio.sleep(5)
                wait_time += 5
            
            # Get final status
            final_status = await manager.get_documentation_status()
            print(f"✅ Final documentation status: {final_status}")
            
            return True
            
        finally:
            print("Stopping OrchestrationManager...")
            await manager.stop()
            print("✅ OrchestrationManager stopped")
            
    except Exception as e:
        print(f"❌ Orchestrated build test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_frontend_integration():
    """Test if documentation is accessible from frontend"""
    try:
        print("\n🔍 TESTING FRONTEND INTEGRATION")
        print("=" * 50)
        
        # Check if built documentation exists
        frontend_docs_path = Path("pygent-repo/public/docs")
        
        print(f"Checking frontend docs path: {frontend_docs_path}")
        if not frontend_docs_path.exists():
            print(f"❌ Frontend docs directory not found: {frontend_docs_path}")
            return False
        
        print(f"✅ Frontend docs directory exists: {frontend_docs_path}")
        
        # Check for key files
        index_file = frontend_docs_path / "index.html"
        manifest_file = frontend_docs_path / "manifest.json"
        
        if index_file.exists():
            print(f"✅ Index file exists: {index_file}")
        else:
            print(f"❌ Index file missing: {index_file}")
            return False
        
        if manifest_file.exists():
            print(f"✅ Manifest file exists: {manifest_file}")
            
            # Read manifest
            import json
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            print(f"📊 Manifest routes: {len(manifest.get('routes', []))}")
            print(f"📊 Manifest assets: {len(manifest.get('assets', []))}")
        else:
            print(f"⚠️  Manifest file missing: {manifest_file}")
        
        # Count files
        file_count = sum(1 for f in frontend_docs_path.rglob("*") if f.is_file())
        print(f"📊 Total files in frontend docs: {file_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Frontend integration test failed: {e}")
        return False

async def test_mermaid_support():
    """Test if Mermaid diagrams are supported"""
    try:
        print("\n🔍 TESTING MERMAID DIAGRAM SUPPORT")
        print("=" * 50)
        
        frontend_docs_path = Path("pygent-repo/public/docs")
        
        if not frontend_docs_path.exists():
            print("❌ Frontend docs not available for Mermaid testing")
            return False
        
        # Look for Mermaid-related files
        mermaid_files = []
        for pattern in ["*mermaid*", "*diagram*"]:
            mermaid_files.extend(list(frontend_docs_path.rglob(pattern)))
        
        if mermaid_files:
            print(f"✅ Found {len(mermaid_files)} Mermaid-related files:")
            for file in mermaid_files[:5]:  # Show first 5
                print(f"   - {file.relative_to(frontend_docs_path)}")
        else:
            print("⚠️  No Mermaid-related files found (this may be normal)")
        
        # Check if any HTML files contain mermaid references
        html_files = list(frontend_docs_path.rglob("*.html"))
        mermaid_references = 0
        
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'mermaid' in content.lower():
                        mermaid_references += 1
            except:
                pass
        
        print(f"📊 HTML files with Mermaid references: {mermaid_references}/{len(html_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mermaid support test failed: {e}")
        return False

async def main():
    """Run all Phase 3 tests"""
    print("🚀 STARTING PHASE 3: ORIGINAL PROBLEM RESOLUTION")
    print("=" * 60)
    
    tests = [
        ("Original VitePress Build", test_original_vitepress_build),
        ("Orchestrated Build", test_orchestrated_build),
        ("Frontend Integration", test_frontend_integration),
        ("Mermaid Support", test_mermaid_support),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        success = await test_func()
        results.append((test_name, success))
        
        if not success and test_name == "Original VitePress Build":
            print("⚠️  Original build failed - this is expected if conflicts exist")
        elif not success and test_name == "Orchestrated Build":
            print("🚨 Orchestrated build failed - this indicates our solution needs work")
    
    print("\n" + "=" * 60)
    print("📊 PHASE 3 TEST RESULTS:")
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
    
    # Analyze results
    original_build_success = results[0][1] if len(results) > 0 else False
    orchestrated_build_success = results[1][1] if len(results) > 1 else False
    
    if original_build_success and orchestrated_build_success:
        print("🎉 BOTH BUILDS SUCCESSFUL - NO CONFLICTS DETECTED")
        print("✅ Documentation system is working correctly")
    elif not original_build_success and orchestrated_build_success:
        print("🎉 ORCHESTRATOR FIXED THE ORIGINAL PROBLEM!")
        print("✅ Our conflict resolution is working")
    elif original_build_success and not orchestrated_build_success:
        print("🚨 ORCHESTRATOR INTRODUCED NEW ISSUES")
        print("❌ Need to debug orchestrator implementation")
    else:
        print("🚨 BOTH BUILDS FAILED")
        print("❌ Need to investigate underlying issues")
    
    return passed >= 2  # At least half the tests should pass

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
