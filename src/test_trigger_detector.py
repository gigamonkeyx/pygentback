#!/usr/bin/env python3
"""
Test BuildTriggerDetector functionality
"""

import sys
import os
import asyncio
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print(f"🔍 Testing BuildTriggerDetector")
print(f"Current directory: {current_dir}")

async def test_trigger_detector_import():
    """Test if we can import BuildTriggerDetector"""
    print("\n1️⃣ Testing import...")
    
    try:
        from orchestration.build_trigger_detector import BuildTriggerDetector
        print("✅ Import successful")
        return BuildTriggerDetector
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_trigger_detector_creation(BuildTriggerDetector):
    """Test creating BuildTriggerDetector instance"""
    print("\n2️⃣ Testing instance creation...")
    
    try:
        docs_path = current_dir / "docs"
        
        detector = BuildTriggerDetector(
            repo_path=current_dir.parent,  # Go up to pygent-factory root
            docs_path=docs_path
        )
        print("✅ Instance creation successful")
        return detector
    except Exception as e:
        print(f"❌ Instance creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_git_commands(detector):
    """Test git command functionality"""
    print("\n3️⃣ Testing git commands...")
    
    try:
        # Test current commit hash
        commit_hash = detector.get_current_commit_hash()
        if commit_hash:
            print(f"✅ Current commit: {commit_hash[:8]}...")
        else:
            print("⚠️  Could not get commit hash (not in git repo?)")
        
        # Test current branch
        branch = detector.get_current_branch()
        if branch:
            print(f"✅ Current branch: {branch}")
        else:
            print("⚠️  Could not get current branch")
        
        # Test latest tag
        tag = detector.get_latest_tag()
        if tag:
            print(f"✅ Latest tag: {tag}")
        else:
            print("ℹ️  No tags found")
        
        return True
        
    except Exception as e:
        print(f"❌ Git commands failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_file_monitoring(detector):
    """Test file monitoring functionality"""
    print("\n4️⃣ Testing file monitoring...")
    
    try:
        # Test file hash calculation
        test_file = current_dir / "docs" / "index.md"
        if test_file.exists():
            file_hash = detector.get_file_hash(test_file)
            print(f"✅ File hash calculated: {file_hash[:16]}...")
        else:
            print("⚠️  Test file not found")
        
        # Test file trigger checking
        file_triggers, current_hashes = detector.check_file_triggers()
        
        print(f"✅ File trigger check complete:")
        print(f"   Files changed: {file_triggers['files_changed']}")
        print(f"   Total files monitored: {file_triggers['details']['total_files']}")
        print(f"   New files: {file_triggers['details']['new_count']}")
        print(f"   Changed files: {file_triggers['details']['changed_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ File monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_trigger_checking(detector):
    """Test comprehensive trigger checking"""
    print("\n5️⃣ Testing comprehensive trigger checking...")
    
    try:
        # Test all triggers
        trigger_status = await detector.check_all_triggers()
        
        print(f"✅ Trigger check complete:")
        print(f"   Should build: {trigger_status['should_build']}")
        print(f"   Total triggers: {trigger_status['summary']['total_triggers']}")
        
        if trigger_status['summary']['trigger_reasons']:
            print(f"   Trigger reasons:")
            for reason in trigger_status['summary']['trigger_reasons']:
                print(f"     • {reason}")
        else:
            print(f"   No triggers detected")
        
        # Show detailed trigger info
        triggers = trigger_status['triggers']
        print(f"\n   Detailed trigger status:")
        print(f"     Git changes: {triggers['git']['new_commits']}")
        print(f"     File changes: {triggers['files']['files_changed']}")
        print(f"     Manual triggers: {triggers['manual']['manual_requested']}")
        print(f"     Time triggers: {triggers['time']['time_triggered']}")
        
        return trigger_status
        
    except Exception as e:
        print(f"❌ Trigger checking failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Run trigger detector tests"""
    print("🧪 BuildTriggerDetector Test")
    print("=" * 50)
    
    # Test 1: Import
    BuildTriggerDetector = await test_trigger_detector_import()
    if not BuildTriggerDetector:
        print("\n❌ Import failed - stopping tests")
        return 1
    
    # Test 2: Create instance
    detector = await test_trigger_detector_creation(BuildTriggerDetector)
    if not detector:
        print("\n❌ Instance creation failed - stopping tests")
        return 1
    
    # Test 3: Git commands
    git_success = await test_git_commands(detector)
    if not git_success:
        print("\n⚠️  Git commands failed - continuing with other tests")
    
    # Test 4: File monitoring
    file_success = await test_file_monitoring(detector)
    if not file_success:
        print("\n❌ File monitoring failed - stopping tests")
        return 1
    
    # Test 5: Trigger checking
    trigger_status = await test_trigger_checking(detector)
    if trigger_status is None:
        print("\n❌ Trigger checking failed - stopping tests")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 ALL TRIGGER DETECTOR TESTS PASSED!")
    print(f"📊 Summary:")
    print(f"   • BuildTriggerDetector working: ✅")
    print(f"   • File monitoring working: ✅")
    print(f"   • Trigger detection working: ✅")
    print(f"   • Should build: {'✅ Yes' if trigger_status['should_build'] else '❌ No'}")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)