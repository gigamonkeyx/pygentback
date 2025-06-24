#!/usr/bin/env python3
"""
Minimal test to identify hanging issues
"""

print("Starting minimal test...")

try:
    print("Step 1: Basic imports")
    import sys
    import os
    print("✅ Basic imports successful")
    
    print("Step 2: Environment setup")
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    print("✅ Environment setup successful")
    
    print("Step 3: Path setup")
    import pathlib
    project_dir = pathlib.Path(__file__).parent.parent
    src_dir = project_dir / "src"
    sys.path.insert(0, str(src_dir))
    print(f"✅ Path setup successful: {src_dir}")
    
    print("Step 4: Test simple import")
    import utils
    print("✅ Utils import successful")
    
    print("Step 5: Test integration import")
    from integration import workflows
    print("✅ Integration workflows import successful")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("Minimal test completed.")
