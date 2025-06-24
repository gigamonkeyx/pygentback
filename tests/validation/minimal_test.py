#!/usr/bin/env python3
"""Minimal test to isolate import issues"""

import sys
print("Python version:", sys.version)
print("Python path:", sys.path[:3])

try:
    print("Testing basic imports...")
    from pathlib import Path
    from dataclasses import dataclass
    from enum import Enum
    print("✅ Basic imports successful")
    
    print("Testing documentation models...")
    from src.orchestration.documentation_models import DocumentationConfig
    print("✅ DocumentationConfig imported")
    
    config = DocumentationConfig()
    print(f"✅ Config created: {config.docs_source_path}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
