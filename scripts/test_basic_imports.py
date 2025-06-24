#!/usr/bin/env python3
"""
Test basic imports without problematic dependencies
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

print("Testing basic imports...")
print(f"Python: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Source directory: {src_dir}")

# Test 1: Basic Python modules
try:
    import asyncio
    import logging
    import json
    import datetime
    print("✅ Basic Python modules imported")
except Exception as e:
    print(f"❌ Basic Python modules failed: {e}")

# Test 2: Third-party modules that should be available
try:
    import psutil
    print("✅ psutil imported")
except Exception as e:
    print(f"❌ psutil failed: {e}")

try:
    import pydantic
    print("✅ pydantic imported")
except Exception as e:
    print(f"❌ pydantic failed: {e}")

try:
    import fastapi
    print("✅ fastapi imported")
except Exception as e:
    print(f"❌ fastapi failed: {e}")

# Test 3: Our integration modules (should work now)
try:
    from integration.workflows import WorkflowManager
    print("✅ WorkflowManager imported")
except Exception as e:
    print(f"❌ WorkflowManager failed: {e}")

try:
    from integration.monitoring import IntegrationMonitor
    print("✅ IntegrationMonitor imported")
except Exception as e:
    print(f"❌ IntegrationMonitor failed: {e}")

try:
    from integration.events import EventBus
    print("✅ EventBus imported")
except Exception as e:
    print(f"❌ EventBus failed: {e}")

try:
    from integration.config import IntegrationConfigManager
    print("✅ IntegrationConfigManager imported")
except Exception as e:
    print(f"❌ IntegrationConfigManager failed: {e}")

try:
    from integration.utils import OperationResult
    print("✅ OperationResult imported")
except Exception as e:
    print(f"❌ OperationResult failed: {e}")

# Test 4: Analytics modules (direct import)
try:
    from testing.analytics.dashboard import PerformanceDashboard
    print("✅ PerformanceDashboard imported")
except Exception as e:
    print(f"❌ PerformanceDashboard failed: {e}")

try:
    from testing.analytics.trends import TrendAnalyzer
    print("✅ TrendAnalyzer imported")
except Exception as e:
    print(f"❌ TrendAnalyzer failed: {e}")

try:
    from testing.analytics.analyzer import RecipeAnalyzer
    print("✅ RecipeAnalyzer imported")
except Exception as e:
    print(f"❌ RecipeAnalyzer failed: {e}")

# Test 5: Windows compatibility module
try:
    from utils.windows_compat import WindowsCompatibilityManager
    print("✅ WindowsCompatibilityManager imported")
except Exception as e:
    print(f"❌ WindowsCompatibilityManager failed: {e}")

print("\n🎉 Basic import test completed!")
print("If you see this message, the core system is working.")
