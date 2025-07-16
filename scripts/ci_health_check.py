#!/usr/bin/env python3
"""
CI/CD Health Check Script
Validates essential components before running tests
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python {version.major}.{version.minor} not supported. Requires Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_essential_files():
    """Check for essential files"""
    essential_files = [
        "requirements.txt",
        "requirements-dev.txt", 
        "main.py",
        "src/__init__.py",
        "pytest.ini",
        ".env.example"
    ]
    
    missing_files = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing essential files: {missing_files}")
        return False
    
    return True

def check_import_paths():
    """Check critical import paths"""
    try:
        sys.path.append('src')
        
        # Test critical imports
        import core
        print("✅ Core module imports")
        
        import orchestration
        print("✅ Orchestration module imports")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_environment():
    """Check environment variables"""
    required_env = {
        'PYTHONIOENCODING': 'utf-8',
        'LANG': 'en_US.UTF-8',
        'LC_ALL': 'en_US.UTF-8'
    }
    
    for key, expected in required_env.items():
        actual = os.environ.get(key)
        if actual != expected:
            print(f"⚠️  {key}={actual} (expected {expected})")
        else:
            print(f"✅ {key}={actual}")
    
    return True

def main():
    """Run all health checks"""
    print("🔍 CI/CD Health Check Starting...")
    
    checks = [
        ("Python Version", check_python_version),
        ("Essential Files", check_essential_files),
        ("Import Paths", check_import_paths),
        ("Environment", check_environment)
    ]
    
    failed_checks = []
    
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        try:
            if not check_func():
                failed_checks.append(name)
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            failed_checks.append(name)
    
    print(f"\n📊 Health Check Summary:")
    print(f"✅ Passed: {len(checks) - len(failed_checks)}")
    print(f"❌ Failed: {len(failed_checks)}")
    
    if failed_checks:
        print(f"Failed checks: {failed_checks}")
        sys.exit(1)
    else:
        print("🎉 All health checks passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
