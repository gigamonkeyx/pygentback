#!/usr/bin/env python3
"""
EMERGENCY MOCK AUDIT - Find ALL Mock Bullshit

This script performs a comprehensive audit to find every piece of mock code
that's killing the PyGent Factory project and driving developers away.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def find_mock_bullshit(root_dir: str = "src") -> Dict[str, List[str]]:
    """Find all mock bullshit in the codebase."""
    
    # Patterns that indicate mock/fake/placeholder code
    MOCK_PATTERNS = [
        # Direct mock indicators
        (r'\bclass\s+Mock\w+', "Mock class definition"),
        (r'\bmock_\w+\s*=', "Mock variable assignment"),
        (r'def\s+mock_\w+', "Mock function definition"),
        (r'return\s+.*mock', "Returns mock data"),
        (r'MockBuilder|MockValidator|MockSettings', "Mock infrastructure"),
        
        # Fake/simulation indicators
        (r'# Simulate|# SIMULATE', "Simulation comment"),
        (r'await asyncio\.sleep.*# Simulate', "Fake async delay"),
        (r'f["\'].*mock.*["\']', "Mock string formatting"),
        (r'fake_\w+|dummy_\w+', "Fake/dummy variables"),
        
        # Placeholder indicators
        (r'# TODO:|# FIXME:|# HACK:', "Placeholder comments"),
        (r'placeholder.*implementation', "Placeholder implementation"),
        (r'raise NotImplementedError', "Not implemented"),
        (r'pass\s*#.*placeholder', "Placeholder pass"),
        
        # Test simulation in production
        (r'for i in range.*# Generate fake', "Fake data generation"),
        (r'return \{.*"fake"|"mock"|"test"', "Returns fake data"),
        (r'hardcoded|HARDCODED', "Hardcoded values"),
        
        # Fallback mock implementations
        (r'except.*:\s*return.*mock', "Mock fallback"),
        (r'if.*not.*available.*return.*fake', "Fake fallback"),
    ]
    
    mock_findings = {}
    
    # Scan all Python files in src directory
    for py_file in Path(root_dir).rglob("*.py"):
        # Skip test files and demonstration files
        if any(skip in str(py_file) for skip in ["test_", "_test", "demo", "example"]):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            file_issues = []
            
            for pattern, description in MOCK_PATTERNS:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = content.split('\n')[line_num - 1].strip()
                    
                    file_issues.append(f"Line {line_num}: {description} - {line_content}")
            
            if file_issues:
                mock_findings[str(py_file)] = file_issues
                
        except Exception as e:
            print(f"Error scanning {py_file}: {e}")
    
    return mock_findings

def find_specific_mock_classes(root_dir: str = "src") -> List[Tuple[str, str, int]]:
    """Find specific mock class implementations."""
    mock_classes = []
    
    for py_file in Path(root_dir).rglob("*.py"):
        if any(skip in str(py_file) for skip in ["test_", "_test", "demo"]):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                if re.search(r'class\s+Mock\w+', line, re.IGNORECASE):
                    mock_classes.append((str(py_file), line.strip(), i))
                    
        except Exception as e:
            continue
    
    return mock_classes

def find_fake_data_generation(root_dir: str = "src") -> List[Tuple[str, str, int]]:
    """Find fake data generation code."""
    fake_data = []
    
    FAKE_PATTERNS = [
        r'for i in range.*Document.*query',
        r'mock_docs\s*=\s*\[',
        r'fake_.*=.*\[',
        r'return.*\[.*f["\'].*{i}.*["\']',
        r'Document {i} for query',
        r'f["\']Document about.*["\']',
    ]
    
    for py_file in Path(root_dir).rglob("*.py"):
        if any(skip in str(py_file) for skip in ["test_", "_test", "demo"]):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for pattern in FAKE_PATTERNS:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        fake_data.append((str(py_file), line.strip(), i))
                        
        except Exception as e:
            continue
    
    return fake_data

def main():
    """Run the emergency mock audit."""
    print("🚨 EMERGENCY MOCK AUDIT - FINDING ALL MOCK BULLSHIT")
    print("=" * 70)
    print("🎯 Scanning for mock code that's killing the project...")
    print()
    
    # Find all mock bullshit
    mock_findings = find_mock_bullshit()
    
    if not mock_findings:
        print("✅ NO MOCK BULLSHIT FOUND!")
        print("🎉 Codebase appears to be clean of mock implementations")
        return True
    
    print(f"🚨 FOUND MOCK BULLSHIT IN {len(mock_findings)} FILES:")
    print("=" * 70)
    
    total_issues = 0
    for file_path, issues in mock_findings.items():
        print(f"\n📁 {file_path}")
        print("-" * 50)
        for issue in issues:
            print(f"   ❌ {issue}")
            total_issues += 1
    
    print(f"\n🚨 TOTAL MOCK ISSUES: {total_issues}")
    
    # Find specific mock classes
    print("\n🔍 SPECIFIC MOCK CLASSES:")
    print("-" * 40)
    mock_classes = find_specific_mock_classes()
    if mock_classes:
        for file_path, line, line_num in mock_classes:
            print(f"❌ {file_path}:{line_num} - {line}")
    else:
        print("✅ No mock classes found")
    
    # Find fake data generation
    print("\n🔍 FAKE DATA GENERATION:")
    print("-" * 40)
    fake_data = find_fake_data_generation()
    if fake_data:
        for file_path, line, line_num in fake_data:
            print(f"❌ {file_path}:{line_num} - {line}")
    else:
        print("✅ No fake data generation found")
    
    print("\n" + "=" * 70)
    if total_issues > 0:
        print("🚨 MOCK BULLSHIT DETECTED - PROJECT IS COMPROMISED")
        print("💀 This explains why developers are quitting")
        print("🔥 ALL MOCK CODE MUST BE ELIMINATED IMMEDIATELY")
        return False
    else:
        print("✅ CODEBASE IS CLEAN - NO MOCK BULLSHIT FOUND")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
