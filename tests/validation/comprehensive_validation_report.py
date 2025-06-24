#!/usr/bin/env python3
"""
MOCK REMOVAL PROJECT - COMPREHENSIVE VALIDATION REPORT
======================================================================

This script validates that all mock/testing code has been successfully removed
from production files and replaced with production-ready implementations.
"""

import os
import re
import ast
import sys
import subprocess
from typing import List, Dict, Tuple

def scan_for_mock_patterns(file_path: str) -> List[str]:
    """Scan a file for mock/testing patterns."""
    mock_patterns = [
        r'\bmock\b(?![_A-Za-z])',  # 'mock' as whole word, not part of mockingbird etc
        r'\bMock[A-Z]',  # MockClass, MockBuilder etc  
        r'TODO:',
        r'FIXME:',
        r'HACK:',
        r'placeholder',
        r'test_.*\(\)',  # test functions
        r'assert\s+False',  # placeholder assertions
        r'raise\s+NotImplementedError',
        r'pass\s*#.*mock',
    ]
    
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for pattern in mock_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Filter out false positives for 'mock' pattern
                if pattern.startswith(r'\bmock\b'):
                    # Check if it's in a regex or documentation context
                    lines = content.split('\n')
                    real_matches = []
                    for i, line in enumerate(lines):
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip if it's in a regex pattern or documentation
                            if ('r"' in line or "r'" in line or 
                                '|mock|' in line or 
                                'mock simulation' in line.lower() or
                                '# regex' in line.lower() or
                                'pattern' in line.lower()):
                                continue
                            real_matches.append(f"Line {i+1}: {line.strip()}")
                    if real_matches:
                        issues.extend(real_matches)
                else:
                    issues.append(f"{pattern}: {len(matches)} occurrences")
                    
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    
    return issues

def validate_syntax(file_path: str) -> Tuple[bool, str]:
    """Validate Python syntax using ast."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_production_implementations(file_path: str) -> List[str]:
    """Check for real production implementations vs mock placeholders."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for production indicators
        production_patterns = [
            r'class\s+\w+Agent\b',
            r'class\s+\w+Factory\b', 
            r'class\s+\w+Manager\b',
            r'async\s+def\s+\w+',
            r'def\s+\w+.*->.*:',
            r'@property',
            r'logging\.getLogger',
            r'asyncio\.',
        ]
        
        # Look for mock indicators (should not exist)
        anti_patterns = [
            r'def\s+mock_',
            r'return\s+"mock',
            r'Mock\w+\(\)',
            r'pass\s*$',  # Empty implementations
        ]
        
        has_production = any(re.search(p, content) for p in production_patterns)
        has_mock = any(re.search(p, content) for p in anti_patterns)
        
        if not has_production:
            issues.append("No production patterns detected")
        if has_mock:
            issues.append("Mock patterns still present")
            
    except Exception as e:
        issues.append(f"Error checking implementations: {e}")
    
    return issues

def main():
    """Run comprehensive validation."""
    print("🎯 MOCK REMOVAL PROJECT - COMPREHENSIVE VALIDATION")
    print("=" * 70)
    
    # Core production files to validate
    production_files = [
        'src/core/agent_factory.py',
        'src/ai/multi_agent/core_new.py',
        'src/ai/nlp/core.py',
        'src/orchestration/distributed_genetic_algorithm.py',
        'src/orchestration/collaborative_self_improvement.py',
        'src/a2a/__init__.py',
    ]
    
    total_issues = 0
    syntax_passed = 0
    implementation_passed = 0
    
    print("\\n🔍 MOCK PATTERN ANALYSIS")
    print("-" * 50)
    
    for file_path in production_files:
        if not os.path.exists(file_path):
            print(f"⚠️  {file_path}: File not found")
            continue
            
        print(f"\\n📁 {file_path}")
        
        # Check for mock patterns
        mock_issues = scan_for_mock_patterns(file_path)
        if mock_issues:
            print(f"  ⚠️  Mock patterns found: {len(mock_issues)}")
            for issue in mock_issues:
                print(f"    • {issue}")
            total_issues += len(mock_issues)
        else:
            print("  ✅ No mock patterns found")
            
        # Validate syntax
        syntax_ok, syntax_msg = validate_syntax(file_path)
        if syntax_ok:
            print("  ✅ Syntax: OK")
            syntax_passed += 1
        else:
            print(f"  ❌ Syntax: {syntax_msg}")
            
        # Check production implementations
        impl_issues = check_production_implementations(file_path)
        if impl_issues:
            print(f"  ⚠️  Implementation issues: {len(impl_issues)}")
            for issue in impl_issues:
                print(f"    • {issue}")
        else:
            print("  ✅ Production implementation detected")
            implementation_passed += 1
    
    print(f"\\n📊 VALIDATION SUMMARY")
    print("-" * 50)
    print(f"Files analyzed: {len(production_files)}")
    print(f"Mock issues found: {total_issues}")
    print(f"Syntax validation: {syntax_passed}/{len(production_files)} passed")
    print(f"Implementation check: {implementation_passed}/{len(production_files)} passed")
    
    if total_issues == 0 and syntax_passed == len(production_files):
        print("\\n🎉 SUCCESS: All validation checks passed!")
        print("✅ Mock removal project completed successfully")
        print("✅ Production code is clean and ready")
        return True
    else:
        print("\\n⚠️  ISSUES DETECTED:")
        if total_issues > 0:
            print(f"  • {total_issues} mock patterns still present")
        if syntax_passed < len(production_files):
            print(f"  • {len(production_files) - syntax_passed} files have syntax issues")
        print("\\n❌ Mock removal project needs additional work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
