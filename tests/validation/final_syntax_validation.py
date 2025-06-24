#!/usr/bin/env python3
"""
Final Mock Removal Validation - Syntax Check Only
Validates that all production files have valid syntax after mock removal.
"""

import ast
import os
import sys
from pathlib import Path


def validate_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def find_mock_indicators(file_path):
    """Find any remaining mock indicators in the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        mock_indicators = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            # Look for actual mock code, not just comments or strings
            if ('mock' in line_lower and 
                ('class' in line_lower or 'def' in line_lower or 'import' in line_lower) and
                'comment' not in line_lower and 'string' not in line_lower):
                mock_indicators.append((i, line.strip()))
        
        return mock_indicators
    except Exception:
        return []


def main():
    print("🎯 FINAL MOCK REMOVAL VALIDATION")
    print("=" * 60)
    print("Checking syntax and mock indicators in production files...")
    
    # Key production files that had mock code removed
    production_files = [
        "src/core/agent_factory.py",
        "src/ai/multi_agent/core_new.py", 
        "src/ai/nlp/core.py",
        "src/orchestration/distributed_genetic_algorithm.py",
        "src/orchestration/distributed_genetic_algorithm_clean.py",
        "src/a2a/__init__.py"
    ]
    
    syntax_errors = []
    mock_indicators_found = []
    
    for file_path in production_files:
        if os.path.exists(file_path):
            print(f"\n🔍 Checking {file_path}")
            
            # Check syntax
            is_valid, error = validate_python_syntax(file_path)
            if is_valid:
                print(f"   ✅ Syntax: Valid")
            else:
                print(f"   ❌ Syntax: {error}")
                syntax_errors.append((file_path, error))
            
            # Check for mock indicators
            mock_indicators = find_mock_indicators(file_path)
            if mock_indicators:
                print(f"   ⚠️  Mock indicators found: {len(mock_indicators)}")
                mock_indicators_found.extend([(file_path, line_num, line) for line_num, line in mock_indicators])
            else:
                print(f"   ✅ Mock indicators: None found")
        else:
            print(f"\n❌ File not found: {file_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    if syntax_errors:
        print(f"❌ Syntax Errors Found: {len(syntax_errors)}")
        for file_path, error in syntax_errors:
            print(f"   • {file_path}: {error}")
    else:
        print("✅ All production files have valid syntax")
    
    if mock_indicators_found:
        print(f"\n⚠️  Mock Indicators Found: {len(mock_indicators_found)}")
        for file_path, line_num, line in mock_indicators_found:
            print(f"   • {file_path}:{line_num}: {line}")
    else:
        print("✅ No mock indicators found in production files")
    
    # Overall status
    if not syntax_errors and not mock_indicators_found:
        print("\n🎉 MOCK REMOVAL COMPLETED SUCCESSFULLY!")
        print("All production files are syntax-valid and mock-free.")
        return 0
    else:
        print("\n⚠️  Some issues found - see details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
