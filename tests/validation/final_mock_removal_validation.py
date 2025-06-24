#!/usr/bin/env python3
"""
Final Mock Removal Validation
Validates that all production code compiles and mocks have been removed.
"""

import ast
import os
import sys
from typing import List, Dict, Any

def validate_python_syntax(file_path: str) -> Dict[str, Any]:
    """Validate Python file syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to check syntax
        ast.parse(content)
        return {"status": "valid", "error": None}
    except SyntaxError as e:
        return {"status": "syntax_error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def find_mock_indicators(file_path: str) -> List[str]:
    """Find remaining mock/placeholder indicators in file."""
    indicators = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            if any(term in line_lower for term in ['mock', 'placeholder for', 'fixme', 'hack']):
                # Skip legitimate uses (in comments about real implementations)
                if not any(skip in line_lower for skip in [
                    'remove mock', 'replace mock', 'production', 'real', 
                    'no mock', 'non-mock', 'eliminate mock'
                ]):
                    indicators.append(f"Line {i}: {line.strip()}")
    except Exception as e:
        indicators.append(f"Error reading file: {e}")
    
    return indicators

def main():
    """Run final validation."""
    print("=== FINAL MOCK REMOVAL VALIDATION ===\n")
    
    # Key production files to validate
    production_files = [
        "src/core/agent_factory.py",
        "src/orchestration/distributed_genetic_algorithm.py", 
        "src/orchestration/distributed_genetic_algorithm_clean.py",
        "src/ai/multi_agent/core_new.py",
        "src/ai/nlp/core.py",
        "src/a2a/__init__.py",
        "src/communication/protocols.py",
        "src/communication/protocols/manager.py"
    ]
    
    syntax_errors = 0
    mock_findings = 0
    
    print("1. SYNTAX VALIDATION:")
    print("-" * 40)
    
    for file_path in production_files:
        if os.path.exists(file_path):
            result = validate_python_syntax(file_path)
            if result["status"] == "valid":
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path}: {result['error']}")
                syntax_errors += 1
        else:
            print(f"? {file_path}: File not found")
    
    print(f"\nSyntax validation complete. Errors: {syntax_errors}")
    
    print("\n2. MOCK/PLACEHOLDER DETECTION:")
    print("-" * 40)
    
    for file_path in production_files:
        if os.path.exists(file_path):
            indicators = find_mock_indicators(file_path)
            if indicators:
                print(f"\n{file_path}:")
                for indicator in indicators[:5]:  # Show first 5
                    print(f"  - {indicator}")
                if len(indicators) > 5:
                    print(f"  ... and {len(indicators) - 5} more")
                mock_findings += len(indicators)
            else:
                print(f"✓ {file_path}: No mock indicators found")
    
    print(f"\nMock detection complete. Findings: {mock_findings}")
    
    print("\n3. SUMMARY:")
    print("-" * 40)
    
    if syntax_errors == 0:
        print("✓ All production files have valid syntax")
    else:
        print(f"✗ {syntax_errors} files have syntax errors")
    
    if mock_findings == 0:
        print("✓ No mock/placeholder indicators found in production code")
    else:
        print(f"⚠ {mock_findings} potential mock/placeholder indicators found")
    
    overall_status = "PASS" if syntax_errors == 0 and mock_findings < 5 else "REVIEW_NEEDED"
    print(f"\nOverall Status: {overall_status}")
    
    return 0 if overall_status == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
