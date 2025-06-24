#!/usr/bin/env python3
"""
Final Test Battery - Mock Removal Validation
Tests the complete cleanup and production readiness of the codebase.
"""

import os
import re
import subprocess
import sys

def test_mock_patterns():
    """Check for remaining mock patterns in production files."""
    print("🔍 SCANNING FOR MOCK/PLACEHOLDER PATTERNS")
    print("=" * 60)
    
    production_files = [
        'src/core/agent_factory.py',
        'src/ai/multi_agent/core_new.py', 
        'src/ai/nlp/core.py',
        'src/orchestration/distributed_genetic_algorithm.py',
        'src/a2a/__init__.py',
        'src/orchestration/collaborative_self_improvement.py'
    ]
    
    mock_patterns = [r'\bmock\b', r'\bMock\b', r'\bMOCK\b', r'TODO:', r'FIXME:', r'HACK:']
    
    total_issues = 0
    for file_path in production_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                issues = []
                for pattern in mock_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        issues.append(f'{pattern}({len(matches)})')
                
                if issues:
                    print(f'⚠️  {file_path}: {", ".join(issues)}')
                    total_issues += len(issues)
                else:
                    print(f'✅ {file_path}: Clean')
    
    print(f'\n📊 TOTAL ISSUES FOUND: {total_issues}')
    return total_issues == 0

def test_syntax_compilation():
    """Test that all production files compile without syntax errors."""
    print("\n🐍 SYNTAX COMPILATION TESTS")
    print("=" * 60)
    
    production_files = [
        'src/core/agent_factory.py',
        'src/ai/multi_agent/core_new.py', 
        'src/ai/nlp/core.py',
        'src/orchestration/distributed_genetic_algorithm.py',
        'src/orchestration/collaborative_self_improvement.py'
    ]
    
    all_passed = True
    for file_path in production_files:
        if os.path.exists(file_path):
            try:
                result = subprocess.run([sys.executable, '-m', 'py_compile', file_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f'✅ {file_path}: Syntax OK')
                else:
                    print(f'❌ {file_path}: {result.stderr.strip()}')
                    all_passed = False
            except Exception as e:
                print(f'❌ {file_path}: {e}')
                all_passed = False
    
    return all_passed

def test_core_imports():
    """Test basic imports of core production modules."""
    print("\n📦 CORE IMPORT TESTS")
    print("=" * 60)
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    test_imports = [
        ('src.ai.multi_agent.core_new', 'Agent'),
        ('src.ai.nlp.core', 'NLPCore'),
    ]
    
    all_passed = True
    for module_name, class_name in test_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f'✅ {module_name}.{class_name}: Import OK')
        except Exception as e:
            print(f'❌ {module_name}.{class_name}: {e}')
            all_passed = False
    
    return all_passed

def test_a2a_functionality():
    """Test A2A module basic functionality."""
    print("\n🤝 A2A MODULE TESTS")
    print("=" * 60)
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.a2a import A2AServer
        print('✅ A2AServer: Import OK')
        
        # Test basic instantiation
        server = A2AServer()
        print('✅ A2AServer: Instantiation OK')
        
        return True
    except Exception as e:
        print(f'❌ A2A Tests: {e}')
        return False

def run_full_test_battery():
    """Run the complete test battery."""
    print("🎯 MOCK REMOVAL VALIDATION - FULL TEST BATTERY")
    print("=" * 70)
    
    tests = [
        ("Mock Pattern Detection", test_mock_patterns),
        ("Syntax Compilation", test_syntax_compilation),
        ("Core Import Tests", test_core_imports),
        ("A2A Functionality", test_a2a_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: Exception - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n🏁 TEST SUMMARY")
    print("=" * 60)
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Production code is clean and ready.")
    else:
        print("⚠️  Some tests failed. Review issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_test_battery()
    sys.exit(0 if success else 1)
