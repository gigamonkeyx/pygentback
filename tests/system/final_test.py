import py_compile

print('🔍 TESTING PRODUCTION IMPLEMENTATIONS')
print('=' * 50)

files = [
    'src/core/agent_factory.py',
    'src/orchestration/distributed_genetic_algorithm.py',
    'src/ai/multi_agent/core_new.py', 
    'src/ai/nlp/core.py',
    'src/a2a/__init__.py',
    'src/orchestration/collaborative_self_improvement.py'
]

syntax_ok = 0
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'✅ {f.split("/")[-1]} - Syntax OK')
        syntax_ok += 1
    except Exception as e:
        print(f'❌ {f.split("/")[-1]} - Error: {e}')

print(f'\n📊 Syntax Check: {syntax_ok}/{len(files)} files pass')

# Also check for remaining mock patterns
import re
mock_patterns = [
    'MockBuilder', 'MockValidator', 'MockSettings',
    'TODO:', 'FIXME:', 'HACK:'
]

print('\n🔍 CHECKING FOR REMAINING MOCK PATTERNS')
print('=' * 50)

total_mock_issues = 0
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        file_issues = 0
        for pattern in mock_patterns:
            if pattern in content:
                matches = content.count(pattern)
                print(f'⚠️  {f.split("/")[-1]}: Found {matches} instances of "{pattern}"')
                file_issues += matches
        
        if file_issues == 0:
            print(f'✅ {f.split("/")[-1]} - No mock patterns found')
        total_mock_issues += file_issues
            
    except Exception as e:
        print(f'❌ Error checking {f}: {e}')

print(f'\n📊 Mock Pattern Check: {total_mock_issues} total issues found')

if syntax_ok == len(files) and total_mock_issues == 0:
    print('\n🎉 ALL TESTS PASSED! Mock removal successful!')
else:
    print(f'\n⚠️  Issues found: {len(files) - syntax_ok} syntax errors, {total_mock_issues} mock patterns')
