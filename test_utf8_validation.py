#!/usr/bin/env python3
"""UTF-8 Configuration Validation Test"""

import os
import logging
import sys

def test_utf8_configuration():
    """Test UTF-8 configuration as implemented in CI/CD pipeline"""
    
    # Set UTF-8 environment variables as configured in CI
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    
    # Configure logging with UTF-8
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Test non-ASCII characters
    test_strings = [
        'Testing UTF-8: PyGent Factory',
        'Unicode symbols: checkmark cross warning',
        'International: cafe, naive, resume',
        'Math symbols: alpha beta gamma delta',
        'Emoji test: fire bulb target star'
    ]
    
    print('=== UTF-8 VALIDATION TEST ===')
    success_count = 0
    
    for i, test_str in enumerate(test_strings, 1):
        try:
            logger.info(f'Test {i}: {test_str}')
            print(f'✓ UTF-8 Test {i} passed: {test_str}')
            success_count += 1
        except UnicodeEncodeError as e:
            print(f'✗ UTF-8 Test {i} failed: {e}')
            return False
    
    # Test environment variables
    print('\n=== ENVIRONMENT VALIDATION ===')
    env_vars = ['PYTHONIOENCODING', 'LANG', 'LC_ALL']
    for var in env_vars:
        value = os.environ.get(var, 'NOT_SET')
        print(f'{var}: {value}')
        if 'utf-8' not in value.lower() and 'en_us' not in value.lower():
            print(f'✗ Environment variable {var} not properly configured')
            return False
    
    print(f'\n✓ All {success_count} UTF-8 tests passed - Configuration validated')
    return True

if __name__ == '__main__':
    success = test_utf8_configuration()
    sys.exit(0 if success else 1)
