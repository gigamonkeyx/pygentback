#!/usr/bin/env python3
"""YAML Validation Script for GitHub Actions Workflows"""

import yaml
import sys

def validate_yaml_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
        print(f'✅ {filepath}: Valid YAML syntax')
        
        # Additional GitHub Actions specific validation
        if 'jobs' in content:
            print(f'  📋 Found {len(content["jobs"])} jobs')
            for job_name, job_config in content['jobs'].items():
                if 'needs' in job_config:
                    needs = job_config['needs']
                    if isinstance(needs, str):
                        needs = [needs]
                    print(f'    🔗 {job_name} depends on: {needs}')
        
        return True
    except yaml.YAMLError as e:
        print(f'❌ {filepath}: YAML syntax error - {e}')
        return False
    except Exception as e:
        print(f'❌ {filepath}: Error reading file - {e}')
        return False

def main():
    # Validate both workflow files
    files = [
        '.github/workflows/ci-cd.yml',
        '.github/workflows/intelligent-docs-build.yml'
    ]

    all_valid = True
    for file in files:
        if not validate_yaml_file(file):
            all_valid = False
        print()

    if all_valid:
        print('✅ All workflow YAML files are syntactically valid and ready for deployment')
        return 0
    else:
        print('❌ YAML validation failed')
        return 1

if __name__ == '__main__':
    sys.exit(main())
