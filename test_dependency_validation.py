#!/usr/bin/env python3
"""Dependency Management Validation Test"""

import subprocess
import sys
import pkg_resources
from typing import List, Dict, Tuple

def check_critical_dependencies() -> Tuple[bool, List[str]]:
    """Check critical dependencies for PyGent Factory core functionality"""
    
    critical_deps = {
        'numpy': '1.26.4',  # Pinned for binary compatibility
        'pandas': '>=2.1.4',  # Pinned minimum version
        'torch': '>=1.13.0,<2.0.0',  # Pinned range
        'networkx': '>=2.8.0,<3.0.0',  # Pinned range
        'fastapi': '>=0.115.9',  # Core API framework
        'pydantic': '>=2.11.0',  # Data validation
        'sqlalchemy': '>=2.0.0',  # Database ORM
    }
    
    issues = []
    success = True
    
    print('=== CRITICAL DEPENDENCY VALIDATION ===')
    
    for package, version_spec in critical_deps.items():
        try:
            installed = pkg_resources.get_distribution(package)
            print(f'✓ {package}: {installed.version} (required: {version_spec})')
            
            # Check if version meets requirement
            req = pkg_resources.Requirement.parse(f'{package}{version_spec}')
            if installed not in req:
                issues.append(f'{package} {installed.version} does not meet requirement {version_spec}')
                success = False
                print(f'  ⚠️  Version mismatch detected')
                
        except pkg_resources.DistributionNotFound:
            issues.append(f'{package} not installed')
            success = False
            print(f'✗ {package}: NOT INSTALLED (required: {version_spec})')
    
    return success, issues

def test_import_compatibility() -> Tuple[bool, List[str]]:
    """Test that critical imports work without conflicts"""
    
    critical_imports = [
        'numpy',
        'pandas', 
        'torch',
        'networkx',
        'fastapi',
        'pydantic',
        'sqlalchemy'
    ]
    
    issues = []
    success = True
    
    print('\n=== IMPORT COMPATIBILITY TEST ===')
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f'✓ {module}: Import successful')
        except ImportError as e:
            issues.append(f'{module}: Import failed - {e}')
            success = False
            print(f'✗ {module}: Import failed - {e}')
        except Exception as e:
            issues.append(f'{module}: Import error - {e}')
            success = False
            print(f'⚠️ {module}: Import warning - {e}')
    
    return success, issues

def analyze_dependency_conflicts() -> Dict[str, int]:
    """Analyze pip check output for conflict patterns"""
    
    try:
        result = subprocess.run(['pip', 'check'], 
                              capture_output=True, 
                              text=True, 
                              timeout=30)
        
        if result.returncode == 0:
            return {'total_conflicts': 0}
        
        conflicts = result.stdout.split('\n')
        conflict_types = {
            'version_mismatch': 0,
            'missing_dependency': 0,
            'incompatible_version': 0
        }
        
        for conflict in conflicts:
            if 'has requirement' in conflict:
                conflict_types['version_mismatch'] += 1
            elif 'requires' in conflict and 'not installed' in conflict:
                conflict_types['missing_dependency'] += 1
            elif 'incompatible' in conflict.lower():
                conflict_types['incompatible_version'] += 1
        
        conflict_types['total_conflicts'] = len([c for c in conflicts if c.strip()])
        return conflict_types
        
    except Exception as e:
        print(f'Error analyzing conflicts: {e}')
        return {'error': str(e)}

def main():
    """Main validation function"""
    
    print('=== DEPENDENCY MANAGEMENT VALIDATION ===\n')
    
    # Test critical dependencies
    deps_ok, dep_issues = check_critical_dependencies()
    
    # Test import compatibility  
    imports_ok, import_issues = test_import_compatibility()
    
    # Analyze conflicts
    print('\n=== CONFLICT ANALYSIS ===')
    conflicts = analyze_dependency_conflicts()
    
    if 'error' not in conflicts:
        print(f"Total conflicts detected: {conflicts.get('total_conflicts', 0)}")
        print(f"Version mismatches: {conflicts.get('version_mismatch', 0)}")
        print(f"Missing dependencies: {conflicts.get('missing_dependency', 0)}")
        print(f"Incompatible versions: {conflicts.get('incompatible_version', 0)}")
    
    # Summary
    print('\n=== VALIDATION SUMMARY ===')
    
    if deps_ok and imports_ok:
        print('✓ Core dependencies validated successfully')
        if conflicts.get('total_conflicts', 0) > 0:
            print(f'⚠️ {conflicts["total_conflicts"]} non-critical conflicts detected')
            print('  These may be resolved by dependency pinning in CI environment')
        return True
    else:
        print('✗ Critical dependency issues detected:')
        for issue in dep_issues + import_issues:
            print(f'  - {issue}')
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
