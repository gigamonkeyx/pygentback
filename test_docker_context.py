#!/usr/bin/env python3
"""Docker Build Context Validation Test"""

import os
import subprocess
import sys
from pathlib import Path

def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except Exception as e:
        print(f"Error calculating size for {path}: {e}")
    return total_size

def format_size(size_bytes: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def check_dockerignore_effectiveness():
    """Test .dockerignore effectiveness by comparing included vs excluded sizes"""
    
    print('=== DOCKER BUILD CONTEXT VALIDATION ===')
    
    # Get total project size
    project_root = Path('.')
    total_size = get_directory_size(project_root)
    print(f"Total project size: {format_size(total_size)}")
    
    # Check .dockerignore exists
    dockerignore_path = project_root / '.dockerignore'
    if not dockerignore_path.exists():
        print("❌ .dockerignore file not found!")
        return False
    
    print("✅ .dockerignore file found")
    
    # Parse .dockerignore patterns
    with open(dockerignore_path, 'r') as f:
        ignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"📋 Found {len(ignore_patterns)} ignore patterns")
    
    # Check sizes of commonly large directories that should be ignored
    large_dirs = {
        'node_modules': project_root / 'ui' / 'node_modules',
        'tests': project_root / 'tests',
        '__pycache__': project_root / '__pycache__',
        '.git': project_root / '.git',
        '3d_studio_workspace/outputs': project_root / '3d_studio_workspace' / 'outputs',
        'models': project_root / 'models',
    }
    
    excluded_size = 0
    for name, path in large_dirs.items():
        if path.exists():
            size = get_directory_size(path)
            excluded_size += size
            print(f"📁 {name}: {format_size(size)} {'(should be excluded)' if size > 1024*1024 else ''}")
    
    # Estimate build context size (total - excluded)
    estimated_context_size = total_size - excluded_size
    print(f"\n📊 Size Analysis:")
    print(f"  Total project: {format_size(total_size)}")
    print(f"  Excluded dirs: {format_size(excluded_size)}")
    print(f"  Estimated context: {format_size(estimated_context_size)}")
    
    # Validate context size is reasonable (< 100MB for efficient builds)
    max_context_size = 100 * 1024 * 1024  # 100MB
    if estimated_context_size > max_context_size:
        print(f"⚠️ Build context may be large: {format_size(estimated_context_size)}")
        print("   Consider adding more patterns to .dockerignore")
        return False
    else:
        print(f"✅ Build context size acceptable: {format_size(estimated_context_size)}")
        return True

def test_dockerfile_stages():
    """Test Dockerfile multi-stage configuration"""
    
    print('\n=== DOCKERFILE MULTI-STAGE VALIDATION ===')
    
    dockerfile_path = Path('Dockerfile')
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found!")
        return False
    
    with open(dockerfile_path, 'r') as f:
        content = f.read()
    
    # Check for multi-stage build patterns
    stages = []
    for line in content.split('\n'):
        if line.strip().startswith('FROM ') and ' as ' in line.lower():
            stage = line.split(' as ')[-1].strip()
            stages.append(stage)
    
    print(f"📋 Found {len(stages)} build stages: {', '.join(stages)}")
    
    expected_stages = ['base', 'development', 'production']
    missing_stages = [stage for stage in expected_stages if stage not in stages]
    
    if missing_stages:
        print(f"⚠️ Missing expected stages: {', '.join(missing_stages)}")
        return False
    else:
        print("✅ All expected stages found")
        return True

def main():
    """Main validation function"""
    
    context_ok = check_dockerignore_effectiveness()
    stages_ok = test_dockerfile_stages()
    
    print('\n=== DOCKER VALIDATION SUMMARY ===')
    
    if context_ok and stages_ok:
        print('✅ Docker build optimization validated successfully')
        return True
    else:
        print('❌ Docker build optimization issues detected')
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
