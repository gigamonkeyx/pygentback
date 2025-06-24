#!/usr/bin/env python3
"""
Setup Git Hooks for Documentation Build Triggers

Installs git hooks that automatically detect when documentation
should be rebuilt based on commits and file changes.
"""

import os
import sys
import stat
from pathlib import Path
import subprocess

def create_pre_commit_hook():
    """Create pre-commit hook to detect documentation changes"""
    hook_content = '''#!/bin/bash
# Pre-commit hook for documentation build triggers

# Check if documentation files are being committed
docs_changed=$(git diff --cached --name-only | grep -E "\\.(md|vue|mmd|ts|js)$" | grep -E "^src/docs/")

if [ ! -z "$docs_changed" ]; then
    echo "📝 Documentation files changed, marking for rebuild:"
    echo "$docs_changed"
    
    # Create trigger file
    touch src/docs/.force_rebuild
    echo "✅ Documentation rebuild trigger set"
fi

exit 0
'''
    return hook_content

def create_post_commit_hook():
    """Create post-commit hook to check for version tags"""
    hook_content = '''#!/bin/bash
# Post-commit hook for documentation build triggers

# Check if this commit created a version tag
current_commit=$(git rev-parse HEAD)
tags_on_commit=$(git tag --points-at $current_commit)

if [ ! -z "$tags_on_commit" ]; then
    echo "🏷️  Version tag detected: $tags_on_commit"
    echo "📚 Triggering documentation rebuild for release"
    
    # Create trigger file
    touch src/docs/.force_rebuild
    echo "✅ Documentation rebuild trigger set for release"
fi

# Check if we're on main/master branch
current_branch=$(git branch --show-current)
if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
    echo "🌟 Commit to $current_branch branch detected"
    
    # Check if significant files changed
    changed_files=$(git diff --name-only HEAD~1 HEAD | grep -E "\\.(py|md|vue|mmd|ts|js|json)$")
    
    if [ ! -z "$changed_files" ]; then
        echo "📝 Significant files changed, marking for documentation rebuild"
        touch src/docs/.force_rebuild
        echo "✅ Documentation rebuild trigger set"
    fi
fi

exit 0
'''
    return hook_content

def create_post_merge_hook():
    """Create post-merge hook for pull request merges"""
    hook_content = '''#!/bin/bash
# Post-merge hook for documentation build triggers

echo "🔀 Merge detected, checking for documentation updates"

# Check if documentation files were affected by the merge
docs_changed=$(git diff --name-only HEAD~1 HEAD | grep -E "\\.(md|vue|mmd|ts|js)$" | grep -E "^src/docs/")

if [ ! -z "$docs_changed" ]; then
    echo "📝 Documentation files affected by merge:"
    echo "$docs_changed"
    
    # Create trigger file
    touch src/docs/.force_rebuild
    echo "✅ Documentation rebuild trigger set for merge"
fi

exit 0
'''
    return hook_content

def install_git_hook(hook_name: str, hook_content: str, git_dir: Path):
    """Install a git hook"""
    hooks_dir = git_dir / "hooks"
    hook_file = hooks_dir / hook_name
    
    # Create hooks directory if it doesn't exist
    hooks_dir.mkdir(exist_ok=True)
    
    # Write hook content
    with open(hook_file, 'w') as f:
        f.write(hook_content)
    
    # Make executable
    hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
    
    print(f"✅ Installed {hook_name} hook")

def setup_git_hooks():
    """Setup all git hooks for documentation build triggers"""
    
    # Find git directory
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True
        )
        git_dir = Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        print("❌ Error: Not in a git repository")
        return False
    
    print(f"📁 Git directory: {git_dir}")
    
    # Install hooks
    hooks = [
        ("pre-commit", create_pre_commit_hook()),
        ("post-commit", create_post_commit_hook()),
        ("post-merge", create_post_merge_hook())
    ]
    
    for hook_name, hook_content in hooks:
        install_git_hook(hook_name, hook_content, git_dir)
    
    print("\n🎉 Git hooks installed successfully!")
    print("\nThe following triggers will now automatically mark documentation for rebuild:")
    print("  • Changes to .md, .vue, .mmd, .ts, .js files in src/docs/")
    print("  • Commits to main/master branch with significant changes")
    print("  • Version tag creation")
    print("  • Pull request merges affecting documentation")
    print("\nTo manually trigger a rebuild, run:")
    print("  touch src/docs/.force_rebuild")
    
    return True

def remove_git_hooks():
    """Remove documentation git hooks"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True
        )
        git_dir = Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        print("❌ Error: Not in a git repository")
        return False
    
    hooks_dir = git_dir / "hooks"
    hooks_to_remove = ["pre-commit", "post-commit", "post-merge"]
    
    for hook_name in hooks_to_remove:
        hook_file = hooks_dir / hook_name
        if hook_file.exists():
            hook_file.unlink()
            print(f"🗑️  Removed {hook_name} hook")
    
    print("✅ Git hooks removed")
    return True

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "remove":
        remove_git_hooks()
    else:
        setup_git_hooks()

if __name__ == "__main__":
    main()
