#!/usr/bin/env python3
"""
Git Hook Setup Script

This script sets up git hooks for automated feature registry management.
Run this script to install the hooks into your git repository.
"""

import os
import shutil
import stat
from pathlib import Path

def setup_git_hooks():
    """Setup git hooks for feature registry integration"""
    
    project_root = Path(__file__).parent
    hooks_source = project_root / "hooks"
    git_hooks_dir = project_root / ".git" / "hooks"
    
    if not git_hooks_dir.exists():
        print("❌ .git/hooks directory not found. Are you in a git repository?")
        return False
    
    print("🔧 Setting up git hooks for feature registry...")
    
    hooks_to_install = [
        "pre-commit",
        "post-commit"
    ]
    
    for hook_name in hooks_to_install:
        source_hook = hooks_source / hook_name
        dest_hook = git_hooks_dir / hook_name
        
        if not source_hook.exists():
            print(f"⚠️  Source hook not found: {source_hook}")
            continue
        
        # Backup existing hook if it exists
        if dest_hook.exists():
            backup_path = dest_hook.with_suffix('.backup')
            shutil.copy2(dest_hook, backup_path)
            print(f"📋 Backed up existing {hook_name} to {backup_path}")
        
        # Copy hook
        shutil.copy2(source_hook, dest_hook)
          # Make executable (Unix/Linux/Mac)
        if os.name != 'nt':  # Not Windows
            current_permissions = dest_hook.stat().st_mode
            dest_hook.chmod(current_permissions | stat.S_IEXEC)
        
        print(f"✅ Installed {hook_name} hook")
    
    print("""
🎉 Git hooks installed successfully!

The following hooks are now active:
• pre-commit: Checks feature registry before commits
• post-commit: Updates feature registry after commits

To run a manual feature audit, use:
  python feature_workflow_integration.py daily-audit

To temporarily bypass pre-commit hooks:
  git commit --no-verify

For more information, see:
  docs/COMPLETE_FEATURE_REGISTRY.md
""")
    
    return True

def main():
    """Main function"""
    success = setup_git_hooks()
    if success:
        print("✅ Setup completed successfully")
    else:
        print("❌ Setup failed")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
