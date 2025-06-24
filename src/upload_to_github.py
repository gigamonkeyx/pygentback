"""
GitHub Upload Script

Uploads all deployment files to GitHub repository using the provided token.
"""

import os
import subprocess
import shutil
from pathlib import Path

def upload_to_github():
    """Upload deployment files to GitHub repository."""
    
    # Configuration
    github_token = "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"
    repo_url = f"https://{github_token}@github.com/gigamonkeyx/pygent.git"
    deployment_path = Path(__file__).parent / "deployment_ready"
    temp_repo_path = Path(__file__).parent / "temp_pygent_repo"
    
    print("🚀 Starting GitHub upload...")
    
    try:
        # Clean up any existing temp directory
        if temp_repo_path.exists():
            shutil.rmtree(temp_repo_path)
        
        # Clone the repository
        print("📥 Cloning repository...")
        subprocess.run([
            "git", "clone", repo_url, str(temp_repo_path)
        ], check=True, capture_output=True)
        
        # Copy all deployment files
        print("📁 Copying deployment files...")
        for item in deployment_path.rglob('*'):
            if item.is_file():
                relative_path = item.relative_to(deployment_path)
                target_path = temp_repo_path / relative_path
                
                # Create parent directories if they don't exist
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(item, target_path)
                print(f"  ✅ Copied: {relative_path}")
        
        # Change to repo directory
        os.chdir(temp_repo_path)
        
        # Configure git user
        subprocess.run(["git", "config", "user.email", "timpayne@gmail.com"], check=True)
        subprocess.run(["git", "config", "user.name", "gigamonkeyx"], check=True)
        
        # Add all files
        print("📦 Adding files to git...")
        subprocess.run(["git", "add", "."], check=True)
        
        # Commit
        print("💾 Committing changes...")
        commit_message = """🚀 PyGent Factory UI - Complete Deployment

- Complete React 18 + TypeScript application
- Multi-agent chat interface with real-time WebSocket
- Tree of Thought reasoning visualization
- System monitoring dashboard
- MCP marketplace integration
- Zero mock code architecture
- Production-ready build configuration
- Cloudflare Pages optimized
- Performance optimized with bundle splitting
- Mobile responsive design
- Professional UI/UX

Features:
✅ Advanced AI reasoning system
✅ Real-time multi-agent orchestration
✅ Professional UI/UX design
✅ Performance optimized
✅ Mobile responsive
✅ Zero mock code maintained

Ready for deployment to timpayne.net/pygent"""
        
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push to GitHub
        print("🚀 Pushing to GitHub...")
        subprocess.run(["git", "push", "origin", "main"], check=True)
        
        print("\n🎉 SUCCESS! Files uploaded to GitHub!")
        print("📍 Repository: https://github.com/gigamonkeyx/pygent")
        print("🌐 Ready for Cloudflare Pages setup!")
        
        # Clean up
        os.chdir(Path(__file__).parent)
        shutil.rmtree(temp_repo_path)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")
        print(f"Command output: {e.output}")
        return False
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False
    finally:
        # Clean up temp directory
        if temp_repo_path.exists():
            try:
                os.chdir(Path(__file__).parent)
                shutil.rmtree(temp_repo_path)
            except:
                pass

if __name__ == "__main__":
    print("🤖 GitHub Upload Script")
    print("=" * 50)
    
    success = upload_to_github()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. ✅ GitHub upload complete")
        print("2. 🌐 Set up Cloudflare Pages")
        print("3. 🔗 Configure custom domain")
        print("4. 🚀 Go live!")
    else:
        print("\n❌ Upload failed - check errors above")
    
    input("\nPress Enter to exit...")