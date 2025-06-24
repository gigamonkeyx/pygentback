#!/usr/bin/env python3
"""
Trigger deployment and check status
"""

import requests
import time
import json
from datetime import datetime

def check_github_repository():
    """Check if GitHub repository is properly configured"""
    print("🔍 Checking GitHub Repository...")
    
    try:
        # Check repository exists and has recent commits
        response = requests.get("https://api.github.com/repos/gigamonkeyx/pygent")
        if response.status_code == 200:
            repo_data = response.json()
            print(f"   ✅ Repository exists: {repo_data['full_name']}")
            print(f"   📅 Last updated: {repo_data['updated_at']}")
            print(f"   📊 Size: {repo_data['size']} KB")
            return True
        else:
            print(f"   ❌ Repository check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Repository check error: {e}")
        return False

def check_recent_commits():
    """Check recent commits to verify our deployment was pushed"""
    print("📝 Checking Recent Commits...")
    
    try:
        response = requests.get("https://api.github.com/repos/gigamonkeyx/pygent/commits")
        if response.status_code == 200:
            commits = response.json()
            
            print(f"   📊 Found {len(commits)} recent commits")
            for i, commit in enumerate(commits[:3]):  # Show last 3 commits
                message = commit['commit']['message']
                date = commit['commit']['author']['date']
                sha = commit['sha'][:8]
                print(f"   {i+1}. [{sha}] {message} ({date})")
            
            # Check if our deployment commit is there
            deployment_commit = any("Deploy complete PyGent Factory UI" in commit['commit']['message'] 
                                  for commit in commits[:5])
            
            if deployment_commit:
                print("   ✅ Deployment commit found in recent commits")
                return True
            else:
                print("   ⚠️  Deployment commit not found in recent commits")
                return False
                
        else:
            print(f"   ❌ Commits check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Commits check error: {e}")
        return False

def check_vite_config():
    """Check if vite.config.ts has the correct base path"""
    print("⚙️  Checking Vite Configuration...")
    
    try:
        response = requests.get("https://raw.githubusercontent.com/gigamonkeyx/pygent/master/vite.config.ts")
        if response.status_code == 200:
            config_content = response.text
            
            if "base: '/pygent/'" in config_content:
                print("   ✅ Vite config has correct base path: '/pygent/'")
                return True
            else:
                print("   ❌ Vite config missing base path")
                print("   📄 Current config preview:")
                lines = config_content.split('\n')[:10]
                for line in lines:
                    print(f"      {line}")
                return False
        else:
            print(f"   ❌ Vite config check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Vite config check error: {e}")
        return False

def check_package_json():
    """Check if package.json has all required dependencies"""
    print("📦 Checking Package Dependencies...")
    
    try:
        response = requests.get("https://raw.githubusercontent.com/gigamonkeyx/pygent/master/package.json")
        if response.status_code == 200:
            package_data = response.json()
            
            required_deps = [
                'react', 'react-dom', 'react-router-dom',
                '@radix-ui/react-dialog', 'zustand', 'vite'
            ]
            
            dependencies = package_data.get('dependencies', {})
            dev_dependencies = package_data.get('devDependencies', {})
            all_deps = {**dependencies, **dev_dependencies}
            
            missing_deps = []
            for dep in required_deps:
                if dep not in all_deps:
                    missing_deps.append(dep)
            
            if not missing_deps:
                print(f"   ✅ All required dependencies present ({len(all_deps)} total)")
                return True
            else:
                print(f"   ❌ Missing dependencies: {missing_deps}")
                return False
        else:
            print(f"   ❌ Package.json check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Package.json check error: {e}")
        return False

def check_cloudflare_pages_status():
    """Check if Cloudflare Pages might be configured"""
    print("☁️  Checking Cloudflare Pages Status...")
    
    try:
        # Try to access the pygent path and analyze the response
        response = requests.get("https://timpayne.net/pygent", timeout=30)
        
        print(f"   📊 Status Code: {response.status_code}")
        print(f"   🌐 Server: {response.headers.get('Server', 'Unknown')}")
        
        if 'cloudflare' in response.headers.get('Server', '').lower():
            print("   ✅ Request is going through Cloudflare")
        
        # Check if it's a Cloudflare Pages response
        cf_ray = response.headers.get('CF-RAY')
        if cf_ray:
            print(f"   📡 CF-RAY: {cf_ray}")
        
        # Analyze the response content
        content = response.text[:500]  # First 500 chars
        
        if "HARDCORE GRAPHICS SHOWCASE" in content:
            print("   ⚠️  Getting graphics showcase content (wrong deployment)")
            print("   🔧 This suggests Cloudflare Pages is not configured for pygent repository")
            return False
        elif "PyGent Factory" in content:
            print("   ✅ Getting PyGent Factory content!")
            return True
        elif response.status_code == 404:
            print("   ❌ Getting 404 - Cloudflare Pages likely not configured")
            return False
        else:
            print("   ❓ Getting unexpected content")
            print(f"   📄 Content preview: {content[:100]}...")
            return False
            
    except Exception as e:
        print(f"   ❌ Cloudflare Pages check error: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("🚀 PyGent Factory Deployment Diagnostic")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all checks
    checks = [
        ("GitHub Repository", check_github_repository),
        ("Recent Commits", check_recent_commits),
        ("Vite Configuration", check_vite_config),
        ("Package Dependencies", check_package_json),
        ("Cloudflare Pages", check_cloudflare_pages_status)
    ]
    
    results = {}
    for check_name, check_func in checks:
        print()
        result = check_func()
        results[check_name] = result
    
    print()
    print("=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} {status}")
    
    print()
    print(f"Overall: {passed}/{total} checks passed")
    
    if results.get("Cloudflare Pages", False):
        print("🎉 DEPLOYMENT IS WORKING!")
    elif all(results[k] for k in ["GitHub Repository", "Vite Configuration", "Package Dependencies"]):
        print("🔧 REPOSITORY IS READY - Cloudflare Pages needs configuration")
        print()
        print("📋 Next Steps:")
        print("1. Go to https://dash.cloudflare.com/pages")
        print("2. Create new project from gigamonkeyx/pygent repository")
        print("3. Use build settings from cloudflare_pages_setup_guide.md")
    else:
        print("❌ REPOSITORY ISSUES DETECTED")
        print("   Fix repository issues before configuring Cloudflare Pages")

if __name__ == "__main__":
    main()
