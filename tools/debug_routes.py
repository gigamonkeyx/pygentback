#!/usr/bin/env python3
"""
Debug FastAPI Routes
Lists all registered routes in the application
"""

import sys
from pathlib import Path
import asyncio

# Add the src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

async def debug_routes():
    """Debug registered routes"""
    print("🔍 Analyzing FastAPI Routes...")
    
    try:
        from api.main import create_app
        
        # Create the app
        app = create_app()
        
        print(f"\n📋 Total routes registered: {len(app.routes)}")
        
        # List all routes
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                methods = getattr(route, 'methods', [])
                print(f"  {route.path} [{', '.join(methods)}]")
            elif hasattr(route, 'path'):
                print(f"  {route.path} [MOUNT/OTHER]")
        
        # Look for our specific routes
        target_routes = ['/api/providers', '/api/models', '/api/debug-test', '/api/mcp/servers']
        print(f"\n🎯 Looking for target routes...")
        
        for target in target_routes:
            found = False
            for route in app.routes:
                if hasattr(route, 'path') and route.path == target:
                    methods = getattr(route, 'methods', [])
                    print(f"  ✅ {target} [{', '.join(methods)}]")
                    found = True
                    break
                elif hasattr(route, 'path') and hasattr(route, 'path_regex'):
                    # Check if it matches the pattern
                    import re
                    if re.match(route.path_regex, target):
                        methods = getattr(route, 'methods', [])
                        print(f"  ✅ {target} (pattern: {route.path}) [{', '.join(methods)}]")
                        found = True
                        break
            
            if not found:
                print(f"  ❌ {target} - NOT FOUND")
        
        return True
        
    except Exception as e:
        print(f"❌ Route analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(debug_routes())
