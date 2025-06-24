#!/usr/bin/env python3

"""
Real MCP Filesystem Server - Python Implementation
Provides secure filesystem access via MCP protocol
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the filesystem module to Python path
current_dir = Path(__file__).parent
filesystem_path = current_dir / "filesystem"
sys.path.insert(0, str(filesystem_path))

# Import after path modification
try:
    from server import serve
except ImportError:
    print("Error: Could not import filesystem server module", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    # Default to current working directory if no path provided
    if len(sys.argv) < 2:
        root_path = os.getcwd()
    else:
        root_path = sys.argv[1]
    
    # Ensure the path exists
    if not os.path.exists(root_path):
        print(f"Error: Path {root_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Run the server
    asyncio.run(serve(root_path))
