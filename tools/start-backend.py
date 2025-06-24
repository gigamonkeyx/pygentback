#!/usr/bin/env python3
"""
Simple backend starter script
"""
import os
import sys
import subprocess
from pathlib import Path

# Change to the project directory
project_dir = Path(__file__).parent
os.chdir(project_dir)

# Add src to Python path
sys.path.insert(0, str(project_dir / "src"))

try:
    print(f"Starting PyGent Factory backend from: {project_dir}")
    print("Loading main.py...")
    
    # Run the main script
    result = subprocess.run([sys.executable, "main.py", "--mode", "server"], 
                          cwd=project_dir, 
                          capture_output=False)
    
    if result.returncode != 0:
        print(f"Backend exited with code: {result.returncode}")
    
except KeyboardInterrupt:
    print("\n🛑 Backend stopped by user")
except Exception as e:
    print(f"❌ Error starting backend: {e}")
    import traceback
    traceback.print_exc()
