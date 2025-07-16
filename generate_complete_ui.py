#!/usr/bin/env python3
"""
Generate Complete PyGent Factory UI
Create each file individually for a complete application
"""

import requests
import json
from pathlib import Path


def call_deepseek(prompt: str) -> str:
    """Call DeepSeek-Coder"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "deepseek-coder:6.7b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "max_tokens": 4000
        }
    }
    
    response = requests.post(url, json=payload, timeout=180)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "")
    else:
        raise Exception(f"DeepSeek error: {response.status_code}")


def generate_file(file_name: str, prompt: str, ui_dir: Path) -> bool:
    """Generate a single file"""
    
    print(f"📝 Generating {file_name}...")
    
    try:
        response = call_deepseek(prompt)
        
        # Clean the response
        clean_code = response
        if "```" in response:
            # Extract code from markdown blocks
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are code blocks
                    clean_code = part
                    break
        
        # Remove language specifiers
        lines = clean_code.split('\n')
        if lines and lines[0].strip() in ['tsx', 'typescript', 'javascript', 'json', 'css']:
            lines = lines[1:]
        
        clean_code = '\n'.join(lines).strip()
        
        # Save file
        file_path = ui_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_code)
        
        print(f"   ✅ Created {file_name} ({len(clean_code)} chars)")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to generate {file_name}: {e}")
        return False


def create_complete_ui():
    """Generate complete UI file by file"""
    
    print("🚀 CREATING COMPLETE PYGENT FACTORY UI")
    print("=" * 60)
    
    ui_dir = Path("pygent-factory-ui")
    ui_dir.mkdir(exist_ok=True)
    
    files_to_generate = [
        {
            "name": "package.json",
            "prompt": """Create a package.json for a React TypeScript application called "pygent-factory-ui".

Include:
- React 18, TypeScript, React Router
- TailwindCSS for styling
- Axios for API calls
- React Query for data fetching
- Lucide React for icons
- Development scripts (dev, build, preview)
- Modern dependencies

Generate only the JSON, no explanations."""
        },
        {
            "name": "src/App.tsx", 
            "prompt": """Create App.tsx for PyGent Factory - an AI agent management platform.

Include:
- React Router setup with routes for Dashboard, Agents, Tasks
- Layout component wrapper
- Error boundary
- Theme provider
- Modern React 18 with TypeScript
- Clean, professional structure

Generate only the TSX code, no explanations."""
        },
        {
            "name": "src/components/Layout.tsx",
            "prompt": """Create Layout.tsx for PyGent Factory with:

- Top navigation bar with logo and menu items (Dashboard, Agents, Tasks)
- Sidebar navigation
- Main content area
- Dark/light theme toggle
- Responsive design with TailwindCSS
- Professional styling
- TypeScript

Generate only the TSX code, no explanations."""
        },
        {
            "name": "src/pages/Dashboard.tsx",
            "prompt": """Create Dashboard.tsx for PyGent Factory showing:

- Agent status overview (total, active, idle)
- Recent tasks list
- System health metrics
- Quick action buttons
- Charts/graphs for agent performance
- Real-time updates
- TailwindCSS styling
- TypeScript

Generate only the TSX code, no explanations."""
        },
        {
            "name": "src/pages/Agents.tsx",
            "prompt": """Create Agents.tsx for PyGent Factory with:

- Agent list/grid view
- Add new agent button
- Agent search and filtering
- Agent status indicators
- Edit/delete agent actions
- Agent performance metrics
- TailwindCSS styling
- TypeScript

Generate only the TSX code, no explanations."""
        },
        {
            "name": "src/pages/Tasks.tsx",
            "prompt": """Create Tasks.tsx for PyGent Factory with:

- Task list with status (pending, running, completed, failed)
- Create new task form
- Task filtering and search
- Task progress indicators
- Task details modal
- Real-time task updates
- TailwindCSS styling
- TypeScript

Generate only the TSX code, no explanations."""
        },
        {
            "name": "src/components/AgentCard.tsx",
            "prompt": """Create AgentCard.tsx component for displaying individual agents:

- Agent name, type, and status
- Performance metrics
- Action buttons (start, stop, edit, delete)
- Status indicators with colors
- Responsive card design
- TailwindCSS styling
- TypeScript props interface

Generate only the TSX code, no explanations."""
        },
        {
            "name": "src/types/index.ts",
            "prompt": """Create TypeScript type definitions for PyGent Factory:

- Agent interface (id, name, type, status, capabilities, etc.)
- Task interface (id, title, description, status, agent_id, etc.)
- API response types
- Component prop types
- Enum types for status values

Generate only the TypeScript code, no explanations."""
        },
        {
            "name": "src/api/client.ts",
            "prompt": """Create API client for PyGent Factory backend:

- Axios-based HTTP client
- Base URL configuration
- Agent CRUD operations
- Task CRUD operations
- Error handling
- TypeScript types
- Request/response interceptors

Generate only the TypeScript code, no explanations."""
        }
    ]
    
    successful = 0
    total = len(files_to_generate)
    
    for file_info in files_to_generate:
        if generate_file(file_info["name"], file_info["prompt"], ui_dir):
            successful += 1
    
    print(f"\n📊 GENERATION COMPLETE:")
    print(f"   Files created: {successful}/{total}")
    print(f"   Success rate: {successful/total:.1%}")
    
    # List all created files
    print(f"\n📁 Created files in {ui_dir}/:")
    for file_path in sorted(ui_dir.rglob("*")):
        if file_path.is_file():
            size = file_path.stat().st_size
            rel_path = file_path.relative_to(ui_dir)
            print(f"   📄 {rel_path} ({size:,} bytes)")
    
    # Show key file previews
    show_previews(ui_dir)
    
    if successful >= total * 0.8:  # 80% success rate
        print(f"\n🎉 REAL UI SUCCESSFULLY CREATED!")
        print(f"📁 Location: {ui_dir.absolute()}")
        print(f"🚀 Ready for: npm install && npm run dev")
        return True
    else:
        print(f"\n🔄 PARTIAL SUCCESS - Some files need manual creation")
        return False


def show_previews(ui_dir: Path):
    """Show previews of key files"""
    
    key_files = ['package.json', 'src/App.tsx', 'src/pages/Dashboard.tsx']
    
    print(f"\n📝 KEY FILE PREVIEWS:")
    print("-" * 50)
    
    for file_name in key_files:
        file_path = ui_dir / file_name
        if file_path.exists():
            print(f"\n📄 {file_name}:")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content[:400])
                if len(content) > 400:
                    print("...")


if __name__ == "__main__":
    success = create_complete_ui()
    
    if success:
        print("\n✅ SUCCESS: Complete PyGent Factory UI created!")
    else:
        print("\n🔄 PARTIAL: Some files created, manual work needed")
