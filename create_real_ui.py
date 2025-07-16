#!/usr/bin/env python3
"""
Create Real PyGent Factory UI
Generate a complete, production-ready UI using DeepSeek-Coder
"""

import requests
import json
from pathlib import Path


def call_deepseek(prompt: str) -> str:
    """Call DeepSeek-Coder directly"""
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "deepseek-coder:6.7b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8000
        }
    }
    
    response = requests.post(url, json=payload, timeout=300)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "")
    else:
        raise Exception(f"DeepSeek API error: {response.status_code}")


def create_real_ui():
    """Generate a complete PyGent Factory UI"""
    
    print("🚀 CREATING REAL PYGENT FACTORY UI")
    print("=" * 60)
    
    # The prompt for a complete UI
    prompt = """
Create a complete, production-ready React TypeScript application for PyGent Factory.

This is an AI agent management platform. Create:

1. App.tsx - Main application with routing
2. components/Layout.tsx - Main layout with navigation
3. pages/Dashboard.tsx - Dashboard with agent overview
4. pages/Agents.tsx - Agent management page
5. pages/Tasks.tsx - Task management page
6. components/AgentCard.tsx - Individual agent display
7. components/TaskCard.tsx - Individual task display
8. types/index.ts - TypeScript type definitions
9. api/client.ts - API client for backend communication
10. package.json - Dependencies and scripts

Requirements:
- Modern React 18 with TypeScript
- React Router for navigation
- TailwindCSS for styling
- Responsive design
- Real API integration with PyGent Factory backend
- Professional UI/UX
- Dark/light theme support
- Real-time updates
- Error handling
- Loading states

Make it production-ready with proper error boundaries, loading states, and professional styling.

Generate each file with clear file path comments and complete, functional code.
"""
    
    print("📝 Generating complete UI with DeepSeek-Coder...")
    print("⏳ This may take a few minutes for a complete application...")
    
    try:
        response = call_deepseek(prompt)
        
        print(f"✅ Generated {len(response):,} characters of code")
        
        # Create UI directory
        ui_dir = Path("pygent-factory-ui")
        ui_dir.mkdir(exist_ok=True)
        
        # Save the complete response
        with open(ui_dir / "generated_code.txt", "w", encoding="utf-8") as f:
            f.write(response)
        
        print(f"💾 Saved complete response to {ui_dir}/generated_code.txt")
        
        # Try to extract individual files
        files_created = extract_files_from_response(response, ui_dir)
        
        print(f"\n📊 RESULTS:")
        print(f"   Total response: {len(response):,} characters")
        print(f"   Files extracted: {files_created}")
        
        # List created files
        print(f"\n📁 Created files in {ui_dir}/:")
        for file_path in sorted(ui_dir.rglob("*")):
            if file_path.is_file():
                size = file_path.stat().st_size
                rel_path = file_path.relative_to(ui_dir)
                print(f"   📄 {rel_path} ({size:,} bytes)")
        
        # Show preview of main files
        show_file_previews(ui_dir)
        
        print(f"\n🎉 REAL UI CREATED!")
        print(f"📁 Location: {ui_dir.absolute()}")
        print(f"🚀 Ready for development and deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def extract_files_from_response(response: str, ui_dir: Path) -> int:
    """Extract individual files from the AI response"""
    
    files_created = 0
    lines = response.split('\n')
    current_file = None
    current_content = []
    in_code_block = False
    
    for line in lines:
        # Check for file path indicators
        if any(pattern in line.lower() for pattern in [
            'app.tsx', 'layout.tsx', 'dashboard.tsx', 'agents.tsx', 'tasks.tsx',
            'agentcard.tsx', 'taskcard.tsx', 'index.ts', 'client.ts', 'package.json',
            '// src/', '// components/', '// pages/', '// types/', '// api/'
        ]):
            # Save previous file
            if current_file and current_content:
                save_extracted_file(ui_dir, current_file, current_content)
                files_created += 1
            
            # Determine new file path
            current_file = determine_file_path(line)
            current_content = []
            in_code_block = False
        
        elif line.strip().startswith('```'):
            in_code_block = not in_code_block
            if not in_code_block and current_file:
                # End of code block, save file
                save_extracted_file(ui_dir, current_file, current_content)
                files_created += 1
                current_file = None
                current_content = []
        
        elif current_file and (in_code_block or not line.strip().startswith('```')):
            current_content.append(line)
    
    # Save last file
    if current_file and current_content:
        save_extracted_file(ui_dir, current_file, current_content)
        files_created += 1
    
    return files_created


def determine_file_path(line: str) -> str:
    """Determine file path from line content"""
    
    line_lower = line.lower()
    
    if 'app.tsx' in line_lower:
        return 'src/App.tsx'
    elif 'layout.tsx' in line_lower:
        return 'src/components/Layout.tsx'
    elif 'dashboard.tsx' in line_lower:
        return 'src/pages/Dashboard.tsx'
    elif 'agents.tsx' in line_lower:
        return 'src/pages/Agents.tsx'
    elif 'tasks.tsx' in line_lower:
        return 'src/pages/Tasks.tsx'
    elif 'agentcard.tsx' in line_lower:
        return 'src/components/AgentCard.tsx'
    elif 'taskcard.tsx' in line_lower:
        return 'src/components/TaskCard.tsx'
    elif 'index.ts' in line_lower and 'types' in line_lower:
        return 'src/types/index.ts'
    elif 'client.ts' in line_lower:
        return 'src/api/client.ts'
    elif 'package.json' in line_lower:
        return 'package.json'
    else:
        return None


def save_extracted_file(ui_dir: Path, file_path: str, content: list):
    """Save extracted file content"""
    
    if not file_path:
        return
    
    full_path = ui_dir / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean content
    clean_content = []
    for line in content:
        if not line.strip().startswith('```'):
            clean_content.append(line)
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(clean_content))


def show_file_previews(ui_dir: Path):
    """Show previews of key files"""
    
    key_files = [
        'src/App.tsx',
        'src/components/Layout.tsx', 
        'src/pages/Dashboard.tsx',
        'package.json'
    ]
    
    print(f"\n📝 FILE PREVIEWS:")
    print("-" * 60)
    
    for file_path in key_files:
        full_path = ui_dir / file_path
        if full_path.exists():
            print(f"\n📄 {file_path}:")
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                preview = content[:300]
                print(preview)
                if len(content) > 300:
                    print("...")


if __name__ == "__main__":
    success = create_real_ui()
    
    if success:
        print("\n✅ SUCCESS: Real PyGent Factory UI created!")
        print("🚀 Ready for npm install and development!")
    else:
        print("\n❌ FAILED: Could not create UI")
