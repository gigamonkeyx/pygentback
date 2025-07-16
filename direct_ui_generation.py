#!/usr/bin/env python3
"""
Direct UI Generation - Create actual Vue.js files
Uses Ollama DeepSeek directly to generate UI without complex agent system.
"""

import asyncio
import json
from pathlib import Path
import aiohttp


async def call_ollama_directly(prompt: str) -> str:
    """Call Ollama API directly"""
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "deepseek-r1:8b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 4000
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("response", "")
            else:
                raise Exception(f"Ollama API error: {response.status}")


async def generate_vue_ui():
    """Generate Vue.js UI using DeepSeek-R1 via Ollama"""
    
    print("🚀 DIRECT UI GENERATION WITH DEEPSEEK-R1")
    print("=" * 60)
    
    # The prompt - no hints about structure
    prompt = """
Create a complete Vue.js application for PyGent Factory - an AI agent management system.

Requirements:
1. Create a modern, professional UI
2. Include pages for: Dashboard, Agents, Tasks, Results
3. Use Vue 3 composition API
4. Include routing with Vue Router
5. Make it responsive
6. Add API integration for agent management
7. Include proper component structure

Generate the following files:
1. App.vue (main application component)
2. router/index.js (routing configuration)
3. views/Dashboard.vue (dashboard page)
4. views/Agents.vue (agent management page)
5. views/Tasks.vue (task management page)
6. components/AgentCard.vue (agent display component)
7. package.json (dependencies)

For each file, start with a comment showing the filename, then provide the complete code.
Make it production-ready with proper error handling and modern design.
"""
    
    print("📝 Sending prompt to DeepSeek-R1...")
    print(f"Prompt length: {len(prompt)} characters")
    
    try:
        # Call DeepSeek-R1
        response = await call_ollama_directly(prompt)
        
        print(f"✅ Received response from DeepSeek-R1")
        print(f"Response length: {len(response)} characters")
        
        if len(response) < 100:
            print("❌ Response too short, something went wrong")
            print(f"Response: {response}")
            return
        
        # Create output directory
        ui_dir = Path("ui-alternative")
        ui_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Creating UI files in {ui_dir}/")
        
        # Save the full response first
        with open(ui_dir / "deepseek_response.txt", "w", encoding="utf-8") as f:
            f.write(response)
        print(f"   💾 Saved full response to deepseek_response.txt")
        
        # Try to extract individual files from the response
        files_created = 0
        
        # Look for file patterns in the response
        lines = response.split('\n')
        current_file = None
        current_content = []
        
        for line in lines:
            # Check if this line indicates a new file
            if any(filename in line.lower() for filename in [
                'app.vue', 'router/index.js', 'dashboard.vue', 'agents.vue', 
                'tasks.vue', 'agentcard.vue', 'package.json'
            ]):
                # Save previous file if we have one
                if current_file and current_content:
                    file_path = ui_dir / current_file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write('\n'.join(current_content))
                    
                    print(f"   📄 Created {current_file} ({len(current_content)} lines)")
                    files_created += 1
                
                # Start new file
                if 'app.vue' in line.lower():
                    current_file = 'App.vue'
                elif 'router/index.js' in line.lower():
                    current_file = 'router/index.js'
                elif 'dashboard.vue' in line.lower():
                    current_file = 'views/Dashboard.vue'
                elif 'agents.vue' in line.lower():
                    current_file = 'views/Agents.vue'
                elif 'tasks.vue' in line.lower():
                    current_file = 'views/Tasks.vue'
                elif 'agentcard.vue' in line.lower():
                    current_file = 'components/AgentCard.vue'
                elif 'package.json' in line.lower():
                    current_file = 'package.json'
                else:
                    current_file = None
                
                current_content = []
            
            elif current_file:
                # Add line to current file content
                current_content.append(line)
        
        # Save the last file
        if current_file and current_content:
            file_path = ui_dir / current_file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write('\n'.join(current_content))
            
            print(f"   📄 Created {current_file} ({len(current_content)} lines)")
            files_created += 1
        
        # If file extraction didn't work well, create a basic structure
        if files_created < 3:
            print("\n🔧 File extraction incomplete, creating basic structure...")
            
            # Create a basic App.vue from the response
            app_vue_content = extract_vue_component(response, "App")
            if app_vue_content:
                with open(ui_dir / "App.vue", "w", encoding="utf-8") as f:
                    f.write(app_vue_content)
                print(f"   📄 Created App.vue from extracted content")
                files_created += 1
        
        # Show what was created
        print(f"\n📊 GENERATION RESULTS:")
        print(f"   Files created: {files_created}")
        print(f"   Response length: {len(response):,} characters")
        
        # List all files in the directory
        print(f"\n📁 Files in {ui_dir}/:")
        for file_path in ui_dir.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   📄 {file_path.relative_to(ui_dir)} ({size:,} bytes)")
        
        # Show a preview of the main response
        print(f"\n📝 DEEPSEEK-R1 RESPONSE PREVIEW:")
        print("-" * 60)
        print(response[:1000])
        if len(response) > 1000:
            print("...")
            print(response[-500:])
        
        print("\n🎉 UI GENERATION COMPLETE!")
        print(f"Check the {ui_dir}/ directory for generated files.")
        
        return response
        
    except Exception as e:
        print(f"❌ Error calling DeepSeek-R1: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_vue_component(text: str, component_name: str) -> str:
    """Extract a Vue component from text"""
    
    # Look for Vue component patterns
    lines = text.split('\n')
    in_component = False
    component_lines = []
    
    for line in lines:
        if '<template>' in line:
            in_component = True
            component_lines = [line]
        elif in_component:
            component_lines.append(line)
            if '</style>' in line or ('export default' in line and '}' in line):
                break
    
    if component_lines and len(component_lines) > 5:
        return '\n'.join(component_lines)
    
    return ""


if __name__ == "__main__":
    result = asyncio.run(generate_vue_ui())
    
    if result:
        print("\n✅ SUCCESS: Vue.js UI generated by DeepSeek-R1!")
    else:
        print("\n❌ FAILED: Could not generate UI")
