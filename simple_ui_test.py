#!/usr/bin/env python3
"""
Simple UI Generation Test with DeepSeek-Coder
"""

import requests
import json
from pathlib import Path


def call_ollama(prompt: str) -> str:
    """Call Ollama API with requests"""
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "deepseek-coder:6.7b",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=payload, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "")
    else:
        raise Exception(f"Ollama API error: {response.status_code}")


def main():
    print("🚀 SIMPLE UI GENERATION TEST")
    print("=" * 50)
    
    prompt = """
Create a complete Vue.js component for PyGent Factory agent management.

Requirements:
- Vue 3 composition API
- Display a list of AI agents
- Include agent cards with name, status, and actions
- Add buttons to create, edit, and delete agents
- Make it responsive and modern
- Include proper styling

Generate a single .vue file with template, script, and style sections.
"""
    
    print("📝 Calling DeepSeek-Coder...")
    
    try:
        response = call_ollama(prompt)
        
        print(f"✅ Response received ({len(response)} characters)")
        
        # Create output directory
        ui_dir = Path("ui-alternative")
        ui_dir.mkdir(exist_ok=True)
        
        # Save the response
        output_file = ui_dir / "AgentManagement.vue"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response)
        
        print(f"💾 Saved to {output_file}")
        print(f"📄 File size: {output_file.stat().st_size} bytes")
        
        # Show preview
        print("\n📝 GENERATED CODE PREVIEW:")
        print("-" * 50)
        print(response[:800])
        if len(response) > 800:
            print("...")
            print(response[-200:])
        
        print("\n🎉 UI GENERATION COMPLETE!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ SUCCESS: DeepSeek-Coder generated Vue.js UI!")
    else:
        print("❌ FAILED: Could not generate UI")
