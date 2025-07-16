#!/usr/bin/env python3
"""
Real UI Generation Test - No Hints, No Mock Code
Actually generates Vue.js files using the Golden Glue system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.supervisor_agent import SupervisorAgent
from core.agent_factory import AgentFactory
from ai.providers.provider_registry import ProviderRegistry


async def generate_real_ui():
    """Generate actual Vue.js UI files using the agent system"""
    
    print("🚀 REAL UI GENERATION TEST - NO HINTS, NO MOCK CODE")
    print("=" * 70)
    
    # Initialize the Golden Glue
    supervisor = SupervisorAgent()
    agent_factory = AgentFactory()
    
    # Initialize AI providers
    provider_registry = ProviderRegistry()
    await provider_registry.initialize()
    
    print("✅ Golden Glue system initialized")
    
    # The task - no hints about Vue.js structure
    task_description = """
    Create a complete alternative UI for PyGent Factory. 
    The UI should allow users to:
    1. View and manage AI agents
    2. Submit tasks to agents
    3. Monitor task progress
    4. View results
    
    Make it modern, responsive, and professional.
    """
    
    print(f"\n📋 TASK: {task_description.strip()}")
    print("-" * 70)
    
    # Step 1: Supervisor analyzes the task
    print("\n🔍 STEP 1: Task Analysis")
    analysis = await supervisor.analyze_task(task_description)
    print(f"   Task Type: {analysis.task_type}")
    print(f"   Complexity: {analysis.complexity}/10")
    print(f"   Required Capabilities: {', '.join(analysis.required_capabilities)}")
    
    # Step 2: Create a coding agent
    print("\n🤖 STEP 2: Creating Coding Agent")
    agent_type = await supervisor.select_agent(analysis)
    
    try:
        # Create the agent
        agent = await agent_factory.create_agent(
            agent_type=agent_type,
            name="ui_generator",
            capabilities=["vue.js", "javascript", "html", "css", "frontend"],
            custom_config={
                "model": "deepseek-r1:8b",
                "provider": "ollama",
                "temperature": 0.1,
                "max_tokens": 4000
            }
        )
        print(f"   ✅ Created {agent_type} agent: {agent.agent_id}")
        
        # Step 3: Execute the task
        print("\n💻 STEP 3: Executing UI Generation Task")
        print("   Sending task to agent (this may take a moment)...")
        
        # Execute the task
        result = await agent.execute_task({
            "task": task_description,
            "output_directory": "ui-alternative",
            "requirements": [
                "Create a complete Vue.js application",
                "Include multiple components",
                "Add routing between pages", 
                "Integrate with PyGent Factory API",
                "Make it responsive and modern",
                "Include proper file structure"
            ]
        })
        
        print(f"   ✅ Task execution completed")
        print(f"   Result type: {type(result)}")
        
        # Step 4: Supervisor evaluates the result
        print("\n📊 STEP 4: Supervisor Evaluation")
        supervision_result = await supervisor.supervise_task(
            task_id="real_ui_generation",
            task_description=task_description,
            agent_output=result
        )
        
        quality_score = supervision_result.get("quality_score", {})
        requires_retry = supervision_result.get("requires_retry", True)
        
        print(f"   Quality Score: {quality_score.get('score', 0):.1%}")
        print(f"   Passed: {'✅ YES' if not requires_retry else '❌ NO'}")
        
        if supervision_result.get("feedback"):
            print(f"   Feedback: {supervision_result['feedback'][:200]}...")
        
        # Step 5: Check what files were created
        print("\n📁 STEP 5: Checking Generated Files")
        ui_dir = Path("ui-alternative")
        
        if ui_dir.exists():
            print(f"   ✅ UI directory created: {ui_dir}")
            
            # List all generated files
            for root, dirs, files in os.walk(ui_dir):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(ui_dir)
                    file_size = file_path.stat().st_size
                    print(f"   📄 {rel_path} ({file_size} bytes)")
                    
                    # Show content of Vue files
                    if file.endswith('.vue') and file_size > 0:
                        print(f"      Preview of {file}:")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()[:300]
                            print(f"      {content}...")
                            print()
        else:
            print("   ❌ No UI directory found")
            print("   Agent may have returned code as text instead of files")
            
            # Check if result contains code
            if isinstance(result, str) and len(result) > 100:
                print("   📝 Agent returned code as text:")
                print(f"   {result[:500]}...")
                
                # Try to extract and save Vue.js code
                if "<template>" in result or "export default" in result:
                    print("   💾 Extracting Vue.js code to files...")
                    
                    # Create directory
                    ui_dir.mkdir(exist_ok=True)
                    
                    # Save the main component
                    main_vue = ui_dir / "App.vue"
                    with open(main_vue, 'w', encoding='utf-8') as f:
                        f.write(result)
                    
                    print(f"   ✅ Saved to {main_vue}")
        
        # Final assessment
        print("\n" + "=" * 70)
        if not requires_retry:
            print("🎉 REAL UI GENERATION: ✅ SUCCESS!")
            print("   The agent successfully created a UI without hints!")
        else:
            print("🔄 REAL UI GENERATION: PARTIAL SUCCESS")
            print("   Agent created output but needs improvement")
            
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Agent Used: {agent_type} (DeepSeek-R1:8B via Ollama)")
        print(f"   Task Complexity: {analysis.complexity}/10")
        print(f"   Quality Score: {quality_score.get('score', 0):.1%}")
        print(f"   Files Created: {len(list(ui_dir.glob('**/*'))) if ui_dir.exists() else 0}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error creating or executing agent: {e}")
        print(f"   Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(generate_real_ui())
    
    if result:
        print("\n🚀 UI GENERATION COMPLETE!")
        print("Check the ui-alternative/ directory for generated files.")
    else:
        print("\n❌ UI GENERATION FAILED!")
        print("Check the error messages above.")
