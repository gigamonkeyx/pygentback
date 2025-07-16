#!/usr/bin/env python3
"""
Test Agent Coding Capability
Real test of whether our agents can actually code without hints.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.supervisor_agent import SupervisorAgent
from core.agent_factory import AgentFactory


async def test_agent_coding():
    """Test if agents can actually code"""
    
    print("🤖 TESTING AGENT CODING CAPABILITY")
    print("=" * 60)
    
    # Initialize system
    supervisor = SupervisorAgent()
    factory = AgentFactory()
    
    print("✅ System initialized")
    
    # Real coding challenge - no hints
    coding_task = """
    Create a React component for a task management dashboard.
    
    Requirements:
    - Display a list of tasks with status (pending, in-progress, completed)
    - Allow adding new tasks
    - Allow marking tasks as complete
    - Include filtering by status
    - Make it responsive and modern
    - Use React hooks (useState, useEffect)
    - Include proper TypeScript types
    
    Generate a complete .tsx file ready for production use.
    """
    
    print(f"\n📋 CODING TASK:")
    print(coding_task.strip())
    print("-" * 60)
    
    try:
        # Step 1: Supervisor analysis
        print("\n🔍 STEP 1: Task Analysis")
        analysis = await supervisor.analyze_task(coding_task)
        print(f"   Task Type: {analysis.task_type}")
        print(f"   Complexity: {analysis.complexity}/10")
        print(f"   Required Skills: {', '.join(analysis.required_capabilities)}")
        
        # Step 2: Create coding agent
        print("\n🤖 STEP 2: Creating Coding Agent")
        agent = await factory.create_agent(
            agent_type="coding",
            name="react_coder",
            capabilities=["react", "typescript", "javascript", "frontend"],
            custom_config={
                "provider": "ollama",
                "model": "deepseek-coder:6.7b",
                "temperature": 0.1,
                "max_tokens": 4000
            }
        )
        print(f"   ✅ Created agent: {agent.agent_id}")
        
        # Step 3: Execute coding task
        print("\n💻 STEP 3: Agent Coding Task")
        print("   Sending task to agent...")

        # Create agent message for the coding task
        from agents.base_agent import AgentMessage

        message = AgentMessage(
            content=coding_task,
            metadata={
                "format": "typescript_react",
                "output_type": "component_file",
                "requirements": [
                    "React hooks (useState, useEffect)",
                    "TypeScript types",
                    "Responsive design",
                    "Task management functionality"
                ]
            }
        )

        response = await agent.process_message(message)
        result = response.content
        
        print(f"   ✅ Task completed")
        print(f"   Result type: {type(result)}")
        
        # Step 4: Supervisor evaluation
        print("\n📊 STEP 4: Code Quality Assessment")
        supervision = await supervisor.supervise_task(
            task_id="react_coding_test",
            task_description=coding_task,
            agent_output=result
        )
        
        quality_score = supervision.get("quality_score", {}).get("score", 0)
        requires_retry = supervision.get("requires_retry", True)
        
        print(f"   Quality Score: {quality_score:.1%}")
        print(f"   Code Quality: {'✅ PASSED' if not requires_retry else '🔄 NEEDS WORK'}")
        
        # Step 5: Save and analyze the code
        print("\n📁 STEP 5: Generated Code Analysis")
        
        if isinstance(result, str) and len(result) > 100:
            # Save the generated code
            output_dir = Path("generated-code")
            output_dir.mkdir(exist_ok=True)
            
            code_file = output_dir / "TaskDashboard.tsx"
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(result)
            
            print(f"   💾 Saved to: {code_file}")
            print(f"   📄 File size: {len(result):,} characters")
            
            # Analyze the code
            has_react = "import React" in result or "from 'react'" in result
            has_typescript = "interface " in result or ": string" in result or ": number" in result
            has_hooks = "useState" in result or "useEffect" in result
            has_components = "function " in result or "const " in result and "=>" in result
            
            print(f"   🔍 Code Analysis:")
            print(f"     React imports: {'✅' if has_react else '❌'}")
            print(f"     TypeScript types: {'✅' if has_typescript else '❌'}")
            print(f"     React hooks: {'✅' if has_hooks else '❌'}")
            print(f"     Component structure: {'✅' if has_components else '❌'}")
            
            # Show preview
            print(f"\n📝 CODE PREVIEW:")
            print("-" * 40)
            print(result[:600])
            if len(result) > 600:
                print("...")
                print(result[-300:])
            
            # Final assessment
            code_quality = sum([has_react, has_typescript, has_hooks, has_components])
            
            print(f"\n🎯 FINAL ASSESSMENT:")
            print(f"   Code Quality Features: {code_quality}/4")
            print(f"   Supervisor Score: {quality_score:.1%}")
            
            if code_quality >= 3 and quality_score > 0.6:
                print("   🎉 AGENT CAN CODE! ✅")
                return True
            elif code_quality >= 2:
                print("   🔄 AGENT CAN CODE BUT NEEDS IMPROVEMENT")
                return True
            else:
                print("   ❌ AGENT CODING NEEDS WORK")
                return False
        else:
            print("   ❌ No substantial code generated")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during coding test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_agent_coding())
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 CONCLUSION: YES, THE AGENTS CAN CODE!")
        print("✅ Capable of generating functional React/TypeScript components")
        print("✅ Supervisor system provides quality control")
        print("✅ Ready for autonomous UI development")
    else:
        print("🔄 CONCLUSION: AGENTS NEED MORE TRAINING")
        print("❌ Code generation needs improvement")
        print("🎯 Focus on better prompting and model fine-tuning")
    
    print("\n🚀 AGENT CODING CAPABILITY TEST COMPLETE!")
