#!/usr/bin/env python3
"""
Quick Agent Coding Test
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_coding():
    print("🤖 QUICK AGENT CODING TEST")
    print("=" * 40)
    
    try:
        from core.agent_factory import AgentFactory
        from agents.base_agent import AgentMessage
        
        factory = AgentFactory()
        
        # Create coding agent
        agent = await factory.create_agent(
            agent_type="coding",
            name="test_coder"
        )
        
        print(f"✅ Created agent: {agent.agent_id}")
        
        # Simple coding task
        task = "Create a simple React component that displays a button and counts clicks when pressed. Use TypeScript and React hooks."
        
        message = AgentMessage(content=task)
        response = await agent.process_message(message)
        
        print(f"✅ Got response: {len(response.content)} characters")
        
        # Save the code
        output_dir = Path("generated-code")
        output_dir.mkdir(exist_ok=True)
        
        code_file = output_dir / "ClickCounter.tsx"
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(response.content)
        
        print(f"💾 Saved to: {code_file}")
        
        # Show preview
        print("\n📝 GENERATED CODE:")
        print("-" * 40)
        print(response.content[:800])
        if len(response.content) > 800:
            print("...")
        
        # Check if it looks like React code
        has_react = "React" in response.content or "useState" in response.content
        has_typescript = "interface" in response.content or ": number" in response.content
        has_component = "function" in response.content or "const" in response.content
        
        print(f"\n🔍 CODE ANALYSIS:")
        print(f"   React features: {'✅' if has_react else '❌'}")
        print(f"   TypeScript: {'✅' if has_typescript else '❌'}")
        print(f"   Component structure: {'✅' if has_component else '❌'}")
        
        if has_react and has_component:
            print("\n🎉 SUCCESS: AGENT CAN CODE!")
            return True
        else:
            print("\n🔄 PARTIAL SUCCESS: Needs improvement")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_coding())
    
    if success:
        print("\n✅ AGENTS CAN CODE!")
    else:
        print("\n❌ AGENTS NEED WORK")
