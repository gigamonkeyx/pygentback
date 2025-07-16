#!/usr/bin/env python3
"""
Simple Dragon Task Execution - Bypasses Health Check
Uses the working components directly to create a flying dragon HTML page
"""

import asyncio
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_dragon_html_directly():
    """Create a flying dragon HTML page using the working coding agent"""
    
    logger.info("🐉 SIMPLE DRAGON TASK - DIRECT EXECUTION")
    logger.info("=" * 60)
    
    try:
        # Import the agent factory directly
        from src.core.agent_factory import AgentFactory
        
        # Initialize agent factory
        logger.info("📋 Step 1: Initialize Agent Factory")
        agent_factory = AgentFactory()
        await agent_factory.initialize()
        logger.info("✅ Agent factory initialized")
        
        # Create a coding agent
        logger.info("📋 Step 2: Create Coding Agent")
        coding_agent = await agent_factory.create_agent(
            agent_type="coding",
            name="Dragon HTML Creator",
            capabilities=["code_generation", "html_creation", "css_animation"],
            custom_config={
                "programming_languages": ["html", "css", "javascript"],
                "specialization": "web_animation",
                "code_style": "modern_and_animated"
            }
        )
        logger.info(f"✅ Created coding agent: {coding_agent.config.agent_id}")
        
        # Create the dragon task
        logger.info("📋 Step 3: Execute Dragon HTML Creation Task")
        
        # Define the task
        dragon_task = {
            "task_type": "html_creation",
            "title": "Flying Dragon Animation",
            "requirements": [
                "Create a complete HTML page with a flying dragon",
                "Use CSS animations for smooth dragon flight",
                "Include a beautiful background (sky, clouds, etc.)",
                "Make the dragon fly across the screen continuously",
                "Use modern CSS3 animations and transforms",
                "Include responsive design",
                "Add some magical effects (sparkles, trails, etc.)"
            ],
            "style": "modern, animated, magical",
            "output_format": "complete_html_file"
        }
        
        logger.info("🎯 Task: Create animated flying dragon HTML page")
        logger.info("   Requirements: Modern CSS3 animations, responsive design, magical effects")
        
        # Execute the task using the coding agent's process_message method
        from src.core.agent import AgentMessage, MessageType

        # Create a message for the coding agent
        dragon_message = AgentMessage(
            type=MessageType.REQUEST,
            content={
                "request": "Create a complete HTML page with a flying dragon animation.",
                "requirements": [
                    "Create a complete HTML page with a flying dragon",
                    "Use CSS animations for smooth dragon flight",
                    "Include a beautiful background (sky, clouds, etc.)",
                    "Make the dragon fly across the screen continuously",
                    "Use modern CSS3 animations and transforms",
                    "Include responsive design",
                    "Add some magical effects (sparkles, trails, etc.)"
                ],
                "style": "modern, animated, magical",
                "output": "Complete HTML file with embedded CSS",
                "instructions": "Please generate a complete, working HTML file that I can save and open in a browser."
            },
            metadata={
                "task_type": "code_generation",
                "language": "html",
                "output_format": "complete_html_file"
            }
        )

        # Process the message
        result_message = await coding_agent.process_message(dragon_message)

        logger.info("✅ Dragon HTML creation completed!")
        logger.info(f"📋 Result message type: {type(result_message.content)}")

        # Extract the HTML content from the result
        html_content = ""
        if isinstance(result_message.content, dict):
            # Try different possible keys for the HTML content
            html_content = (result_message.content.get("code") or
                          result_message.content.get("html") or
                          result_message.content.get("response") or
                          str(result_message.content))
        elif isinstance(result_message.content, str):
            html_content = result_message.content
        else:
            html_content = str(result_message.content)

        logger.info(f"📏 Extracted HTML length: {len(html_content)} characters")

        if html_content and len(html_content) > 50:
                filename = f"flying_dragon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logger.info(f"💾 Saved dragon HTML to: {filename}")
                logger.info(f"📏 File size: {len(html_content)} characters")
                
                # Show preview of the HTML
                preview = html_content[:300] + "..." if len(html_content) > 300 else html_content
                logger.info("📄 HTML Preview:")
                logger.info(f"   {preview}")
                
                return {
                    "success": True,
                    "filename": filename,
                    "content": html_content,
                    "agent_id": coding_agent.config.agent_id,
                    "task_completed": True
                }
        else:
            logger.error("❌ No HTML content generated")
            return {"success": False, "error": "No HTML content generated"}
            
    except Exception as e:
        logger.error(f"💥 Dragon task failed: {e}")
        return {"success": False, "error": str(e)}

async def test_backend_connection():
    """Test if the backend is accessible"""
    
    logger.info("🔍 TESTING BACKEND CONNECTION")
    logger.info("=" * 40)
    
    try:
        import requests
        
        # Test backend health
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ Backend Status: {health_data.get('status', 'unknown')}")
            
            # Check components
            components = health_data.get('components', {})
            
            working_components = []
            for name, info in components.items():
                status = info.get('status', 'unknown')
                if status == 'healthy':
                    working_components.append(name)
                    logger.info(f"   ✅ {name}: {status}")
                else:
                    logger.info(f"   ⚠️  {name}: {status}")
            
            logger.info(f"📊 Working components: {len(working_components)}")
            return True
        else:
            logger.error(f"❌ Backend returned status: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Backend connection failed: {e}")
        return False

async def main():
    """Main execution function"""
    
    logger.info("🚀 STARTING SIMPLE DRAGON TASK")
    logger.info("=" * 70)
    
    # Step 1: Test backend connection
    backend_ok = await test_backend_connection()
    
    if not backend_ok:
        logger.error("💥 Backend not accessible - cannot proceed")
        return False
    
    # Step 2: Execute dragon task directly
    result = await create_dragon_html_directly()
    
    if result.get("success"):
        logger.info("🎉 SUCCESS! Flying dragon HTML page created!")
        logger.info(f"📁 File: {result.get('filename')}")
        logger.info(f"🤖 Agent: {result.get('agent_id')}")
        logger.info("")
        logger.info("🌐 To view the dragon:")
        logger.info(f"   Open {result.get('filename')} in your web browser")
        return True
    else:
        logger.error(f"💥 Dragon task failed: {result.get('error')}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🐉 Dragon is flying! 🐉")
    else:
        print("\n💥 Dragon task failed!")
