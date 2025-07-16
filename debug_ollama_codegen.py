#!/usr/bin/env python3
"""
Debug Ollama Code Generation
Test the raw Ollama response to understand the codegen issues
"""

import asyncio
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_raw_ollama():
    """Test raw Ollama API directly"""
    
    logger.info("🔍 TESTING RAW OLLAMA API")
    logger.info("=" * 50)
    
    try:
        import aiohttp
        
        # Test the exact same request the CodingAgent makes
        prompt = """You are an expert programmer. Generate clean, efficient, and well-commented code for the following request:

Create a complete HTML page with a flying dragon animation.

Requirements:
- Create a complete HTML page with a flying dragon
- Use CSS animations for smooth dragon flight
- Include a beautiful background (sky, clouds, etc.)
- Make the dragon fly across the screen continuously
- Use modern CSS3 animations and transforms
- Include responsive design
- Add some magical effects (sparkles, trails, etc.)

Please provide only the code without explanations.

Code:"""
        
        payload = {
            "model": "deepseek-coder-v2:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1000
            }
        }
        
        logger.info("📤 Sending request to Ollama...")
        logger.info(f"   Model: {payload['model']}")
        logger.info(f"   Temperature: {payload['options']['temperature']}")
        logger.info(f"   Max tokens: {payload['options']['num_predict']}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                
                logger.info(f"📥 Response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    
                    logger.info("✅ Raw Ollama Response:")
                    logger.info(f"   Model: {result.get('model', 'unknown')}")
                    logger.info(f"   Done: {result.get('done', 'unknown')}")
                    logger.info(f"   Total duration: {result.get('total_duration', 'unknown')}")
                    logger.info(f"   Response length: {len(result.get('response', ''))}")
                    
                    raw_response = result.get("response", "")
                    
                    logger.info("📄 Response content preview:")
                    preview = raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
                    logger.info(f"   {preview}")
                    
                    # Save full response
                    with open("ollama_raw_response.txt", "w", encoding="utf-8") as f:
                        f.write(raw_response)
                    
                    logger.info("💾 Full response saved to: ollama_raw_response.txt")
                    
                    # Test language detection
                    detected_lang = detect_language_test(prompt)
                    logger.info(f"🔍 Language detection result: {detected_lang}")
                    
                    # Test code formatting
                    formatted_code = format_code_test(raw_response, "html")
                    logger.info(f"🎨 Formatted code length: {len(formatted_code)}")
                    
                    if formatted_code and len(formatted_code) > 50:
                        # Save formatted HTML
                        with open("ollama_generated_dragon.html", "w", encoding="utf-8") as f:
                            f.write(formatted_code)
                        logger.info("💾 Formatted HTML saved to: ollama_generated_dragon.html")
                        
                        return {
                            "success": True,
                            "raw_response": raw_response,
                            "formatted_code": formatted_code,
                            "detected_language": detected_lang
                        }
                    else:
                        logger.error("❌ No usable code generated")
                        return {"success": False, "error": "No usable code generated"}
                        
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Ollama API error: {response.status} - {error_text}")
                    return {"success": False, "error": f"API error: {response.status}"}
                    
    except Exception as e:
        logger.error(f"💥 Test failed: {e}")
        return {"success": False, "error": str(e)}

def detect_language_test(user_input: str) -> str:
    """Test the language detection logic"""
    
    supported_languages = [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", 
        "rust", "php", "ruby", "swift", "kotlin", "scala", "r", "sql",
        "html", "css", "bash", "powershell", "yaml", "json", "xml"
    ]
    
    input_lower = user_input.lower()
    
    # Check for explicit language mentions
    for lang in supported_languages:
        if lang in input_lower:
            return lang
    
    # Check for language-specific keywords
    if any(keyword in input_lower for keyword in ["def ", "import ", "python", ".py"]):
        return "python"
    elif any(keyword in input_lower for keyword in ["function", "const", "let", "var", "javascript", ".js"]):
        return "javascript"
    elif any(keyword in input_lower for keyword in ["class", "public", "private", "java", ".java"]):
        return "java"
    elif any(keyword in input_lower for keyword in ["#include", "int main", "c++", ".cpp"]):
        return "c++"
    elif any(keyword in input_lower for keyword in ["using", "namespace", "c#", ".cs"]):
        return "c#"
    
    # Default to Python if no language detected
    return "python"

def format_code_test(code_response: str, language: str) -> str:
    """Test the code formatting logic"""
    
    # Remove common prefixes/suffixes
    cleaned_code = code_response.strip()
    
    # Remove markdown code blocks if present
    if cleaned_code.startswith("```"):
        lines = cleaned_code.split('\n')
        if len(lines) > 2:
            cleaned_code = '\n'.join(lines[1:-1])
    
    return cleaned_code

async def test_coding_agent_flow():
    """Test the complete CodingAgent flow"""
    
    logger.info("🤖 TESTING CODING AGENT FLOW")
    logger.info("=" * 40)
    
    try:
        # Import the CodingAgent
        from src.core.agent_factory import AgentFactory
        from src.core.agent import AgentMessage, MessageType
        
        # Create agent factory
        agent_factory = AgentFactory()
        await agent_factory.initialize()
        
        # Create coding agent
        coding_agent = await agent_factory.create_agent(
            agent_type="coding",
            name="Debug Coding Agent",
            capabilities=["code_generation"],
            custom_config={
                "model_name": "deepseek-coder-v2:latest",
                "temperature": 0.3,
                "max_tokens": 1000
            }
        )
        
        logger.info(f"✅ Created coding agent: {coding_agent.config.agent_id}")
        
        # Create message
        message = AgentMessage(
            type=MessageType.REQUEST,
            content={
                "content": "Create a complete HTML page with a flying dragon animation using CSS3."
            },
            metadata={"task_type": "code_generation"}
        )
        
        logger.info("📤 Sending message to coding agent...")
        
        # Process message
        response = await coding_agent.process_message(message)
        
        logger.info("📥 Received response from coding agent")
        logger.info(f"   Response type: {type(response.content)}")
        logger.info(f"   Content keys: {list(response.content.keys()) if isinstance(response.content, dict) else 'Not a dict'}")
        
        if isinstance(response.content, dict):
            logger.info("📋 Response analysis:")
            for key, value in response.content.items():
                if isinstance(value, str):
                    logger.info(f"   {key}: {value[:100]}..." if len(value) > 100 else f"   {key}: {value}")
                else:
                    logger.info(f"   {key}: {value}")
        
        return response.content
        
    except Exception as e:
        logger.error(f"💥 Coding agent test failed: {e}")
        return {"success": False, "error": str(e)}

async def main():
    """Main execution function"""
    
    logger.info("🚀 STARTING OLLAMA CODEGEN DEBUG")
    logger.info("=" * 70)
    
    # Test 1: Raw Ollama API
    raw_result = await test_raw_ollama()
    
    print("\n" + "="*50)
    
    # Test 2: CodingAgent flow
    agent_result = await test_coding_agent_flow()
    
    logger.info("🏁 DEBUG COMPLETE")
    logger.info("=" * 30)
    
    if raw_result.get("success"):
        logger.info("✅ Raw Ollama: SUCCESS")
    else:
        logger.info(f"❌ Raw Ollama: {raw_result.get('error')}")
    
    if agent_result and not agent_result.get("error"):
        logger.info("✅ CodingAgent: SUCCESS")
    else:
        logger.info(f"❌ CodingAgent: {agent_result.get('error') if agent_result else 'Failed'}")

if __name__ == "__main__":
    asyncio.run(main())
