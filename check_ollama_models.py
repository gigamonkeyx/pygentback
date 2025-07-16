#!/usr/bin/env python3
"""
Check Available Ollama Models
Use the working PyGent Factory components to check what models are available
"""

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_ollama_models():
    """Check what Ollama models are actually available"""
    
    logger.info("🔍 CHECKING AVAILABLE OLLAMA MODELS")
    logger.info("=" * 60)
    
    try:
        # Use the working Ollama provider
        from src.ai.providers.ollama_provider import OllamaProvider
        
        # Initialize Ollama provider
        ollama_provider = OllamaProvider()
        await ollama_provider.initialize()
        
        logger.info("✅ Ollama provider initialized")
        
        # Get available models
        models = await ollama_provider.get_available_models()
        
        logger.info(f"📊 Found {len(models)} Ollama models:")
        logger.info("")
        
        coding_models = []
        general_models = []
        
        for i, model in enumerate(models, 1):
            logger.info(f"  {i}. ✅ {model}")
            
            # Categorize models
            if 'coder' in model.lower() or 'code' in model.lower():
                coding_models.append(model)
            else:
                general_models.append(model)
        
        logger.info("")
        logger.info("🛠️  CODING MODELS:")
        for model in coding_models:
            logger.info(f"     - {model}")
        
        logger.info("")
        logger.info("🧠 GENERAL MODELS:")
        for model in general_models:
            logger.info(f"     - {model}")
        
        logger.info("")
        logger.info("🚨 PROBLEM ANALYSIS:")
        logger.info(f"   ❌ CodingAgent configured for: deepseek-coder-v2:latest")
        logger.info(f"   ✅ Available coding model: {coding_models[0] if coding_models else 'NONE'}")
        
        if coding_models:
            logger.info("")
            logger.info("💡 SOLUTION:")
            logger.info(f"   Update CodingAgent to use: {coding_models[0]}")
        
        return {
            "total_models": len(models),
            "all_models": models,
            "coding_models": coding_models,
            "general_models": general_models,
            "recommended_coding_model": coding_models[0] if coding_models else None
        }
        
    except Exception as e:
        logger.error(f"💥 Failed to check Ollama models: {e}")
        return None

async def test_model_generation():
    """Test code generation with the correct model"""
    
    logger.info("🧪 TESTING CODE GENERATION WITH CORRECT MODEL")
    logger.info("=" * 60)
    
    try:
        # Get available models first
        model_info = await check_ollama_models()
        
        if not model_info or not model_info["coding_models"]:
            logger.error("❌ No coding models available for testing")
            return False
        
        correct_model = model_info["coding_models"][0]
        logger.info(f"🎯 Testing with model: {correct_model}")
        
        # Test direct Ollama call with correct model
        import aiohttp
        
        prompt = """Create a simple HTML page with a flying dragon animation using CSS3.

Requirements:
- Complete HTML page
- CSS3 animations
- Flying dragon

Please provide only the HTML code:"""
        
        payload = {
            "model": correct_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 500
            }
        }
        
        logger.info("📤 Sending test request to Ollama...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    generated_code = result.get("response", "")
                    
                    logger.info("✅ Code generation successful!")
                    logger.info(f"📏 Generated {len(generated_code)} characters")
                    
                    # Show preview
                    preview = generated_code[:300] + "..." if len(generated_code) > 300 else generated_code
                    logger.info("📄 Generated code preview:")
                    logger.info(f"   {preview}")
                    
                    # Save the generated code
                    if generated_code and len(generated_code) > 50:
                        with open("test_generated_dragon.html", "w", encoding="utf-8") as f:
                            f.write(generated_code)
                        logger.info("💾 Saved generated code to: test_generated_dragon.html")
                        return True
                    else:
                        logger.warning("⚠️ Generated code too short")
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Ollama API error: {response.status} - {error_text}")
                    return False
                    
    except Exception as e:
        logger.error(f"💥 Test failed: {e}")
        return False

async def main():
    """Main execution function"""
    
    logger.info("🚀 STARTING OLLAMA MODEL CHECK")
    logger.info("=" * 70)
    
    # Check available models
    model_info = await check_ollama_models()
    
    if model_info:
        logger.info("")
        logger.info("🧪 Testing code generation...")
        success = await test_model_generation()
        
        logger.info("")
        logger.info("🏁 ANALYSIS COMPLETE")
        logger.info("=" * 30)
        
        if success:
            logger.info("✅ Code generation works with correct model!")
            logger.info(f"🔧 Fix: Update CodingAgent to use {model_info['recommended_coding_model']}")
        else:
            logger.info("❌ Code generation still has issues")
    else:
        logger.error("💥 Could not check Ollama models")

if __name__ == "__main__":
    asyncio.run(main())
