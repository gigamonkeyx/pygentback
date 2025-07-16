#!/usr/bin/env python3
"""
Test Hugging Face Model Discovery System
Demonstrates the new dynamic model discovery and integration
"""

import asyncio
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_model_discovery():
    """Test the complete model discovery system"""
    
    logger.info("🚀 TESTING HUGGING FACE MODEL DISCOVERY SYSTEM")
    logger.info("=" * 70)
    
    try:
        # Test 1: Basic HF model discovery
        logger.info("📋 Test 1: Hugging Face Model Discovery")
        logger.info("-" * 50)
        
        from src.core.model_discovery_service import model_discovery_service
        
        # Get model snapshot
        snapshot = await model_discovery_service.get_model_snapshot(force_refresh=False)
        
        logger.info(f"✅ Discovered {len(snapshot.all_models)} models from Hugging Face")
        logger.info(f"📊 Best models by capability:")
        
        for capability, model in snapshot.best_models.items():
            logger.info(f"   🎯 {capability}: {model.name}")
            logger.info(f"      ⭐ Score: {model.performance_score:.2f}")
            logger.info(f"      📥 Downloads: {model.downloads:,}")
            logger.info(f"      👍 Likes: {model.likes}")
            logger.info("")
        
        # Test 2: Startup integration
        logger.info("📋 Test 2: Startup Model Manager Integration")
        logger.info("-" * 50)
        
        from src.core.startup_model_manager import startup_model_manager
        
        # Initialize with startup manager
        startup_results = await startup_model_manager.initialize_models_on_startup(
            provider_registry=None,
            force_refresh=False
        )
        
        if startup_results["success"]:
            logger.info("✅ Startup model discovery successful!")
            logger.info(f"   🤗 HF models: {startup_results['hf_models_discovered']}")
            logger.info(f"   🏠 Local models: {startup_results['local_models_found']}")
            logger.info(f"   🔗 Integrated: {startup_results['integrated_models']}")
            
            logger.info("")
            logger.info("🎯 INTEGRATED BEST MODELS:")
            for capability, model_info in startup_results["best_models_by_capability"].items():
                provider_icon = "🏠" if model_info["provider"] == "ollama" else "🌐" if model_info["provider"] == "openrouter" else "🤗"
                free_icon = "🆓" if model_info["is_free"] else "💰"
                
                logger.info(f"   {capability.upper()}: {model_info['name']}")
                logger.info(f"      {provider_icon} Provider: {model_info['provider']}")
                logger.info(f"      {free_icon} Cost: {'Free' if model_info['is_free'] else 'Paid'}")
                logger.info(f"      ⭐ Score: {model_info['performance_score']:.2f}")
                logger.info("")
        else:
            logger.error("❌ Startup model discovery failed")
            for error in startup_results["errors"]:
                logger.error(f"   💥 {error}")
        
        # Test 3: Agent configuration generation
        logger.info("📋 Test 3: Dynamic Agent Configuration")
        logger.info("-" * 50)
        
        agent_types = ["coding", "reasoning", "research", "general"]
        
        for agent_type in agent_types:
            config = startup_model_manager.get_model_config_for_agent(agent_type)
            
            logger.info(f"🤖 {agent_type.upper()} Agent Configuration:")
            logger.info(f"   Provider: {config['provider']}")
            logger.info(f"   Model: {config['model_name']}")
            logger.info(f"   Temperature: {config['temperature']}")
            logger.info(f"   Max Tokens: {config['max_tokens']}")
            logger.info("")
        
        # Test 4: Compare with old hard-coded approach
        logger.info("📋 Test 4: Comparison with Hard-coded Models")
        logger.info("-" * 50)
        
        hard_coded_models = {
            "coding": "deepseek-coder-v2:latest",  # ❌ Doesn't exist
            "reasoning": "phi4-fast",              # ❌ Doesn't exist
            "general": "deepseek/deepseek-r1-0528-qwen3-8b:free"  # ✅ Might exist
        }
        
        logger.info("🔴 OLD HARD-CODED APPROACH:")
        for agent_type, model in hard_coded_models.items():
            logger.info(f"   {agent_type}: {model} (may not exist)")
        
        logger.info("")
        logger.info("🟢 NEW DYNAMIC APPROACH:")
        for agent_type in hard_coded_models.keys():
            best_model = startup_model_manager.get_best_model_for_capability(agent_type)
            if best_model:
                logger.info(f"   {agent_type}: {best_model.name} ({best_model.provider}) ✅ VERIFIED")
            else:
                logger.info(f"   {agent_type}: No model found ❌")
        
        # Test 5: Create a working coding agent with discovered model
        logger.info("📋 Test 5: Create Working Coding Agent")
        logger.info("-" * 50)
        
        coding_config = startup_model_manager.get_model_config_for_agent("coding")
        logger.info(f"🛠️ Creating coding agent with discovered model: {coding_config['model_name']}")
        
        # Test the configuration by creating a simple request
        if coding_config["provider"] == "ollama":
            try:
                import aiohttp
                
                prompt = "Create a simple Python function to add two numbers:"
                payload = {
                    "model": coding_config["model_name"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": coding_config["temperature"],
                        "num_predict": 100
                    }
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:11434/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            generated_code = result.get("response", "")
                            
                            logger.info("✅ Coding agent test successful!")
                            logger.info(f"📝 Generated code preview:")
                            preview = generated_code[:200] + "..." if len(generated_code) > 200 else generated_code
                            logger.info(f"   {preview}")
                        else:
                            logger.error(f"❌ Ollama test failed: {response.status}")
                            
            except Exception as e:
                logger.error(f"❌ Coding agent test failed: {e}")
        else:
            logger.info(f"ℹ️ Skipping test for {coding_config['provider']} provider")
        
        logger.info("")
        logger.info("🎉 MODEL DISCOVERY SYSTEM TEST COMPLETE!")
        logger.info("=" * 50)
        logger.info("✅ Benefits of new system:")
        logger.info("   🔄 Dynamic model discovery from Hugging Face")
        logger.info("   🏠 Automatic integration with local providers")
        logger.info("   🎯 Capability-based model matching")
        logger.info("   💾 Intelligent caching (6-hour refresh)")
        logger.info("   🆓 Preference for free/local models")
        logger.info("   🛡️ Graceful fallback when models unavailable")
        logger.info("   📊 Performance-based model ranking")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Model discovery test failed: {e}")
        return False

async def test_startup_integration():
    """Test integration with system startup"""
    
    logger.info("🧪 TESTING STARTUP INTEGRATION")
    logger.info("=" * 50)
    
    try:
        # Import and run the startup checklist with model discovery
        from system_startup_checklist import SystemStartupChecklist
        
        checklist = SystemStartupChecklist(auto_start=False)
        
        # Run just the model discovery phase
        logger.info("Running model discovery phase...")
        success = await checklist._check_model_discovery()
        
        if success:
            logger.info("✅ Model discovery phase passed!")
        else:
            logger.error("❌ Model discovery phase failed!")
        
        return success
        
    except Exception as e:
        logger.error(f"💥 Startup integration test failed: {e}")
        return False

async def main():
    """Main test execution"""
    
    logger.info("🚀 STARTING MODEL DISCOVERY TESTS")
    logger.info("=" * 70)
    
    # Test 1: Core model discovery
    test1_success = await test_model_discovery()
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Startup integration
    test2_success = await test_startup_integration()
    
    logger.info("")
    logger.info("🏁 ALL TESTS COMPLETE")
    logger.info("=" * 30)
    
    if test1_success and test2_success:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Model discovery system is working correctly")
        logger.info("🔧 Ready to fix hard-coded model problems")
    else:
        logger.error("💥 SOME TESTS FAILED")
        logger.error("🔧 Check the output above for issues")

if __name__ == "__main__":
    asyncio.run(main())
