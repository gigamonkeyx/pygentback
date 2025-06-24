#!/usr/bin/env python3
"""
Simple OpenRouter Test

Quick test of the OpenRouter provider with your API key.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ai.providers.openrouter_provider import OpenRouterProvider


async def test_openrouter():
    """Test OpenRouter provider."""
    print("🚀 Testing OpenRouter with your API key...")
    print("=" * 50)
    
    # Create provider
    provider = OpenRouterProvider()
    
    # Test initialization
    print("1. Initializing...")
    success = await provider.initialize()
    print(f"   ✅ Initialization: {'SUCCESS' if success else 'FAILED'}")
    
    if not success:
        print("❌ Failed to initialize OpenRouter")
        return False
    
    # Test health check
    print("\n2. Health check...")
    health = await provider.health_check()
    print(f"   Health: {'✅ HEALTHY' if health.get('healthy') else '❌ UNHEALTHY'}")
    print(f"   Status: {health.get('status_code', 'N/A')}")
    print(f"   Models: {health.get('models_loaded', 0)}")
    
    # Test models
    print("\n3. Available models...")
    models = await provider.get_available_models()
    print(f"   Total: {len(models)}")
    if models:
        print(f"   First 3: {models[:3]}")
    
    # Test text generation
    print("\n4. Testing text generation...")
    if models:
        test_model = "anthropic/claude-3.5-sonnet" if "anthropic/claude-3.5-sonnet" in models else models[0]
        print(f"   Using: {test_model}")
        
        response = await provider.generate_text(
            model=test_model,
            prompt="Say 'Hello from OpenRouter!' to test the connection.",
            max_tokens=20,
            temperature=0.1
        )
        
        print(f"   Response: {response}")
        
        if response:
            print("   ✅ Text generation: SUCCESS")
            return True
        else:
            print("   ❌ Text generation: FAILED")
            return False
    else:
        print("   ⚠️ No models available")
        return False


async def main():
    print("🧪 OpenRouter Quick Test")
    print(f"API Key: sk-or-v1-...{('5715f77a3372c962f219373073f7d34eb9eaa0a65504ff15d0895c9fab3bae56')[-8:]}")
    print("=" * 50)
    
    try:
        result = await test_openrouter()
        
        if result:
            print("\n🎉 SUCCESS! OpenRouter is working with your API key!")
        else:
            print("\n❌ FAILED! Check the errors above.")
        
        return result
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
