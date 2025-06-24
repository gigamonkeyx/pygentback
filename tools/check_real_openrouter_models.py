#!/usr/bin/env python3
"""
Check Real OpenRouter Models

This script checks what models are actually available with your specific API key,
not just generic examples. Let's see what you really have access to.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ai.providers.openrouter_provider import OpenRouterProvider


async def check_real_models():
    """Check what models are actually available with your API key"""
    print("🔍 Checking Real OpenRouter Models with Your API Key...")
    
    # Your API key
    api_key = "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"
    
    provider = OpenRouterProvider(api_key=api_key)
    
    try:
        # Initialize the provider (not start)
        if not await provider.initialize():
            print("❌ Failed to initialize OpenRouter provider")
            return
        
        print("✅ Connected to OpenRouter successfully")
        
        # Get all available models
        models = await provider.get_available_models()
        print(f"📊 Total models available: {len(models)}")
        
        # Filter for models you mentioned
        target_keywords = [
            "deepseek", "qwen", "r1", "7b", "1.5b", "3b", "8b"
        ]
        
        print("\n🎯 Looking for DeepSeek, Qwen, R1, and small models...")
        matching_models = []
        
        for model in models:
            model_lower = model.lower()
            for keyword in target_keywords:
                if keyword in model_lower:
                    matching_models.append(model)
                    break
        
        if matching_models:
            print(f"✅ Found {len(matching_models)} matching models:")
            for model in matching_models[:25]:  # Show first 25
                print(f"   • {model}")
            if len(matching_models) > 25:
                print(f"   ... and {len(matching_models) - 25} more")
        else:
            print("❌ No matching models found for your search terms")
        
        # Show some popular/recent models
        print("\n🔥 First 20 models in your account:")
        for i, model in enumerate(models[:20]):
            print(f"   {i+1:2d}. {model}")
        
        # Test a simple generation with a small model if available
        test_models = [m for m in models if any(keyword in m.lower() for keyword in ["7b", "qwen", "deepseek", "3b"])]
        
        if test_models:
            test_model = test_models[0]
            print(f"\n🧪 Testing generation with {test_model}...")
            
            try:
                response = await provider.generate_text(
                    model=test_model,
                    prompt="Say 'Hello from OpenRouter!' and tell me your model name.",
                    max_tokens=50
                )
                
                if response:
                    print("✅ Test successful!")
                    print(f"   Model: {test_model}")
                    print(f"   Response: {response}")
                else:
                    print("❌ Test failed - no response")
            
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        # Show full model list in sections
        print(f"\n📋 All available models ({len(models)} total):")
        
        # Group by provider
        providers = {}
        for model in models:
            provider_name = model.split('/')[0] if '/' in model else 'other'
            if provider_name not in providers:
                providers[provider_name] = []
            providers[provider_name].append(model)
        
        for provider_name, provider_models in sorted(providers.items()):
            print(f"\n   📁 {provider_name.upper()} ({len(provider_models)} models):")
            for i, model in enumerate(provider_models[:10]):  # Show first 10 per provider
                print(f"      {i+1:2d}. {model}")
            if len(provider_models) > 10:
                print(f"      ... and {len(provider_models) - 10} more {provider_name} models")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function"""
    print("OpenRouter Real Model Check")
    print("Checking what models are actually available with your API key...")
    print("(Not generic examples, but real available models)")
    print("=" * 60)
    
    await check_real_models()


if __name__ == "__main__":
    asyncio.run(main())
