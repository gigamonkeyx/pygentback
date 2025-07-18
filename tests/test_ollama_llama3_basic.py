#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Ollama Llama3 8B Configuration Test
Observer-approved validation for Phase 1 foundation setup
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BasicOllamaTest:
    """Basic test for Ollama Llama3 8B configuration"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model_name = "llama3:8b"
    
    async def test_ollama_server_connection(self) -> bool:
        """Test if Ollama server is running and accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        logger.info(f"✅ Ollama server connected. Available models: {models}")
                        return True
                    else:
                        logger.error(f"❌ Ollama server responded with status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama server: {e}")
            return False
    
    async def test_llama3_8b_availability(self) -> bool:
        """Test if Llama3 8B model is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        
                        if self.model_name in models:
                            logger.info(f"✅ Llama3 8B model available")
                            return True
                        else:
                            logger.error(f"❌ Llama3 8B model not found. Available: {models}")
                            return False
                    else:
                        logger.error(f"❌ Failed to get model list: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error checking model availability: {e}")
            return False
    
    async def test_basic_generation(self) -> bool:
        """Test basic code generation with Llama3 8B"""
        try:
            prompt = "write hello world in Python"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        generated_text = data.get("response", "").strip()
                        
                        if generated_text and "print" in generated_text.lower():
                            logger.info(f"✅ Basic generation successful")
                            logger.info(f"Generated: {generated_text[:100]}...")
                            return True
                        else:
                            logger.error(f"❌ Generation failed or invalid response: {generated_text}")
                            return False
                    else:
                        logger.error(f"❌ Generation request failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error during generation test: {e}")
            return False
    
    async def test_model_performance(self) -> Dict[str, Any]:
        """Test model performance metrics"""
        try:
            prompt = "def fibonacci(n):"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "max_tokens": 200
                }
            }
            
            import time
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        end_time = time.time()
                        
                        response_time = end_time - start_time
                        generated_text = data.get("response", "").strip()
                        
                        metrics = {
                            "response_time_seconds": response_time,
                            "generated_length": len(generated_text),
                            "tokens_per_second": len(generated_text.split()) / response_time if response_time > 0 else 0,
                            "success": True
                        }
                        
                        logger.info(f"✅ Performance test completed")
                        logger.info(f"Response time: {response_time:.2f}s")
                        logger.info(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
                        
                        return metrics
                    else:
                        logger.error(f"❌ Performance test failed: {response.status}")
                        return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"❌ Performance test error: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_coding_capabilities(self) -> bool:
        """Test coding-specific capabilities"""
        coding_prompts = [
            "write a function to reverse a string in Python",
            "create a simple class for a bank account",
            "write a function to check if a number is prime"
        ]
        
        success_count = 0
        
        for prompt in coding_prompts:
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "max_tokens": 300
                    }
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            generated_text = data.get("response", "").strip()
                            
                            # Basic validation - check for Python keywords
                            if any(keyword in generated_text.lower() for keyword in ["def", "class", "return", "if"]):
                                success_count += 1
                                logger.info(f"✅ Coding test passed: {prompt[:30]}...")
                            else:
                                logger.warning(f"⚠️ Coding test questionable: {prompt[:30]}...")
                        else:
                            logger.error(f"❌ Coding test failed: {prompt[:30]}...")
            except Exception as e:
                logger.error(f"❌ Coding test error for '{prompt[:30]}...': {e}")
        
        success_rate = success_count / len(coding_prompts)
        logger.info(f"Coding capabilities success rate: {success_rate:.2%}")
        
        return success_rate >= 0.7  # 70% success rate threshold


async def run_basic_ollama_tests():
    """Run basic Ollama Llama3 8B tests"""
    print("\n🚀 PHASE 1 BASIC VALIDATION: Ollama Llama3 8B Foundation")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_instance = BasicOllamaTest()
    results = {}
    
    try:
        # Test 1: Server connection
        print("\n1. Testing Ollama server connection...")
        results['server_connection'] = await test_instance.test_ollama_server_connection()
        
        if not results['server_connection']:
            print("❌ CRITICAL: Ollama server not accessible. Please ensure Ollama is running.")
            return False
        
        # Test 2: Model availability
        print("\n2. Testing Llama3 8B model availability...")
        results['model_availability'] = await test_instance.test_llama3_8b_availability()
        
        if not results['model_availability']:
            print("❌ CRITICAL: Llama3 8B model not available. Please run 'ollama pull llama3:8b'")
            return False
        
        # Test 3: Basic generation
        print("\n3. Testing basic code generation...")
        results['basic_generation'] = await test_instance.test_basic_generation()
        
        # Test 4: Performance metrics
        print("\n4. Testing model performance...")
        performance_metrics = await test_instance.test_model_performance()
        results['performance'] = performance_metrics.get('success', False)
        
        # Test 5: Coding capabilities
        print("\n5. Testing coding-specific capabilities...")
        results['coding_capabilities'] = await test_instance.test_coding_capabilities()
        
        # Summary
        print("\n" + "=" * 60)
        print("PHASE 1 BASIC VALIDATION RESULTS:")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        success_rate = passed_tests / total_tests
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
        
        if success_rate >= 0.8:  # 80% success threshold
            print("\n🎉 PHASE 1 BASIC VALIDATION: SUCCESS")
            print("✅ Ollama Llama3 8B foundation is ready")
            print("✅ Observer Checkpoint: Basic configuration validated")
            print("🚀 Ready to proceed to Phase 1.2: Enhanced Agent Base Class Integration")
            return True
        else:
            print("\n⚠️ PHASE 1 BASIC VALIDATION: PARTIAL SUCCESS")
            print("Some tests failed. Please review and fix issues before proceeding.")
            return False
            
    except Exception as e:
        print(f"\n❌ PHASE 1 VALIDATION FAILED: {e}")
        logger.error(f"Phase 1 validation error: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(run_basic_ollama_tests())
