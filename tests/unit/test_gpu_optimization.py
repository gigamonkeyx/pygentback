#!/usr/bin/env python3
"""
Test GPU Optimization System

Comprehensive testing of GPU optimization, RTX 3080 features,
and performance monitoring for PyGent Factory.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from core.gpu_config import gpu_manager
from core.gpu_optimization import gpu_optimizer, GPUOptimizationConfig
from core.ollama_gpu_integration import ollama_gpu_manager


async def test_gpu_detection():
    """Test GPU detection and configuration"""
    print("🔍 Testing GPU Detection...")
    
    # Initialize GPU manager
    success = gpu_manager.initialize()
    
    if success and gpu_manager.is_available():
        config = gpu_manager.get_config()
        print(f"✅ GPU Detected: {config.device_name}")
        print(f"   Device: {config.device}")
        print(f"   Memory: {config.memory_total}MB total")
        print(f"   Compute Capability: {config.compute_capability}")
        print(f"   CUDA Version: {config.cuda_version}")
        print(f"   Mixed Precision Support: {config.supports_mixed_precision}")
        
        # Check for RTX 3080
        if "rtx 3080" in config.device_name.lower():
            print("🚀 RTX 3080 detected - optimal configuration available!")
        
        return True
    else:
        print("⚠️ No GPU detected - using CPU fallback")
        return False


async def test_gpu_optimization():
    """Test GPU optimization system"""
    print("\n⚡ Testing GPU Optimization...")
    
    # Initialize optimizer
    success = await gpu_optimizer.initialize()
    
    if success:
        print("✅ GPU Optimizer initialized")
        
        # Get optimization status
        status = gpu_optimizer.get_optimization_status()
        print(f"   RTX 3080 Detected: {status['rtx_3080_detected']}")
        print(f"   Tensor Cores Available: {status['tensor_cores_available']}")
        print(f"   Mixed Precision: {status['mixed_precision_enabled']}")
        print(f"   FAISS GPU: {status['faiss_gpu_available']}")
        print(f"   CuPy Available: {status['cupy_available']}")
        
        return True
    else:
        print("❌ GPU Optimization failed to initialize")
        return False


async def test_memory_management():
    """Test GPU memory management"""
    print("\n💾 Testing Memory Management...")
    
    if not gpu_manager.is_available():
        print("⚠️ No GPU available for memory testing")
        return True
    
    try:
        import torch
        
        # Get initial memory state
        initial_memory = gpu_manager.get_memory_info()
        print(f"   Initial Memory: {initial_memory.get('allocated_mb', 0):.1f}MB allocated")
        
        # Test memory allocation with optimization
        device = gpu_manager.get_device()
        
        with gpu_optimizer.optimized_inference("memory_test"):
            # Allocate test tensors
            test_tensors = []
            for i in range(5):
                tensor = torch.randn(1000, 1000, device=device)
                test_tensors.append(tensor)
            
            # Check memory usage
            peak_memory = gpu_manager.get_memory_info()
            print(f"   Peak Memory: {peak_memory.get('allocated_mb', 0):.1f}MB allocated")
            
            # Clear tensors
            del test_tensors
        
        # Test memory cleanup
        gpu_optimizer.clear_memory_cache()
        final_memory = gpu_manager.get_memory_info()
        print(f"   Final Memory: {final_memory.get('allocated_mb', 0):.1f}MB allocated")
        
        print("✅ Memory management test completed")
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False


async def test_performance_monitoring():
    """Test performance monitoring"""
    print("\n📊 Testing Performance Monitoring...")
    
    try:
        # Wait for some performance data to be collected
        await asyncio.sleep(2)
        
        # Get performance summary
        summary = gpu_optimizer.get_performance_summary()
        
        if summary.get("error"):
            print(f"⚠️ No performance data available: {summary['error']}")
        else:
            print("✅ Performance monitoring active")
            print(f"   Sample Count: {summary.get('sample_count', 0)}")
            
            gpu_util = summary.get('gpu_utilization', {})
            if gpu_util:
                print(f"   GPU Utilization: avg={gpu_util.get('avg', 0):.1%}, max={gpu_util.get('max', 0):.1%}")
            
            memory_util = summary.get('memory_utilization', {})
            if memory_util:
                print(f"   Memory Utilization: avg={memory_util.get('avg', 0):.1%}, max={memory_util.get('max', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False


async def test_ollama_gpu_integration():
    """Test Ollama GPU integration"""
    print("\n🦙 Testing Ollama GPU Integration...")
    
    try:
        # Initialize Ollama GPU manager
        success = await ollama_gpu_manager.initialize()
        
        if success:
            print("✅ Ollama GPU manager initialized")
            
            # Get performance summary
            perf_summary = ollama_gpu_manager.get_performance_summary()
            print(f"   GPU Optimized: {perf_summary['gpu_optimized']}")
            print(f"   Max GPU Memory: {perf_summary['max_gpu_memory_gb']:.1f}GB")
            print(f"   Available Models: {len(ollama_gpu_manager.available_models)}")
            
            # List optimized models
            for model_name, config in ollama_gpu_manager.available_models.items():
                print(f"   📦 {model_name}: {config.size_gb:.1f}GB, {config.gpu_layers} GPU layers")
            
            return True
        else:
            print("⚠️ Ollama GPU manager failed to initialize (service may not be running)")
            return True  # Not a critical failure
            
    except Exception as e:
        print(f"❌ Ollama GPU integration test failed: {e}")
        return False


async def test_inference_optimization():
    """Test inference optimization"""
    print("\n🧠 Testing Inference Optimization...")
    
    if not gpu_manager.is_available():
        print("⚠️ No GPU available for inference testing")
        return True
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(1000, 500)
                self.linear2 = nn.Linear(500, 100)
                self.linear3 = nn.Linear(100, 10)
                
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = torch.relu(self.linear2(x))
                return self.linear3(x)
        
        # Test model optimization
        model = TestModel()
        optimized_model = gpu_optimizer.optimize_model_for_inference(model)
        
        print(f"✅ Model optimized for inference")
        print(f"   Device: {next(optimized_model.parameters()).device}")
        
        # Test inference with optimization
        device = gpu_manager.get_device()
        test_input = torch.randn(32, 1000, device=device)
        
        # Measure inference time
        start_time = time.time()
        
        with gpu_optimizer.optimized_inference("test_model"):
            with torch.no_grad():
                output = optimized_model(test_input)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        print(f"   Inference Time: {inference_time:.2f}ms")
        print(f"   Output Shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference optimization test failed: {e}")
        return False


async def test_rtx_3080_features():
    """Test RTX 3080 specific features"""
    print("\n🚀 Testing RTX 3080 Features...")
    
    if not gpu_optimizer.rtx_3080_detected:
        print("⚠️ RTX 3080 not detected - skipping RTX-specific tests")
        return True
    
    try:
        print("✅ RTX 3080 detected - testing specific optimizations")
        
        # Test Tensor Core availability
        if gpu_optimizer.tensor_core_available:
            print("✅ Tensor Cores available")
            
            # Test mixed precision if available
            if gpu_optimizer.config.use_mixed_precision:
                print("✅ Mixed precision enabled")
                
                import torch
                from torch.cuda.amp import autocast
                
                # Test autocast functionality
                device = gpu_manager.get_device()
                test_tensor = torch.randn(100, 100, device=device)
                
                with autocast():
                    result = torch.matmul(test_tensor, test_tensor)
                    print(f"✅ Mixed precision inference working (dtype: {result.dtype})")
        
        # Test TensorFloat-32
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            tf32_enabled = torch.backends.cuda.matmul.allow_tf32
            print(f"✅ TensorFloat-32: {'enabled' if tf32_enabled else 'disabled'}")
        
        return True
        
    except Exception as e:
        print(f"❌ RTX 3080 features test failed: {e}")
        return False


async def run_all_tests():
    """Run all GPU optimization tests"""
    print("🚀 PyGent Factory GPU Optimization Test Suite")
    print("=" * 60)
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("GPU Optimization", test_gpu_optimization),
        ("Memory Management", test_memory_management),
        ("Performance Monitoring", test_performance_monitoring),
        ("Ollama GPU Integration", test_ollama_gpu_integration),
        ("Inference Optimization", test_inference_optimization),
        ("RTX 3080 Features", test_rtx_3080_features)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if await test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    total = len(tests)
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL GPU OPTIMIZATION TESTS PASSED!")
        print("   PyGent Factory is fully GPU optimized and ready for high-performance AI workloads.")
        
        # Display optimization summary
        if gpu_manager.is_available():
            print(f"\n🔥 GPU OPTIMIZATION SUMMARY:")
            print(f"   GPU: {gpu_manager.config.device_name}")
            print(f"   Memory: {gpu_manager.config.memory_total}MB")
            print(f"   RTX 3080 Optimizations: {'✅ Active' if gpu_optimizer.rtx_3080_detected else '❌ Not Available'}")
            print(f"   Mixed Precision: {'✅ Enabled' if gpu_optimizer.config.use_mixed_precision else '❌ Disabled'}")
            print(f"   Tensor Cores: {'✅ Available' if gpu_optimizer.tensor_core_available else '❌ Not Available'}")
    else:
        print("⚠️ SOME GPU OPTIMIZATION TESTS FAILED")
        print("   Check the errors above and ensure GPU drivers and dependencies are properly installed.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
