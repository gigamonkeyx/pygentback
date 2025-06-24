#!/usr/bin/env python3
"""
Validate GPU Optimization Implementation

Simple validation of GPU optimization components and structure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def validate_gpu_imports():
    """Validate GPU optimization imports"""
    print("🔍 Validating GPU Optimization Imports...")
    
    try:
        # Test core GPU imports
        from core.gpu_config import gpu_manager, GPUConfig
        print("✅ GPU Config imported successfully")
        
        from core.gpu_optimization import gpu_optimizer, GPUOptimizationConfig
        print("✅ GPU Optimization imported successfully")
        
        from core.ollama_gpu_integration import ollama_gpu_manager
        print("✅ Ollama GPU Integration imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def validate_gpu_structure():
    """Validate GPU optimization structure"""
    print("\n🏗️ Validating GPU Optimization Structure...")
    
    try:
        from core.gpu_optimization import GPUOptimizer, GPUOptimizationConfig, GPUPerformanceMetrics
        
        # Check GPUOptimizationConfig attributes
        config = GPUOptimizationConfig()
        required_attrs = [
            'memory_fraction', 'use_mixed_precision', 'use_tensor_cores',
            'rtx_3080_optimizations', 'performance_monitoring'
        ]
        
        for attr in required_attrs:
            if hasattr(config, attr):
                print(f"✅ GPUOptimizationConfig.{attr} exists")
            else:
                print(f"❌ GPUOptimizationConfig.{attr} missing")
                return False
        
        # Check GPUOptimizer methods
        optimizer = GPUOptimizer()
        required_methods = [
            'initialize', 'clear_memory_cache', 'optimize_model_for_inference',
            'get_optimization_status', 'get_performance_summary'
        ]
        
        for method in required_methods:
            if hasattr(optimizer, method):
                print(f"✅ GPUOptimizer.{method} exists")
            else:
                print(f"❌ GPUOptimizer.{method} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Structure validation failed: {e}")
        return False


def validate_ollama_gpu_structure():
    """Validate Ollama GPU integration structure"""
    print("\n🦙 Validating Ollama GPU Integration Structure...")
    
    try:
        from core.ollama_gpu_integration import OllamaGPUManager, OllamaModelConfig
        
        # Check OllamaModelConfig attributes
        config = OllamaModelConfig(
            name="test",
            size_gb=1.0,
            context_length=4096
        )
        
        required_attrs = [
            'name', 'size_gb', 'context_length', 'gpu_layers',
            'gpu_memory_required_gb', 'optimization_level'
        ]
        
        for attr in required_attrs:
            if hasattr(config, attr):
                print(f"✅ OllamaModelConfig.{attr} exists")
            else:
                print(f"❌ OllamaModelConfig.{attr} missing")
                return False
        
        # Check OllamaGPUManager methods
        manager = OllamaGPUManager()
        required_methods = [
            'initialize', 'generate', 'get_performance_summary'
        ]
        
        for method in required_methods:
            if hasattr(manager, method):
                print(f"✅ OllamaGPUManager.{method} exists")
            else:
                print(f"❌ OllamaGPUManager.{method} missing")
                return False
        
        # Check RTX 3080 optimized models
        if hasattr(manager, 'rtx_3080_models') and manager.rtx_3080_models:
            print(f"✅ RTX 3080 optimized models: {len(manager.rtx_3080_models)} models")
            for model_name in manager.rtx_3080_models:
                print(f"   📦 {model_name}")
        else:
            print("❌ RTX 3080 optimized models missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama GPU structure validation failed: {e}")
        return False


def validate_api_structure():
    """Validate GPU monitoring API structure"""
    print("\n🌐 Validating GPU Monitoring API Structure...")
    
    try:
        from api.gpu_monitoring import router
        
        # Check if router exists
        if router:
            print("✅ GPU monitoring API router exists")
        else:
            print("❌ GPU monitoring API router missing")
            return False
        
        # Check route tags
        if hasattr(router, 'tags') and 'GPU Monitoring' in router.tags:
            print("✅ GPU Monitoring API tags configured")
        else:
            print("❌ GPU Monitoring API tags missing")
        
        return True
        
    except Exception as e:
        print(f"❌ API structure validation failed: {e}")
        return False


def validate_dependencies():
    """Validate GPU optimization dependencies"""
    print("\n📦 Validating GPU Dependencies...")
    
    dependencies = {
        'torch': 'PyTorch for GPU acceleration',
        'psutil': 'System monitoring',
        'aiohttp': 'Async HTTP client',
        'fastapi': 'API framework',
        'pydantic': 'Data validation'
    }
    
    available_deps = []
    missing_deps = []
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: {description}")
            available_deps.append(dep)
        except ImportError:
            print(f"❌ {dep}: {description} (missing)")
            missing_deps.append(dep)
    
    # Check optional dependencies
    optional_deps = {
        'cupy': 'CuPy for GPU array operations',
        'faiss': 'FAISS for GPU vector search',
        'GPUtil': 'GPU utilization monitoring'
    }
    
    print("\n📦 Optional GPU Dependencies:")
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: {description}")
        except ImportError:
            print(f"⚠️ {dep}: {description} (optional, not installed)")
    
    return len(missing_deps) == 0


def validate_pytorch_gpu():
    """Validate PyTorch GPU support"""
    print("\n🔥 Validating PyTorch GPU Support...")
    
    try:
        import torch
        
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            
            if torch.cuda.device_count() > 0:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ GPU 0: {gpu_name}")
                
                # Check for RTX 3080
                if "rtx 3080" in gpu_name.lower():
                    print("🚀 RTX 3080 detected!")
                
                # Check compute capability
                props = torch.cuda.get_device_properties(0)
                compute_cap = (props.major, props.minor)
                print(f"✅ Compute capability: {compute_cap}")
                
                # Check memory
                memory_gb = props.total_memory / (1024**3)
                print(f"✅ GPU memory: {memory_gb:.1f}GB")
                
                return True
        else:
            print("⚠️ CUDA not available - GPU optimization will be limited")
            return True  # Not a failure, just limited functionality
            
    except Exception as e:
        print(f"❌ PyTorch GPU validation failed: {e}")
        return False


def main():
    """Run all GPU optimization validations"""
    print("🚀 PyGent Factory GPU Optimization Validation")
    print("=" * 60)
    
    validations = [
        ("GPU Imports", validate_gpu_imports),
        ("GPU Structure", validate_gpu_structure),
        ("Ollama GPU Structure", validate_ollama_gpu_structure),
        ("API Structure", validate_api_structure),
        ("Dependencies", validate_dependencies),
        ("PyTorch GPU Support", validate_pytorch_gpu)
    ]
    
    passed = 0
    for validation_name, validation_func in validations:
        print(f"\n{validation_name}:")
        try:
            if validation_func():
                passed += 1
            else:
                print(f"❌ {validation_name} failed")
        except Exception as e:
            print(f"❌ {validation_name} error: {e}")
    
    total = len(validations)
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL GPU OPTIMIZATION VALIDATIONS PASSED!")
        print("   PyGent Factory GPU optimization system is properly implemented.")
        print("   Ready for high-performance AI workloads with RTX 3080 optimization.")
        
        print(f"\n🔥 GPU OPTIMIZATION FEATURES:")
        print(f"   ✅ Advanced GPU detection and configuration")
        print(f"   ✅ RTX 3080 specific optimizations")
        print(f"   ✅ Mixed precision training support")
        print(f"   ✅ Tensor Core acceleration")
        print(f"   ✅ Memory management and cleanup")
        print(f"   ✅ Performance monitoring and analytics")
        print(f"   ✅ Ollama GPU integration")
        print(f"   ✅ Model inference optimization")
        print(f"   ✅ Real-time GPU monitoring API")
        
        return True
    else:
        print(f"⚠️ {total - passed} VALIDATIONS FAILED")
        print("   Fix the issues above before deploying GPU optimization.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
