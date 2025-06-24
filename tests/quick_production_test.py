#!/usr/bin/env python3
"""
Quick Production Code Test
Tests core production functionality after mock removal.
"""

import sys
import os
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import(module_name, class_name=None):
    """Test importing a module/class."""
    try:
        if class_name:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} imported successfully")
            return cls
        else:
            module = __import__(module_name)
            print(f"✅ {module_name} imported successfully")
            return module
    except Exception as e:
        print(f"❌ {module_name}" + (f".{class_name}" if class_name else "") + f" failed: {e}")
        traceback.print_exc()
        return None

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("🧪 TESTING CORE PRODUCTION IMPORTS")
    print("=" * 50)
    
    # Test A2A components
    a2a_server = test_import("src.a2a", "A2AServer")
    
    # Test Agent Factory
    agent_factory = test_import("src.core.agent_factory", "AgentFactory")
    
    # Test Multi-Agent Core
    multi_agent = test_import("src.ai.multi_agent.core_new", "Agent")
    
    # Test NLP Core
    nlp_core = test_import("src.ai.nlp.core", "NLPCore")
    
    # Test Distributed Genetic Algorithm
    dga = test_import("src.orchestration.distributed_genetic_algorithm", "DistributedGeneticAlgorithm")
    
    print("\n🧪 TESTING BASIC INSTANTIATION")
    print("=" * 50)
    
    # Try to create instances if imports worked
    if nlp_core:
        try:
            nlp_instance = nlp_core()
            print("✅ NLPCore instance created successfully")
        except Exception as e:
            print(f"❌ NLPCore instantiation failed: {e}")
    
    print("\n🧪 TESTING PRODUCTION CODE PATTERNS")
    print("=" * 50)
    
    # Check that key files are free of mock patterns
    mock_patterns = ['mock', 'Mock', 'MOCK', 'TODO', 'FIXME', 'HACK', 'placeholder']
    production_files = [
        'src/core/agent_factory.py',
        'src/ai/multi_agent/core_new.py', 
        'src/ai/nlp/core.py',
        'src/orchestration/distributed_genetic_algorithm.py',
        'src/a2a/__init__.py'
    ]
    
    for file_path in production_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                found_patterns = []
                for pattern in mock_patterns:
                    if pattern in content:
                        # Count occurrences
                        count = content.count(pattern)
                        found_patterns.append(f"{pattern}({count})")
                
                if found_patterns:
                    print(f"⚠️  {file_path}: {', '.join(found_patterns)}")
                else:
                    print(f"✅ {file_path}: Clean of mock patterns")

if __name__ == "__main__":
    test_basic_functionality()
