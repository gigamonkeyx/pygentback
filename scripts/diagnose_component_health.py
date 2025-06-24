#!/usr/bin/env python3
"""
Component Health Diagnostic Tool

Deep analysis of why workflow components are reporting as unhealthy.
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# Setup Windows-compatible environment
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Add src to path
project_dir = Path(__file__).parent.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

print("🔍 PyGent Factory - Component Health Diagnostic")
print("=" * 60)
print(f"🐍 Python: {sys.version}")
print(f"💻 Platform: {sys.platform}")
print(f"📁 Working Directory: {os.getcwd()}")
print(f"📦 Source Directory: {src_dir}")
print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

async def diagnose_adapters():
    """Diagnose individual adapter health"""
    
    print("🔧 ADAPTER INITIALIZATION DIAGNOSIS")
    print("-" * 50)
    
    # Test NLP Adapter
    print("\n📚 Testing NLP Adapter...")
    try:
        from integration.adapters import NLPAdapter
        nlp_adapter = NLPAdapter()
        print(f"✅ NLPAdapter created: {nlp_adapter.component_name}")
        print(f"   Initial status: {nlp_adapter.health_status}")
        print(f"   Is initialized: {nlp_adapter.is_initialized}")
        
        # Try to initialize
        print("   Attempting initialization...")
        init_result = await nlp_adapter.initialize()
        print(f"   Initialization result: {init_result}")
        print(f"   Post-init status: {nlp_adapter.health_status}")
        print(f"   Is initialized: {nlp_adapter.is_initialized}")
        
        # Try health check
        print("   Performing health check...")
        health_result = await nlp_adapter.health_check()
        print(f"   Health check result: {health_result}")
        
    except Exception as e:
        print(f"❌ NLP Adapter failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
    
    # Test Multi-Agent Adapter
    print("\n🤖 Testing Multi-Agent Adapter...")
    try:
        from integration.adapters import MultiAgentAdapter
        ma_adapter = MultiAgentAdapter()
        print(f"✅ MultiAgentAdapter created: {ma_adapter.component_name}")
        print(f"   Initial status: {ma_adapter.health_status}")
        print(f"   Is initialized: {ma_adapter.is_initialized}")
        
        # Try to initialize
        print("   Attempting initialization...")
        init_result = await ma_adapter.initialize()
        print(f"   Initialization result: {init_result}")
        print(f"   Post-init status: {ma_adapter.health_status}")
        print(f"   Is initialized: {ma_adapter.is_initialized}")
        
        # Try health check
        print("   Performing health check...")
        health_result = await ma_adapter.health_check()
        print(f"   Health check result: {health_result}")
        
    except Exception as e:
        print(f"❌ Multi-Agent Adapter failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
    
    # Test Genetic Algorithm Adapter
    print("\n🧬 Testing Genetic Algorithm Adapter...")
    try:
        from integration.adapters import GeneticAlgorithmAdapter
        ga_adapter = GeneticAlgorithmAdapter()
        print(f"✅ GeneticAlgorithmAdapter created: {ga_adapter.component_name}")
        print(f"   Initial status: {ga_adapter.health_status}")
        print(f"   Is initialized: {ga_adapter.is_initialized}")
        
        # Try to initialize
        print("   Attempting initialization...")
        init_result = await ga_adapter.initialize()
        print(f"   Initialization result: {init_result}")
        print(f"   Post-init status: {ga_adapter.health_status}")
        print(f"   Is initialized: {ga_adapter.is_initialized}")
        
        # Try health check
        print("   Performing health check...")
        health_result = await ga_adapter.health_check()
        print(f"   Health check result: {health_result}")
        
    except Exception as e:
        print(f"❌ Genetic Algorithm Adapter failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
    
    # Test Predictive Adapter
    print("\n🔮 Testing Predictive Adapter...")
    try:
        from integration.adapters import PredictiveAdapter
        pred_adapter = PredictiveAdapter()
        print(f"✅ PredictiveAdapter created: {pred_adapter.component_name}")
        print(f"   Initial status: {pred_adapter.health_status}")
        print(f"   Is initialized: {pred_adapter.is_initialized}")
        
        # Try to initialize
        print("   Attempting initialization...")
        init_result = await pred_adapter.initialize()
        print(f"   Initialization result: {init_result}")
        print(f"   Post-init status: {pred_adapter.health_status}")
        print(f"   Is initialized: {pred_adapter.is_initialized}")
        
        # Try health check
        print("   Performing health check...")
        health_result = await pred_adapter.health_check()
        print(f"   Health check result: {health_result}")
        
    except Exception as e:
        print(f"❌ Predictive Adapter failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")

async def diagnose_integration_engine():
    """Diagnose integration engine health monitoring"""
    
    print("\n🔧 INTEGRATION ENGINE DIAGNOSIS")
    print("-" * 50)
    
    try:
        from integration.core import IntegrationEngine
        from integration.adapters import NLPAdapter
        
        # Create integration engine
        engine = IntegrationEngine()
        print(f"✅ IntegrationEngine created")
        print(f"   Status: {engine.status}")
        print(f"   Components registered: {len(engine.components)}")
        
        # Start the engine
        print("   Starting integration engine...")
        await engine.start()
        print(f"   Post-start status: {engine.status}")
        
        # Register a component
        print("   Registering NLP adapter...")
        nlp_adapter = NLPAdapter()
        
        from integration.core import ComponentInfo, ComponentType
        component_info = ComponentInfo(
            component_id="test_nlp",
            component_type=ComponentType.NLP_SYSTEM,
            name="Test NLP Component",
            version="1.0",
            capabilities=["text_analysis"]
        )
        
        engine.register_component(component_info, nlp_adapter)
        print(f"   Components after registration: {len(engine.components)}")
        
        # Initialize the adapter
        print("   Initializing registered adapter...")
        init_result = await nlp_adapter.initialize()
        print(f"   Adapter initialization: {init_result}")
        
        # Perform health checks
        print("   Performing engine health checks...")
        await engine._perform_health_checks()
        
        # Check component status
        for comp_id, comp_info in engine.components.items():
            print(f"   Component {comp_id}:")
            print(f"     Name: {comp_info.name}")
            print(f"     Status: {comp_info.status}")
            print(f"     Health Score: {comp_info.health_score}")
            print(f"     Last Health Check: {comp_info.last_health_check}")
        
        # Stop the engine
        print("   Stopping integration engine...")
        await engine.stop()
        print(f"   Final status: {engine.status}")
        
    except Exception as e:
        print(f"❌ Integration Engine diagnosis failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")

async def diagnose_historical_workflow():
    """Diagnose historical research workflow component registration"""
    
    print("\n🏛️ HISTORICAL WORKFLOW DIAGNOSIS")
    print("-" * 50)
    
    try:
        from workflows.historical_research import HistoricalResearchWorkflow
        
        # Create workflow
        workflow = HistoricalResearchWorkflow()
        print(f"✅ HistoricalResearchWorkflow created")
        
        # Check initial state
        print(f"   Integration engine: {workflow.integration_engine is not None}")
        print(f"   Supported topics: {len(workflow.supported_topics)}")
        
        # Try initialization
        print("   Attempting workflow initialization...")
        init_result = await workflow.initialize()
        print(f"   Initialization result: {init_result}")
        
        # Check integration engine state
        if workflow.integration_engine:
            engine = workflow.integration_engine
            print(f"   Engine status: {engine.status}")
            print(f"   Registered components: {len(engine.components)}")
            
            # List all registered components
            for comp_id, comp_info in engine.components.items():
                print(f"     Component: {comp_id}")
                print(f"       Name: {comp_info.name}")
                print(f"       Type: {comp_info.component_type}")
                print(f"       Status: {comp_info.status}")
                print(f"       Health Score: {comp_info.health_score}")
                
                # Get adapter and check its health
                if comp_id in engine.adapters:
                    adapter = engine.adapters[comp_id]
                    print(f"       Adapter initialized: {adapter.is_initialized}")
                    print(f"       Adapter health status: {adapter.health_status}")
        
    except Exception as e:
        print(f"❌ Historical Workflow diagnosis failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")

async def main():
    """Main diagnostic function"""
    try:
        await diagnose_adapters()
        await diagnose_integration_engine()
        await diagnose_historical_workflow()
        
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print("✅ Diagnostic completed successfully")
        print("📝 Check the output above for specific component issues")
        print("💡 Common issues:")
        print("   • Missing AI component dependencies")
        print("   • Import path problems")
        print("   • Adapter initialization failures")
        print("   • Health check implementation issues")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    finally:
        print(f"\n✨ Diagnostic session ended at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    print("🚀 Starting component health diagnostic...")
    asyncio.run(main())
