#!/usr/bin/env python3
"""
PyGent Factory Main Launcher

Production launcher for the PyGent Factory AI system with support for
different deployment modes and comprehensive system initialization.
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.config_manager import initialize_config, ConfigError


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def check_dependencies():
    """Check for required dependencies"""
    
    missing_deps = []
    optional_deps = []
    
    # Required dependencies
    required = [
        ("numpy", "NumPy for numerical computations"),
        ("yaml", "PyYAML for configuration files")
    ]
    
    for module, description in required:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(f"{module} - {description}")
    
    # Optional dependencies
    optional = [
        ("fastapi", "FastAPI for REST API server"),
        ("uvicorn", "Uvicorn for ASGI server"),
        ("faiss", "FAISS for GPU vector search"),
        ("pydantic", "Pydantic for API validation"),
        ("psutil", "psutil for system monitoring"),
        ("GPUtil", "GPUtil for GPU monitoring")
    ]
    
    for module, description in optional:
        try:
            __import__(module)
        except ImportError:
            optional_deps.append(f"{module} - {description}")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: pip install numpy pyyaml")
        return False
    
    if optional_deps:
        print("⚠️ Optional dependencies not available:")
        for dep in optional_deps:
            print(f"  - {dep}")
        print("\nFor full functionality, install with:")
        print("pip install fastapi uvicorn faiss-gpu pydantic psutil GPUtil")
        print()
    
    return True


async def run_api_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the API server"""

    try:
        import uvicorn
        from src.api.main import create_app

        print(f"🚀 Starting PyGent Factory API Server")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Workers: {workers}")
        print()

        app = create_app()
        if app is None:
            print("❌ Failed to create API application")
            return False

        # Create server configuration
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

        # Create and run server
        server = uvicorn.Server(config)
        print(f"✅ Server starting at http://{host}:{port}")
        print("   Press Ctrl+C to stop the server")
        print()

        await server.serve()

        return True

    except ImportError:
        print("❌ FastAPI/Uvicorn not available - cannot run API server")
        print("Install with: pip install fastapi uvicorn")
        return False
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False


async def run_reasoning_demo():
    """Run reasoning system demonstration"""
    
    print("🧠 PyGent Factory Reasoning Demo")
    print("=" * 40)
    
    try:
        from src.ai.reasoning.unified_pipeline import (
            UnifiedReasoningPipeline, UnifiedConfig, ReasoningMode
        )
        
        # Create professional demo components
        class DemoRetriever:
            async def retrieve(self, query: str, k: int = 5):
                # Simulate realistic document retrieval
                documents = [
                    {"content": f"Technical documentation: {query} involves systematic analysis and implementation strategies.", "score": 0.92},
                    {"content": f"Research findings: Studies show that {query} can be optimized through advanced methodologies.", "score": 0.87},
                    {"content": f"Best practices: Industry standards for {query} emphasize scalability and performance.", "score": 0.83},
                    {"content": f"Case study: Successful implementation of {query} in enterprise environments.", "score": 0.79},
                    {"content": f"Expert analysis: {query} requires careful consideration of multiple factors.", "score": 0.75}
                ]
                return documents[:k]

        class DemoGenerator:
            async def generate(self, query: str, context: str):
                return f"Based on comprehensive analysis and available documentation, here's a detailed response to your query about {query}: The system has processed relevant information and can provide structured insights with supporting evidence from multiple sources."
        
        # Initialize pipeline
        config = UnifiedConfig(reasoning_mode=ReasoningMode.ADAPTIVE)
        pipeline = UnifiedReasoningPipeline(config, DemoRetriever(), DemoGenerator())
        
        # Test queries
        queries = [
            "What is artificial intelligence?",
            "How can I optimize system performance?",
            "Design a strategy for improving code quality and maintainability"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 50)
            
            result = await pipeline.reason(query)
            
            if result.success:
                print(f"✅ Mode: {result.reasoning_mode.value}")
                print(f"✅ Time: {result.total_time:.2f}s")
                print(f"✅ Confidence: {result.confidence_score:.3f}")
                print(f"✅ Response: {result.response[:100]}...")
            else:
                print(f"❌ Failed: {result.error_message}")
        
        print(f"\n📊 Pipeline Statistics:")
        stats = pipeline.get_pipeline_statistics()
        print(f"Total queries: {stats['query_count']}")
        print(f"Mode usage: {stats['mode_usage']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning demo failed: {e}")
        return False


async def run_evolution_demo():
    """Run evolution system demonstration"""
    
    print("\n🧬 PyGent Factory Evolution Demo")
    print("=" * 40)
    
    try:
        from src.evolution.advanced_recipe_evolution import (
            AdvancedRecipeEvolution, EvolutionConfig, Recipe, EvolutionStrategy
        )
        from src.ai.reasoning.unified_pipeline import UnifiedReasoningPipeline, UnifiedConfig
        
        # Create mock components
        class MockRetriever:
            async def retrieve(self, query: str, k: int = 5):
                return [{"content": f"Mock document about {query}", "score": 0.8}]
        
        class MockGenerator:
            async def generate(self, query: str, context: str):
                return f"Optimized approach: {query}"
        
        # Initialize systems
        reasoning_config = UnifiedConfig()
        reasoning_pipeline = UnifiedReasoningPipeline(reasoning_config, MockRetriever(), MockGenerator())
        
        evolution_config = EvolutionConfig(
            population_size=6,
            max_generations=3,
            evolution_strategy=EvolutionStrategy.HYBRID
        )
        evolution_system = AdvancedRecipeEvolution(evolution_config, reasoning_pipeline)
        
        # Create initial recipes
        initial_recipes = [
            Recipe(
                id="recipe1",
                name="Basic Processing",
                description="Simple data processing",
                steps=["Load", "Process", "Save"]
            ),
            Recipe(
                id="recipe2",
                name="Optimized Processing",
                description="Faster data processing",
                steps=["Load in parallel", "Process with caching", "Save efficiently"]
            )
        ]
        
        objectives = ["Improve speed", "Reduce memory usage", "Increase reliability"]
        
        print(f"Starting evolution with {len(initial_recipes)} recipes")
        print(f"Objectives: {', '.join(objectives)}")
        print()
        
        # Run evolution
        result = await evolution_system.evolve_recipes(initial_recipes, objectives)
        
        if result['success']:
            print(f"✅ Evolution completed!")
            print(f"Generations: {result['generations_completed']}")
            print(f"Time: {result['total_time']:.2f}s")
            print(f"Best recipes: {len(result['best_recipes'])}")
            
            # Show best recipe
            if result['best_recipes']:
                best = result['best_recipes'][0]
                print(f"\n🏆 Best Recipe: {best['name']}")
                print(f"Fitness: {best['composite_fitness']:.3f}")
                print(f"Description: {best['description'][:100]}...")
        else:
            print(f"❌ Evolution failed: {result.get('error')}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Evolution demo failed: {e}")
        return False


async def run_research_demo(topic: Optional[str] = None):
    """Run AI-enhanced historical research demonstration"""
    
    print("📚 PyGent Factory AI Historical Research")
    print("=" * 50)
    
    try:
        from src.core.gpu_config import gpu_manager
        from src.core.ollama_integration import ollama_manager
        from src.core.openrouter_integration import openrouter_manager
        from src.core.agent_orchestrator import agent_orchestrator
        from src.core.research_workflow import research_workflow, ResearchWorkflowConfig
        from src.core.academic_pdf import report_generator
        
        # Initialize systems
        print("🔧 Initializing AI research systems...")
        
        # Initialize GPU configuration
        gpu_initialized = gpu_manager.initialize()
        if gpu_initialized and gpu_manager.config.device != "cpu":
            print(f"✅ GPU detected: {gpu_manager.config.device_name} ({gpu_manager.config.memory_total/1024:.1f}GB)")
        else:
            print("⚠️ No GPU detected, using CPU only")
        
        # Initialize Ollama
        if await ollama_manager.initialize():
            print("✅ Ollama local AI initialized")
            available_models = ollama_manager.get_available_models()
            model_names = list(available_models.keys())
            print(f"   Available models: {', '.join(model_names[:3])}")
        else:
            print("⚠️ Ollama not available")
        
        # Initialize OpenRouter
        if await openrouter_manager.initialize():
            print("✅ OpenRouter external AI initialized")
        else:
            print("⚠️ OpenRouter not available")
        
        # Start agent orchestrator
        await agent_orchestrator.start()
        print("✅ AI agent orchestrator started")
        
        # Configure research topic
        if not topic:
            topic = input("\n📝 Enter research topic (or press Enter for demo): ").strip()
            if not topic:
                topic = "The Impact of the Printing Press on 15th Century Europe"
        
        print(f"\n🔍 Starting research on: {topic}")
        
        # Create research configuration
        config = ResearchWorkflowConfig(
            topic=topic,
            scope="comprehensive",
            max_documents=10,
            fact_check_threshold=0.7,
            use_external_models=True,
            budget_limit=5.0,
            timeout_minutes=30
        )
        
        # Start research workflow
        workflow_id = await research_workflow.start_research(config)
        print(f"✅ Research workflow started (ID: {workflow_id})")
        
        # Monitor progress
        print("\n⏳ Monitoring research progress...")
        import time
        start_time = time.time()
        
        while True:
            workflow_status = await research_workflow.get_workflow_status(workflow_id)
            if not workflow_status:
                print("❌ Lost track of workflow")
                return False
            
            elapsed = time.time() - start_time
            print(f"   Status: {workflow_status.status} (elapsed: {elapsed:.1f}s)")
            
            if workflow_status.status in ["completed", "failed"]:
                break
            elif elapsed > config.timeout_minutes * 60:
                print("⏰ Research timed out")
                return False
            
            await asyncio.sleep(2)
        
        # Check results
        if workflow_status.status == "completed":
            print(f"🎉 Research completed in {elapsed:.1f} seconds!")
            
            # Generate academic report
            print("\n📄 Generating academic report...")
            report_path = await report_generator.generate_research_report(
                research_data={
                    "topic": topic,
                    "results": workflow_status.results
                },
                output_format="pdf",
                citation_style="apa"
            )
            
            print(f"✅ Academic report generated: {report_path}")
            
            # Display summary
            synthesis = workflow_status.results.get("synthesis", {})
            metadata = synthesis.get("synthesis_metadata", {})
            
            print("\n📊 Research Summary:")
            print(f"   Documents analyzed: {metadata.get('documents_analyzed', 0)}")
            print(f"   Claims verified: {metadata.get('claims_verified', 0)}")
            print(f"   Credibility score: {metadata.get('credibility_score', 0):.2%}")
            print(f"   Processing time: {metadata.get('workflow_duration', 0):.1f}s")
            
            return True
            
        else:
            print(f"❌ Research failed: {workflow_status.error}")
            return False
        
    except ImportError as e:
        print(f"❌ Missing AI research dependencies: {e}")
        print("Install with: pip install transformers ollama openai")
        return False
    except Exception as e:
        print(f"❌ Research demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            await agent_orchestrator.stop()
        except Exception:
            pass


def run_system_test():
    """Run comprehensive system test"""
    
    print("🧪 PyGent Factory System Test")
    print("=" * 35)
    
    tests = [
        ("Configuration", test_configuration),
        ("Core Imports", test_core_imports),
        ("AI Components", test_ai_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n📈 Test Results: {passed}/{total} passed")
    return passed == total


def test_configuration():
    """Test configuration system"""
    try:
        config_manager = initialize_config()
        
        # Test basic configuration access
        system_config = config_manager.get_section('system')
        if not system_config:
            return False
        
        # Test environment detection
        env = config_manager.get_environment()
        if not env:
            return False
        
        return True
    except Exception:
        return False


def test_core_imports():
    """Test core module imports"""
    try:
        # Test AI reasoning imports
        from src.ai.reasoning.tot.models import ToTConfig, ThoughtState
        from src.ai.reasoning.unified_pipeline import UnifiedReasoningPipeline
        
        # Test RAG imports
        from src.rag.s3.models import S3Config, SearchState
        
        # Test evolution imports
        from src.evolution.advanced_recipe_evolution import Recipe, EvolutionConfig
        
        # Test search imports
        from src.search.gpu_search import VectorSearchConfig
        
        return True
    except ImportError:
        return False


def test_ai_components():
    """Test AI component initialization"""
    try:
        from src.ai.reasoning.tot.models import ToTConfig
        from src.rag.s3.models import S3Config
        from src.evolution.advanced_recipe_evolution import EvolutionConfig
        
        # Test configuration creation
        tot_config = ToTConfig()
        s3_config = S3Config()
        evolution_config = EvolutionConfig()
        
        # Test basic functionality
        assert tot_config.max_depth > 0
        assert s3_config.max_search_iterations > 0
        assert evolution_config.population_size > 0
        
        return True
    except Exception:
        return False


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="PyGent Factory - Advanced AI Reasoning and Optimization System"
    )
    
    parser.add_argument(
        "mode",
        choices=["server", "demo", "test", "reasoning", "evolution", "research"],
        help="Execution mode"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of API server workers (default: 1)"
    )
    
    parser.add_argument(
        "--config-dir",
        help="Configuration directory path"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    print("🚀 PyGent Factory - Advanced AI System")
    print("=" * 45)
    print("Tree of Thought + s3 RAG + GPU Vector Search")
    print()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Initialize configuration
    try:
        initialize_config(args.config_dir)
        logger.info("Configuration initialized successfully")
    except ConfigError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Failed to initialize configuration: {e}")
        return 1
    
    # Execute based on mode
    success = False
    
    if args.mode == "server":
        success = await run_api_server(args.host, args.port, args.workers)
    
    elif args.mode == "test":
        success = run_system_test()
    
    elif args.mode == "reasoning":
        success = await run_reasoning_demo()
    
    elif args.mode == "evolution":
        success = await run_evolution_demo()
    
    elif args.mode == "research":
        success = await run_research_demo()
    
    elif args.mode == "demo":
        print("Running comprehensive demonstration...")
        reasoning_success = await run_reasoning_demo()
        evolution_success = await run_evolution_demo()
        success = reasoning_success and evolution_success
    
    if success:
        print("\n🎉 Execution completed successfully!")
        return 0
    else:
        print("\n❌ Execution completed with errors")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
