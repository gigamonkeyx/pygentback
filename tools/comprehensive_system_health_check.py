#!/usr/bin/env python3
"""
Complete System Health Check for PyGent Factory
Validates all major components and identifies remaining issues
"""

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.logging import get_logger
from core.agent_factory import AgentFactory
from ai.providers.provider_registry import ProviderRegistry
from workflows.research_analysis_orchestrator import ResearchAnalysisOrchestrator
from orchestration.orchestration_manager import OrchestrationManager
from orchestration.task_dispatcher import TaskDispatcher

logger = get_logger(__name__)

class SystemHealthChecker:
    """Comprehensive system health checker"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    async def check_provider_system(self) -> Dict[str, Any]:
        """Check provider registry and providers"""
        logger.info("🔍 Checking Provider System...")
        
        try:
            registry = ProviderRegistry()
            await registry.initialize()
            
            health = await registry.get_system_health()
            providers = await registry.list_available_providers()
            
            return {
                "status": "healthy" if health["status"] == "healthy" else "degraded",
                "providers": providers,
                "health": health
            }
        except Exception as e:
            logger.error(f"Provider system check failed: {e}")
            self.errors.append(f"Provider System: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def check_agent_factory(self) -> Dict[str, Any]:
        """Check agent factory functionality"""
        logger.info("🔍 Checking Agent Factory...")
        
        try:
            factory = AgentFactory()
            await factory.initialize()
            
            # Test agent creation
            agent = await factory.create_agent(
                agent_type="coding",
                provider="ollama",
                model="qwen3:8b"
            )
            
            if agent:
                await factory.destroy_agent(agent.agent_id)
                return {
                    "status": "healthy",
                    "can_create_agents": True,
                    "memory_cleanup": True
                }
            else:
                return {
                    "status": "degraded",
                    "can_create_agents": False,
                    "error": "Agent creation returned None"
                }
                
        except Exception as e:
            logger.error(f"Agent factory check failed: {e}")
            self.errors.append(f"Agent Factory: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def check_research_system(self) -> Dict[str, Any]:
        """Check research orchestration system"""
        logger.info("🔍 Checking Research System...")
        
        try:
            # Check standalone research orchestrator
            research_orchestrator = ResearchAnalysisOrchestrator()
            
            # Check main orchestration system
            orchestration_manager = OrchestrationManager()
            task_dispatcher = TaskDispatcher(orchestration_manager)
            
            return {
                "status": "healthy",
                "research_orchestrator": "available",
                "main_orchestrator": "available",
                "task_dispatcher": "available",
                "integration_gap": True,  # Known issue
                "notes": "Research orchestrator is standalone, not integrated with main system"
            }
            
        except Exception as e:
            logger.error(f"Research system check failed: {e}")
            self.errors.append(f"Research System: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def check_file_structure(self) -> Dict[str, Any]:
        """Check critical file structure"""
        logger.info("🔍 Checking File Structure...")
        
        critical_files = [
            "src/core/agent_factory.py",
            "src/ai/providers/provider_registry.py", 
            "src/ai/providers/ollama_provider.py",
            "src/ai/providers/openrouter_provider.py",
            "src/workflows/research_analysis_orchestrator.py",
            "src/orchestration/orchestration_manager.py",
            "src/orchestration/task_dispatcher.py"
        ]
        
        missing_files = []
        for file_path in critical_files:
            if not (project_root / file_path).exists():
                missing_files.append(file_path)
        
        return {
            "status": "healthy" if not missing_files else "degraded",
            "missing_files": missing_files,
            "critical_files_present": len(critical_files) - len(missing_files)
        }
    
    async def check_import_integrity(self) -> Dict[str, Any]:
        """Check if all critical imports work"""
        logger.info("🔍 Checking Import Integrity...")
        
        import_tests = [
            ("core.agent_factory", "AgentFactory"),
            ("ai.providers.provider_registry", "ProviderRegistry"),
            ("ai.providers.ollama_provider", "OllamaProvider"),
            ("ai.providers.openrouter_provider", "OpenRouterProvider"),
            ("workflows.research_analysis_orchestrator", "ResearchAnalysisOrchestrator"),
            ("orchestration.orchestration_manager", "OrchestrationManager"),
            ("orchestration.task_dispatcher", "TaskDispatcher")
        ]
        
        failed_imports = []
        for module_name, class_name in import_tests:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
            except Exception as e:
                failed_imports.append(f"{module_name}.{class_name}: {str(e)}")
        
        return {
            "status": "healthy" if not failed_imports else "degraded",            "failed_imports": failed_imports,
            "successful_imports": len(import_tests) - len(failed_imports)
        }
    
    async def check_ollama_integration(self) -> Dict[str, Any]:
        """Check Ollama integration specifically"""
        logger.info("🔍 Checking Ollama Integration...")
        
        try:
            from ai.providers.ollama_provider import OllamaProvider
            from core.ollama_manager import get_ollama_manager
            
            # Test provider directly
            provider = OllamaProvider()
            provider_ok = await provider.initialize()
            
            if not provider_ok:
                return {
                    "status": "failed",
                    "error": "Ollama provider failed to initialize",
                    "available_models": [],
                    "manager_status": "not tested"
                }
            
            models = await provider.get_available_models()
            
            # Test manager
            manager = get_ollama_manager()
            manager_ok = await manager.start()
            manager_models = await manager.get_available_models() if manager_ok else []
            
            # Test basic generation if models available
            generation_ok = False
            generation_response = ""
            if models and provider_ok:
                try:
                    test_model = "qwen3:8b" if "qwen3:8b" in models else models[0]
                    response = await provider.generate_text(
                        prompt="Test: What is 1+1?",
                        model=test_model,
                        max_tokens=20
                    )
                    generation_ok = bool(response and len(response.strip()) > 0)
                    generation_response = response[:50] if response else ""
                except Exception as e:
                    logger.warning(f"Generation test failed: {e}")
            
            return {
                "status": "healthy" if provider_ok and manager_ok else "degraded",
                "provider_initialized": provider_ok,
                "manager_started": manager_ok,
                "available_models": models,
                "manager_models": manager_models,
                "generation_test": {
                    "passed": generation_ok,
                    "response": generation_response
                }
            }
            
        except Exception as e:
            logger.error(f"Ollama integration check failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "available_models": [],
                "manager_status": "error"
            }

    async def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all system checks"""
        logger.info("🚀 Starting Comprehensive System Health Check...")
        
        checks = {
            "file_structure": await self.check_file_structure(),
            "import_integrity": await self.check_import_integrity(),
            "provider_system": await self.check_provider_system(),
            "agent_factory": await self.check_agent_factory(),
            "research_system": await self.check_research_system(),
            "ollama_integration": await self.check_ollama_integration()
        }
        
        # Calculate overall health
        healthy_checks = sum(1 for check in checks.values() if check.get("status") == "healthy")
        total_checks = len(checks)
        health_percentage = (healthy_checks / total_checks) * 100
        
        overall_status = "healthy" if health_percentage > 80 else "degraded" if health_percentage > 50 else "critical"
        
        return {
            "overall_status": overall_status,
            "health_percentage": health_percentage,
            "checks": checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self._generate_recommendations(checks)
        }
    
    def _generate_recommendations(self, checks: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on check results"""
        recommendations = []
        
        # Check for critical issues
        if checks["file_structure"]["status"] != "healthy":
            recommendations.append("🔧 Fix missing critical files before deployment")
        
        if checks["import_integrity"]["status"] != "healthy":
            recommendations.append("🔧 Resolve import errors - system won't function properly")
        
        if checks["provider_system"]["status"] == "failed":
            recommendations.append("🚨 Provider system is broken - agents cannot be created")
        
        if checks["agent_factory"]["status"] == "failed":
            recommendations.append("🚨 Agent factory is broken - core functionality unavailable")
        
        # Check for known integration gaps
        if checks["research_system"].get("integration_gap"):
            recommendations.append("📋 Integrate research orchestrator with main task dispatcher")
        
        # Performance recommendations
        if len(self.errors) == 0 and len(recommendations) == 0:
            recommendations.append("✅ System is production ready!")
            recommendations.append("📋 Consider implementing performance monitoring")
            recommendations.append("📋 Add automated testing pipeline")
        
        return recommendations

async def main():
    """Main execution function"""
    print("=" * 80)
    print("🏥 PYGENT FACTORY - COMPREHENSIVE SYSTEM HEALTH CHECK")
    print("=" * 80)
    
    checker = SystemHealthChecker()
    
    try:
        results = await checker.run_comprehensive_check()
        
        print(f"\n📊 OVERALL SYSTEM STATUS: {results['overall_status'].upper()}")
        print(f"📈 Health Score: {results['health_percentage']:.1f}%")
        
        print("\n" + "=" * 50)
        print("📋 DETAILED RESULTS")
        print("=" * 50)
        
        for check_name, check_result in results["checks"].items():
            status_emoji = {"healthy": "✅", "degraded": "⚠️", "failed": "❌"}.get(check_result["status"], "❓")
            print(f"\n{status_emoji} {check_name.replace('_', ' ').title()}: {check_result['status'].upper()}")
            
            # Print details
            for key, value in check_result.items():
                if key != "status" and key != "error":
                    if isinstance(value, list) and value:
                        print(f"   • {key}: {len(value)} items")
                        for item in value[:3]:  # Show first 3 items
                            print(f"     - {item}")
                        if len(value) > 3:
                            print(f"     ... and {len(value) - 3} more")
                    elif isinstance(value, (str, int, float, bool)):
                        print(f"   • {key}: {value}")
            
            if "error" in check_result:
                print(f"   ❌ Error: {check_result['error']}")
        
        # Show errors
        if results["errors"]:
            print("\n" + "=" * 50)
            print("❌ CRITICAL ERRORS")
            print("=" * 50)
            for error in results["errors"]:
                print(f"• {error}")
        
        # Show warnings
        if results["warnings"]:
            print("\n" + "=" * 50)
            print("⚠️ WARNINGS")
            print("=" * 50)
            for warning in results["warnings"]:
                print(f"• {warning}")
        
        # Show recommendations
        print("\n" + "=" * 50)
        print("💡 RECOMMENDATIONS")
        print("=" * 50)
        for rec in results["recommendations"]:
            print(f"{rec}")
        
        print("\n" + "=" * 80)
        print("✅ HEALTH CHECK COMPLETE")
        print("=" * 80)
        
        return results["overall_status"] == "healthy"
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        print(f"\n❌ HEALTH CHECK FAILED: {str(e)}")
        print(f"📋 Stacktrace:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
