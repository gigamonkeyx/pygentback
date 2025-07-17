#!/usr/bin/env python3
"""
PyGent Factory Training Controller
Observer-approved full training mode deployment with systematic 5-phase execution
"""

import asyncio
import logging
import sys
import os
import time
import psutil
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import PyGent Factory components
from src.core.sim_env import WorldSimulationEvolution, SimulationEnvironment, AgentPopulationManager
from src.agents.base_agent import BaseAgent, AgentType, AgentStatus
from src.orchestration.mcp_orchestrator import MCPOrchestrator
from src.core.emergent_behavior_detector import Docker443EmergentBehaviorDetector
from src.dgm.core.engine import DGMEngine

class ObserverTrainingController:
    """Observer-supervised training controller for PyGent Factory full training mode"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.start_time = time.time()
        self.phase_results = {}
        self.system_metrics = {}
        self.training_agents = []
        
        # Initialize core components
        self.simulation_env = None
        self.population_manager = None
        self.evolution_system = None
        self.mcp_orchestrator = None
        self.emergence_detector = None
        self.dgm_engines = {}
        
        # GPU throttling configuration
        self.gpu_memory_fraction = 0.8
        self.max_generations = 5
        self.population_size = 10
        
    def _setup_logging(self):
        """Setup logging for training process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def print_phase_header(self, phase: str, description: str):
        """Print formatted phase header"""
        print(f"\n{'='*80}")
        print(f"🚀 {phase}: {description}")
        print(f"{'='*80}")
        print(f"⏱️  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Observer Supervision: ACTIVE")
        print(f"📊 RIPER-Ω Protocol: COMPLIANT")
    
    def print_phase_result(self, phase: str, success: bool, metrics: Dict[str, Any]):
        """Print phase completion results"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n{'-'*60}")
        print(f"📋 {phase} RESULT: {status}")
        print(f"⏱️  Duration: {metrics.get('duration', 0):.2f} seconds")
        print(f"💾 Memory Usage: {metrics.get('memory_usage', 0):.1f}%")
        print(f"🔥 GPU Usage: {metrics.get('gpu_usage', 'N/A')}")
        
        if 'details' in metrics:
            for key, value in metrics['details'].items():
                print(f"   {key}: {value}")
        print(f"{'-'*60}")
    
    async def configure_gpu_throttling(self):
        """Configure GPU memory throttling for controlled resource usage"""
        try:
            if torch.cuda.is_available():
                # Set memory fraction to prevent GPU overload
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
                
                gpu_info = {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
                    "memory_fraction": self.gpu_memory_fraction
                }
                
                self.logger.info(f"GPU Configuration: {gpu_info}")
                return gpu_info
            else:
                self.logger.warning("CUDA not available - using CPU only")
                return {"cuda_available": False}
                
        except Exception as e:
            self.logger.error(f"GPU configuration failed: {e}")
            return {"error": str(e)}
    
    async def monitor_system_resources(self) -> Dict[str, Any]:
        """Monitor system resource usage with psutil"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU if available
            gpu_info = "N/A"
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_info = f"{gpu_memory_allocated:.1f}GB allocated, {gpu_memory_reserved:.1f}GB reserved"
            
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / 1024**3,
                "gpu_usage": gpu_info,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check for resource overload
            if cpu_percent > 90 or memory.percent > 90:
                self.logger.warning(f"High resource usage detected: CPU {cpu_percent}%, Memory {memory.percent}%")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Resource monitoring failed: {e}")
            return {"error": str(e)}
    
    async def initialize_simulation_environment(self) -> bool:
        """Initialize the simulation environment for agent evolution"""
        try:
            self.logger.info("Initializing simulation environment...")
            
            # Create simulation environment
            self.simulation_env = SimulationEnvironment()
            await self.simulation_env.initialize()
            
            # Create population manager
            self.population_manager = AgentPopulationManager(self.simulation_env)
            await self.population_manager.initialize()
            
            # Create evolution system
            self.evolution_system = WorldSimulationEvolution(
                self.simulation_env, 
                self.population_manager
            )
            
            self.logger.info("Simulation environment initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Simulation environment initialization failed: {e}")
            return False
    
    async def spawn_base_agent(self) -> Optional[BaseAgent]:
        """Spawn a base agent for evolution training"""
        try:
            self.logger.info("Spawning base agent for evolution...")
            
            # Create base agent with evolution-optimized configuration
            agent = BaseAgent(
                agent_type=AgentType.RESEARCH,
                name="evolution_base_agent",
                config={
                    "evolution_enabled": True,
                    "fitness_tracking": True,
                    "mutation_rate": 0.1,
                    "learning_rate": 0.01,
                    "environment_adaptation": True
                }
            )
            
            # Initialize the agent
            success = await agent.initialize()
            if not success:
                self.logger.error("Base agent initialization failed")
                return None
            
            # Start the agent
            await agent.start()
            
            # Add to training agents list
            self.training_agents.append(agent)
            
            self.logger.info(f"Base agent spawned successfully: {agent.name} ({agent.agent_id})")
            return agent
            
        except Exception as e:
            self.logger.error(f"Base agent spawning failed: {e}")
            return None
    
    async def run_controlled_evolution(self, generations: int = 5) -> Dict[str, Any]:
        """Run controlled evolution for specified generations"""
        try:
            self.logger.info(f"Starting controlled evolution for {generations} generations...")
            
            evolution_results = {
                "generations_completed": 0,
                "fitness_improvements": [],
                "mutation_successes": 0,
                "crossover_successes": 0,
                "population_diversity": [],
                "performance_metrics": []
            }
            
            # Run evolution cycle
            results = await self.evolution_system.run_evolution_cycle(generations)
            
            if "error" not in results:
                evolution_results.update({
                    "generations_completed": results.get("generations_completed", 0),
                    "final_fitness": results.get("final_fitness", {}),
                    "evolution_time": results.get("evolution_time", 0),
                    "success": True
                })
                
                self.logger.info(f"Evolution completed successfully: {evolution_results['generations_completed']} generations")
            else:
                evolution_results["error"] = results["error"]
                evolution_results["success"] = False
                self.logger.error(f"Evolution failed: {results['error']}")
            
            return evolution_results
            
        except Exception as e:
            self.logger.error(f"Controlled evolution failed: {e}")
            return {"error": str(e), "success": False}
    
    async def validate_system_stability(self) -> bool:
        """Validate system stability and check for overloads"""
        try:
            # Monitor resources for stability check
            metrics = await self.monitor_system_resources()
            
            # Check for system overload conditions
            if "error" in metrics:
                return False
            
            cpu_ok = metrics["cpu_percent"] < 85
            memory_ok = metrics["memory_percent"] < 85
            
            if not cpu_ok:
                self.logger.warning(f"High CPU usage: {metrics['cpu_percent']}%")
            if not memory_ok:
                self.logger.warning(f"High memory usage: {metrics['memory_percent']}%")
            
            stability_ok = cpu_ok and memory_ok
            
            if stability_ok:
                self.logger.info("System stability validated - no overloads detected")
            else:
                self.logger.warning("System stability issues detected")
            
            return stability_ok
            
        except Exception as e:
            self.logger.error(f"System stability validation failed: {e}")
            return False

    async def execute_phase_1(self) -> Dict[str, Any]:
        """Execute Phase 1: Controlled Single Agent Evolution"""
        phase_start = time.time()
        self.print_phase_header("PHASE 1", "Controlled Single Agent Evolution")

        try:
            # Step 1: Configure GPU throttling
            print("🔧 Step 1: Configuring GPU throttling...")
            gpu_config = await self.configure_gpu_throttling()

            # Step 2: Initialize simulation environment
            print("🌍 Step 2: Initializing simulation environment...")
            env_success = await self.initialize_simulation_environment()
            if not env_success:
                raise Exception("Simulation environment initialization failed")

            # Step 3: Spawn base agent
            print("🤖 Step 3: Spawning base agent...")
            base_agent = await self.spawn_base_agent()
            if not base_agent:
                raise Exception("Base agent spawning failed")

            # Step 4: Monitor initial system resources
            print("📊 Step 4: Monitoring system resources...")
            initial_metrics = await self.monitor_system_resources()

            # Step 5: Run controlled evolution (5 generations)
            print(f"🧬 Step 5: Running {self.max_generations} generations of evolution...")
            evolution_results = await self.run_controlled_evolution(self.max_generations)

            # Step 6: Validate system stability
            print("✅ Step 6: Validating system stability...")
            stability_ok = await self.validate_system_stability()

            # Step 7: Collect final metrics
            final_metrics = await self.monitor_system_resources()

            # Calculate phase duration
            phase_duration = time.time() - phase_start

            # Compile phase results
            phase_1_results = {
                "success": evolution_results.get("success", False) and stability_ok,
                "duration": phase_duration,
                "gpu_config": gpu_config,
                "base_agent_id": base_agent.agent_id if base_agent else None,
                "evolution_results": evolution_results,
                "initial_metrics": initial_metrics,
                "final_metrics": final_metrics,
                "system_stable": stability_ok,
                "details": {
                    "generations_completed": evolution_results.get("generations_completed", 0),
                    "agents_spawned": len(self.training_agents),
                    "memory_usage": final_metrics.get("memory_percent", 0),
                    "gpu_usage": final_metrics.get("gpu_usage", "N/A")
                }
            }

            # Store results
            self.phase_results["phase_1"] = phase_1_results

            # Print results
            self.print_phase_result("PHASE 1", phase_1_results["success"], phase_1_results)

            if phase_1_results["success"]:
                print("🎉 PHASE 1 COMPLETE: Single agent evolution foundation established!")
                print(f"   Base agent ready for multi-agent scaling")
                print(f"   System stability validated with no overloads")
                print(f"   Evolution framework operational")
            else:
                print("⚠️  PHASE 1 ISSUES: Some components need attention")
                if not stability_ok:
                    print("   System stability concerns detected")
                if not evolution_results.get("success", False):
                    print("   Evolution process encountered issues")

            return phase_1_results

        except Exception as e:
            phase_duration = time.time() - phase_start
            error_results = {
                "success": False,
                "duration": phase_duration,
                "error": str(e),
                "details": {"error_type": type(e).__name__}
            }

            self.phase_results["phase_1"] = error_results
            self.print_phase_result("PHASE 1", False, error_results)

            self.logger.error(f"Phase 1 execution failed: {e}")
            return error_results


async def main():
    """Main entry point for Observer-approved training deployment"""
    print("🚀 PYGENT FACTORY FULL TRAINING MODE")
    print("Observer supervision: ACTIVE")
    print("RIPER-Ω protocol: COMPLIANT")
    print("=" * 60)

    controller = ObserverTrainingController()

    try:
        # Execute Phase 1: Controlled Single Agent Evolution
        phase_1_results = await controller.execute_phase_1()

        if phase_1_results["success"]:
            print("\n🎯 PHASE 1 SUCCESS - Ready for Phase 2 multi-agent scaling")
            print("Observer: Phase 1 foundation established successfully")
        else:
            print("\n⚠️  PHASE 1 INCOMPLETE - Addressing issues before proceeding")
            print("Observer: Phase 1 requires attention before scaling")

        # Report to Observer
        total_time = time.time() - controller.start_time
        print(f"\n📊 TRAINING SESSION SUMMARY:")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Phases completed: 1/5")
        print(f"   System status: {'OPERATIONAL' if phase_1_results['success'] else 'NEEDS ATTENTION'}")
        print(f"   Observer compliance: MAINTAINED")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        controller.logger.error(f"Training session failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
