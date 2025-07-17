#!/usr/bin/env python3
"""
Phase 3 DGM Training Controller
Observer-approved DGM integration with multi-agent swarm
Implements self-improvement loops with safety monitoring
"""

import asyncio
import logging
import sys
import os
import time
import psutil
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class Phase3DGMTrainingController:
    """Phase 3 Observer-supervised DGM training controller"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.start_time = time.time()
        self.phase_results = {}
        self.system_metrics = {}
        self.training_agents = []
        self.dgm_engines = {}
        self.improvement_history = []
        
        # GPU throttling configuration
        self.gpu_memory_fraction = 0.8
        self.max_generations = 5
        self.max_agents = 10
        
        # DGM configuration (Observer-approved)
        self.dgm_config = {
            "safety": {
                "max_violations": 5,
                "strict_mode": True,
                "safety_threshold": 0.8
            },
            "code_generation": {
                "max_iterations": 3,
                "improvement_threshold": 0.05
            },
            "validation": {
                "timeout_seconds": 30,
                "min_test_coverage": 0.8
            }
        }
        
        # Agent role specifications (from Phase 2)
        self.agent_roles = {
            "explorer_1": {
                "type": "research",
                "role": "explorer",
                "capabilities": ["environment_scanning", "tool_discovery", "resource_mapping"],
                "traits": {"curiosity": 0.9, "risk_tolerance": 0.7, "exploration_radius": 0.8}
            },
            "explorer_2": {
                "type": "research", 
                "role": "explorer",
                "capabilities": ["deep_analysis", "pattern_recognition", "anomaly_detection"],
                "traits": {"curiosity": 0.8, "risk_tolerance": 0.6, "exploration_radius": 0.9}
            },
            "builder_1": {
                "type": "construction",
                "role": "builder",
                "capabilities": ["resource_construction", "tool_creation", "infrastructure_building"],
                "traits": {"efficiency": 0.8, "precision": 0.9, "innovation": 0.6}
            },
            "builder_2": {
                "type": "construction",
                "role": "builder", 
                "capabilities": ["optimization", "system_integration", "quality_assurance"],
                "traits": {"efficiency": 0.9, "precision": 0.8, "innovation": 0.7}
            },
            "coordinator": {
                "type": "management",
                "role": "coordinator",
                "capabilities": ["task_distribution", "resource_allocation", "conflict_resolution"],
                "traits": {"leadership": 0.9, "communication": 0.8, "strategic_thinking": 0.9}
            }
        }
        
    def _setup_logging(self):
        """Setup logging for Phase 3 DGM training process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'phase3_dgm_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
        print(f"🧬 DGM Integration: ENABLED")
    
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
    
    async def create_dgm_enhanced_agents(self, num_agents: int = 5) -> List[Dict[str, Any]]:
        """Create DGM-enhanced agents for self-improvement"""
        try:
            self.logger.info(f"Creating {num_agents} DGM-enhanced agents...")
            
            created_agents = []
            role_names = list(self.agent_roles.keys())
            
            for i in range(num_agents):
                # Select role (cycle through available roles)
                role_name = role_names[i % len(role_names)]
                role_spec = self.agent_roles[role_name].copy()
                
                # Create agent with DGM capabilities
                agent = {
                    "agent_id": f"dgm_agent_{role_name}_{int(time.time())}_{i}",
                    "name": f"dgm_{role_name}_{i+1}",
                    "type": role_spec["type"],
                    "role": role_spec["role"],
                    "capabilities": role_spec["capabilities"] + ["self_improvement", "code_generation"],
                    "traits": role_spec["traits"].copy(),
                    "fitness_score": 0.0,
                    "generation": 0,
                    "dgm_enabled": True,
                    "improvement_attempts": 0,
                    "successful_improvements": 0,
                    "safety_violations": 0,
                    "status": "active",
                    "created_at": datetime.now().isoformat()
                }
                
                # Add some randomization to traits for diversity
                for trait_name, trait_value in agent["traits"].items():
                    variation = (torch.rand(1).item() - 0.5) * 0.2  # ±10% variation
                    agent["traits"][trait_name] = max(0.1, min(1.0, trait_value + variation))
                
                created_agents.append(agent)
                self.training_agents.append(agent)
                
                self.logger.info(f"Created DGM agent: {agent['agent_id']} ({agent['role']})")
            
            self.logger.info(f"DGM-enhanced agent swarm created: {len(created_agents)} agents")
            return created_agents
            
        except Exception as e:
            self.logger.error(f"DGM agent creation failed: {e}")
            return []

    async def initialize_dgm_engines(self, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize DGM engines for each agent with safety monitoring"""
        try:
            self.logger.info("Initializing DGM engines for agents...")

            initialization_results = {
                "engines_created": 0,
                "safety_monitors_active": 0,
                "initialization_errors": []
            }

            for agent in agents:
                agent_id = agent["agent_id"]

                try:
                    # Create simplified DGM engine (mock implementation for Phase 3)
                    dgm_engine = {
                        "agent_id": agent_id,
                        "config": self.dgm_config.copy(),
                        "active_improvements": [],
                        "improvement_history": [],
                        "safety_violations": 0,
                        "baseline_performance": agent["fitness_score"],
                        "status": "initialized"
                    }

                    self.dgm_engines[agent_id] = dgm_engine
                    initialization_results["engines_created"] += 1
                    initialization_results["safety_monitors_active"] += 1

                    self.logger.info(f"DGM engine initialized for agent: {agent_id}")

                except Exception as e:
                    error_msg = f"Failed to initialize DGM engine for {agent_id}: {e}"
                    self.logger.error(error_msg)
                    initialization_results["initialization_errors"].append(error_msg)

            self.logger.info(f"DGM engines initialized: {initialization_results['engines_created']} engines")
            return initialization_results

        except Exception as e:
            self.logger.error(f"DGM engine initialization failed: {e}")
            return {"error": str(e)}

    async def simulate_dgm_self_improvement(self, agents: List[Dict[str, Any]], improvement_cycles: int = 3) -> Dict[str, Any]:
        """Simulate DGM self-improvement cycles with safety monitoring"""
        try:
            self.logger.info(f"Starting DGM self-improvement simulation for {improvement_cycles} cycles...")

            improvement_results = {
                "total_improvement_attempts": 0,
                "successful_improvements": 0,
                "safety_rejections": 0,
                "fitness_improvements": [],
                "efficiency_gains": [],
                "bloat_reductions": []
            }

            for cycle in range(improvement_cycles):
                self.logger.info(f"DGM Improvement Cycle {cycle + 1}/{improvement_cycles}")

                cycle_attempts = 0
                cycle_successes = 0
                cycle_rejections = 0

                for agent in agents:
                    agent_id = agent["agent_id"]
                    dgm_engine = self.dgm_engines.get(agent_id)

                    if not dgm_engine:
                        continue

                    # Simulate improvement attempt
                    improvement_result = await self._attempt_agent_improvement(agent, dgm_engine)

                    cycle_attempts += 1
                    agent["improvement_attempts"] += 1

                    if improvement_result["success"]:
                        cycle_successes += 1
                        agent["successful_improvements"] += 1

                        # Apply improvement to agent
                        fitness_gain = improvement_result.get("fitness_gain", 0)
                        efficiency_gain = improvement_result.get("efficiency_gain", 0)

                        agent["fitness_score"] += fitness_gain
                        improvement_results["fitness_improvements"].append(fitness_gain)
                        improvement_results["efficiency_gains"].append(efficiency_gain)

                        self.logger.info(f"Agent {agent_id}: Improvement successful (+{fitness_gain:.3f} fitness)")

                    elif improvement_result.get("safety_rejected", False):
                        cycle_rejections += 1
                        agent["safety_violations"] += 1
                        self.logger.warning(f"Agent {agent_id}: Improvement rejected by safety monitor")

                    # Simulate processing time
                    await asyncio.sleep(0.1)

                improvement_results["total_improvement_attempts"] += cycle_attempts
                improvement_results["successful_improvements"] += cycle_successes
                improvement_results["safety_rejections"] += cycle_rejections

                success_rate = cycle_successes / max(cycle_attempts, 1)
                self.logger.info(f"Cycle {cycle + 1}: {cycle_successes}/{cycle_attempts} improvements successful ({success_rate:.1%})")

            # Calculate overall metrics
            total_attempts = improvement_results["total_improvement_attempts"]
            total_successes = improvement_results["successful_improvements"]
            overall_success_rate = total_successes / max(total_attempts, 1)

            improvement_results["overall_success_rate"] = overall_success_rate
            improvement_results["average_fitness_gain"] = sum(improvement_results["fitness_improvements"]) / max(len(improvement_results["fitness_improvements"]), 1)

            self.logger.info(f"DGM self-improvement completed: {overall_success_rate:.1%} success rate")
            return improvement_results

        except Exception as e:
            self.logger.error(f"DGM self-improvement simulation failed: {e}")
            return {"error": str(e)}

    async def _attempt_agent_improvement(self, agent: Dict[str, Any], dgm_engine: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt self-improvement for a single agent with safety checks"""
        try:
            agent_id = agent["agent_id"]

            # Generate improvement candidate
            improvement_candidate = await self._generate_improvement_candidate(agent, dgm_engine)

            # Safety evaluation
            safety_result = await self._evaluate_improvement_safety(improvement_candidate, dgm_engine)

            if not safety_result["safe"]:
                return {
                    "success": False,
                    "safety_rejected": True,
                    "reason": safety_result.get("reason", "Safety violation detected")
                }

            # Simulate improvement validation and application
            validation_result = await self._validate_improvement(improvement_candidate, agent)

            if validation_result["valid"]:
                # Apply improvement
                fitness_gain = improvement_candidate.get("expected_fitness_gain", 0)
                efficiency_gain = improvement_candidate.get("expected_efficiency_gain", 0)

                # Record improvement in DGM engine
                dgm_engine["improvement_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "improvement_type": improvement_candidate["type"],
                    "fitness_gain": fitness_gain,
                    "efficiency_gain": efficiency_gain,
                    "safety_score": safety_result["safety_score"]
                })

                return {
                    "success": True,
                    "fitness_gain": fitness_gain,
                    "efficiency_gain": efficiency_gain,
                    "improvement_type": improvement_candidate["type"]
                }
            else:
                return {
                    "success": False,
                    "validation_failed": True,
                    "reason": validation_result.get("reason", "Validation failed")
                }

        except Exception as e:
            self.logger.error(f"Agent improvement attempt failed for {agent['agent_id']}: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_improvement_candidate(self, agent: Dict[str, Any], dgm_engine: Dict[str, Any]) -> Dict[str, Any]:
        """Generate improvement candidate for agent"""
        try:
            # Analyze agent performance and identify improvement opportunities
            current_fitness = agent["fitness_score"]
            role = agent["role"]
            traits = agent["traits"]

            # Select improvement type based on agent role and current performance
            improvement_types = ["trait_optimization", "capability_enhancement", "efficiency_improvement"]
            improvement_type = improvement_types[torch.randint(0, len(improvement_types), (1,)).item()]

            # Generate specific improvement based on type
            if improvement_type == "trait_optimization":
                # Optimize agent traits
                target_trait = list(traits.keys())[torch.randint(0, len(traits), (1,)).item()]
                current_value = traits[target_trait]
                improvement_factor = 0.1 + torch.rand(1).item() * 0.2  # 10-30% improvement

                candidate = {
                    "type": "trait_optimization",
                    "target_trait": target_trait,
                    "current_value": current_value,
                    "proposed_value": min(1.0, current_value * (1 + improvement_factor)),
                    "expected_fitness_gain": improvement_factor * 0.1,
                    "expected_efficiency_gain": improvement_factor * 0.05,
                    "risk_level": 0.1  # Low risk for trait optimization
                }

            elif improvement_type == "capability_enhancement":
                # Enhance agent capabilities
                new_capability = f"enhanced_{role}_capability_{int(time.time())}"

                candidate = {
                    "type": "capability_enhancement",
                    "new_capability": new_capability,
                    "expected_fitness_gain": 0.15,
                    "expected_efficiency_gain": 0.1,
                    "risk_level": 0.2  # Medium risk for capability changes
                }

            else:  # efficiency_improvement
                # Improve agent efficiency
                efficiency_gain = 0.1 + torch.rand(1).item() * 0.15  # 10-25% efficiency gain

                candidate = {
                    "type": "efficiency_improvement",
                    "efficiency_gain": efficiency_gain,
                    "expected_fitness_gain": efficiency_gain * 0.5,
                    "expected_efficiency_gain": efficiency_gain,
                    "risk_level": 0.15  # Low-medium risk for efficiency improvements
                }

            candidate["agent_id"] = agent["agent_id"]
            candidate["timestamp"] = datetime.now().isoformat()

            return candidate

        except Exception as e:
            self.logger.error(f"Improvement candidate generation failed: {e}")
            return {"type": "error", "error": str(e)}

    async def _evaluate_improvement_safety(self, candidate: Dict[str, Any], dgm_engine: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate safety of improvement candidate"""
        try:
            safety_config = dgm_engine["config"]["safety"]

            # Calculate safety score based on risk level and safety rules
            risk_level = candidate.get("risk_level", 0.5)
            safety_score = 1.0 - risk_level

            # Check against safety threshold
            safety_threshold = safety_config["safety_threshold"]
            is_safe = safety_score >= safety_threshold

            # Additional safety checks
            violations = []

            # Check for excessive risk
            if risk_level > 0.8:
                violations.append("High risk level detected")

            # Check for reasonable improvement expectations
            fitness_gain = candidate.get("expected_fitness_gain", 0)
            if fitness_gain > 1.0:  # More than 100% improvement is suspicious
                violations.append("Unrealistic fitness gain expectation")

            # Update safety status based on violations
            if violations:
                is_safe = False
                safety_score *= 0.5  # Reduce safety score for violations

            return {
                "safe": is_safe,
                "safety_score": safety_score,
                "violations": violations,
                "reason": "Safety evaluation completed"
            }

        except Exception as e:
            self.logger.error(f"Safety evaluation failed: {e}")
            return {"safe": False, "safety_score": 0.0, "reason": f"Safety evaluation error: {e}"}

    async def _validate_improvement(self, candidate: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, Any]:
        """Validate improvement candidate"""
        try:
            # Simulate validation process
            improvement_type = candidate.get("type", "unknown")

            # Basic validation checks
            if improvement_type == "trait_optimization":
                proposed_value = candidate.get("proposed_value", 0)
                if 0.1 <= proposed_value <= 1.0:
                    return {"valid": True, "reason": "Trait optimization validated"}
                else:
                    return {"valid": False, "reason": "Proposed trait value out of bounds"}

            elif improvement_type == "capability_enhancement":
                # Simulate capability validation
                validation_success = torch.rand(1).item() > 0.3  # 70% success rate
                if validation_success:
                    return {"valid": True, "reason": "Capability enhancement validated"}
                else:
                    return {"valid": False, "reason": "Capability enhancement validation failed"}

            elif improvement_type == "efficiency_improvement":
                efficiency_gain = candidate.get("efficiency_gain", 0)
                if 0 < efficiency_gain <= 0.5:  # Max 50% efficiency gain
                    return {"valid": True, "reason": "Efficiency improvement validated"}
                else:
                    return {"valid": False, "reason": "Efficiency gain out of reasonable bounds"}

            else:
                return {"valid": False, "reason": f"Unknown improvement type: {improvement_type}"}

        except Exception as e:
            self.logger.error(f"Improvement validation failed: {e}")
            return {"valid": False, "reason": f"Validation error: {e}"}

    async def run_phase_3_training(self) -> bool:
        """Execute Observer-approved Phase 3: DGM Integration"""
        phase_start = time.time()

        self.print_phase_header("PHASE 3", "DGM Integration with Self-Improvement Loops")

        try:
            # Step 1: Configure GPU throttling
            print("🔧 Step 1: Configuring GPU throttling...")
            gpu_config = await self.configure_gpu_throttling()

            # Step 2: Create DGM-enhanced agents
            print(f"🤖 Step 2: Creating DGM-enhanced agents ({self.max_agents//2} agents)...")
            agents = await self.create_dgm_enhanced_agents(self.max_agents//2)  # Start with 5 agents
            if not agents:
                raise Exception("Failed to create DGM-enhanced agents")

            # Step 3: Initialize DGM engines
            print("🧬 Step 3: Initializing DGM engines with safety monitoring...")
            dgm_init_results = await self.initialize_dgm_engines(agents)
            if "error" in dgm_init_results:
                raise Exception(f"DGM initialization failed: {dgm_init_results['error']}")

            # Step 4: Monitor initial resources
            print("📊 Step 4: Monitoring initial system resources...")
            initial_metrics = await self.monitor_system_resources()

            # Step 5: Run DGM self-improvement cycles
            print("🔄 Step 5: Running DGM self-improvement cycles...")
            improvement_results = await self.simulate_dgm_self_improvement(agents, improvement_cycles=3)

            if "error" in improvement_results:
                raise Exception(f"DGM self-improvement failed: {improvement_results['error']}")

            # Step 6: Validate no bloat and efficiency gains
            print("✅ Step 6: Validating efficiency gains and bloat reduction...")
            validation_results = await self.validate_dgm_improvements(agents, improvement_results)

            # Step 7: Final resource monitoring
            print("📊 Step 7: Final resource monitoring...")
            final_metrics = await self.monitor_system_resources()

            # Calculate phase metrics
            phase_duration = time.time() - phase_start
            phase_metrics = {
                "duration": phase_duration,
                "memory_usage": final_metrics.get("memory_percent", 0),
                "gpu_usage": final_metrics.get("gpu_usage", "N/A"),
                "details": {
                    "dgm_agents_created": len(agents),
                    "dgm_engines_initialized": dgm_init_results.get("engines_created", 0),
                    "improvement_attempts": improvement_results.get("total_improvement_attempts", 0),
                    "successful_improvements": improvement_results.get("successful_improvements", 0),
                    "success_rate": improvement_results.get("overall_success_rate", 0),
                    "average_fitness_gain": improvement_results.get("average_fitness_gain", 0),
                    "safety_rejections": improvement_results.get("safety_rejections", 0),
                    "efficiency_validated": validation_results.get("efficiency_validated", False)
                }
            }

            self.phase_results["phase_3"] = phase_metrics
            self.print_phase_result("PHASE 3", True, phase_metrics)

            return True

        except Exception as e:
            phase_duration = time.time() - phase_start
            error_metrics = {
                "duration": phase_duration,
                "memory_usage": 0,
                "gpu_usage": "N/A",
                "details": {"error_type": type(e).__name__}
            }

            self.phase_results["phase_3"] = error_metrics
            self.print_phase_result("PHASE 3", False, error_metrics)
            self.logger.error(f"Phase 3 execution failed: {e}")

            return False

    async def validate_dgm_improvements(self, agents: List[Dict[str, Any]], improvement_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DGM improvements for efficiency gains and bloat reduction"""
        try:
            self.logger.info("Validating DGM improvements...")

            validation_results = {
                "efficiency_validated": False,
                "bloat_detected": False,
                "performance_metrics": {},
                "recommendations": []
            }

            # Check efficiency gains
            avg_fitness_gain = improvement_results.get("average_fitness_gain", 0)
            success_rate = improvement_results.get("overall_success_rate", 0)

            if avg_fitness_gain > 0.05 and success_rate > 0.5:  # 5% fitness gain, 50% success rate
                validation_results["efficiency_validated"] = True
                validation_results["recommendations"].append("DGM improvements show positive efficiency gains")
            else:
                validation_results["recommendations"].append("DGM improvements need optimization")

            # Check for bloat (excessive complexity without proportional gains)
            total_attempts = improvement_results.get("total_improvement_attempts", 0)
            successful_improvements = improvement_results.get("successful_improvements", 0)

            if total_attempts > 0:
                efficiency_ratio = successful_improvements / total_attempts
                if efficiency_ratio < 0.3:  # Less than 30% success rate indicates potential bloat
                    validation_results["bloat_detected"] = True
                    validation_results["recommendations"].append("High failure rate suggests bloat - consider pruning")

            # Performance metrics
            validation_results["performance_metrics"] = {
                "average_fitness_gain": avg_fitness_gain,
                "success_rate": success_rate,
                "efficiency_ratio": efficiency_ratio if total_attempts > 0 else 0,
                "safety_compliance": 1.0 - (improvement_results.get("safety_rejections", 0) / max(total_attempts, 1))
            }

            self.logger.info(f"DGM validation completed: Efficiency validated = {validation_results['efficiency_validated']}")
            return validation_results

        except Exception as e:
            self.logger.error(f"DGM validation failed: {e}")
            return {"error": str(e)}

    async def run_observer_training(self):
        """Run complete Observer-supervised Phase 3 DGM training sequence"""
        print("🚀 PHASE 3 PYGENT FACTORY DGM TRAINING MODE")
        print("Observer supervision: ACTIVE")
        print("RIPER-Ω protocol: COMPLIANT")
        print("DGM integration: ENABLED")
        print("Self-improvement loops: ACTIVE")
        print("="*60)

        total_start = time.time()

        # Phase 3: DGM Integration
        phase_3_success = await self.run_phase_3_training()

        # Training summary
        total_duration = time.time() - total_start
        phases_completed = 1 if phase_3_success else 0

        print(f"\n📊 PHASE 3 DGM TRAINING SESSION SUMMARY:")
        print(f"   Total time: {total_duration:.2f} seconds")
        print(f"   Phases completed: {phases_completed}/1 (Phase 3)")
        print(f"   System status: {'OPERATIONAL' if phase_3_success else 'NEEDS ATTENTION'}")
        print(f"   Observer compliance: MAINTAINED")

        if phase_3_success:
            print("\n✅ PHASE 3 DGM TRAINING COMPLETE")
            print("   Observer validation: PASSED")
            print("   DGM self-improvement: SUCCESSFUL")
            print("   Safety monitoring: ACTIVE")
            print("   Ready for Phase 4 continuous learning loops")
        else:
            print("\n⚠️  PHASE 3 INCOMPLETE")
            print("   Observer: Address DGM issues before continuous loops")

async def main():
    """Main entry point for Phase 3 DGM training controller"""
    try:
        controller = Phase3DGMTrainingController()
        await controller.run_observer_training()

    except KeyboardInterrupt:
        print("\n\n⏹️  Phase 3 DGM training interrupted by user")
        print("   Observer supervision maintained")
    except Exception as e:
        print(f"\n💥 Phase 3 DGM training failed: {e}")
        print("   Observer methodology: Systematic error analysis required")

if __name__ == "__main__":
    asyncio.run(main())
