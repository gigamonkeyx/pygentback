#!/usr/bin/env python3
"""
World Simulation Monitoring API Endpoints

Provides real-time monitoring endpoints for world simulation system:
- /sim/status: Current simulation status and metrics
- /sim/metrics: Performance metrics and statistics
- /sim/agents: Agent population state and performance
- /sim/behaviors: Emergent behavior detection results
- /sim/evolution: Evolution system status and history

RIPER-Ω Protocol compliant with observer supervision.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta

from core.sim_env import (
    SimulationEnvironment,
    AgentPopulationManager,
    WorldSimulationEvolution,
    AgentInteractionSystem,
    EmergentBehaviorMonitor,
    RIPEROmegaIntegration
)

logger = logging.getLogger(__name__)

# Global simulation system instance (will be initialized by main app)
_simulation_system: Optional[SimulationEnvironment] = None

def get_simulation_system() -> SimulationEnvironment:
    """Get the global simulation system instance"""
    global _simulation_system
    if _simulation_system is None:
        raise HTTPException(status_code=503, detail="World simulation system not initialized")
    return _simulation_system

def set_simulation_system(sim_system: SimulationEnvironment):
    """Set the global simulation system instance"""
    global _simulation_system
    _simulation_system = sim_system

# Create router for world simulation endpoints
router = APIRouter(prefix="/sim", tags=["world_simulation"])


@router.get("/status")
async def get_simulation_status(sim_env: SimulationEnvironment = Depends(get_simulation_system)) -> Dict[str, Any]:
    """
    Get current simulation status and basic metrics.
    
    Returns:
        - Environment status and uptime
        - Resource utilization
        - Agent count and distribution
        - Current simulation cycle
        - RIPER-Ω mode status
    """
    try:
        # Get environment state
        env_state = await sim_env.get_environment_state()
        
        # Get population state if available
        population_info = {}
        if hasattr(sim_env, 'population_manager'):
            pop_state = await sim_env.population_manager.get_population_state()
            population_info = {
                "population_size": pop_state.get("population_size", 0),
                "role_distribution": pop_state.get("role_distribution", {}),
                "average_performance": pop_state.get("average_performance", 0.0)
            }
        
        # Get RIPER-Ω status if available
        riperω_info = {}
        if hasattr(sim_env, 'riperω_integration'):
            riperω_status = await sim_env.riperω_integration.get_riperω_status()
            riperω_info = {
                "current_mode": riperω_status.get("current_mode", "unknown"),
                "confidence_scores": riperω_status.get("confidence_scores", {}),
                "stagnation_generations": riperω_status.get("stagnation_generations", 0)
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "id": env_state["environment_id"],
                "status": env_state["status"],
                "uptime_seconds": env_state["uptime"],
                "cycle_count": env_state["cycle_count"]
            },
            "resources": env_state["resources"],
            "population": population_info,
            "riperω_protocol": riperω_info,
            "system_health": "operational" if env_state["status"] == "active" else "degraded"
        }
        
    except Exception as e:
        logger.error(f"Failed to get simulation status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve simulation status: {str(e)}")


@router.get("/metrics")
async def get_simulation_metrics(sim_env: SimulationEnvironment = Depends(get_simulation_system)) -> Dict[str, Any]:
    """
    Get detailed performance metrics and statistics.
    
    Returns:
        - Resource utilization trends
        - Evolution system performance
        - Interaction statistics
        - Emergent behavior metrics
        - Performance benchmarks
    """
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "resource_metrics": {},
            "evolution_metrics": {},
            "interaction_metrics": {},
            "behavior_metrics": {},
            "performance_benchmarks": {}
        }
        
        # Resource metrics
        env_state = await sim_env.get_environment_state()
        total_utilization = sum(
            res["utilization"] for res in env_state["resources"].values()
        ) / len(env_state["resources"])
        
        metrics["resource_metrics"] = {
            "average_utilization": total_utilization,
            "resource_breakdown": env_state["resources"],
            "efficiency_score": 1.0 - total_utilization if total_utilization < 1.0 else 0.0
        }
        
        # Evolution metrics
        if hasattr(sim_env, 'evolution_system'):
            evolution_system = sim_env.evolution_system
            metrics["evolution_metrics"] = {
                "current_generation": evolution_system.generation,
                "fitness_history_length": len(evolution_system.fitness_history),
                "convergence_threshold": evolution_system.convergence_threshold,
                "mutation_rate": evolution_system.mutation_rate,
                "crossover_rate": evolution_system.crossover_rate
            }
            
            if evolution_system.fitness_history:
                latest_fitness = evolution_system.fitness_history[-1]
                metrics["evolution_metrics"]["latest_fitness_stats"] = latest_fitness
        
        # Interaction metrics
        if hasattr(sim_env, 'interaction_system'):
            interaction_summary = await sim_env.interaction_system.get_interaction_summary()
            metrics["interaction_metrics"] = interaction_summary
        
        # Behavior metrics
        if hasattr(sim_env, 'behavior_monitor'):
            behavior_summary = await sim_env.behavior_monitor.get_behavior_summary()
            metrics["behavior_metrics"] = behavior_summary
        
        # Performance benchmarks
        metrics["performance_benchmarks"] = {
            "cycles_per_minute": env_state["cycle_count"] / max(env_state["uptime"] / 60, 1),
            "memory_efficiency": "good",  # Placeholder - could integrate actual memory monitoring
            "response_time_ms": 50  # Placeholder - could measure actual response times
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get simulation metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve simulation metrics: {str(e)}")


@router.get("/agents")
async def get_agent_population(sim_env: SimulationEnvironment = Depends(get_simulation_system)) -> Dict[str, Any]:
    """
    Get agent population state and performance data.
    
    Returns:
        - Individual agent details
        - Performance metrics per agent
        - Role distribution
        - Agent interaction networks
    """
    try:
        if not hasattr(sim_env, 'population_manager'):
            raise HTTPException(status_code=404, detail="Population manager not available")
        
        pop_state = await sim_env.population_manager.get_population_state()
        
        # Enhance with additional agent details
        enhanced_agents = {}
        for agent_name, agent_data in pop_state["agents"].items():
            enhanced_agents[agent_name] = {
                **agent_data,
                "health_status": "healthy" if agent_data["performance"]["efficiency_score"] > 0.6 else "needs_attention",
                "specialization": agent_data.get("specialization", "general"),
                "last_activity": datetime.now().isoformat()  # Placeholder
            }
        
        # Add network analysis if available
        network_analysis = {}
        if hasattr(sim_env, 'interaction_system') and sim_env.interaction_system.interaction_graph:
            try:
                import networkx as nx
                graph = sim_env.interaction_system.interaction_graph
                network_analysis = {
                    "total_connections": graph.number_of_edges(),
                    "network_density": nx.density(graph),
                    "most_connected_agents": [
                        {"agent": node, "connections": graph.degree(node)}
                        for node in sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)[:5]
                    ]
                }
            except ImportError:
                network_analysis = {"error": "NetworkX not available"}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "population_summary": {
                "total_agents": pop_state["population_size"],
                "role_distribution": pop_state["role_distribution"],
                "average_performance": pop_state["average_performance"]
            },
            "agents": enhanced_agents,
            "network_analysis": network_analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent population: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent population: {str(e)}")


@router.get("/behaviors")
async def get_emergent_behaviors(sim_env: SimulationEnvironment = Depends(get_simulation_system)) -> Dict[str, Any]:
    """
    Get emergent behavior detection results and patterns.
    
    Returns:
        - Detected behavior patterns
        - Cooperation metrics
        - Resource optimization patterns
        - Alliance formations
        - Adaptive triggers
    """
    try:
        if not hasattr(sim_env, 'behavior_monitor'):
            raise HTTPException(status_code=404, detail="Behavior monitor not available")
        
        behavior_summary = await sim_env.behavior_monitor.get_behavior_summary()
        
        # Get recent behavior monitoring results
        recent_behaviors = {}
        if sim_env.behavior_monitor.detected_behaviors:
            latest_detection = sim_env.behavior_monitor.detected_behaviors[-1]
            recent_behaviors = latest_detection["behaviors"]
        
        # Get active adaptation triggers
        active_triggers = sim_env.behavior_monitor.adaptation_triggers[-10:] if sim_env.behavior_monitor.adaptation_triggers else []
        
        # Get feedback loop history
        recent_feedback = sim_env.behavior_monitor.feedback_history[-10:] if sim_env.behavior_monitor.feedback_history else []
        
        return {
            "timestamp": datetime.now().isoformat(),
            "behavior_summary": behavior_summary,
            "recent_detections": recent_behaviors,
            "active_triggers": active_triggers,
            "feedback_loops": recent_feedback,
            "emergence_indicators": {
                "cooperation_level": "high" if behavior_summary.get("behavior_summary", {}).get("cooperation_patterns", 0) > 2 else "low",
                "optimization_activity": "active" if behavior_summary.get("behavior_summary", {}).get("optimization_patterns", 0) > 1 else "minimal",
                "network_formation": "growing" if behavior_summary.get("behavior_summary", {}).get("sharing_networks", 0) > 0 else "stable"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get emergent behaviors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve emergent behaviors: {str(e)}")


@router.get("/evolution")
async def get_evolution_status(sim_env: SimulationEnvironment = Depends(get_simulation_system)) -> Dict[str, Any]:
    """
    Get evolution system status and history.
    
    Returns:
        - Current generation status
        - Fitness progression
        - Mutation/crossover statistics
        - Convergence analysis
        - DGM validation results
    """
    try:
        if not hasattr(sim_env, 'evolution_system'):
            raise HTTPException(status_code=404, detail="Evolution system not available")
        
        evolution_system = sim_env.evolution_system
        
        # Get fitness progression
        fitness_progression = []
        if evolution_system.fitness_history:
            fitness_progression = [
                {
                    "generation": i,
                    "mean_fitness": stats["mean_fitness"],
                    "max_fitness": stats["max_fitness"],
                    "min_fitness": stats["min_fitness"]
                }
                for i, stats in enumerate(evolution_system.fitness_history)
            ]
        
        # Calculate convergence status
        convergence_status = "unknown"
        if len(evolution_system.fitness_history) >= 3:
            recent_fitness = [gen["mean_fitness"] for gen in evolution_system.fitness_history[-3:]]
            fitness_improvement = max(recent_fitness) - min(recent_fitness)
            convergence_status = "converged" if fitness_improvement < evolution_system.convergence_threshold else "evolving"
        
        # Get DGM validation results if available
        dgm_results = {}
        if hasattr(sim_env, 'dgm_validator'):
            dgm_summary = await sim_env.dgm_validator.get_validation_summary()
            dgm_results = dgm_summary
        
        return {
            "timestamp": datetime.now().isoformat(),
            "evolution_status": {
                "current_generation": evolution_system.generation,
                "convergence_status": convergence_status,
                "stagnation_generations": getattr(evolution_system, 'stagnation_generations', 0)
            },
            "parameters": {
                "mutation_rate": evolution_system.mutation_rate,
                "crossover_rate": evolution_system.crossover_rate,
                "elite_ratio": evolution_system.elite_ratio,
                "bloat_penalty_rate": evolution_system.bloat_penalty_rate
            },
            "fitness_progression": fitness_progression,
            "dgm_validation": dgm_results
        }
        
    except Exception as e:
        logger.error(f"Failed to get evolution status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve evolution status: {str(e)}")


@router.post("/control/start")
async def start_simulation(sim_env: SimulationEnvironment = Depends(get_simulation_system)) -> Dict[str, Any]:
    """Start or resume world simulation"""
    try:
        if sim_env.status == "shutdown":
            raise HTTPException(status_code=400, detail="Cannot start shutdown simulation")
        
        # Initialize if needed
        if sim_env.status == "initializing":
            success = await sim_env.initialize()
            if not success:
                raise HTTPException(status_code=500, detail="Failed to initialize simulation")
        
        return {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "environment_id": sim_env.environment_id
        }
        
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start simulation: {str(e)}")


@router.post("/control/stop")
async def stop_simulation(sim_env: SimulationEnvironment = Depends(get_simulation_system)) -> Dict[str, Any]:
    """Gracefully stop world simulation"""
    try:
        await sim_env.shutdown()
        
        return {
            "status": "stopped",
            "timestamp": datetime.now().isoformat(),
            "environment_id": sim_env.environment_id
        }
        
    except Exception as e:
        logger.error(f"Failed to stop simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop simulation: {str(e)}")


@router.get("/health")
async def simulation_health_check() -> Dict[str, Any]:
    """Health check endpoint for simulation system"""
    try:
        global _simulation_system
        
        if _simulation_system is None:
            return {
                "status": "not_initialized",
                "timestamp": datetime.now().isoformat(),
                "healthy": False
            }
        
        env_state = await _simulation_system.get_environment_state()
        
        return {
            "status": env_state["status"],
            "timestamp": datetime.now().isoformat(),
            "healthy": env_state["status"] in ["active", "degraded"],
            "uptime": env_state["uptime"],
            "cycle_count": env_state["cycle_count"]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "healthy": False,
            "error": str(e)
        }
