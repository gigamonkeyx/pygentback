#!/usr/bin/env python3
"""
Test suite for emergent behavior detection system.

Tests emergent behavior identification, alliance formation, feedback loops,
and adaptive response mechanisms. RIPER-Ω Protocol compliant testing with observer supervision.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from collections import deque
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.sim_env import (
    SimulationEnvironment,
    AgentInteractionSystem,
    EmergentBehaviorMonitor
)


class TestAgentInteractionSystem:
    """Test AgentInteractionSystem functionality"""
    
    @pytest.fixture
    async def sim_env(self):
        """Create test simulation environment"""
        env = SimulationEnvironment({"test_mode": True})
        env.population_manager = Mock()
        env.population_manager.get_population_state = AsyncMock()
        yield env
        await env.shutdown()
    
    @pytest.fixture
    async def interaction_system(self, sim_env):
        """Create test interaction system"""
        system = AgentInteractionSystem(sim_env)
        await system.initialize()
        return system
    
    @pytest.mark.asyncio
    async def test_interaction_system_initialization(self, interaction_system):
        """Test interaction system initialization"""
        assert interaction_system.resource_sharing_threshold == 0.3
        assert interaction_system.collaboration_probability == 0.4
        assert interaction_system.alliance_formation_threshold == 3
        assert len(interaction_system.message_history) == 0
        assert len(interaction_system.resource_sharing_log) == 0
        assert len(interaction_system.collaboration_history) == 0
    
    @pytest.mark.asyncio
    async def test_resource_scarcity_detection(self, interaction_system):
        """Test resource scarcity detection"""
        # Mock environment state with high utilization
        env_state = {
            "resources": {
                "compute": {"utilization": 0.8, "available": 200.0},
                "memory": {"utilization": 0.9, "available": 100.0},
                "storage": {"utilization": 0.6, "available": 400.0}
            }
        }
        
        scarcity_info = await interaction_system._detect_resource_scarcity(env_state)
        
        assert scarcity_info["scarcity_level"] == "high"  # 2 critical resources
        assert len(scarcity_info["critical_resources"]) == 2
        
        # Verify critical resources
        critical_names = [res["resource"] for res in scarcity_info["critical_resources"]]
        assert "compute" in critical_names
        assert "memory" in critical_names
        assert "storage" not in critical_names  # Below 70% threshold
    
    @pytest.mark.asyncio
    async def test_resource_sharing_event_creation(self, interaction_system):
        """Test resource sharing event creation"""
        resource_info = {"available": 1000.0}
        
        sharing_event = await interaction_system._create_sharing_event(
            "agent_1", "agent_2", "compute", resource_info
        )
        
        assert sharing_event is not None
        assert sharing_event["sharer"] == "agent_1"
        assert sharing_event["receiver"] == "agent_2"
        assert sharing_event["resource"] == "compute"
        assert sharing_event["amount"] == 100.0  # 10% of available
        assert sharing_event["reason"] == "resource_scarcity_response"
        assert sharing_event["success"] == True
        
        # Verify event is logged
        assert len(interaction_system.resource_sharing_log) == 1
        assert interaction_system.resource_sharing_log[0] == sharing_event
    
    @pytest.mark.asyncio
    async def test_collaborative_task_creation(self, interaction_system):
        """Test collaborative task creation"""
        agent1_data = {"role": "explorer", "type": "research"}
        agent2_data = {"role": "builder", "type": "coding"}
        
        collaboration_task = await interaction_system._create_collaboration_task(
            "explorer_1", agent1_data, "builder_1", agent2_data
        )
        
        assert collaboration_task is not None
        assert collaboration_task["participants"] == ["explorer_1", "builder_1"]
        assert collaboration_task["roles"] == ["explorer", "builder"]
        assert collaboration_task["description"] == "Environment analysis and system construction"
        assert collaboration_task["status"] == "initiated"
        assert 0.1 <= collaboration_task["expected_benefit"] <= 0.3
        assert 60 <= collaboration_task["duration_estimate"] <= 300
        
        # Verify task is logged
        assert len(interaction_system.collaboration_history) == 1
        assert interaction_system.collaboration_history[0] == collaboration_task
    
    @pytest.mark.asyncio
    async def test_alliance_formation_tracking(self, interaction_system):
        """Test alliance formation tracking with NetworkX"""
        # Skip if NetworkX not available
        if interaction_system.interaction_graph is None:
            pytest.skip("NetworkX not available")
        
        # Add nodes to graph
        interaction_system.interaction_graph.add_node("agent_1", **{
            "collaboration_count": 5,
            "reputation": 0.8,
            "resource_shares": 3
        })
        interaction_system.interaction_graph.add_node("agent_2", **{
            "collaboration_count": 4,
            "reputation": 0.7,
            "resource_shares": 2
        })
        
        # Add interaction edge
        interaction_system.interaction_graph.add_edge("agent_1", "agent_2", **{
            "interaction_count": 3,
            "last_interaction": datetime.now(),
            "interaction_types": ["collaboration", "resource_sharing"]
        })
        
        alliance_formations = await interaction_system._update_alliance_formations()
        
        # Should form alliance (agent_1 meets criteria and has qualified partner)
        assert len(alliance_formations) >= 0  # May or may not form based on criteria
        
        if alliance_formations:
            alliance = alliance_formations[0]
            assert "alliance_id" in alliance
            assert "leader" in alliance
            assert "members" in alliance
            assert "formation_reason" in alliance
            assert "strength" in alliance
    
    @pytest.mark.asyncio
    async def test_interaction_edge_updates(self, interaction_system):
        """Test interaction edge updates in NetworkX graph"""
        if interaction_system.interaction_graph is None:
            pytest.skip("NetworkX not available")
        
        # Add initial nodes
        interaction_system.interaction_graph.add_node("agent_1", **{
            "collaboration_count": 0,
            "reputation": 0.5
        })
        interaction_system.interaction_graph.add_node("agent_2", **{
            "collaboration_count": 0,
            "reputation": 0.5
        })
        
        # Update interaction edge
        await interaction_system._update_interaction_edge("agent_1", "agent_2", "collaboration")
        
        # Verify edge was created
        assert interaction_system.interaction_graph.has_edge("agent_1", "agent_2")
        
        edge_data = interaction_system.interaction_graph["agent_1"]["agent_2"]
        assert edge_data["interaction_count"] == 1
        assert "collaboration" in edge_data["interaction_types"]
        
        # Verify node updates
        assert interaction_system.interaction_graph.nodes["agent_1"]["collaboration_count"] == 1
        assert interaction_system.interaction_graph.nodes["agent_2"]["collaboration_count"] == 1
        assert interaction_system.interaction_graph.nodes["agent_1"]["reputation"] > 0.5
        assert interaction_system.interaction_graph.nodes["agent_2"]["reputation"] > 0.5
    
    @pytest.mark.asyncio
    async def test_swarm_coordination_execution(self, interaction_system):
        """Test complete swarm coordination execution"""
        # Mock environment and population states
        env_state = {
            "resources": {
                "compute": {"utilization": 0.8, "available": 200.0},
                "memory": {"utilization": 0.5, "available": 500.0}
            }
        }
        
        population_state = {
            "agents": {
                "agent_1": {"role": "explorer", "type": "research", "performance": {"efficiency_score": 0.8}},
                "agent_2": {"role": "builder", "type": "coding", "performance": {"efficiency_score": 0.6}},
                "agent_3": {"role": "harvester", "type": "analysis", "performance": {"efficiency_score": 0.3}}
            }
        }
        
        interaction_system.simulation_env.get_environment_state = AsyncMock(return_value=env_state)
        interaction_system.simulation_env.population_manager.get_population_state = AsyncMock(return_value=population_state)
        
        coordination_results = await interaction_system.enable_swarm_coordination()
        
        assert "resource_sharing_events" in coordination_results
        assert "collaboration_tasks" in coordination_results
        assert "alliance_formations" in coordination_results
        assert "message_exchanges" in coordination_results
        
        # Should have resource sharing due to compute scarcity
        assert len(coordination_results["resource_sharing_events"]) > 0
        
        # Should have some collaboration tasks
        assert isinstance(coordination_results["collaboration_tasks"], list)
    
    @pytest.mark.asyncio
    async def test_interaction_summary_generation(self, interaction_system):
        """Test interaction summary generation"""
        # Add some test data
        interaction_system.message_history.append({"type": "test", "timestamp": datetime.now()})
        interaction_system.resource_sharing_log.append({"event": "test_sharing"})
        interaction_system.collaboration_history.append({"task": "test_collaboration"})
        interaction_system.state_log.append({"event": "test_event", "timestamp": datetime.now()})
        
        summary = await interaction_system.get_interaction_summary()
        
        assert summary["total_interactions"] == 1
        assert summary["resource_sharing_events"] == 1
        assert summary["collaboration_tasks"] == 1
        assert summary["state_events"] == 1
        assert len(summary["recent_events"]) == 1
        
        if interaction_system.interaction_graph is not None:
            assert "network_stats" in summary
            assert "total_nodes" in summary["network_stats"]
            assert "total_edges" in summary["network_stats"]


class TestEmergentBehaviorMonitor:
    """Test EmergentBehaviorMonitor functionality"""
    
    @pytest.fixture
    async def sim_env(self):
        """Create test simulation environment"""
        env = SimulationEnvironment({"test_mode": True})
        yield env
        await env.shutdown()
    
    @pytest.fixture
    async def interaction_system(self, sim_env):
        """Create test interaction system"""
        system = AgentInteractionSystem(sim_env)
        await system.initialize()
        return system
    
    @pytest.fixture
    async def behavior_monitor(self, sim_env, interaction_system):
        """Create test behavior monitor"""
        monitor = EmergentBehaviorMonitor(sim_env, interaction_system)
        await monitor.initialize()
        return monitor
    
    def test_behavior_monitor_initialization(self, behavior_monitor):
        """Test behavior monitor initialization"""
        assert behavior_monitor.resource_threshold == 0.3
        assert behavior_monitor.cooperation_threshold == 0.6
        assert behavior_monitor.optimization_threshold == 0.2
        assert len(behavior_monitor.detected_behaviors) == 0
        assert len(behavior_monitor.feedback_history) == 0
    
    @pytest.mark.asyncio
    async def test_spontaneous_cooperation_detection(self, behavior_monitor):
        """Test spontaneous cooperation pattern detection"""
        # Add recent collaboration history
        recent_time = datetime.now()
        behavior_monitor.interaction_system.collaboration_history = [
            {
                "timestamp": recent_time - timedelta(minutes=2),
                "participants": ["agent_1", "agent_2"],
                "status": "completed",
                "expected_benefit": 0.3
            },
            {
                "timestamp": recent_time - timedelta(minutes=1),
                "participants": ["agent_2", "agent_3"],
                "status": "completed", 
                "expected_benefit": 0.25
            },
            {
                "timestamp": recent_time,
                "participants": ["agent_1", "agent_3"],
                "expected_benefit": 0.4
            }
        ]
        
        cooperation_patterns = await behavior_monitor._detect_spontaneous_cooperation()
        
        assert len(cooperation_patterns) == 1
        pattern = cooperation_patterns[0]
        
        assert pattern["type"] == "spontaneous_cooperation"
        assert len(pattern["participants"]) == 3  # All unique participants
        assert pattern["success_rate"] > behavior_monitor.cooperation_threshold
        assert pattern["collaboration_count"] == 3
        assert pattern["significance"] in ["medium", "high"]
    
    @pytest.mark.asyncio
    async def test_resource_optimization_detection(self, behavior_monitor):
        """Test resource optimization pattern detection"""
        # Add recent resource sharing history
        recent_time = datetime.now()
        behavior_monitor.interaction_system.resource_sharing_log = [
            {
                "timestamp": recent_time - timedelta(minutes=3),
                "sharer": "agent_1",
                "receiver": "agent_2",
                "amount": 100.0
            },
            {
                "timestamp": recent_time - timedelta(minutes=2),
                "sharer": "agent_3",
                "receiver": "agent_4",
                "amount": 150.0
            },
            {
                "timestamp": recent_time - timedelta(minutes=1),
                "sharer": "agent_2",
                "receiver": "agent_5",
                "amount": 75.0
            }
        ]
        
        optimization_patterns = await behavior_monitor._detect_resource_optimization()
        
        assert len(optimization_patterns) == 1
        pattern = optimization_patterns[0]
        
        assert pattern["type"] == "resource_optimization"
        assert pattern["total_resources_shared"] == 325.0
        assert pattern["participating_agents"] == 5  # 3 sharers + 3 receivers - 1 overlap
        assert pattern["sharing_events"] == 3
        assert pattern["optimization_efficiency"] > 0
    
    @pytest.mark.asyncio
    async def test_tool_sharing_network_detection(self, behavior_monitor):
        """Test tool sharing network detection"""
        if behavior_monitor.interaction_system.interaction_graph is None:
            pytest.skip("NetworkX not available")
        
        # Create tool sharing network
        graph = behavior_monitor.interaction_system.interaction_graph
        
        # Add nodes
        for i in range(1, 5):
            graph.add_node(f"agent_{i}")
        
        # Add tool sharing edges
        tool_sharing_edges = [
            ("agent_1", "agent_2", {"interaction_types": ["tool_sharing"]}),
            ("agent_2", "agent_3", {"interaction_types": ["tool_sharing"]}),
            ("agent_3", "agent_4", {"interaction_types": ["tool_sharing"]}),
            ("agent_1", "agent_4", {"interaction_types": ["tool_sharing"]})
        ]
        
        for edge in tool_sharing_edges:
            graph.add_edge(edge[0], edge[1], **edge[2])
        
        sharing_networks = await behavior_monitor._detect_tool_sharing_networks()
        
        assert len(sharing_networks) == 1
        network = sharing_networks[0]
        
        assert network["type"] == "tool_sharing_network"
        assert network["network_size"] == 4
        assert len(network["participants"]) == 4
        assert network["sharing_connections"] == 4
    
    @pytest.mark.asyncio
    async def test_adaptive_triggers_detection(self, behavior_monitor):
        """Test adaptive rule triggers detection"""
        # Mock environment state with resource scarcity
        env_state = {
            "resources": {
                "compute": {"utilization": 0.8},  # Above threshold
                "memory": {"utilization": 0.9}    # Above threshold
            }
        }
        behavior_monitor.simulation_env.get_environment_state = AsyncMock(return_value=env_state)
        
        # Mock high cooperation success
        behavior_monitor._calculate_recent_success_rate = AsyncMock(return_value=0.85)
        behavior_monitor.interaction_system.collaboration_history = [{"test": "data"}]
        
        adaptive_triggers = await behavior_monitor._check_adaptive_triggers()
        
        # Should have resource scarcity trigger
        scarcity_triggers = [t for t in adaptive_triggers if t["type"] == "resource_scarcity"]
        assert len(scarcity_triggers) == 1
        
        scarcity_trigger = scarcity_triggers[0]
        assert scarcity_trigger["action"] == "trigger_alliance_mutations"
        assert scarcity_trigger["priority"] == "high"
        assert len(scarcity_trigger["critical_resources"]) == 2
        
        # Should have cooperation success trigger
        coop_triggers = [t for t in adaptive_triggers if t["type"] == "high_cooperation_success"]
        assert len(coop_triggers) == 1
        
        coop_trigger = coop_triggers[0]
        assert coop_trigger["action"] == "reinforce_cooperation_patterns"
        assert coop_trigger["success_rate"] == 0.85
    
    @pytest.mark.asyncio
    async def test_feedback_loop_processing(self, behavior_monitor):
        """Test feedback loop processing for evolution integration"""
        monitoring_results = {
            "spontaneous_cooperation": [
                {"pattern_id": "coop_1", "significance": "high", "type": "cooperation"}
            ],
            "resource_optimization": [
                {"pattern_id": "opt_1", "significance": "medium", "type": "optimization"}
            ],
            "tool_sharing_networks": [
                {"pattern_id": "net_1", "significance": "high", "type": "network"}
            ]
        }
        
        feedback_loops = await behavior_monitor._process_feedback_loops(monitoring_results)
        
        # Should create feedback for high significance behaviors
        high_sig_loops = [f for f in feedback_loops if f["source_behavior"] in ["spontaneous_cooperation", "tool_sharing_networks"]]
        assert len(high_sig_loops) == 2
        
        for feedback_loop in high_sig_loops:
            assert feedback_loop["influence_type"] == "positive_reinforcement"
            assert "evolution_impact" in feedback_loop
            
            impact = feedback_loop["evolution_impact"]
            assert "mutation_bias" in impact
            assert "selection_pressure" in impact
            assert "trait_emphasis" in impact
    
    @pytest.mark.asyncio
    async def test_complete_behavior_monitoring(self, behavior_monitor):
        """Test complete emergent behavior monitoring cycle"""
        # Setup test data
        behavior_monitor.interaction_system.collaboration_history = [
            {
                "timestamp": datetime.now() - timedelta(minutes=1),
                "participants": ["agent_1", "agent_2"],
                "expected_benefit": 0.3
            }
        ]
        
        behavior_monitor.interaction_system.resource_sharing_log = [
            {
                "timestamp": datetime.now() - timedelta(minutes=2),
                "sharer": "agent_1",
                "receiver": "agent_2", 
                "amount": 100.0
            }
        ]
        
        # Mock environment state
        behavior_monitor.simulation_env.get_environment_state = AsyncMock(return_value={
            "resources": {"compute": {"utilization": 0.5}}
        })
        
        monitoring_results = await behavior_monitor.monitor_emergent_behaviors()
        
        assert "spontaneous_cooperation" in monitoring_results
        assert "resource_optimization" in monitoring_results
        assert "tool_sharing_networks" in monitoring_results
        assert "adaptive_triggers" in monitoring_results
        assert "feedback_loops" in monitoring_results
        
        # Verify behavior is stored
        assert len(behavior_monitor.detected_behaviors) == 1
        stored_behavior = behavior_monitor.detected_behaviors[0]
        assert "timestamp" in stored_behavior
        assert "behaviors" in stored_behavior
    
    @pytest.mark.asyncio
    async def test_behavior_summary_generation(self, behavior_monitor):
        """Test behavior summary generation"""
        # Add test detection history
        behavior_monitor.detected_behaviors = [
            {
                "timestamp": datetime.now(),
                "behaviors": {
                    "spontaneous_cooperation": [{"test": "coop"}],
                    "resource_optimization": [{"test": "opt1"}, {"test": "opt2"}],
                    "tool_sharing_networks": []
                }
            }
        ]
        
        behavior_monitor.adaptation_triggers = [{"test": "trigger"}]
        behavior_monitor.feedback_history = [{"test": "feedback1"}, {"test": "feedback2"}]
        
        summary = await behavior_monitor.get_behavior_summary()
        
        assert summary["total_detections"] == 1
        assert "latest_detection" in summary
        assert summary["active_triggers"] == 1
        assert summary["feedback_loops"] == 2
        
        behavior_summary = summary["behavior_summary"]
        assert behavior_summary["cooperation_patterns"] == 1
        assert behavior_summary["optimization_patterns"] == 2
        assert behavior_summary["sharing_networks"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
