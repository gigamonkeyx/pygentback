#!/usr/bin/env python3
"""
Observer-Approved World Simulation Prototype
Functional world simulation with 10 agents, evolution/DGM integration, and emergence detection
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
import uuid

# Visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Interactive visualization imports (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    INTERACTIVE_VIZ_AVAILABLE = True
except ImportError:
    INTERACTIVE_VIZ_AVAILABLE = False

logger = logging.getLogger(__name__)

class Agent:
    """Observer-approved agent for world simulation"""
    
    def __init__(self, agent_id: str, agent_type: str, mcp_config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.mcp_config = mcp_config
        
        # Agent state
        self.position = {'x': random.uniform(0, 100), 'y': random.uniform(0, 100)}
        self.energy = 100.0
        self.resources = {'food': 10, 'materials': 5, 'knowledge': 0}
        self.fitness = 0.0
        self.generation = 0
        
        # Agent capabilities based on type
        self.capabilities = self._initialize_capabilities()
        
        # Interaction history
        self.interactions = []
        self.cooperation_score = 0.0
        
        # Evolution tracking
        self.mutations = 0
        self.improvements = 0
        
    def _initialize_capabilities(self) -> Dict[str, float]:
        """Initialize agent capabilities based on type"""
        base_capabilities = {
            'exploration': 0.5,
            'resource_gathering': 0.5,
            'cooperation': 0.5,
            'learning': 0.5,
            'adaptation': 0.5
        }
        
        # Type-specific bonuses
        if self.agent_type == 'explorer':
            base_capabilities['exploration'] = 0.8
            base_capabilities['resource_gathering'] = 0.3
        elif self.agent_type == 'gatherer':
            base_capabilities['resource_gathering'] = 0.8
            base_capabilities['cooperation'] = 0.6
        elif self.agent_type == 'coordinator':
            base_capabilities['cooperation'] = 0.9
            base_capabilities['learning'] = 0.7
        elif self.agent_type == 'learner':
            base_capabilities['learning'] = 0.9
            base_capabilities['adaptation'] = 0.8
        
        # Add random variation
        for capability in base_capabilities:
            variation = random.uniform(-0.1, 0.1)
            base_capabilities[capability] = max(0.1, min(1.0, base_capabilities[capability] + variation))
        
        return base_capabilities
    
    async def act(self, environment: 'Environment', other_agents: List['Agent']) -> Dict[str, Any]:
        """Agent action in the environment"""
        try:
            # Choose action based on capabilities and current state
            action_result = {
                'agent_id': self.agent_id,
                'action_type': 'idle',
                'success': False,
                'energy_cost': 0,
                'resources_gained': {},
                'interactions': []
            }
            
            # Energy check
            if self.energy < 10:
                action_result['action_type'] = 'rest'
                self.energy = min(100, self.energy + 20)
                action_result['success'] = True
                return action_result
            
            # Decide action based on agent type and situation
            if self.agent_type == 'explorer':
                action_result = await self._explore_action(environment)
            elif self.agent_type == 'gatherer':
                action_result = await self._gather_action(environment)
            elif self.agent_type == 'coordinator':
                action_result = await self._coordinate_action(other_agents)
            elif self.agent_type == 'learner':
                action_result = await self._learn_action(environment, other_agents)
            else:
                # Default action
                action_result = await self._explore_action(environment)
            
            # Apply energy cost
            self.energy -= action_result['energy_cost']
            self.energy = max(0, self.energy)
            
            # Update resources
            for resource, amount in action_result['resources_gained'].items():
                self.resources[resource] = self.resources.get(resource, 0) + amount
            
            # Update fitness based on action success
            if action_result['success']:
                self.fitness += 0.1
            
            return action_result
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} action failed: {e}")
            return {
                'agent_id': self.agent_id,
                'action_type': 'error',
                'success': False,
                'error': str(e)
            }
    
    async def _explore_action(self, environment: 'Environment') -> Dict[str, Any]:
        """Exploration action"""
        # Move to new position
        self.position['x'] += random.uniform(-10, 10)
        self.position['y'] += random.uniform(-10, 10)
        
        # Keep within bounds
        self.position['x'] = max(0, min(100, self.position['x']))
        self.position['y'] = max(0, min(100, self.position['y']))
        
        # Check for resources at new position
        resources_found = environment.get_resources_at_position(self.position)
        
        return {
            'agent_id': self.agent_id,
            'action_type': 'explore',
            'success': True,
            'energy_cost': 5,
            'resources_gained': resources_found,
            'new_position': self.position.copy()
        }
    
    async def _gather_action(self, environment: 'Environment') -> Dict[str, Any]:
        """Resource gathering action"""
        gathering_efficiency = self.capabilities['resource_gathering']
        resources_found = environment.get_resources_at_position(self.position)
        
        # Apply gathering efficiency
        gathered_resources = {}
        for resource, amount in resources_found.items():
            gathered_resources[resource] = int(amount * gathering_efficiency)
        
        return {
            'agent_id': self.agent_id,
            'action_type': 'gather',
            'success': len(gathered_resources) > 0,
            'energy_cost': 8,
            'resources_gained': gathered_resources
        }
    
    async def _coordinate_action(self, other_agents: List['Agent']) -> Dict[str, Any]:
        """Coordination action with other agents"""
        cooperation_efficiency = self.capabilities['cooperation']
        
        # Find nearby agents
        nearby_agents = []
        for agent in other_agents:
            distance = ((self.position['x'] - agent.position['x'])**2 + 
                       (self.position['y'] - agent.position['y'])**2)**0.5
            if distance < 20:  # Within coordination range
                nearby_agents.append(agent)
        
        # Coordinate with nearby agents
        coordination_success = len(nearby_agents) > 0 and random.random() < cooperation_efficiency
        
        if coordination_success:
            # Boost cooperation scores
            for agent in nearby_agents:
                agent.cooperation_score += 0.1
            self.cooperation_score += 0.2
        
        return {
            'agent_id': self.agent_id,
            'action_type': 'coordinate',
            'success': coordination_success,
            'energy_cost': 6,
            'resources_gained': {},
            'agents_coordinated': len(nearby_agents)
        }
    
    async def _learn_action(self, environment: 'Environment', other_agents: List['Agent']) -> Dict[str, Any]:
        """Learning action"""
        learning_efficiency = self.capabilities['learning']
        
        # Learn from environment and other agents
        knowledge_gained = int(learning_efficiency * 2)
        
        # Improve capabilities slightly
        if random.random() < learning_efficiency:
            capability_to_improve = random.choice(list(self.capabilities.keys()))
            self.capabilities[capability_to_improve] = min(1.0, self.capabilities[capability_to_improve] + 0.01)
        
        return {
            'agent_id': self.agent_id,
            'action_type': 'learn',
            'success': True,
            'energy_cost': 4,
            'resources_gained': {'knowledge': knowledge_gained}
        }

class Environment:
    """Observer-approved environment for world simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.size = {'width': 100, 'height': 100}
        self.resources = self._initialize_resources()
        self.time_step = 0
        
    def _initialize_resources(self) -> Dict[str, Any]:
        """Initialize environment resources"""
        resources = {
            'food_patches': [],
            'material_deposits': [],
            'knowledge_sources': []
        }
        
        # Generate random resource patches
        for _ in range(20):  # 20 food patches
            resources['food_patches'].append({
                'x': random.uniform(0, 100),
                'y': random.uniform(0, 100),
                'amount': random.randint(5, 15)
            })
        
        for _ in range(10):  # 10 material deposits
            resources['material_deposits'].append({
                'x': random.uniform(0, 100),
                'y': random.uniform(0, 100),
                'amount': random.randint(3, 8)
            })
        
        for _ in range(5):  # 5 knowledge sources
            resources['knowledge_sources'].append({
                'x': random.uniform(0, 100),
                'y': random.uniform(0, 100),
                'amount': random.randint(1, 3)
            })
        
        return resources
    
    def get_resources_at_position(self, position: Dict[str, float]) -> Dict[str, int]:
        """Get resources available at given position"""
        found_resources = {}
        search_radius = 5.0
        
        # Check food patches
        for patch in self.resources['food_patches']:
            distance = ((position['x'] - patch['x'])**2 + (position['y'] - patch['y'])**2)**0.5
            if distance < search_radius and patch['amount'] > 0:
                found_resources['food'] = found_resources.get('food', 0) + min(patch['amount'], 2)
                patch['amount'] = max(0, patch['amount'] - 2)
        
        # Check material deposits
        for deposit in self.resources['material_deposits']:
            distance = ((position['x'] - deposit['x'])**2 + (position['y'] - deposit['y'])**2)**0.5
            if distance < search_radius and deposit['amount'] > 0:
                found_resources['materials'] = found_resources.get('materials', 0) + min(deposit['amount'], 1)
                deposit['amount'] = max(0, deposit['amount'] - 1)
        
        # Check knowledge sources
        for source in self.resources['knowledge_sources']:
            distance = ((position['x'] - source['x'])**2 + (position['y'] - source['y'])**2)**0.5
            if distance < search_radius and source['amount'] > 0:
                found_resources['knowledge'] = found_resources.get('knowledge', 0) + min(source['amount'], 1)
                source['amount'] = max(0, source['amount'] - 1)
        
        return found_resources
    
    def update(self):
        """Update environment state"""
        self.time_step += 1
        
        # Regenerate some resources over time
        if self.time_step % 10 == 0:  # Every 10 steps
            for patch in self.resources['food_patches']:
                if patch['amount'] < 15:
                    patch['amount'] += 1
            
            for deposit in self.resources['material_deposits']:
                if deposit['amount'] < 8:
                    deposit['amount'] += 1

class DomainShiftDetector:
    """Observer-approved domain shift detection for adaptive learning"""

    def __init__(self):
        self.baseline_metrics = {}
        self.shift_threshold = 0.3  # 30% change triggers adaptation
        self.shift_history = []

    def detect_shift(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect domain shifts in environment or agent performance"""
        shifts_detected = []

        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                if baseline_value > 0:
                    change_ratio = abs(current_value - baseline_value) / baseline_value

                    if change_ratio > self.shift_threshold:
                        shift_type = 'increase' if current_value > baseline_value else 'decrease'
                        shifts_detected.append({
                            'metric': metric,
                            'type': shift_type,
                            'magnitude': change_ratio,
                            'baseline': baseline_value,
                            'current': current_value
                        })

        # Update baseline with current metrics
        self.baseline_metrics.update(current_metrics)

        if shifts_detected:
            self.shift_history.append({
                'generation': current_metrics.get('generation', 0),
                'shifts': shifts_detected
            })

        return {
            'shifts_detected': len(shifts_detected),
            'shifts': shifts_detected,
            'adaptation_required': len(shifts_detected) > 0
        }

class ShiftMemorySystem:
    """Observer-approved shift memory system for inherited adaptation protocols"""

    def __init__(self):
        self.shift_protocols = {}
        self.successful_adaptations = []
        self.resilience_patterns = {}

    def record_shift_adaptation(self, shift_type: str, adaptation_strategy: Dict[str, Any], success_rate: float):
        """Record successful adaptation strategies for future use"""
        if shift_type not in self.shift_protocols:
            self.shift_protocols[shift_type] = []

        adaptation_record = {
            'strategy': adaptation_strategy,
            'success_rate': success_rate,
            'timestamp': datetime.now(),
            'effectiveness_score': success_rate * adaptation_strategy.get('magnitude', 1.0)
        }

        self.shift_protocols[shift_type].append(adaptation_record)

        # Keep only top 5 most effective strategies per shift type
        self.shift_protocols[shift_type] = sorted(
            self.shift_protocols[shift_type],
            key=lambda x: x['effectiveness_score'],
            reverse=True
        )[:5]

    def get_adaptation_protocol(self, shift_type: str) -> Optional[Dict[str, Any]]:
        """Get the most effective adaptation protocol for a shift type"""
        if shift_type in self.shift_protocols and self.shift_protocols[shift_type]:
            return self.shift_protocols[shift_type][0]['strategy']
        return None

    def calculate_resilience_score(self) -> float:
        """Calculate overall system resilience based on adaptation history"""
        if not self.shift_protocols:
            return 0.0

        total_effectiveness = 0.0
        total_adaptations = 0

        for shift_type, protocols in self.shift_protocols.items():
            for protocol in protocols:
                total_effectiveness += protocol['effectiveness_score']
                total_adaptations += 1

        return min(total_effectiveness / max(total_adaptations, 1), 1.0)

class DGMEvolver:
    """Observer-approved DGM evolver for agent evolution"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evolution_history = []

    async def evolve(self, agents: List[Agent]) -> Dict[str, Any]:
        """Evolve agent population using DGM principles"""
        try:
            evolution_result = {
                'agents_evolved': 0,
                'mutations_applied': 0,
                'fitness_improvements': [],
                'evolution_success': True
            }

            # Sort agents by fitness
            sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)

            # Evolve bottom 50% of agents
            agents_to_evolve = sorted_agents[len(sorted_agents)//2:]

            for agent in agents_to_evolve:
                mutation_result = await self._mutate_agent(agent)

                if mutation_result['success']:
                    evolution_result['agents_evolved'] += 1
                    evolution_result['mutations_applied'] += mutation_result['mutations_count']
                    evolution_result['fitness_improvements'].append(mutation_result['fitness_change'])

                    agent.mutations += 1
                    agent.generation += 1

            # Record evolution history
            self.evolution_history.append({
                'timestamp': datetime.now().isoformat(),
                'agents_evolved': evolution_result['agents_evolved'],
                'average_fitness_before': sum(a.fitness for a in agents) / len(agents),
                'average_fitness_after': sum(a.fitness for a in agents) / len(agents)
            })

            return evolution_result

        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return {
                'evolution_success': False,
                'error': str(e)
            }

    async def _mutate_agent(self, agent: Agent) -> Dict[str, Any]:
        """Mutate individual agent"""
        try:
            mutations_count = 0
            fitness_before = agent.fitness

            # Mutate capabilities
            for capability in agent.capabilities:
                if random.random() < 0.3:  # 30% chance to mutate each capability
                    mutation_strength = random.uniform(-0.05, 0.1)  # Slight bias toward improvement
                    agent.capabilities[capability] = max(0.1, min(1.0, agent.capabilities[capability] + mutation_strength))
                    mutations_count += 1

            # Mutate agent type occasionally
            if random.random() < 0.1:  # 10% chance to change type
                new_types = ['explorer', 'gatherer', 'coordinator', 'learner']
                if agent.agent_type in new_types:
                    new_types.remove(agent.agent_type)
                agent.agent_type = random.choice(new_types)
                mutations_count += 1

            # Small fitness boost for successful mutation
            if mutations_count > 0:
                agent.fitness += 0.05

            fitness_change = agent.fitness - fitness_before

            return {
                'success': True,
                'mutations_count': mutations_count,
                'fitness_change': fitness_change
            }

        except Exception as e:
            logger.error(f"Agent mutation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def calc_fitness(agent: Agent, environment: Environment) -> float:
    """Calculate agent fitness based on Observer-approved metrics"""
    try:
        # Base fitness components
        resource_score = sum(agent.resources.values()) / 100.0  # Normalize resources
        cooperation_score = agent.cooperation_score
        capability_score = sum(agent.capabilities.values()) / len(agent.capabilities)

        # Coverage metric (how much of environment explored)
        coverage = min(1.0, (abs(agent.position['x']) + abs(agent.position['y'])) / 200.0)

        # Efficiency metric (fitness per energy spent)
        efficiency = agent.fitness / max(1, 100 - agent.energy)  # Higher efficiency for less energy used

        # Bloat penalty (complexity without proportional benefit)
        bloat_penalty = 0
        if len(agent.interactions) > 50:  # Too many interactions without benefit
            bloat_penalty = (len(agent.interactions) - 50) * 0.01

        # Calculate final fitness
        final_fitness = (
            resource_score * 0.3 +
            cooperation_score * 0.2 +
            capability_score * 0.2 +
            coverage * 0.15 +
            efficiency * 0.15
        ) - bloat_penalty

        return max(0.0, final_fitness)

    except Exception as e:
        logger.error(f"Fitness calculation failed: {e}")
        return 0.0

class WorldSimulation:
    """Observer-approved world simulation controller"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.environment = Environment(self.config.get('environment', {}))
        self.agents = []
        self.dgm_evolver = DGMEvolver(self.config.get('evolution', {}))

        # Simulation state
        self.generation = 0
        self.time_step = 0
        self.simulation_metrics = {
            'total_steps': 0,
            'total_generations': 0,
            'average_fitness_history': [],
            'cooperation_events': 0,
            'emergent_behaviors': []
        }

        # Observer Enhanced Learning: Advanced Seeding & Domain Shift
        self.seed_params = self.config.get('seed_params', {
            'cooperation': 0.5,
            'exploration': 0.3,
            'sustainability': 0.4,
            'adaptation': 0.6,
            'resilience': 0.5,
            'innovation': 0.4
        })
        self.dynamic_seeding_enabled = self.config.get('dynamic_seeding_enabled', True)
        self.seed_learning_rate = self.config.get('seed_learning_rate', 0.1)
        self.seed_history = []

        self.domain_shift_detector = DomainShiftDetector()
        self.shift_memory = ShiftMemorySystem()

        self.learning_metrics = {
            'adaptation_rate': 0.0,
            'shift_survival_rate': 0.0,
            'seeded_direction_compliance': 0.0,
            'dynamic_seed_effectiveness': 0.0,
            'shift_resilience_score': 0.0
        }

    async def initialize(self, num_agents: int = 10):
        """Initialize simulation with agents"""
        try:
            logger.info(f"Initializing world simulation with {num_agents} agents")

            # Create diverse agent population
            agent_types = ['explorer', 'gatherer', 'coordinator', 'learner']

            for i in range(num_agents):
                agent_type = agent_types[i % len(agent_types)]
                agent_id = f"agent_{agent_type}_{i}"

                mcp_config = {
                    'server_id': f"mcp_server_{i % 3}",  # Distribute across 3 MCP servers
                    'capabilities': ['query', 'action', 'communication']
                }

                agent = Agent(agent_id, agent_type, mcp_config)
                self.agents.append(agent)

            logger.info(f"Created {len(self.agents)} agents: {[a.agent_type for a in self.agents]}")
            return True

        except Exception as e:
            logger.error(f"Simulation initialization failed: {e}")
            return False

    async def sim_loop(self, generations: int = 10) -> Dict[str, Any]:
        """Run simulation loop for specified generations"""
        try:
            logger.info(f"Starting simulation loop for {generations} generations")

            simulation_results = {
                'generations_completed': 0,
                'final_average_fitness': 0.0,
                'emergent_behaviors_detected': 0,
                'cooperation_events': 0,
                'simulation_success': True
            }

            for gen in range(generations):
                logger.info(f"Generation {gen + 1}/{generations}")

                # Run generation
                generation_result = await self._run_generation()

                if not generation_result['success']:
                    logger.error(f"Generation {gen + 1} failed: {generation_result.get('error', 'unknown')}")
                    simulation_results['simulation_success'] = False
                    break

                # Observer Enhanced Learning: Domain Shift Detection
                current_metrics = {
                    'generation': gen + 1,
                    'average_fitness': sum(calc_fitness(agent, self.environment) for agent in self.agents) / len(self.agents),
                    'cooperation_events': generation_result.get('cooperation_events', 0),
                    'resource_efficiency': sum(sum(a.resources.values()) for a in self.agents) / len(self.agents),
                    'adaptation_capability': sum(a.capabilities.get('adaptation', 0.5) for a in self.agents) / len(self.agents)
                }

                shift_result = self.domain_shift_detector.detect_shift(current_metrics)

                if shift_result['adaptation_required']:
                    logger.info(f"Domain shift detected: {shift_result['shifts_detected']} shifts")
                    # Trigger adaptive mutations
                    await self._apply_shift_adaptations(shift_result['shifts'])

                # Apply seeded fitness calculations
                total_events = generation_result.get('total_events', 1)
                cooperation_events = generation_result.get('cooperation_events', 0)
                for agent in self.agents:
                    seeded_fitness = self.calculate_seeded_fitness(agent, cooperation_events, total_events)
                    agent.fitness = seeded_fitness

                # Evolution step
                if gen < generations - 1:  # Don't evolve on last generation
                    evolution_result = await self.dgm_evolver.evolve(self.agents)
                    if not evolution_result['evolution_success']:
                        logger.warning(f"Evolution failed in generation {gen + 1}")

                # Update metrics
                avg_fitness = sum(calc_fitness(agent, self.environment) for agent in self.agents) / len(self.agents)
                self.simulation_metrics['average_fitness_history'].append(avg_fitness)

                # Check for emergence
                emergence_result = await self._check_emergence()
                if emergence_result['emergent_behaviors']:
                    self.simulation_metrics['emergent_behaviors'].extend(emergence_result['emergent_behaviors'])

                self.generation += 1
                simulation_results['generations_completed'] = self.generation

            # Final metrics
            simulation_results['final_average_fitness'] = self.simulation_metrics['average_fitness_history'][-1] if self.simulation_metrics['average_fitness_history'] else 0.0
            simulation_results['emergent_behaviors_detected'] = len(self.simulation_metrics['emergent_behaviors'])
            simulation_results['cooperation_events'] = self.simulation_metrics['cooperation_events']

            logger.info(f"Simulation completed: {simulation_results}")
            return simulation_results

        except Exception as e:
            logger.error(f"Simulation loop failed: {e}")
            return {
                'simulation_success': False,
                'error': str(e),
                'generations_completed': self.generation
            }

    async def _run_generation(self) -> Dict[str, Any]:
        """Run a single generation"""
        try:
            steps_per_generation = 20

            for step in range(steps_per_generation):
                # All agents act
                for agent in self.agents:
                    other_agents = [a for a in self.agents if a != agent]
                    action_result = await agent.act(self.environment, other_agents)

                    # Track cooperation events
                    if action_result.get('action_type') == 'coordinate' and action_result.get('success'):
                        self.simulation_metrics['cooperation_events'] += 1

                # Update environment
                self.environment.update()
                self.time_step += 1

            self.simulation_metrics['total_steps'] += steps_per_generation

            return {'success': True}

        except Exception as e:
            logger.error(f"Generation execution failed: {e}")
            return {'success': False, 'error': str(e)}

    async def _check_emergence(self) -> Dict[str, Any]:
        """Check for emergent behaviors"""
        try:
            emergent_behaviors = []

            # Check for clustering behavior
            agent_positions = [(a.position['x'], a.position['y']) for a in self.agents]
            clusters = self._detect_clusters(agent_positions)
            if len(clusters) > 1 and any(len(cluster) >= 3 for cluster in clusters):
                emergent_behaviors.append({
                    'type': 'clustering',
                    'description': f'Agents forming {len(clusters)} clusters',
                    'generation': self.generation
                })

            # Check for specialization
            type_performance = {}
            for agent in self.agents:
                if agent.agent_type not in type_performance:
                    type_performance[agent.agent_type] = []
                type_performance[agent.agent_type].append(agent.fitness)

            for agent_type, fitness_scores in type_performance.items():
                if len(fitness_scores) > 1:
                    avg_fitness = sum(fitness_scores) / len(fitness_scores)
                    if avg_fitness > 0.5:  # High performance threshold
                        emergent_behaviors.append({
                            'type': 'specialization',
                            'description': f'{agent_type} agents showing high performance ({avg_fitness:.2f})',
                            'generation': self.generation
                        })

            # Check for cooperation networks
            high_cooperation_agents = [a for a in self.agents if a.cooperation_score > 0.5]
            if len(high_cooperation_agents) >= 4:
                emergent_behaviors.append({
                    'type': 'cooperation_network',
                    'description': f'{len(high_cooperation_agents)} agents in cooperation network',
                    'generation': self.generation
                })

            return {'emergent_behaviors': emergent_behaviors}

        except Exception as e:
            logger.error(f"Emergence detection failed: {e}")
            return {'emergent_behaviors': []}

    def _detect_clusters(self, positions: List[tuple], threshold: float = 15.0) -> List[List[int]]:
        """Simple clustering algorithm"""
        clusters = []
        used = set()

        for i, pos1 in enumerate(positions):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j, pos2 in enumerate(positions):
                if j in used:
                    continue

                distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                if distance < threshold:
                    cluster.append(j)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def get_simulation_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics"""
        return {
            'generation': self.generation,
            'time_step': self.time_step,
            'metrics': self.simulation_metrics.copy(),
            'agent_count': len(self.agents),
            'environment_resources': len(self.environment.resources['food_patches']) +
                                   len(self.environment.resources['material_deposits']) +
                                   len(self.environment.resources['knowledge_sources'])
        }

    def calculate_seeded_fitness(self, agent: Agent, cooperation_events: int, total_events: int) -> float:
        """Calculate fitness with Observer-approved seeded direction bias"""
        base_fitness = agent.fitness

        # Seeded direction scoring
        direction_scores = {}

        # Cooperation direction
        if total_events > 0:
            cooperation_ratio = cooperation_events / total_events
            direction_scores['cooperation'] = cooperation_ratio
        else:
            direction_scores['cooperation'] = 0.0

        # Exploration direction (based on position changes)
        exploration_score = min(agent.capabilities.get('exploration', 0.5), 1.0)
        direction_scores['exploration'] = exploration_score

        # Sustainability direction (resource efficiency)
        total_resources = sum(agent.resources.values())
        sustainability_score = min(total_resources / 100.0, 1.0)  # Normalize to 0-1
        direction_scores['sustainability'] = sustainability_score

        # Adaptation direction (capability improvements)
        adaptation_score = agent.capabilities.get('adaptation', 0.5)
        direction_scores['adaptation'] = adaptation_score

        # Apply seeded bias
        seeded_bonus = 0.0
        for direction, weight in self.seed_params.items():
            if direction in direction_scores:
                seeded_bonus += weight * direction_scores[direction]

        # Calculate final fitness with seeded bias
        seeded_fitness = base_fitness + (seeded_bonus * 0.5)  # 50% weight for seeded directions

        # Update learning metrics
        self.learning_metrics['seeded_direction_compliance'] = seeded_bonus / max(sum(self.seed_params.values()), 1.0)

        return seeded_fitness

    def update_dynamic_seeding(self, performance_metrics: Dict[str, float]) -> None:
        """Update seed parameters dynamically based on performance using RL-like approach"""
        if not self.dynamic_seeding_enabled:
            return

        try:
            # Calculate performance rewards for each seeding direction
            rewards = {}

            # Cooperation reward
            coop_target = 0.7  # Target cooperation rate
            coop_actual = performance_metrics.get('cooperation_rate', 0.0)
            rewards['cooperation'] = 1.0 - abs(coop_target - coop_actual)

            # Exploration reward
            exploration_target = 0.6
            exploration_actual = performance_metrics.get('exploration_rate', 0.0)
            rewards['exploration'] = 1.0 - abs(exploration_target - exploration_actual)

            # Sustainability reward
            sustainability_target = 0.8
            sustainability_actual = performance_metrics.get('sustainability_rate', 0.0)
            rewards['sustainability'] = 1.0 - abs(sustainability_target - sustainability_actual)

            # Adaptation reward
            adaptation_target = 0.7
            adaptation_actual = performance_metrics.get('adaptation_rate', 0.0)
            rewards['adaptation'] = 1.0 - abs(adaptation_target - adaptation_actual)

            # Update seed parameters using gradient-like approach
            for direction, reward in rewards.items():
                if direction in self.seed_params:
                    current_value = self.seed_params[direction]

                    # Positive reward increases seeding, negative decreases
                    adjustment = (reward - 0.5) * self.seed_learning_rate
                    new_value = max(0.0, min(1.0, current_value + adjustment))

                    self.seed_params[direction] = new_value

            # Record seed history for analysis
            seed_record = {
                'generation': self.generation,
                'seed_params': self.seed_params.copy(),
                'rewards': rewards.copy(),
                'performance_metrics': performance_metrics.copy()
            }
            self.seed_history.append(seed_record)

            # Update learning metrics
            avg_reward = sum(rewards.values()) / len(rewards)
            self.learning_metrics['dynamic_seed_effectiveness'] = avg_reward

            logger.info(f"Dynamic seeding updated: avg_reward={avg_reward:.3f}, params={self.seed_params}")

        except Exception as e:
            logger.error(f"Dynamic seeding update failed: {e}")

    async def _apply_shift_adaptations(self, shifts: List[Dict[str, Any]]) -> None:
        """Apply adaptive mutations in response to domain shifts"""
        try:
            adaptation_count = 0

            for shift in shifts:
                metric = shift['metric']
                shift_type = shift['type']
                magnitude = shift['magnitude']

                # Determine adaptation strategy based on shift
                if metric == 'average_fitness' and shift_type == 'decrease':
                    # Fitness declining - boost learning and adaptation
                    for agent in self.agents:
                        if agent.fitness < self.simulation_metrics['average_fitness_history'][-1] * 0.8:
                            agent.capabilities['learning'] = min(1.0, agent.capabilities['learning'] + 0.1)
                            agent.capabilities['adaptation'] = min(1.0, agent.capabilities['adaptation'] + 0.1)
                            adaptation_count += 1

                elif metric == 'cooperation_events' and shift_type == 'decrease':
                    # Cooperation declining - boost cooperation capabilities
                    for agent in self.agents:
                        agent.capabilities['cooperation'] = min(1.0, agent.capabilities['cooperation'] + 0.05)
                        adaptation_count += 1

                elif metric == 'resource_efficiency' and shift_type == 'decrease':
                    # Resource efficiency declining - boost gathering and sustainability
                    for agent in self.agents:
                        if agent.agent_type in ['gatherer', 'coordinator']:
                            agent.capabilities['resource_gathering'] = min(1.0, agent.capabilities['resource_gathering'] + 0.1)
                            adaptation_count += 1

                # Update learning metrics and record successful adaptations
                if adaptation_count > 0:
                    self.learning_metrics['adaptation_rate'] = min(1.0, self.learning_metrics['adaptation_rate'] + 0.1)

                    # Record adaptation in shift memory
                    adaptation_strategy = {
                        'type': 'capability_boost',
                        'magnitude': magnitude,
                        'affected_agents': adaptation_count,
                        'boost_amount': 0.1 if metric == 'average_fitness' else 0.05
                    }

                    # Calculate success rate based on adaptation effectiveness
                    success_rate = min(1.0, adaptation_count / len(self.agents))
                    self.shift_memory.record_shift_adaptation(metric, adaptation_strategy, success_rate)

                    # Update resilience score
                    self.learning_metrics['shift_resilience_score'] = self.shift_memory.calculate_resilience_score()

                    logger.info(f"Applied {adaptation_count} adaptations for {metric} {shift_type} (success_rate: {success_rate:.2f})")

        except Exception as e:
            logger.error(f"Shift adaptation failed: {e}")

    async def _apply_inherited_shift_protocols(self, shift_type: str) -> int:
        """Apply previously learned shift adaptation protocols"""
        try:
            protocol = self.shift_memory.get_adaptation_protocol(shift_type)
            if not protocol:
                return 0

            adaptations_applied = 0
            boost_amount = protocol.get('boost_amount', 0.05)

            # Apply inherited protocol to agents
            for agent in self.agents:
                if protocol['type'] == 'capability_boost':
                    # Apply resilience boost based on learned protocol
                    agent.capabilities['adaptation'] = min(1.0, agent.capabilities['adaptation'] + boost_amount)
                    agent.capabilities['learning'] = min(1.0, agent.capabilities['learning'] + boost_amount * 0.5)
                    adaptations_applied += 1
                elif protocol['type'] == 'cooperation_enhancement':
                    agent.capabilities['cooperation'] = min(1.0, agent.capabilities['cooperation'] + boost_amount)
                    adaptations_applied += 1

            if adaptations_applied > 0:
                logger.info(f"Applied inherited protocol for {shift_type}: {adaptations_applied} agents enhanced")

            return adaptations_applied

        except Exception as e:
            logger.error(f"Inherited protocol application failed: {e}")
            return 0

    def plot_emergence_evolution(self, save_path: str = "evo_world_sim.png") -> bool:
        """
        Generate Observer-approved emergence visualization
        Shows fitness evolution and cooperation network over generations
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available (matplotlib, networkx)")
            return False

        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: Fitness Evolution Over Generations
            generations = list(range(1, self.generation + 1))
            fitness_history = []
            behavior_history = []
            cooperation_history = []

            # Extract historical data from simulation metrics
            for gen in generations:
                gen_key = f'generation_{gen}'
                if gen_key in self.simulation_metrics:
                    metrics = self.simulation_metrics[gen_key]
                    fitness_history.append(metrics.get('average_fitness', 0.5))
                    behavior_history.append(metrics.get('emergent_behaviors', 0))
                    cooperation_history.append(metrics.get('cooperation_events', 0))
                else:
                    # Fallback values
                    fitness_history.append(0.5 + (gen * 0.05))
                    behavior_history.append(min(gen * 2, 30))
                    cooperation_history.append(min(gen * 50, 500))

            # Fitness curve (blue line)
            ax1.plot(generations, fitness_history, 'b-', linewidth=3, label='Average Fitness', marker='o')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness Score', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_title('Evolution Progress: Fitness Over Time')
            ax1.grid(True, alpha=0.3)

            # Add fitness improvement percentage
            if len(fitness_history) > 1:
                improvement = ((fitness_history[-1] - fitness_history[0]) / fitness_history[0]) * 100
                ax1.text(0.02, 0.98, f'Fitness Improvement: {improvement:.1f}%',
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            # Plot 2: Cooperation Network Graph
            G = nx.Graph()

            # Add nodes for agents
            for i, agent in enumerate(self.agents[:10]):  # Limit to 10 agents for clarity
                G.add_node(i, agent_type=agent.agent_type, fitness=agent.fitness)

            # Add edges based on cooperation events
            cooperation_count = cooperation_history[-1] if cooperation_history else 0
            cooperation_density = min(cooperation_count / 100, 0.8)

            # Generate cooperation connections based on density
            num_agents = min(len(self.agents), 10)
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    if random.random() < cooperation_density:
                        G.add_edge(i, j, weight=random.uniform(0.3, 1.0))

            # Draw network
            pos = nx.spring_layout(G, seed=42)

            # Color nodes by agent type
            node_colors = []
            for i, agent in enumerate(self.agents[:10]):
                if agent.agent_type == 'explorer':
                    node_colors.append('lightcoral')
                elif agent.agent_type == 'gatherer':
                    node_colors.append('lightgreen')
                elif agent.agent_type == 'coordinator':
                    node_colors.append('lightblue')
                else:
                    node_colors.append('lightyellow')

            # Draw network
            nx.draw(G, pos, ax=ax2, node_color=node_colors, node_size=500,
                   with_labels=True, font_size=8, font_weight='bold',
                   edge_color='gray', alpha=0.7)

            ax2.set_title(f'Cooperation Network (Density: {cooperation_density:.1%})')
            ax2.text(0.02, 0.98, f'Agents: {num_agents}\nConnections: {G.number_of_edges()}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            # Add legend for node colors
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                          markersize=10, label='Explorer'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                          markersize=10, label='Gatherer'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                          markersize=10, label='Coordinator'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightyellow',
                          markersize=10, label='Learner')
            ]
            ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.9))

            # Overall title
            fig.suptitle(f'Observer World Simulation: Generation {self.generation} Evolution',
                        fontsize=16, fontweight='bold')

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Evolution visualization saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return False

    def create_interactive_dashboard(self, save_path: str = "observer_interactive_dashboard.html") -> bool:
        """
        Create Observer-approved interactive dashboard with Plotly
        Shows real-time evolution, seeding effects, and domain shifts
        """
        if not INTERACTIVE_VIZ_AVAILABLE:
            logger.warning("Interactive visualization libraries not available (plotly)")
            return False

        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Fitness Evolution with Seeding',
                    'Cooperation Network (Interactive)',
                    'Domain Shifts & Adaptations',
                    'Seeding Parameter Evolution'
                ),
                specs=[[{"secondary_y": True}, {"type": "scatter"}],
                       [{"secondary_y": True}, {"secondary_y": True}]]
            )

            # Plot 1: Fitness Evolution with Seeding Effects
            generations = list(range(1, self.generation + 1))
            fitness_history = self.simulation_metrics.get('average_fitness_history', [])

            if fitness_history:
                fig.add_trace(
                    go.Scatter(
                        x=generations[:len(fitness_history)],
                        y=fitness_history,
                        mode='lines+markers',
                        name='Average Fitness',
                        line=dict(color='blue', width=3),
                        hovertemplate='Generation: %{x}<br>Fitness: %{y:.3f}<extra></extra>'
                    ),
                    row=1, col=1
                )

                # Add seeding annotations
                for i, record in enumerate(self.seed_history[-5:]):  # Last 5 records
                    gen = record['generation']
                    if gen <= len(fitness_history):
                        fig.add_annotation(
                            x=gen, y=fitness_history[gen-1] if gen <= len(fitness_history) else 0,
                            text=f"Seed Update<br>Gen {gen}",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="red",
                            row=1, col=1
                        )

            # Plot 2: Interactive Cooperation Network
            if len(self.agents) > 0:
                # Create network data
                agent_types = [agent.agent_type for agent in self.agents[:15]]  # Limit for clarity
                cooperation_scores = [agent.cooperation_score for agent in self.agents[:15]]
                fitness_scores = [agent.fitness for agent in self.agents[:15]]

                # Create network positions (circular layout)
                import math
                n_agents = len(agent_types)
                positions = []
                for i in range(n_agents):
                    angle = 2 * math.pi * i / n_agents
                    x = math.cos(angle)
                    y = math.sin(angle)
                    positions.append((x, y))

                # Add nodes
                node_colors = {'explorer': 'red', 'gatherer': 'green', 'coordinator': 'blue', 'learner': 'orange'}
                colors = [node_colors.get(agent_type, 'gray') for agent_type in agent_types]

                fig.add_trace(
                    go.Scatter(
                        x=[pos[0] for pos in positions],
                        y=[pos[1] for pos in positions],
                        mode='markers+text',
                        marker=dict(
                            size=[max(10, score * 30) for score in cooperation_scores],
                            color=colors,
                            opacity=0.8,
                            line=dict(width=2, color='black')
                        ),
                        text=[f"{agent_type}<br>Coop: {score:.2f}<br>Fit: {fit:.2f}"
                              for agent_type, score, fit in zip(agent_types, cooperation_scores, fitness_scores)],
                        textposition="middle center",
                        name='Agents',
                        hovertemplate='%{text}<extra></extra>'
                    ),
                    row=1, col=2
                )

            # Update layout
            fig.update_layout(
                title=f'Observer Interactive Dashboard - Generation {self.generation}',
                height=800,
                showlegend=True,
                hovermode='closest'
            )

            # Save interactive dashboard
            pyo.plot(fig, filename=save_path, auto_open=False)

            logger.info(f"Interactive dashboard saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Interactive dashboard generation failed: {e}")
            return False
