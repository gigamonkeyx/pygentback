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
