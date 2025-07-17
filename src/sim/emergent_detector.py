#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emergent Behavior Detector with Shift Learning
Observer-approved system for detecting emergent behaviors and RL-based shift adaptation
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmergentBehavior:
    """Detected emergent behavior"""
    behavior_id: str
    behavior_type: str
    agents_involved: List[str]
    strength: float
    timestamp: datetime
    description: str
    fitness_impact: float = 0.0

@dataclass
class EnvironmentShift:
    """Detected environment shift"""
    shift_id: str
    shift_type: str
    magnitude: float
    timestamp: datetime
    affected_agents: List[str]
    survival_rate: float = 0.0

class ShiftLearningRL:
    """
    Observer-approved RL system for shift adaptation learning
    Rewards mutations that improve survival during environment shifts
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shift_threshold = config.get('shift_threshold', 0.3)
        self.survival_reward_threshold = config.get('survival_reward_threshold', 0.9)
        self.learning_rate = config.get('learning_rate', 0.1)
        
        # RL state tracking
        self.shift_history = []
        self.adaptation_rewards = []
        self.mutation_success_rates = {}
        
        logger.info("ShiftLearningRL initialized for adaptation learning")
    
    def detect_environment_shift(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> Optional[EnvironmentShift]:
        """Detect significant environment shifts"""
        try:
            if not previous_state:
                return None
            
            # Calculate shift magnitude across key metrics
            shift_factors = []
            
            # Resource availability shift
            if 'resource_availability' in current_state and 'resource_availability' in previous_state:
                resource_shift = abs(current_state['resource_availability'] - previous_state['resource_availability'])
                shift_factors.append(resource_shift)
            
            # Agent population shift
            if 'agent_count' in current_state and 'agent_count' in previous_state:
                population_shift = abs(current_state['agent_count'] - previous_state['agent_count']) / max(1, previous_state['agent_count'])
                shift_factors.append(population_shift)
            
            # Fitness variance shift
            if 'avg_fitness' in current_state and 'avg_fitness' in previous_state:
                fitness_shift = abs(current_state['avg_fitness'] - previous_state['avg_fitness']) / max(0.1, previous_state['avg_fitness'])
                shift_factors.append(fitness_shift)
            
            # Calculate overall shift magnitude
            if shift_factors:
                shift_magnitude = sum(shift_factors) / len(shift_factors)
                
                if shift_magnitude > self.shift_threshold:
                    # Determine shift type
                    shift_type = self._classify_shift_type(current_state, previous_state)
                    
                    shift = EnvironmentShift(
                        shift_id=f"shift_{int(time.time())}",
                        shift_type=shift_type,
                        magnitude=shift_magnitude,
                        timestamp=datetime.now(),
                        affected_agents=current_state.get('agent_ids', [])
                    )
                    
                    self.shift_history.append(shift)
                    logger.info(f"Environment shift detected: {shift_type} (magnitude: {shift_magnitude:.3f})")
                    
                    return shift
            
            return None
            
        except Exception as e:
            logger.error(f"Environment shift detection failed: {e}")
            return None
    
    def calculate_adaptation_reward(self, agent_id: str, pre_shift_fitness: float, post_shift_fitness: float, survival: bool) -> float:
        """Calculate RL reward for shift adaptation"""
        try:
            # Base survival reward
            survival_reward = 1.0 if survival else 0.0
            
            # Fitness improvement reward
            if survival and pre_shift_fitness > 0:
                fitness_improvement = (post_shift_fitness - pre_shift_fitness) / pre_shift_fitness
                fitness_reward = max(0.0, fitness_improvement)
            else:
                fitness_reward = 0.0
            
            # Combined reward with Observer-approved weighting
            total_reward = (survival_reward * 0.6) + (fitness_reward * 0.4)
            
            # Bonus for exceeding survival threshold
            if survival and post_shift_fitness > pre_shift_fitness * 1.1:  # 10% improvement
                total_reward += 0.3  # Observer bonus
            
            # Record reward
            self.adaptation_rewards.append({
                'agent_id': agent_id,
                'reward': total_reward,
                'survival': survival,
                'fitness_improvement': fitness_improvement if survival else -1.0,
                'timestamp': datetime.now()
            })
            
            logger.debug(f"Adaptation reward for {agent_id}: {total_reward:.3f} (survival: {survival}, fitness_improvement: {fitness_improvement:.3f})")
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Adaptation reward calculation failed: {e}")
            return 0.0
    
    def get_mutation_bias(self, agent_type: str, current_environment: Dict[str, Any]) -> Dict[str, float]:
        """Get RL-learned mutation bias for current environment"""
        try:
            # Base mutation probabilities
            base_mutations = {
                'cooperation': 0.2,
                'exploration': 0.2,
                'resource_gathering': 0.2,
                'adaptation': 0.2,
                'efficiency': 0.2
            }
            
            # Apply learned biases based on recent shift adaptations
            if self.adaptation_rewards:
                recent_rewards = self.adaptation_rewards[-10:]  # Last 10 adaptations
                
                # Calculate success rates for different mutation types
                for reward_record in recent_rewards:
                    if reward_record['survival'] and reward_record['reward'] > 0.5:
                        # Successful adaptation - bias toward similar mutations
                        if 'resource_scarcity' in str(current_environment):
                            base_mutations['resource_gathering'] += 0.1
                            base_mutations['efficiency'] += 0.1
                        
                        if 'high_competition' in str(current_environment):
                            base_mutations['cooperation'] += 0.1
                            base_mutations['adaptation'] += 0.1
                        
                        if 'environmental_change' in str(current_environment):
                            base_mutations['exploration'] += 0.1
                            base_mutations['adaptation'] += 0.1
            
            # Normalize probabilities
            total_prob = sum(base_mutations.values())
            if total_prob > 0:
                for mutation_type in base_mutations:
                    base_mutations[mutation_type] /= total_prob
            
            logger.debug(f"Mutation bias for {agent_type}: {base_mutations}")
            return base_mutations
            
        except Exception as e:
            logger.error(f"Mutation bias calculation failed: {e}")
            return {'cooperation': 0.2, 'exploration': 0.2, 'resource_gathering': 0.2, 'adaptation': 0.2, 'efficiency': 0.2}
    
    def _classify_shift_type(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> str:
        """Classify the type of environment shift"""
        try:
            # Resource scarcity shift
            if 'resource_availability' in current_state and 'resource_availability' in previous_state:
                if current_state['resource_availability'] < previous_state['resource_availability'] * 0.7:
                    return 'resource_scarcity'
                elif current_state['resource_availability'] > previous_state['resource_availability'] * 1.3:
                    return 'resource_abundance'
            
            # Population pressure shift
            if 'agent_count' in current_state and 'agent_count' in previous_state:
                if current_state['agent_count'] > previous_state['agent_count'] * 1.5:
                    return 'population_boom'
                elif current_state['agent_count'] < previous_state['agent_count'] * 0.5:
                    return 'population_crash'
            
            # Fitness pressure shift
            if 'avg_fitness' in current_state and 'avg_fitness' in previous_state:
                if current_state['avg_fitness'] < previous_state['avg_fitness'] * 0.8:
                    return 'fitness_pressure'
                elif current_state['avg_fitness'] > previous_state['avg_fitness'] * 1.2:
                    return 'fitness_relaxation'
            
            return 'general_environmental_change'
            
        except Exception as e:
            logger.warning(f"Shift classification failed: {e}")
            return 'unknown_shift'
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get shift learning statistics"""
        try:
            if not self.adaptation_rewards:
                return {"no_data": True}
            
            # Calculate statistics
            total_adaptations = len(self.adaptation_rewards)
            successful_adaptations = sum(1 for r in self.adaptation_rewards if r['survival'])
            avg_reward = sum(r['reward'] for r in self.adaptation_rewards) / total_adaptations
            
            # Survival rate
            survival_rate = successful_adaptations / total_adaptations
            
            # Recent performance (last 10 adaptations)
            recent_rewards = self.adaptation_rewards[-10:]
            recent_survival_rate = sum(1 for r in recent_rewards if r['survival']) / len(recent_rewards)
            
            return {
                'total_adaptations': total_adaptations,
                'successful_adaptations': successful_adaptations,
                'survival_rate': survival_rate,
                'avg_reward': avg_reward,
                'recent_survival_rate': recent_survival_rate,
                'total_shifts_detected': len(self.shift_history),
                'learning_effectiveness': min(1.0, survival_rate * 1.2)  # Effectiveness metric
            }
            
        except Exception as e:
            logger.error(f"Learning stats calculation failed: {e}")
            return {"error": str(e)}

class EmergentBehaviorDetector:
    """
    Observer-approved emergent behavior detector with shift learning
    Detects emergent behaviors and applies RL-based adaptation learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.behavior_threshold = config.get('behavior_threshold', 0.5)
        self.detection_window = config.get('detection_window', 10)
        
        # Initialize shift learning RL
        self.shift_learning = ShiftLearningRL(config.get('shift_learning', {}))
        
        # Behavior tracking
        self.detected_behaviors = []
        self.behavior_patterns = {}
        
        logger.info("EmergentDetector initialized with shift learning capabilities")
    
    def detect_emergent_behaviors(self, agents: List[Any], environment: Dict[str, Any]) -> List[EmergentBehavior]:
        """Detect emergent behaviors in agent population"""
        try:
            behaviors = []
            
            # Detect cooperation clusters
            cooperation_behavior = self._detect_cooperation_clusters(agents)
            if cooperation_behavior:
                behaviors.append(cooperation_behavior)
            
            # Detect resource sharing networks
            sharing_behavior = self._detect_resource_sharing(agents)
            if sharing_behavior:
                behaviors.append(sharing_behavior)
            
            # Detect exploration coordination
            exploration_behavior = self._detect_exploration_coordination(agents)
            if exploration_behavior:
                behaviors.append(exploration_behavior)
            
            # Store detected behaviors
            self.detected_behaviors.extend(behaviors)
            
            if behaviors:
                logger.info(f"Detected {len(behaviors)} emergent behaviors")
            
            return behaviors
            
        except Exception as e:
            logger.error(f"Emergent behavior detection failed: {e}")
            return []
    
    def _detect_cooperation_clusters(self, agents: List[Any]) -> Optional[EmergentBehavior]:
        """Detect cooperation cluster formation"""
        try:
            if len(agents) < 3:
                return None
            
            # Calculate cooperation density
            cooperation_events = 0
            total_interactions = 0
            
            for agent in agents:
                if hasattr(agent, 'interactions'):
                    total_interactions += len(agent.interactions)
                    cooperation_events += sum(1 for interaction in agent.interactions if 'cooperation' in str(interaction))
            
            if total_interactions > 0:
                cooperation_density = cooperation_events / total_interactions
                
                if cooperation_density > self.behavior_threshold:
                    return EmergentBehavior(
                        behavior_id=f"cooperation_cluster_{int(time.time())}",
                        behavior_type="cooperation_cluster",
                        agents_involved=[agent.agent_id for agent in agents if hasattr(agent, 'agent_id')],
                        strength=cooperation_density,
                        timestamp=datetime.now(),
                        description=f"Cooperation cluster with {cooperation_density:.1%} cooperation rate",
                        fitness_impact=cooperation_density * 0.5
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Cooperation cluster detection failed: {e}")
            return None
    
    def _detect_resource_sharing(self, agents: List[Any]) -> Optional[EmergentBehavior]:
        """Detect resource sharing networks"""
        try:
            if len(agents) < 2:
                return None
            
            # Calculate resource sharing frequency
            sharing_events = 0
            total_agents = len(agents)
            
            for agent in agents:
                if hasattr(agent, 'resources') and hasattr(agent, 'interactions'):
                    # Check for resource sharing in interactions
                    for interaction in agent.interactions:
                        if 'resource' in str(interaction) and 'share' in str(interaction):
                            sharing_events += 1
            
            sharing_rate = sharing_events / max(1, total_agents)
            
            if sharing_rate > self.behavior_threshold:
                return EmergentBehavior(
                    behavior_id=f"resource_sharing_{int(time.time())}",
                    behavior_type="resource_sharing",
                    agents_involved=[agent.agent_id for agent in agents if hasattr(agent, 'agent_id')],
                    strength=sharing_rate,
                    timestamp=datetime.now(),
                    description=f"Resource sharing network with {sharing_rate:.2f} sharing rate",
                    fitness_impact=sharing_rate * 0.3
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Resource sharing detection failed: {e}")
            return None
    
    def _detect_exploration_coordination(self, agents: List[Any]) -> Optional[EmergentBehavior]:
        """Detect coordinated exploration patterns"""
        try:
            if len(agents) < 2:
                return None
            
            # Calculate exploration coordination
            exploration_agents = []
            for agent in agents:
                if hasattr(agent, 'agent_type') and 'explorer' in agent.agent_type:
                    exploration_agents.append(agent)
            
            if len(exploration_agents) >= 2:
                # Check for coordinated movement patterns
                coordination_score = len(exploration_agents) / len(agents)
                
                if coordination_score > self.behavior_threshold:
                    return EmergentBehavior(
                        behavior_id=f"exploration_coordination_{int(time.time())}",
                        behavior_type="exploration_coordination",
                        agents_involved=[agent.agent_id for agent in exploration_agents if hasattr(agent, 'agent_id')],
                        strength=coordination_score,
                        timestamp=datetime.now(),
                        description=f"Coordinated exploration with {len(exploration_agents)} explorers",
                        fitness_impact=coordination_score * 0.4
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Exploration coordination detection failed: {e}")
            return None
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get emergent behavior detection statistics"""
        try:
            if not self.detected_behaviors:
                return {"no_data": True}
            
            # Behavior type distribution
            behavior_types = {}
            total_strength = 0.0
            
            for behavior in self.detected_behaviors:
                behavior_type = behavior.behavior_type
                behavior_types[behavior_type] = behavior_types.get(behavior_type, 0) + 1
                total_strength += behavior.strength
            
            avg_strength = total_strength / len(self.detected_behaviors)
            
            # Include shift learning stats
            shift_stats = self.shift_learning.get_learning_stats()
            
            return {
                'total_behaviors_detected': len(self.detected_behaviors),
                'behavior_type_distribution': behavior_types,
                'avg_behavior_strength': avg_strength,
                'shift_learning_stats': shift_stats
            }
            
        except Exception as e:
            logger.error(f"Detection stats calculation failed: {e}")
            return {"error": str(e)}

    def log_mcp_audit_chain(
        self,
        agent_id: str,
        mcp_action: Dict[str, Any],
        outcome: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Observer-approved MCP audit chain logging
        Logs MCP calls with hashes and proofs for visualization
        """
        try:
            import hashlib
            import json

            # Generate audit hash
            audit_data = {
                'agent_id': agent_id,
                'mcp_action': mcp_action,
                'outcome': outcome,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }

            audit_hash = hashlib.sha256(
                json.dumps(audit_data, sort_keys=True).encode()
            ).hexdigest()[:16]  # Short hash for visualization

            # Create audit entry
            audit_entry = {
                'audit_id': f"mcp_{agent_id}_{audit_hash}",
                'agent_id': agent_id,
                'timestamp': datetime.now(),
                'mcp_action': mcp_action,
                'outcome': outcome,
                'context': context,
                'audit_hash': audit_hash,
                'appropriateness_score': context.get('context_appropriateness', 0.5),
                'success': outcome.get('success', False),
                'env_improvement': outcome.get('env_improvement', 0.0),
                'gaming_detected': self._detect_gaming_in_audit(mcp_action, outcome)
            }

            # Store audit entry
            if not hasattr(self, 'mcp_audit_log'):
                self.mcp_audit_log = []

            self.mcp_audit_log.append(audit_entry)

            # Limit audit log size
            if len(self.mcp_audit_log) > 1000:
                self.mcp_audit_log = self.mcp_audit_log[-1000:]

            logger.debug(f"MCP audit logged: {audit_entry['audit_id']}")
            return audit_entry['audit_id']

        except Exception as e:
            logger.error(f"MCP audit chain logging failed: {e}")
            return ""

    def _detect_gaming_in_audit(self, mcp_action: Dict[str, Any], outcome: Dict[str, Any]) -> bool:
        """Detect gaming patterns in MCP audit"""
        try:
            # Gaming indicators
            action_str = str(mcp_action).lower()
            gaming_keywords = ['dummy', 'test', 'fake', 'hack']

            # Check for gaming patterns
            if any(keyword in action_str for keyword in gaming_keywords):
                return True

            # Minimal compliance detection
            if (outcome.get('success', False) and
                outcome.get('env_improvement', 0) <= 0.001):
                return True

            return False

        except Exception as e:
            logger.warning(f"Gaming detection in audit failed: {e}")
            return False

    def generate_audit_heatmap_data(self, generations: int = 10) -> Dict[str, Any]:
        """
        Generate data for MCP audit heatmap visualization
        Shows usage vs generations with gaming attempts colored
        """
        try:
            if not hasattr(self, 'mcp_audit_log') or not self.mcp_audit_log:
                return {"no_data": True}

            # Group audits by generation (approximate from timestamp)
            generation_data = {}

            for i, audit in enumerate(self.mcp_audit_log[-generations*50:]):  # Last N generations worth
                # Approximate generation from index
                generation = (i // 50) + 1  # Assume 50 audits per generation

                if generation not in generation_data:
                    generation_data[generation] = {
                        'total_calls': 0,
                        'successful_calls': 0,
                        'gaming_attempts': 0,
                        'avg_appropriateness': 0.0,
                        'avg_improvement': 0.0,
                        'agents': set()
                    }

                gen_data = generation_data[generation]
                gen_data['total_calls'] += 1
                gen_data['agents'].add(audit['agent_id'])

                if audit['success']:
                    gen_data['successful_calls'] += 1

                if audit['gaming_detected']:
                    gen_data['gaming_attempts'] += 1

                gen_data['avg_appropriateness'] += audit['appropriateness_score']
                gen_data['avg_improvement'] += audit['env_improvement']

            # Calculate averages
            for gen, data in generation_data.items():
                total = data['total_calls']
                if total > 0:
                    data['avg_appropriateness'] /= total
                    data['avg_improvement'] /= total
                    data['success_rate'] = data['successful_calls'] / total
                    data['gaming_rate'] = data['gaming_attempts'] / total
                    data['unique_agents'] = len(data['agents'])

                # Remove set for JSON serialization
                del data['agents']

            # Prepare heatmap data
            heatmap_data = {
                'generations': list(generation_data.keys()),
                'total_calls': [data['total_calls'] for data in generation_data.values()],
                'success_rates': [data['success_rate'] for data in generation_data.values()],
                'gaming_rates': [data['gaming_rate'] for data in generation_data.values()],
                'appropriateness_scores': [data['avg_appropriateness'] for data in generation_data.values()],
                'generation_data': generation_data
            }

            logger.info(f"Generated audit heatmap data for {len(generation_data)} generations")
            return heatmap_data

        except Exception as e:
            logger.error(f"Audit heatmap data generation failed: {e}")
            return {"error": str(e)}

    def get_mcp_audit_summary(self) -> Dict[str, Any]:
        """Get summary of MCP audit data"""
        try:
            if not hasattr(self, 'mcp_audit_log') or not self.mcp_audit_log:
                return {"no_data": True}

            total_audits = len(self.mcp_audit_log)
            successful_audits = sum(1 for audit in self.mcp_audit_log if audit['success'])
            gaming_audits = sum(1 for audit in self.mcp_audit_log if audit['gaming_detected'])

            avg_appropriateness = sum(audit['appropriateness_score'] for audit in self.mcp_audit_log) / total_audits
            avg_improvement = sum(audit['env_improvement'] for audit in self.mcp_audit_log) / total_audits

            unique_agents = len(set(audit['agent_id'] for audit in self.mcp_audit_log))

            return {
                'total_mcp_audits': total_audits,
                'successful_audits': successful_audits,
                'gaming_audits': gaming_audits,
                'success_rate': successful_audits / total_audits,
                'gaming_rate': gaming_audits / total_audits,
                'avg_appropriateness_score': avg_appropriateness,
                'avg_environment_improvement': avg_improvement,
                'unique_agents_monitored': unique_agents,
                'enforcement_effectiveness': 1.0 - (gaming_audits / total_audits),
                'audit_coverage': 'comprehensive' if total_audits > 100 else 'moderate' if total_audits > 50 else 'limited'
            }

        except Exception as e:
            logger.error(f"MCP audit summary calculation failed: {e}")
            return {"error": str(e)}
