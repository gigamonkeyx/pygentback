#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP-Seeded Environment
Observer-approved environment with MCP context-aware auto-bias for directed learning
"""

import logging
import random
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MCPBias:
    """MCP-based bias configuration for environment seeding"""
    bias_type: str
    bias_strength: float
    target_behavior: str
    appropriateness_threshold: float
    success_weight: float
    context_weight: float

class MCPSeededEnvironment:
    """
    Observer-approved MCP-seeded environment
    Auto-biases environment based on MCP audit patterns for directed learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # MCP bias configuration
        self.mcp_weight_base = config.get('mcp_weight_base', 0.5)
        self.appropriateness_threshold = config.get('appropriateness_threshold', 0.7)
        self.auto_bias_enabled = config.get('auto_bias_enabled', True)
        
        # Environment parameters
        self.resource_distribution = config.get('resource_distribution', 'uniform')
        self.cooperation_incentive = config.get('cooperation_incentive', 0.3)
        self.exploration_reward = config.get('exploration_reward', 0.2)
        
        # MCP bias tracking
        self.active_biases = []
        self.bias_history = []
        self.mcp_audit_data = []
        
        # Environment state
        self.generation = 0
        self.environment_state = {
            'resource_availability': 0.5,
            'cooperation_density': 0.3,
            'exploration_coverage': 0.2,
            'mcp_appropriateness': 0.5
        }
        
        logger.info("MCP-seeded environment initialized with auto-bias system")
    
    def seed_from_mcp_audits(self, audit_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Observer-approved MCP audit seeding
        Seeds environment bias based on MCP audit patterns
        """
        try:
            if not audit_data:
                logger.warning("No MCP audit data provided for seeding")
                return {"no_seeding": True}
            
            # Store audit data
            self.mcp_audit_data = audit_data
            
            # Analyze audit patterns
            audit_analysis = self._analyze_mcp_patterns(audit_data)
            
            # Generate bias recommendations
            bias_recommendations = self._generate_bias_recommendations(audit_analysis)
            
            # Apply auto-bias if enabled
            if self.auto_bias_enabled:
                applied_biases = self._apply_auto_bias(bias_recommendations)
            else:
                applied_biases = []
            
            seeding_result = {
                'audit_data_count': len(audit_data),
                'audit_analysis': audit_analysis,
                'bias_recommendations': bias_recommendations,
                'applied_biases': applied_biases,
                'auto_bias_enabled': self.auto_bias_enabled,
                'environment_state': self.environment_state.copy()
            }
            
            logger.info(f"MCP seeding complete: {len(applied_biases)} biases applied")
            return seeding_result
            
        except Exception as e:
            logger.error(f"MCP audit seeding failed: {e}")
            return {"error": str(e)}
    
    def _analyze_mcp_patterns(self, audit_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze MCP audit patterns for bias generation"""
        try:
            if not audit_data:
                return {"no_data": True}
            
            # Calculate pattern metrics
            total_audits = len(audit_data)
            successful_audits = sum(1 for audit in audit_data if audit.get('success', False))
            gaming_audits = sum(1 for audit in audit_data if audit.get('gaming_detected', False))
            
            # Appropriateness analysis
            appropriateness_scores = [audit.get('appropriateness_score', 0.5) for audit in audit_data]
            avg_appropriateness = sum(appropriateness_scores) / len(appropriateness_scores)
            high_appropriateness_count = sum(1 for score in appropriateness_scores if score >= self.appropriateness_threshold)
            
            # Environment improvement analysis
            improvement_scores = [audit.get('env_improvement', 0.0) for audit in audit_data]
            avg_improvement = sum(improvement_scores) / len(improvement_scores)
            positive_improvement_count = sum(1 for score in improvement_scores if score > 0)
            
            # Agent behavior analysis
            agent_ids = list(set(audit.get('agent_id', 'unknown') for audit in audit_data))
            agent_performance = {}
            
            for agent_id in agent_ids:
                agent_audits = [audit for audit in audit_data if audit.get('agent_id') == agent_id]
                if agent_audits:
                    agent_success_rate = sum(1 for audit in agent_audits if audit.get('success', False)) / len(agent_audits)
                    agent_appropriateness = sum(audit.get('appropriateness_score', 0.5) for audit in agent_audits) / len(agent_audits)
                    agent_performance[agent_id] = {
                        'success_rate': agent_success_rate,
                        'appropriateness': agent_appropriateness,
                        'audit_count': len(agent_audits)
                    }
            
            # Pattern classification
            pattern_classification = self._classify_mcp_patterns(
                avg_appropriateness, avg_improvement, gaming_audits / total_audits
            )
            
            return {
                'total_audits': total_audits,
                'success_rate': successful_audits / total_audits,
                'gaming_rate': gaming_audits / total_audits,
                'avg_appropriateness': avg_appropriateness,
                'high_appropriateness_rate': high_appropriateness_count / total_audits,
                'avg_environment_improvement': avg_improvement,
                'positive_improvement_rate': positive_improvement_count / total_audits,
                'unique_agents': len(agent_ids),
                'agent_performance': agent_performance,
                'pattern_classification': pattern_classification
            }
            
        except Exception as e:
            logger.error(f"MCP pattern analysis failed: {e}")
            return {"error": str(e)}
    
    def _classify_mcp_patterns(self, avg_appropriateness: float, avg_improvement: float, gaming_rate: float) -> str:
        """Classify MCP usage patterns for bias recommendations"""
        try:
            # Pattern classification logic
            if gaming_rate > 0.2:
                return "high_gaming_pattern"
            elif avg_appropriateness >= 0.8 and avg_improvement > 0.1:
                return "optimal_usage_pattern"
            elif avg_appropriateness >= 0.6 and avg_improvement > 0.05:
                return "good_usage_pattern"
            elif avg_appropriateness < 0.4 or avg_improvement <= 0:
                return "poor_usage_pattern"
            else:
                return "moderate_usage_pattern"
                
        except Exception as e:
            logger.warning(f"MCP pattern classification failed: {e}")
            return "unknown_pattern"
    
    def _generate_bias_recommendations(self, audit_analysis: Dict[str, Any]) -> List[MCPBias]:
        """Generate bias recommendations based on audit analysis"""
        try:
            if audit_analysis.get('error') or audit_analysis.get('no_data'):
                return []
            
            recommendations = []
            pattern = audit_analysis.get('pattern_classification', 'unknown_pattern')
            avg_appropriateness = audit_analysis.get('avg_appropriateness', 0.5)
            gaming_rate = audit_analysis.get('gaming_rate', 0.0)
            
            # High appropriateness bias
            if avg_appropriateness >= self.appropriateness_threshold:
                mcp_weight = min(0.9, self.mcp_weight_base + (avg_appropriateness - self.appropriateness_threshold))
                recommendations.append(MCPBias(
                    bias_type="high_appropriateness_amplification",
                    bias_strength=mcp_weight,
                    target_behavior="mcp_smart_usage",
                    appropriateness_threshold=self.appropriateness_threshold,
                    success_weight=0.8,
                    context_weight=0.9
                ))
            
            # Anti-gaming bias
            if gaming_rate > 0.1:
                anti_gaming_strength = min(0.8, gaming_rate * 2.0)
                recommendations.append(MCPBias(
                    bias_type="anti_gaming_enforcement",
                    bias_strength=anti_gaming_strength,
                    target_behavior="gaming_resistance",
                    appropriateness_threshold=0.8,
                    success_weight=0.9,
                    context_weight=0.95
                ))
            
            # Context awareness bias
            if pattern in ["optimal_usage_pattern", "good_usage_pattern"]:
                context_strength = 0.7 if pattern == "optimal_usage_pattern" else 0.5
                recommendations.append(MCPBias(
                    bias_type="context_awareness_enhancement",
                    bias_strength=context_strength,
                    target_behavior="context_smart_decisions",
                    appropriateness_threshold=0.6,
                    success_weight=0.7,
                    context_weight=0.8
                ))
            
            # Improvement focus bias
            avg_improvement = audit_analysis.get('avg_environment_improvement', 0.0)
            if avg_improvement > 0.05:
                improvement_strength = min(0.8, avg_improvement * 5.0)
                recommendations.append(MCPBias(
                    bias_type="improvement_focus_amplification",
                    bias_strength=improvement_strength,
                    target_behavior="environment_improvement",
                    appropriateness_threshold=0.5,
                    success_weight=0.6,
                    context_weight=0.7
                ))
            
            logger.info(f"Generated {len(recommendations)} bias recommendations for pattern: {pattern}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Bias recommendation generation failed: {e}")
            return []
    
    def _apply_auto_bias(self, bias_recommendations: List[MCPBias]) -> List[Dict[str, Any]]:
        """Apply auto-bias based on recommendations"""
        try:
            applied_biases = []
            
            for bias in bias_recommendations:
                # Apply bias to environment state
                bias_application = self._apply_single_bias(bias)
                
                if bias_application.get('success', False):
                    applied_biases.append(bias_application)
                    self.active_biases.append(bias)
            
            # Update environment state based on applied biases
            self._update_environment_state()
            
            logger.info(f"Applied {len(applied_biases)} auto-biases to environment")
            return applied_biases
            
        except Exception as e:
            logger.error(f"Auto-bias application failed: {e}")
            return []
    
    def _apply_single_bias(self, bias: MCPBias) -> Dict[str, Any]:
        """Apply a single bias to the environment"""
        try:
            bias_application = {
                'bias_type': bias.bias_type,
                'bias_strength': bias.bias_strength,
                'target_behavior': bias.target_behavior,
                'application_timestamp': datetime.now(),
                'success': False
            }
            
            # Apply bias based on type
            if bias.bias_type == "high_appropriateness_amplification":
                # Increase cooperation incentive and resource availability
                self.cooperation_incentive = min(0.9, self.cooperation_incentive + bias.bias_strength * 0.2)
                self.environment_state['cooperation_density'] = min(0.9, self.environment_state['cooperation_density'] + bias.bias_strength * 0.1)
                bias_application['effects'] = ['increased_cooperation_incentive', 'enhanced_cooperation_density']
                
            elif bias.bias_type == "anti_gaming_enforcement":
                # Increase exploration requirements and reduce easy rewards
                self.exploration_reward = max(0.1, self.exploration_reward - bias.bias_strength * 0.1)
                self.environment_state['resource_availability'] = max(0.3, self.environment_state['resource_availability'] - bias.bias_strength * 0.1)
                bias_application['effects'] = ['reduced_easy_rewards', 'increased_exploration_requirements']
                
            elif bias.bias_type == "context_awareness_enhancement":
                # Increase environmental complexity and context requirements
                self.environment_state['mcp_appropriateness'] = min(0.9, self.environment_state['mcp_appropriateness'] + bias.bias_strength * 0.2)
                bias_application['effects'] = ['enhanced_context_requirements', 'increased_appropriateness_focus']
                
            elif bias.bias_type == "improvement_focus_amplification":
                # Increase rewards for environment improvement
                self.exploration_reward = min(0.8, self.exploration_reward + bias.bias_strength * 0.15)
                self.environment_state['exploration_coverage'] = min(0.8, self.environment_state['exploration_coverage'] + bias.bias_strength * 0.1)
                bias_application['effects'] = ['increased_improvement_rewards', 'enhanced_exploration_coverage']
            
            bias_application['success'] = True
            bias_application['environment_state_after'] = self.environment_state.copy()
            
            return bias_application
            
        except Exception as e:
            logger.error(f"Single bias application failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _update_environment_state(self):
        """Update environment state based on active biases"""
        try:
            # Calculate cumulative bias effects
            total_appropriateness_bias = sum(bias.context_weight for bias in self.active_biases if bias.bias_type.endswith('_enhancement'))
            total_anti_gaming_bias = sum(bias.bias_strength for bias in self.active_biases if 'anti_gaming' in bias.bias_type)
            
            # Update MCP appropriateness based on biases
            self.environment_state['mcp_appropriateness'] = min(0.95, 
                self.environment_state['mcp_appropriateness'] + total_appropriateness_bias * 0.1)
            
            # Update other state parameters
            if total_anti_gaming_bias > 0:
                self.environment_state['resource_availability'] = max(0.2, 
                    self.environment_state['resource_availability'] - total_anti_gaming_bias * 0.05)
            
            logger.debug(f"Environment state updated with {len(self.active_biases)} active biases")
            
        except Exception as e:
            logger.error(f"Environment state update failed: {e}")
    
    def test_directed_growth(self, test_behaviors: List[str]) -> Dict[str, Any]:
        """Test directed growth with MCP-seeded biases"""
        try:
            test_results = {}
            
            for behavior in test_behaviors:
                # Simulate behavior testing
                behavior_score = self._simulate_behavior_test(behavior)
                
                test_results[behavior] = {
                    'behavior_score': behavior_score,
                    'target_threshold': 0.88,
                    'meets_threshold': behavior_score >= 0.88,
                    'bias_influence': self._calculate_bias_influence(behavior)
                }
            
            # Calculate overall directed growth effectiveness
            successful_behaviors = sum(1 for result in test_results.values() if result['meets_threshold'])
            growth_effectiveness = successful_behaviors / len(test_behaviors) if test_behaviors else 0.0
            
            return {
                'test_behaviors': test_behaviors,
                'behavior_test_results': test_results,
                'successful_behaviors': successful_behaviors,
                'growth_effectiveness': growth_effectiveness,
                'target_effectiveness': 0.88,
                'directed_growth_working': growth_effectiveness >= 0.88
            }
            
        except Exception as e:
            logger.error(f"Directed growth testing failed: {e}")
            return {"error": str(e)}
    
    def _simulate_behavior_test(self, behavior: str) -> float:
        """Simulate testing a specific behavior"""
        try:
            # Base behavior score
            base_score = random.uniform(0.5, 0.8)
            
            # Apply bias influences
            bias_boost = 0.0
            for bias in self.active_biases:
                if behavior.lower() in bias.target_behavior.lower():
                    bias_boost += bias.bias_strength * 0.2
            
            # Apply environment state influence
            env_influence = self.environment_state.get('mcp_appropriateness', 0.5) * 0.1
            
            # Calculate final score
            final_score = min(0.95, base_score + bias_boost + env_influence)
            
            return final_score
            
        except Exception as e:
            logger.warning(f"Behavior test simulation failed: {e}")
            return 0.5
    
    def _calculate_bias_influence(self, behavior: str) -> float:
        """Calculate bias influence on a specific behavior"""
        try:
            total_influence = 0.0
            
            for bias in self.active_biases:
                if behavior.lower() in bias.target_behavior.lower():
                    total_influence += bias.bias_strength
            
            return min(1.0, total_influence)
            
        except Exception as e:
            logger.warning(f"Bias influence calculation failed: {e}")
            return 0.0
    
    def get_seeding_stats(self) -> Dict[str, Any]:
        """Get MCP seeding statistics"""
        try:
            return {
                'active_biases': len(self.active_biases),
                'bias_history': len(self.bias_history),
                'mcp_audit_data_count': len(self.mcp_audit_data),
                'auto_bias_enabled': self.auto_bias_enabled,
                'current_generation': self.generation,
                'environment_state': self.environment_state.copy(),
                'mcp_weight_base': self.mcp_weight_base,
                'appropriateness_threshold': self.appropriateness_threshold
            }
            
        except Exception as e:
            logger.error(f"Seeding stats calculation failed: {e}")
            return {"error": str(e)}
