#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observer Clone System
Observer-approved autonomous monitoring agents spawned from DGM evolution
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from .models import ImprovementCandidate, ImprovementType
from .core.validator import DGMValidator

logger = logging.getLogger(__name__)

@dataclass
class ObserverMetrics:
    """Metrics tracked by observer clones"""
    timestamp: datetime
    system_health: float
    performance_score: float
    safety_compliance: float
    efficiency_rating: float
    anomaly_count: int
    recommendations: List[str]

class ObserverClone:
    """
    Observer-approved autonomous monitoring agent
    Spawned from DGM evolution to provide hands-off system monitoring
    """
    
    def __init__(self, clone_id: str, config: Dict[str, Any]):
        self.clone_id = clone_id
        self.config = config
        
        # Observer clone capabilities
        self.monitoring_enabled = True
        self.autonomous_actions = config.get('autonomous_actions', True)
        self.alert_threshold = config.get('alert_threshold', 0.7)
        self.intervention_threshold = config.get('intervention_threshold', 0.5)
        
        # Monitoring history
        self.metrics_history = []
        self.alerts_generated = []
        self.interventions_performed = []
        
        # Clone specialization
        self.specialization = config.get('specialization', 'general')
        self.monitoring_scope = config.get('monitoring_scope', ['performance', 'safety', 'efficiency'])
        
        logger.info(f"Observer clone {clone_id} initialized with specialization: {self.specialization}")
    
    async def monitor_system(self, system_state: Dict[str, Any]) -> ObserverMetrics:
        """Monitor system state and generate metrics"""
        try:
            monitoring_start = time.time()
            
            # Calculate system health
            system_health = self._calculate_system_health(system_state)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(system_state)
            
            # Calculate safety compliance
            safety_compliance = self._calculate_safety_compliance(system_state)
            
            # Calculate efficiency rating
            efficiency_rating = self._calculate_efficiency_rating(system_state)
            
            # Detect anomalies
            anomaly_count = self._detect_anomalies(system_state)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                system_health, performance_score, safety_compliance, efficiency_rating, anomaly_count
            )
            
            # Create metrics
            metrics = ObserverMetrics(
                timestamp=datetime.now(),
                system_health=system_health,
                performance_score=performance_score,
                safety_compliance=safety_compliance,
                efficiency_rating=efficiency_rating,
                anomaly_count=anomaly_count,
                recommendations=recommendations
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Check for alerts
            await self._check_alerts(metrics)
            
            # Check for interventions
            await self._check_interventions(metrics, system_state)
            
            monitoring_time = time.time() - monitoring_start
            logger.info(f"Observer clone {self.clone_id} monitoring completed in {monitoring_time:.3f}s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Observer clone {self.clone_id} monitoring failed: {e}")
            return ObserverMetrics(
                timestamp=datetime.now(),
                system_health=0.0,
                performance_score=0.0,
                safety_compliance=0.0,
                efficiency_rating=0.0,
                anomaly_count=999,
                recommendations=[f"Monitoring error: {str(e)}"]
            )
    
    def _calculate_system_health(self, system_state: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        try:
            health_factors = []
            
            # Check evolution success rate
            if 'evolution_results' in system_state:
                evolution_success = system_state['evolution_results'].get('success', False)
                health_factors.append(1.0 if evolution_success else 0.0)
            
            # Check simulation success rate
            if 'simulation_results' in system_state:
                sim_success = system_state['simulation_results'].get('simulation_success', False)
                health_factors.append(1.0 if sim_success else 0.0)
            
            # Check agent count
            if 'agent_count' in system_state:
                agent_count = system_state['agent_count']
                expected_count = system_state.get('expected_agent_count', 10)
                agent_health = min(1.0, agent_count / expected_count)
                health_factors.append(agent_health)
            
            # Check error rate
            if 'error_count' in system_state:
                error_count = system_state['error_count']
                error_health = max(0.0, 1.0 - (error_count * 0.1))
                health_factors.append(error_health)
            
            return sum(health_factors) / len(health_factors) if health_factors else 0.5
            
        except Exception as e:
            logger.warning(f"System health calculation failed: {e}")
            return 0.5
    
    def _calculate_performance_score(self, system_state: Dict[str, Any]) -> float:
        """Calculate system performance score"""
        try:
            performance_factors = []
            
            # Check fitness scores
            if 'best_fitness' in system_state:
                fitness = system_state['best_fitness']
                performance_factors.append(min(1.0, fitness / 2.0))  # Normalize to 200% target
            
            # Check generation efficiency
            if 'generations_completed' in system_state and 'target_generations' in system_state:
                efficiency = system_state['generations_completed'] / system_state['target_generations']
                performance_factors.append(min(1.0, efficiency))
            
            # Check cooperation events
            if 'cooperation_events' in system_state:
                cooperation = min(1.0, system_state['cooperation_events'] / 1000.0)
                performance_factors.append(cooperation)
            
            return sum(performance_factors) / len(performance_factors) if performance_factors else 0.5
            
        except Exception as e:
            logger.warning(f"Performance score calculation failed: {e}")
            return 0.5
    
    def _calculate_safety_compliance(self, system_state: Dict[str, Any]) -> float:
        """Calculate safety compliance score"""
        try:
            safety_factors = []
            
            # Check validation success rate
            if 'validation_results' in system_state:
                validation_success = system_state['validation_results'].get('success', False)
                safety_factors.append(1.0 if validation_success else 0.0)
            
            # Check safety scores
            if 'safety_score' in system_state:
                safety_score = system_state['safety_score']
                safety_factors.append(safety_score)
            
            # Check for safety violations
            if 'safety_violations' in system_state:
                violations = system_state['safety_violations']
                violation_penalty = max(0.0, 1.0 - (violations * 0.2))
                safety_factors.append(violation_penalty)
            
            return sum(safety_factors) / len(safety_factors) if safety_factors else 0.8
            
        except Exception as e:
            logger.warning(f"Safety compliance calculation failed: {e}")
            return 0.8
    
    def _calculate_efficiency_rating(self, system_state: Dict[str, Any]) -> float:
        """Calculate system efficiency rating"""
        try:
            efficiency_factors = []
            
            # Check runtime efficiency
            if 'runtime' in system_state and 'target_runtime' in system_state:
                runtime_efficiency = min(1.0, system_state['target_runtime'] / max(0.1, system_state['runtime']))
                efficiency_factors.append(runtime_efficiency)
            
            # Check resource utilization
            if 'resource_usage' in system_state:
                resource_efficiency = max(0.0, 1.0 - system_state['resource_usage'])
                efficiency_factors.append(resource_efficiency)
            
            # Check RL reward efficiency
            if 'rl_reward' in system_state:
                rl_efficiency = min(1.0, system_state['rl_reward'])
                efficiency_factors.append(rl_efficiency)
            
            return sum(efficiency_factors) / len(efficiency_factors) if efficiency_factors else 0.6
            
        except Exception as e:
            logger.warning(f"Efficiency rating calculation failed: {e}")
            return 0.6
    
    def _detect_anomalies(self, system_state: Dict[str, Any]) -> int:
        """Detect system anomalies"""
        try:
            anomalies = 0
            
            # Check for fitness stagnation
            if 'fitness_history' in system_state:
                fitness_history = system_state['fitness_history']
                if len(fitness_history) >= 3:
                    recent_variance = max(fitness_history[-3:]) - min(fitness_history[-3:])
                    if recent_variance < 0.01:
                        anomalies += 1
            
            # Check for excessive runtime
            if 'runtime' in system_state:
                if system_state['runtime'] > 300:  # 5 minutes
                    anomalies += 1
            
            # Check for low cooperation
            if 'cooperation_events' in system_state:
                if system_state['cooperation_events'] < 10:
                    anomalies += 1
            
            # Check for validation failures
            if 'validation_results' in system_state:
                if not system_state['validation_results'].get('success', True):
                    anomalies += 1
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return 0
    
    def _generate_recommendations(
        self, 
        system_health: float, 
        performance_score: float, 
        safety_compliance: float, 
        efficiency_rating: float, 
        anomaly_count: int
    ) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        try:
            if system_health < 0.7:
                recommendations.append("System health below threshold - investigate agent failures")
            
            if performance_score < 0.6:
                recommendations.append("Performance below target - consider parameter tuning")
            
            if safety_compliance < 0.8:
                recommendations.append("Safety compliance low - review validation thresholds")
            
            if efficiency_rating < 0.5:
                recommendations.append("Efficiency poor - optimize runtime and resource usage")
            
            if anomaly_count > 2:
                recommendations.append(f"Multiple anomalies detected ({anomaly_count}) - system review needed")
            
            if not recommendations:
                recommendations.append("System operating within normal parameters")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations due to error"]
    
    async def _check_alerts(self, metrics: ObserverMetrics):
        """Check if alerts should be generated"""
        try:
            alert_conditions = []
            
            if metrics.system_health < self.alert_threshold:
                alert_conditions.append(f"System health: {metrics.system_health:.2f}")
            
            if metrics.safety_compliance < self.alert_threshold:
                alert_conditions.append(f"Safety compliance: {metrics.safety_compliance:.2f}")
            
            if metrics.anomaly_count > 3:
                alert_conditions.append(f"Anomaly count: {metrics.anomaly_count}")
            
            if alert_conditions:
                alert = {
                    'timestamp': metrics.timestamp,
                    'clone_id': self.clone_id,
                    'conditions': alert_conditions,
                    'severity': 'high' if len(alert_conditions) > 2 else 'medium'
                }
                self.alerts_generated.append(alert)
                logger.warning(f"Observer clone {self.clone_id} generated alert: {alert_conditions}")
            
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
    
    async def _check_interventions(self, metrics: ObserverMetrics, system_state: Dict[str, Any]):
        """Check if autonomous interventions should be performed"""
        try:
            if not self.autonomous_actions:
                return
            
            intervention_needed = False
            intervention_type = None
            
            # Critical system health
            if metrics.system_health < self.intervention_threshold:
                intervention_needed = True
                intervention_type = "system_restart"
            
            # Critical safety compliance
            elif metrics.safety_compliance < self.intervention_threshold:
                intervention_needed = True
                intervention_type = "safety_adjustment"
            
            # Excessive anomalies
            elif metrics.anomaly_count > 5:
                intervention_needed = True
                intervention_type = "anomaly_mitigation"
            
            if intervention_needed:
                intervention = {
                    'timestamp': metrics.timestamp,
                    'clone_id': self.clone_id,
                    'type': intervention_type,
                    'metrics': metrics,
                    'system_state': system_state
                }
                self.interventions_performed.append(intervention)
                logger.info(f"Observer clone {self.clone_id} performed intervention: {intervention_type}")
            
        except Exception as e:
            logger.error(f"Intervention checking failed: {e}")
    
    def get_clone_status(self) -> Dict[str, Any]:
        """Get current clone status and metrics"""
        try:
            recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
            
            return {
                'clone_id': self.clone_id,
                'specialization': self.specialization,
                'monitoring_enabled': self.monitoring_enabled,
                'total_monitoring_cycles': len(self.metrics_history),
                'alerts_generated': len(self.alerts_generated),
                'interventions_performed': len(self.interventions_performed),
                'recent_metrics': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'system_health': m.system_health,
                        'performance_score': m.performance_score,
                        'safety_compliance': m.safety_compliance,
                        'efficiency_rating': m.efficiency_rating,
                        'anomaly_count': m.anomaly_count
                    } for m in recent_metrics
                ],
                'latest_recommendations': recent_metrics[-1].recommendations if recent_metrics else []
            }
            
        except Exception as e:
            logger.error(f"Clone status retrieval failed: {e}")
            return {'error': str(e)}

class ObserverCloneManager:
    """Manager for multiple observer clones"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clones = {}
        self.clone_count = config.get('clone_count', 3)
        
    async def spawn_observer_clones(self) -> bool:
        """Spawn observer clones from DGM evolution"""
        try:
            logger.info(f"Spawning {self.clone_count} observer clones")
            
            clone_specializations = ['performance', 'safety', 'efficiency', 'general']
            
            for i in range(self.clone_count):
                clone_id = f"observer_clone_{i}"
                specialization = clone_specializations[i % len(clone_specializations)]
                
                clone_config = {
                    'specialization': specialization,
                    'autonomous_actions': True,
                    'alert_threshold': 0.7,
                    'intervention_threshold': 0.5,
                    'monitoring_scope': ['performance', 'safety', 'efficiency']
                }
                
                clone = ObserverClone(clone_id, clone_config)
                self.clones[clone_id] = clone
                
                logger.info(f"Spawned observer clone: {clone_id} ({specialization})")
            
            return True
            
        except Exception as e:
            logger.error(f"Observer clone spawning failed: {e}")
            return False
    
    async def monitor_with_clones(self, system_state: Dict[str, Any]) -> Dict[str, ObserverMetrics]:
        """Monitor system using all observer clones"""
        try:
            clone_results = {}
            
            for clone_id, clone in self.clones.items():
                metrics = await clone.monitor_system(system_state)
                clone_results[clone_id] = metrics
            
            return clone_results
            
        except Exception as e:
            logger.error(f"Clone monitoring failed: {e}")
            return {}
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get manager and all clone statuses"""
        try:
            return {
                'total_clones': len(self.clones),
                'active_clones': sum(1 for clone in self.clones.values() if clone.monitoring_enabled),
                'clone_statuses': {clone_id: clone.get_clone_status() for clone_id, clone in self.clones.items()}
            }
            
        except Exception as e:
            logger.error(f"Manager status retrieval failed: {e}")
            return {'error': str(e)}
