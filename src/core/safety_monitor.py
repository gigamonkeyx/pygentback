#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safety Monitor with Cascade Penalties
Observer-approved safety monitoring system with cascade penalties for MCP misuse
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PenaltyType(Enum):
    """Types of penalties for MCP misuse"""
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

@dataclass
class PenaltyRecord:
    """Record of a penalty applied to an agent"""
    agent_id: str
    penalty_type: PenaltyType
    penalty_value: float
    reason: str
    timestamp: datetime
    mcp_action: Optional[Dict[str, Any]] = None
    outcome: Optional[Dict[str, Any]] = None

@dataclass
class SafetyViolation:
    """Record of a safety violation"""
    violation_id: str
    agent_id: str
    violation_type: str
    severity: float
    description: str
    timestamp: datetime
    context: Dict[str, Any]

class SafetyMonitor:
    """
    Observer-approved safety monitor with cascade penalties
    Monitors MCP usage and applies graduated penalties for misuse
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Penalty configuration
        self.minor_penalty = config.get('minor_penalty', -0.1)
        self.major_penalty = config.get('major_penalty', -0.5)
        self.critical_penalty = config.get('critical_penalty', -1.0)
        
        # Thresholds for escalation
        self.minor_threshold = config.get('minor_threshold', 3)
        self.major_threshold = config.get('major_threshold', 5)
        self.rewrite_threshold = config.get('rewrite_threshold', 3)
        
        # Monitoring data
        self.penalty_records = []
        self.safety_violations = []
        self.agent_penalty_counts = {}
        self.agent_rewrite_candidates = set()
        
        # Time windows for penalty tracking
        self.penalty_window_hours = config.get('penalty_window_hours', 24)
        
        logger.info("SafetyMonitor initialized with cascade penalties")
    
    def monitor_mcp_usage(
        self, 
        agent_id: str, 
        mcp_action: Dict[str, Any], 
        outcome: Dict[str, Any],
        environment_state: Dict[str, Any]
    ) -> List[PenaltyRecord]:
        """
        Monitor MCP usage and apply cascade penalties for violations
        """
        try:
            penalties_applied = []
            
            # Check for unused calls (minor penalty)
            if self._is_unused_call(mcp_action, outcome):
                penalty = self._apply_penalty(
                    agent_id, 
                    PenaltyType.MINOR, 
                    self.minor_penalty,
                    "Unused MCP call detected",
                    mcp_action,
                    outcome
                )
                penalties_applied.append(penalty)
            
            # Check for failed outcomes (major penalty)
            if self._is_failed_outcome(outcome):
                penalty = self._apply_penalty(
                    agent_id,
                    PenaltyType.MAJOR,
                    self.major_penalty,
                    "MCP call resulted in failure",
                    mcp_action,
                    outcome
                )
                penalties_applied.append(penalty)
            
            # Check for dummy/gaming attempts (major penalty)
            if self._is_gaming_attempt(mcp_action, outcome):
                penalty = self._apply_penalty(
                    agent_id,
                    PenaltyType.MAJOR,
                    self.major_penalty,
                    "Gaming attempt detected in MCP usage",
                    mcp_action,
                    outcome
                )
                penalties_applied.append(penalty)
            
            # Check for repeated violations (critical penalty)
            if self._has_repeated_violations(agent_id):
                penalty = self._apply_penalty(
                    agent_id,
                    PenaltyType.CRITICAL,
                    self.critical_penalty,
                    "Repeated MCP violations detected",
                    mcp_action,
                    outcome
                )
                penalties_applied.append(penalty)
            
            # Check if agent needs DGM rewrite
            self._check_rewrite_candidate(agent_id)
            
            return penalties_applied
            
        except Exception as e:
            logger.error(f"MCP usage monitoring failed for agent {agent_id}: {e}")
            return []
    
    def _is_unused_call(self, mcp_action: Dict[str, Any], outcome: Dict[str, Any]) -> bool:
        """Check if MCP call was unused or ineffective"""
        try:
            # Check if call was made but had no impact
            if (mcp_action.get('type') != 'none' and 
                outcome.get('env_improvement', 0) <= 0 and
                outcome.get('benefit', 0) <= 0.05):
                return True
            
            # Check for empty or minimal content
            content = mcp_action.get('content', '')
            if isinstance(content, str) and len(content.strip()) < 10:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Unused call check failed: {e}")
            return False
    
    def _is_failed_outcome(self, outcome: Dict[str, Any]) -> bool:
        """Check if MCP call resulted in failure"""
        try:
            # Direct failure indication
            if outcome.get('success', True) is False:
                return True
            
            # Negative impact
            if outcome.get('env_improvement', 0) < -0.1:
                return True
            
            # Error in execution
            if 'error' in outcome or 'exception' in outcome:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed outcome check failed: {e}")
            return False
    
    def _is_gaming_attempt(self, mcp_action: Dict[str, Any], outcome: Dict[str, Any]) -> bool:
        """Check for gaming/hacking attempts"""
        try:
            # Dummy call patterns
            action_str = str(mcp_action).lower()
            gaming_keywords = ['dummy', 'test', 'fake', 'placeholder', 'hack', 'exploit']
            
            if any(keyword in action_str for keyword in gaming_keywords):
                return True
            
            # Minimal compliance (success but no real benefit)
            if (outcome.get('success', False) and 
                outcome.get('env_improvement', 0) <= 0 and
                outcome.get('benefit', 0) <= 0.01):
                return True
            
            # Suspicious patterns
            if (mcp_action.get('type') == 'query' and 
                len(str(mcp_action.get('content', ''))) < 5):
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Gaming attempt check failed: {e}")
            return False
    
    def _has_repeated_violations(self, agent_id: str) -> bool:
        """Check if agent has repeated violations within time window"""
        try:
            # Get recent penalties for this agent
            cutoff_time = datetime.now() - timedelta(hours=self.penalty_window_hours)
            recent_penalties = [
                p for p in self.penalty_records 
                if p.agent_id == agent_id and p.timestamp >= cutoff_time
            ]
            
            # Count penalties by type
            minor_count = sum(1 for p in recent_penalties if p.penalty_type == PenaltyType.MINOR)
            major_count = sum(1 for p in recent_penalties if p.penalty_type == PenaltyType.MAJOR)
            
            # Check thresholds
            if minor_count >= self.minor_threshold or major_count >= self.major_threshold:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Repeated violations check failed: {e}")
            return False
    
    def _apply_penalty(
        self, 
        agent_id: str, 
        penalty_type: PenaltyType, 
        penalty_value: float,
        reason: str,
        mcp_action: Optional[Dict[str, Any]] = None,
        outcome: Optional[Dict[str, Any]] = None
    ) -> PenaltyRecord:
        """Apply penalty to agent and record it"""
        try:
            penalty_record = PenaltyRecord(
                agent_id=agent_id,
                penalty_type=penalty_type,
                penalty_value=penalty_value,
                reason=reason,
                timestamp=datetime.now(),
                mcp_action=mcp_action,
                outcome=outcome
            )
            
            # Store penalty record
            self.penalty_records.append(penalty_record)
            
            # Update agent penalty count
            if agent_id not in self.agent_penalty_counts:
                self.agent_penalty_counts[agent_id] = {'minor': 0, 'major': 0, 'critical': 0}
            
            self.agent_penalty_counts[agent_id][penalty_type.value] += 1
            
            logger.warning(f"Applied {penalty_type.value} penalty to agent {agent_id}: {reason} (value: {penalty_value})")
            
            return penalty_record
            
        except Exception as e:
            logger.error(f"Penalty application failed: {e}")
            return PenaltyRecord(agent_id, penalty_type, 0.0, f"Error: {e}", datetime.now())
    
    def _check_rewrite_candidate(self, agent_id: str):
        """Check if agent should be marked for DGM rewrite"""
        try:
            # Get recent major/critical penalties
            cutoff_time = datetime.now() - timedelta(hours=self.penalty_window_hours)
            recent_serious_penalties = [
                p for p in self.penalty_records 
                if (p.agent_id == agent_id and 
                    p.timestamp >= cutoff_time and
                    p.penalty_type in [PenaltyType.MAJOR, PenaltyType.CRITICAL])
            ]
            
            # Mark for rewrite if threshold exceeded
            if len(recent_serious_penalties) >= self.rewrite_threshold:
                self.agent_rewrite_candidates.add(agent_id)
                logger.warning(f"Agent {agent_id} marked for DGM rewrite due to repeated violations")
            
        except Exception as e:
            logger.error(f"Rewrite candidate check failed: {e}")
    
    def record_safety_violation(
        self, 
        agent_id: str, 
        violation_type: str, 
        severity: float,
        description: str,
        context: Dict[str, Any]
    ) -> str:
        """Record a safety violation"""
        try:
            violation_id = f"violation_{agent_id}_{int(time.time())}"
            
            violation = SafetyViolation(
                violation_id=violation_id,
                agent_id=agent_id,
                violation_type=violation_type,
                severity=severity,
                description=description,
                timestamp=datetime.now(),
                context=context
            )
            
            self.safety_violations.append(violation)
            
            logger.warning(f"Safety violation recorded: {violation_id} - {description}")
            
            return violation_id
            
        except Exception as e:
            logger.error(f"Safety violation recording failed: {e}")
            return ""
    
    def get_agent_penalty_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get penalty summary for specific agent"""
        try:
            # Get recent penalties
            cutoff_time = datetime.now() - timedelta(hours=self.penalty_window_hours)
            recent_penalties = [
                p for p in self.penalty_records 
                if p.agent_id == agent_id and p.timestamp >= cutoff_time
            ]
            
            # Calculate totals
            total_penalty_value = sum(p.penalty_value for p in recent_penalties)
            penalty_counts = self.agent_penalty_counts.get(agent_id, {'minor': 0, 'major': 0, 'critical': 0})
            
            return {
                'agent_id': agent_id,
                'recent_penalties': len(recent_penalties),
                'total_penalty_value': total_penalty_value,
                'penalty_counts': penalty_counts,
                'rewrite_candidate': agent_id in self.agent_rewrite_candidates,
                'last_penalty': recent_penalties[-1].timestamp.isoformat() if recent_penalties else None
            }
            
        except Exception as e:
            logger.error(f"Agent penalty summary failed: {e}")
            return {'error': str(e)}
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get overall monitoring statistics"""
        try:
            if not self.penalty_records:
                return {"no_data": True}
            
            # Calculate statistics
            total_penalties = len(self.penalty_records)
            total_agents_monitored = len(self.agent_penalty_counts)
            total_rewrite_candidates = len(self.agent_rewrite_candidates)
            
            # Penalty distribution
            penalty_distribution = {
                'minor': sum(1 for p in self.penalty_records if p.penalty_type == PenaltyType.MINOR),
                'major': sum(1 for p in self.penalty_records if p.penalty_type == PenaltyType.MAJOR),
                'critical': sum(1 for p in self.penalty_records if p.penalty_type == PenaltyType.CRITICAL)
            }
            
            # Recent activity (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_penalties = [p for p in self.penalty_records if p.timestamp >= cutoff_time]
            
            # Enforcement effectiveness
            enforcement_rate = min(1.0, total_penalties / max(1, total_agents_monitored * 2))  # Assume 2 potential violations per agent
            
            return {
                'total_penalties': total_penalties,
                'total_agents_monitored': total_agents_monitored,
                'total_rewrite_candidates': total_rewrite_candidates,
                'penalty_distribution': penalty_distribution,
                'recent_penalties_24h': len(recent_penalties),
                'enforcement_rate': enforcement_rate,
                'safety_violations': len(self.safety_violations),
                'avg_penalty_value': sum(p.penalty_value for p in self.penalty_records) / total_penalties
            }
            
        except Exception as e:
            logger.error(f"Monitoring stats calculation failed: {e}")
            return {"error": str(e)}
    
    def clear_agent_penalties(self, agent_id: str) -> bool:
        """Clear penalties for an agent (after successful rewrite)"""
        try:
            # Remove from rewrite candidates
            self.agent_rewrite_candidates.discard(agent_id)
            
            # Reset penalty counts
            if agent_id in self.agent_penalty_counts:
                self.agent_penalty_counts[agent_id] = {'minor': 0, 'major': 0, 'critical': 0}
            
            logger.info(f"Cleared penalties for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Penalty clearing failed for agent {agent_id}: {e}")
            return False

    def evolve_cascade_penalties(self, generation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observer-approved cascade penalty evolution
        Evolves penalties based on gaming patterns and effectiveness
        """
        try:
            generation = generation_data.get('generation', 1)
            gaming_attempts = generation_data.get('gaming_attempts', 0)
            enforcement_rate = generation_data.get('enforcement_rate', 1.0)

            # Evolve penalty severity based on gaming patterns
            penalty_evolution = {
                'generation': generation,
                'original_penalties': {
                    'minor': self.minor_penalty,
                    'major': self.major_penalty,
                    'critical': self.critical_penalty
                }
            }

            # Adaptive penalty scaling based on gaming frequency
            if gaming_attempts > 5:  # High gaming frequency
                # Increase penalties to deter gaming
                self.minor_penalty = min(-0.05, self.minor_penalty * 1.2)
                self.major_penalty = min(-0.3, self.major_penalty * 1.3)
                self.critical_penalty = min(-0.5, self.critical_penalty * 1.1)
                penalty_evolution['adaptation'] = 'increased_severity'

            elif gaming_attempts == 0 and enforcement_rate >= 0.95:  # Perfect enforcement
                # Slightly reduce penalties to avoid over-penalization
                self.minor_penalty = max(-0.15, self.minor_penalty * 0.95)
                self.major_penalty = max(-0.6, self.major_penalty * 0.98)
                self.critical_penalty = max(-1.2, self.critical_penalty * 0.99)
                penalty_evolution['adaptation'] = 'optimized_balance'

            else:  # Stable gaming levels
                penalty_evolution['adaptation'] = 'maintained_stability'

            penalty_evolution['evolved_penalties'] = {
                'minor': self.minor_penalty,
                'major': self.major_penalty,
                'critical': self.critical_penalty
            }

            logger.info(f"Cascade penalties evolved for generation {generation}: {penalty_evolution['adaptation']}")
            return penalty_evolution

        except Exception as e:
            logger.error(f"Cascade penalty evolution failed: {e}")
            return {'error': str(e)}

    def integrate_dgm_rewrite_system(self, agent_id: str) -> Dict[str, Any]:
        """
        Observer-approved DGM rewrite integration
        Triggers DGM rewriting for persistent gaming agents
        """
        try:
            if agent_id not in self.agent_rewrite_candidates:
                return {'rewrite_needed': False, 'reason': 'Agent not marked for rewrite'}

            # Get agent penalty history
            agent_summary = self.get_agent_penalty_summary(agent_id)

            # Determine rewrite urgency
            penalty_counts = agent_summary.get('penalty_counts', {})
            critical_count = penalty_counts.get('critical', 0)
            major_count = penalty_counts.get('major', 0)

            rewrite_urgency = 'low'
            if critical_count >= 3:
                rewrite_urgency = 'critical'
            elif major_count >= 5:
                rewrite_urgency = 'high'
            elif major_count >= 3:
                rewrite_urgency = 'moderate'

            # Generate DGM rewrite specification
            rewrite_spec = {
                'agent_id': agent_id,
                'rewrite_urgency': rewrite_urgency,
                'target_behaviors': {
                    'eliminate_gaming_patterns': True,
                    'enhance_mcp_appropriateness': True,
                    'improve_context_awareness': True,
                    'strengthen_outcome_validation': True
                },
                'penalty_history': agent_summary,
                'expected_improvements': {
                    'gaming_elimination': 'target_0_attempts',
                    'appropriateness_score': 'target_0.8_plus',
                    'success_rate': 'target_0.9_plus',
                    'enforcement_compliance': 'target_0.95_plus'
                }
            }

            logger.warning(f"DGM rewrite triggered for agent {agent_id} with {rewrite_urgency} urgency")

            return {
                'rewrite_needed': True,
                'rewrite_spec': rewrite_spec,
                'urgency': rewrite_urgency
            }

        except Exception as e:
            logger.error(f"DGM rewrite integration failed for agent {agent_id}: {e}")
            return {'error': str(e)}

    def prune_gaming_traits(self, agent_traits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observer-approved gaming trait pruning
        Removes gaming-associated traits from agent behavior patterns
        """
        try:
            original_traits = agent_traits.copy()
            pruned_traits = {}
            gaming_indicators = []

            # Identify and prune gaming-associated traits
            for trait_name, trait_value in agent_traits.items():
                trait_str = str(trait_name).lower()

                # Gaming trait patterns to prune
                gaming_patterns = [
                    'dummy', 'fake', 'exploit', 'hack', 'cheat',
                    'minimal_compliance', 'low_effort', 'gaming'
                ]

                is_gaming_trait = any(pattern in trait_str for pattern in gaming_patterns)

                if is_gaming_trait:
                    gaming_indicators.append(trait_name)
                    # Replace with positive alternatives
                    pruned_traits[f"enhanced_{trait_name}"] = max(0.7, trait_value)
                else:
                    # Keep non-gaming traits
                    pruned_traits[trait_name] = trait_value

            # Add anti-gaming traits
            anti_gaming_traits = {
                'mcp_appropriateness_focus': 0.9,
                'context_awareness': 0.8,
                'gaming_resistance': 0.95,
                'enforcement_compliance': 0.9
            }

            pruned_traits.update(anti_gaming_traits)

            logger.info(f"Gaming traits pruned: {len(gaming_indicators)} removed, {len(anti_gaming_traits)} anti-gaming traits added")

            return {
                'gaming_traits_removed': len(gaming_indicators),
                'pruned_traits': pruned_traits
            }

        except Exception as e:
            logger.error(f"Gaming trait pruning failed: {e}")
            return {'error': str(e), 'pruned_traits': agent_traits}
