"""
Emergent Behavior Detector

Advanced system for detecting, analyzing, and cataloging emergent behaviors
in multi-agent orchestration systems. Identifies novel coordination patterns
and breakthrough capabilities.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

from .coordination_models import (
    PerformanceMetrics, OrchestrationConfig, AgentID, ServerID, TaskID
)

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """Types of emergent behaviors."""
    COORDINATION_PATTERN = "coordination_pattern"
    PERFORMANCE_BREAKTHROUGH = "performance_breakthrough"
    ADAPTIVE_STRATEGY = "adaptive_strategy"
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"
    NOVEL_SOLUTION = "novel_solution"
    EFFICIENCY_OPTIMIZATION = "efficiency_optimization"
    FAULT_TOLERANCE = "fault_tolerance"
    UNKNOWN = "unknown"


class BehaviorSignificance(Enum):
    """Significance levels of emergent behaviors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BREAKTHROUGH = "breakthrough"


@dataclass
class BehaviorPattern:
    """Detected emergent behavior pattern."""
    pattern_id: str
    behavior_type: BehaviorType
    significance: BehaviorSignificance
    description: str
    
    # Pattern characteristics
    agents_involved: Set[AgentID]
    servers_involved: Set[ServerID]
    tasks_involved: Set[TaskID]
    
    # Performance metrics
    performance_improvement: float
    efficiency_gain: float
    reliability_increase: float
    
    # Pattern data
    pattern_data: Dict[str, Any] = field(default_factory=dict)
    detection_confidence: float = 0.0
    
    # Temporal information
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)
    observation_count: int = 1
    
    # Reproducibility
    reproduction_attempts: int = 0
    successful_reproductions: int = 0
    
    @property
    def reproduction_rate(self) -> float:
        """Calculate reproduction success rate."""
        if self.reproduction_attempts == 0:
            return 0.0
        return self.successful_reproductions / self.reproduction_attempts
    
    @property
    def pattern_stability(self) -> float:
        """Calculate pattern stability over time."""
        if self.observation_count < 2:
            return 0.0
        
        duration = (self.last_observed - self.first_observed).total_seconds()
        if duration == 0:
            return 1.0
        
        # Stability based on observation frequency
        expected_observations = duration / 3600  # Expected hourly observations
        stability = min(1.0, self.observation_count / max(expected_observations, 1))
        return stability


@dataclass
class BehaviorEvent:
    """Individual behavior event observation."""
    event_id: str
    pattern_id: Optional[str]
    timestamp: datetime
    agents_involved: Set[AgentID]
    servers_involved: Set[ServerID]
    performance_metrics: Dict[str, float]
    context_data: Dict[str, Any] = field(default_factory=dict)


class EmergentBehaviorDetector:
    """
    Advanced emergent behavior detection and analysis system.
    
    Features:
    - Real-time pattern recognition
    - Performance anomaly detection
    - Collective intelligence identification
    - Novel solution discovery
    - Behavior cataloging and reproduction
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        
        # Behavior tracking
        self.detected_patterns: Dict[str, BehaviorPattern] = {}
        self.behavior_events: deque = deque(maxlen=10000)
        self.pattern_history: Dict[str, List[BehaviorEvent]] = defaultdict(list)
        
        # Detection parameters
        self.detection_threshold = 0.7
        self.significance_thresholds = {
            BehaviorSignificance.LOW: 0.05,
            BehaviorSignificance.MEDIUM: 0.15,
            BehaviorSignificance.HIGH: 0.30,
            BehaviorSignificance.BREAKTHROUGH: 0.50
        }
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        self.baseline_window = timedelta(hours=24)
        
        # Pattern recognition
        self.pattern_recognizers: List[Callable] = []
        self.anomaly_detectors: List[Callable] = []
        
        # Analysis state
        self.is_running = False
        self.analysis_task: Optional[asyncio.Task] = None
        
        # Initialize built-in detectors
        self._initialize_detectors()
        
        logger.info("Emergent Behavior Detector initialized")
    
    async def start(self):
        """Start the behavior detection system."""
        self.is_running = True
        
        # Start analysis loop
        self.analysis_task = asyncio.create_task(self._analysis_loop())
        
        logger.info("Emergent Behavior Detector started")
    
    async def stop(self):
        """Stop the behavior detection system."""
        self.is_running = False
        
        if self.analysis_task:
            self.analysis_task.cancel()
        
        logger.info("Emergent Behavior Detector stopped")
    
    async def observe_system_state(self, 
                                 agents_state: Dict[AgentID, Dict[str, Any]],
                                 servers_state: Dict[ServerID, Dict[str, Any]],
                                 performance_metrics: PerformanceMetrics,
                                 task_data: Dict[TaskID, Dict[str, Any]]):
        """Observe current system state for behavior detection."""
        try:
            # Create behavior event
            event = BehaviorEvent(
                event_id=f"event_{datetime.utcnow().timestamp()}",
                pattern_id=None,
                timestamp=datetime.utcnow(),
                agents_involved=set(agents_state.keys()),
                servers_involved=set(servers_state.keys()),
                performance_metrics={
                    'task_success_rate': performance_metrics.task_success_rate,
                    'avg_response_time': performance_metrics.avg_response_time,
                    'coordination_efficiency': performance_metrics.coordination_efficiency,
                    'agent_utilization': performance_metrics.avg_agent_utilization,
                    'system_health': performance_metrics.system_health_score
                },
                context_data={
                    'agents_state': agents_state,
                    'servers_state': servers_state,
                    'task_data': task_data
                }
            )
            
            # Store event
            self.behavior_events.append(event)
            
            # Trigger immediate analysis for significant events
            await self._analyze_event(event)
            
        except Exception as e:
            logger.error(f"System state observation failed: {e}")
    
    async def detect_patterns(self, analysis_window: timedelta = None) -> List[BehaviorPattern]:
        """Detect emergent behavior patterns in recent events."""
        analysis_window = analysis_window or timedelta(hours=1)
        cutoff_time = datetime.utcnow() - analysis_window
        
        try:
            # Get recent events
            recent_events = [
                event for event in self.behavior_events
                if event.timestamp >= cutoff_time
            ]
            
            if len(recent_events) < 10:
                return []
            
            detected_patterns = []
            
            # Run pattern recognizers
            for recognizer in self.pattern_recognizers:
                try:
                    patterns = await recognizer(recent_events)
                    detected_patterns.extend(patterns)
                except Exception as e:
                    logger.error(f"Pattern recognizer failed: {e}")
            
            # Filter and validate patterns
            validated_patterns = []
            for pattern in detected_patterns:
                if await self._validate_pattern(pattern):
                    validated_patterns.append(pattern)
            
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return []
    
    async def analyze_performance_breakthrough(self, 
                                             current_metrics: PerformanceMetrics) -> Optional[BehaviorPattern]:
        """Analyze for performance breakthrough behaviors."""
        try:
            # Update baselines
            await self._update_performance_baselines(current_metrics)
            
            # Check for significant improvements
            improvements = {}
            
            for metric_name, current_value in {
                'task_success_rate': current_metrics.task_success_rate,
                'coordination_efficiency': current_metrics.coordination_efficiency,
                'system_health': current_metrics.system_health_score
            }.items():
                baseline = self.performance_baselines.get(metric_name, current_value)
                if baseline > 0:
                    improvement = (current_value - baseline) / baseline
                    improvements[metric_name] = improvement
            
            # Find maximum improvement
            max_improvement = max(improvements.values()) if improvements else 0.0
            
            # Check if improvement is significant
            if max_improvement >= self.significance_thresholds[BehaviorSignificance.MEDIUM]:
                significance = self._calculate_significance(max_improvement)
                
                pattern = BehaviorPattern(
                    pattern_id=f"breakthrough_{datetime.utcnow().timestamp()}",
                    behavior_type=BehaviorType.PERFORMANCE_BREAKTHROUGH,
                    significance=significance,
                    description=f"Performance breakthrough detected: {max_improvement:.2%} improvement",
                    agents_involved=set(),  # Will be populated by analysis
                    servers_involved=set(),
                    tasks_involved=set(),
                    performance_improvement=max_improvement,
                    efficiency_gain=improvements.get('coordination_efficiency', 0.0),
                    reliability_increase=improvements.get('system_health', 0.0),
                    pattern_data={
                        'improvements': improvements,
                        'baseline_metrics': dict(self.performance_baselines),
                        'current_metrics': {
                            'task_success_rate': current_metrics.task_success_rate,
                            'coordination_efficiency': current_metrics.coordination_efficiency,
                            'system_health': current_metrics.system_health_score
                        }
                    },
                    detection_confidence=min(0.9, max_improvement * 2)
                )
                
                return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Performance breakthrough analysis failed: {e}")
            return None
    
    async def identify_collective_intelligence(self, 
                                             agents_data: Dict[AgentID, Dict[str, Any]]) -> Optional[BehaviorPattern]:
        """Identify collective intelligence behaviors."""
        try:
            if len(agents_data) < 3:
                return None
            
            # Analyze agent collaboration patterns
            collaboration_score = await self._calculate_collaboration_score(agents_data)
            
            # Check for emergent collective behavior
            if collaboration_score >= 0.8:
                # Analyze the nature of the collective behavior
                behavior_characteristics = await self._analyze_collective_characteristics(agents_data)
                
                pattern = BehaviorPattern(
                    pattern_id=f"collective_{datetime.utcnow().timestamp()}",
                    behavior_type=BehaviorType.COLLECTIVE_INTELLIGENCE,
                    significance=self._calculate_significance(collaboration_score),
                    description=f"Collective intelligence behavior: {collaboration_score:.2%} collaboration efficiency",
                    agents_involved=set(agents_data.keys()),
                    servers_involved=set(),
                    tasks_involved=set(),
                    performance_improvement=collaboration_score - 0.5,  # Baseline assumption
                    efficiency_gain=behavior_characteristics.get('efficiency_gain', 0.0),
                    reliability_increase=behavior_characteristics.get('reliability_gain', 0.0),
                    pattern_data={
                        'collaboration_score': collaboration_score,
                        'characteristics': behavior_characteristics,
                        'agent_roles': await self._identify_agent_roles(agents_data)
                    },
                    detection_confidence=collaboration_score
                )
                
                return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Collective intelligence identification failed: {e}")
            return None
    
    async def catalog_behavior(self, pattern: BehaviorPattern) -> bool:
        """Catalog a detected behavior pattern."""
        try:
            # Check if pattern already exists
            existing_pattern = await self._find_similar_pattern(pattern)
            
            if existing_pattern:
                # Update existing pattern
                existing_pattern.last_observed = datetime.utcnow()
                existing_pattern.observation_count += 1
                
                # Update performance metrics if improved
                if pattern.performance_improvement > existing_pattern.performance_improvement:
                    existing_pattern.performance_improvement = pattern.performance_improvement
                    existing_pattern.efficiency_gain = pattern.efficiency_gain
                    existing_pattern.reliability_increase = pattern.reliability_increase
                
                logger.info(f"Updated existing pattern: {existing_pattern.pattern_id}")
                return True
            else:
                # Add new pattern
                self.detected_patterns[pattern.pattern_id] = pattern
                logger.info(f"Cataloged new behavior pattern: {pattern.pattern_id} ({pattern.behavior_type.value})")
                return True
                
        except Exception as e:
            logger.error(f"Behavior cataloging failed: {e}")
            return False
    
    async def attempt_reproduction(self, pattern_id: str) -> bool:
        """Attempt to reproduce a detected behavior pattern."""
        try:
            pattern = self.detected_patterns.get(pattern_id)
            if not pattern:
                return False
            
            pattern.reproduction_attempts += 1
            
            # Implement reproduction logic based on pattern type
            success = await self._reproduce_pattern(pattern)
            
            if success:
                pattern.successful_reproductions += 1
                logger.info(f"Successfully reproduced pattern: {pattern_id}")
            else:
                logger.warning(f"Failed to reproduce pattern: {pattern_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Pattern reproduction failed for {pattern_id}: {e}")
            return False
    
    async def _analysis_loop(self):
        """Background analysis loop."""
        while self.is_running:
            try:
                # Detect new patterns
                new_patterns = await self.detect_patterns()
                
                # Catalog new patterns
                for pattern in new_patterns:
                    await self.catalog_behavior(pattern)
                
                # Analyze pattern evolution
                await self._analyze_pattern_evolution()
                
                # Clean up old events
                await self._cleanup_old_events()
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_event(self, event: BehaviorEvent):
        """Analyze individual event for immediate pattern detection."""
        try:
            # Check for anomalies
            for detector in self.anomaly_detectors:
                try:
                    anomaly = await detector(event)
                    if anomaly:
                        logger.info(f"Anomaly detected: {anomaly}")
                except Exception as e:
                    logger.error(f"Anomaly detector failed: {e}")
            
            # Check for pattern matches
            for pattern_id, pattern in self.detected_patterns.items():
                if await self._event_matches_pattern(event, pattern):
                    event.pattern_id = pattern_id
                    self.pattern_history[pattern_id].append(event)
                    pattern.last_observed = event.timestamp
                    pattern.observation_count += 1
            
        except Exception as e:
            logger.error(f"Event analysis failed: {e}")
    
    def _initialize_detectors(self):
        """Initialize built-in pattern recognizers and anomaly detectors."""
        # Pattern recognizers
        self.pattern_recognizers.extend([
            self._detect_coordination_patterns,
            self._detect_adaptive_strategies,
            self._detect_efficiency_optimizations
        ])
        
        # Anomaly detectors
        self.anomaly_detectors.extend([
            self._detect_performance_anomalies,
            self._detect_coordination_anomalies
        ])
    
    async def _detect_coordination_patterns(self, events: List[BehaviorEvent]) -> List[BehaviorPattern]:
        """Detect novel coordination patterns."""
        patterns = []
        
        try:
            # Analyze agent interaction patterns
            interaction_matrix = self._build_interaction_matrix(events)
            
            # Find unusual coordination patterns
            unusual_patterns = self._find_unusual_interactions(interaction_matrix)
            
            for pattern_data in unusual_patterns:
                pattern = BehaviorPattern(
                    pattern_id=f"coord_{datetime.utcnow().timestamp()}",
                    behavior_type=BehaviorType.COORDINATION_PATTERN,
                    significance=BehaviorSignificance.MEDIUM,
                    description=f"Novel coordination pattern detected",
                    agents_involved=pattern_data['agents'],
                    servers_involved=pattern_data['servers'],
                    tasks_involved=set(),
                    performance_improvement=pattern_data['improvement'],
                    efficiency_gain=pattern_data['efficiency'],
                    reliability_increase=pattern_data['reliability'],
                    pattern_data=pattern_data,
                    detection_confidence=pattern_data['confidence']
                )
                patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Coordination pattern detection failed: {e}")
        
        return patterns
    
    async def _detect_adaptive_strategies(self, events: List[BehaviorEvent]) -> List[BehaviorPattern]:
        """Detect adaptive strategy emergence."""
        patterns = []
        
        try:
            # Analyze strategy changes over time
            strategy_evolution = self._analyze_strategy_evolution(events)
            
            # Find adaptive improvements
            for evolution_data in strategy_evolution:
                if evolution_data['adaptation_score'] > 0.7:
                    pattern = BehaviorPattern(
                        pattern_id=f"adaptive_{datetime.utcnow().timestamp()}",
                        behavior_type=BehaviorType.ADAPTIVE_STRATEGY,
                        significance=self._calculate_significance(evolution_data['adaptation_score']),
                        description=f"Adaptive strategy evolution detected",
                        agents_involved=evolution_data['agents'],
                        servers_involved=evolution_data['servers'],
                        tasks_involved=set(),
                        performance_improvement=evolution_data['improvement'],
                        efficiency_gain=evolution_data['efficiency_gain'],
                        reliability_increase=evolution_data['reliability_gain'],
                        pattern_data=evolution_data,
                        detection_confidence=evolution_data['adaptation_score']
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Adaptive strategy detection failed: {e}")
        
        return patterns
    
    async def _detect_efficiency_optimizations(self, events: List[BehaviorEvent]) -> List[BehaviorPattern]:
        """Detect efficiency optimization behaviors."""
        patterns = []
        
        try:
            # Analyze efficiency trends
            efficiency_trends = self._analyze_efficiency_trends(events)
            
            for trend_data in efficiency_trends:
                if trend_data['optimization_score'] > 0.6:
                    pattern = BehaviorPattern(
                        pattern_id=f"efficiency_{datetime.utcnow().timestamp()}",
                        behavior_type=BehaviorType.EFFICIENCY_OPTIMIZATION,
                        significance=self._calculate_significance(trend_data['optimization_score']),
                        description=f"Efficiency optimization detected",
                        agents_involved=trend_data['agents'],
                        servers_involved=trend_data['servers'],
                        tasks_involved=set(),
                        performance_improvement=trend_data['improvement'],
                        efficiency_gain=trend_data['efficiency_gain'],
                        reliability_increase=0.0,
                        pattern_data=trend_data,
                        detection_confidence=trend_data['optimization_score']
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Efficiency optimization detection failed: {e}")
        
        return patterns
    
    async def _detect_performance_anomalies(self, event: BehaviorEvent) -> Optional[Dict[str, Any]]:
        """Detect performance anomalies in events."""
        try:
            anomalies = []
            
            # Check for unusual performance metrics
            for metric_name, value in event.performance_metrics.items():
                baseline = self.performance_baselines.get(metric_name, value)
                if baseline > 0:
                    deviation = abs(value - baseline) / baseline
                    if deviation > 0.5:  # 50% deviation threshold
                        anomalies.append({
                            'metric': metric_name,
                            'value': value,
                            'baseline': baseline,
                            'deviation': deviation
                        })
            
            if anomalies:
                return {
                    'type': 'performance_anomaly',
                    'event_id': event.event_id,
                    'anomalies': anomalies,
                    'timestamp': event.timestamp
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Performance anomaly detection failed: {e}")
            return None
    
    async def _detect_coordination_anomalies(self, event: BehaviorEvent) -> Optional[Dict[str, Any]]:
        """Detect coordination anomalies in events."""
        try:
            # Check for unusual agent/server combinations
            unusual_combinations = []
            
            # Simple heuristic: check if agent-server combination is rare
            combination_key = f"{len(event.agents_involved)}_{len(event.servers_involved)}"
            
            # This would be enhanced with historical data analysis
            if len(event.agents_involved) > 10 or len(event.servers_involved) > 5:
                unusual_combinations.append({
                    'type': 'high_coordination_complexity',
                    'agents_count': len(event.agents_involved),
                    'servers_count': len(event.servers_involved)
                })
            
            if unusual_combinations:
                return {
                    'type': 'coordination_anomaly',
                    'event_id': event.event_id,
                    'anomalies': unusual_combinations,
                    'timestamp': event.timestamp
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Coordination anomaly detection failed: {e}")
            return None
    
    def _calculate_significance(self, improvement_score: float) -> BehaviorSignificance:
        """Calculate behavior significance based on improvement score."""
        if improvement_score >= self.significance_thresholds[BehaviorSignificance.BREAKTHROUGH]:
            return BehaviorSignificance.BREAKTHROUGH
        elif improvement_score >= self.significance_thresholds[BehaviorSignificance.HIGH]:
            return BehaviorSignificance.HIGH
        elif improvement_score >= self.significance_thresholds[BehaviorSignificance.MEDIUM]:
            return BehaviorSignificance.MEDIUM
        else:
            return BehaviorSignificance.LOW
    
    async def _update_performance_baselines(self, metrics: PerformanceMetrics):
        """Update performance baselines with exponential moving average."""
        alpha = 0.1  # Learning rate
        
        current_metrics = {
            'task_success_rate': metrics.task_success_rate,
            'coordination_efficiency': metrics.coordination_efficiency,
            'system_health': metrics.system_health_score,
            'avg_response_time': metrics.avg_response_time
        }
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.performance_baselines:
                # Exponential moving average
                self.performance_baselines[metric_name] = (
                    alpha * current_value + 
                    (1 - alpha) * self.performance_baselines[metric_name]
                )
            else:
                self.performance_baselines[metric_name] = current_value
    
    # Placeholder implementations for complex analysis methods
    def _build_interaction_matrix(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Build agent-server interaction matrix."""
        return {'placeholder': 'interaction_matrix'}
    
    def _find_unusual_interactions(self, matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find unusual interaction patterns."""
        return []
    
    def _analyze_strategy_evolution(self, events: List[BehaviorEvent]) -> List[Dict[str, Any]]:
        """Analyze strategy evolution patterns."""
        return []
    
    def _analyze_efficiency_trends(self, events: List[BehaviorEvent]) -> List[Dict[str, Any]]:
        """Analyze efficiency optimization trends."""
        return []
    
    async def _calculate_collaboration_score(self, agents_data: Dict[AgentID, Dict[str, Any]]) -> float:
        """Calculate collaboration score between agents."""
        return 0.5  # Placeholder
    
    async def _analyze_collective_characteristics(self, agents_data: Dict[AgentID, Dict[str, Any]]) -> Dict[str, float]:
        """Analyze characteristics of collective behavior."""
        return {'efficiency_gain': 0.1, 'reliability_gain': 0.05}
    
    async def _identify_agent_roles(self, agents_data: Dict[AgentID, Dict[str, Any]]) -> Dict[AgentID, str]:
        """Identify roles of agents in collective behavior."""
        return {agent_id: 'participant' for agent_id in agents_data.keys()}
    
    async def _validate_pattern(self, pattern: BehaviorPattern) -> bool:
        """Validate detected pattern."""
        return pattern.detection_confidence > self.detection_threshold
    
    async def _find_similar_pattern(self, pattern: BehaviorPattern) -> Optional[BehaviorPattern]:
        """Find similar existing pattern."""
        for existing_pattern in self.detected_patterns.values():
            if (existing_pattern.behavior_type == pattern.behavior_type and
                len(existing_pattern.agents_involved.intersection(pattern.agents_involved)) > 0):
                return existing_pattern
        return None
    
    async def _event_matches_pattern(self, event: BehaviorEvent, pattern: BehaviorPattern) -> bool:
        """Check if event matches existing pattern."""
        return len(event.agents_involved.intersection(pattern.agents_involved)) > 0
    
    async def _reproduce_pattern(self, pattern: BehaviorPattern) -> bool:
        """Attempt to reproduce a behavior pattern."""
        # Placeholder for pattern reproduction logic
        return False
    
    async def _analyze_pattern_evolution(self):
        """Analyze how patterns evolve over time."""
        pass
    
    async def _cleanup_old_events(self):
        """Clean up old behavior events."""
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        # Remove old events from pattern history
        for pattern_id in self.pattern_history:
            self.pattern_history[pattern_id] = [
                event for event in self.pattern_history[pattern_id]
                if event.timestamp >= cutoff_time
            ]
    
    async def get_detector_metrics(self) -> Dict[str, Any]:
        """Get comprehensive detector metrics."""
        return {
            'detected_patterns': len(self.detected_patterns),
            'behavior_events': len(self.behavior_events),
            'pattern_types': {
                behavior_type.value: len([
                    p for p in self.detected_patterns.values()
                    if p.behavior_type == behavior_type
                ])
                for behavior_type in BehaviorType
            },
            'significance_distribution': {
                significance.value: len([
                    p for p in self.detected_patterns.values()
                    if p.significance == significance
                ])
                for significance in BehaviorSignificance
            },
            'performance_baselines': dict(self.performance_baselines),
            'is_running': self.is_running
        }