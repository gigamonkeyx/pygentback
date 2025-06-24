"""
Capability Matcher

Advanced matching system for MCP capabilities and requirements.
Uses semantic analysis, machine learning, and heuristics to match capabilities.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from .mcp_analyzer import MCPServerProfile, MCPCapability, CapabilityType, PerformanceLevel

logger = logging.getLogger(__name__)


class MatchType(Enum):
    """Types of capability matches"""
    EXACT = "exact"
    SEMANTIC = "semantic"
    FUNCTIONAL = "functional"
    PARTIAL = "partial"
    INFERRED = "inferred"


class MatchConfidence(Enum):
    """Confidence levels for matches"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class MatchScore:
    """Score for a capability match"""
    overall_score: float = 0.0
    semantic_score: float = 0.0
    functional_score: float = 0.0
    performance_score: float = 0.0
    compatibility_score: float = 0.0
    confidence: MatchConfidence = MatchConfidence.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'overall_score': self.overall_score,
            'semantic_score': self.semantic_score,
            'functional_score': self.functional_score,
            'performance_score': self.performance_score,
            'compatibility_score': self.compatibility_score,
            'confidence': self.confidence.value
        }


@dataclass
class MatchResult:
    """Result of capability matching"""
    capability: MCPCapability
    server_profile: MCPServerProfile
    match_type: MatchType
    match_score: MatchScore
    matching_tools: List[str] = field(default_factory=list)
    match_explanation: str = ""
    alternative_capabilities: List[CapabilityType] = field(default_factory=list)
    requirements_coverage: Dict[str, float] = field(default_factory=dict)
    
    def is_good_match(self, threshold: float = 0.7) -> bool:
        """Check if this is a good match"""
        return self.match_score.overall_score >= threshold
    
    def get_match_quality(self) -> str:
        """Get qualitative match quality"""
        score = self.match_score.overall_score
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "fair"
        elif score >= 0.3:
            return "poor"
        else:
            return "very_poor"


class CapabilityMatcher:
    """
    Advanced capability matching system for MCP servers.
    
    Matches requirements against available capabilities using multiple
    techniques including semantic analysis, functional matching, and ML.
    """
    
    def __init__(self):
        # Semantic similarity mappings
        self.semantic_mappings = self._initialize_semantic_mappings()
        
        # Functional equivalence mappings
        self.functional_mappings = self._initialize_functional_mappings()
        
        # Performance weights for different match types
        self.match_type_weights = {
            MatchType.EXACT: 1.0,
            MatchType.SEMANTIC: 0.9,
            MatchType.FUNCTIONAL: 0.8,
            MatchType.PARTIAL: 0.6,
            MatchType.INFERRED: 0.4
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            MatchConfidence.VERY_HIGH: 0.9,
            MatchConfidence.HIGH: 0.75,
            MatchConfidence.MEDIUM: 0.6,
            MatchConfidence.LOW: 0.4,
            MatchConfidence.VERY_LOW: 0.0
        }
        
        # Match history for learning
        self.match_history = []
        self.feedback_data = []
    
    def _initialize_semantic_mappings(self) -> Dict[CapabilityType, List[CapabilityType]]:
        """Initialize semantic similarity mappings between capabilities"""
        return {
            CapabilityType.FILE_OPERATIONS: [
                CapabilityType.DATA_PROCESSING,
                CapabilityType.SYSTEM_INTEGRATION
            ],
            CapabilityType.DATABASE_ACCESS: [
                CapabilityType.DATA_PROCESSING,
                CapabilityType.ANALYTICS
            ],
            CapabilityType.WEB_SCRAPING: [
                CapabilityType.API_INTEGRATION,
                CapabilityType.DATA_PROCESSING
            ],
            CapabilityType.API_INTEGRATION: [
                CapabilityType.WEB_SCRAPING,
                CapabilityType.COMMUNICATION
            ],
            CapabilityType.DATA_PROCESSING: [
                CapabilityType.ANALYTICS,
                CapabilityType.FILE_OPERATIONS,
                CapabilityType.DATABASE_ACCESS
            ],
            CapabilityType.CODE_EXECUTION: [
                CapabilityType.DEVELOPMENT_TOOLS,
                CapabilityType.AUTOMATION
            ],
            CapabilityType.NATURAL_LANGUAGE: [
                CapabilityType.COMMUNICATION,
                CapabilityType.ANALYTICS
            ],
            CapabilityType.IMAGE_PROCESSING: [
                CapabilityType.DATA_PROCESSING,
                CapabilityType.ANALYTICS
            ],
            CapabilityType.AUDIO_PROCESSING: [
                CapabilityType.DATA_PROCESSING,
                CapabilityType.NATURAL_LANGUAGE
            ],
            CapabilityType.SYSTEM_INTEGRATION: [
                CapabilityType.FILE_OPERATIONS,
                CapabilityType.AUTOMATION
            ],
            CapabilityType.SECURITY_TOOLS: [
                CapabilityType.SYSTEM_INTEGRATION,
                CapabilityType.DEVELOPMENT_TOOLS
            ],
            CapabilityType.DEVELOPMENT_TOOLS: [
                CapabilityType.CODE_EXECUTION,
                CapabilityType.AUTOMATION
            ],
            CapabilityType.COMMUNICATION: [
                CapabilityType.API_INTEGRATION,
                CapabilityType.NATURAL_LANGUAGE
            ],
            CapabilityType.ANALYTICS: [
                CapabilityType.DATA_PROCESSING,
                CapabilityType.DATABASE_ACCESS
            ],
            CapabilityType.AUTOMATION: [
                CapabilityType.CODE_EXECUTION,
                CapabilityType.SYSTEM_INTEGRATION
            ]
        }
    
    def _initialize_functional_mappings(self) -> Dict[str, List[CapabilityType]]:
        """Initialize functional equivalence mappings"""
        return {
            "data_manipulation": [
                CapabilityType.DATA_PROCESSING,
                CapabilityType.FILE_OPERATIONS,
                CapabilityType.DATABASE_ACCESS
            ],
            "content_generation": [
                CapabilityType.NATURAL_LANGUAGE,
                CapabilityType.CODE_EXECUTION,
                CapabilityType.IMAGE_PROCESSING
            ],
            "information_retrieval": [
                CapabilityType.WEB_SCRAPING,
                CapabilityType.API_INTEGRATION,
                CapabilityType.DATABASE_ACCESS
            ],
            "system_control": [
                CapabilityType.SYSTEM_INTEGRATION,
                CapabilityType.AUTOMATION,
                CapabilityType.CODE_EXECUTION
            ],
            "communication_tasks": [
                CapabilityType.COMMUNICATION,
                CapabilityType.API_INTEGRATION,
                CapabilityType.NATURAL_LANGUAGE
            ],
            "analysis_tasks": [
                CapabilityType.ANALYTICS,
                CapabilityType.DATA_PROCESSING,
                CapabilityType.NATURAL_LANGUAGE
            ]
        }
    
    async def match_capabilities(self, 
                                required_capabilities: List[CapabilityType],
                                available_servers: List[MCPServerProfile],
                                context: Optional[Dict[str, Any]] = None) -> List[MatchResult]:
        """
        Match required capabilities against available servers.
        
        Args:
            required_capabilities: List of required capabilities
            available_servers: List of available server profiles
            context: Optional context for matching
            
        Returns:
            List of match results sorted by score
        """
        try:
            logger.info(f"Matching {len(required_capabilities)} capabilities "
                       f"against {len(available_servers)} servers")
            
            all_matches = []
            
            for capability_type in required_capabilities:
                capability_matches = await self._match_single_capability(
                    capability_type, available_servers, context
                )
                all_matches.extend(capability_matches)
            
            # Sort by overall score
            all_matches.sort(key=lambda m: m.match_score.overall_score, reverse=True)
            
            # Record matches for learning
            self._record_matches(required_capabilities, all_matches, context)
            
            logger.info(f"Found {len(all_matches)} capability matches")
            
            return all_matches
            
        except Exception as e:
            logger.error(f"Capability matching failed: {e}")
            return []
    
    async def _match_single_capability(self, 
                                     capability_type: CapabilityType,
                                     available_servers: List[MCPServerProfile],
                                     context: Optional[Dict[str, Any]]) -> List[MatchResult]:
        """Match a single capability against available servers"""
        matches = []
        
        for server in available_servers:
            # Check for exact matches
            exact_capability = server.get_capability_by_type(capability_type)
            if exact_capability:
                match_result = await self._create_match_result(
                    exact_capability, server, MatchType.EXACT, capability_type, context
                )
                matches.append(match_result)
                continue
            
            # Check for semantic matches
            semantic_matches = await self._find_semantic_matches(
                capability_type, server, context
            )
            matches.extend(semantic_matches)
            
            # Check for functional matches
            functional_matches = await self._find_functional_matches(
                capability_type, server, context
            )
            matches.extend(functional_matches)
            
            # Check for inferred matches
            inferred_matches = await self._find_inferred_matches(
                capability_type, server, context
            )
            matches.extend(inferred_matches)
        
        return matches
    
    async def _find_semantic_matches(self, 
                                   capability_type: CapabilityType,
                                   server: MCPServerProfile,
                                   context: Optional[Dict[str, Any]]) -> List[MatchResult]:
        """Find semantically similar capabilities"""
        matches = []
        
        similar_capabilities = self.semantic_mappings.get(capability_type, [])
        
        for similar_cap_type in similar_capabilities:
            capability = server.get_capability_by_type(similar_cap_type)
            if capability:
                match_result = await self._create_match_result(
                    capability, server, MatchType.SEMANTIC, capability_type, context
                )
                matches.append(match_result)
        
        return matches
    
    async def _find_functional_matches(self, 
                                     capability_type: CapabilityType,
                                     server: MCPServerProfile,
                                     context: Optional[Dict[str, Any]]) -> List[MatchResult]:
        """Find functionally equivalent capabilities"""
        matches = []
        
        # Find functional categories that include the required capability
        for function_name, capabilities in self.functional_mappings.items():
            if capability_type in capabilities:
                # Check if server has other capabilities in this functional group
                for other_cap_type in capabilities:
                    if other_cap_type != capability_type:
                        capability = server.get_capability_by_type(other_cap_type)
                        if capability:
                            match_result = await self._create_match_result(
                                capability, server, MatchType.FUNCTIONAL, capability_type, context
                            )
                            match_result.match_explanation = f"Functional match via {function_name}"
                            matches.append(match_result)
        
        return matches
    
    async def _find_inferred_matches(self, 
                                   capability_type: CapabilityType,
                                   server: MCPServerProfile,
                                   context: Optional[Dict[str, Any]]) -> List[MatchResult]:
        """Find inferred capabilities based on available tools"""
        matches = []
        
        # Analyze tools to infer capabilities
        capability_keywords = {
            CapabilityType.FILE_OPERATIONS: ["file", "read", "write", "directory"],
            CapabilityType.DATABASE_ACCESS: ["database", "sql", "query", "table"],
            CapabilityType.WEB_SCRAPING: ["scrape", "web", "html", "extract"],
            CapabilityType.API_INTEGRATION: ["api", "rest", "http", "request"],
            CapabilityType.DATA_PROCESSING: ["process", "transform", "parse", "convert"],
            CapabilityType.CODE_EXECUTION: ["execute", "run", "code", "script"],
            CapabilityType.NATURAL_LANGUAGE: ["text", "language", "nlp", "sentiment"],
            CapabilityType.IMAGE_PROCESSING: ["image", "photo", "visual", "picture"],
            CapabilityType.COMMUNICATION: ["email", "message", "send", "notify"]
        }
        
        keywords = capability_keywords.get(capability_type, [])
        
        # Check if server tools contain relevant keywords
        matching_tools = []
        for tool in server.available_tools:
            tool_lower = tool.lower()
            for keyword in keywords:
                if keyword in tool_lower:
                    matching_tools.append(tool)
                    break
        
        if matching_tools:
            # Create inferred capability
            inferred_capability = MCPCapability(
                capability_type=capability_type,
                tools=matching_tools,
                performance_level=PerformanceLevel.UNKNOWN,
                complexity_support="basic"
            )
            
            match_result = await self._create_match_result(
                inferred_capability, server, MatchType.INFERRED, capability_type, context
            )
            match_result.match_explanation = f"Inferred from tools: {', '.join(matching_tools[:3])}"
            matches.append(match_result)
        
        return matches
    
    async def _create_match_result(self, 
                                 capability: MCPCapability,
                                 server: MCPServerProfile,
                                 match_type: MatchType,
                                 required_capability: CapabilityType,
                                 context: Optional[Dict[str, Any]]) -> MatchResult:
        """Create a match result with calculated scores"""
        
        # Calculate match score
        match_score = await self._calculate_match_score(
            capability, server, match_type, required_capability, context
        )
        
        # Create match result
        match_result = MatchResult(
            capability=capability,
            server_profile=server,
            match_type=match_type,
            match_score=match_score,
            matching_tools=capability.tools.copy()
        )
        
        # Add explanation based on match type
        if match_type == MatchType.EXACT:
            match_result.match_explanation = f"Exact match for {required_capability.value}"
        elif match_type == MatchType.SEMANTIC:
            match_result.match_explanation = f"Semantic match: {capability.capability_type.value} ≈ {required_capability.value}"
        elif match_type == MatchType.FUNCTIONAL:
            match_result.match_explanation = f"Functional equivalent: {capability.capability_type.value}"
        elif match_type == MatchType.PARTIAL:
            match_result.match_explanation = f"Partial match for {required_capability.value}"
        
        return match_result
    
    async def _calculate_match_score(self, 
                                   capability: MCPCapability,
                                   server: MCPServerProfile,
                                   match_type: MatchType,
                                   required_capability: CapabilityType,
                                   context: Optional[Dict[str, Any]]) -> MatchScore:
        """Calculate comprehensive match score"""
        
        # Base score from match type
        base_score = self.match_type_weights[match_type]
        
        # Semantic similarity score
        semantic_score = self._calculate_semantic_score(
            capability.capability_type, required_capability
        )
        
        # Functional similarity score
        functional_score = self._calculate_functional_score(
            capability.capability_type, required_capability
        )
        
        # Performance score
        performance_scores = {
            PerformanceLevel.EXCELLENT: 1.0,
            PerformanceLevel.GOOD: 0.8,
            PerformanceLevel.AVERAGE: 0.6,
            PerformanceLevel.POOR: 0.3,
            PerformanceLevel.UNKNOWN: 0.5
        }
        performance_score = performance_scores[capability.performance_level]
        
        # Compatibility score
        compatibility_score = self._calculate_compatibility_score(
            capability, server, context
        )
        
        # Overall score calculation
        overall_score = (
            base_score * 0.3 +
            semantic_score * 0.2 +
            functional_score * 0.2 +
            performance_score * 0.2 +
            compatibility_score * 0.1
        )
        
        # Determine confidence level
        confidence = self._determine_confidence(overall_score, match_type)
        
        return MatchScore(
            overall_score=overall_score,
            semantic_score=semantic_score,
            functional_score=functional_score,
            performance_score=performance_score,
            compatibility_score=compatibility_score,
            confidence=confidence
        )
    
    def _calculate_semantic_score(self, 
                                capability_type: CapabilityType,
                                required_capability: CapabilityType) -> float:
        """Calculate semantic similarity score"""
        if capability_type == required_capability:
            return 1.0
        
        # Check if they are semantically related
        similar_caps = self.semantic_mappings.get(required_capability, [])
        if capability_type in similar_caps:
            return 0.8
        
        # Check reverse mapping
        similar_caps = self.semantic_mappings.get(capability_type, [])
        if required_capability in similar_caps:
            return 0.8
        
        # Check for shared semantic categories
        req_similar = set(self.semantic_mappings.get(required_capability, []))
        cap_similar = set(self.semantic_mappings.get(capability_type, []))
        
        if req_similar & cap_similar:  # Intersection
            return 0.6
        
        return 0.2  # Minimal similarity
    
    def _calculate_functional_score(self, 
                                  capability_type: CapabilityType,
                                  required_capability: CapabilityType) -> float:
        """Calculate functional similarity score"""
        if capability_type == required_capability:
            return 1.0
        
        # Check if they belong to the same functional group
        for function_name, capabilities in self.functional_mappings.items():
            if capability_type in capabilities and required_capability in capabilities:
                return 0.9
        
        return 0.1  # Minimal functional similarity
    
    def _calculate_compatibility_score(self, 
                                     capability: MCPCapability,
                                     server: MCPServerProfile,
                                     context: Optional[Dict[str, Any]]) -> float:
        """Calculate compatibility score based on context"""
        score = 1.0
        
        if not context:
            return score
        
        # Check resource requirements
        max_memory = context.get('max_memory_mb')
        if max_memory and server.memory_usage_mb > max_memory:
            score *= 0.5
        
        max_cpu = context.get('max_cpu_percent')
        if max_cpu and server.cpu_usage_percent > max_cpu:
            score *= 0.5
        
        # Check latency requirements
        max_latency = context.get('max_latency_ms')
        if max_latency and capability.latency_ms > max_latency:
            score *= 0.3
        
        # Check success rate requirements
        min_success_rate = context.get('min_success_rate')
        if min_success_rate and capability.success_rate < min_success_rate:
            score *= 0.4
        
        return score
    
    def _determine_confidence(self, overall_score: float, match_type: MatchType) -> MatchConfidence:
        """Determine confidence level for the match"""
        # Adjust score based on match type
        if match_type == MatchType.EXACT:
            adjusted_score = overall_score
        elif match_type == MatchType.SEMANTIC:
            adjusted_score = overall_score * 0.9
        elif match_type == MatchType.FUNCTIONAL:
            adjusted_score = overall_score * 0.8
        elif match_type == MatchType.INFERRED:
            adjusted_score = overall_score * 0.6
        else:
            adjusted_score = overall_score * 0.7
        
        # Determine confidence level
        for confidence, threshold in sorted(self.confidence_thresholds.items(), 
                                          key=lambda x: x[1], reverse=True):
            if adjusted_score >= threshold:
                return confidence
        
        return MatchConfidence.VERY_LOW
    
    def _record_matches(self, 
                       required_capabilities: List[CapabilityType],
                       matches: List[MatchResult],
                       context: Optional[Dict[str, Any]]):
        """Record matches for learning and analysis"""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'required_capabilities': [cap.value for cap in required_capabilities],
            'total_matches': len(matches),
            'match_types': [match.match_type.value for match in matches],
            'average_score': np.mean([match.match_score.overall_score for match in matches]) if matches else 0.0,
            'context': context or {}
        }
        
        self.match_history.append(record)
        
        # Keep limited history
        if len(self.match_history) > 1000:
            self.match_history = self.match_history[-1000:]
    
    def add_match_feedback(self, match_result: MatchResult, 
                          success: bool, effectiveness_rating: float,
                          feedback_notes: str = ""):
        """Add feedback about match effectiveness"""
        feedback = {
            'timestamp': datetime.utcnow().isoformat(),
            'match_id': id(match_result),
            'server_name': match_result.server_profile.server_name,
            'capability_type': match_result.capability.capability_type.value,
            'match_type': match_result.match_type.value,
            'predicted_score': match_result.match_score.overall_score,
            'actual_success': success,
            'effectiveness_rating': effectiveness_rating,
            'feedback_notes': feedback_notes
        }
        
        self.feedback_data.append(feedback)
        
        # Keep limited feedback
        if len(self.feedback_data) > 1000:
            self.feedback_data = self.feedback_data[-1000:]
    
    def get_matching_statistics(self) -> Dict[str, Any]:
        """Get statistics about capability matching"""
        if not self.match_history:
            return {"error": "No matching history available"}
        
        # Match type distribution
        all_match_types = []
        for record in self.match_history:
            all_match_types.extend(record['match_types'])
        
        match_type_counts = {}
        for match_type in all_match_types:
            match_type_counts[match_type] = match_type_counts.get(match_type, 0) + 1
        
        # Average scores
        avg_scores = [record['average_score'] for record in self.match_history]
        
        # Feedback analysis
        feedback_stats = {}
        if self.feedback_data:
            success_rate = sum(1 for f in self.feedback_data if f['actual_success']) / len(self.feedback_data)
            avg_effectiveness = np.mean([f['effectiveness_rating'] for f in self.feedback_data])
            
            # Accuracy analysis (predicted vs actual)
            predicted_scores = [f['predicted_score'] for f in self.feedback_data]
            actual_ratings = [f['effectiveness_rating'] for f in self.feedback_data]
            correlation = np.corrcoef(predicted_scores, actual_ratings)[0, 1] if len(predicted_scores) > 1 else 0.0
            
            feedback_stats = {
                'success_rate': success_rate,
                'average_effectiveness': avg_effectiveness,
                'prediction_accuracy': correlation,
                'total_feedback': len(self.feedback_data)
            }
        
        return {
            'total_matching_sessions': len(self.match_history),
            'match_type_distribution': match_type_counts,
            'average_match_score': np.mean(avg_scores) if avg_scores else 0.0,
            'feedback_statistics': feedback_stats
        }
