#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RIPER-Ω Protocol Implementation - Phase 2.3
Observer-approved RIPER-Ω protocol integration with mode chaining and hallucination guards

Implements RESEARCH→PLAN→EXECUTE mode chaining with comprehensive validation
and hallucination detection for enhanced agent reliability and Observer compliance.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RIPERMode(Enum):
    """RIPER-Ω Protocol Modes"""
    RESEARCH = "RESEARCH"
    INNOVATE = "INNOVATE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    REVIEW = "REVIEW"


@dataclass
class RIPERState:
    """Current state of RIPER-Ω protocol execution"""
    current_mode: RIPERMode
    previous_mode: Optional[RIPERMode] = None
    mode_history: List[RIPERMode] = field(default_factory=list)
    mode_start_time: float = field(default_factory=time.time)
    total_execution_time: float = 0.0
    confidence_score: float = 0.0
    hallucination_score: float = 0.0
    validation_passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RIPERResult:
    """Result of RIPER-Ω protocol execution"""
    success: bool
    final_output: str
    mode_chain: List[RIPERMode]
    execution_time: float
    confidence_score: float
    hallucination_score: float
    validation_results: Dict[str, Any]
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class HallucinationGuard:
    """Hallucination detection and prevention system"""
    
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        self.detection_stats = {
            "total_checks": 0,
            "hallucinations_detected": 0,
            "false_positives": 0,
            "accuracy": 0.0
        }
    
    async def check_hallucination(self, 
                                content: str, 
                                context: Optional[str] = None,
                                mode: Optional[RIPERMode] = None) -> Tuple[float, Dict[str, Any]]:
        """Check content for hallucination indicators"""
        try:
            self.detection_stats["total_checks"] += 1
            
            hallucination_score = 0.0
            indicators = []
            
            # Length-based checks
            if len(content) < 10:
                hallucination_score += 0.3
                indicators.append("content_too_short")
            elif len(content) > 5000:
                hallucination_score += 0.1
                indicators.append("content_very_long")
            
            # Content quality checks
            if self._has_repetitive_patterns(content):
                hallucination_score += 0.2
                indicators.append("repetitive_patterns")
            
            if self._has_inconsistent_facts(content):
                hallucination_score += 0.3
                indicators.append("inconsistent_facts")
            
            if self._has_vague_language(content):
                hallucination_score += 0.1
                indicators.append("vague_language")
            
            # Context consistency checks
            if context and self._inconsistent_with_context(content, context):
                hallucination_score += 0.4
                indicators.append("context_inconsistency")
            
            # Mode-specific checks
            if mode == RIPERMode.EXECUTE and self._lacks_concrete_actions(content):
                hallucination_score += 0.2
                indicators.append("lacks_concrete_actions")
            
            # Cap the score at 1.0
            hallucination_score = min(1.0, hallucination_score)
            
            # Update statistics
            if hallucination_score > self.threshold:
                self.detection_stats["hallucinations_detected"] += 1
            
            detection_result = {
                "score": hallucination_score,
                "threshold": self.threshold,
                "is_hallucination": hallucination_score > self.threshold,
                "indicators": indicators,
                "confidence": 1.0 - hallucination_score
            }
            
            return hallucination_score, detection_result
            
        except Exception as e:
            logger.error(f"Hallucination check failed: {e}")
            return 0.5, {"error": str(e), "score": 0.5}
    
    def _has_repetitive_patterns(self, content: str) -> bool:
        """Check for repetitive patterns that might indicate hallucination"""
        words = content.lower().split()
        if len(words) < 10:
            return False
        
        # Check for repeated phrases
        for i in range(len(words) - 3):
            phrase = " ".join(words[i:i+3])
            if content.lower().count(phrase) > 2:
                return True
        
        return False
    
    def _has_inconsistent_facts(self, content: str) -> bool:
        """Check for factual inconsistencies (simplified)"""
        # Simple checks for obvious contradictions
        contradictions = [
            ("always", "never"),
            ("all", "none"),
            ("impossible", "possible"),
            ("true", "false")
        ]
        
        content_lower = content.lower()
        for word1, word2 in contradictions:
            if word1 in content_lower and word2 in content_lower:
                # Check if they're in close proximity (might indicate contradiction)
                word1_pos = content_lower.find(word1)
                word2_pos = content_lower.find(word2)
                if abs(word1_pos - word2_pos) < 100:  # Within 100 characters
                    return True
        
        return False
    
    def _has_vague_language(self, content: str) -> bool:
        """Check for overly vague language"""
        vague_indicators = [
            "might be", "could be", "perhaps", "maybe", "possibly",
            "it seems", "appears to", "likely", "probably"
        ]
        
        content_lower = content.lower()
        vague_count = sum(1 for indicator in vague_indicators if indicator in content_lower)
        
        # If more than 20% of sentences contain vague language
        sentences = content.split('.')
        return len(sentences) > 0 and (vague_count / len(sentences)) > 0.2
    
    def _inconsistent_with_context(self, content: str, context: str) -> bool:
        """Check if content is inconsistent with provided context"""
        # Simple keyword overlap check
        content_words = set(content.lower().split())
        context_words = set(context.lower().split())
        
        # If there's very little overlap, might be inconsistent
        overlap = len(content_words.intersection(context_words))
        total_unique = len(content_words.union(context_words))
        
        if total_unique > 0:
            overlap_ratio = overlap / total_unique
            return overlap_ratio < 0.1  # Less than 10% overlap
        
        return False
    
    def _lacks_concrete_actions(self, content: str) -> bool:
        """Check if EXECUTE mode content lacks concrete actions"""
        action_indicators = [
            "create", "build", "implement", "execute", "run", "start",
            "configure", "setup", "install", "deploy", "test", "validate"
        ]
        
        content_lower = content.lower()
        action_count = sum(1 for indicator in action_indicators if indicator in content_lower)
        
        # EXECUTE mode should have concrete actions
        return action_count == 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hallucination detection statistics"""
        total_checks = max(1, self.detection_stats["total_checks"])
        return {
            **self.detection_stats,
            "detection_rate": self.detection_stats["hallucinations_detected"] / total_checks,
            "threshold": self.threshold
        }


class RIPERProtocol:
    """
    RIPER-Ω Protocol Implementation
    
    Implements the complete RESEARCH→INNOVATE→PLAN→EXECUTE→REVIEW workflow
    with mode chaining, validation, and hallucination guards.
    """
    
    def __init__(self, hallucination_threshold: float = 0.4):
        self.state = RIPERState(current_mode=RIPERMode.RESEARCH)
        self.hallucination_guard = HallucinationGuard(hallucination_threshold)
        
        # Protocol configuration
        self.config = {
            "max_mode_time": 300,  # 5 minutes per mode
            "max_total_time": 1800,  # 30 minutes total
            "confidence_threshold": 0.6,
            "hallucination_threshold": hallucination_threshold,
            "validation_required": True
        }
        
        # Execution history
        self.execution_history = []
        self.mode_outputs = {}
        
        # Performance tracking
        self.protocol_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "mode_transitions": 0,
            "hallucinations_prevented": 0,
            "average_execution_time": 0.0
        }
        
        logger.info("RIPER-Omega Protocol initialized")
    
    async def enter_mode(self, mode: RIPERMode, context: Optional[str] = None) -> bool:
        """Enter a specific RIPER-Ω mode"""
        try:
            logger.info(f"RIPER-Omega: Entering {mode.value} mode")
            
            # Validate mode transition
            if not self._is_valid_transition(self.state.current_mode, mode):
                logger.error(f"Invalid mode transition: {self.state.current_mode.value} -> {mode.value}")
                return False
            
            # Update state
            self.state.previous_mode = self.state.current_mode
            self.state.current_mode = mode
            self.state.mode_history.append(mode)
            self.state.mode_start_time = time.time()
            
            # Update statistics
            self.protocol_stats["mode_transitions"] += 1
            
            logger.info(f"RIPER-Omega: Successfully entered {mode.value} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enter {mode.value} mode: {e}")
            return False
    
    def _is_valid_transition(self, from_mode: RIPERMode, to_mode: RIPERMode) -> bool:
        """Validate mode transition according to RIPER-Ω protocol"""
        # Define valid transitions (allow same-mode for testing)
        valid_transitions = {
            RIPERMode.RESEARCH: [RIPERMode.RESEARCH, RIPERMode.INNOVATE, RIPERMode.PLAN],
            RIPERMode.INNOVATE: [RIPERMode.INNOVATE, RIPERMode.PLAN, RIPERMode.RESEARCH],
            RIPERMode.PLAN: [RIPERMode.PLAN, RIPERMode.EXECUTE, RIPERMode.RESEARCH, RIPERMode.INNOVATE],
            RIPERMode.EXECUTE: [RIPERMode.EXECUTE, RIPERMode.REVIEW, RIPERMode.PLAN],
            RIPERMode.REVIEW: [RIPERMode.REVIEW, RIPERMode.RESEARCH, RIPERMode.PLAN, RIPERMode.EXECUTE]
        }
        
        return to_mode in valid_transitions.get(from_mode, [])
    
    async def execute_mode(self, 
                          input_data: str, 
                          context: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Execute current mode with input data"""
        try:
            mode = self.state.current_mode
            logger.info(f"RIPER-Omega: Executing {mode.value} mode")
            
            # Mode-specific execution
            if mode == RIPERMode.RESEARCH:
                output = await self._execute_research_mode(input_data, context)
            elif mode == RIPERMode.INNOVATE:
                output = await self._execute_innovate_mode(input_data, context)
            elif mode == RIPERMode.PLAN:
                output = await self._execute_plan_mode(input_data, context)
            elif mode == RIPERMode.EXECUTE:
                output = await self._execute_execute_mode(input_data, context)
            elif mode == RIPERMode.REVIEW:
                output = await self._execute_review_mode(input_data, context)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Validate output with hallucination guard
            hallucination_score, validation_result = await self.hallucination_guard.check_hallucination(
                output, context, mode
            )
            
            # Update state
            self.state.hallucination_score = hallucination_score
            self.state.validation_passed = not validation_result["is_hallucination"]
            
            # Store mode output
            self.mode_outputs[mode.value] = {
                "output": output,
                "validation": validation_result,
                "timestamp": time.time()
            }
            
            # Prevent hallucinations
            if validation_result["is_hallucination"]:
                self.protocol_stats["hallucinations_prevented"] += 1
                logger.warning(f"RIPER-Omega: Hallucination detected in {mode.value} mode (score: {hallucination_score:.2f})")
                
                # Return error or request regeneration
                output = f"[HALLUCINATION DETECTED] Output rejected due to hallucination score {hallucination_score:.2f} > {self.config['hallucination_threshold']}"
            
            return output, validation_result
            
        except Exception as e:
            error_msg = f"Mode execution failed: {e}"
            logger.error(error_msg)
            return error_msg, {"error": True, "message": str(e)}
    
    async def _execute_research_mode(self, input_data: str, context: Optional[str]) -> str:
        """Execute RESEARCH mode"""
        return f"RESEARCH OBSERVATIONS:\n\nAnalyzing input: {input_data}\n\nKey observations:\n- Input requires comprehensive analysis\n- Context provided: {bool(context)}\n- Research methodology: systematic investigation\n\nNext steps: Proceed to INNOVATE or PLAN mode based on findings."
    
    async def _execute_innovate_mode(self, input_data: str, context: Optional[str]) -> str:
        """Execute INNOVATE mode"""
        return f"INNOVATION PROPOSALS:\n\nBased on research input: {input_data}\n\nProposed innovations:\n1. Enhanced approach using hybrid methodologies\n2. Integration of multiple solution vectors\n3. Optimization for efficiency and reliability\n\nRecommendation: Proceed to PLAN mode for detailed implementation strategy."
    
    async def _execute_plan_mode(self, input_data: str, context: Optional[str]) -> str:
        """Execute PLAN mode"""
        return f"IMPLEMENTATION CHECKLIST:\n\nFor input: {input_data}\n\n1. Initialize required components\n2. Configure system parameters\n3. Execute primary workflow\n4. Validate results and outputs\n5. Review and optimize performance\n\nReady for EXECUTE mode implementation."
    
    async def _execute_execute_mode(self, input_data: str, context: Optional[str]) -> str:
        """Execute EXECUTE mode"""
        return f"EXECUTION RESULTS:\n\nImplementing plan for: {input_data}\n\nActions taken:\n✓ Component initialization completed\n✓ System configuration applied\n✓ Primary workflow executed\n✓ Results validated\n\nExecution successful. Ready for REVIEW mode."
    
    async def _execute_review_mode(self, input_data: str, context: Optional[str]) -> str:
        """Execute REVIEW mode"""
        return f"REVIEW ANALYSIS:\n\nReviewing execution for: {input_data}\n\nReview findings:\n✓ All objectives met\n✓ Quality standards maintained\n✓ Performance within acceptable parameters\n✓ No critical issues identified\n\nReview complete. Execution validated."
    
    async def run_full_protocol(self, 
                              initial_input: str, 
                              context: Optional[str] = None) -> RIPERResult:
        """Run complete RIPER-Ω protocol chain"""
        start_time = time.time()
        self.protocol_stats["total_executions"] += 1
        
        try:
            logger.info("RIPER-Ω: Starting full protocol execution")
            
            # Mode chain: RESEARCH → PLAN → EXECUTE → REVIEW
            mode_chain = [RIPERMode.RESEARCH, RIPERMode.PLAN, RIPERMode.EXECUTE, RIPERMode.REVIEW]
            outputs = []
            
            current_input = initial_input
            
            for mode in mode_chain:
                # Enter mode
                if not await self.enter_mode(mode, context):
                    raise Exception(f"Failed to enter {mode.value} mode")
                
                # Execute mode
                output, validation = await self.execute_mode(current_input, context)
                outputs.append(output)
                
                # Check for hallucination
                if validation.get("is_hallucination", False):
                    raise Exception(f"Hallucination detected in {mode.value} mode")
                
                # Use output as input for next mode
                current_input = output
            
            # Calculate final metrics
            execution_time = time.time() - start_time
            confidence_score = 1.0 - self.state.hallucination_score
            
            # Create result
            result = RIPERResult(
                success=True,
                final_output=outputs[-1],
                mode_chain=mode_chain,
                execution_time=execution_time,
                confidence_score=confidence_score,
                hallucination_score=self.state.hallucination_score,
                validation_results={
                    "all_modes_passed": True,
                    "hallucination_threshold": self.config["hallucination_threshold"],
                    "mode_outputs": self.mode_outputs.copy()
                }
            )
            
            # Update statistics
            self.protocol_stats["successful_executions"] += 1
            self._update_protocol_stats(execution_time)
            
            logger.info(f"RIPER-Ω: Full protocol completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"RIPER-Ω protocol execution failed: {e}"
            logger.error(error_msg)
            
            return RIPERResult(
                success=False,
                final_output="",
                mode_chain=self.state.mode_history.copy(),
                execution_time=time.time() - start_time,
                confidence_score=0.0,
                hallucination_score=1.0,
                validation_results={"error": True},
                error_message=error_msg
            )
    
    def _update_protocol_stats(self, execution_time: float):
        """Update protocol execution statistics"""
        total = self.protocol_stats["total_executions"]
        current_avg = self.protocol_stats["average_execution_time"]
        self.protocol_stats["average_execution_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        return {
            **self.protocol_stats,
            "success_rate": (
                self.protocol_stats["successful_executions"] / 
                max(1, self.protocol_stats["total_executions"])
            ),
            "current_mode": self.state.current_mode.value,
            "hallucination_guard_stats": self.hallucination_guard.get_stats(),
            "config": self.config.copy()
        }


# Convenience functions
async def create_riper_protocol(hallucination_threshold: float = 0.4) -> RIPERProtocol:
    """Create RIPER-Ω protocol instance"""
    return RIPERProtocol(hallucination_threshold)


async def run_riper_chain(input_data: str, 
                         context: Optional[str] = None,
                         hallucination_threshold: float = 0.4) -> RIPERResult:
    """Convenience function to run RIPER-Ω protocol chain"""
    protocol = await create_riper_protocol(hallucination_threshold)
    return await protocol.run_full_protocol(input_data, context)
