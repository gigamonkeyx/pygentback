"""Safety Monitor for DGM Engine

This module provides safety monitoring and constraint enforcement for the DGM system.
"""

import logging
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..models import DGMProgram, DGMEvolution

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for safety monitoring"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    id: str
    risk_level: RiskLevel
    message: str
    context: Dict[str, Any]
    timestamp: float
    resolved: bool = False


class SafetyMonitor:
    """Safety monitoring and constraint enforcement for DGM"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        self.max_violations = config.get("max_violations", 100)
        self.violations: List[SafetyViolation] = []
        self.lock = threading.Lock()
        self.safety_rules = self._initialize_safety_rules()
        logger.info("SafetyMonitor initialized")
    
    def _initialize_safety_rules(self) -> Dict[str, Any]:
        """Initialize safety rules and constraints"""
        return {
            "max_program_size": 10000,  # Maximum program size in characters
            "max_execution_time": 300,   # Maximum execution time in seconds
            "forbidden_modules": ["os", "subprocess", "sys"],
            "forbidden_functions": ["exec", "eval", "__import__"],
            "max_memory_usage": 500 * 1024 * 1024,  # 500MB
        }
    
    def check_program_safety(self, program: DGMProgram) -> List[SafetyViolation]:
        """Check if a program meets safety constraints"""
        violations = []
        
        # Check program size
        if len(program.code) > self.safety_rules["max_program_size"]:
            violations.append(SafetyViolation(
                id=f"size_violation_{program.id}",
                risk_level=RiskLevel.HIGH,
                message=f"Program size exceeds maximum allowed ({len(program.code)} > {self.safety_rules['max_program_size']})",
                context={"program_id": program.id, "size": len(program.code)},
                timestamp=program.timestamp
            ))
        
        # Check for forbidden modules
        for module in self.safety_rules["forbidden_modules"]:
            if f"import {module}" in program.code:
                violations.append(SafetyViolation(
                    id=f"module_violation_{program.id}_{module}",
                    risk_level=RiskLevel.CRITICAL,
                    message=f"Program contains forbidden module: {module}",
                    context={"program_id": program.id, "module": module},
                    timestamp=program.timestamp
                ))
        
        # Check for forbidden functions
        for func in self.safety_rules["forbidden_functions"]:
            if func in program.code:
                violations.append(SafetyViolation(
                    id=f"function_violation_{program.id}_{func}",
                    risk_level=RiskLevel.CRITICAL,
                    message=f"Program contains forbidden function: {func}",
                    context={"program_id": program.id, "function": func},
                    timestamp=program.timestamp
                ))
        
        return violations
    
    def check_evolution_safety(self, evolution: DGMEvolution) -> List[SafetyViolation]:
        """Check if an evolution meets safety constraints"""
        violations = []
        
        # Check evolution fitness bounds
        if evolution.fitness < -1000 or evolution.fitness > 1000:
            violations.append(SafetyViolation(
                id=f"fitness_violation_{evolution.id}",
                risk_level=RiskLevel.MEDIUM,
                message=f"Evolution fitness out of bounds: {evolution.fitness}",
                context={"evolution_id": evolution.id, "fitness": evolution.fitness},
                timestamp=evolution.timestamp
            ))
        
        return violations
    
    def record_violation(self, violation: SafetyViolation):
        """Record a safety violation"""
        with self.lock:
            self.violations.append(violation)
            if len(self.violations) > self.max_violations:
                self.violations.pop(0)  # Remove oldest violation
            
            logger.warning(f"Safety violation recorded: {violation.message}")
    
    def get_violations(self, risk_level: Optional[RiskLevel] = None) -> List[SafetyViolation]:
        """Get safety violations, optionally filtered by risk level"""
        with self.lock:
            if risk_level:
                return [v for v in self.violations if v.risk_level == risk_level]
            return self.violations.copy()
    
    def resolve_violation(self, violation_id: str) -> bool:
        """Mark a violation as resolved"""
        with self.lock:
            for violation in self.violations:
                if violation.id == violation_id:
                    violation.resolved = True
                    logger.info(f"Safety violation resolved: {violation_id}")
                    return True
            return False
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        with self.lock:
            unresolved = [v for v in self.violations if not v.resolved]
            critical = [v for v in unresolved if v.risk_level == RiskLevel.CRITICAL]
            high = [v for v in unresolved if v.risk_level == RiskLevel.HIGH]
            
            return {
                "total_violations": len(self.violations),
                "unresolved_violations": len(unresolved),
                "critical_violations": len(critical),
                "high_risk_violations": len(high),
                "safety_rules": self.safety_rules,
                "system_safe": len(critical) == 0
            }
    
    def is_safe_to_execute(self, program: DGMProgram) -> bool:
        """Check if it's safe to execute a program"""
        violations = self.check_program_safety(program)
        critical_violations = [v for v in violations if v.risk_level == RiskLevel.CRITICAL]
        return len(critical_violations) == 0
    
    def shutdown(self):
        """Cleanup safety monitor resources"""
        with self.lock:
            logger.info("SafetyMonitor shutting down")
            self.violations.clear()
    
    async def evaluate_candidate_safety(self, candidate) -> Dict[str, Any]:
        """Evaluate the safety of an improvement candidate"""
        from ..models import DGMProgram
        
        # Create a temporary program from the candidate for evaluation
        temp_program = DGMProgram(
            id=f"temp_{candidate.id}",
            code=str(candidate.code_changes),  # Convert to string for evaluation
            description=candidate.description,
            author="dgm_engine"
        )
        
        # Check program safety
        violations = self.check_program_safety(temp_program)
        
        # Determine overall safety score
        critical_count = len([v for v in violations if v.risk_level == RiskLevel.CRITICAL])
        high_count = len([v for v in violations if v.risk_level == RiskLevel.HIGH])
        
        safety_score = max(0.0, 1.0 - (critical_count * 0.5 + high_count * 0.2))
        
        return {
            "safe": critical_count == 0,
            "safety_score": safety_score,
            "violations": violations,
            "recommendation": "approve" if critical_count == 0 else "reject"
        }
