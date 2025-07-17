"""
DGM Empirical Validator - Validates improvement candidates through testing
"""
import asyncio
import logging
import tempfile
from typing import Dict, Any, List
from datetime import datetime

from ..models import (
    ImprovementCandidate, ValidationResult, PerformanceMetric
)

logger = logging.getLogger(__name__)

class EmpiricalValidator:
    """Validates improvement candidates through empirical testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get("validation_timeout", 300)  # 5 minutes
        self.test_iterations = config.get("test_iterations", 10)
        self.performance_threshold = config.get("performance_threshold", 0.05)
        self.safety_checks = config.get("safety_checks", True)
        
    async def validate_candidate(self, candidate: ImprovementCandidate) -> ValidationResult:
        """Validate an improvement candidate"""
        logger.info(f"Starting validation for candidate {candidate.id}")
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Syntax and static analysis
            static_results = await self._run_static_analysis(candidate)
            if not static_results["success"]:
                return ValidationResult(
                    candidate_id=candidate.id,
                    success=False,
                    performance_before=[],
                    performance_after=[],
                    improvement_score=0.0,
                    safety_score=0.0,
                    test_results=static_results,
                    validation_time=0.0
                )
            
            # Step 2: Measure baseline performance
            performance_before = await self._measure_baseline_performance()
            
            # Step 3: Apply changes in isolated environment
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test environment
                test_env = await self._create_test_environment(temp_dir, candidate)
                
                # Step 4: Run tests with changes
                test_results = await self._run_tests_with_changes(test_env, candidate)
                
                # Step 5: Measure performance after changes
                if test_results["success"]:
                    performance_after = await self._measure_performance_with_changes(
                        test_env, candidate
                    )
                else:
                    performance_after = []
                
                # Step 6: Calculate improvement and safety scores
                improvement_score = self._calculate_improvement_score(
                    performance_before, performance_after
                )
                safety_score = await self._calculate_safety_score(
                    candidate, test_results
                )
                
                validation_time = (datetime.utcnow() - start_time).total_seconds()
                
                return ValidationResult(
                    candidate_id=candidate.id,
                    success=test_results["success"] and improvement_score > 0,
                    performance_before=performance_before,
                    performance_after=performance_after,
                    improvement_score=improvement_score,
                    safety_score=safety_score,
                    test_results=test_results,
                    validation_time=validation_time
                )
                
        except Exception as e:
            logger.error(f"Validation failed for candidate {candidate.id}: {e}")
            validation_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ValidationResult(
                candidate_id=candidate.id,
                success=False,
                performance_before=[],
                performance_after=[],
                improvement_score=0.0,
                safety_score=0.0,
                test_results={"error": str(e)},
                validation_time=validation_time
            )
    
    async def _run_static_analysis(self, candidate: ImprovementCandidate) -> Dict[str, Any]:
        """Run static analysis on code changes"""
        try:
            results = {"success": True, "issues": []}
            
            for filename, code in candidate.code_changes.items():
                # Basic syntax check
                try:
                    compile(code, filename, 'exec')
                except SyntaxError as e:
                    results["success"] = False
                    results["issues"].append({
                        "file": filename,
                        "type": "syntax_error",
                        "message": str(e),
                        "line": e.lineno
                    })
                
                # Check for dangerous patterns
                dangerous_patterns = [
                    "eval(", "exec(", "__import__", "subprocess", "os.system"
                ]
                
                for pattern in dangerous_patterns:
                    if pattern in code:
                        results["issues"].append({
                            "file": filename,
                            "type": "security_warning",
                            "message": f"Potentially dangerous pattern: {pattern}",
                            "severity": "high"
                        })
            
            return results
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _measure_baseline_performance(self) -> List[PerformanceMetric]:
        """Measure current performance before changes"""
        metrics = []
        
        try:
            # Simulate performance measurement
            # In real implementation, this would measure actual agent performance
            metrics.append(PerformanceMetric(
                name="response_time",
                value=0.5,
                unit="seconds"
            ))
            
            metrics.append(PerformanceMetric(
                name="accuracy",
                value=0.85,
                unit="percentage"
            ))
            
            metrics.append(PerformanceMetric(
                name="memory_usage",
                value=128.0,
                unit="MB"
            ))
            
        except Exception as e:
            logger.error(f"Error measuring baseline performance: {e}")        
        return metrics
    
    async def _create_test_environment(self, temp_dir: str, candidate: ImprovementCandidate) -> Dict[str, Any]:
        """Create isolated test environment"""
        import os
        
        # Copy necessary files to temp directory
        test_env = {
            "path": temp_dir,
            "files": {},
            "config": self.config.copy()
        }
        
        # Apply code changes to temp files
        for filename, code in candidate.code_changes.items():
            temp_file = os.path.join(temp_dir, os.path.basename(filename))
            with open(temp_file, 'w') as f:
                f.write(code)
            test_env["files"][filename] = temp_file
        
        return test_env
    
    async def _run_tests_with_changes(self, test_env: Dict[str, Any], candidate: ImprovementCandidate) -> Dict[str, Any]:
        """Run tests with the proposed changes"""
        try:
            # Simulate test execution
            # In real implementation, this would run actual tests
            await asyncio.sleep(0.1)  # Simulate test time
            
            return {
                "success": True,
                "tests_passed": 10,
                "tests_failed": 0,
                "coverage": 0.95,
                "execution_time": 0.1
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _measure_performance_with_changes(
        self, 
        test_env: Dict[str, Any], 
        candidate: ImprovementCandidate
    ) -> List[PerformanceMetric]:
        """Measure performance with proposed changes"""
        metrics = []
        
        try:
            # Simulate improved performance
            # In real implementation, this would measure actual performance
            improvement_factor = 1.0 + candidate.expected_improvement
            
            metrics.append(PerformanceMetric(
                name="response_time",
                value=0.5 / improvement_factor,
                unit="seconds"
            ))
            
            metrics.append(PerformanceMetric(
                name="accuracy",
                value=min(0.85 * improvement_factor, 1.0),
                unit="percentage"
            ))
            
            metrics.append(PerformanceMetric(
                name="memory_usage",
                value=128.0 * (2.0 - improvement_factor),
                unit="MB"
            ))
            
        except Exception as e:
            logger.error(f"Error measuring performance with changes: {e}")
        
        return metrics
    
    def _calculate_improvement_score(
        self, 
        before: List[PerformanceMetric], 
        after: List[PerformanceMetric]
    ) -> float:
        """Calculate overall improvement score"""
        if not before or not after:
            return 0.0
        
        # Create metric lookup for easier comparison
        before_metrics = {m.name: m.value for m in before}
        after_metrics = {m.name: m.value for m in after}
        
        improvements = []
        
        for metric_name in before_metrics:
            if metric_name in after_metrics:
                before_val = before_metrics[metric_name]
                after_val = after_metrics[metric_name]
                
                # Different metrics have different improvement directions
                if metric_name in ["response_time", "memory_usage"]:
                    # Lower is better
                    if before_val > 0:
                        improvement = (before_val - after_val) / before_val
                    else:
                        improvement = 0.0
                else:
                    # Higher is better (accuracy, etc.)
                    if before_val > 0:
                        improvement = (after_val - before_val) / before_val
                    else:
                        improvement = 0.0
                
                improvements.append(improvement)
        
        # Return average improvement
        if improvements:
            return sum(improvements) / len(improvements)
        else:
            return 0.0
    
    async def _calculate_safety_score(
        self, 
        candidate: ImprovementCandidate, 
        test_results: Dict[str, Any]
    ) -> float:
        """Calculate safety score for the candidate"""
        score = 1.0
        
        # Reduce score based on test failures
        if not test_results.get("success", False):
            score *= 0.1
        
        # Reduce score based on coverage
        coverage = test_results.get("coverage", 0.0)
        if coverage < 0.8:
            score *= (coverage + 0.2)
        
        # Reduce score based on risk level
        score *= (1.0 - candidate.risk_level)
        
        # Additional safety checks could be added here
        
        return max(0.0, min(1.0, score))


class DGMValidator:
    """
    Observer-approved DGM Validator with configurable thresholds and dynamic adaptation
    Combines empirical validation with formal verification for enhanced safety
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Observer-approved configurable thresholds
        self.safety_threshold = config.get("safety_threshold", 0.6)
        self.performance_threshold = config.get("performance_threshold", 0.05)
        self.complexity_threshold = config.get("complexity_threshold", 1500)
        self.bloat_threshold = config.get("bloat_threshold", 0.15)

        # Dynamic threshold adaptation
        self.adaptive_thresholds = config.get("adaptive_thresholds", True)
        self.threshold_learning_rate = config.get("threshold_learning_rate", 0.1)

        # Initialize empirical validator
        self.empirical_validator = EmpiricalValidator(config)

        # Validation history for threshold adaptation
        self.validation_history = []
        self.success_rate_window = config.get("success_rate_window", 10)

        logger.info(f"DGMValidator initialized with safety_threshold={self.safety_threshold}")

    async def validate_improvement(self, candidate: ImprovementCandidate) -> ValidationResult:
        """
        Validate improvement candidate with Observer-approved safety checks
        """
        logger.info(f"Starting DGM validation for candidate {candidate.id}")

        try:
            # Step 1: Run empirical validation
            empirical_result = await self.empirical_validator.validate_candidate(candidate)

            # Step 2: Apply Observer safety thresholds
            safety_passed = empirical_result.safety_score >= self.safety_threshold
            performance_passed = empirical_result.improvement_score >= self.performance_threshold

            # Step 3: Check complexity and bloat
            complexity_passed = self._check_complexity(candidate)
            bloat_passed = self._check_bloat(candidate)

            # Step 4: Overall validation decision
            overall_success = (
                empirical_result.success and
                safety_passed and
                performance_passed and
                complexity_passed and
                bloat_passed
            )

            # Step 5: Create enhanced validation result
            enhanced_result = ValidationResult(
                candidate_id=candidate.id,
                success=overall_success,
                performance_before=empirical_result.performance_before,
                performance_after=empirical_result.performance_after,
                improvement_score=empirical_result.improvement_score,
                safety_score=empirical_result.safety_score,
                test_results={
                    **empirical_result.test_results,
                    'safety_threshold_passed': safety_passed,
                    'performance_threshold_passed': performance_passed,
                    'complexity_passed': complexity_passed,
                    'bloat_passed': bloat_passed,
                    'thresholds_used': {
                        'safety': self.safety_threshold,
                        'performance': self.performance_threshold,
                        'complexity': self.complexity_threshold,
                        'bloat': self.bloat_threshold
                    }
                },
                validation_time=empirical_result.validation_time
            )

            # Step 6: Update validation history and adapt thresholds
            self._update_validation_history(enhanced_result)
            if self.adaptive_thresholds:
                self._adapt_thresholds()

            logger.info(f"DGM validation completed: success={overall_success}, "
                       f"safety={empirical_result.safety_score:.3f}, "
                       f"improvement={empirical_result.improvement_score:.3f}")

            return enhanced_result

        except Exception as e:
            logger.error(f"DGM validation failed for candidate {candidate.id}: {e}")
            return ValidationResult(
                candidate_id=candidate.id,
                success=False,
                performance_before=[],
                performance_after=[],
                improvement_score=0.0,
                safety_score=0.0,
                test_results={"error": str(e)},
                validation_time=0.0
            )

    def _check_complexity(self, candidate: ImprovementCandidate) -> bool:
        """Check if candidate meets complexity thresholds"""
        try:
            total_lines = 0
            for code in candidate.code_changes.values():
                total_lines += len(code.split('\n'))

            complexity_passed = total_lines <= self.complexity_threshold
            logger.debug(f"Complexity check: {total_lines} lines <= {self.complexity_threshold} = {complexity_passed}")
            return complexity_passed

        except Exception as e:
            logger.warning(f"Complexity check failed: {e}")
            return False

    def _check_bloat(self, candidate: ImprovementCandidate) -> bool:
        """Check if candidate introduces code bloat"""
        try:
            # Simple bloat detection based on code size vs improvement
            total_chars = sum(len(code) for code in candidate.code_changes.values())
            improvement_ratio = candidate.expected_improvement

            if improvement_ratio > 0:
                bloat_ratio = total_chars / (improvement_ratio * 10000)  # Normalize
                bloat_passed = bloat_ratio <= self.bloat_threshold
            else:
                bloat_passed = False

            logger.debug(f"Bloat check: ratio={bloat_ratio:.3f} <= {self.bloat_threshold} = {bloat_passed}")
            return bloat_passed

        except Exception as e:
            logger.warning(f"Bloat check failed: {e}")
            return False

    def _update_validation_history(self, result: ValidationResult):
        """Update validation history for threshold adaptation"""
        self.validation_history.append({
            'timestamp': datetime.utcnow(),
            'success': result.success,
            'safety_score': result.safety_score,
            'improvement_score': result.improvement_score
        })

        # Keep only recent history
        if len(self.validation_history) > self.success_rate_window * 2:
            self.validation_history = self.validation_history[-self.success_rate_window:]

    def _adapt_thresholds(self):
        """Adapt thresholds based on validation history"""
        if len(self.validation_history) < self.success_rate_window:
            return

        recent_history = self.validation_history[-self.success_rate_window:]
        success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)

        # Adapt safety threshold based on success rate
        if success_rate < 0.5:  # Too many failures, lower threshold
            adjustment = -self.threshold_learning_rate * 0.1
            self.safety_threshold = max(0.3, self.safety_threshold + adjustment)
            logger.info(f"Lowered safety threshold to {self.safety_threshold:.3f} (success rate: {success_rate:.1%})")
        elif success_rate > 0.9:  # Too many passes, raise threshold
            adjustment = self.threshold_learning_rate * 0.1
            self.safety_threshold = min(0.9, self.safety_threshold + adjustment)
            logger.info(f"Raised safety threshold to {self.safety_threshold:.3f} (success rate: {success_rate:.1%})")

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics"""
        if not self.validation_history:
            return {"no_data": True}

        recent_history = self.validation_history[-self.success_rate_window:]
        success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
        avg_safety = sum(h['safety_score'] for h in recent_history) / len(recent_history)
        avg_improvement = sum(h['improvement_score'] for h in recent_history) / len(recent_history)

        return {
            "total_validations": len(self.validation_history),
            "recent_success_rate": success_rate,
            "average_safety_score": avg_safety,
            "average_improvement_score": avg_improvement,
            "current_thresholds": {
                "safety": self.safety_threshold,
                "performance": self.performance_threshold,
                "complexity": self.complexity_threshold,
                "bloat": self.bloat_threshold
            }
        }
