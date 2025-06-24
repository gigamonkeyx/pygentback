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
