"""
Quality Evaluator - Output Assessment System
Evaluates agent outputs against standards and provides quality scores.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    completeness: float  # 0.0-1.0
    correctness: float   # 0.0-1.0
    clarity: float       # 0.0-1.0
    functionality: float # 0.0-1.0
    overall: float       # 0.0-1.0


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    metrics: QualityMetrics
    passed: bool
    issues: List[str]
    suggestions: List[str]
    detailed_feedback: str


class QualityEvaluator:
    """
    Quality Evaluator for Agent Outputs
    
    Provides comprehensive quality assessment including:
    - Completeness checking
    - Correctness validation
    - Clarity assessment
    - Functionality verification
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QualityEvaluator")
        self.quality_standards = {
            "min_completeness": 0.7,
            "min_correctness": 0.8,
            "min_clarity": 0.6,
            "min_functionality": 0.7,
            "min_overall": 0.7
        }
    
    async def evaluate_ui_creation(self, output: Any, task_description: str) -> QualityReport:
        """Evaluate UI creation task output"""
        
        issues = []
        suggestions = []
        
        # Initialize metrics
        completeness = 0.0
        correctness = 0.0
        clarity = 0.0
        functionality = 0.0
        
        if isinstance(output, str):
            # Check for Vue.js components
            vue_indicators = [
                "<template>", "<script>", "<style>",
                "export default", "Vue.component",
                ".vue", "vue"
            ]
            
            vue_score = sum(1 for indicator in vue_indicators if indicator in output) / len(vue_indicators)
            completeness += vue_score * 0.4
            
            # Check for proper structure
            if "<template>" in output and "</template>" in output:
                completeness += 0.2
                correctness += 0.3
            else:
                issues.append("Missing Vue template structure")
                suggestions.append("Include <template> section with proper HTML structure")
            
            if "<script>" in output and "</script>" in output:
                completeness += 0.2
                correctness += 0.3
            else:
                issues.append("Missing Vue script section")
                suggestions.append("Include <script> section with component logic")
            
            # Check for component exports
            if "export default" in output:
                correctness += 0.2
                functionality += 0.3
            else:
                issues.append("Missing component export")
                suggestions.append("Add 'export default' for Vue component")
            
            # Check for proper naming
            if re.search(r'name:\s*["\'][A-Z][a-zA-Z]*["\']', output):
                clarity += 0.3
                correctness += 0.2
            else:
                issues.append("Component name not found or improperly formatted")
                suggestions.append("Add proper component name in PascalCase")
            
            # Check for basic functionality indicators
            functionality_indicators = ["data()", "methods:", "computed:", "props:"]
            func_score = sum(0.2 for indicator in functionality_indicators if indicator in output)
            functionality += min(0.4, func_score)
            
            # Clarity assessment
            lines = output.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            if len(non_empty_lines) >= 10:
                clarity += 0.3
            else:
                issues.append("Output seems too minimal")
                suggestions.append("Provide more comprehensive implementation")
            
            # Check for comments or documentation
            if "//" in output or "/*" in output or "<!--" in output:
                clarity += 0.4
            else:
                suggestions.append("Add comments to explain component functionality")
        
        else:
            issues.append("Output is not in expected string format")
            completeness = 0.1
            correctness = 0.1
            clarity = 0.1
            functionality = 0.1
        
        # Ensure metrics are within bounds
        completeness = min(1.0, max(0.0, completeness))
        correctness = min(1.0, max(0.0, correctness))
        clarity = min(1.0, max(0.0, clarity))
        functionality = min(1.0, max(0.0, functionality))
        
        # Calculate overall score
        overall = (completeness + correctness + clarity + functionality) / 4
        
        metrics = QualityMetrics(
            completeness=completeness,
            correctness=correctness,
            clarity=clarity,
            functionality=functionality,
            overall=overall
        )
        
        # Determine if passed
        passed = (
            completeness >= self.quality_standards["min_completeness"] and
            correctness >= self.quality_standards["min_correctness"] and
            clarity >= self.quality_standards["min_clarity"] and
            functionality >= self.quality_standards["min_functionality"] and
            overall >= self.quality_standards["min_overall"]
        )
        
        # Generate detailed feedback
        detailed_feedback = self._generate_detailed_feedback(metrics, issues, suggestions)
        
        return QualityReport(
            metrics=metrics,
            passed=passed,
            issues=issues,
            suggestions=suggestions,
            detailed_feedback=detailed_feedback
        )
    
    async def evaluate_coding_task(self, output: Any, task_description: str) -> QualityReport:
        """Evaluate general coding task output"""
        
        issues = []
        suggestions = []
        
        # Initialize metrics
        completeness = 0.0
        correctness = 0.0
        clarity = 0.0
        functionality = 0.0
        
        if isinstance(output, str):
            # Basic completeness check
            if len(output.strip()) > 50:
                completeness += 0.4
            else:
                issues.append("Output too short for coding task")
                suggestions.append("Provide more comprehensive implementation")
            
            # Check for code structure
            code_indicators = ["def ", "class ", "function ", "const ", "let ", "var "]
            if any(indicator in output for indicator in code_indicators):
                completeness += 0.3
                correctness += 0.3
            else:
                issues.append("No clear code structure detected")
                suggestions.append("Include proper functions, classes, or variables")
            
            # Check for error handling
            error_handling = ["try:", "catch", "except:", "throw", "raise"]
            if any(handler in output for handler in error_handling):
                correctness += 0.3
                functionality += 0.3
            else:
                suggestions.append("Consider adding error handling")
            
            # Check for comments
            if "//" in output or "#" in output or "/*" in output:
                clarity += 0.4
            else:
                suggestions.append("Add comments to explain code functionality")
            
            # Check for proper formatting
            lines = output.split('\n')
            indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
            if indented_lines > 0:
                clarity += 0.3
            else:
                issues.append("Code appears to lack proper indentation")
                suggestions.append("Use proper indentation for code structure")
            
            # Functionality assessment
            if len(lines) >= 5:
                functionality += 0.4
            
        else:
            issues.append("Output is not in expected string format")
            completeness = 0.1
            correctness = 0.1
            clarity = 0.1
            functionality = 0.1
        
        # Ensure metrics are within bounds
        completeness = min(1.0, max(0.0, completeness))
        correctness = min(1.0, max(0.0, correctness))
        clarity = min(1.0, max(0.0, clarity))
        functionality = min(1.0, max(0.0, functionality))
        
        # Calculate overall score
        overall = (completeness + correctness + clarity + functionality) / 4
        
        metrics = QualityMetrics(
            completeness=completeness,
            correctness=correctness,
            clarity=clarity,
            functionality=functionality,
            overall=overall
        )
        
        # Determine if passed
        passed = overall >= self.quality_standards["min_overall"]
        
        # Generate detailed feedback
        detailed_feedback = self._generate_detailed_feedback(metrics, issues, suggestions)
        
        return QualityReport(
            metrics=metrics,
            passed=passed,
            issues=issues,
            suggestions=suggestions,
            detailed_feedback=detailed_feedback
        )
    
    async def evaluate_output(self, output: Any, task_description: str, task_type: str = "general") -> QualityReport:
        """Main evaluation method that routes to specific evaluators"""
        
        try:
            if "ui" in task_type.lower() or "vue" in task_description.lower():
                return await self.evaluate_ui_creation(output, task_description)
            elif "cod" in task_type.lower():
                return await self.evaluate_coding_task(output, task_description)
            else:
                # Default general evaluation
                return await self.evaluate_coding_task(output, task_description)
                
        except Exception as e:
            self.logger.error(f"Error evaluating output: {e}")
            
            # Return minimal failing report
            metrics = QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
            return QualityReport(
                metrics=metrics,
                passed=False,
                issues=[f"Evaluation error: {str(e)}"],
                suggestions=["Please retry the task"],
                detailed_feedback=f"Quality evaluation failed: {str(e)}"
            )
    
    def _generate_detailed_feedback(self, metrics: QualityMetrics, issues: List[str], suggestions: List[str]) -> str:
        """Generate detailed feedback report"""
        
        feedback_parts = []
        
        # Metrics summary
        feedback_parts.append("Quality Assessment:")
        feedback_parts.append(f"- Completeness: {metrics.completeness:.1%}")
        feedback_parts.append(f"- Correctness: {metrics.correctness:.1%}")
        feedback_parts.append(f"- Clarity: {metrics.clarity:.1%}")
        feedback_parts.append(f"- Functionality: {metrics.functionality:.1%}")
        feedback_parts.append(f"- Overall Score: {metrics.overall:.1%}")
        feedback_parts.append("")
        
        # Issues
        if issues:
            feedback_parts.append("Issues Found:")
            for issue in issues:
                feedback_parts.append(f"- {issue}")
            feedback_parts.append("")
        
        # Suggestions
        if suggestions:
            feedback_parts.append("Suggestions for Improvement:")
            for suggestion in suggestions:
                feedback_parts.append(f"- {suggestion}")
        
        return "\n".join(feedback_parts)
    
    def update_standards(self, new_standards: Dict[str, float]) -> None:
        """Update quality standards"""
        self.quality_standards.update(new_standards)
        self.logger.info(f"Updated quality standards: {self.quality_standards}")
